//! BF16 row projections and affine parameter forms.

pub use affine::{Affine, Bias, output, query};
pub(crate) use affine::{Coefficients, Inputs, Outputs};
pub(crate) use decode::Matrix;
pub use experts::Experts;
pub(crate) use rows::Projection;

use core::borrow::Borrow;
use core::marker::PhantomData;

use furiosa_opt_std::prelude::*;

use crate::Op;
use crate::load::{self, Load, LoadError};
use crate::ops::normalization::Spread;
use crate::tensor::form::Weight;
use crate::tensor::form::{BlockScaled, Part, Projections};
use crate::tensor::mapping;
use crate::tensor::{axis::Lanes, live::Live};

pub mod affine;
mod decode;
pub(crate) mod experts;
pub(crate) mod fp8;
pub(crate) mod kernel;
pub mod modulation;
pub(crate) mod nvfp4;
mod projection;
mod quantize;
pub(crate) mod rows;
pub(crate) mod split;
pub(crate) mod stream;

/// Multiply BF16 input rows by a retained [Out, In] weight matrix transposed: its checkpoint's
/// `weight`, held as `T`, the HBM tensor or a borrow of it in a borrowed tree.
#[derive(DeviceSend, Load)]
pub struct Linear<
    In: AxisName,
    Out: AxisName,
    Chips: M = m![1],
    T = HbmTensor<bf16, Chips, m![Out, In]>,
> {
    #[load(rename = "")]
    pub(crate) affine: Affine<T, ()>,
    shape: PhantomData<fn() -> (In, Out, Chips)>,
}

impl<In: AxisName, Out: AxisName, Chips: M, T> Linear<In, Out, Chips, T> {
    /// Retain the BF16 [Out, In] matrix, owned or borrowed, without copying it.
    pub fn new(weight: Weight<m![Out, In], Chips, T>) -> Self {
        Self {
            affine: Affine::new(weight.into_tensor(), ()),
            shape: PhantomData,
        }
    }
}

/// A matrix whose value is built on host, as a converted one's is, held as its tensor is.
impl<In: AxisName, Out: AxisName, Chips: M + 'static, T>
    Load<HostTensor<bf16, m![{ Chips }, { m![Out, In] }]>> for Linear<In, Out, Chips, T>
where
    T: Load<HostTensor<bf16, m![{ Chips }, { m![Out, In] }]>>,
{
    fn load(
        context: &mut load::Context<'_>,
        host: HostTensor<bf16, m![{ Chips }, { m![Out, In] }]>,
    ) -> Result<Self, LoadError> {
        T::load(context, host).map(|weight| Self {
            affine: Affine::new(weight, ()),
            shape: PhantomData,
        })
    }
}

/// The projections as one matrix, their rows in order on every chip.
impl<'a, In: AxisName, Out: AxisName, Chips: M + 'static, T, const N: usize>
    Load<Projections<'a, N>> for Linear<In, Out, Chips, T>
where
    T: Load<HostTensor<bf16, m![{ Chips }, { m![Out, In] }]>>,
{
    fn load(
        context: &mut load::Context<'_>,
        projections: Projections<'a, N>,
    ) -> Result<Self, LoadError> {
        let parts = projections.read(context, |from, cx| {
            cx.field("weight", "weight", |cx| {
                from.host::<
                    bf16,
                    m![{ Symbol<Part<Out, N>> }, In],
                    m![{ Chips }, { m![{ Symbol<Part<Out, N>> }, In] }],
                >(cx)
            })
        })?;
        let parts = parts.map(HostTensor::into_vec);
        let share = Part::<Out, N>::SIZE * In::SIZE;
        let rows = (0..Chips::SIZE)
            .flat_map(|chip| {
                parts
                    .iter()
                    .flat_map(move |part| &part[chip * share..][..share])
            })
            .copied();
        T::load(context, HostTensor::from_vec(rows)).map(|weight| Self::new(Weight::new(weight)))
    }
}

/// The projections as one channel-scaled matrix, their rows in order on every chip.
impl<'a, In: AxisName, Out: AxisName, Chips: M + 'static, W, T, const N: usize>
    Load<Projections<'a, N>>
    for Linear<In, Out, Chips, BlockScaled<In, Out, f8e4m3, f32, m![Out, 1 # 8], (), Chips, W, T>>
where
    BlockScaled<In, Out, f8e4m3, f32, m![Out, 1 # 8], (), Chips, W, T>: Load<Projections<'a, N>>,
{
    fn load(
        context: &mut load::Context<'_>,
        projections: Projections<'a, N>,
    ) -> Result<Self, LoadError> {
        Load::load(context, projections).map(|weight| Self::new(Weight::new(weight)))
    }
}

/// A matrix built on host written into the held tensor's storage.
impl<'t, In: AxisName, Out: AxisName, Chips: M + 'static, T>
    Load<(
        &'t mut Linear<In, Out, Chips, T>,
        HostTensor<bf16, m![{ Chips }, { m![Out, In] }]>,
    )> for &'t mut Linear<In, Out, Chips, T>
where
    for<'u> &'u mut T: Load<(&'u mut T, HostTensor<bf16, m![{ Chips }, { m![Out, In] }]>)>,
{
    fn load(
        context: &mut load::Context<'_>,
        (into, host): (
            &'t mut Linear<In, Out, Chips, T>,
            HostTensor<bf16, m![{ Chips }, { m![Out, In] }]>,
        ),
    ) -> Result<Self, LoadError> {
        <&mut T as Load<_>>::load(context, (&mut into.affine.weight, host))?;
        Ok(into)
    }
}

impl<In: AxisName, Out: AxisName, Chips: M, T> Linear<In, Out, Chips, T>
where
    T: Borrow<HbmTensor<bf16, Chips, m![Out, In]>>,
{
    /// Borrow the contiguous [Out, In] weight matrix on each chip, owned or borrowed by the tree.
    pub fn tensor(&self) -> &HbmTensor<bf16, Chips, m![Out, In]> {
        self.affine.weight.borrow()
    }
}

/// The lane groups of `Rows`, spelled as a mapping so that a row type naming them names `Rows`.
pub(crate) type Groups<Rows> = m![Rows / ({ Symbol<Lanes<Rows>> })];

/// Projected rows in the resident contraction's native mapping.
pub(crate) type Projected<In, Out, Rows> = kernel::Result<In, Out, Rows>;

// Rows stay explicit while the feature partition belongs to the input tensor. A consumer does
// not require the producer's hidden width or a particular cluster/slice placement.
impl<'x, In: AxisName, Out: AxisName, Rows: AxisName, Chips: M, S: M, F: M, T>
    Op<&'x DmTensor<bf16, m![1], Chips, S, m![Rows, { F }]>> for Linear<In, Out, m![1], T>
where
    T: Matrix<In = In, Out = Out>,
{
    type Output = Projected<In, Out, Rows>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'x DmTensor<bf16, m![1], Chips, S, m![Rows, { F }]>,
    ) -> Projected<In, Out, Rows> {
        self.affine.forward(
            device,
            (Live::all().with(PhantomData), input.view(), |at| {
                mapping::Rows::<Rows>::group(input.view(), at)
            }),
        )
    }
}

impl<'x, In: AxisName, Out: AxisName, Rows: AxisName, Chips: M, S: M, F: M, T>
    Op<(
        Live<(), Rows>,
        &'x DmTensor<bf16, m![1], Chips, S, m![Rows, { F }]>,
    )> for Linear<In, Out, m![1], T>
where
    T: Matrix<In = In, Out = Out>,
{
    type Output = Projected<In, Out, Rows>;

    fn forward(
        &self,
        device: &mut Device,
        (live, input): (
            Live<(), Rows>,
            &'x DmTensor<bf16, m![1], Chips, S, m![Rows, { F }]>,
        ),
    ) -> Projected<In, Out, Rows> {
        self.affine.forward(
            device,
            (live.with(PhantomData), input.view(), |at| {
                mapping::Rows::<Rows>::group(input.view(), at)
            }),
        )
    }
}

/// Rows in HBM, too many for data memory, project block by block into the caller's rows.
impl<'x, 'o, In: AxisName, Out: AxisName, Rows: AxisName, Chips: M, T>
    Op<(
        &'x HbmTensor<bf16, Chips, m![Rows, In]>,
        &'o mut HbmTensor<bf16, Chips, m![Rows, Out]>,
    )> for Linear<In, Out, Chips, T>
where
    T: Borrow<HbmTensor<bf16, Chips, m![Out, In]>>,
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (input, output): (
            &'x HbmTensor<bf16, Chips, m![Rows, In]>,
            &'o mut HbmTensor<bf16, Chips, m![Rows, Out]>,
        ),
    ) {
        rows::Projection::new(self.tensor(), &()).forward(device, (input, output));
    }
}

/// Rows a composition moves in project as borrowed rows do.
impl<'o, In: AxisName, Out: AxisName, Rows: AxisName, Chips: M, T>
    Op<(
        HbmTensor<bf16, Chips, m![Rows, In]>,
        &'o mut HbmTensor<bf16, Chips, m![Rows, Out]>,
    )> for Linear<In, Out, Chips, T>
where
    T: Borrow<HbmTensor<bf16, Chips, m![Out, In]>>,
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (input, output): (
            HbmTensor<bf16, Chips, m![Rows, In]>,
            &'o mut HbmTensor<bf16, Chips, m![Rows, Out]>,
        ),
    ) {
        self.forward(device, (&input, output));
    }
}

/// Rows in HBM project into new rows.
impl<'x, In: AxisName, Out: AxisName, Rows: AxisName, Chips: M, T>
    Op<&'x HbmTensor<bf16, Chips, m![Rows, In]>> for Linear<In, Out, Chips, T>
where
    T: Borrow<HbmTensor<bf16, Chips, m![Out, In]>>,
{
    type Output = HbmTensor<bf16, Chips, m![Rows, Out]>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'x HbmTensor<bf16, Chips, m![Rows, In]>,
    ) -> HbmTensor<bf16, Chips, m![Rows, Out]> {
        let mut output = HbmTensor::new();
        self.forward(device, (input, &mut output));
        output
    }
}

/// BF16 coefficients with an FP32 bias whose FP32 sums come out unrounded: its checkpoint's
/// `weight` and `bias`, held as `W` and `B`, the HBM tensors or borrows of them.
#[derive(DeviceSend, Load)]
pub struct Unrounded<
    In: AxisName,
    Out: AxisName,
    Chips: M = m![1],
    W = HbmTensor<bf16, Chips, m![Out, In]>,
    B = HbmTensor<f32, Chips, m![Out]>,
>(pub Affine<W, B>, pub PhantomData<fn() -> (In, Out, Chips)>);

/// Rows in HBM project block by block into FP32 rows in HBM, each output's bias added to its sum.
impl<'x, In: AxisName, Out: AxisName, Rows: AxisName, Chips: M, W, B>
    Op<&'x HbmTensor<bf16, Chips, m![Rows, In]>> for Unrounded<In, Out, Chips, W, B>
where
    W: Borrow<HbmTensor<bf16, Chips, m![Out, In]>>,
    B: Borrow<HbmTensor<f32, Chips, m![Out]>>,
{
    type Output = HbmTensor<f32, Chips, m![Rows, Out]>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'x HbmTensor<bf16, Chips, m![Rows, In]>,
    ) -> HbmTensor<f32, Chips, m![Rows, Out]> {
        let mut output = HbmTensor::new();
        let (weight, bias): (
            &HbmTensor<bf16, Chips, m![Out, In]>,
            &HbmTensor<f32, Chips, m![Out]>,
        ) = (self.0.weight.borrow(), self.0.bias.borrow());
        rows::Projection::new(weight, bias).forward(device, (input, &mut output));
        output
    }
}

/// A biased projection whose output rows come out head-major, each head's `H` features in order,
/// from its checkpoint's `weight` and `bias` held as `W` and `B`, owned or borrowed.
#[derive(DeviceSend, Load)]
pub struct PerHead<
    In: AxisName,
    Out: AxisName,
    H: AxisName,
    Chips: M = m![1],
    W = HbmTensor<bf16, Chips, m![Out, In]>,
    B = HbmTensor<f32, Chips, m![Out]>,
>(
    pub Affine<W, B>,
    pub PhantomData<fn() -> (In, Out, H, Chips)>,
);

/// Rows in HBM project block by block into head-major rows in HBM.
impl<'x, In: AxisName, Out: AxisName, H: AxisName, Rows: AxisName, Chips: M, W, B>
    Op<&'x HbmTensor<bf16, Chips, m![Rows, In]>> for PerHead<In, Out, H, Chips, W, B>
where
    W: Borrow<HbmTensor<bf16, Chips, m![Out, In]>>,
    B: Borrow<HbmTensor<f32, Chips, m![Out]>>,
{
    type Output = HbmTensor<bf16, Chips, m![Out / ({ Symbol<H> }), Rows, Out % ({ Symbol<H> })]>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'x HbmTensor<bf16, Chips, m![Rows, In]>,
    ) -> Self::Output {
        let mut output = HbmTensor::new();
        let (weight, bias): (
            &HbmTensor<bf16, Chips, m![Out, In]>,
            &HbmTensor<f32, Chips, m![Out]>,
        ) = (self.0.weight.borrow(), self.0.bias.borrow());
        rows::Projection::new(weight, bias).forward(device, (input, &mut output));
        output
    }
}

/// Head-major rows in HBM, each head's rows in order, project block by block into HBM; the owned
/// projection runs as its borrowed view.
impl<'x, In: AxisName, Out: AxisName, H: AxisName, Rows: AxisName, Chips: M>
    Op<&'x HbmTensor<bf16, Chips, m![In / ({ Symbol<H> }), Rows, In % ({ Symbol<H> })]>>
    for Biased<In, Out, Chips>
{
    type Output = HbmTensor<bf16, Chips, m![Rows, Out]>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'x HbmTensor<bf16, Chips, m![In / ({ Symbol<H> }), Rows, In % ({ Symbol<H> })]>,
    ) -> HbmTensor<bf16, Chips, m![Rows, Out]> {
        Affine::new(&self.weight, &self.bias).forward(device, input)
    }
}

impl<'x, 'w, In: AxisName, Out: AxisName, H: AxisName, Rows: AxisName, Chips: M>
    Op<&'x HbmTensor<bf16, Chips, m![In / ({ Symbol<H> }), Rows, In % ({ Symbol<H> })]>>
    for Affine<&'w HbmTensor<bf16, Chips, m![Out, In]>, &'w HbmTensor<f32, Chips, m![Out]>>
{
    type Output = HbmTensor<bf16, Chips, m![Rows, Out]>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'x HbmTensor<bf16, Chips, m![In / ({ Symbol<H> }), Rows, In % ({ Symbol<H> })]>,
    ) -> HbmTensor<bf16, Chips, m![Rows, Out]> {
        let mut output = HbmTensor::new();
        rows::Projection::new(self.weight, self.bias).forward(device, (input, &mut output));
        output
    }
}

/// BF16 coefficients with an FP32 bias.
pub type Biased<In, Out, Chips = m![1]> =
    Affine<HbmTensor<bf16, Chips, m![Out, In]>, HbmTensor<f32, Chips, m![Out]>>;

/// Rows in HBM project block by block into HBM, each output's bias added to its FP32 sum before
/// the one rounding; the owned projection runs as its borrowed view.
impl<'x, In: AxisName, Out: AxisName, Rows: AxisName, Chips: M>
    Op<&'x HbmTensor<bf16, Chips, m![Rows, In]>> for Biased<In, Out, Chips>
{
    type Output = HbmTensor<bf16, Chips, m![Rows, Out]>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'x HbmTensor<bf16, Chips, m![Rows, In]>,
    ) -> HbmTensor<bf16, Chips, m![Rows, Out]> {
        Affine::new(&self.weight, &self.bias).forward(device, input)
    }
}

impl<'x, 'w, In: AxisName, Out: AxisName, Rows: AxisName, Chips: M>
    Op<&'x HbmTensor<bf16, Chips, m![Rows, In]>>
    for Affine<&'w HbmTensor<bf16, Chips, m![Out, In]>, &'w HbmTensor<f32, Chips, m![Out]>>
{
    type Output = HbmTensor<bf16, Chips, m![Rows, Out]>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'x HbmTensor<bf16, Chips, m![Rows, In]>,
    ) -> HbmTensor<bf16, Chips, m![Rows, Out]> {
        let mut output = HbmTensor::new();
        rows::Projection::new(self.weight, self.bias).forward(device, (input, &mut output));
        output
    }
}

impl<In: AxisName, Out: AxisName, Rows: AxisName, Chips: M, S: M, F: M, T>
    Op<DmTensor<bf16, m![1], Chips, S, m![Rows, { F }]>> for Linear<In, Out, m![1], T>
where
    T: Matrix<In = In, Out = Out>,
{
    type Output = Projected<In, Out, Rows>;

    fn forward(
        &self,
        device: &mut Device,
        input: DmTensor<bf16, m![1], Chips, S, m![Rows, { F }]>,
    ) -> Projected<In, Out, Rows> {
        self.affine.forward(
            device,
            (Live::all().with(PhantomData), input.view(), |at| {
                mapping::Rows::<Rows>::group(input.view(), at)
            }),
        )
    }
}

/// The live row groups of a launch's rows projected by the dense contraction, in the feature
/// slices the next projection reads; the projected rows keep the launch's live rows.
impl<'x, In: AxisName, Out: AxisName, Rows: AxisName, Chips: M, S: M, F: M, W>
    Op<&'x Live<DmTensor<bf16, m![1], Chips, S, m![Rows, { F }]>, Rows>> for Affine<W, ()>
where
    W: Matrix<In = In, Out = Out>,
{
    type Output = Live<Spread<Out, Rows>, Rows>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'x Live<DmTensor<bf16, m![1], Chips, S, m![Rows, { F }]>, Rows>,
    ) -> Self::Output {
        let mut output = DmTensor::new();
        self.forward(
            &mut *device,
            (
                input.with(PhantomData::<Rows>),
                input.value().view(),
                |at| mapping::Rows::<Rows>::group(input.value().view(), at),
            ),
        )
        .view()
        .to_dm_view(&mut device.tdma, output.view_mut());
        input.with(output)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use furiosa_opt_std::prelude::*;
    use furiosa_opt_std::runtime::launch;

    use crate::ops::attention::geometry;
    use crate::ops::linear::kernel::Input;
    use crate::ops::linear::{self};
    use crate::ops::normalization::RowMajor;
    use crate::ops::normalization::tests::{blocked, random as values};
    use crate::tensor::axis;
    use crate::tensor::axis::{Eight, Single, Sixteen};
    use crate::tensor::live::Live;
    use crate::tensor::{live::tests::count, scalar::Value};
    use crate::tests::DEVICE;
    use crate::{Op, ops::linear::Linear, tensor::form::Weight};
    use core::marker::PhantomData;

    struct Fixture;

    type Hidden = geometry::Hidden<2048>;
    type QkvWidth = geometry::ProjectionWidth<4096>;
    type Chip = m![1];
    type Rows<In, Out, Count> = HbmTensor<bf16, Chip, Input<In, Out, Count>>;
    type Projected<Out, Count> = HbmTensor<bf16, Chip, RowMajor<Out, Count>>;

    /// `R` rows in HBM project as the FP64 product rounded once to BF16: the `whole` rows, and
    /// every `stride`-th row at a stride of columns that crosses every slice, pass and cluster.
    impl Fixture {
        async fn bands_match_the_product<In: AxisName, Out: AxisName, R: AxisName>(
            whole: &[usize],
            stride: usize,
        ) {
            let _lease = DEVICE.lock().await;
            let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
            let weight = values(Out::SIZE * In::SIZE, 0x9e37_79b9_7f4a_7c15);
            let inputs = values(R::SIZE * In::SIZE, 0x2545_f491_4f6c_dd1d);
            let widened = |values: &[bf16]| values.iter().map(|v| v.to_f32()).collect::<Vec<f32>>();
            let (rows, coefficients) = (widened(&inputs), widened(&weight));
            let linear: Linear<In, Out> = Linear::new(Weight::new(
                HostTensor::<bf16, m![Out, In]>::from_vec(weight)
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
            ));
            let input = HostTensor::<bf16, m![R, In]>::from_vec(inputs)
                .to_hbm::<Chip, m![R, In]>(&mut device.pdma)
                .await
                .unwrap();
            let output = launch(project_hbm, (&mut device, &linear, &input))
                .await
                .unwrap()
                .to_host::<m![R, Out]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            let check = |row: usize, column: usize| {
                let expected: f64 = rows[row * In::SIZE..][..In::SIZE]
                    .iter()
                    .zip(&coefficients[column * In::SIZE..][..In::SIZE])
                    .map(|(x, w)| f64::from(*x) * f64::from(*w))
                    .sum();
                let got = f64::from(output[row * Out::SIZE + column].to_f32());
                assert!(
                    (got - expected).abs() <= expected.abs() / 256.0 + 0.01,
                    "row {row} column {column}: {got} vs {expected}"
                );
            };
            for &row in whole {
                (0..Out::SIZE).for_each(|column| check(row, column));
            }
            for row in (0..R::SIZE).step_by(stride) {
                (0..Out::SIZE)
                    .step_by(97)
                    .for_each(|column| check(row, column));
            }
        }
    }

    /// `R` rows in HBM project as the FP64 product plus the column's bias, rounded once to BF16:
    /// every `stride`-th row at a stride of columns that crosses every slice and pass.
    impl Fixture {
        async fn biased_bands_match<In: AxisName, Out: AxisName, R: AxisName>(stride: usize) {
            let _lease = DEVICE.lock().await;
            let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
            let weight = values(Out::SIZE * In::SIZE, 0x9e37_79b9_7f4a_7c15);
            let inputs = values(R::SIZE * In::SIZE, 0x2545_f491_4f6c_dd1d);
            let bias: Vec<f32> = values(Out::SIZE, 0x1405_7b7e_f767_814f)
                .iter()
                .map(|v| v.to_f32() * 4.0)
                .collect();
            let linear = linear::Biased::<In, Out>::new(
                HostTensor::<bf16, m![Out, In]>::from_vec(weight.clone())
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
                HostTensor::<f32, m![Out]>::from_vec(bias.clone())
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
            );
            let input = HostTensor::<bf16, m![R, In]>::from_vec(inputs.clone())
                .to_hbm::<Chip, m![R, In]>(&mut device.pdma)
                .await
                .unwrap();
            let output = launch(project_biased, (&mut device, &linear, &input))
                .await
                .unwrap()
                .to_host::<m![R, Out]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            for row in (0..R::SIZE).step_by(stride) {
                for column in (0..Out::SIZE).step_by(37) {
                    let expected: f64 = inputs[row * In::SIZE..][..In::SIZE]
                        .iter()
                        .zip(&weight[column * In::SIZE..][..In::SIZE])
                        .map(|(x, w)| f64::from(x.to_f32()) * f64::from(w.to_f32()))
                        .sum::<f64>()
                        + f64::from(bias[column]);
                    let got = f64::from(output[row * Out::SIZE + column].to_f32());
                    assert!(
                        (got - expected).abs() <= expected.abs() / 256.0 + 0.01,
                        "row {row} column {column}: {got} vs {expected}"
                    );
                }
            }
        }
    }

    /// The input chunk a slice of the projection reads, in elements.
    const fn chunk<In: AxisName, Out: AxisName>() -> usize {
        <Linear<In, Out>::Chunk as M>::SIZE
    }

    /// `rows x In` times `Out x In` transposed, accumulated in f32 row by row as the reference.
    fn oracle(rows: &[bf16], weight: &[bf16], width: usize) -> Vec<f32> {
        let outputs = weight.len() / width;
        let mut output = Vec::with_capacity(rows.len() / width * outputs);
        for input in rows.chunks_exact(width) {
            for weight in weight.chunks_exact(width) {
                output.push(
                    input
                        .iter()
                        .zip(weight)
                        .map(|(x, w)| x.to_f32() * w.to_f32())
                        .sum(),
                );
            }
        }
        output
    }

    /// The largest error relative to the expected value, absolute where that is within one, so a
    /// near-zero sum is not judged in its own vanishing units.
    fn worst_error(actual: impl IntoIterator<Item = f32>, expected: &[f32]) -> f32 {
        let mut count = 0;
        let worst = actual
            .into_iter()
            .zip(expected)
            .map(|(a, e)| {
                count += 1;
                (a - e).abs() / e.abs().max(1.0)
            })
            .fold(0.0f32, f32::max);
        assert_eq!(count, expected.len());
        worst
    }

    #[device(chip = 1)]
    fn project_rows<In: AxisName, Out: AxisName, R: AxisName>(
        device: &mut Device,
        rows: &Value,
        dense: &linear::Affine<HbmTensor<bf16, Chip, m![Out, In]>, ()>,
        input: &Rows<In, Out, R>,
    ) -> Projected<Out, R> {
        let value: DmTensor<bf16, Chip, m![2], m![1 # 256], m![R, In]> =
            input.to_dm(&mut device.tdma);
        let input = Live::<(), R>::read(rows).with(value);
        dense
            .forward(&mut *device, &input)
            .value()
            .to_hbm(&mut device.tdma)
    }

    /// `Count` rows through the fused projection give what the host computes for each, and what
    /// one row gives alone, bit for bit: a lane holds one row and nothing else.
    async fn rows_project_as_alone<In: AxisName, Out: AxisName, Count: AxisName>() {
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let weight = values(Out::SIZE * In::SIZE, 0x9e37_79b9_7f4a_7c15);
        let inputs = values(Count::SIZE * In::SIZE, 0x2545_f491_4f6c_dd1d);
        let expected = oracle(&inputs, &weight, In::SIZE);
        let qkv = linear::Affine::new(
            HostTensor::<bf16, m![Out, In]>::from_vec(weight)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            (),
        );
        let rows_hbm = HostTensor::<bf16, Linear<In, Out>::Input<Count>>::from_vec(blocked(
            &inputs,
            In::SIZE,
            chunk::<In, Out>(),
        ))
        .to_hbm::<Chip, Linear<In, Out>::Input<Count>>(&mut device.pdma)
        .await
        .unwrap();
        let row_hbm = HostTensor::<bf16, Linear<In, Out>::Input<Single>>::from_vec(
            inputs[..In::SIZE].to_vec(),
        )
        .to_hbm::<Chip, Linear<In, Out>::Input<Single>>(&mut device.pdma)
        .await
        .unwrap();

        let every = count(Count::SIZE).to_hbm(&mut device.pdma).await.unwrap();
        let one = count(1).to_hbm(&mut device.pdma).await.unwrap();
        let batched = launch(project_rows, (&mut device, &every, &qkv, &rows_hbm))
            .await
            .unwrap()
            .to_host::<RowMajor<Out, Count>>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let single = launch(project_rows, (&mut device, &one, &qkv, &row_hbm))
            .await
            .unwrap()
            .to_host::<RowMajor<Out, Single>>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();

        let worst = worst_error(batched.iter().map(|value| value.to_f32()), &expected);
        assert!(worst <= 0.01, "{worst} relative error from the oracle");
        assert_eq!(
            single,
            batched[..Out::SIZE],
            "one row alone differs from row 0"
        );
        for row in 1..Count::SIZE {
            assert_ne!(
                batched[row * Out::SIZE..][..Out::SIZE],
                batched[..Out::SIZE],
                "row {row} repeats row 0"
            );
        }
    }

    /// Independent rows match the FP32 projection reference and the first row projected alone.
    #[tokio::test(flavor = "current_thread")]
    async fn eight_rows_project_as_alone() {
        rows_project_as_alone::<Hidden, QkvWidth, Eight>().await;
    }

    /// Sixteen rows take two passes over one weight in DM; each pass writes its own eight.
    #[tokio::test(flavor = "current_thread")]
    async fn sixteen_rows_project_as_alone() {
        rows_project_as_alone::<Hidden, QkvWidth, Sixteen>().await;
    }

    /// At every live row count of sixteen rows the groups holding a live row give what the
    /// unrolled projection gives, bit for bit, and a skipped group does not: its pass never ran.
    /// The scratch a skipped group writes is whatever an earlier launch left, so full and partial
    /// launches over other rows precede each one that is compared.
    async fn counts_groups<In: AxisName, Out: AxisName>() {
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let weight = values(Out::SIZE * In::SIZE, 0x9e37_79b9_7f4a_7c15);
        let qkv = linear::Affine::new(
            HostTensor::<bf16, m![Out, In]>::from_vec(weight)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            (),
        );
        let mut rows = Vec::new();
        for seed in [0x2545_f491_4f6c_dd1d, 0x0123_4567_89ab_cdef] {
            let inputs = values(Sixteen::SIZE * In::SIZE, seed);
            rows.push(
                HostTensor::<bf16, Linear<In, Out>::Input<Sixteen>>::from_vec(blocked(
                    &inputs,
                    In::SIZE,
                    chunk::<In, Out>(),
                ))
                .to_hbm::<Chip, Linear<In, Out>::Input<Sixteen>>(&mut device.pdma)
                .await
                .unwrap(),
            );
        }
        let [rows, other] = rows.as_slice() else {
            unreachable!("two seeds")
        };
        let mut counts = Vec::new();
        for live in [16, 9, 8, 1, 0] {
            counts.push((live, count(live).to_hbm(&mut device.pdma).await.unwrap()));
        }
        let unrolled = launch(project_rows, (&mut device, &counts[0].1, &qkv, rows))
            .await
            .unwrap();
        let read = async |device: &mut Device, output: Projected<Out, Sixteen>| {
            output
                .to_host::<RowMajor<Out, Sixteen>>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec()
        };
        let unrolled = read(&mut device, unrolled).await;
        let group = 8 * Out::SIZE;
        for (live, rows_live) in &counts {
            for (_, scratch) in &counts[..2] {
                launch(project_rows, (&mut device, scratch, &qkv, other))
                    .await
                    .unwrap();
            }
            let output = launch(project_rows, (&mut device, rows_live, &qkv, rows))
                .await
                .unwrap();
            let got = read(&mut device, output).await;
            for at in 0..2 {
                let range = at * group..(at + 1) * group;
                if at * 8 < *live {
                    assert_eq!(
                        got[range.clone()],
                        unrolled[range],
                        "{live} live rows: group {at} differs from the unrolled projection"
                    );
                } else {
                    assert_ne!(
                        got[range.clone()],
                        unrolled[range],
                        "{live} live rows: group {at} was computed"
                    );
                }
            }
        }
    }

    /// Rows in HBM projected into HBM.
    #[device(chip = 1)]
    fn project_hbm<In: AxisName, Out: AxisName, R: AxisName>(
        device: &mut Device,
        linear: &Linear<In, Out>,
        input: &HbmTensor<bf16, Chip, m![R, In]>,
    ) -> HbmTensor<bf16, Chip, m![R, Out]> {
        linear.forward(device, input)
    }
    /// The fused query, key and value projection over a 128-row text envelope, and one of its
    /// parts.
    #[tokio::test(flavor = "current_thread")]
    async fn qkv_bands_match_the_product() {
        Fixture::bands_match_the_product::<
            geometry::Hidden<5376>,
            geometry::ProjectionWidth<21504>,
            axis::Rows<128>,
        >(&[0, 1, 63, 64, 127], 1)
        .await;
        Fixture::bands_match_the_product::<
            geometry::Hidden<5376>,
            geometry::ProjectionWidth<7168>,
            axis::Rows<128>,
        >(&[0, 63, 64, 127], 1)
        .await;
    }

    /// The attention output projection and a 5,120-feature input projection over 128 rows.
    #[tokio::test(flavor = "current_thread")]
    async fn output_bands_match_the_product() {
        Fixture::bands_match_the_product::<
            geometry::Hidden<7168>,
            geometry::ProjectionWidth<5376>,
            axis::Rows<128>,
        >(&[0, 63, 64, 127], 1)
        .await;
        Fixture::bands_match_the_product::<
            geometry::Hidden<5120>,
            geometry::ProjectionWidth<5376>,
            axis::Rows<128>,
        >(&[0, 63, 64, 127], 1)
        .await;
    }

    /// One chip's parts when four chips split 37,888 rows: a quarter of the query, key and value
    /// columns over every row, then output, gate/up and down over a quarter of the rows.
    #[tokio::test(flavor = "current_thread")]
    async fn shards_match_the_product() {
        type Rows<const N: usize> = axis::Rows<N>;
        Fixture::bands_match_the_product::<
            geometry::Hidden<5376>,
            geometry::ProjectionWidth<5376>,
            Rows<37888>,
        >(&[0, 37_887], 1001)
        .await;
        Fixture::bands_match_the_product::<
            geometry::Hidden<7168>,
            geometry::ProjectionWidth<5376>,
            Rows<9472>,
        >(&[0, 9471], 501)
        .await;
        Fixture::bands_match_the_product::<
            geometry::Hidden<5376>,
            geometry::ProjectionWidth<28672>,
            Rows<9472>,
        >(&[0, 9471], 501)
        .await;
        Fixture::bands_match_the_product::<
            geometry::Hidden<14336>,
            geometry::ProjectionWidth<5376>,
            Rows<9472>,
        >(&[0, 9471], 501)
        .await;
    }

    #[device(chip = 1)]
    fn project_biased<In: AxisName, Out: AxisName, R: AxisName>(
        device: &mut Device,
        linear: &linear::Biased<In, Out>,
        input: &HbmTensor<bf16, Chip, m![R, In]>,
    ) -> HbmTensor<bf16, Chip, m![R, Out]> {
        linear.forward(device, input)
    }
    #[device(chip = 1)]
    fn project_unrounded<In: AxisName, Out: AxisName, R: AxisName>(
        device: &mut Device,
        linear: &linear::Unrounded<In, Out>,
        input: &HbmTensor<bf16, Chip, m![R, In]>,
    ) -> HbmTensor<f32, Chip, m![R, Out]> {
        linear.forward(device, input)
    }

    /// A 2,048-row tile's 2048 -> 3072 projection gives every FP32 sum plus its bias unrounded,
    /// within FP32 accumulation error of the FP64 product.
    #[tokio::test(flavor = "current_thread")]
    async fn patch_projection_writes_fp32_sums() {
        let _lease = DEVICE.lock().await;
        type In = geometry::Hidden<2048>;
        type Out = geometry::ProjectionWidth<3072>;
        type R = axis::Rows<2048>;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let weight = values(Out::SIZE * In::SIZE, 0x9e37_79b9_7f4a_7c15);
        let inputs = values(R::SIZE * In::SIZE, 0x2545_f491_4f6c_dd1d);
        let bias: Vec<f32> = values(Out::SIZE, 0x1405_7b7e_f767_814f)
            .iter()
            .map(|v| v.to_f32() * 4.0)
            .collect();
        let linear = linear::Unrounded::<In, Out>(
            linear::Biased::new(
                HostTensor::<bf16, m![Out, In]>::from_vec(weight.clone())
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
                HostTensor::<f32, m![Out]>::from_vec(bias.clone())
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
            ),
            PhantomData,
        );
        let input = HostTensor::<bf16, m![R, In]>::from_vec(inputs.clone())
            .to_hbm::<Chip, m![R, In]>(&mut device.pdma)
            .await
            .unwrap();
        let output = launch(project_unrounded, (&mut device, &linear, &input))
            .await
            .unwrap()
            .to_host::<m![R, Out]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for row in (0..R::SIZE).step_by(7) {
            for column in (0..Out::SIZE).step_by(37) {
                let expected: f64 = inputs[row * In::SIZE..][..In::SIZE]
                    .iter()
                    .zip(&weight[column * In::SIZE..][..In::SIZE])
                    .map(|(x, w)| f64::from(x.to_f32()) * f64::from(w.to_f32()))
                    .sum::<f64>()
                    + f64::from(bias[column]);
                let got = f64::from(output[row * Out::SIZE + column]);
                assert!(
                    (got - expected).abs() <= 1e-4 * (expected.abs() + 1.0),
                    "row {row} column {column}: {got} vs {expected}"
                );
            }
        }
    }

    /// The biased projections of a 2,048-row tile: query, key and value, output, gate/up and down.
    #[tokio::test(flavor = "current_thread")]
    async fn tile_projections_add_their_bias() {
        type Rows = axis::Rows<2048>;
        type In<const N: usize> = geometry::Hidden<N>;
        type Out<const N: usize> = geometry::ProjectionWidth<N>;
        Fixture::biased_bands_match::<In<2048>, Out<6144>, Rows>(7).await;
        Fixture::biased_bands_match::<In<2048>, Out<2048>, Rows>(7).await;
        Fixture::biased_bands_match::<In<2048>, Out<16384>, Rows>(7).await;
        Fixture::biased_bands_match::<In<8192>, Out<2048>, Rows>(7).await;
    }

    /// A media embedder's projections into a 3840-wide text model: merged patches, audio frames
    /// and soft tokens.
    #[tokio::test(flavor = "current_thread")]
    async fn embedder_projections_add_their_bias() {
        type Rows = axis::Rows<256>;
        type In<const N: usize> = geometry::Hidden<N>;
        type Out<const N: usize> = geometry::ProjectionWidth<N>;
        Fixture::biased_bands_match::<In<6912>, Out<3840>, Rows>(7).await;
        Fixture::biased_bands_match::<In<3840>, Out<3840>, Rows>(7).await;
        Fixture::biased_bands_match::<In<640>, Out<3840>, Rows>(7).await;
    }

    #[device(chip = 1)]
    fn project_wide<In: AxisName, Out: AxisName, R: AxisName>(
        device: &mut Device,
        linear: &Linear<In, Out>,
        input: &HbmTensor<bf16, Chip, m![R, In]>,
    ) -> HbmTensor<f32, Chip, m![R, Out]> {
        let mut output = HbmTensor::new();
        linear.forward(&mut *device, (input, &mut output));
        output
    }

    /// Rows in HBM hold every FP32 sum unrounded: each output within FP32 accumulation error of
    /// the FP64 product.
    #[tokio::test(flavor = "current_thread")]
    async fn wide_bands_write_fp32_sums() {
        let _lease = DEVICE.lock().await;
        type In = geometry::Hidden<5376>;
        type Out = geometry::ProjectionWidth<192>;
        type R = axis::Rows<128>;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let weight = values(Out::SIZE * In::SIZE, 0x9e37_79b9_7f4a_7c15);
        let inputs = values(R::SIZE * In::SIZE, 0x2545_f491_4f6c_dd1d);
        let expected: Vec<f64> = inputs
            .chunks_exact(In::SIZE)
            .flat_map(|row| {
                weight.chunks_exact(In::SIZE).map(move |w| {
                    row.iter()
                        .zip(w)
                        .map(|(x, w)| f64::from(x.to_f32()) * f64::from(w.to_f32()))
                        .sum()
                })
            })
            .collect();
        let linear: Linear<In, Out> = Linear::new(Weight::new(
            HostTensor::<bf16, m![Out, In]>::from_vec(weight)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
        ));
        let input = HostTensor::<bf16, m![R, In]>::from_vec(inputs)
            .to_hbm::<Chip, m![R, In]>(&mut device.pdma)
            .await
            .unwrap();
        let output = launch(project_wide, (&mut device, &linear, &input))
            .await
            .unwrap()
            .to_host::<m![R, Out]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (index, (got, expected)) in output.iter().zip(&expected).enumerate() {
            assert!(
                (f64::from(*got) - expected).abs() <= 1e-4 * (expected.abs() + 1.0),
                "row {} column {}: {got} vs {expected}",
                index / Out::SIZE,
                index % Out::SIZE
            );
        }
    }
    /// Every projection shape a dense model uses counts its live groups.
    #[tokio::test(flavor = "current_thread")]
    async fn only_the_counted_groups_are_projected() {
        counts_groups::<geometry::Hidden<1024>, geometry::ProjectionWidth<4096>>().await;
        counts_groups::<geometry::Hidden<2048>, geometry::ProjectionWidth<1024>>().await;
        counts_groups::<geometry::Hidden<2048>, geometry::ProjectionWidth<4096>>().await;
        counts_groups::<geometry::Hidden<2048>, geometry::ProjectionWidth<2048>>().await;
        counts_groups::<geometry::Hidden<2560>, geometry::ProjectionWidth<6144>>().await;
        counts_groups::<geometry::Hidden<4096>, geometry::ProjectionWidth<2560>>().await;
        counts_groups::<geometry::Hidden<4096>, geometry::ProjectionWidth<6144>>().await;
    }
}
