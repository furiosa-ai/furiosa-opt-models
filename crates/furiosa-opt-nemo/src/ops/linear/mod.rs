//! BF16 row projections and affine parameter forms.

mod affine;
pub mod contraction;
pub use affine::{mlp, output, pair, query};
pub mod experts;
pub mod fp8;
pub(crate) mod kernel;
pub mod nvfp4;
mod projection;
pub(crate) mod rows;

pub use affine::{
    Affine, Bias, COLUMNS, Coefficients, Columns, Context, Inputs, Nvfp4, Outputs, Weights,
};
pub use experts::Experts;
pub(crate) use kernel::Matrix;
pub use kernel::{Input, Output};
pub(crate) use rows::Projection;

use core::marker::PhantomData;

use furiosa_opt_std::prelude::*;

use crate::Module;
use crate::ops::layout::{Dm, Live, axes::Lanes};
use crate::ops::normalization::Spread;
use crate::ops::weight::Weight;

/// Multiply BF16 input rows by a retained [Out, In] weight matrix transposed.
#[derive(DeviceSend)]
pub struct Linear<In: AxisName, Out: AxisName, Chips: M = m![1]> {
    pub(crate) affine: Affine<HbmTensor<bf16, Chips, m![Out, In]>, ()>,
}

impl<In: AxisName, Out: AxisName, Chips: M> Linear<In, Out, Chips> {
    /// Retain the BF16 [Out, In] matrix without copying it.
    pub fn new(weight: Weight<m![Out, In], Chips>) -> Self {
        Self {
            affine: Affine::new(weight.into_tensor(), ()),
        }
    }

    /// Borrow the contiguous [Out, In] weight matrix on each chip.
    pub fn tensor(&self) -> &HbmTensor<bf16, Chips, m![Out, In]> {
        &self.affine.weight
    }
}

pub type Groups<Rows> = crate::ops::layout::Rows<Rows>::Groups;

/// Projected rows in the resident contraction's native mapping.
pub type Projected<In, Out, Rows> = kernel::Result<In, Out, Rows>;

// Rows stay explicit while the feature partition belongs to the input tensor. A consumer does
// not require the producer's hidden width or a particular cluster/slice placement.
impl<'d, 'x, In: AxisName, Out: AxisName, Rows: AxisName, Chips: M, S: M, F: M>
    Module<(
        &'d mut Device,
        &'x DmTensor<bf16, m![1], Chips, S, m![Rows, { F }]>,
    )> for Linear<In, Out>
{
    type Output = Projected<In, Out, Rows>;

    fn forward(
        &self,
        (device, input): (
            &'d mut Device,
            &'x DmTensor<bf16, m![1], Chips, S, m![Rows, { F }]>,
        ),
    ) -> Projected<In, Out, Rows> {
        self.affine.forward((
            device,
            (Live::<Rows>::over(Rows::SIZE), input.view(), |at| {
                crate::ops::layout::Rows::<Rows>::group(input.view(), at)
            }),
        ))
    }
}

impl<'d, 'x, In: AxisName, Out: AxisName, Rows: AxisName, Chips: M, S: M, F: M>
    Module<(
        &'d mut Device,
        (
            Live<Rows>,
            &'x DmTensor<bf16, m![1], Chips, S, m![Rows, { F }]>,
        ),
    )> for Linear<In, Out>
{
    type Output = Projected<In, Out, Rows>;

    fn forward(
        &self,
        (device, (live, input)): (
            &'d mut Device,
            (
                Live<Rows>,
                &'x DmTensor<bf16, m![1], Chips, S, m![Rows, { F }]>,
            ),
        ),
    ) -> Projected<In, Out, Rows> {
        self.affine.forward((
            device,
            (live, input.view(), |at| {
                crate::ops::layout::Rows::<Rows>::group(input.view(), at)
            }),
        ))
    }
}

/// Rows in HBM, too many for data memory, project block by block into HBM.
impl<'d, 'x, In: AxisName, Out: AxisName, Rows: AxisName, X: M>
    Module<(&'d mut Device, &'x HbmTensor<bf16, X, m![Rows, In]>)> for Linear<In, Out, X>
{
    type Output = HbmTensor<bf16, X, m![Rows, Out]>;

    fn forward(
        &self,
        (device, input): (&'d mut Device, &'x HbmTensor<bf16, X, m![Rows, In]>),
    ) -> HbmTensor<bf16, X, m![Rows, Out]> {
        let mut output = HbmTensor::new();
        rows::Projection::new(&self.affine.weight, &()).forward((device, (input, &mut output)));
        output
    }
}

/// BF16 coefficients with an FP32 bias whose FP32 sums land unrounded.
#[derive(DeviceSend)]
pub struct Unrounded<In: AxisName, Out: AxisName, Chips: M = m![1]>(pub Biased<In, Out, Chips>);

/// Rows in HBM project block by block into FP32 rows in HBM, each output's bias added to its sum.
impl<'d, 'x, In: AxisName, Out: AxisName, Rows: AxisName, Chips: M>
    Module<(&'d mut Device, &'x HbmTensor<bf16, Chips, m![Rows, In]>)>
    for Unrounded<In, Out, Chips>
{
    type Output = HbmTensor<f32, Chips, m![Rows, Out]>;

    fn forward(
        &self,
        (device, input): (&'d mut Device, &'x HbmTensor<bf16, Chips, m![Rows, In]>),
    ) -> HbmTensor<f32, Chips, m![Rows, Out]> {
        let mut output = HbmTensor::new();
        rows::Projection::new(&self.0.weight, &self.0.bias).forward((device, (input, &mut output)));
        output
    }
}

/// A biased projection whose output rows land head-major, every head of `H` features its rows in
/// order.
#[derive(DeviceSend)]
pub struct PerHead<In: AxisName, Out: AxisName, H: AxisName, Chips: M = m![1]>(
    pub Biased<In, Out, Chips>,
    pub PhantomData<H>,
);

/// Rows in HBM project block by block into head-major rows in HBM.
impl<'d, 'x, In: AxisName, Out: AxisName, H: AxisName, Rows: AxisName, Chips: M>
    Module<(&'d mut Device, &'x HbmTensor<bf16, Chips, m![Rows, In]>)>
    for PerHead<In, Out, H, Chips>
{
    type Output = HbmTensor<bf16, Chips, m![Out / ({ Symbol<H> }), Rows, Out % ({ Symbol<H> })]>;

    fn forward(
        &self,
        (device, input): (&'d mut Device, &'x HbmTensor<bf16, Chips, m![Rows, In]>),
    ) -> Self::Output {
        let mut output = HbmTensor::new();
        rows::Projection::new(&self.0.weight, &self.0.bias).forward((device, (input, &mut output)));
        output
    }
}

/// Head-major rows in HBM, each head's rows in order, project block by block into HBM.
impl<'d, 'x, In: AxisName, Out: AxisName, H: AxisName, Rows: AxisName, Chips: M>
    Module<(
        &'d mut Device,
        &'x HbmTensor<bf16, Chips, m![In / ({ Symbol<H> }), Rows, In % ({ Symbol<H> })]>,
    )> for Biased<In, Out, Chips>
{
    type Output = HbmTensor<bf16, Chips, m![Rows, Out]>;

    fn forward(
        &self,
        (device, input): (
            &'d mut Device,
            &'x HbmTensor<bf16, Chips, m![In / ({ Symbol<H> }), Rows, In % ({ Symbol<H> })]>,
        ),
    ) -> HbmTensor<bf16, Chips, m![Rows, Out]> {
        let mut output = HbmTensor::new();
        rows::Projection::new(&self.weight, &self.bias).forward((device, (input, &mut output)));
        output
    }
}

/// BF16 coefficients with an FP32 bias.
pub type Biased<In, Out, Chips = m![1]> =
    Affine<HbmTensor<bf16, Chips, m![Out, In]>, HbmTensor<f32, Chips, m![Out]>>;

/// Rows in HBM project block by block into HBM, each output's bias added to its FP32 sum before
/// the one rounding.
impl<'d, 'x, In: AxisName, Out: AxisName, Rows: AxisName, Chips: M>
    Module<(&'d mut Device, &'x HbmTensor<bf16, Chips, m![Rows, In]>)> for Biased<In, Out, Chips>
{
    type Output = HbmTensor<bf16, Chips, m![Rows, Out]>;

    fn forward(
        &self,
        (device, input): (&'d mut Device, &'x HbmTensor<bf16, Chips, m![Rows, In]>),
    ) -> HbmTensor<bf16, Chips, m![Rows, Out]> {
        let mut output = HbmTensor::new();
        rows::Projection::new(&self.weight, &self.bias).forward((device, (input, &mut output)));
        output
    }
}

impl<'d, In: AxisName, Out: AxisName, Rows: AxisName, Chips: M, S: M, F: M>
    Module<(
        &'d mut Device,
        DmTensor<bf16, m![1], Chips, S, m![Rows, { F }]>,
    )> for Linear<In, Out>
{
    type Output = Projected<In, Out, Rows>;

    fn forward(
        &self,
        (device, input): (
            &'d mut Device,
            DmTensor<bf16, m![1], Chips, S, m![Rows, { F }]>,
        ),
    ) -> Projected<In, Out, Rows> {
        self.affine.forward((
            device,
            (Live::<Rows>::over(Rows::SIZE), input.view(), |at| {
                crate::ops::layout::Rows::<Rows>::group(input.view(), at)
            }),
        ))
    }
}

/// The live row groups of a launch's rows projected by the dense contraction, in the feature
/// slices the next projection reads; the projected rows keep the launch's live tiles.
impl<'d, 'x, In: AxisName, Out: AxisName, Rows: AxisName, Chips: M, S: M, F: M, W>
    Module<(
        &'d mut Device,
        &'x Dm<Lanes<Rows>, DmTensor<bf16, m![1], Chips, S, m![Rows, { F }]>>,
    )> for Affine<W, ()>
where
    W: Matrix<In = In, Out = Out>,
{
    type Output = Dm<Lanes<Rows>, Spread<Out, Rows>>;

    fn forward(
        &self,
        (device, input): (
            &'d mut Device,
            &'x Dm<Lanes<Rows>, DmTensor<bf16, m![1], Chips, S, m![Rows, { F }]>>,
        ),
    ) -> Self::Output {
        let mut output = DmTensor::new();
        self.forward((
            &mut *device,
            (input.tiles, input.value.view(), |at| {
                crate::ops::layout::Rows::<Rows>::group(input.value.view(), at)
            }),
        ))
        .view()
        .to_dm_view(&mut device.tdma, output.view_mut());
        Dm {
            value: output,
            tiles: input.tiles,
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    struct Fixture;
    use furiosa_opt_std::prelude::*;
    use furiosa_opt_std::runtime::launch;

    use crate::compose::{Count, tests::count};
    use crate::ops::attention::geometry;
    use crate::ops::layout::axes::{Eight, Single, Sixteen};
    use crate::ops::linear::{self};
    use crate::ops::normalization::RowMajor;
    use crate::ops::normalization::tests::{blocked, random as values};
    use crate::{Module, ops::linear::Linear, ops::weight::Weight};

    type Hidden = geometry::Hidden<2048>;
    type QkvWidth = geometry::ProjectionWidth<4096>;
    type Chip = m![1];
    type Rows<In, Out, Count> = HbmTensor<bf16, Chip, linear::Input<In, Out, Count>>;
    type Projected<Out, Count> = HbmTensor<bf16, Chip, RowMajor<Out, Count>>;

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
        rows: &Count,
        dense: &linear::Affine<HbmTensor<bf16, Chip, m![Out, In]>, ()>,
        input: &Rows<In, Out, R>,
    ) -> Projected<Out, R> {
        let input: crate::ops::layout::Dm<_, DmTensor<bf16, Chip, m![2], m![1 # 256], m![R, In]>> =
            crate::ops::layout::Dm {
                value: input.to_dm(&mut device.tdma),
                tiles: crate::ops::layout::Live::<R>::read(rows),
            };
        dense
            .forward((&mut *device, &input))
            .value
            .to_hbm(&mut device.tdma)
    }

    /// `Count` rows through the fused projection land what the host computes for each, and what
    /// one row lands alone, bit for bit: a lane holds one row and nothing else.
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

    /// Sixteen rows take two passes over one weight in DM; each pass lands its own eight.
    #[tokio::test(flavor = "current_thread")]
    async fn sixteen_rows_project_as_alone() {
        rows_project_as_alone::<Hidden, QkvWidth, Sixteen>().await;
    }

    /// At every live row count of sixteen rows the groups holding a live row land what the
    /// unrolled projection lands, bit for bit, and a skipped group does not: its pass never ran.
    /// The scratch a skipped group lands is whatever an earlier launch left, so full and partial
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
        let land = async |device: &mut Device, output: Projected<Out, Sixteen>| {
            output
                .to_host::<RowMajor<Out, Sixteen>>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec()
        };
        let unrolled = land(&mut device, unrolled).await;
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
            let landed = land(&mut device, output).await;
            for at in 0..2 {
                let range = at * group..(at + 1) * group;
                if at * 8 < *live {
                    assert_eq!(
                        landed[range.clone()],
                        unrolled[range],
                        "{live} live rows: group {at} differs from the unrolled projection"
                    );
                } else {
                    assert_ne!(
                        landed[range.clone()],
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
        linear.forward((device, input))
    }

    /// `R` rows in HBM project as the FP64 product rounded once to BF16: the `whole` rows, and
    /// every `stride`-th row at a stride of columns that crosses every slice, pass and cluster.
    impl Fixture {
        async fn bands_match_the_product<In: AxisName, Out: AxisName, R: AxisName>(
            whole: &[usize],
            stride: usize,
        ) {
            let _lease = crate::tests::DEVICE.lock().await;
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
    /// The fused query, key and value projection over a 128-row text envelope, and one of its
    /// parts.
    #[tokio::test(flavor = "current_thread")]
    async fn qkv_bands_match_the_product() {
        Fixture::bands_match_the_product::<
            geometry::Hidden<5376>,
            geometry::ProjectionWidth<21504>,
            crate::ops::layout::axes::Rows<128>,
        >(&[0, 1, 63, 64, 127], 1)
        .await;
        Fixture::bands_match_the_product::<
            geometry::Hidden<5376>,
            geometry::ProjectionWidth<7168>,
            crate::ops::layout::axes::Rows<128>,
        >(&[0, 63, 64, 127], 1)
        .await;
    }

    /// The attention output projection and a 5,120-feature input projection over 128 rows.
    #[tokio::test(flavor = "current_thread")]
    async fn output_bands_match_the_product() {
        Fixture::bands_match_the_product::<
            geometry::Hidden<7168>,
            geometry::ProjectionWidth<5376>,
            crate::ops::layout::axes::Rows<128>,
        >(&[0, 63, 64, 127], 1)
        .await;
        Fixture::bands_match_the_product::<
            geometry::Hidden<5120>,
            geometry::ProjectionWidth<5376>,
            crate::ops::layout::axes::Rows<128>,
        >(&[0, 63, 64, 127], 1)
        .await;
    }

    /// One chip's parts when four chips split 37,888 rows: a quarter of the query, key and value
    /// columns over every row, then output, gate/up and down over a quarter of the rows.
    #[tokio::test(flavor = "current_thread")]
    async fn shards_match_the_product() {
        type Rows<const N: usize> = crate::ops::layout::axes::Rows<N>;
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
        linear.forward((device, input))
    }

    /// `R` rows in HBM project as the FP64 product plus the column's bias, rounded once to BF16:
    /// every `stride`-th row at a stride of columns that crosses every slice and pass.
    impl Fixture {
        async fn biased_bands_match<In: AxisName, Out: AxisName, R: AxisName>(stride: usize) {
            let _lease = crate::tests::DEVICE.lock().await;
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
    #[device(chip = 1)]
    fn project_unrounded<In: AxisName, Out: AxisName, R: AxisName>(
        device: &mut Device,
        linear: &linear::Unrounded<In, Out>,
        input: &HbmTensor<bf16, Chip, m![R, In]>,
    ) -> HbmTensor<f32, Chip, m![R, Out]> {
        linear.forward((device, input))
    }

    /// A 2,048-row tile's 2048 -> 3072 projection lands every FP32 sum plus its bias unrounded,
    /// within FP32 accumulation error of the FP64 product.
    #[tokio::test(flavor = "current_thread")]
    async fn patch_projection_lands_fp32_sums() {
        let _lease = crate::tests::DEVICE.lock().await;
        type In = geometry::Hidden<2048>;
        type Out = geometry::ProjectionWidth<3072>;
        type R = crate::ops::layout::axes::Rows<2048>;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let weight = values(Out::SIZE * In::SIZE, 0x9e37_79b9_7f4a_7c15);
        let inputs = values(R::SIZE * In::SIZE, 0x2545_f491_4f6c_dd1d);
        let bias: Vec<f32> = values(Out::SIZE, 0x1405_7b7e_f767_814f)
            .iter()
            .map(|v| v.to_f32() * 4.0)
            .collect();
        let linear = linear::Unrounded::<In, Out>(linear::Biased::new(
            HostTensor::<bf16, m![Out, In]>::from_vec(weight.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            HostTensor::<f32, m![Out]>::from_vec(bias.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
        ));
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
        type Rows = crate::ops::layout::axes::Rows<2048>;
        type In<const N: usize> = geometry::Hidden<N>;
        type Out<const N: usize> = geometry::ProjectionWidth<N>;
        Fixture::biased_bands_match::<In<2048>, Out<6144>, Rows>(7).await;
        Fixture::biased_bands_match::<In<2048>, Out<2048>, Rows>(7).await;
        Fixture::biased_bands_match::<In<2048>, Out<16384>, Rows>(7).await;
        Fixture::biased_bands_match::<In<8192>, Out<2048>, Rows>(7).await;
    }

    #[device(chip = 1)]
    fn project_wide<In: AxisName, Out: AxisName, R: AxisName>(
        device: &mut Device,
        linear: &Linear<In, Out>,
        input: &HbmTensor<bf16, Chip, m![R, In]>,
    ) -> HbmTensor<f32, Chip, m![R, Out]> {
        let mut output = HbmTensor::new();
        linear.forward((&mut *device, (input, &mut output)));
        output
    }

    /// Rows in HBM land every FP32 sum unrounded: each output within FP32 accumulation error of
    /// the FP64 product.
    #[tokio::test(flavor = "current_thread")]
    async fn wide_bands_land_fp32_sums() {
        let _lease = crate::tests::DEVICE.lock().await;
        type In = geometry::Hidden<5376>;
        type Out = geometry::ProjectionWidth<192>;
        type R = crate::ops::layout::axes::Rows<128>;
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
