//! Rows times transposed projection weights, with an optional per-output bias.

use crate::Op;
use crate::load::Load;
use crate::modules::generic::schedule::{IntoStream, RowWise, Stream};
use crate::ops::index::gather;
use crate::ops::normalization;
use crate::ops::normalization::{Cluster, FeatureSlices};
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

pub mod output;
pub mod query;

/// Projection coefficients in the device form a projection computes with.
#[derive(Clone, DeviceSend, Load)]
pub struct Affine<Matrix, Offset> {
    /// Projection weights in [output, input] order, or a quantized weight/scale tuple.
    pub weight: Matrix,
    /// Per-output offsets; the chosen weight form determines the no-bias representation.
    pub bias: Offset,
}

impl<Matrix, Offset> Affine<Matrix, Offset> {
    /// Retain the weight and bias representations for the selected projection form.
    pub fn new(weight: Matrix, bias: Offset) -> Self {
        Self { weight, bias }
    }
}

/// Relabel a feature axis as the projection's input coordinate.
#[derive(Clone, Debug)]
pub struct Inputs<A>(PhantomData<A>);

impl<A: AxisName> AxisName for Inputs<A> {
    const NAME: Ident = Ident::new("Inputs");
    const SIZE: usize = A::SIZE;
}

/// Relabel a feature axis as the projection's output coordinate.
#[derive(Clone, Debug)]
pub struct Outputs<A>(PhantomData<A>);

impl<A: AxisName> AxisName for Outputs<A> {
    const NAME: Ident = Ident::new("Outputs");
    const SIZE: usize = A::SIZE;
}

/// Dense [output, input] coefficients with zero or one per-output bias tensor.
pub(crate) type Coefficients<D, In, Out, const BIAS: usize> = Affine<
    HbmTensor<D, m![1], m![{ Symbol<Outputs<Out>> }, { Symbol<Inputs<In>> }]>,
    [HbmTensor<D, m![1], m![{ Symbol<Outputs<Out>> }]>; BIAS],
>;

impl<D: Scalar, W: Scalar, In: AxisName, Out: AxisName, Rows: AxisName, const BIAS: usize>
    Op<&HbmTensor<D, m![1], m![Rows, In]>> for Coefficients<W, In, Out, BIAS>
{
    type Output = HbmTensor<D, m![1], m![Rows, Out]>;

    fn forward(&self, _: &mut Device, _: &HbmTensor<D, m![1], m![Rows, In]>) -> Self::Output {
        const { assert!(BIAS <= 1) };
        todo!("NPU affine projection over contiguous rows")
    }
}

impl<D: Scalar, W: Scalar, In: AxisName, Out: AxisName, Rows: AxisName, const BIAS: usize>
    Op<HbmTensor<D, m![1], m![Rows, In]>> for Coefficients<W, In, Out, BIAS>
{
    type Output = HbmTensor<D, m![1], m![Rows, Out]>;

    fn forward(
        &self,
        device: &mut Device,
        input: HbmTensor<D, m![1], m![Rows, In]>,
    ) -> Self::Output {
        self.forward(device, &input)
    }
}

impl<D: Scalar, W: Scalar, In: AxisName, Out: AxisName, Rows: AxisName, const BIAS: usize>
    Op<&HbmTensor<D, m![1], m![Rows, In]>>
    for Affine<HbmTensor<W, m![1], m![Out, In]>, [HbmTensor<W, m![1], m![Out]>; BIAS]>
{
    type Output = HbmTensor<D, m![1], m![Rows, Out]>;

    fn forward(&self, _: &mut Device, _: &HbmTensor<D, m![1], m![Rows, In]>) -> Self::Output {
        const { assert!(BIAS <= 1) };
        todo!("NPU affine projection over contiguous rows")
    }
}

/// One contiguous input-width FP32 weight row for each runtime output column.
pub type Weights<In> = Slice<HbmTensor<f32, m![1], m![In]>>;

/// Per-output FP32 bias at element zero of each four-element padded row.
pub type Bias = Slice<HbmTensor<f32, m![1], m![4]>>;

/// The most output columns a runtime-width projection computes.
pub const COLUMNS: usize = 256;

/// The output columns a runtime-width projection compiles: one per weight row, then padding.
#[derive(Clone, Debug)]
pub struct Columns;

impl AxisName for Columns {
    const NAME: Ident = Ident::new("Columns");
    const SIZE: usize = COLUMNS;
}

/// One output column of every row, among the columns the other weight rows write.
type Column<Rows> = m![Rows, Columns = 1 #{!} Columns];

/// Each row's FP32 sum, held by every slice of both clusters.
type Sums<Rows> = DmTensor<f32, m![1], Cluster, m![256], m![Rows, 1 # 8]>;

/// Column `c` of each row is the row's dot product with weight row `c`, plus bias `c` when the bias
/// is not empty, in FP32; columns past the weight rows are left as they were.
impl<X: MaterializableScalar + FetchCast<f32>, In: AxisName, Rows: AxisName>
    Op<(
        &HbmTensor<X, m![1], m![Rows, In]>,
        &mut HbmTensor<f32, m![1], m![Rows, Columns]>,
    )> for Affine<Weights<In>, Bias>
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (input, output): (
            &HbmTensor<X, m![1], m![Rows, In]>,
            &mut HbmTensor<f32, m![1], m![Rows, Columns]>,
        ),
    ) {
        type Chunk<In> = Symbol<normalization::Chunk<In>>;
        let input: DmTensor<X, m![1], Cluster, FeatureSlices<In>, m![Rows, In % ({ Chunk<In> })]> =
            input.to_dm(&mut device.tdma);
        let store = |device: &mut Device,
                     sums: Sums<Rows>,
                     column: HbmTensorViewMut<'_, f32, m![1], Column<Rows>>| {
            // SAFETY: every slice of both clusters holds the same sums; the transfer reads one.
            let sums: DmTensor<f32, m![1], m![1 # 2], m![1 # 256], m![Rows, 1 # 8]> =
                unsafe { sums.reshape() };
            let padded: HbmTensor<f32, m![1], m![Rows, 1 # 4]> = sums.to_hbm(&mut device.tdma);
            padded.view().to_hbm_view(&mut device.tdma, column);
        };
        for index in 0..COLUMNS {
            if index < self.weight.len() {
                let weight: DmTensor<
                    f32,
                    m![1],
                    Cluster,
                    FeatureSlices<In>,
                    m![In % ({ Chunk<In> })],
                > = self.weight[index].to_dm(&mut device.tdma);
                let weight: VrfTensor<
                    f32,
                    m![1],
                    Cluster,
                    FeatureSlices<In>,
                    m![In % ({ Chunk<In> })],
                > = device
                    .sub
                    .begin(weight.view())
                    .fetch::<m![In % ({ Chunk<In> }) / 8], m![In % 8]>()
                    .collect::<m![In % ({ Chunk<In> }) / 8], m![In % 8]>()
                    .to_vrf();
                let sums: Sums<Rows> = device
                    .main
                    .begin(input.view())
                    .fetch::<m![Rows, In % ({ Chunk<In> }) / 8], m![In % 8]>()
                    .fetch_cast::<f32>()
                    .collect::<m![Rows, In % ({ Chunk<In> }) / 8], m![In % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<
                        m![Rows, In % ({ Chunk<In> }) / 8, In % 8 / 4 % 2],
                        m![In % 4],
                    >()
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight)
                    .vector_intra_slice_reduce::<
                        In,
                        m![Rows],
                        m![1 # 4],
                    >(IntraSliceReduceOpF32::Add)
                    .vector_widen_pad::<m![1 # 8]>()
                    .vector_inter_slice_reduce::<m![256], m![Rows]>(InterSliceReduceOpF32::Add)
                    .vector_final()
                    .commit_trim::<m![1 # 8]>()
                    .commit();
                let column = output
                    .view_mut()
                    .tile::<m![Columns], 1, Column<Rows>>(index);
                if index < self.bias.len() {
                    // SAFETY: a bias row holds its value at element zero and zeros after it.
                    let bias = unsafe { self.bias[index].view().reshape::<m![1], m![1 # 4]>() };
                    let bias: DmTensor<f32, m![1], Cluster, m![256], m![1 # 8]> =
                        bias.to_dm(&mut device.tdma);
                    let bias: VrfTensor<f32, m![1], Cluster, m![256], m![1 # 8]> = device
                        .sub
                        .begin(bias.view())
                        .fetch::<m![1], m![1 # 8]>()
                        .collect::<m![1], m![1 # 8]>()
                        .to_vrf();
                    let biased: Sums<Rows> = device
                        .main
                        .begin(sums.view())
                        .fetch::<m![Rows], m![1 # 8]>()
                        .collect::<m![Rows], m![1 # 8]>()
                        .vector_init()
                        .vector_intra_slice_tag(TagMode::Zero)
                        .vector_narrow_trim::<m![1 # 4]>()
                        .vector_fp_binary(FpBinaryOp::AddF, &bias)
                        .vector_widen_pad::<m![1 # 8]>()
                        .vector_final()
                        .commit_trim::<m![1 # 8]>()
                        .commit();
                    store(device, biased, column);
                } else {
                    store(device, sums, column);
                }
            }
        }
    }
}

/// Borrowed input rows and writable output rows for a runtime-sized affine projection.
#[derive(DeviceSend)]
pub struct Context<'a, D: Scalar, In: AxisName, Rows: AxisName> {
    /// The live input rows of the compiled row envelope.
    pub input: gather::Rows<'a, D, In, Rows>,
    /// One writable FP32 row of output columns per input row.
    pub output: HbmTensor<f32, m![1], m![Rows, Columns]>,
}

impl<X: MaterializableScalar + FetchCast<f32>, In: AxisName, Rows: AxisName>
    Op<&mut Context<'_, X, In, Rows>> for Affine<Weights<In>, Bias>
{
    type Output = ();

    fn forward(&self, device: &mut Device, context: &mut Context<'_, X, In, Rows>) {
        let (input, _) = context.input.stage(device);
        self.forward(device, (&input, &mut context.output));
    }
}

impl<D: Scalar, W: Scalar, In: AxisName, Out: AxisName, Rows: AxisName, const BIAS: usize>
    Op<HbmTensor<D, m![1], m![Rows, In]>>
    for Affine<HbmTensor<W, m![1], m![Out, In]>, [HbmTensor<W, m![1], m![Out]>; BIAS]>
{
    type Output = HbmTensor<D, m![1], m![Rows, Out]>;

    fn forward(
        &self,
        device: &mut Device,
        input: HbmTensor<D, m![1], m![Rows, In]>,
    ) -> Self::Output {
        self.forward(device, &input)
    }
}

/// A projection of a borrowed tree's matrix streams one `forward` a tile.
impl<'m, 's, 'w, U: Stream, E, S, D: Scalar, C: M, F: M, B> IntoStream<U, E, &'s mut S>
    for &'m Affine<&'w HbmTensor<D, C, F>, B>
where
    RowWise<'s, &'m Affine<&'w HbmTensor<D, C, F>, B>, E, S, U>: Stream,
{
    type Stream = RowWise<'s, &'m Affine<&'w HbmTensor<D, C, F>, B>, E, S, U>;

    fn into_stream(self, env: E, state: &'s mut S, upstream: U) -> Self::Stream {
        RowWise::new(self, env, state, upstream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::index::gather::{RowStorage, Rows};
    use crate::tensor::axis::SixtyFour;
    use crate::tensor::memory::{alias, blocks};
    use crate::tests::DEVICE;
    use crate::tests::forward;
    use furiosa_opt_std::runtime::{HbmBuffer, HostBufferSlice, launch};
    use std::any;

    axes![Width = 1024];

    /// A scalar of the head's features, which holds each fixture value exactly.
    trait Exact: MaterializableScalar + FetchCast<f32> {
        fn exact(value: f32) -> Self;
    }

    impl Exact for bf16 {
        fn exact(value: f32) -> Self {
            let converted = bf16::from_f32(value);
            assert_eq!(
                converted.to_f32(),
                value,
                "a BF16 fixture value must not round"
            );
            converted
        }
    }

    impl Exact for f32 {
        fn exact(value: f32) -> Self {
            value
        }
    }

    /// Products and sums are multiples of 2^-14 below 2^9, exact in FP32 in any order;
    /// fine values retain bits that BF16 drops.
    fn values(count: usize, mut state: u64, fine: bool) -> Vec<f32> {
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let sixteenths = (state >> 60) as f32 / 16.0 - 0.5;
                sixteenths
                    + if fine {
                        ((state >> 56) & 0xf) as f32 / 1024.0
                    } else {
                        0.0
                    }
            })
            .collect()
    }

    /// Consecutive `E`-shaped `D` tensors of `values`, from one upload.
    async fn upload<E: M, D: Exact>(
        device: &mut Device,
        values: &[f32],
    ) -> Slice<HbmTensor<D, m![1], E>> {
        let converted: Vec<D> = values.iter().map(|&value| D::exact(value)).collect();
        let buffer = HostBufferSlice::from(&D::to_buf(&converted))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        blocks(buffer, E::SIZE * size_of::<D>()).unwrap().into()
    }

    async fn check<X: Exact>(
        device: &mut Device,
        features: &[f32],
        (weights, bias): (&[f32], &[f32]),
    ) {
        let (live, labels) = (features.len() / Width::SIZE, weights.len() / Width::SIZE);
        let source: RowStorage<Width, X> = upload(device, features).await;
        let padded: Vec<f32> = bias
            .iter()
            .flat_map(|&offset| [offset, 0.0, 0.0, 0.0])
            .collect();
        let head = Affine::<Weights<Width>, Bias>::new(
            upload(device, weights).await,
            if bias.is_empty() {
                Slice::from(Vec::new())
            } else {
                upload(device, &padded).await
            },
        );
        let output =
            HbmBuffer::alloc(device, SixtyFour::SIZE * COLUMNS * size_of::<f32>()).unwrap();
        let mut context = Context::<X, Width, SixtyFour> {
            input: Rows::new(device, &source).await.unwrap(),
            output: alias(&output, ..).unwrap(),
        };
        launch(forward, (&mut *device, &head, (), &mut (), &mut context))
            .await
            .unwrap();
        let scores: HbmTensor<f32, m![1], m![{ Symbol<SixtyFour> }, Columns]> =
            output.try_into().unwrap();
        let scores = scores
            .to_host::<m![{ Symbol<SixtyFour> }, Columns]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let form = any::type_name::<X>();
        for row in 0..live {
            for label in 0..labels {
                let expected = features[row * Width::SIZE..][..Width::SIZE]
                    .iter()
                    .zip(&weights[label * Width::SIZE..][..Width::SIZE])
                    .map(|(feature, weight)| feature * weight)
                    .sum::<f32>()
                    + bias.get(label).copied().unwrap_or(0.0);
                assert_eq!(
                    scores[row * COLUMNS + label],
                    expected,
                    "{form} features, {labels} labels, row {row} of {live}, label {label}"
                );
            }
        }
    }

    /// Fixtures for `X` features: FP32 values carry bits BF16 drops in every bias and in exactly
    /// one factor of each product, the features when they are FP32, else the coefficients.
    fn fixtures<X: Exact>(rows: usize, labels: usize) -> [Vec<f32>; 3] {
        let fine = X::BITS == 32;
        [
            values(rows * Width::SIZE, 0x2545_f491_4f6c_dd1d, fine),
            values(
                labels * Width::SIZE,
                0x9e37_79b9_7f4a_7c15 ^ labels as u64,
                !fine,
            ),
            values(labels, 0x6a09_e667_f3bc_c908, true),
        ]
    }

    /// Every label count from one to the compiled columns, with and without a bias, for one live
    /// row up to a whole envelope.
    async fn forms<X: Exact>(device: &mut Device) {
        for labels in [1, 2, 3, 255, 256] {
            let [features, weights, bias] = fixtures::<X>(SixtyFour::SIZE, labels);
            for live in [1, 5, 64] {
                for bias in [&bias[..], &[]] {
                    let rows = &features[..live * Width::SIZE];
                    check::<X>(device, rows, (&weights, bias)).await;
                }
            }
        }
    }

    /// Live scores equal each label's FP32 dot product plus bias exactly.
    #[tokio::test(flavor = "current_thread")]
    async fn scores_live_rows() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        forms::<bf16>(&mut device).await;
        forms::<f32>(&mut device).await;
    }

    /// Heads that share the compiled entry each score with their own parameters: a head that
    /// differs from another in one weight, or in one bias, differs in that label's column only.
    #[tokio::test(flavor = "current_thread")]
    async fn keeps_head_parameters() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let [features, weights, bias] = fixtures::<f32>(5, 3);
        let mut weight = weights.clone();
        weight[Width::SIZE + 7] += 0.25;
        let mut offset = bias.clone();
        offset[2] -= 0.5;
        for (w, b) in [
            (&weights, &bias),
            (&weight, &bias),
            (&weights, &offset),
            (&weights, &bias),
        ] {
            check::<f32>(&mut device, &features, (w, b)).await;
        }
    }
}
