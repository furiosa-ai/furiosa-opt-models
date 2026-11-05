//! Rows times transposed projection weights, with an optional per-output bias.

use crate::Module;
use crate::ops::normalization::{Cluster, FeatureSlices};
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

pub mod mlp;
pub mod output;
pub mod pair;
pub mod query;

/// Projection coefficients in the device form a projection computes with.
#[derive(Clone, DeviceSend)]
pub struct Affine<Matrix, Offset> {
    /// Projection weights in [output, input] order, or a quantized weight/scale tuple.
    pub weight: Matrix,
    /// Per-output offsets; the chosen weight form determines the no-bias representation.
    pub bias: Offset,
}

/// E2M1 weights with E4M3 scales per 16 inputs and an FP32 tensor scale.
pub type Nvfp4<In, Out> = Affine<
    (
        HbmTensor<f4e2m1, m![1], m![Out, In]>,
        HbmTensor<f8e4m3, m![1], m![Out, In / 16]>,
        HbmTensor<f32, m![1], m![1 # 8]>,
    ),
    (),
>;

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
pub type Coefficients<D, In, Out, const BIAS: usize> = Affine<
    HbmTensor<D, m![1], m![{ Symbol<Outputs<Out>> }, { Symbol<Inputs<In>> }]>,
    [HbmTensor<D, m![1], m![{ Symbol<Outputs<Out>> }]>; BIAS],
>;

impl<D: Scalar, W: Scalar, In: AxisName, Out: AxisName, Rows: AxisName, const BIAS: usize>
    Module<(&mut Device, &HbmTensor<D, m![1], m![Rows, In]>)> for Coefficients<W, In, Out, BIAS>
{
    type Output = HbmTensor<D, m![1], m![Rows, Out]>;

    fn forward(&self, _: (&mut Device, &HbmTensor<D, m![1], m![Rows, In]>)) -> Self::Output {
        const { assert!(BIAS <= 1) };
        todo!("NPU affine projection over contiguous rows")
    }
}

impl<D: Scalar, W: Scalar, In: AxisName, Out: AxisName, Rows: AxisName, const BIAS: usize>
    Module<(&mut Device, HbmTensor<D, m![1], m![Rows, In]>)> for Coefficients<W, In, Out, BIAS>
{
    type Output = HbmTensor<D, m![1], m![Rows, Out]>;

    fn forward(
        &self,
        (device, input): (&mut Device, HbmTensor<D, m![1], m![Rows, In]>),
    ) -> Self::Output {
        self.forward((device, &input))
    }
}

impl<D: Scalar, W: Scalar, In: AxisName, Out: AxisName, Rows: AxisName, const BIAS: usize>
    Module<(&mut Device, &HbmTensor<D, m![1], m![Rows, In]>)>
    for Affine<HbmTensor<W, m![1], m![Out, In]>, [HbmTensor<W, m![1], m![Out]>; BIAS]>
{
    type Output = HbmTensor<D, m![1], m![Rows, Out]>;

    fn forward(&self, _: (&mut Device, &HbmTensor<D, m![1], m![Rows, In]>)) -> Self::Output {
        const { assert!(BIAS <= 1) };
        todo!("NPU affine projection over contiguous rows")
    }
}

/// One contiguous input-width weight row for each runtime output column.
pub type Weights<D, In> = Slice<HbmTensor<D, m![1], m![In]>>;

/// Per-output bias at element zero of each four-element padded row.
pub type Bias<D> = Slice<HbmTensor<D, m![1], m![4]>>;

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
/// is not empty, accumulated in FP32; columns past the weight rows are left as they were.
impl<X, D, In: AxisName, Rows: AxisName>
    Module<(
        &mut Device,
        (
            &HbmTensor<X, m![1], m![Rows, In]>,
            &mut HbmTensor<f32, m![1], m![Rows, Columns]>,
        ),
    )> for Affine<Weights<D, In>, Bias<D>>
where
    X: MaterializableScalar + FetchCast<f32>,
    D: MaterializableScalar + FetchCast<f32>,
{
    type Output = ();

    fn forward(
        &self,
        (device, (input, output)): (
            &mut Device,
            (
                &HbmTensor<X, m![1], m![Rows, In]>,
                &mut HbmTensor<f32, m![1], m![Rows, Columns]>,
            ),
        ),
    ) {
        type Chunk<In> = Symbol<crate::ops::normalization::Chunk<In>>;
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
                    D,
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
                    .fetch_cast::<f32>()
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
                    .vector_narrow_split::<m![Rows, In % ({ Chunk<In> }) / 8, In % 8 / 4 % 2], m![In % 4]>()
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight)
                    .vector_intra_slice_reduce::<In, m![Rows], m![1 # 4]>(IntraSliceReduceOpF32::Add)
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
                    let bias: DmTensor<D, m![1], Cluster, m![256], m![1 # 8]> =
                        bias.to_dm(&mut device.tdma);
                    let bias: VrfTensor<f32, m![1], Cluster, m![256], m![1 # 8]> = device
                        .sub
                        .begin(bias.view())
                        .fetch::<m![1], m![1 # 8]>()
                        .fetch_cast::<f32>()
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
    pub input: crate::ops::indexing::rows::Rows<'a, D, In, Rows>,
    /// One writable FP32 row of output columns per input row.
    pub output: HbmTensor<f32, m![1], m![Rows, Columns]>,
}

impl<X, D, In: AxisName, Rows: AxisName> Module<(&mut Device, &mut Context<'_, X, In, Rows>)>
    for Affine<Weights<D, In>, Bias<D>>
where
    X: MaterializableScalar + FetchCast<f32>,
    D: MaterializableScalar + FetchCast<f32>,
{
    type Output = ();

    fn forward(&self, (device, context): (&mut Device, &mut Context<'_, X, In, Rows>)) {
        let (input, _) = context.input.stage(device);
        self.forward((device, (&input, &mut context.output)));
    }
}

impl<D: Scalar, W: Scalar, In: AxisName, Out: AxisName, Rows: AxisName, const BIAS: usize>
    Module<(&mut Device, HbmTensor<D, m![1], m![Rows, In]>)>
    for Affine<HbmTensor<W, m![1], m![Out, In]>, [HbmTensor<W, m![1], m![Out]>; BIAS]>
{
    type Output = HbmTensor<D, m![1], m![Rows, Out]>;

    fn forward(
        &self,
        (device, input): (&mut Device, HbmTensor<D, m![1], m![Rows, In]>),
    ) -> Self::Output {
        self.forward((device, &input))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::indexing::rows::Rows;
    use crate::ops::layout::axes::SixtyFour;
    use crate::ops::normalization::tests::random;
    use crate::ops::tensor::alias;
    use furiosa_opt_std::runtime::{HbmBuffer, HostBufferSlice, launch};

    axes![Width = 1024];

    const LABELS: usize = 3;

    /// Consecutive `E`-shaped BF16 tensors of `values`, from one upload.
    async fn upload<E: M>(
        device: &mut Device,
        values: &[bf16],
    ) -> Slice<HbmTensor<bf16, m![1], E>> {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect();
        let buffer = HostBufferSlice::from(&bytes)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let size = E::SIZE * size_of::<bf16>();
        (0..values.len() / E::SIZE)
            .map(|index| alias(&buffer, index * size..(index + 1) * size).unwrap())
            .collect()
    }

    /// Each live row's column for each label is its FP32 dot product with the label's weight row
    /// plus the label's bias, with and without a bias, and for one live row or several.
    #[tokio::test(flavor = "current_thread")]
    async fn scores_each_live_row_per_label() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let features = random(5 * Width::SIZE, 0x2545_f491_4f6c_dd1d);
        let coefficients = random(LABELS * Width::SIZE, 0x9e37_79b9_7f4a_7c15);
        let offsets = [0.5f32, -1.25, 2.0];
        for (live, biased) in [(5, true), (1, false)] {
            let source: crate::ops::indexing::RowStorage<Width, bf16> =
                upload(&mut device, &features[..live * Width::SIZE]).await;
            let weight: Weights<bf16, Width> = upload(&mut device, &coefficients).await;
            let padded: Vec<bf16> = offsets
                .iter()
                .flat_map(|&offset| {
                    [
                        bf16::from_f32(offset),
                        bf16::from_f32(0.0),
                        bf16::from_f32(0.0),
                        bf16::from_f32(0.0),
                    ]
                })
                .collect();
            let bias: Bias<bf16> = if biased {
                upload(&mut device, &padded).await
            } else {
                Slice::from(Vec::new())
            };
            let affine = Affine::new(weight, bias);
            let output =
                HbmBuffer::alloc(&device, SixtyFour::SIZE * COLUMNS * size_of::<f32>()).unwrap();
            let mut context = Context::<bf16, Width, SixtyFour> {
                input: Rows::new(&mut device, &source).await.unwrap(),
                output: alias(&output, ..).unwrap(),
            };
            launch(crate::models::forward, (&mut device, &affine, &mut context))
                .await
                .unwrap();
            let scores: HbmTensor<f32, m![1], m![{ Symbol<SixtyFour> }, Columns]> =
                output.try_into().unwrap();
            let scores = scores
                .to_host::<m![{ Symbol<SixtyFour> }, Columns]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            for label in 0..LABELS {
                for row in 0..live {
                    let expected: f32 = (0..Width::SIZE)
                        .map(|feature| {
                            features[row * Width::SIZE + feature].to_f32()
                                * coefficients[label * Width::SIZE + feature].to_f32()
                        })
                        .sum::<f32>()
                        + if biased { offsets[label] } else { 0.0 };
                    let got = scores[row * COLUMNS + label];
                    assert!(
                        (got - expected).abs() <= 1e-3 * (1.0 + expected.abs()),
                        "label {label} row {row} of {live}: {got} != {expected}"
                    );
                }
            }
        }
    }
}
