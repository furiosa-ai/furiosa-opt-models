//! Elementwise FP32 arithmetic with source-dtype output over bounded device-memory windows.

use crate::Module;
use crate::ops::layout::Dm;
use crate::ops::normalization::{Chunk, Cluster, Element, FeatureSlices, Spread};
use crate::ops::vector::scalar;
use furiosa_opt_std::prelude::*;

/// Flattened logical element count, independent of the input's named axes.
#[derive(Clone, Debug)]
pub(crate) struct Flat<S>(std::marker::PhantomData<S>);

impl<S: M + 'static> AxisName for Flat<S> {
    const NAME: Ident = Ident::new("Flat");
    const SIZE: usize = S::SIZE;
}

/// Elements per slice, an eight-element multiple bounded by 512.
#[derive(Clone, Debug)]
pub struct Part<S>(std::marker::PhantomData<S>);

impl<S: M + 'static> AxisName for Part<S> {
    const NAME: Ident = Ident::new("Part");

    const SIZE: usize = {
        assert!(
            S::SIZE > 0 && S::SIZE.is_multiple_of(8),
            "pointwise tensors require complete eight-element vectors"
        );
        let minimum = S::SIZE.div_ceil(256).next_multiple_of(8);
        let mut part = if minimum < 512 { minimum } else { 512 };
        while S::SIZE % part != 0 {
            part -= 8;
        }
        part
    };
}

/// One transfer window covering at most 256 equal slices and dividing the flattened input.
#[derive(Clone, Debug)]
pub struct Window<S>(std::marker::PhantomData<S>);

impl<S: M + 'static> AxisName for Window<S> {
    const NAME: Ident = Ident::new("Window");

    const SIZE: usize = {
        let parts = S::SIZE / Part::<S>::SIZE;
        let mut lanes = if parts < 256 { parts } else { 256 };
        while parts % lanes != 0 {
            lanes -= 1;
        }
        lanes * Part::<S>::SIZE
    };
}

type Width<S> = Symbol<Flat<S>>;
type Extent<S> = Symbol<Window<S>>;
type Segment<S> = Symbol<Part<S>>;
type Batch<S> = m![{ Width<S> } / ({ Extent<S> }) = 1 # ({ Width<S> } / ({ Extent<S> })), { Width<S> } % ({ Extent<S> })];
type BatchMut<S> = m![{ Width<S> } / ({ Extent<S> }) = 1 #{!} ({ Width<S> } / ({ Extent<S> })), { Width<S> } % ({ Extent<S> })];
type Slices<S> = m![{ Width<S> } % ({ Extent<S> }) / ({ Segment<S> }) # 256];
type Elements<S> = m![{ Width<S> } % ({ Segment<S> })];
type Rows<D, S, Chips = m![1]> = DmTensor<D, Chips, m![1 # 2], Slices<S>, Elements<S>>;
type Registers<S, Chips = m![1]> = VrfTensor<f32, Chips, m![1 # 2], Slices<S>, Elements<S>>;

#[derive(Clone, Debug)]
struct Vector<S>(std::marker::PhantomData<S>);

impl<S: M + 'static> AxisName for Vector<S> {
    const NAME: Ident = Ident::new("Flat");
    const SIZE: usize = if S::SIZE == 1 { 8 } else { S::SIZE };
}

type Shape<S> = Symbol<Vector<S>>;

/// Scalars widened to FP32 for arithmetic and converted back at the window boundary.
pub(crate) trait Float: MaterializableScalar + FetchCast<f32> {
    fn narrow<S: M + 'static, Chips: M>(
        device: &mut Device,
        input: Rows<f32, S, Chips>,
    ) -> Rows<Self, S, Chips>;
}

impl Float for f32 {
    fn narrow<S: M + 'static, Chips: M>(
        _: &mut Device,
        input: Rows<f32, S, Chips>,
    ) -> Rows<Self, S, Chips> {
        input
    }
}

impl Float for bf16 {
    fn narrow<S: M + 'static, Chips: M>(
        device: &mut Device,
        input: Rows<f32, S, Chips>,
    ) -> Rows<Self, S, Chips> {
        device
            .main
            .begin(input.view())
            .fetch::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
            .collect::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
            .cast::<bf16, m![{ Width<S> } % 8 # 16]>()
            .commit_trim::<m![{ Width<S> } % 8]>()
            .commit()
    }
}

/// Unary formulas; extra passes keep clipping and activation stages in separate DM round trips.
#[derive(Clone, Copy, PartialEq, Eq, std::marker::ConstParamTy)]
pub(crate) enum Unary {
    /// Clamp (x + 1) / 2 to [0, 1].
    Denormalize,
    /// Hyperbolic tangent.
    Tanh,
    /// x / 2 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³))).
    Gelu,
    /// x * sigmoid(x).
    Silu,
    /// 1 / (1 + exp(-x)).
    Sigmoid,
}

impl Unary {
    fn apply<D: Float, S: M + 'static, const OP: Unary, Chips: M>(
        device: &mut Device,
        input: Rows<D, S, Chips>,
    ) -> Rows<D, S, Chips> {
        let source: Registers<S, Chips> = device
            .sub
            .begin(input.view())
            .fetch::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
            .to_vrf();
        let value = device
            .main
            .begin(input.view())
            .fetch::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ Width<S> } % ({ Segment<S> }) / 4], m![{ Width<S> } % 4]>();
        let value = match OP {
            Unary::Denormalize => value
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.5_f32)
                .vector_fp_binary(FpBinaryOp::AddF, 0.5_f32),
            Unary::Tanh => value.vector_fp_unary(FpUnaryOp::Tanh),
            Unary::Silu => value
                .vector_fp_unary(FpUnaryOp::Sigmoid)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &source),
            Unary::Gelu => value
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &source)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), 0.044715_f32)
                .vector_fp_binary(FpBinaryOp::AddF, 1_f32),
            Unary::Sigmoid => value.vector_fp_unary(FpUnaryOp::Sigmoid),
        };
        let output: Rows<f32, S, Chips> = value
            .vector_widen_concat::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
            .vector_final()
            .commit_trim::<m![{ Width<S> } % 8]>()
            .commit();
        let output = if const { matches!(OP, Unary::Denormalize) } {
            device
                .main
                .begin(output.view())
                .fetch::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
                .collect::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_clip(ClipBinaryOpF32::Max, 0.0_f32)
                .vector_final()
                .commit_trim::<m![{ Width<S> } % 8]>()
                .commit()
        } else {
            output
        };
        let output = if const { matches!(OP, Unary::Denormalize) } {
            device
                .main
                .begin(output.view())
                .fetch::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
                .collect::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_clip(ClipBinaryOpF32::Min, 1.0_f32)
                .vector_final()
                .commit_trim::<m![{ Width<S> } % 8]>()
                .commit()
        } else {
            output
        };
        let output = if const { matches!(OP, Unary::Gelu) } {
            device.main.begin(output.view())
                .fetch::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
                .collect::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![{ Width<S> } % ({ Segment<S> }) / 4], m![{ Width<S> } % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &source)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), 0.79788456_f32)
                .vector_fp_unary(FpUnaryOp::Tanh)
                .vector_widen_concat::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
                .vector_final()
                .commit_trim::<m![{ Width<S> } % 8]>()
                .commit()
        } else {
            output
        };
        let output = if const { matches!(OP, Unary::Gelu) } {
            device.main.begin(output.view())
                .fetch::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
                .collect::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![{ Width<S> } % ({ Segment<S> }) / 4], m![{ Width<S> } % 4]>()
                .vector_fp_binary(FpBinaryOp::AddF, 1_f32)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &source)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), 0.5_f32)
                .vector_widen_concat::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
                .vector_final()
                .commit_trim::<m![{ Width<S> } % 8]>()
                .commit()
        } else {
            output
        };
        D::narrow(device, output)
    }
}

impl Unary {
    pub(in crate::ops) fn rows<D: Float, S: M + 'static, const OP: Unary, Chips: M>(
        device: &mut Device,
        input: &HbmTensor<D, Chips, S>,
    ) -> HbmTensor<D, Chips, S> {
        if const { S::SIZE == 1 } {
            type Vector = m![8];
            // SAFETY: flattening preserves the contiguous element order.
            let input = unsafe { input.view().reshape::<Chips, m![{ Width<S> }]>() };
            let input: DmTensor<D, Chips, m![1 # 2], m![1 # 256], m![{ Width<S> } = 1 # 8]> = input
                .pad::<m![{ Width<S> } = 1 # 8]>()
                .to_dm(&mut device.tdma);
            // SAFETY: both mappings cover the same eight contiguous cells; only lane 0 reaches HBM.
            let input: Rows<D, Vector, Chips> = unsafe { input.reshape() };
            let mut result = HbmTensor::<D, Chips, m![{ Width<S> }]>::new();
            Unary::apply::<D, Vector, OP, Chips>(device, input)
                .view()
                .tile::<m![{ Width<Vector> }], 1, m![{ Width<Vector> } = 1 # 8]>(0)
                .to_hbm_view(
                    &mut device.tdma,
                    result
                        .view_mut()
                        .tile::<m![{ Width<S> }], 1, m![{ Width<S> } = 1]>(0),
                );
            // SAFETY: flattening preserves the contiguous element order.
            return unsafe { result.reshape() };
        }

        let mut result = HbmTensor::<D, Chips, m![{ Width<S> }]>::new();
        for index in 0..S::SIZE / Window::<Shape<S>>::SIZE {
            let input: Rows<D, Shape<S>, Chips> =
                unsafe { input.view().reshape::<Chips, m![{ Width<S> }]>() }
                    .tile::<m![{ Width<Shape<S>> } / ({ Extent<Shape<S>> })], 1, Batch<Shape<S>>>(
                        index,
                    )
                    .to_dm(&mut device.tdma);
            Unary::apply::<D, Shape<S>, OP, Chips>(device, input)
            .view()
            .to_hbm_view(
            &mut device.tdma,
            result
                .view_mut()
                .tile::<m![{ Width<Shape<S>> } / ({ Extent<Shape<S>> })], 1, BatchMut<Shape<S>>>(
                    index,
                ),
        );
        }
        // SAFETY: flattening preserves the contiguous element order.
        unsafe { result.reshape() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Module, ops::activation::Tanh};
    use furiosa_opt_std::runtime::launch;

    #[device(chip = 1)]
    fn tanh<S: M + 'static>(
        device: &mut Device,
        input: &HbmTensor<f32, m![1], S>,
    ) -> HbmTensor<f32, m![1], S> {
        Tanh.forward((device, input))
    }

    #[tokio::test(flavor = "current_thread")]
    async fn scalar_tanh() {
        let mut device = Device::new(tanh.topology()).unwrap();
        for value in [-3.0_f32, -0.25, 0.0, 0.25, 3.0] {
            let input: HbmTensor<f32, m![1], m![1]> =
                HostTensor::<f32, m![1]>::from_vec(vec![value])
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap();
            let output = launch(tanh, (&mut device, &input)).await.unwrap();
            let output = output
                .to_host::<m![1]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            assert_eq!(output.len(), 1);
            assert!(
                (output[0] - value.tanh()).abs() <= 1e-5,
                "tanh({value}) = {}",
                output[0]
            );
        }
    }

    type Envelope = crate::ops::layout::axes::Rows<8>;
    type Row = m![Features];
    axes![Features = 4096];

    #[device(chip = 1)]
    fn gated_rows(
        device: &mut Device,
        (gate, up): (
            &HbmTensor<bf16, m![1], m![Envelope, Features]>,
            &HbmTensor<bf16, m![1], m![Envelope, Features]>,
        ),
        live: &crate::compose::Count,
    ) -> HbmTensor<bf16, m![1], m![Envelope, Features]> {
        use crate::ops::layout::{Live, axes::Lanes};
        use crate::ops::{activation::Silu, indexing::Store, multiply::Multiply};
        let tiles = Live::<Envelope>::read(live);
        let gate: Dm<Lanes<Envelope>, Spread<Features, Envelope>> = Dm {
            value: gate.to_dm(&mut device.tdma),
            tiles,
        };
        let up: Dm<Lanes<Envelope>, Spread<Features, Envelope>> = Dm {
            value: up.to_dm(&mut device.tdma),
            tiles,
        };
        let active = Silu.forward((&mut *device, &gate));
        let product = Multiply.forward((&mut *device, (&active, &up)));
        Store.forward((device, &product))
    }

    #[device(chip = 1)]
    fn gated_row(
        device: &mut Device,
        (gate, up): (&HbmTensor<bf16, m![1], Row>, &HbmTensor<bf16, m![1], Row>),
    ) -> HbmTensor<bf16, m![1], Row> {
        use crate::ops::{activation::Silu, multiply::Multiply};
        let active = Silu.forward((&mut *device, gate));
        Multiply.forward((device, (&active, up)))
    }

    /// A call's live rows in feature slices pass the activation and product exactly as each row
    /// alone passes them.
    #[tokio::test(flavor = "current_thread")]
    async fn gates_rows_as_alone() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(gated_rows.topology()).unwrap();
        let values = |seed| crate::ops::normalization::tests::random(8 * 4096, seed);
        let (gates, ups) = (values(3), values(5));
        async fn upload<E: M>(device: &mut Device, values: &[bf16]) -> HbmTensor<bf16, m![1], E> {
            HostTensor::<bf16, E>::from_vec(values.iter().map(|v| bf16::from_f32(v.to_f32() * 4.0)))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap()
        }
        let gate = upload::<m![Envelope, Features]>(&mut device, &gates).await;
        let up = upload::<m![Envelope, Features]>(&mut device, &ups).await;
        let live = crate::compose::tests::count(5)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let rows = launch(gated_rows, (&mut device, (&gate, &up), &live))
            .await
            .unwrap()
            .to_host::<m![Envelope, Features]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let bits = |values: &[bf16]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        for row in 0..5 {
            let gate = upload::<Row>(&mut device, &gates[row * 4096..][..4096]).await;
            let up = upload::<Row>(&mut device, &ups[row * 4096..][..4096]).await;
            let alone = launch(gated_row, (&mut device, (&gate, &up)))
                .await
                .unwrap()
                .to_host::<Row>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            assert_eq!(bits(&rows[row * 4096..][..4096]), bits(&alone), "row {row}");
        }
    }
}

impl<S: M + 'static> Window<S> {
    pub(in crate::ops) fn combine<D: Float, const MULTIPLY: bool, Chips: M>(
        device: &mut Device,
        lhs: &HbmTensor<D, Chips, S>,
        rhs: &HbmTensor<D, Chips, S>,
    ) -> HbmTensor<D, Chips, S> {
        let mut result = HbmTensor::<D, Chips, m![{ Width<S> }]>::new();
        for index in 0..S::SIZE / Window::<S>::SIZE {
            let lhs: Rows<D, S, Chips> = unsafe { lhs.view().reshape::<Chips, m![{ Width<S> }]>() }
                .tile::<m![{ Width<S> } / ({ Extent<S> })], 1, Batch<S>>(index)
                .to_dm(&mut device.tdma);
            let rhs: Rows<D, S, Chips> = unsafe { rhs.view().reshape::<Chips, m![{ Width<S> }]>() }
                .tile::<m![{ Width<S> } / ({ Extent<S> })], 1, Batch<S>>(index)
                .to_dm(&mut device.tdma);
            let rhs: Registers<S, Chips> = device
                .sub
                .begin(rhs.view())
                .fetch::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
                .to_vrf();
            let output: Rows<f32, S, Chips> = device
            .main
            .begin(lhs.view())
            .fetch::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ Width<S> } % ({ Segment<S> }) / 4], m![{ Width<S> } % 4]>()
            .vector_fp_binary(
                if MULTIPLY {
                    FpBinaryOp::MulF(FpMulAlu::Mul0)
                } else {
                    FpBinaryOp::AddF
                },
                &rhs,
            )
            .vector_widen_concat::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
            .vector_final()
            .commit_trim::<m![{ Width<S> } % 8]>()
            .commit();
            D::narrow(device, output).view().to_hbm_view(
                &mut device.tdma,
                result
                    .view_mut()
                    .tile::<m![{ Width<S> } / ({ Extent<S> })], 1, BatchMut<S>>(index),
            );
        }
        // SAFETY: flattening preserves the contiguous element order.
        unsafe { result.reshape() }
    }
}

impl<D: Float, S: M + 'static> Module<(&mut Device, &HbmTensor<D, m![1], S>)>
    for super::multiply::Scale<HbmTensor<f32, m![1], m![1 # 8]>>
{
    type Output = HbmTensor<D, m![1], S>;

    fn forward(&self, (device, input): (&mut Device, &HbmTensor<D, m![1], S>)) -> Self::Output {
        Self::rows(device, input, &self.0)
    }
}

/// `input` times the FP32 scalar in the first cell of `factor`, rounded to `D` once.
impl<F: M> super::multiply::Scale<HbmTensor<f32, m![1], F>> {
    pub(in crate::ops) fn rows<D: Float, S: M + 'static>(
        device: &mut Device,
        input: &HbmTensor<D, m![1], S>,
        factor: &HbmTensor<f32, m![1], F>,
    ) -> HbmTensor<D, m![1], S> {
        let mut result = HbmTensor::<D, m![1], m![{ Width<S> }]>::new();
        for index in 0..S::SIZE / Window::<S>::SIZE {
            let input: Rows<D, S> = unsafe { input.view().reshape::<m![1], m![{ Width<S> }]>() }
                .tile::<m![{ Width<S> } / ({ Extent<S> })], 1, Batch<S>>(index)
                .to_dm(&mut device.tdma);
            let scale: DmTensor<f32, m![1], m![1 # 2], Slices<S>, m![1 # 8]> =
                factor.to_dm(&mut device.tdma);
            let scale = scalar::register(device, &scale);
            let output: Rows<f32, S> = device
            .main
            .begin(input.view())
            .fetch::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ Width<S> } % ({ Segment<S> }) / 4], m![{ Width<S> } % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
            .vector_widen_concat::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
            .vector_final()
            .commit_trim::<m![{ Width<S> } % 8]>()
            .commit();
            D::narrow(device, output).view().to_hbm_view(
                &mut device.tdma,
                result
                    .view_mut()
                    .tile::<m![{ Width<S> } / ({ Extent<S> })], 1, BatchMut<S>>(index),
            );
        }
        // SAFETY: flattening preserves the contiguous element order.
        unsafe { result.reshape() }
    }
}

/// One feature-slice tile of every row: its time and packet axes.
type SpreadTime<W, R> = m![R, W % ({ Symbol<Chunk<W>> }) / 8];
type SpreadPacket<W> = m![W % 8];

/// `x * sigmoid(x)` of every row of one launch in feature slices, in FP32, rounded to BF16 once;
/// the result keeps their live tiles.
impl<'x, W: AxisName, R: AxisName, S: crate::compose::Space<Envelope = R>>
    Module<(&mut Device, &'x Dm<S, Spread<W, R>>)> for super::activation::Silu
{
    type Output = Dm<S, Spread<W, R>>;

    fn forward(&self, (device, input): (&mut Device, &'x Dm<S, Spread<W, R>>)) -> Self::Output {
        let source: VrfTensor<f32, m![1], Cluster, FeatureSlices<W>, Element<W, R>> = device
            .sub
            .begin(input.value.view())
            .fetch::<SpreadTime<W, R>, SpreadPacket<W>>()
            .fetch_cast::<f32>()
            .collect::<SpreadTime<W, R>, SpreadPacket<W>>()
            .to_vrf();
        let value = device
            .main
            .begin(input.value.view())
            .fetch::<SpreadTime<W, R>, SpreadPacket<W>>()
            .fetch_cast::<f32>()
            .collect::<SpreadTime<W, R>, SpreadPacket<W>>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![R, W % ({ Symbol<Chunk<W>> }) / 4], m![W % 4]>()
            .vector_fp_unary(FpUnaryOp::Sigmoid)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &source)
            .vector_widen_concat::<SpreadTime<W, R>, SpreadPacket<W>>()
            .vector_final()
            .cast::<bf16, m![W % 8 # 16]>()
            .commit_trim::<m![W % 8]>()
            .commit();
        Dm {
            value,
            tiles: input.tiles,
        }
    }
}

/// The elementwise product of the rows of one launch in feature slices, in FP32, rounded to BF16
/// once; the product keeps their live tiles.
impl<'x, 'y, W: AxisName, R: AxisName, S: crate::compose::Space<Envelope = R>>
    Module<(
        &mut Device,
        (&'x Dm<S, Spread<W, R>>, &'y Dm<S, Spread<W, R>>),
    )> for super::multiply::Multiply
{
    type Output = Dm<S, Spread<W, R>>;

    fn forward(
        &self,
        (device, (left, right)): (
            &mut Device,
            (&'x Dm<S, Spread<W, R>>, &'y Dm<S, Spread<W, R>>),
        ),
    ) -> Self::Output {
        let right: VrfTensor<f32, m![1], Cluster, FeatureSlices<W>, Element<W, R>> = device
            .sub
            .begin(right.value.view())
            .fetch::<SpreadTime<W, R>, SpreadPacket<W>>()
            .fetch_cast::<f32>()
            .collect::<SpreadTime<W, R>, SpreadPacket<W>>()
            .to_vrf();
        let value = device
            .main
            .begin(left.value.view())
            .fetch::<SpreadTime<W, R>, SpreadPacket<W>>()
            .fetch_cast::<f32>()
            .collect::<SpreadTime<W, R>, SpreadPacket<W>>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![R, W % ({ Symbol<Chunk<W>> }) / 4], m![W % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &right)
            .vector_widen_concat::<SpreadTime<W, R>, SpreadPacket<W>>()
            .vector_final()
            .cast::<bf16, m![W % 8 # 16]>()
            .commit_trim::<m![W % 8]>()
            .commit();
        Dm {
            value,
            tiles: left.tiles,
        }
    }
}
