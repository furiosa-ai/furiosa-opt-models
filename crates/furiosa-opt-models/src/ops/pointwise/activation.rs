//! Activation functions: each one function of one element.

use super::{
    Factors, Float, Registers, Rows, Segment, SpreadPacket, SpreadPairs, SpreadTime, Width, Window,
};
use crate::Op;
use crate::load::{Checkpoint, Context, Load, LoadError};
use crate::ops::index::packed::Timestep;
use crate::ops::linear::{Groups, stream};
use crate::ops::normalization::{Chunk, Cluster, Element, FeatureSlices, Spread};
use crate::ops::vector::Silu as _;
use crate::tensor::{live::Live, mapping::Rows as Layout, parallel::Chip, scalar};
use furiosa_opt_std::prelude::*;
use std::borrow::Borrow;

/// Elementwise activation selected by a model's configuration.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// max(x, 0).
    Relu,
    /// x * Phi(x), using the normal cumulative distribution.
    Gelu,
    /// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    NewGelu,
    /// x * sigmoid(x).
    Silu,
    /// x * sigmoid(1.702 * x).
    QuickGelu,
}

/// max(x, 0) on a single-chip tensor.
#[derive(Default, Clone, Copy, DeviceSend, Load)]
pub struct Relu;

impl<D: Scalar, S: M, Chips: M + 'static> Op<&HbmTensor<D, Chips, S>> for Relu {
    type Output = HbmTensor<D, Chips, S>;

    fn forward(&self, _: &mut Device, _: &HbmTensor<D, Chips, S>) -> Self::Output {
        todo!("NPU rectified linear unit")
    }
}

/// x * sigmoid(1.702 * x) on a single-chip tensor.
#[derive(Default, Clone, Copy, DeviceSend, Load)]
pub struct QuickGelu;

impl<D: Float, S: M + 'static, Chips: M + 'static> Op<&HbmTensor<D, Chips, S>> for QuickGelu {
    type Output = HbmTensor<D, Chips, S>;

    /// Each window's `x * sigmoid(1.702 x)` in FP32, `x` held in the stash.
    fn forward(&self, device: &mut Device, input: &HbmTensor<D, Chips, S>) -> Self::Output {
        Window::<S>::map(device, input, |device, input| {
            Window::<S>::stream(&mut device.main, &input)
                .vector_stash()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.702_f32)
                .vector_fp_unary(FpUnaryOp::Sigmoid)
                .vector_fp_binary_with_mode(
                    FpBinaryOp::MulF(FpMulAlu::Mul1),
                    BinaryArgMode::Mode10,
                    Stash,
                )
                .vector_widen_concat::<
                    m![{ Width<S> } % ({ Segment<S> }) / 8],
                    m![{ Width<S> } % 8],
                >()
                .vector_final()
                .commit_trim::<m![{ Width<S> } % 8]>()
                .commit()
        })
    }
}

/// Exact x * Phi(x) on a single-chip tensor.
#[derive(Default, Clone, Copy, DeviceSend, Load)]
pub struct Gelu;

impl<D: Float, S: M + 'static, Chips: M + 'static> Op<&HbmTensor<D, Chips, S>> for Gelu {
    type Output = HbmTensor<D, Chips, S>;

    /// Each window's `x / 2 * (1 + erf(x / sqrt(2)))` in FP32, in two DM round trips: the error
    /// function, then its product with `x`.
    fn forward(&self, device: &mut Device, input: &HbmTensor<D, Chips, S>) -> Self::Output {
        Window::<S>::map(device, input, |device, input| {
            let source: Registers<S, Chips> = device
                .sub
                .begin(input.view())
                .fetch::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
                .to_vrf();
            let erf: Rows<f32, S, Chips> = Window::<S>::stream(&mut device.main, &input)
                .vector_fp_binary(
                    FpBinaryOp::MulF(FpMulAlu::Mul0),
                    core::f32::consts::FRAC_1_SQRT_2,
                )
                .vector_fp_unary(FpUnaryOp::Erf)
                .vector_widen_concat::<
                    m![{ Width<S> } % ({ Segment<S> }) / 8],
                    m![{ Width<S> } % 8],
                >()
                .vector_final()
                .commit_trim::<m![{ Width<S> } % 8]>()
                .commit();
            device
                .main
                .begin(erf.view())
                .fetch::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
                .collect::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<
                    m![{ Width<S> } % ({ Segment<S> }) / 4],
                    m![{ Width<S> } % 4],
                >()
                .vector_fp_binary(FpBinaryOp::AddF, 1_f32)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &source)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), 0.5_f32)
                .vector_widen_concat::<
                    m![{ Width<S> } % ({ Segment<S> }) / 8],
                    m![{ Width<S> } % 8],
                >()
                .vector_final()
                .commit_trim::<m![{ Width<S> } % 8]>()
                .commit()
        })
    }
}

/// Tanh GELU approximation on a single-chip tensor, with coefficient 0.044715.
#[derive(Default, Clone, Copy, DeviceSend, Load)]
pub struct NewGelu;

/// The tanh GELU's FP32 coefficients: of the cube, and `sqrt(2 / pi)`.
const CUBIC: f32 = 0.044715;
const ROOT: f32 = 0.797_884_6;

impl<D: Float, S: M + 'static, Chips: M + 'static> Op<&HbmTensor<D, Chips, S>> for NewGelu {
    type Output = HbmTensor<D, Chips, S>;

    /// Each window's `x / 2 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x³)))` in FP32 over three DM
    /// round trips: `1 + 0.044715 x²`, the tanh of it times `sqrt(2/pi) x`, the product.
    fn forward(&self, device: &mut Device, input: &HbmTensor<D, Chips, S>) -> Self::Output {
        Window::<S>::map(device, input, |device, input| {
            let source: Registers<S, Chips> = device
                .sub
                .begin(input.view())
                .fetch::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
                .to_vrf();
            let output: Rows<f32, S, Chips> = Window::<S>::stream(&mut device.main, &input)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &source)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), CUBIC)
                .vector_fp_binary(FpBinaryOp::AddF, 1_f32)
                .vector_widen_concat::<
                    m![{ Width<S> } % ({ Segment<S> }) / 8],
                    m![{ Width<S> } % 8],
                >()
                .vector_final()
                .commit_trim::<m![{ Width<S> } % 8]>()
                .commit();
            let output: Rows<f32, S, Chips> = device
                .main
                .begin(output.view())
                .fetch::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
                .collect::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<
                    m![{ Width<S> } % ({ Segment<S> }) / 4],
                    m![{ Width<S> } % 4],
                >()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &source)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), ROOT)
                .vector_fp_unary(FpUnaryOp::Tanh)
                .vector_widen_concat::<
                    m![{ Width<S> } % ({ Segment<S> }) / 8],
                    m![{ Width<S> } % 8],
                >()
                .vector_final()
                .commit_trim::<m![{ Width<S> } % 8]>()
                .commit();
            device
                .main
                .begin(output.view())
                .fetch::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
                .collect::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<
                    m![{ Width<S> } % ({ Segment<S> }) / 4],
                    m![{ Width<S> } % 4],
                >()
                .vector_fp_binary(FpBinaryOp::AddF, 1_f32)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &source)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), 0.5_f32)
                .vector_widen_concat::<
                    m![{ Width<S> } % ({ Segment<S> }) / 8],
                    m![{ Width<S> } % 8],
                >()
                .vector_final()
                .commit_trim::<m![{ Width<S> } % 8]>()
                .commit()
        })
    }
}

/// The tanh GELU of a launch's rows in feature slices in a single-chip tensor's FP32 steps, their
/// live rows kept; later passes read `x` beside the pass before, so rows never fill the registers.
impl<'x, W: AxisName, R: AxisName, Chips: M + 'static> Op<&'x Live<Spread<W, R, Chips>, R>>
    for NewGelu
{
    type Output = Live<Spread<W, R, Chips>, R>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'x Live<Spread<W, R, Chips>, R>,
    ) -> Self::Output {
        type Wide<W, R, Chips> = DmTensor<f32, Chips, Cluster, FeatureSlices<W>, Element<W, R>>;
        type Narrow<W, R> = m![R, W % ({ Symbol<Chunk<W>> }) / 4];
        let x: Wide<W, R, Chips> = device
            .main
            .begin(input.value().view())
            .fetch::<SpreadTime<W, R>, SpreadPacket<W>>()
            .fetch_cast::<f32>()
            .collect::<SpreadTime<W, R>, SpreadPacket<W>>()
            .commit_trim::<SpreadPacket<W>>()
            .commit();
        let inner: Wide<W, R, Chips> = device
            .main
            .begin(input.value().view())
            .fetch::<SpreadTime<W, R>, SpreadPacket<W>>()
            .fetch_cast::<f32>()
            .collect::<SpreadTime<W, R>, SpreadPacket<W>>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<Narrow<W, R>, m![W % 4]>()
            .vector_fp_binary_with_mode(
                FpBinaryOp::MulF(FpMulAlu::Mul0),
                BinaryArgMode::Mode00,
                0.0f32,
            )
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), CUBIC)
            .vector_fp_binary(FpBinaryOp::AddF, 1_f32)
            .vector_widen_concat::<SpreadTime<W, R>, SpreadPacket<W>>()
            .vector_final()
            .commit_trim::<SpreadPacket<W>>()
            .commit();
        let tanh: Wide<W, R, Chips> = device
            .main
            .begin_interleaved::<Factors, _, _, _, _, _>(inner.view(), x.view())
            .fetch::<SpreadPairs<W, R>, SpreadPacket<W>>()
            .collect::<SpreadPairs<W, R>, SpreadPacket<W>>()
            .vector_init()
            .vector_intra_slice_unzip::<Factors, SpreadTime<W, R>>()
            .vector_narrow_split::<Narrow<W, R>, m![W % 4]>()
            .vector_fp_zip(FpBinaryOp::MulF(FpMulAlu::Mul0))
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), ROOT)
            .vector_fp_unary(FpUnaryOp::Tanh)
            .vector_widen_concat::<SpreadTime<W, R>, SpreadPacket<W>>()
            .vector_final()
            .commit_trim::<SpreadPacket<W>>()
            .commit();
        let value = device
            .main
            .begin_interleaved::<Factors, _, _, _, _, _>(tanh.view(), x.view())
            .fetch::<SpreadPairs<W, R>, SpreadPacket<W>>()
            .collect::<SpreadPairs<W, R>, SpreadPacket<W>>()
            .vector_init()
            .vector_intra_slice_unzip::<Factors, SpreadTime<W, R>>()
            .vector_narrow_split::<Narrow<W, R>, m![W % 4]>()
            .vector_fp_binary(FpBinaryOp::AddF, 1_f32, ())
            .vector_fp_zip(FpBinaryOp::MulF(FpMulAlu::Mul0))
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), 0.5_f32)
            .vector_widen_concat::<SpreadTime<W, R>, SpreadPacket<W>>()
            .vector_final()
            .cast::<bf16, m![W % 8 # 16]>()
            .commit_trim::<m![W % 8]>()
            .commit();
        input.with(value)
    }
}

/// x * sigmoid(x), preserving any input chip mapping.
#[derive(Clone, Copy, Default, DeviceSend, Load)]
pub struct Silu;

impl<D: Float, S: M + 'static, Chips: M> Op<&HbmTensor<D, Chips, S>> for Silu {
    type Output = HbmTensor<D, Chips, S>;

    /// Each window's `x * sigmoid(x)` in FP32, `x` held in the stash.
    fn forward(&self, device: &mut Device, input: &HbmTensor<D, Chips, S>) -> Self::Output {
        Window::<S>::map(device, input, |device, input| {
            Window::<S>::stream(&mut device.main, &input)
                .vector_silu()
                .vector_widen_concat::<
                    m![{ Width<S> } % ({ Segment<S> }) / 8],
                    m![{ Width<S> } % 8],
                >()
                .vector_final()
                .commit_trim::<m![{ Width<S> } % 8]>()
                .commit()
        })
    }
}

/// 1 / (1 + exp(-x)), preserving any input chip mapping.
#[derive(Clone, Copy, Default, DeviceSend, Load)]
pub struct Sigmoid;

impl<D: Float, S: M + 'static, Chips: M> Op<&HbmTensor<D, Chips, S>> for Sigmoid {
    type Output = HbmTensor<D, Chips, S>;

    fn forward(&self, device: &mut Device, input: &HbmTensor<D, Chips, S>) -> Self::Output {
        Window::<S>::map(device, input, |device, input| {
            Window::<S>::stream(&mut device.main, &input)
                .vector_fp_unary(FpUnaryOp::Sigmoid)
                .vector_widen_concat::<
                    m![{ Width<S> } % ({ Segment<S> }) / 8],
                    m![{ Width<S> } % 8],
                >()
                .vector_final()
                .commit_trim::<m![{ Width<S> } % 8]>()
                .commit()
        })
    }
}

/// Hyperbolic tangent on a single-chip tensor.
#[derive(Default, Clone, Copy, DeviceSend, Load)]
pub struct Tanh;

impl<D: Float, S: M + 'static, Chips: M + 'static> Op<&HbmTensor<D, Chips, S>> for Tanh {
    type Output = HbmTensor<D, Chips, S>;

    fn forward(&self, device: &mut Device, input: &HbmTensor<D, Chips, S>) -> Self::Output {
        Window::<S>::map(device, input, |device, input| {
            Window::<S>::stream(&mut device.main, &input)
                .vector_fp_unary(FpUnaryOp::Tanh)
                .vector_widen_concat::<
                    m![{ Width<S> } % ({ Segment<S> }) / 8],
                    m![{ Width<S> } % 8],
                >()
                .vector_final()
                .commit_trim::<m![{ Width<S> } % 8]>()
                .commit()
        })
    }
}

/// `c * tanh(x / c)`, which bounds each value by `c` and keeps the values near zero; `c` and `1 / c`
/// are FP32 scalars in the first cell of eight a load places.
#[derive(DeviceSend, Load)]
#[load(tree)]
pub struct Softcap<Factor = HbmTensor<f32, Chip, m![1 # 8]>> {
    inverse: Factor,
    bound: Factor,
}

/// `c` is the positive finite config value at the key the softcap's position enters, and `1 / c`
/// rounds once from its FP64 reciprocal, as PyTorch's CUDA kernels round a scalar's.
impl<'a, Factor> Load<&'a Checkpoint> for Softcap<Factor>
where
    Factor: Load<HostTensor<f32, m![{ Chip }, { m![1 # 8] }]>>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let key = cx.config_key().map(str::to_owned);
            let key = key.expect("a softcap reads its bound under a config key");
            let (name, bound) = cx.config_number(from, &key)?;
            if !bound.is_finite() || bound <= 0.0 {
                return Err(LoadError::InvalidData {
                    name,
                    reason: format!("found {bound}, expected a positive finite value"),
                });
            }
            let packet = |value: f64| {
                HostTensor::from_vec([value as f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            };
            let inverse = cx.field("inverse", "", |cx| Factor::load(cx, packet(1.0 / bound)))?;
            let bound = cx.field("bound", "", |cx| Factor::load(cx, packet(bound)))?;
            Ok(Self { inverse, bound })
        })
    }
}

/// The bound and its reciprocal are no update targets.
impl<'t, 's, Factor> Load<(&'t mut Softcap<Factor>, &'s Checkpoint)> for &'t mut Softcap<Factor> {
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut Softcap<Factor>, &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |_| Ok(()))?;
        Ok(into)
    }
}

impl<Factor> Softcap<Factor> {
    /// The bound and its reciprocal in the vector registers of a vocabulary pass's logits, for the
    /// passes of a call to read.
    pub(crate) fn to_vrf<Hidden: AxisName, V: AxisName, Chips: M + 'static>(
        &self,
        device: &mut Device,
    ) -> Softcap<Register<Hidden, V, Chips>>
    where
        Factor: Borrow<HbmTensor<f32, Chips, m![1 # 8]>>,
    {
        type Placed<Hidden, V, Chips> =
            DmTensor<f32, Chips, stream::Cluster<V>, stream::Reduced<Hidden, V>, m![1 # 8]>;
        let inverse: Placed<Hidden, V, Chips> = self.inverse.borrow().to_dm(&mut device.tdma);
        let bound: Placed<Hidden, V, Chips> = self.bound.borrow().to_dm(&mut device.tdma);
        Softcap {
            inverse: scalar::register(device, &inverse),
            bound: scalar::register(device, &bound),
        }
    }
}

/// Each logit `z` of a vocabulary pass becomes `c * tanh(z * (1 / c))` in FP32, the reciprocal
/// multiplying as PyTorch's CUDA kernels divide by a scalar.
impl<H: AxisName, V: AxisName, R: AxisName, Chips: M + 'static> Op<&stream::Logits<H, V, R, Chips>>
    for Softcap<Register<H, V, Chips>>
{
    type Output = stream::Logits<H, V, R, Chips>;

    fn forward(
        &self,
        device: &mut Device,
        logits: &stream::Logits<H, V, R, Chips>,
    ) -> Self::Output {
        type Time<H, V, R> = m![V % ({ stream::PassRows<H, V> }), { Groups<R> }];
        type Lanes<R> = m![{ Layout<R>::Lanes } # 8];
        device
            .main
            .begin(logits.view())
            .fetch::<Time<H, V, R>, Lanes<R>>()
            .collect::<Time<H, V, R>, Lanes<R>>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<
                m![V % ({ stream::PassRows<H, V> }), { Groups<R> }, { Layout<R>::Lanes } # 8 / 4],
                m![{ Layout<R>::Lanes } # 8 % 4],
            >()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &self.inverse)
            .vector_fp_unary(FpUnaryOp::Tanh)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &self.bound)
            .vector_widen_concat::<Time<H, V, R>, Lanes<R>>()
            .vector_final()
            .commit_trim::<Lanes<R>>()
            .commit()
    }
}

/// One FP32 scalar in every slice of a vocabulary pass's logits.
type Register<Hidden, V, Chips> =
    VrfTensor<f32, Chips, stream::Cluster<V>, stream::Reduced<Hidden, V>, m![1 # 8]>;

/// `x * sigmoid(x)` of every row of one launch in feature slices, in FP32, rounded to BF16 once;
/// the result keeps their live rows.
impl<'x, W: AxisName, R: AxisName, Chips: M + 'static> Op<&'x Live<Spread<W, R, Chips>, R>>
    for Silu
{
    type Output = Live<Spread<W, R, Chips>, R>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'x Live<Spread<W, R, Chips>, R>,
    ) -> Self::Output {
        let value = device
            .main
            .begin(input.value().view())
            .fetch::<SpreadTime<W, R>, SpreadPacket<W>>()
            .fetch_cast::<f32>()
            .collect::<SpreadTime<W, R>, SpreadPacket<W>>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![R, W % ({ Symbol<Chunk<W>> }) / 4], m![W % 4]>()
            .vector_silu()
            .vector_widen_concat::<SpreadTime<W, R>, SpreadPacket<W>>()
            .vector_final()
            .cast::<bf16, m![W % 8 # 16]>()
            .commit_trim::<m![W % 8]>()
            .commit();
        input.with(value)
    }
}

/// `bf16(silu(t))` of each timestep's time embedding, FP32 inside and rounded once.
impl<In: AxisName, Chips: M> Op<HbmTensorView<'_, f32, Chips, m![Timestep, In]>> for Silu {
    type Output = DmTensor<bf16, Chips, m![2], m![1 # 256], m![Timestep, In]>;

    fn forward(
        &self,
        device: &mut Device,
        time: HbmTensorView<'_, f32, Chips, m![Timestep, In]>,
    ) -> Self::Output {
        let in_dm: DmTensor<f32, Chips, m![2], m![1 # 256], m![Timestep, In]> =
            time.to_dm(&mut device.tdma);
        device
            .main
            .begin(in_dm.view())
            .fetch::<m![Timestep, In / 8], m![In % 8]>()
            .collect::<m![Timestep, In / 8], m![In % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Timestep, In / 8, In % 8 / 4 % 2], m![In % 4]>()
            .vector_silu()
            .vector_widen_concat::<m![Timestep, In / 8], m![In % 8]>()
            .vector_final()
            .cast::<bf16, m![In % 8 # 16]>()
            .commit_trim::<m![In % 8]>()
            .commit()
    }
}

#[cfg(test)]
mod tests {
    use super::{Gelu, NewGelu, QuickGelu, Sigmoid, Tanh};
    use crate::Op;
    use crate::ops::index::scatter::Store;
    use crate::ops::normalization::Spread;
    use crate::tensor::live::{Live, tests::count};
    use crate::tensor::scalar::Value;
    use crate::tests::DEVICE;
    use furiosa_opt_std::{prelude::*, runtime::launch};

    axes![Gates = 4096, Rows = 8, Expanded = 15360];

    #[device(chip = 1)]
    fn sigmoid(
        device: &mut Device,
        input: &HbmTensor<bf16, m![1], m![Gates]>,
    ) -> HbmTensor<bf16, m![1], m![Gates]> {
        Sigmoid.forward(device, input)
    }

    /// The logistic function of each BF16 gate is its FP32 value rounded once, saturating at
    /// both ends.
    #[tokio::test(flavor = "current_thread")]
    async fn sigmoid_rounds_once() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(sigmoid.topology()).unwrap();
        let gates: Vec<bf16> = (0..Gates::SIZE)
            .map(|at| bf16::from_f32((at as f32 - 2048.0) / 50.0))
            .collect();
        let input = HostTensor::<bf16, m![Gates]>::from_vec(gates.clone())
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let output = launch(sigmoid, (&mut device, &input))
            .await
            .unwrap()
            .to_host::<m![Gates]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (gate, output) in gates.iter().zip(output) {
            let expected = bf16::from_f32(1.0 / (1.0 + (-gate.to_f32()).exp()));
            let gate = gate.to_f32();
            assert_eq!(output.to_bits(), expected.to_bits(), "sigmoid({gate})");
        }
    }

    #[device(chip = 1)]
    fn tanh<S: M + 'static>(
        device: &mut Device,
        input: &HbmTensor<f32, m![1], S>,
    ) -> HbmTensor<f32, m![1], S> {
        Tanh.forward(device, input)
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

    #[device(chip = 1)]
    fn activated<A, S: M + 'static>(
        device: &mut Device,
        activation: &A,
        input: &HbmTensor<bf16, m![1], S>,
    ) -> HbmTensor<bf16, m![1], S>
    where
        A: DeviceSend
            + for<'x> Op<&'x HbmTensor<bf16, m![1], S>, Output = HbmTensor<bf16, m![1], S>>,
    {
        activation.forward(device, input)
    }

    /// `activation` of each BF16 input of shape `S`, cycling over [-8, 8), on the device, within
    /// one BF16 step of `definition` in FP64, and within 2^-16 of it near zero.
    async fn matches<A, S: M + 'static>(activation: A, definition: fn(f64) -> f64)
    where
        A: DeviceSend
            + for<'x> Op<&'x HbmTensor<bf16, m![1], S>, Output = HbmTensor<bf16, m![1], S>>,
    {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let inputs: Vec<bf16> = (0..S::SIZE)
            .map(|at| bf16::from_f32((at % 4096) as f32 / 256.0 - 8.0))
            .collect();
        let input = HostTensor::<bf16, S>::from_vec(inputs.clone())
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let output = launch(activated, (&mut device, &activation, &input))
            .await
            .unwrap()
            .to_host::<S>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (&x, got) in inputs.iter().zip(output) {
            assert_close(x, got, definition);
        }
    }

    /// `got` for input `x` is within one BF16 step of `definition` in FP64, and within 2^-16 of it
    /// near zero.
    fn assert_close(x: bf16, got: bf16, definition: fn(f64) -> f64) {
        let (x, got) = (f64::from(x.to_f32()), f64::from(got.to_f32()));
        let want = definition(x);
        let step = (want.abs().log2().floor() - 7.0).exp2();
        assert!(
            (got - want).abs() <= step.max(2f64.powi(-16)),
            "{x}: {got} vs {want}"
        );
    }

    /// `x / 2 * (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`.
    fn tanh_approximation(x: f64) -> f64 {
        x / 2.0 * (1.0 + ((2.0 / core::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
    }

    /// GELU is `x Phi(x)` through the error function, not its tanh approximation.
    #[tokio::test(flavor = "current_thread")]
    async fn gelu_is_exact() {
        matches::<_, m![Gates]>(Gelu, |x| x / 2.0 * (1.0 + (x / 2f64.sqrt()).erf())).await;
    }

    /// Quick GELU is `x sigmoid(1.702 x)`.
    #[tokio::test(flavor = "current_thread")]
    async fn quick_gelu_is_a_scaled_sigmoid() {
        matches::<_, m![Gates]>(QuickGelu, |x| x / (1.0 + (-1.702 * x).exp())).await;
    }

    /// New GELU is the tanh approximation `x / 2 * (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`, also
    /// over rows of a width no power of two divides.
    #[tokio::test(flavor = "current_thread")]
    async fn new_gelu_is_the_tanh_approximation() {
        matches::<_, m![Gates]>(NewGelu, tanh_approximation).await;
        matches::<_, m![Rows, Expanded]>(NewGelu, tanh_approximation).await;
    }

    #[device(chip = 1)]
    fn activated_rows<A>(
        device: &mut Device,
        activation: &A,
        input: &HbmTensor<bf16, m![1], m![Rows, Expanded]>,
        (start, rows): (&Value, &Value),
        output: &mut HbmTensor<bf16, m![1], m![Rows, Expanded]>,
    ) where
        A: DeviceSend
            + for<'x> Op<
                &'x Live<Spread<Expanded, Rows>, Rows>,
                Output = Live<Spread<Expanded, Rows>, Rows>,
            >,
    {
        let live =
            Live::<(), Rows>::all().subrange(start.extent::<m![Rows]>(), rows.extent::<m![Rows]>());
        let input: Live<Spread<Expanded, Rows>, Rows> = live.with(input.to_dm(&mut device.tdma));
        let result = activation.forward(&mut *device, &input);
        Store.forward(device, ((result.value(), result.with(())), output));
    }

    /// New GELU of a launch's live rows in feature slices is each row's tanh approximation, bit for
    /// bit as the rows alone give it, wherever the live rows start; the other rows keep the output's.
    #[tokio::test(flavor = "current_thread")]
    async fn new_gelu_of_live_rows_is_the_tanh_approximation() {
        const KEPT: f32 = -3.0;
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let values: Vec<bf16> = (0..Rows::SIZE * Expanded::SIZE)
            .map(|at| bf16::from_f32((at % 4096) as f32 / 256.0 - 8.0))
            .collect();
        let upload = async |device: &mut Device, values: Vec<bf16>| {
            HostTensor::<bf16, m![Rows, Expanded]>::from_vec(values)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap()
        };
        let whole = upload(&mut device, values.clone()).await;
        let alone = launch(activated, (&mut device, &NewGelu, &whole))
            .await
            .unwrap()
            .to_host::<m![Rows, Expanded]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (first, rows) in [(3, 4), (0, 8), (7, 1)] {
            let live = |at: usize| (first..first + rows).contains(&(at / Expanded::SIZE));
            let input = values.iter().enumerate().map(|(at, &x)| match live(at) {
                true => x,
                false => bf16::from_f32(f32::NAN),
            });
            let input = upload(&mut device, input.collect()).await;
            let mut output = upload(
                &mut device,
                vec![bf16::from_f32(KEPT); Rows::SIZE * Expanded::SIZE],
            )
            .await;
            let (start, count) = (
                count(first).to_hbm(&mut device.pdma).await.unwrap(),
                count(rows).to_hbm(&mut device.pdma).await.unwrap(),
            );
            launch(
                activated_rows,
                (&mut device, &NewGelu, &input, (&start, &count), &mut output),
            )
            .await
            .unwrap();
            let output = output
                .to_host::<m![Rows, Expanded]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            for (at, &got) in output.iter().enumerate() {
                let want = if live(at) {
                    alone[at]
                } else {
                    bf16::from_f32(KEPT)
                };
                assert_eq!(got.to_bits(), want.to_bits(), "rows {first}+{rows}: {at}");
                if live(at) {
                    assert_close(values[at], got, tanh_approximation);
                }
            }
        }
    }
}
