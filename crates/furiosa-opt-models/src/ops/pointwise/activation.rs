//! Activation functions: each one function of one element.

use super::{Float, Registers, Rows, Segment, SpreadPacket, SpreadTime, Width, Window};
use crate::Op;
use crate::load::Load;
use crate::ops::index::packed::Timestep;
use crate::ops::normalization::{Chunk, Spread};
use crate::ops::vector::Silu as _;
use crate::tensor::live::Live;
use furiosa_opt_std::prelude::*;

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
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), 0.044715_f32)
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
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), 0.79788456_f32)
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
        for (x, got) in inputs.iter().zip(output) {
            let (x, got) = (f64::from(x.to_f32()), f64::from(got.to_f32()));
            let want = definition(x);
            let step = (want.abs().log2().floor() - 7.0).exp2();
            assert!(
                (got - want).abs() <= step.max(2f64.powi(-16)),
                "{x}: {got} vs {want}"
            );
        }
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
        let definition = |x: f64| {
            x / 2.0
                * (1.0 + ((2.0 / core::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
        };
        matches::<_, m![Gates]>(NewGelu, definition).await;
        matches::<_, m![Rows, Expanded]>(NewGelu, definition).await;
    }
}
