//! torch's `softplus` with its default beta and threshold: `log(1 + exp(x))`, and `x` above 20.

use super::{Float, Registers, Rows, Segment, Width, Window};
use crate::Op;
use crate::load::Load;
use furiosa_opt_std::prelude::*;

/// Above it torch's `softplus` returns its argument.
const THRESHOLD: f32 = 20.0;

/// `log(1 + exp(x))` for `x <= 20` and `x` above, preserving any input chip mapping.
#[derive(Clone, Copy, Default, DeviceSend, Load)]
pub struct Softplus;

impl<D: Float, S: M + 'static, Chips: M> Op<&HbmTensor<D, Chips, S>> for Softplus {
    type Output = HbmTensor<D, Chips, S>;

    /// Each window's `max(log(1 + exp(min(x, 20))), x)` in FP32, in three DM round trips; the log
    /// exceeds `x` exactly where `x` is at most the threshold.
    fn forward(&self, device: &mut Device, input: &HbmTensor<D, Chips, S>) -> Self::Output {
        Window::<S>::map(device, [input], |device, [input]| {
            let source: Registers<S, Chips> = device
                .sub
                .begin(input.view())
                .fetch::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
                .to_vrf();
            let held: Rows<f32, S, Chips> = device
                .main
                .begin(input.view())
                .fetch::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_clip(ClipBinaryOpF32::Min, THRESHOLD)
                .vector_final()
                .commit_trim::<m![{ Width<S> } % 8]>()
                .commit();
            let grown: Rows<f32, S, Chips> = Window::<S>::stream(&mut device.main, &held)
                .vector_fp_unary(FpUnaryOp::Exp)
                .vector_fp_binary(FpBinaryOp::AddF, 1_f32)
                .vector_widen_concat::<
                    m![{ Width<S> } % ({ Segment<S> }) / 8],
                    m![{ Width<S> } % 8],
                >()
                .vector_final()
                .commit_trim::<m![{ Width<S> } % 8]>()
                .commit();
            Window::<S>::stream(&mut device.main, &grown)
                .vector_fp_unary(FpUnaryOp::Log)
                .vector_widen_concat::<
                    m![{ Width<S> } % ({ Segment<S> }) / 8],
                    m![{ Width<S> } % 8],
                >()
                .vector_clip(ClipBinaryOpF32::Max, &source)
                .vector_final()
                .commit_trim::<m![{ Width<S> } % 8]>()
                .commit()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::Softplus;
    use crate::Op;
    use crate::tests::DEVICE;
    use furiosa_opt_std::{prelude::*, runtime::launch};

    axes![Steps = 4096];

    #[device(chip = 1)]
    fn softplus(
        device: &mut Device,
        input: &HbmTensor<f32, m![1], m![Steps]>,
    ) -> HbmTensor<f32, m![1], m![Steps]> {
        Softplus.forward(device, input)
    }

    /// Every element is torch's `softplus(x)` to FP32 precision, `x` itself above the threshold.
    #[tokio::test(flavor = "current_thread")]
    async fn matches_definition() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(softplus.topology()).unwrap();
        let values: Vec<f32> = (0..Steps::SIZE)
            .map(|at| (at as f32 - 2048.0) / 64.0)
            .collect();
        let input = HostTensor::<f32, m![Steps]>::from_vec(values.iter().copied())
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let output = launch(softplus, (&mut device, &input))
            .await
            .unwrap()
            .to_host::<m![Steps]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (&x, &y) in values.iter().zip(&output) {
            let expected = if x > 20.0 {
                x
            } else {
                f64::from(x).exp().ln_1p() as f32
            };
            assert!(
                (y - expected).abs() <= 1e-5 * expected.max(1.0),
                "softplus({x}) = {y}, expected {expected}"
            );
        }
    }
}
