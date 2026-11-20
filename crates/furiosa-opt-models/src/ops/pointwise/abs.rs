//! Elementwise magnitude, preserving the source precision and physical layout.

use super::{Float, Rows, Segment, Width, Window};
use crate::Op;
use crate::load::Load;
use crate::tensor::memory::residency::Residency;
use furiosa_opt_std::prelude::*;

/// Elementwise magnitude: clear each element's sign bit in FP32 and retain its element type.
#[derive(Clone, Default, DeviceSend, Load, Residency)]
pub struct Abs;

impl<D: Float, S: M + 'static, Chips: M> Op<&HbmTensor<D, Chips, S>> for Abs {
    type Output = HbmTensor<D, Chips, S>;

    fn forward(&self, device: &mut Device, input: &HbmTensor<D, Chips, S>) -> Self::Output {
        Window::<S>::map(device, input, |device, input| magnitude(device, &input))
    }
}

/// One window, including an already rounded intermediate from another pointwise operation.
impl<D: Float, S: M + 'static, Chips: M> Op<&Rows<D, S, Chips>> for Abs {
    type Output = Rows<D, S, Chips>;

    fn forward(&self, device: &mut Device, input: &Rows<D, S, Chips>) -> Self::Output {
        let output = magnitude(device, input);
        D::narrow(device, output)
    }
}

fn magnitude<D: Float, S: M + 'static, Chips: M>(
    device: &mut Device,
    input: &Rows<D, S, Chips>,
) -> Rows<f32, S, Chips> {
    device
        .main
        .begin(input.view())
        .fetch::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, i32::MAX)
        .vector_reinterpret::<f32>()
        .vector_final()
        .commit_trim::<m![{ Width<S> } % 8]>()
        .commit()
}

#[cfg(test)]
mod tests {
    use super::Abs;
    use crate::Op;
    use furiosa_opt_std::{prelude::*, runtime::launch};

    type Input<D, S> = HbmTensor<D, m![1], S>;

    #[device(chip = 1)]
    fn absolute<S: M + 'static>(
        device: &mut Device,
        (fp32, bf16): (&Input<f32, S>, &Input<bf16, S>),
    ) -> (Input<f32, S>, Input<bf16, S>) {
        (Abs.forward(&mut *device, fp32), Abs.forward(device, bf16))
    }

    async fn matches<S: M + 'static>() {
        let mut device = Device::new(absolute.topology()).unwrap();
        let cases = [
            -0.0_f32,
            0.0,
            -1.0,
            1.0,
            -1.00390625,
            1.00390625,
            f32::NEG_INFINITY,
            f32::INFINITY,
        ];
        let fp32: Vec<f32> = (0..S::SIZE).map(|at| cases[at % cases.len()]).collect();
        let bf16: Vec<bf16> = fp32.iter().map(|value| bf16::from_f32(*value)).collect();
        let fp32_input: Input<f32, S> = HostTensor::<f32, S>::from_vec(fp32.clone())
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let bf16_input: Input<bf16, S> = HostTensor::<bf16, S>::from_vec(bf16.clone())
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let (fp32_output, bf16_output) =
            launch(absolute, (&mut device, (&fp32_input, &bf16_input)))
                .await
                .unwrap();
        let fp32_output = fp32_output
            .to_host::<S>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let bf16_output = bf16_output
            .to_host::<S>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (index, (got, input)) in fp32_output.iter().zip(&fp32).enumerate() {
            assert_eq!(got.to_bits(), input.abs().to_bits(), "FP32 element {index}");
        }
        for (index, (got, input)) in bf16_output.iter().zip(&bf16).enumerate() {
            assert_eq!(
                got.to_bits(),
                bf16::from_f32(input.to_f32().abs()).to_bits(),
                "BF16 element {index}"
            );
        }
    }

    /// HBM mapping preserves each magnitude exactly in both precisions, including signed zero,
    /// scalar padding, and a complete-vector tail eight elements past 512.
    #[tokio::test(flavor = "current_thread")]
    async fn preserves_magnitudes() {
        let _lease = crate::tests::DEVICE.lock().await;
        matches::<m![1]>().await;
        matches::<m![520]>().await;
    }
}
