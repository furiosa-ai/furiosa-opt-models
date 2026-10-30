//! Affine maps of each element: a scale and a shift, and decoded image values into [0, 1].

use super::{Float, Rows, Segment, Width, Window};
use crate::Op;
use crate::load::Load;
use crate::ops::pointwise::clamp::Clamp;
use crate::tensor::scalar::F32;
use furiosa_opt_std::prelude::*;

/// Convert decoded image values from [-1, 1] into bounded [0, 1] samples.
#[derive(DeviceSend, Load)]
pub struct Denormalize;

impl<S: M + 'static> Op<&HbmTensor<f32, m![1], S>> for Denormalize {
    type Output = HbmTensor<f32, m![1], S>;

    /// Each window `x / 2 + 1 / 2`, then held in [0, 1], in two DM round trips.
    fn forward(&self, device: &mut Device, input: &HbmTensor<f32, m![1], S>) -> Self::Output {
        type Half = F32<{ 0.5f32.to_bits() as usize }>;
        type Unit = Clamp<F32<{ 0.0f32.to_bits() as usize }>, F32<{ 1.0f32.to_bits() as usize }>>;
        Window::<S>::map(device, [input], |device, [input]| {
            let scaled = Rescale::<Half, Half>::window(device, &input);
            Unit::window(device, &scaled)
        })
    }
}

/// Apply elementwise `input * scale + shift` with scalar broadcast.
#[derive(DeviceSend)]
pub struct Rescale<Scale = HbmTensor<f32, m![1], m![1]>, Shift = HbmTensor<f32, m![1], m![1]>> {
    /// Multiplicative scalar.
    pub scale: Scale,
    /// Additive scalar.
    pub shift: Shift,
}

impl<S: M> Op<&HbmTensor<f32, m![1], S>> for Rescale {
    type Output = HbmTensor<f32, m![1], S>;

    fn forward(&self, _: &mut Device, _: &HbmTensor<f32, m![1], S>) -> Self::Output {
        todo!("NPU elementwise input * scale + shift with scalar broadcast")
    }
}

/// [`F32`] scalars, fixed when the kernel compiles.
impl<const SCALE: usize, const SHIFT: usize> Rescale<F32<SCALE>, F32<SHIFT>> {
    /// A window scaled, then shifted, in one DM round trip.
    fn window<D: Float, S: M + 'static, Chips: M>(
        device: &mut Device,
        input: &Rows<D, S, Chips>,
    ) -> Rows<f32, S, Chips> {
        Window::<S>::stream(&mut device.main, input)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), f32::from(F32::<SCALE>))
            .vector_fp_binary(FpBinaryOp::AddF, f32::from(F32::<SHIFT>))
            .vector_widen_concat::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
            .vector_final()
            .commit_trim::<m![{ Width<S> } % 8]>()
            .commit()
    }
}

#[cfg(test)]
mod tests {
    use super::Denormalize;
    use crate::Op;
    use crate::tests::DEVICE;
    use furiosa_opt_std::{prelude::*, runtime::launch};

    axes![Pixels = 4096];

    #[device(chip = 1)]
    fn denormalized(
        device: &mut Device,
        input: &HbmTensor<f32, m![1], m![Pixels]>,
    ) -> HbmTensor<f32, m![1], m![Pixels]> {
        Denormalize.forward(device, input)
    }

    /// Every value is `x / 2 + 1 / 2` held in [0, 1], bit for bit, inside the interval and past
    /// both ends.
    #[tokio::test(flavor = "current_thread")]
    async fn holds_halved_values_in_the_unit_interval() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(denormalized.topology()).unwrap();
        let values: Vec<f32> = (0..Pixels::SIZE)
            .map(|i| (i as f32 - 2048.0) / 683.0)
            .collect();
        let input = HostTensor::<f32, m![Pixels]>::from_vec(values.iter().copied())
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let output = launch(denormalized, (&mut device, &input))
            .await
            .unwrap()
            .to_host::<m![Pixels]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (x, y) in values.iter().zip(&output) {
            let expected = (x * 0.5 + 0.5).clamp(0.0, 1.0);
            assert_eq!(y.to_bits(), expected.to_bits(), "denormalize({x})");
        }
    }
}
