//! torch's `clamp`: every element held between two bounds fixed when the kernel compiles.

use super::{Float, Rows, Segment, Width, Window};
use crate::Op;
use crate::load::Load;
use crate::tensor::scalar::F32;
use core::marker::PhantomData;
use furiosa_opt_std::prelude::*;

/// `min(max(x, Lower), Upper)` with [`F32`] bounds, preserving any input chip mapping.
#[derive(DeviceSend, Load)]
pub struct Clamp<Lower, Upper>(PhantomData<(Lower, Upper)>);

impl<Lower, Upper> Default for Clamp<Lower, Upper> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<D: Float, S: M + 'static, Chips: M, const LOWER: usize, const UPPER: usize>
    Op<&HbmTensor<D, Chips, S>> for Clamp<F32<LOWER>, F32<UPPER>>
{
    type Output = HbmTensor<D, Chips, S>;

    fn forward(&self, device: &mut Device, input: &HbmTensor<D, Chips, S>) -> Self::Output {
        Window::<S>::map(device, [input], |device, [input]| {
            Self::window(device, &input)
        })
    }
}

impl<const LOWER: usize, const UPPER: usize> Clamp<F32<LOWER>, F32<UPPER>> {
    /// A window raised to the lower bound, then lowered to the upper one, in one DM round trip.
    pub(super) fn window<D: Float, S: M + 'static, Chips: M>(
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
            .vector_clip(ClipBinaryOpF32::Max, f32::from(F32::<LOWER>))
            .vector_clip(ClipBinaryOpF32::Min, f32::from(F32::<UPPER>))
            .vector_final()
            .commit_trim::<m![{ Width<S> } % 8]>()
            .commit()
    }
}

#[cfg(test)]
mod tests {
    use super::Clamp;
    use crate::Op;
    use crate::tensor::scalar::F32;
    use crate::tests::DEVICE;
    use furiosa_opt_std::{prelude::*, runtime::launch};

    axes![Features = 4096];
    type Unit = Clamp<F32<{ (-1.0f32).to_bits() as usize }>, F32<{ 1.0f32.to_bits() as usize }>>;

    #[device(chip = 1)]
    fn clamp(
        device: &mut Device,
        input: &HbmTensor<f32, m![1], m![Features]>,
    ) -> HbmTensor<f32, m![1], m![Features]> {
        Unit::default().forward(device, input)
    }

    /// Every element equals torch's `clamp(x, -1, 1)`, bit for bit.
    #[tokio::test(flavor = "current_thread")]
    async fn matches_definition() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(clamp.topology()).unwrap();
        let values: Vec<f32> = (0..4096).map(|i| (i as f32 - 2048.0) / 512.0).collect();
        let input = HostTensor::<f32, m![Features]>::from_vec(values.iter().copied())
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let output = launch(clamp, (&mut device, &input))
            .await
            .unwrap()
            .to_host::<m![Features]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (x, y) in values.iter().zip(&output) {
            assert_eq!(y.to_bits(), x.clamp(-1.0, 1.0).to_bits(), "clamp({x})");
        }
    }
}
