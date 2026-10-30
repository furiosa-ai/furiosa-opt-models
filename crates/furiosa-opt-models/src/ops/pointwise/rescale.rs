//! Affine maps of each element: a scale and a shift, and decoded image values into [0, 1].

use super::{Rows, Segment, Width, Window};
use crate::Op;
use crate::load::Load;
use furiosa_opt_std::prelude::*;

/// Convert decoded image values from [-1, 1] into bounded [0, 1] samples.
#[derive(DeviceSend, Load)]
pub struct Denormalize;

impl<S: M + 'static> Op<&HbmTensor<f32, m![1], S>> for Denormalize {
    type Output = HbmTensor<f32, m![1], S>;

    /// Each window in three DM round trips: `x / 2 + 1 / 2`, then the clip at 0, then the clip
    /// at 1.
    fn forward(&self, device: &mut Device, input: &HbmTensor<f32, m![1], S>) -> Self::Output {
        Window::<S>::map(device, input, |device, input| {
            let output: Rows<f32, S> = Window::<S>::stream(&mut device.main, &input)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.5_f32)
                .vector_fp_binary(FpBinaryOp::AddF, 0.5_f32)
                .vector_widen_concat::<
                    m![{ Width<S> } % ({ Segment<S> }) / 8],
                    m![{ Width<S> } % 8],
                >()
                .vector_final()
                .commit_trim::<m![{ Width<S> } % 8]>()
                .commit();
            let output: Rows<f32, S> = device
                .main
                .begin(output.view())
                .fetch::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
                .collect::<m![{ Width<S> } % ({ Segment<S> }) / 8], m![{ Width<S> } % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_clip(ClipBinaryOpF32::Max, 0.0_f32)
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
                .vector_clip(ClipBinaryOpF32::Min, 1.0_f32)
                .vector_final()
                .commit_trim::<m![{ Width<S> } % 8]>()
                .commit()
        })
    }
}

/// Apply elementwise `input * scale + shift` with scalar broadcast.
#[derive(DeviceSend)]
pub struct Rescale {
    /// Multiplicative scalar.
    pub scale: HbmTensor<f32, m![1], m![1]>,
    /// Additive scalar.
    pub shift: HbmTensor<f32, m![1], m![1]>,
}

impl<S: M> Op<&HbmTensor<f32, m![1], S>> for Rescale {
    type Output = HbmTensor<f32, m![1], S>;

    fn forward(&self, _: &mut Device, _: &HbmTensor<f32, m![1], S>) -> Self::Output {
        todo!("NPU elementwise input * scale + shift with scalar broadcast")
    }
}
