//! Normalization of each channel group over its channels and spatial extent.

use crate::Op;
use crate::tensor::image::Image;
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

/// Normalize each channel group over its channels and spatial extent with FP32 statistics.
#[derive(DeviceSend)]
pub struct GroupNorm<Channels: AxisName, Groups: AxisName> {
    /// Learned scale per channel.
    pub weight: HbmTensor<f32, m![1], m![Channels]>,
    /// Learned offset per channel.
    pub bias: HbmTensor<f32, m![1], m![Channels]>,
    /// Device-resident scalar added to each group's variance.
    pub epsilon: HbmTensor<f32, m![1], m![1]>,
    /// Compile-time channel group count; Channels must divide evenly into Groups.
    pub groups: PhantomData<Groups>,
}

impl<Channels: AxisName, Groups: AxisName, H: AxisName, W: AxisName> Op<&Image<Channels, H, W>>
    for GroupNorm<Channels, Groups>
{
    type Output = Image<Channels, H, W>;

    fn forward(&self, _: &mut Device, _: &Image<Channels, H, W>) -> Self::Output {
        const { assert!(Groups::SIZE > 0 && Channels::SIZE % Groups::SIZE == 0) };
        todo!(
            "NPU group normalization with FP32 statistics over each channel group and its spatial extent"
        )
    }
}
