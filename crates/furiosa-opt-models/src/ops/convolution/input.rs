//! A latent's rows lifted for the convolution that folds the input projection into its own taps.
//!
//! - The input projection is linear and the convolution after it zero-pads its input, so the two
//!   fold into one convolution over the latent channels and a channel of ones that is zero on the
//!   padding rows, which carries the projection's bias exactly at the edges too.
//! - The latent's denormalization (`z * std + mean`) folds into the same coefficients.
use furiosa_opt_std::prelude::*;

use super::frames::{self, Frames, Stereo, Window};
use crate::Op;
use crate::load::Load;
use crate::tensor::parallel::Chip;

axes![Latent = 32, Halves = 2];

/// A latent's channels, a ones channel, then channels the folded coefficients weigh by zero.
#[derive(Clone, Debug)]
pub struct Lifted;

impl AxisName for Lifted {
    const NAME: Ident = Ident::new("Lifted");
    const SIZE: usize = 2 * Latent::SIZE;
}

/// The latent's rows, the span's rows after the live samples ignored.
pub type Rows<S, X = Chip> = HbmTensor<f32, X, m![Stereo, { Symbol<S> }, Latent]>;

/// `[z, 1, 1, 0, ...]` on every live row, zero on the padding rows.
#[derive(Clone, Copy, Debug, Default, DeviceSend, Load)]
pub struct Lift;

impl<'x, S: Window, X: M> Op<&'x Rows<S, X>> for Lift {
    type Output = Frames<f32, Lifted, S, X>;

    fn forward(&self, device: &mut Device, latent: &'x Rows<S, X>) -> Self::Output {
        let mut lifted = Frames::<f32, Lifted, S, X>::new();
        let mut ones: DmTensor<f32, X, m![1 # 2], m![1 # 256], m![Stereo, Latent]> =
            DmTensor::new();
        ones.view_mut().memset(0.0, &mut device.sub);
        // Device memsets cover whole eight bytes; the second one is written on a zero coefficient.
        ones.view_mut()
            .tile::<m![Latent], 2, m![Stereo, Latent = 2 #{!} Latent]>(0)
            .memset(1.0, &mut device.sub);
        {
            // SAFETY: the lifted channels are the latent's, then the ones channel's group.
            let interior: HbmTensorViewMut<
                '_,
                f32,
                X,
                m![Stereo, { Symbol<S> } #{!} ({ frames::FrameRows<S> }), Halves, Latent],
            > = unsafe { S::interior(&mut lifted).reshape() };
            latent.view().to_hbm_view(
                &mut device.tdma,
                interior.tile::<m![Halves], 1, m![
                    Stereo,
                    { Symbol<S> } #{!} ({ frames::FrameRows<S> }),
                    Halves = 1 #{!} Halves,
                    Latent
                ]>(0),
            );
        }
        {
            // SAFETY: the lifted channels are the latent's, then the ones channel's group.
            let interior: HbmTensorViewMut<
                '_,
                f32,
                X,
                m![Stereo, { Symbol<S> } #{!} ({ frames::FrameRows<S> }), Halves, Latent],
            > = unsafe { S::interior(&mut lifted).reshape() };
            ones.view().to_hbm_view(
                &mut device.tdma,
                interior.tile::<m![Halves], 1, m![
                    Stereo,
                    { Symbol<S> } #{!} ({ frames::FrameRows<S> }),
                    Halves = 1 #{!} Halves,
                    Latent
                ]>(1),
            );
        }
        frames::Zero.forward(&mut *device, (&mut lifted, S::SAMPLES));
        lifted
    }
}
