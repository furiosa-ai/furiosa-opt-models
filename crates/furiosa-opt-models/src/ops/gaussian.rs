//! The mode of a diagonal Gaussian over moments of channels [mean, log-variance], and the moment
//! and latent images its relation reads.

use crate::Op;
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

/// Channels containing the mean followed by the log-variance.
#[derive(Clone, Debug)]
pub struct Moments<Channels>(PhantomData<Channels>);

impl<Channels: AxisName> AxisName for Moments<Channels> {
    const NAME: Ident = Ident::new("Moments");
    const SIZE: usize = 2 * Channels::SIZE;
}

/// Channel-first mean and log-variance images.
pub type Input<C, H, W> = HbmTensor<f32, m![1], m![{ Symbol<Moments<C>> }, H, W]>;

/// Channel-first latent image with one channel per Gaussian variable.
pub type Latent<C, H, W> = HbmTensor<f32, m![1], m![C, H, W]>;

type Mean<C> = Padding<m![{ Symbol<Moments<C>> } % C], m![{ Symbol<Moments<C>> }]>;

/// Return the mean channels of the Gaussian moments.
#[derive(DeviceSend)]
pub struct Mode;

impl<C: AxisName, H: AxisName, W: AxisName> Op<&Input<C, H, W>> for Mode {
    type Output = Latent<C, H, W>;

    fn forward(&self, device: &mut Device, parameters: &Input<C, H, W>) -> Self::Output {
        let mean = parameters
            .view()
            .tile::<m![{ Symbol<Moments<C>> } / C], 1, m![{ Mean<C> }, H, W]>(0);
        let mut output = HbmTensor::<f32, m![1], m![{ Symbol<Moments<C>> } % C, H, W]>::new();
        mean.to_hbm_view(&mut device.tdma, output.view_mut());
        // Relabel the copied channel half after its original buffer extent is removed.
        unsafe { output.reshape::<m![1], m![C, H, W]>() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::DEVICE;
    use crate::tests::forward;

    axes![Channels = 4, Height = 2, Width = 8];

    /// The mode of a diagonal Gaussian is its mean: the first half of the moment channels, as
    /// Diffusers' `DiagonalGaussianDistribution.mode()` returns it.
    #[tokio::test(flavor = "current_thread")]
    async fn returns_mean_channels() {
        type C = Channels;
        type H = Height;
        type W = Width;
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(forward.topology()).unwrap();
        let mean: Vec<_> = (0..C::SIZE * H::SIZE * W::SIZE)
            .map(|index| index as f32 / 8.0 - 3.0)
            .collect();
        let parameters = HostTensor::<f32, m![{ Symbol<Moments<C>> }, H, W]>::from_vec(
            mean.iter()
                .copied()
                .chain((0..C::SIZE * H::SIZE * W::SIZE).map(|index| -1000.0 - index as f32))
                .collect::<Vec<_>>(),
        );
        let parameters: Input<C, H, W> = parameters.to_hbm(&mut device.pdma).await.unwrap();
        let result = launch(forward, (&mut device, &Mode, (), &mut (), &parameters))
            .await
            .unwrap();
        let result: HostTensor<f32, m![C, H, W]> = result.to_host(&mut device.pdma).await.unwrap();
        let actual = result.into_vec();
        assert_eq!(actual, mean);
    }
}
