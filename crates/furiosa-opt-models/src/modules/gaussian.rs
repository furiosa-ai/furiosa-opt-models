//! A diagonal Gaussian over moments of channels [mean, log-variance]: its mode, or a sample from
//! noise the caller supplies.
use crate::Module;
use crate::ops::gaussian::{Input, Latent};
use furiosa_opt_std::prelude::*;

/// A diagonal Gaussian: its mode without a noise operand, a sample with one.
#[derive(Clone, DeviceSend)]
pub struct Diagonal<Mode, Sample> {
    pub mode: Mode,
    pub sample: Sample,
}

impl<X, Y, Mode, Sample, Env: Copy> Module<(&X, &()), Env, &mut ()> for Diagonal<Mode, Sample>
where
    Mode: for<'x, 's> Module<&'x X, Env, &'s mut (), Output = Y>,
{
    type Output = Y;

    fn forward(&self, device: &mut Device, env: Env, _: &mut (), (parameters, ()): (&X, &())) -> Y {
        self.mode.forward(device, env, &mut (), parameters)
    }
}

impl<X, N, Y, Mode, Sample, Env: Copy> Module<(&X, &(N,)), Env, &mut ()> for Diagonal<Mode, Sample>
where
    Sample: for<'x, 'n, 's> Module<(&'x X, &'n N), Env, &'s mut (), Output = Y>,
{
    type Output = Y;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        (parameters, (noise,)): (&X, &(N,)),
    ) -> Y {
        self.sample
            .forward(device, env, &mut (), (parameters, noise))
    }
}

/// Sample `mean + exp(0.5 * clamp(log_variance, -30, 20)) * noise` in f32.
#[derive(DeviceSend)]
pub struct Sample;

impl<C: AxisName, H: AxisName, W: AxisName, Env>
    Module<(&Input<C, H, W>, &Latent<C, H, W>), Env, &mut ()> for Sample
{
    type Output = Latent<C, H, W>;

    fn forward(
        &self,
        _: &mut Device,
        _: Env,
        _: &mut (),
        _: (&Input<C, H, W>, &Latent<C, H, W>),
    ) -> Self::Output {
        todo!("NPU float32 mean + exp(0.5 * clamp(log_variance, -30, 20)) * supplied noise")
    }
}

impl<C: AxisName, H: AxisName, W: AxisName, Env: Copy>
    Module<&(Input<C, H, W>, Latent<C, H, W>), Env, &mut ()> for Sample
{
    type Output = Latent<C, H, W>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        (parameters, noise): &(Input<C, H, W>, Latent<C, H, W>),
    ) -> Self::Output {
        self.forward(device, env, &mut (), (parameters, noise))
    }
}
