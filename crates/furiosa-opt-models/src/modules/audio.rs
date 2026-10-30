//! Waveform readouts from spectral features and caller-owned history.

use crate::Module;
use furiosa_opt_std::prelude::{Device, DeviceSend};

/// Converts projected spectral features to a waveform using history, then trims to the extent.
#[derive(DeviceSend)]
pub struct Waveform<Projection, Spectrum, Inverse, Trim> {
    pub projection: Projection,
    pub spectrum: Spectrum,
    pub inverse: Inverse,
    pub trim: Trim,
}

/// The extent is the input's beside the features; the state is the inverse transform's history,
/// lent to it beside the spectrum.
impl<'s, X, P, S, W, O, H, T, Projection, Spectrum, Inverse, Trim, Env: Copy>
    Module<(&X, T), Env, &'s mut H> for Waveform<Projection, Spectrum, Inverse, Trim>
where
    Projection: for<'x, 'd> Module<&'x X, Env, &'d mut (), Output = P>,
    Spectrum: for<'x, 'd> Module<&'x P, Env, &'d mut (), Output = S>,
    Inverse: for<'x, 'd> Module<(&'x S, &'s mut H), Env, &'d mut (), Output = W>,
    Trim: for<'d> Module<(W, T), Env, &'d mut (), Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        history: &'s mut H,
        (hidden, extent): (&X, T),
    ) -> O {
        let projected = self.projection.forward(&mut *device, env, &mut (), hidden);
        let spectrum = self
            .spectrum
            .forward(&mut *device, env, &mut (), &projected);
        let waveform = self
            .inverse
            .forward(&mut *device, env, &mut (), (&spectrum, history));
        self.trim.forward(device, env, &mut (), (waveform, extent))
    }
}
