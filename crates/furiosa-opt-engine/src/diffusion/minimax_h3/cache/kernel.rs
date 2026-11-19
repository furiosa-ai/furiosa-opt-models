//! Four-chip entry bindings for the model's separate TeaCache probe and statistics stages.

use crate::diffusion::minimax_h3::request;
use furiosa_opt_models::Module;
use furiosa_opt_models::models::diffusion::minimax_h3::denoiser::cache::{NormInput, Operands};
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::DeviceOutput;

pub(super) use furiosa_opt_models::models::diffusion::minimax_h3::denoiser::cache::{
    Hidden, L1, Modulation, Modulations, Norm, NormTables, Totals,
};

/// Keep the generic module boundary while selecting the probe's four-chip topology.
/// The engine's default generic entry is explicitly one-chip and cannot be selected here.
#[device(chip = 4)]
pub(super) fn modulate<M, I, E, S, O>(
    device: &mut Device,
    module: &M,
    env: E,
    state: S,
    input: I,
) -> O
where
    M: DeviceSend + Module<I, E, S, Output = O>,
    I: DeviceSend,
    E: DeviceSend,
    S: DeviceSend,
    O: DeviceOutput,
{
    module.forward(device, env, state, input)
}

/// Prime with `Loadable::<(&mut Device, &Norm<'_, '_>, &request::Inputs<R, K, T, A, V>,
/// &mut Hidden<R>, (&Hidden<R>, &NormTables))>::load(norm, device)`.
/// Pass `&modulations.0.0` as the norm tables; the entry's device output is `()`.
#[device(chip = 4)]
pub(super) fn norm<R: AxisName, K: AxisName, T, A, V>(
    device: &mut Device,
    module: &Norm<'_, '_>,
    inputs: &request::Inputs<R, K, T, A, V>,
    output: &mut Hidden<R>,
    source: NormInput<'_, R>,
) {
    module.forward(device, inputs, output, source)
}

/// Prime with `Loadable::<(&mut Device, &L1, &request::Inputs<R, K, T, A, V>, &mut (),
/// (&Hidden<R>, &Hidden<R>))>::load(l1, device)`. The entry returns `Totals`.
#[device(chip = 4)]
pub(super) fn l1<R: AxisName, K: AxisName, T, A, V>(
    device: &mut Device,
    module: &L1,
    inputs: &request::Inputs<R, K, T, A, V>,
    state: &mut (),
    source: Operands<'_, R>,
) -> Totals {
    module.forward(device, inputs, state, source)
}
