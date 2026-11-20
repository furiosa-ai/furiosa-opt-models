//! Separate four-chip TeaCache row launches over the request's initialized prefix.
//!
//! The caller retains operands, metadata and outputs until all readers finish, including
//! after cancellation or submission failure, and supplies writable copy state.

use super::kernel::Hidden;
use crate::diffusion::minimax_h3::request;
use furiosa_opt_models::Module;
use furiosa_opt_models::models::diffusion::minimax_h3::denoiser::cache::Operands;
use furiosa_opt_std::prelude::*;

pub(super) use furiosa_opt_models::models::diffusion::minimax_h3::denoiser::cache::{
    Add, Copy, Subtract,
};

/// Prime with `Loadable::<(&mut Device, &Add, &request::Inputs<R, K, T, A, V>, &mut (),
/// (&Hidden<R>, &Hidden<R>))>::load(add, device)`. The entry returns `Hidden<R>`;
/// only this request's checked prefix is initialized.
#[device(chip = 4)]
pub(super) fn add<R: AxisName, K: AxisName, T, A, V>(
    device: &mut Device,
    module: &Add,
    inputs: &request::Inputs<R, K, T, A, V>,
    state: &mut (),
    source: Operands<'_, R>,
) -> Hidden<R> {
    module.forward(device, inputs, state, source)
}

/// Prime with `Loadable::<(&mut Device, &Subtract, &request::Inputs<R, K, T, A, V>,
/// &mut (), (&Hidden<R>, &Hidden<R>))>::load(subtract, device)`.
/// The entry returns `Hidden<R>` with only this request's checked prefix initialized.
#[device(chip = 4)]
pub(super) fn subtract<R: AxisName, K: AxisName, T, A, V>(
    device: &mut Device,
    module: &Subtract,
    inputs: &request::Inputs<R, K, T, A, V>,
    state: &mut (),
    source: Operands<'_, R>,
) -> Hidden<R> {
    module.forward(device, inputs, state, source)
}

/// Prime with `Loadable::<(&mut Device, &Copy, &request::Inputs<R, K, T, A, V>,
/// &mut Hidden<R>, &Hidden<R>)>::load(copy, device)`. The entry returns `()`;
/// the writable destination is state, and its suffix remains untouched.
#[device(chip = 4)]
pub(super) fn copy<R: AxisName, K: AxisName, T, A, V>(
    device: &mut Device,
    module: &Copy,
    inputs: &request::Inputs<R, K, T, A, V>,
    output: &mut Hidden<R>,
    source: &Hidden<R>,
) {
    module.forward(device, inputs, output, source)
}
