//! Separate four-chip TeaCache row launches over the request's initialized prefix.
//!
//! Add and subtract return the existing leaves' BF16 results; only the checked prefix is
//! initialized. Copy writes that same prefix into caller-owned storage, preserving its suffix.
//! The caller retains all operands, request metadata and accepted outputs until their readers
//! finish, including after cancellation or submission failure, and supplies writable copy state.

use super::kernel::Hidden;
use crate::diffusion::minimax_h3::request;
use furiosa_opt_models::ops::{
    index::scatter::Store,
    pointwise::{add::Add as AddRows, subtract::Subtract as SubtractRows},
};
use furiosa_opt_models::{Module, Op};
use furiosa_opt_std::prelude::*;

type Operands<'a, R> = (&'a Hidden<R>, &'a Hidden<R>);

/// Add a retained residual to the current rows without reading allocation suffixes.
#[derive(DeviceSend)]
pub(super) struct Add;

impl<'i, 'e, R: AxisName, K: AxisName, T, A, V>
    Module<Operands<'i, R>, &'e request::Inputs<R, K, T, A, V>, &mut ()> for Add
{
    type Output = Hidden<R>;

    fn forward(
        &self,
        device: &mut Device,
        inputs: &'e request::Inputs<R, K, T, A, V>,
        _: &mut (),
        source: Operands<'i, R>,
    ) -> Self::Output {
        let call = inputs.blocks.read();
        let operands = call.prefix().with(source);
        Op::forward(&AddRows, device, &operands).into_value()
    }
}

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

/// Subtract the original embedded rows from a computed result to form its retained residual.
#[derive(DeviceSend)]
pub(super) struct Subtract;

impl<'i, 'e, R: AxisName, K: AxisName, T, A, V>
    Module<Operands<'i, R>, &'e request::Inputs<R, K, T, A, V>, &mut ()> for Subtract
{
    type Output = Hidden<R>;

    fn forward(
        &self,
        device: &mut Device,
        inputs: &'e request::Inputs<R, K, T, A, V>,
        _: &mut (),
        source: Operands<'i, R>,
    ) -> Self::Output {
        let call = inputs.blocks.read();
        let operands = call.prefix().with(source);
        Op::forward(&SubtractRows, device, &operands).into_value()
    }
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

/// Retain initialized probe or residual rows in the writable storage supplied by the caller.
#[derive(DeviceSend)]
pub(super) struct Copy;

impl<'i, 'e, R: AxisName, K: AxisName, T, A, V>
    Module<&'i Hidden<R>, &'e request::Inputs<R, K, T, A, V>, &mut Hidden<R>> for Copy
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        inputs: &'e request::Inputs<R, K, T, A, V>,
        output: &mut Hidden<R>,
        source: &'i Hidden<R>,
    ) {
        let call = inputs.blocks.read();
        let source = call.prefix().with(source);
        Op::forward(&Store, device, (&source, output));
    }
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
