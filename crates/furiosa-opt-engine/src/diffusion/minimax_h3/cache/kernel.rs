//! Separately launched first-block TeaCache probe stages and per-chip scalar L1 statistics.
//!
//! The caller lends the already loaded first block's modulation and first adaptive RMS norm.
//! Modulation copies all six returned tables into caller-owned local storage before the next
//! launch reads them, preserving the original `step.rs` table relayout boundary. Norm writes
//! only the request's initialized work, including exact-zero guards. L1 reads that same work.
//!
//! Modulation and norm return `()`: destinations travel behind mutable state, rather than
//! through automatic device-output allocation. The caller must allocate distinct chip-local
//! destinations and retain them, every input and request metadata through accepted completion,
//! including submission failure or cancellation. These entries do not manage those lifetimes.

use crate::diffusion::minimax_h3::{kernel::EmbeddedTime, request};
use furiosa_opt_models::models::diffusion::minimax_h3::{
    embeddings::{Chips, Time, Width},
    transformer,
};
use furiosa_opt_models::ops::{
    index::{packed::Table, scatter::Store},
    pointwise::{abs::Abs, subtract::Subtract, sum::Sum},
};
use furiosa_opt_models::tensor::memory::Hbm;
use furiosa_opt_models::{Module, Then};
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::DeviceOutput;

/// The first block's existing modulation projection, borrowed without loading another tree.
/// Obtain `model` from `first.0.modulation.0` of the admitted `denoiser::Block<&Hbm>`.
#[derive(DeviceSend)]
pub(super) struct LocalTables<'a, M> {
    pub(super) model: &'a M,
}

pub(super) type Modulation<'a, 'w> = LocalTables<'a, transformer::Modulation<Time, Width, &'w Hbm>>;

/// A branch's `(shift, scale)` tables, each retaining its original BF16 boundary.
pub(super) type NormTables = (Table<Width, Chips>, Table<Width, Chips>);
/// A branch's `((shift, scale), gate)` tables.
type BranchTables = (NormTables, Table<Width, Chips>);
/// Attention then feed-forward tables, in the original modulation output order.
pub(super) type Modulations = (BranchTables, BranchTables);

impl<'i, M> Module<&'i EmbeddedTime, (), &mut Modulations> for LocalTables<'_, M>
where
    M: for<'s> Module<&'i EmbeddedTime, (), &'s mut (), Output = Modulations>,
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        _: (),
        output: &mut Modulations,
        time: &'i EmbeddedTime,
    ) {
        let (((shift, scale), gate), ((shift_ff, scale_ff), gate_ff)): Modulations =
            self.model.forward(&mut *device, (), &mut (), time);
        let (((shift_out, scale_out), gate_out), ((shift_ff_out, scale_ff_out), gate_ff_out)) =
            output;
        // The projection assembles broadcast tables using all four chips. Store preserves
        // their BF16 values while copying each chip's complete table into its local destination.
        furiosa_opt_models::Op::forward(&Store, &mut *device, (&shift, shift_out));
        furiosa_opt_models::Op::forward(&Store, &mut *device, (&scale, scale_out));
        furiosa_opt_models::Op::forward(&Store, &mut *device, (&gate, gate_out));
        furiosa_opt_models::Op::forward(&Store, &mut *device, (&shift_ff, shift_ff_out));
        furiosa_opt_models::Op::forward(&Store, &mut *device, (&scale_ff, scale_ff_out));
        furiosa_opt_models::Op::forward(&Store, device, (&gate_ff, gate_ff_out));
    }
}

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

/// The same first block's attention norm, borrowed separately from its projection.
/// Obtain `model` from `first.0.norm.0` of the same admitted block used for modulation.
#[derive(DeviceSend)]
pub(super) struct Norm<'a, 'w: 'a> {
    pub(super) model: &'a transformer::AdaptiveNorm<Width, &'w Hbm>,
}

pub(super) type Hidden<R> = HbmTensor<bf16, Chips, m![R, Width]>;
type NormInput<'a, R> = (&'a Hidden<R>, &'a NormTables);

impl<'i, 'e, R: AxisName, K: AxisName, T, A, V>
    Module<NormInput<'i, R>, &'e request::Inputs<R, K, T, A, V>, &mut Hidden<R>> for Norm<'_, '_>
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        inputs: &'e request::Inputs<R, K, T, A, V>,
        output: &mut Hidden<R>,
        (hidden, modulation): NormInput<'i, R>,
    ) {
        let call = inputs.blocks.read();
        self.model.forward(
            device,
            (),
            &mut (),
            (hidden, output, modulation, call.spans()),
        );
    }
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

/// Each chip's rounded-difference L1 distance and the right operand's L1 norm.
/// The caller must bind both operands to this request's initialized prefix before launching.
#[derive(DeviceSend)]
pub(super) struct L1;

/// Only lane zero of each FP32 eight-lane total is defined, independently on each chip.
pub(super) type Totals = (
    furiosa_opt_models::ops::pointwise::sum::Total<Chips>,
    furiosa_opt_models::ops::pointwise::sum::Total<Chips>,
);
type Magnitudes = (Sum<Then<Subtract, Abs>>, Sum<Abs>);
type Operands<'a, R> = (&'a Hidden<R>, &'a Hidden<R>);

impl<'i, 'e, R: AxisName, K: AxisName, T, A, V>
    Module<Operands<'i, R>, &'e request::Inputs<R, K, T, A, V>, &mut ()> for L1
{
    type Output = Totals;

    fn forward(
        &self,
        device: &mut Device,
        inputs: &'e request::Inputs<R, K, T, A, V>,
        _: &mut (),
        (left, right): Operands<'i, R>,
    ) -> Self::Output {
        let call = inputs.blocks.read();
        // Both sums receive exactly the same checked BlockPrefix. Allocation suffixes are
        // unread, and the BF16 Subtract -> Abs boundary stays inside the existing Sum child.
        let distance = call.prefix().with((left, right));
        let norm_rows = call.prefix().with(right);
        let magnitudes: Magnitudes = (Sum(Then(Subtract, Abs)), Sum(Abs));
        magnitudes.forward(device, (), &mut (((), ()), ()), (&distance, &norm_rows))
    }
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
