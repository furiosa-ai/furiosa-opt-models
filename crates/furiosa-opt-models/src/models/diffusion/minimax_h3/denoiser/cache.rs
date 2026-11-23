//! TeaCache probe, statistics and residual compositions over checked H3 rows.
//!
//! Modulation copies its six BF16 tables into caller-owned local storage. Norm and row
//! operations use only the checked initialized prefix, retaining every rounding boundary.
//! Callers own destinations, operands and metadata through completion of their readers.

use super::super::{request, transformer};
use super::launch::EmbeddedTime;
use crate::models::diffusion::minimax_h3::embeddings::{Chips, Time, Width};
use crate::ops::{
    index::{packed::Table, scatter::Store},
    pointwise::{abs::Abs, add::Add as AddRows, subtract::Subtract as SubtractRows, sum::Sum},
};
use crate::tensor::memory::Hbm;
use crate::{Module, Then};
use furiosa_opt_std::prelude::*;

pub use super::launch::Hidden;

/// The first block's existing modulation projection, borrowed without loading another tree.
/// Obtain `model` from `first.0.modulation.0` of the admitted `denoiser::Block<&Hbm>`.
#[derive(DeviceSend)]
pub struct LocalTables<'a, M> {
    /// Borrowed first-block modulation weights.
    pub model: &'a M,
}

pub type Modulation<'a, 'w> = LocalTables<'a, transformer::Modulation<Time, Width, &'w Hbm>>;

/// A branch's `(shift, scale)` tables, stored at the modulation projection's BF16 precision.
pub type NormTables = (Table<Width, Chips>, Table<Width, Chips>);
/// A branch's `((shift, scale), gate)` tables.
type BranchTables = (NormTables, Table<Width, Chips>);
/// Attention then feed-forward tables, matching the modulation projection's output order.
pub type Modulations = (BranchTables, BranchTables);

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
        crate::Op::forward(&Store, &mut *device, (&shift, shift_out));
        crate::Op::forward(&Store, &mut *device, (&scale, scale_out));
        crate::Op::forward(&Store, &mut *device, (&gate, gate_out));
        crate::Op::forward(&Store, &mut *device, (&shift_ff, shift_ff_out));
        crate::Op::forward(&Store, &mut *device, (&scale_ff, scale_ff_out));
        crate::Op::forward(&Store, device, (&gate_ff, gate_ff_out));
    }
}

/// The same first block's attention norm, borrowed separately from its projection.
/// Obtain `model` from `first.0.norm.0` of the same admitted block used for modulation.
#[derive(DeviceSend)]
pub struct Norm<'a, 'w: 'a> {
    /// Borrowed first-block adaptive normalization weights.
    pub model: &'a transformer::AdaptiveNorm<Width, &'w Hbm>,
}

pub type NormInput<'a, R> = (&'a Hidden<R>, &'a NormTables);

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

/// Each chip's rounded-difference L1 distance and the right operand's L1 norm.
/// The caller must bind both operands to this request's initialized prefix before launching.
#[derive(DeviceSend)]
pub struct L1;

/// Only lane zero of each FP32 eight-lane total is defined, independently on each chip.
pub type Totals = (
    crate::ops::pointwise::sum::Total<Chips>,
    crate::ops::pointwise::sum::Total<Chips>,
);
type Magnitudes = (Sum<Then<SubtractRows, Abs>>, Sum<Abs>);
pub type Operands<'a, R> = (&'a Hidden<R>, &'a Hidden<R>);

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
        let magnitudes: Magnitudes = (Sum(Then(SubtractRows, Abs)), Sum(Abs));
        magnitudes.forward(device, (), &mut (((), ()), ()), (&distance, &norm_rows))
    }
}

/// Add a retained residual to the current rows without reading allocation suffixes.
#[derive(DeviceSend)]
pub struct Add;

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
        crate::Op::forward(&AddRows, device, &operands).into_value()
    }
}

/// Subtract the original embedded rows from a computed result to form its retained residual.
#[derive(DeviceSend)]
pub struct Subtract;

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
        crate::Op::forward(&SubtractRows, device, &operands).into_value()
    }
}

/// Retain initialized probe or residual rows in the writable storage supplied by the caller.
#[derive(DeviceSend)]
pub struct Copy;

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
        crate::Op::forward(&Store, device, (&source, output));
    }
}
