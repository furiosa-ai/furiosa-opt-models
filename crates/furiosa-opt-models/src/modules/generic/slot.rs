//! The state slot a module takes in a call that holds nothing for it, folded over the tree as each
//! module's `Module` impl takes state: a composition's slot holds its parts', an op's is `()`.

use super::{Fork, Sequential, Then};
use crate::load::Named;
use crate::modules::attention::{NormalizedQkv, Positioned, Projection};
use crate::modules::parallel::RowParallel;
use crate::modules::residual::Sum;
use crate::ops::linear::{Affine, fp8::TensorScaled};
use crate::ops::normalization::{RmsNorm, UnitRmsNorm};
use crate::ops::pointwise::{activation::NewGelu, multiply::Multiply};
use crate::tensor::form::BlockScaled;
use crate::tensor::mapping::Reshape;
use crate::tensor::parallel::Parallelism;
use furiosa_opt_std::prelude::*;

/// The slot a module's forward takes when the call holds nothing for it: a launch whose leaves
/// read no state of the call makes their slots from this.
pub trait Slot {
    type Empty: Default;
}

impl Slot for () {
    type Empty = ();
}

impl<const NAME: &'static str, A: Slot> Slot for Named<NAME, A> {
    type Empty = A::Empty;
}

impl<A: Slot, B: Slot> Slot for Sequential<A, B> {
    type Empty = (A::Empty, B::Empty);
}

impl<A: Slot, B: Slot> Slot for Then<A, B> {
    type Empty = (A::Empty, B::Empty);
}

impl<A: Slot, B: Slot> Slot for (A, B) {
    type Empty = (A::Empty, B::Empty);
}

impl<A: Slot, B: Slot, C: Slot> Slot for (A, B, C) {
    type Empty = (A::Empty, B::Empty, C::Empty);
}

/// A fork's parts take its slot.
impl<T: Slot> Slot for Fork<T> {
    type Empty = T::Empty;
}

impl<Parts: Slot, Merge: Slot> Slot for Sum<Parts, Merge> {
    type Empty = (Parts::Empty, Merge::Empty);
}

/// The projection takes the slot; the positions run with none.
impl<P: Slot, Position> Slot for Positioned<P, Position> {
    type Empty = P::Empty;
}

impl<P: Slot, QueryNorm: Slot, KeyNorm: Slot> Slot for NormalizedQkv<P, QueryNorm, KeyNorm> {
    type Empty = (P::Empty, QueryNorm::Empty, KeyNorm::Empty);
}

impl<Query: Slot, Key: Slot, Value: Slot> Slot for Projection<Query, Key, Value> {
    type Empty = (Query::Empty, Key::Empty, Value::Empty);
}

/// A row-parallel layer runs its operation with no slot.
impl<L, P: Parallelism> Slot for RowParallel<L, P> {
    type Empty = ();
}

// # Ops

impl<Matrix, Offset> Slot for Affine<Matrix, Offset> {
    type Empty = ();
}

impl<In: AxisName, Out: AxisName, E: Scalar, S: Scalar, B: M, C, Chips: M, W, T> Slot
    for BlockScaled<In, Out, E, S, B, C, Chips, W, T>
{
    type Empty = ();
}

impl<In: AxisName, Out: AxisName, D: Scalar, Chips: M, W, S> Slot
    for TensorScaled<In, Out, D, Chips, W, S>
{
    type Empty = ();
}

impl<Output> Slot for Reshape<Output> {
    type Empty = ();
}

impl<Width: AxisName, const EPSILON: usize, Chips: M, Parameter> Slot
    for RmsNorm<Width, EPSILON, Chips, Parameter>
{
    type Empty = ();
}

impl<Width: AxisName, const EPSILON: usize, Chips: M, T> Slot
    for UnitRmsNorm<Width, EPSILON, Chips, T>
{
    type Empty = ();
}

impl Slot for NewGelu {
    type Empty = ();
}

impl Slot for Multiply {
    type Empty = ();
}
