//! Tensor-parallel linear layers: row split followed by an all-reduce, or rows split over the
//! chips gathered before, or exchanged for column parts after, the layer. A column split needs no
//! layer of its own: its weight's mapping holds each chip's columns.

use crate::Module;
use crate::load::Load;
use crate::ops::collective::{AllGather, AllReduce, Exchange};
use crate::tensor::parallel::{Parallel, Parallelism as Placement, Replicated};
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

/// Sums chip-local partial products of the input columns into a replicated output.
#[derive(DeviceSend, Load)]
pub struct RowParallel<L, P: Placement = Parallel<1>> {
    #[load(rename = "")]
    pub operation: L,
    placement: PhantomData<P>,
}

/// A composition, not an op.
impl<I, L, P: Placement> !crate::Op<I> for RowParallel<L, P> {}

impl<L, P: Placement> RowParallel<L, P> {
    pub const fn new(operation: L) -> Self {
        Self {
            operation,
            placement: PhantomData,
        }
    }
}

impl<I, O, L, P: Placement, Env: Copy> Module<I, Env, &mut ()> for RowParallel<L, P>
where
    L: for<'s> Module<I, Env, &'s mut (), Output = O>,
    AllReduce<P>: for<'s> Module<O, Env, &'s mut (), Output = O>,
{
    type Output = O;

    fn forward(&self, device: &mut Device, env: Env, _: &mut (), input: I) -> Self::Output {
        let partial = self.operation.forward(&mut *device, env, &mut (), input);
        AllReduce::<P>::default().forward(device, env, &mut (), partial)
    }
}

/// Gathers every chip's part of the input rows onto every chip, then computes chip-local output
/// columns of all the rows.
#[derive(DeviceSend, Load)]
pub struct Gathered<L, P: Placement> {
    #[load(rename = "")]
    pub operation: L,
    placement: PhantomData<P>,
}

/// A composition, not an op.
impl<X, L, P: Placement> !crate::Op<&X> for Gathered<L, P> {}

impl<L, P: Placement> Gathered<L, P> {
    pub const fn new(operation: L) -> Self {
        Self {
            operation,
            placement: PhantomData,
        }
    }
}

/// Every chip already holds every row.
impl<'x, X, E, S, L, const N: usize> Module<&'x X, E, S> for Gathered<L, Replicated<N>>
where
    L: Module<&'x X, E, S>,
{
    type Output = L::Output;

    fn forward(&self, device: &mut Device, env: E, state: S, input: &'x X) -> Self::Output {
        self.operation.forward(device, env, state, input)
    }
}

impl<'x, X, Y, O, L, const N: usize, Env: Copy> Module<&'x X, Env, &mut ()>
    for Gathered<L, Parallel<N>>
where
    Parallel<N>: Placement,
    AllGather<Parallel<N>>: for<'a, 's> Module<&'a X, Env, &'s mut (), Output = Y>,
    L: for<'a, 's> Module<&'a Y, Env, &'s mut (), Output = O>,
{
    type Output = O;

    fn forward(&self, device: &mut Device, env: Env, _: &mut (), input: &'x X) -> Self::Output {
        let gathered =
            AllGather::<Parallel<N>>::default().forward(&mut *device, env, &mut (), input);
        self.operation.forward(device, env, &mut (), &gathered)
    }
}

/// Exchanges each chip's columns of all the rows for all the columns of the chip's part of the
/// rows, then computes the part's output rows.
#[derive(DeviceSend, Load)]
pub struct Exchanged<L, P: Placement> {
    #[load(rename = "")]
    pub operation: L,
    placement: PhantomData<P>,
}

/// A composition, not an op.
impl<X, L, P: Placement> !crate::Op<&X> for Exchanged<L, P> {}

impl<L, P: Placement> Exchanged<L, P> {
    pub const fn new(operation: L) -> Self {
        Self {
            operation,
            placement: PhantomData,
        }
    }
}

/// Every chip already holds every head of every row.
impl<'x, X, E, S, L, const N: usize> Module<&'x X, E, S> for Exchanged<L, Replicated<N>>
where
    L: Module<&'x X, E, S>,
{
    type Output = L::Output;

    fn forward(&self, device: &mut Device, env: E, state: S, input: &'x X) -> Self::Output {
        self.operation.forward(device, env, state, input)
    }
}

impl<'x, X, Y, O, L, const N: usize, Env: Copy> Module<&'x X, Env, &mut ()>
    for Exchanged<L, Parallel<N>>
where
    Parallel<N>: Placement,
    Exchange<Parallel<N>>: for<'a, 's> Module<&'a X, Env, &'s mut (), Output = Y>,
    L: for<'a, 's> Module<&'a Y, Env, &'s mut (), Output = O>,
{
    type Output = O;

    fn forward(&self, device: &mut Device, env: Env, _: &mut (), input: &'x X) -> Self::Output {
        let exchanged =
            Exchange::<Parallel<N>>::default().forward(&mut *device, env, &mut (), input);
        self.operation.forward(device, env, &mut (), &exchanged)
    }
}
