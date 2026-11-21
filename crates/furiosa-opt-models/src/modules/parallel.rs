//! Tensor-parallel linear layers: row split followed by an all-reduce, or rows split over the
//! chips gathered before, or exchanged for column parts after, the layer. A column split needs no
//! layer of its own: its weight's mapping holds each chip's columns.

use crate::Module;
use crate::load::Load;
use crate::ops::collective::prefix::{self, Extents, Prefix};
use crate::ops::collective::{AllGather, AllReduce, Exchange, Pieces, Whole};
use crate::tensor::axis::Rows;
use crate::tensor::live::{BlockPrefix, Live};
use crate::tensor::memory::residency::Residency;
use crate::tensor::parallel::{Parallel, Parallelism as Placement, Part, Replicated, Shard};
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

type ChipMap<const N: usize> = m![{ Symbol<Shard<N>> }];
type Split<R, W, const N: usize> =
    HbmTensor<bf16, ChipMap<N>, m![{ Symbol<Pieces<R, N>> }, { Symbol<Part<W, N>> }]>;
type Local<R, W, const N: usize> = HbmTensor<bf16, ChipMap<N>, m![R, W]>;

/// Sums chip-local partial products of the input columns into a replicated output.
#[derive(DeviceSend, Load, Residency)]
#[residency(children(L))]
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
#[derive(DeviceSend, Load, Residency)]
#[residency(children(L))]
pub struct Gathered<L, P: Placement, Strategy = Whole> {
    #[load(rename = "")]
    pub operation: L,
    placement: PhantomData<(P, Strategy)>,
}

/// A composition, not an op.
impl<X, L, P: Placement, Strategy> !crate::Op<&X> for Gathered<L, P, Strategy> {}

impl<L, P: Placement, Strategy> Gathered<L, P, Strategy> {
    pub const fn new(operation: L) -> Self {
        Self {
            operation,
            placement: PhantomData,
        }
    }
}

/// Every chip already holds every row.
impl<'x, X, E, S, L, const N: usize, Strategy> Module<&'x X, E, S>
    for Gathered<L, Replicated<N>, Strategy>
where
    L: Module<&'x X, E, S>,
{
    type Output = L::Output;

    fn forward(&self, device: &mut Device, env: E, state: S, input: &'x X) -> Self::Output {
        self.operation.forward(device, env, state, input)
    }
}

impl<'x, X, Y, O, L, const N: usize, Env: Copy, Strategy> Module<&'x X, Env, &mut ()>
    for Gathered<L, Parallel<N>, Strategy>
where
    Parallel<N>: Placement,
    AllGather<Parallel<N>, Strategy>: for<'a, 's> Module<&'a X, Env, &'s mut (), Output = Y>,
    L: for<'a, 's> Module<&'a Y, Env, &'s mut (), Output = O>,
{
    type Output = O;

    fn forward(&self, device: &mut Device, env: Env, _: &mut (), input: &'x X) -> Self::Output {
        let gathered = AllGather::<Parallel<N>, Strategy>::default().forward(
            &mut *device,
            env,
            &mut (),
            input,
        );
        self.operation.forward(device, env, &mut (), &gathered)
    }
}

/// Exchanges each chip's columns of all the rows for all the columns of the chip's part of the
/// rows, then computes the part's output rows.
#[derive(DeviceSend, Load, Residency)]
#[residency(children(L))]
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

impl<L, const N: usize> Exchanged<L, Parallel<N>>
where
    Parallel<N>: Placement,
{
    /// Borrow this composition for Live compact work within the extent list `G`.
    ///
    /// The input's checked prefix restricts work to positive, 256-row-aligned extents in `G`.
    /// Its source parts use that compact work stride within the allocation capacity.
    /// One exchange kernel selects the DMA mapping; every request uses this same Module.
    ///
    /// The exchange allocates its usual destination, then lends the same checked prefix to the
    /// child. Environment and state pass through unchanged. The view holds no additional weight
    /// or state; the owning composition's loading and residency names stay unchanged.
    pub fn prefix<G: Extents>(&self) -> ExchangedPrefix<'_, L, G, N> {
        ExchangedPrefix {
            operation: &self.operation,
            work: PhantomData,
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

/// A borrowed [`Exchanged`] call for the fixed extent list `G` and runtime compact work.
///
/// Created by [`Exchanged::prefix`]. The checked input's [`BlockPrefix`] describes local work within
/// capacity `R`; its tensor holds `N` compact parts of that work in a `Pieces<R,N>` allocation.
/// The child's returned [`Live`] value cannot borrow the temporary exchange destination.
pub struct ExchangedPrefix<'a, L, G, const N: usize> {
    operation: &'a L,
    work: PhantomData<G>,
}

impl<I, L, G, const N: usize> !crate::Op<I> for ExchangedPrefix<'_, L, G, N> {}

impl<'i, 'p, R: AxisName, W: AxisName, G: Extents, L, E, State, O, const N: usize>
    Module<&'p Prefix<&'i Split<R, W, N>, R, G, Rows<256>>, E, State>
    for ExchangedPrefix<'_, L, G, N>
where
    Parallel<N>: Placement,
    L: for<'x, 'v> Module<
            &'v BlockPrefix<&'x Local<R, W, N>, R, Rows<256>>,
            E,
            State,
            Output = Live<O, R>,
        >,
{
    type Output = Live<O, R>;

    fn forward(
        &self,
        device: &mut Device,
        env: E,
        state: State,
        input: &'p Prefix<&'i Split<R, W, N>, R, G, Rows<256>>,
    ) -> Self::Output {
        const {
            assert!(R::SIZE.is_multiple_of(256));
            assert!(prefix::aligned::<G>(256));
        }
        let mut exchanged = Local::<R, W, N>::new();
        Exchange::<Parallel<N>>::default().write_prefix::<bf16, R, W, G, Rows<256>>(
            device,
            input,
            &mut exchanged,
        );
        self.operation
            .forward(device, env, state, &input.prefix().with(&exchanged))
    }
}
