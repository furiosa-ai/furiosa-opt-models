//! A launch's rows as its query, key and value projections read them: each span stored once in
//! HBM as it arrives, then, once the last is in, its periods in order, the pairs of tiles first and
//! a last single tile. The value projection's form begins the pairs, if any, and stages each
//! period its own way.
//!
//! - Every step compares a counter of its own with a count of the live rows and does nothing
//!   else on its poll, so a run-time count rolls the polls of each step into one loop.

use super::Tile;
use super::group::{Group, TilePair, Tiles};
use crate::Module;
use crate::modules::generic::Either;
use crate::modules::generic::schedule::{Begin, Spill};
use crate::ops::linear::Affine;
use crate::ops::normalization::span::{Owned, Span};
use crate::tensor::live::Live;
use crate::tensor::parallel::Chip;
use core::marker::PhantomData;
use core::task::Poll;
use furiosa_opt_std::prelude::*;

/// The normalized rows a launch's projections land from, in HBM: groups of eight rows, sixteen
/// features a block.
pub(crate) type Normalized<Hidden, All> =
    HbmTensor<bf16, Chip, m![All / 8, Hidden / 16, All % 8, Hidden % 16]>;

/// A launch's live rows `live` and its spans stored in HBM, the spans it took, whether it began
/// its pairs, the pairs and single tiles it gave, and `staging`: what the value projection's form
/// keeps from one period to the next.
pub struct Launch<Hidden: AxisName, F, All: AxisName> {
    normalized: Normalized<Hidden, All>,
    live: Live<(), All>,
    taken: usize,
    begun: usize,
    pairs: usize,
    singles: usize,
    staging: F,
}

impl<Hidden: AxisName, F, All: AxisName> Launch<Hidden, F, All> {
    /// The launch's live rows, before any span arrives.
    pub(crate) fn new(live: Live<(), All>, staging: F) -> Self {
        Self {
            normalized: HbmTensor::new(),
            live,
            taken: 0,
            begun: 0,
            pairs: 0,
            singles: 0,
            staging,
        }
    }

    /// Period `at` of group `G` over the launch's spans that `rows` stores, staged by its form.
    pub(crate) fn period<'l, G>(
        &'l mut self,
        rows: &'l Normalized<Hidden, All>,
        at: usize,
    ) -> Next<'l, G, Hidden, F, All> {
        Next {
            rows,
            staging: &mut self.staging,
            live: self.live,
            at,
            group: PhantomData,
        }
    }
}

/// Period `at` of a launch's pairs of tiles, or its last single tile, with `G` the period's
/// group: the launch's stored rows, its live rows and its form's staging.
pub(crate) struct Next<'l, G, Hidden: AxisName, F, All: AxisName> {
    pub(crate) rows: &'l Normalized<Hidden, All>,
    pub(crate) staging: &'l mut F,
    pub(crate) live: Live<(), All>,
    pub(crate) at: usize,
    pub(crate) group: PhantomData<G>,
}

/// A span of the launch's rows stored where its periods land from.
impl<W, Hidden: AxisName, F, All: AxisName, Env>
    Module<(&mut Launch<Hidden, F, All>, Live<Owned<Hidden, Span>, All>), Env, &mut ()>
    for Affine<W, ()>
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        _: Env,
        _: &mut (),
        (launch, span): (&mut Launch<Hidden, F, All>, Live<Owned<Hidden, Span>, All>),
    ) {
        let rows = launch.live.rows();
        launch
            .normalized
            .spill(device, span.value(), (span.with(()), rows));
        launch.taken += 1;
    }
}

/// A form that keeps nothing between periods begins its pairs with nothing.
impl<W, Hidden: AxisName, All: AxisName, Env>
    Module<Begin<Next<'_, TilePair, Hidden, (), All>>, Env, &mut ()> for Affine<W, ()>
{
    type Output = ();

    fn forward(
        &self,
        _: &mut Device,
        _: Env,
        _: &mut (),
        _: Begin<Next<'_, TilePair, Hidden, (), All>>,
    ) {
    }
}

/// The launch's next step once its last span is in: the form begins its pairs, if it has any,
/// then gives them in order and its last single tile, each staged by the form; `Ready(None)`
/// before and after, when the launch needs its upstream.
impl<W, Hidden: AxisName, F, All: AxisName, P, S, Env: Copy>
    Module<&mut Launch<Hidden, F, All>, Env, &mut ()> for Affine<W, ()>
where
    Self: for<'l, 's> Module<Begin<Next<'l, TilePair, Hidden, F, All>>, Env, &'s mut (), Output = ()>
        + for<'l, 's> Module<Next<'l, TilePair, Hidden, F, All>, Env, &'s mut (), Output = P>
        + for<'l, 's> Module<Next<'l, Tiles<1>, Hidden, F, All>, Env, &'s mut (), Output = S>,
{
    type Output = Poll<Option<Live<Either<P, S>, All>>>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        launch: &mut Launch<Hidden, F, All>,
    ) -> Self::Output {
        type Pair = <TilePair as Group>::Period;
        let live = launch.live;
        if launch.taken < live.count::<Span>() {
            return Poll::Ready(None);
        }
        let paired = live.count::<Tile>() / (Pair::SIZE / Tile::SIZE);
        let (rows, staging) = (&launch.normalized, &mut launch.staging);
        // One step if the launch has pairs, none if not: a count, not a test of `paired`.
        let most = All::SIZE / Pair::SIZE;
        if launch.begun < (paired + most - 1) / most {
            let next = Next::<TilePair, _, _, _> {
                rows,
                staging,
                live,
                at: 0,
                group: PhantomData,
            };
            self.forward(device, env, &mut (), Begin(next));
            launch.begun += 1;
            return Poll::Pending;
        }
        if launch.pairs < paired {
            let at = launch.pairs;
            let next = Next::<TilePair, _, _, _> {
                rows,
                staging,
                live,
                at,
                group: PhantomData,
            };
            let period = self.forward(device, env, &mut (), next);
            launch.pairs += 1;
            return Poll::Ready(Some(live.tile::<Pair>(at).with(Either::Left(period))));
        }
        if launch.singles < live.count::<Tile>() % 2 {
            let next = Next::<Tiles<1>, _, _, _> {
                rows,
                staging,
                live,
                at: paired,
                group: PhantomData,
            };
            let period = self.forward(device, env, &mut (), next);
            launch.singles += 1;
            return Poll::Ready(Some(
                live.tile::<Tile>(2 * paired).with(Either::Right(period)),
            ));
        }
        Poll::Ready(None)
    }
}

/// A projection of a period of a pair of tiles or of a single tile.
impl<'r, W, P, S, OP, OS, Env: Copy> Module<&'r Either<P, S>, Env, &mut ()> for Affine<W, ()>
where
    Self: for<'s> Module<&'r P, Env, &'s mut (), Output = OP>
        + for<'s> Module<&'r S, Env, &'s mut (), Output = OS>,
{
    type Output = Either<OP, OS>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        period: &'r Either<P, S>,
    ) -> Self::Output {
        match period {
            Either::Left(rows) => Either::Left(self.forward(device, env, &mut (), rows)),
            Either::Right(rows) => Either::Right(self.forward(device, env, &mut (), rows)),
        }
    }
}

/// The span rows a norm leaves, by groups of eight rows; the rows of a launch of at most eight
/// rows, or of a span's single wave, move alone.
impl<'r, Hidden: AxisName, Capacity: AxisName>
    Spill<Device, &'r Owned<Hidden, Span>, (Live<(), Capacity>, usize)>
    for Normalized<Hidden, Capacity>
{
    fn spill(
        &mut self,
        context: &mut Device,
        normalized: &'r Owned<Hidden, Span>,
        (live, rows): (Live<(), Capacity>, usize),
    ) {
        const { assert!(Capacity::SIZE.is_multiple_of(Span::SIZE)) };
        let span = self.view_mut().tile::<m![Capacity / 8], { Span::SIZE / 8 },
            m![Capacity / 8 = { Span::SIZE / 8 } #{!} (Capacity / 8), Hidden / 16, Capacity % 8,
                Hidden % 16]>(live.start() / 8);
        if rows <= 8 {
            // SAFETY: the row groups retain their order when split into the two cluster halves.
            let halves = unsafe {
                span.reshape::<Chip, m![Span / (Span / 2) #{!} (Capacity / (Span / 2)),
                    Span % (Span / 2) / 8, Hidden / 16, Span % 8, Hidden % 16]>(
                )
            };
            normalized.view().tile::<m![Span % (Span / 2)], 16,
                m![Span % (Span / 2) = 16 # (Span / 2), Hidden % 16]>(0)
                .to_hbm_view(&mut context.tdma, halves.tile::<m![Span % (Span / 2) / 8], 2,
                    m![Span / (Span / 2) #{!} (Capacity / (Span / 2)),
                        Span % (Span / 2) / 8 = 2 #{!} (Span % (Span / 2) / 8), Hidden / 16,
                        Span % 8, Hidden % 16]>(0));
        } else if live.count::<Tile>() == 1 {
            // SAFETY: the row groups retain their order when split into the two cluster halves.
            let halves = unsafe {
                span.reshape::<Chip, m![Span / (Span / 2) #{!} (Capacity / (Span / 2)),
                    Span % (Span / 2) / 8, Hidden / 16, Span % 8, Hidden % 16]>(
                )
            };
            normalized.view().tile::<m![Span % (Span / 2)], 128,
                m![Span % (Span / 2) = 128 # (Span / 2), Hidden % 16]>(0)
                .to_hbm_view(&mut context.tdma, halves.tile::<m![Span % (Span / 2) / 8], 16,
                    m![Span / (Span / 2) #{!} (Capacity / (Span / 2)),
                        Span % (Span / 2) / 8 = 16 #{!} (Span % (Span / 2) / 8), Hidden / 16,
                        Span % 8, Hidden % 16]>(0));
        } else {
            normalized.view().to_hbm_view(&mut context.tdma, span);
        }
    }
}
