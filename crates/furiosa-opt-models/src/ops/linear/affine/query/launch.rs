//! A launch's rows as its query, key and value projections read them: its spans stored in HBM,
//! then its periods in order, the pairs of tiles first and a last single tile. The value
//! projection's form begins the pairs, if any, and stages each period its own way.

use super::Tile;
use super::group::TilePair;
use crate::Module;
use crate::modules::generic::Either;
use crate::modules::generic::schedule::{Begin, Spill};
use crate::ops::linear::Affine;
use crate::ops::normalization::span::{Owned, Span};
use crate::tensor::live::Live;
use crate::tensor::parallel::Chip;
use core::marker::PhantomData;
use furiosa_opt_std::prelude::*;

/// The normalized rows a launch's projections land from, in HBM: groups of eight rows, sixteen
/// features a block.
pub(crate) type Normalized<Hidden, All> =
    HbmTensor<bf16, Chip, m![All / 8, Hidden / 16, All % 8, Hidden % 16]>;

/// A launch's live rows `live` and `staging`: what the value projection's form keeps from one
/// period to the next.
pub struct Launch<Hidden: AxisName, F, All: AxisName> {
    live: Live<(), All>,
    staging: F,
    hidden: PhantomData<Hidden>,
}

impl<Hidden: AxisName, F, All: AxisName> Launch<Hidden, F, All> {
    /// The launch's live rows, before any period.
    pub(crate) fn new(live: Live<(), All>, staging: F) -> Self {
        Self {
            live,
            staging,
            hidden: PhantomData,
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
