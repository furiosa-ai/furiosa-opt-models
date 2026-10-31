//! The cut of a sandwich-norm layer at its attention, on a launch's rows as its ordinary leaves
//! take them: each tile of the envelope's rows in the feature slices, the heads as attention reads
//! them, the history in pages of the layer's cache.
//!
//! - Its start runs the input norm and the positioned query/key/value stage, and stores the heads.
//! - Its end runs the output projection, the post-attention norm, the attention residual, the
//!   pre-feed-forward norm and the feed-forward; its update the post-feed-forward norm, the
//!   feed-forward residual and the layer's tail.
//! - Its leaves land nothing ahead, so its front and output projection's landing are `()`.

use super::{Cut, Output, Places, Pool, Staged, Stored, attend_stored};
use crate::Module;
use crate::models::decoder;
use crate::modules::attention::{self, Attention, KeyValue, Positioned};
use crate::modules::residual::Residual;
use crate::modules::transformer::SandwichNorm;
use crate::ops::attention::Heads;
use crate::ops::attention::gqa::{Attended, Pages};
use crate::ops::index::scatter::Store;
use crate::ops::normalization::{RmsNorm, RowMajor, Spread};
use crate::tensor::axis::Single;
use crate::tensor::live::Live;
use crate::tensor::parallel::Chip;
use crate::{Sequential, Then};
use furiosa_opt_std::prelude::*;

/// A call's context of `R` rows: their positions `P`, the slot each writes and the pages of
/// `Windows` each reads.
type Context<P, R, Windows> =
    attention::Context<P, HbmTensor<i32, Chip, m![R]>, Pages<R, Windows, Chip>>;

/// The query heads of group `A` on `R` rows, as attention reads them.
type QueryHeads<A, R> = Heads<
    R,
    <A as decoder::Attention>::Heads,
    <A as decoder::Attention>::Groups,
    <A as decoder::Attention>::Head,
>;

/// The key or value heads of group `A` on `R` rows.
type KeyHeads<A, R> =
    Heads<R, <A as decoder::Attention>::Heads, Single, <A as decoder::Attention>::Head>;

/// What attention gives of group `A` on `R` rows.
type AttendedHeads<A, R> = Attended<
    R,
    <A as decoder::Attention>::Heads,
    <A as decoder::Attention>::Groups,
    <A as decoder::Attention>::Head,
>;

/// The sandwich-norm layer and its tail `L` in every form whose leaves take a whole envelope of
/// `R` rows of `Hidden` features, attending in group `A` over pages of its cache. Its leaves run
/// with the slots `QS` of its query/key/value stage, `PS` of its output projection and `MS` of
/// its feed-forward.
impl<
    'c,
    A,
    P,
    R,
    Windows,
    Hidden,
    const EPSILON: usize,
    NC,
    NP,
    Qkv,
    Position,
    Write,
    Attend,
    Project,
    FeedForward,
    Add,
    Tail,
    QS,
    PS,
    MS,
> Cut<A, &'c Context<P, R, Windows>, (QS, PS, MS)>
    for Then<
        SandwichNorm<
            Attention<Positioned<Qkv, Position>, KeyValue<Write, ()>, Attend, Project>,
            FeedForward,
            RmsNorm<Hidden, EPSILON, NC, NP>,
            Add,
        >,
        Tail,
    >
where
    A: decoder::Attention,
    R: AxisName,
    Windows: AxisName,
    Hidden: AxisName,
    NC: M,
    RmsNorm<Hidden, EPSILON, NC, NP>: for<'r, 'x> Module<
            &'r Live<Spread<Hidden, R, Chip>, R>,
            (),
            &'x mut (),
            Output = Live<Spread<Hidden, R, Chip>, R>,
        >,
    Positioned<Qkv, Position>: for<'r, 'x> Module<
            (&'r Live<Spread<Hidden, R, Chip>, R>, &'c P),
            (),
            &'x mut QS,
            Output = (
                Live<QueryHeads<A, R>, R>,
                Live<KeyHeads<A, R>, R>,
                Live<KeyHeads<A, R>, R>,
            ),
        >,
    Write: for<'d, 'r, 'k> Module<
            (
                &'r mut Pool<A>,
                &'r HbmTensor<i32, Chip, m![R]>,
                (&'k Live<KeyHeads<A, R>, R>, &'k Live<KeyHeads<A, R>, R>),
            ),
            (),
            &'d mut (),
            Output = (),
        >,
    Attend: for<'q, 'b, 'h, 'x> Module<
            (
                &'q Live<QueryHeads<A, R>, R>,
                &'b Pool<A>,
                &'h Pages<R, Windows, Chip>,
            ),
            (),
            &'x mut (),
            Output = Live<AttendedHeads<A, R>, R>,
        >,
    Project: for<'r, 'x> Module<
            &'r Live<AttendedHeads<A, R>, R>,
            (),
            &'x mut PS,
            Output = Live<Spread<Hidden, R, Chip>, R>,
        >,
    Add: for<'r, 's, 'x> Module<
            (
                &'r Live<Spread<Hidden, R, Chip>, R>,
                &'s Live<Spread<Hidden, R, Chip>, R>,
            ),
            (),
            &'x mut (),
            Output = Live<Spread<Hidden, R, Chip>, R>,
        >,
    FeedForward: for<'r, 'x> Module<
            &'r Live<Spread<Hidden, R, Chip>, R>,
            (),
            &'x mut MS,
            Output = Live<Spread<Hidden, R, Chip>, R>,
        >,
    Tail: for<'r, 'x> Module<
            &'r Live<Spread<Hidden, R, Chip>, R>,
            (),
            &'x mut (),
            Output = Live<Spread<Hidden, R, Chip>, R>,
        >,
{
    type Rows = Spread<Hidden, R, Chip>;
    type Envelope = R;
    type Tile = R;
    type Reach = R;
    type Home = HbmTensor<bf16, Chip, RowMajor<Hidden, R>>;
    type Front = ();
    type Landed = ();
    type Staged = Staged<Live<Self::Rows, R>, Live<Self::Rows, R>>;

    fn fill(&self, device: &mut Device, home: &Self::Home, _: Live<(), R>) -> Self::Rows {
        home.to_dm(&mut device.tdma)
    }

    fn spill(&self, device: &mut Device, home: &mut Self::Home, rows: Live<&Self::Rows, R>) {
        let live = rows.with(());
        crate::Op::forward(&Store, device, ((rows.into_value(), live), home));
    }

    fn front(&self, _: &mut Device, _: Live<(), R>) {}

    fn start(
        &self,
        device: &mut Device,
        call: &'c Context<P, R, Windows>,
        (): (),
        rows: Live<Self::Rows, R>,
        (stored, slots): (&mut Stored<A, R>, &mut (QS, PS, MS)),
    ) -> Live<Self::Rows, R> {
        let Then(Sequential(Residual(norm, Then(attention, _), _), _), _) = self;
        let normalized = norm.forward(&mut *device, (), &mut (), &rows);
        let input = (&normalized, &call.positions);
        let (query, key, value) = (attention.0.qkv).forward(&mut *device, (), &mut slots.0, input);
        let Stored { queries, retained } = stored;
        let tdma = &mut device.tdma;
        query.value().view().to_hbm_view(tdma, queries.view_mut());
        key.value().view().to_hbm_view(tdma, retained.0.view_mut());
        value
            .value()
            .view()
            .to_hbm_view(tdma, retained.1.view_mut());
        rows
    }

    fn attend(
        &self,
        device: &mut Device,
        call: &'c Context<P, R, Windows>,
        (pool, live): (&mut &mut Pool<A>, Live<(), R>),
        places: &mut Places<A, R>,
    ) {
        let Then(Sequential(Residual(_, Then(attention, _), _), _), _) = self;
        let Attention { cache, attend, .. } = &attention.0;
        let Places { stored, attended } = places;
        let write = &call.write;
        let operands = (&write, &call.read);
        attend_stored(
            device,
            (cache, operands),
            (pool, &*stored, live),
            |device, pool| {
                let query = live.with(stored.queries.to_dm(&mut device.tdma));
                let rows = attend.forward(&mut *device, (), &mut (), (&query, pool, &call.read));
                (rows.value().view()).to_hbm_view(&mut device.tdma, attended.0.view_mut());
            },
        )
    }

    fn end(
        &self,
        device: &mut Device,
        (attended, ()): Output<'_, A, R, R, ()>,
        input: Live<Self::Rows, R>,
        slots: &mut (QS, PS, MS),
    ) -> Self::Staged {
        let Then(Sequential(first, Residual(norm, Then(mlp, _), _)), _) = self;
        let Residual(_, Then(attention, post), add) = first;
        let heads = attended.with(attended.value().0.to_dm(&mut device.tdma));
        let projected = (attention.0.output).forward(&mut *device, (), &mut slots.1, &heads);
        let branch = post.forward(&mut *device, (), &mut (), &projected);
        let residual = add.forward(&mut *device, (), &mut (), (&input, &branch));
        let normalized = norm.forward(&mut *device, (), &mut (), &residual);
        let ready = mlp.forward(device, (), &mut slots.2, &normalized);
        Staged {
            ready,
            attended: residual,
        }
    }

    fn update(
        &self,
        device: &mut Device,
        Staged { ready, attended }: Self::Staged,
        _: &mut (QS, PS, MS),
    ) -> Live<Self::Rows, R> {
        let Then(Sequential(_, Residual(_, Then(_, post), add)), tail) = self;
        let branch = post.forward(&mut *device, (), &mut (), &ready);
        let rows = add.forward(&mut *device, (), &mut (), (&attended, &branch));
        tail.forward(device, (), &mut (), &rows)
    }
}
