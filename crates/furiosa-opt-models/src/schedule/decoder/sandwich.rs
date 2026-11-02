//! The cut of a sandwich-norm layer at its attention, on a launch's rows as its ordinary leaves
//! take them: each tile of the envelope's rows in the feature slices, the heads as attention reads
//! them, the history in pages of the layer's cache.
//!
//! - Its start runs the input norm and the positioned query/key/value stage, and stores the heads.
//! - Its end runs the output projection, the post-attention norm, the attention residual, the
//!   pre-feed-forward norm and the feed-forward; its update the post-feed-forward norm, the
//!   feed-forward residual and the layer's tail.
//! - Its feed-forward's projections land their weights one step ahead, under work that issues no
//!   transfer: the gate projection's under the post-attention and pre-feed-forward norms, whose
//!   gains land first, the up projection's under the gate's passes, the down projection's under
//!   the activation and the up projection's passes. Its front and output projection's landing are
//!   `()`; its leaves hold nothing in a call, so it takes `()` and runs each with its empty slot.

use super::{Cut, Output, Places, Pool, Staged, Stored, attend_stored};
use crate::models::decoder;
use crate::modules::attention::{self, Attention, KeyValue, Positioned};
use crate::modules::feedforward::glu::Glu;
use crate::modules::parallel::RowParallel;
use crate::modules::residual::{Residual, Sum};
use crate::modules::transformer::SandwichNorm;
use crate::ops::attention::Heads;
use crate::ops::attention::gqa::{Attended, Pages};
use crate::ops::index::scatter::Store;
use crate::ops::normalization::{RmsNorm, RowMajor, Spread};
use crate::tensor::axis::Single;
use crate::tensor::live::Live;
use crate::tensor::parallel::{Chip, Parallelism};
use crate::{Begin, Fork, Module, Sequential, Slot, Then};
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
/// `R` rows of `Hidden` features, attending in group `A` over pages of its cache. Its query/key/
/// value stage and output projection run with their empty slot. Its norms land their gains as
/// `NB`. Its gated feed-forward's gate, up and down projections land as `GL`, `UL`, `DL` and give
/// `GO`, `UO` and the rows; the activation gives `AO` of the gate's, the product `MO`.
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
    NB,
    Qkv,
    Position,
    Write,
    Attend,
    Project,
    Gate,
    Activation,
    Up,
    Multiply,
    Down,
    Placement,
    GL,
    GO,
    AO,
    UL,
    UO,
    MO,
    DL,
    Add,
    Tail,
> Cut<A, &'c Context<P, R, Windows>, ()>
    for Then<
        SandwichNorm<
            Attention<Positioned<Qkv, Position>, KeyValue<Write, ()>, Attend, Project>,
            Glu<Gate, Activation, Up, Multiply, RowParallel<Down, Placement>>,
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
        > + for<'x> Module<Begin<()>, (), &'x mut (), Output = NB>,
    NB: for<'r, 'x> Module<
            &'r Live<Spread<Hidden, R, Chip>, R>,
            (),
            &'x mut (),
            Output = Live<Spread<Hidden, R, Chip>, R>,
        >,
    Positioned<Qkv, Position>: Slot
        + for<'r, 'x> Module<
            (&'r Live<Spread<Hidden, R, Chip>, R>, &'c P),
            (),
            &'x mut <Positioned<Qkv, Position> as Slot>::Empty,
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
    Project: Slot
        + for<'r, 'x> Module<
            &'r Live<AttendedHeads<A, R>, R>,
            (),
            &'x mut <Project as Slot>::Empty,
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
    Placement: Parallelism,
    Gate: for<'x> Module<Begin<()>, (), &'x mut (), Output = GL>
        + for<'r, 'x> Module<(GL, &'r Live<Spread<Hidden, R, Chip>, R>), (), &'x mut (), Output = GO>,
    Activation: for<'r, 'x> Module<&'r GO, (), &'x mut (), Output = AO>,
    Up: for<'x> Module<Begin<()>, (), &'x mut (), Output = UL>
        + for<'r, 'x> Module<(UL, &'r Live<Spread<Hidden, R, Chip>, R>), (), &'x mut (), Output = UO>,
    Multiply: for<'g, 'u, 'x> Module<(&'g AO, &'u UO), (), &'x mut (), Output = MO>,
    Down: for<'x> Module<Begin<()>, (), &'x mut (), Output = DL>,
    RowParallel<Down, Placement>:
        for<'r, 'x> Module<(DL, &'r MO), (), &'x mut (), Output = Live<Spread<Hidden, R, Chip>, R>>,
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
        (stored, _): (&mut Stored<A, R>, &mut ()),
    ) -> Live<Self::Rows, R> {
        let Then(Sequential(Residual(norm, Then(attention, _), _), _), _) = self;
        let normalized = norm.forward(&mut *device, (), &mut (), &rows);
        let input = (&normalized, &call.positions);
        let slot = &mut Default::default();
        let (query, key, value) = (attention.0.qkv).forward(&mut *device, (), slot, input);
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
        _: &mut (),
    ) -> Self::Staged {
        let Then(Sequential(first, Residual(norm, Then(mlp, _), _)), _) = self;
        let Residual(_, Then(attention, post), add) = first;
        let Sum(Fork((Then(gate, activation), up)), Then(multiply, down)) = &mlp.0;
        let heads = attended.with(attended.value().0.to_dm(&mut device.tdma));
        let slot = &mut Default::default();
        let projected = (attention.0.output).forward(&mut *device, (), slot, &heads);
        let post = post.forward(&mut *device, (), &mut (), Begin(()));
        let norm = norm.forward(&mut *device, (), &mut (), Begin(()));
        let gate_landed = gate.forward(&mut *device, (), &mut (), Begin(()));
        let branch = post.forward(&mut *device, (), &mut (), &projected);
        let residual = add.forward(&mut *device, (), &mut (), (&input, &branch));
        let normalized = norm.forward(&mut *device, (), &mut (), &residual);
        let up_landed = up.forward(&mut *device, (), &mut (), Begin(()));
        let gated = gate.forward(&mut *device, (), &mut (), (gate_landed, &normalized));
        let down_landed = (down.0.operation).forward(&mut *device, (), &mut (), Begin(()));
        let activated = activation.forward(&mut *device, (), &mut (), &gated);
        let upped = up.forward(&mut *device, (), &mut (), (up_landed, &normalized));
        let product = multiply.forward(&mut *device, (), &mut (), (&activated, &upped));
        let ready = down.forward(device, (), &mut (), (down_landed, &product));
        Staged {
            ready,
            attended: residual,
        }
    }

    fn update(
        &self,
        device: &mut Device,
        Staged { ready, attended }: Self::Staged,
        _: &mut (),
    ) -> Live<Self::Rows, R> {
        let Then(Sequential(_, Residual(_, Then(_, post), add)), tail) = self;
        let branch = post.forward(&mut *device, (), &mut (), &ready);
        let rows = add.forward(&mut *device, (), &mut (), (&attended, &branch));
        tail.forward(device, (), &mut (), &rows)
    }
}
