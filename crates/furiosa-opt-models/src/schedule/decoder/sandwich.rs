//! The cut of a sandwich-norm layer at its attention, on a launch's rows as its ordinary leaves
//! take them: each tile of the envelope's rows in the feature slices, the heads as attention reads
//! them, the history in pages of the layer's cache.
//!
//! - Its front lands the query, key and value projections' weights; its start runs the input norm
//!   and the positioned query/key/value stage over them, and stores the heads.
//! - Its end runs the output projection, the post-attention norm, the attention residual, the
//!   pre-feed-forward norm and the feed-forward up to the down projection begun with its input,
//!   whose landing it issues; its update the down projection, the post-feed-forward norm, the
//!   feed-forward residual and the layer's tail, whose factor it lands first. The next layer's
//!   front lands between them, behind the down projection's input and under its passes.
//! - Its feed-forward's projections land their weights one step ahead, under work that issues no
//!   transfer: the gate projection's under the post-attention and pre-feed-forward norms, whose
//!   gains land first, the up projection's under the gate's passes, the down projection's under
//!   the activation and the up projection's passes. Its output projection's landing is `()`; its
//!   leaves hold nothing in a call, so it takes `()` and runs each with its empty slot.

use super::{Cut, Output, Places, Pool, Staged, Stored, attend_stored};
use crate::models::decoder;
use crate::modules::attention::{self, Attention, KeyValue, NormalizedQkv, Positioned, Projection};
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
use core::marker::PhantomData;
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
/// `R` rows of `Hidden` features, attending in group `A` over pages of its cache. Its query, key
/// and value projections land as `QL`, `KL`, `VL` and give `QO`, `KO` and the value heads, which
/// the head norms leave as `NQ`, `NK`; its output projection runs with its empty slot. Its norms
/// land their gains as `NB`. Its gated feed-forward's gate, up and down projections land as `GL`,
/// `UL`, `DL` and give `GO`, `UO` and the rows; the activation gives `AO` of the gate's, the
/// product `MO`, with which the down projection begins as `DR`. Its tail lands its factor as `TB`.
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
    Query,
    Key,
    Value,
    ValueNorm,
    QueryNorm,
    KeyNorm,
    Position,
    QL,
    KL,
    VL,
    QO,
    KO,
    NQ,
    NK,
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
    DR,
    Add,
    Tail,
    TB,
> Cut<A, &'c Context<P, R, Windows>, ()>
    for Then<
        SandwichNorm<
            Attention<
                Positioned<
                    NormalizedQkv<
                        Projection<Query, Key, Then<Value, ValueNorm>>,
                        QueryNorm,
                        KeyNorm,
                    >,
                    Position,
                >,
                KeyValue<Write, ()>,
                Attend,
                Project,
            >,
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
    Query: for<'x> Module<Begin<()>, (), &'x mut (), Output = QL>
        + for<'r, 'x> Module<(QL, &'r Live<Spread<Hidden, R, Chip>, R>), (), &'x mut (), Output = QO>,
    Key: for<'x> Module<Begin<()>, (), &'x mut (), Output = KL>
        + for<'r, 'x> Module<(KL, &'r Live<Spread<Hidden, R, Chip>, R>), (), &'x mut (), Output = KO>,
    Value: for<'x> Module<Begin<()>, (), &'x mut (), Output = VL>,
    Then<Value, ValueNorm>: for<'r, 'x> Module<
            (VL, &'r Live<Spread<Hidden, R, Chip>, R>),
            (),
            &'x mut ((), ()),
            Output = Live<KeyHeads<A, R>, R>,
        >,
    QueryNorm: for<'r, 'x> Module<&'r QO, (), &'x mut (), Output = NQ>,
    KeyNorm: for<'r, 'x> Module<&'r KO, (), &'x mut (), Output = NK>,
    Position: for<'x> Module<
            ((NQ, NK), &'c P),
            (),
            &'x mut (),
            Output = (Live<QueryHeads<A, R>, R>, Live<KeyHeads<A, R>, R>),
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
    Down: for<'x> Module<Begin<()>, (), &'x mut (), Output = DL>
        + for<'r, 'x> Module<Begin<(DL, &'r MO)>, (), &'x mut (), Output = DR>,
    RowParallel<Down, Placement>:
        for<'x> Module<DR, (), &'x mut (), Output = Live<Spread<Hidden, R, Chip>, R>>,
    Tail: for<'x> Module<Begin<PhantomData<Spread<Hidden, R, Chip>>>, (), &'x mut (), Output = TB>,
    TB: for<'r, 'x> Module<
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
    type Front = (QL, KL, VL);
    type Landed = ();
    type Staged = Staged<DR, Live<Self::Rows, R>>;

    fn fill(&self, device: &mut Device, home: &Self::Home, _: Live<(), R>) -> Self::Rows {
        home.to_dm(&mut device.tdma)
    }

    fn spill(&self, device: &mut Device, home: &mut Self::Home, rows: Live<&Self::Rows, R>) {
        let live = rows.with(());
        crate::Op::forward(&Store, device, ((rows.into_value(), live), home));
    }

    fn front(&self, device: &mut Device, _: Live<(), R>) -> Self::Front {
        let Then(Sequential(Residual(_, Then(attention, _), _), _), _) = self;
        let Positioned { projection, .. } = &attention.0.qkv;
        let Projection {
            query,
            key,
            value: Then(value, _),
        } = &projection.projection;
        (
            query.forward(&mut *device, (), &mut (), Begin(())),
            key.forward(&mut *device, (), &mut (), Begin(())),
            value.forward(device, (), &mut (), Begin(())),
        )
    }

    fn start(
        &self,
        device: &mut Device,
        call: &'c Context<P, R, Windows>,
        (query, key, value): Self::Front,
        rows: Live<Self::Rows, R>,
        (stored, _): (&mut Stored<A, R>, &mut ()),
    ) -> Live<Self::Rows, R> {
        let Then(Sequential(Residual(norm, Then(attention, _), _), _), _) = self;
        let Positioned {
            projection:
                NormalizedQkv {
                    projection: projections,
                    query_norm,
                    key_norm,
                },
            positions,
        } = &attention.0.qkv;
        let normalized = norm.forward(&mut *device, (), &mut (), &rows);
        let input = &normalized;
        let query = (projections.query).forward(&mut *device, (), &mut (), (query, input));
        let key = (projections.key).forward(&mut *device, (), &mut (), (key, input));
        let value = (projections.value).forward(&mut *device, (), &mut ((), ()), (value, input));
        let query = query_norm.forward(&mut *device, (), &mut (), &query);
        let key = key_norm.forward(&mut *device, (), &mut (), &key);
        let positioned = ((query, key), &call.positions);
        let (query, key) = positions.forward(&mut *device, (), &mut (), positioned);
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
        let begun = Begin((down_landed, &product));
        Staged {
            ready: (down.0.operation).forward(&mut *device, (), &mut (), begun),
            attended: residual,
        }
    }

    fn update(
        &self,
        device: &mut Device,
        Staged { ready, attended }: Self::Staged,
        _: &mut (),
    ) -> Live<Self::Rows, R> {
        let Then(Sequential(_, Residual(_, Then(mlp, post), add)), tail) = self;
        let Sum(_, Then(_, down)) = &mlp.0;
        let tail = tail.forward(&mut *device, (), &mut (), Begin(PhantomData::<Self::Rows>));
        let ready = down.forward(&mut *device, (), &mut (), ready);
        let branch = post.forward(&mut *device, (), &mut (), &ready);
        let rows = add.forward(&mut *device, (), &mut (), (&attended, &branch));
        tail.forward(device, (), &mut (), &rows)
    }
}
