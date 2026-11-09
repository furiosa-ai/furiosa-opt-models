//! The cut of a sandwich-norm layer at its attention, on a launch's rows as its ordinary leaves
//! take them: each tile of the envelope's rows in the feature slices, the heads as attention reads
//! them, the history in pages of the layer's cache.
//!
//! - Its front lands the input norm's gain and the query, key and value projections' weights; its
//!   start runs the input norm and the positioned query/key/value stage over them, and stores the
//!   heads.
//! - Its end runs the output projection, the post-attention norm, the attention residual, the
//!   pre-feed-forward norm and the feed-forward up to the down projection begun with its input,
//!   whose landing it issues; its update the down projection, behind the layer's tail's factor,
//!   and the post-feed-forward norm's gain behind its output; its finish that norm, the
//!   feed-forward residual and the tail. The next layer's front lands between its end and its
//!   update, behind the down projection's input and under its passes, and what the next layer
//!   lands ahead between its update and its finish, behind the down projection's output.
//! - Its feed-forward's projections land their weights ahead, under work that issues no transfer:
//!   the gate projection's codes ahead of its start, behind the down projection's output of the
//!   layer before it and under that layer's finish and its own input norm, into the place the
//!   decoder carries across its attention to its end, and its scales, behind the post-attention
//!   and pre-feed-forward norms' gains, under the output projection's passes, which issue no
//!   transfer once their rows' move is issued, or, with every row group live, under those norms
//!   behind the projection's output, as passes over every row group read data memory at full
//!   rate; the up projection's under those norms and the gate's passes; the down projection's
//!   under the activation and the up projection's passes; its output projection's inside its
//!   attention, after the attention's first window lands or before its last is prepared. Its
//!   leaves hold nothing in a call, so it takes `()`.
//! - Its tile is its envelope, so its attention hands its end the attended heads where they are,
//!   with the landed output projection, and writes no HBM place.

use super::{Cut, Output, Places, Pool, QueryRows, Scales, Staged, Stored, attend_stored};
use crate::load::Named;
use crate::models::decoder;
use crate::modules::attention::{self, Attention, KeyValue, NormalizedQkv, Positioned, Projection};
use crate::modules::feedforward::glu::Glu;
use crate::modules::parallel::RowParallel;
use crate::modules::residual::{Residual, Sum};
use crate::modules::transformer::SandwichNorm;
use crate::ops::attention::gqa::{Attended, Pages};
use crate::ops::attention::{Heads, SplitHeads};
use crate::ops::index::scatter::Store;
use crate::ops::linear::Grouped;
use crate::ops::normalization::{self, RmsNorm, RowMajor, Spread};
use crate::tensor::axis::{Lanes, Single};
use crate::tensor::live::Live;
use crate::tensor::mapping::Rows as Layout;
use crate::tensor::parallel::{Chip, Parallelism};
use crate::{Begin, Fork, Module, Sequential, Then};
use core::marker::PhantomData;
use furiosa_opt_std::prelude::*;

/// A call's context of `R` rows: their positions `P`, the slot each writes and the pages of
/// `Windows` each reads, the tokens past their windows held as `Past`.
type Context<P, R, Windows, Past> =
    attention::Context<P, HbmTensor<i32, Chip, m![R]>, Pages<R, Windows, Chip, Past>>;

/// The query heads of group `A` on `R` rows, as their projection's split leaves them.
type QueryHeads<A, R> = SplitHeads<
    R,
    <A as decoder::Attention>::Heads,
    <A as decoder::Attention>::Groups,
    <A as decoder::Attention>::Head,
>;

/// The key or value heads of group `A` on `R` rows, as their projection's split leaves them.
type KeyHeads<A, R> =
    SplitHeads<R, <A as decoder::Attention>::Heads, Single, <A as decoder::Attention>::Head>;

/// The key or value heads of group `A` on `R` rows, as the cache's write takes them.
type CacheHeads<A, R> =
    Heads<R, <A as decoder::Attention>::Heads, Single, <A as decoder::Attention>::Head>;

/// What attention gives of group `A` on `R` rows.
type AttendedHeads<A, R> = Attended<
    R,
    <A as decoder::Attention>::Heads,
    <A as decoder::Attention>::Groups,
    <A as decoder::Attention>::Head,
>;

/// The sandwich-norm layer and its tail `L` in every form whose leaves take a whole envelope of
/// `R` rows of `Hidden` features, attending in group `A` over pages of its cache that hold the
/// tokens past their windows as `Past`. Its query, key
/// and value projections land as `QL`, `KL`, `VL` and give `QO`, `KO` and the value heads, which
/// the head norms leave as `NQ`, `NK`; its output projection lands as `OL` in its attention. Its
/// norms land their gains as `NB`. Its gated feed-forward's gate, up and down projections land as
/// `GL`, `UL`, `DL` and give `GO`, `UO` and the rows; the activation gives `AO` of the gate's, the
/// product `MO`, with which the down projection begins as `DR`. Its tail lands its factor as `TB`.
impl<
    'c,
    A,
    P,
    R,
    Windows,
    Past,
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
    KB,
    KG,
    QO,
    KO,
    NQ,
    NK,
    Write,
    Attend,
    const OUTPUT: &'static str,
    O,
    OP,
    OL,
    Gate,
    Activation,
    Up,
    Multiply,
    Down,
    Placement,
    GL,
    GS,
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
> Cut<A, &'c Context<P, R, Windows, Past>, ()>
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
                Named<OUTPUT, RowParallel<O, OP>>,
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
        > + Default,
    Query: for<'x> Module<Begin<()>, (), &'x mut (), Output = QL>
        + for<'r, 'x> Module<
            (QL, Begin<Grouped<&'r Live<Spread<Hidden, R, Chip>, R>>>),
            (),
            &'x mut (),
            Output = (QO, KG),
        > + for<'r, 'x> Module<
            (QL, Begin<&'r Live<Spread<Hidden, R, Chip>, R>>),
            (),
            &'x mut (),
            Output = (QO, KB),
        >,
    Key: for<'x> Module<Begin<()>, (), &'x mut (), Output = KL>
        + for<'r, 'g, 'x> Module<
            (KL, (Grouped<&'r Live<Spread<Hidden, R, Chip>, R>>, &'g KG)),
            (),
            &'x mut (),
            Output = KO,
        > + for<'b, 'x> Module<(KL, &'b KB), (), &'x mut (), Output = KO>,
    Value: for<'x> Module<Begin<()>, (), &'x mut (), Output = VL>,
    Then<Value, ValueNorm>: for<'r, 'g, 'x> Module<
            (VL, (Grouped<&'r Live<Spread<Hidden, R, Chip>, R>>, &'g KG)),
            (),
            &'x mut ((), ()),
            Output = Live<KeyHeads<A, R>, R>,
        > + for<'b, 'x> Module<(VL, &'b KB), (), &'x mut ((), ()), Output = Live<KeyHeads<A, R>, R>>,
    QueryNorm: for<'r, 'x> Module<&'r QO, (), &'x mut (), Output = NQ>,
    KeyNorm: for<'r, 'x> Module<&'r KO, (), &'x mut (), Output = NK>,
    Position: for<'x> Module<
            ((NQ, NK), &'c P),
            (),
            &'x mut (),
            Output = (Live<QueryHeads<A, R>, R>, Live<KeyHeads<A, R>, R>),
        >,
    Write: for<'l, 'x> Module<&'l mut Scales<A>, (), &'x mut (), Output = ()>
        + for<'d, 'r, 'o, 's, 'l, 'k> Module<
            (
                &'r mut Pool<A>,
                &'o (&'s HbmTensor<i32, Chip, m![R]>, &'l Scales<A>),
                (&'k Live<CacheHeads<A, R>, R>, &'k Live<CacheHeads<A, R>, R>),
            ),
            (),
            &'d mut (),
            Output = (),
        >,
    Attend: for<'q, 'b, 's, 'h, 'o, 'l, 'x> Module<
            (
                Live<&'q QueryRows<A, R>, R>,
                (&'b Pool<A>, &'s Scales<A>),
                &'h Pages<R, Windows, Chip, Past>,
                (&'o O, &'l mut OL),
            ),
            (),
            &'x mut (),
            Output = Live<AttendedHeads<A, R>, R>,
        >,
    O: for<'x> Module<Begin<()>, (), &'x mut (), Output = OL>
        + for<'l, 'x> Module<&'l mut OL, (), &'x mut (), Output = ()>,
    OL: Default,
    OP: Parallelism,
    Named<OUTPUT, RowParallel<O, OP>>: for<'l, 'r, 'b, 'g, 'x> Module<
            (
                &'l OL,
                &'r Live<AttendedHeads<A, R>, R>,
                (
                    &'b Landings<
                        'b,
                        Gains<
                            'b,
                            Named<"post_attention_layernorm", RmsNorm<Hidden, EPSILON, NC, NP>>,
                            Named<"pre_feedforward_layernorm", RmsNorm<Hidden, EPSILON, NC, NP>>,
                        >,
                        Named<"gate_proj", Gate>,
                    >,
                    &'g mut ((NB, NB), GS),
                ),
            ),
            (),
            &'x mut (),
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
        + for<'x> Module<Begin<GL>, (), &'x mut (), Output = GL>
        + for<'l, 'x> Module<Begin<&'l GL>, (), &'x mut (), Output = GS>
        + for<'l, 'x> Module<&'l mut GS, (), &'x mut (), Output = ()>
        + for<'l, 'r, 'x> Module<
            ((GS, &'l GL), &'r Live<Spread<Hidden, R, Chip>, R>),
            (),
            &'x mut (),
            Output = GO,
        >,
    GL: Default,
    GS: Default,
    Activation: for<'r, 'x> Module<&'r GO, (), &'x mut (), Output = AO>,
    Up: for<'x> Module<Begin<()>, (), &'x mut (), Output = UL>
        + for<'r, 'x> Module<(UL, &'r Live<Spread<Hidden, R, Chip>, R>), (), &'x mut (), Output = UO>,
    Multiply: for<'g, 'u, 'x> Module<(&'g AO, &'u UO), (), &'x mut (), Output = MO>,
    Down: for<'x> Module<Begin<()>, (), &'x mut (), Output = DL>
        + for<'x> Module<Begin<(DL, MO)>, (), &'x mut (), Output = DR>,
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
    type Front = (NB, (QL, KL, VL));
    type Landed = (OL, Live<AttendedHeads<A, R>, R>);
    type Ahead = GL;
    type Staged = Staged<DR, Live<Self::Rows, R>>;
    type Updated = Staged<(Live<Self::Rows, R>, NB, TB), Live<Self::Rows, R>>;

    fn fill(&self, device: &mut Device, home: &Self::Home, tile: Live<(), R>) -> Self::Rows {
        type Feature<W> = Symbol<normalization::Chunk<W>>;
        let mut rows = Self::Rows::new();
        // One transfer, as its setup costs about as much as all its rows: the live rows' lane group,
        // or the envelope when they span groups.
        let (first, count) = tile.groups::<Lanes<R>>();
        if count > 1 {
            home.view().to_dm_view(&mut device.tdma, rows.view_mut());
        } else {
            (home.view())
                .tile::<Layout<R>::Groups, 1, Layout<R>::Elements<m![Hidden]>>(first)
                .to_dm_view(
                    &mut device.tdma,
                    (rows.view_mut()).tile::<Layout<R>::Groups, 1, Layout<R>::GroupMut<
                        m![Hidden % ({ Feature<Hidden> })],
                    >>(first),
                );
        }
        rows
    }

    fn spill(&self, device: &mut Device, home: &mut Self::Home, rows: Live<&Self::Rows, R>) {
        let live = rows.with(());
        crate::Op::forward(&Store, device, ((rows.into_value(), live), home));
    }

    fn front(&self, device: &mut Device, _: Live<(), R>) -> Self::Front {
        let Then(Sequential(Residual(norm, Then(attention, _), _), _), _) = self;
        let Positioned { projection, .. } = &attention.0.qkv;
        let Projection {
            query,
            key,
            value: Then(value, _),
        } = &projection.projection;
        (
            norm.forward(&mut *device, (), &mut (), Begin(())),
            (
                query.forward(&mut *device, (), &mut (), Begin(())),
                key.forward(&mut *device, (), &mut (), Begin(())),
                value.forward(device, (), &mut (), Begin(())),
            ),
        )
    }

    fn ahead(&self, device: &mut Device, gate_landed: GL) -> GL {
        let Then(Sequential(_, Residual(_, Then(mlp, _), _)), _) = self;
        let Sum(Fork((Then(gate, _), _)), _) = &mlp.0;
        gate.forward(device, (), &mut (), Begin(gate_landed))
    }

    fn start(
        &self,
        device: &mut Device,
        call: &'c Context<P, R, Windows, Past>,
        (gain, (query, key, value)): Self::Front,
        rows: Live<Self::Rows, R>,
        (stored, _): (&mut Stored<A, R>, &mut ()),
    ) -> Live<Self::Rows, R> {
        let Then(Sequential(Residual(_, Then(attention, _), _), _), _) = self;
        let Positioned {
            projection:
                NormalizedQkv {
                    projection: projections,
                    query_norm,
                    key_norm,
                },
            positions,
        } = &attention.0.qkv;
        let normalized = gain.forward(&mut *device, (), &mut (), &rows);
        (attention.0.cache.write).forward(&mut *device, (), &mut (), &mut stored.scales);
        let input = &normalized;
        // The query projection keeps the rows it staged for the key and value projections: every
        // row group live, in the tensor registers; else what its form stages for the live groups.
        let projected = |device: &mut Device| {
            if input.groups::<Lanes<R>>().1 == <Layout<R>::Groups as M>::SIZE {
                let begun = (query, Begin(input));
                let (query, banks) = (projections.query).forward(&mut *device, (), &mut (), begun);
                return (
                    query,
                    (projections.key).forward(&mut *device, (), &mut (), (key, &banks)),
                    (projections.value).forward(device, (), &mut ((), ()), (value, &banks)),
                );
            }
            let begun = (query, Begin(Grouped(input)));
            let (query, kept) = (projections.query).forward(&mut *device, (), &mut (), begun);
            (
                query,
                (projections.key).forward(
                    &mut *device,
                    (),
                    &mut (),
                    (key, (Grouped(input), &kept)),
                ),
                (projections.value).forward(
                    device,
                    (),
                    &mut ((), ()),
                    (value, (Grouped(input), &kept)),
                ),
            )
        };
        let (query, key, value) = projected(&mut *device);
        let query = query_norm.forward(&mut *device, (), &mut (), &query);
        let key = key_norm.forward(&mut *device, (), &mut (), &key);
        let positioned = ((query, key), &call.positions);
        let (query, key) = positions.forward(&mut *device, (), &mut (), positioned);
        let Stored {
            queries, retained, ..
        } = stored;
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
        call: &'c Context<P, R, Windows, Past>,
        (pool, live): (&mut &mut Pool<A>, Live<(), R>),
        places: &mut Places<A, R>,
    ) -> Self::Landed {
        let Then(Sequential(Residual(_, Then(attention, _), _), _), _) = self;
        let Attention {
            cache,
            attend,
            output,
            ..
        } = &attention.0;
        let Places { stored, .. } = places;
        let write = (&call.write, &stored.scales);
        let operands = (&write, &call.read);
        attend_stored(
            device,
            (cache, operands),
            (pool, &*stored, live),
            |device, pool| {
                let mut landed = OL::default();
                let boundary = (&output.0.operation, &mut landed);
                let blocks = (pool, &stored.scales);
                let input = (live.with(&stored.queries), blocks, &call.read, boundary);
                let rows = attend.forward(&mut *device, (), &mut (), input);
                (landed, rows)
            },
        )
    }

    fn end(
        &self,
        device: &mut Device,
        ((_, landed), gate_landed): (Output<'_, A, R, R, Self::Landed>, &GL),
        input: Live<Self::Rows, R>,
        _: &mut (),
    ) -> Self::Staged {
        let Then(Sequential(first, Residual(norm, Then(mlp, _), _)), _) = self;
        let Residual(_, Then(attention, post), add) = first;
        let Sum(Fork((Then(gate, activation), up)), Then(multiply, down)) = &mlp.0;
        let (landed, heads) = &*landed;
        let mut begun = <((NB, NB), GS)>::default();
        let boundary = (&Landings(Gains(post, norm), gate), &mut begun);
        let projected =
            (attention.0.output).forward(&mut *device, (), &mut (), (landed, heads, boundary));
        let ((post, norm), mut gate_scales) = begun;
        if projected.groups::<Lanes<R>>().1 == <Layout<R>::Groups as M>::SIZE {
            gate.forward(&mut *device, (), &mut (), &mut gate_scales);
        }
        let up_landed = up.forward(&mut *device, (), &mut (), Begin(()));
        let branch = post.forward(&mut *device, (), &mut (), &projected);
        let residual = add.forward(&mut *device, (), &mut (), (&input, &branch));
        let normalized = norm.forward(&mut *device, (), &mut (), &residual);
        let begun = (gate_scales, gate_landed);
        let gated = gate.forward(&mut *device, (), &mut (), (begun, &normalized));
        let down_landed = (down.0.operation).forward(&mut *device, (), &mut (), Begin(()));
        let activated = activation.forward(&mut *device, (), &mut (), &gated);
        let upped = up.forward(&mut *device, (), &mut (), (up_landed, &normalized));
        let product = multiply.forward(&mut *device, (), &mut (), (&activated, &upped));
        let begun = Begin((down_landed, product));
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
    ) -> Self::Updated {
        let Then(Sequential(_, Residual(_, Then(mlp, post), _)), tail) = self;
        let Sum(_, Then(_, down)) = &mlp.0;
        let tail = tail.forward(&mut *device, (), &mut (), Begin(PhantomData::<Self::Rows>));
        let ready = down.forward(&mut *device, (), &mut (), ready);
        let gain = post.forward(&mut *device, (), &mut (), Begin(()));
        Staged {
            ready: (ready, gain, tail),
            attended,
        }
    }

    fn finish(
        &self,
        device: &mut Device,
        Staged {
            ready: (ready, gain, tail),
            attended,
        }: Self::Updated,
        _: &mut (),
    ) -> Live<Self::Rows, R> {
        let Then(Sequential(_, Residual(_, _, add)), _) = self;
        let branch = gain.forward(&mut *device, (), &mut (), &ready);
        let rows = add.forward(&mut *device, (), &mut (), (&attended, &branch));
        tail.forward(device, (), &mut (), &rows)
    }
}

/// The gains a sandwich-norm layer lands while its output projection's passes run: those of its
/// two norms after the attention, each into its place.
struct Gains<'m, P, N>(&'m P, &'m N);

impl<'m, 'p, 'x, P, N, NB> Module<&'p mut (NB, NB), (), &'x mut ()> for Gains<'m, P, N>
where
    P: for<'s, 'y> Module<&'s mut NB, (), &'y mut (), Output = ()>,
    N: for<'s, 'y> Module<&'s mut NB, (), &'y mut (), Output = ()>,
{
    type Output = ();

    fn forward(&self, device: &mut Device, _: (), _: &'x mut (), (post, norm): &'p mut (NB, NB)) {
        self.0.forward(&mut *device, (), &mut (), post);
        self.1.forward(device, (), &mut (), norm);
    }
}

/// What a sandwich-norm layer lands under its output projection's passes, each into its place: its
/// norms' gains, then its gate projection's scales where fewer than every row group is live.
struct Landings<'m, N, W>(N, &'m W);

/// Every row group live: the gains alone, as passes over every row group read data memory at full
/// rate and the scales' writes would slow them; those land behind the projection's output.
impl<'m, 'p, 'x, N, W, G, S> Module<&'p mut (G, S), (), &'x mut ()> for Landings<'m, N, W>
where
    N: for<'s, 'y> Module<&'s mut G, (), &'y mut (), Output = ()>,
{
    type Output = ();

    fn forward(&self, device: &mut Device, _: (), _: &'x mut (), (gains, _): &'p mut (G, S)) {
        self.0.forward(device, (), &mut (), gains);
    }
}

/// Fewer than every row group live: the gains, then the scales.
impl<'m, 'p, 'x, N, W, G, S> Module<Grouped<&'p mut (G, S)>, (), &'x mut ()> for Landings<'m, N, W>
where
    N: for<'s, 'y> Module<&'s mut G, (), &'y mut (), Output = ()>,
    W: for<'s, 'y> Module<&'s mut S, (), &'y mut (), Output = ()>,
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        _: (),
        _: &'x mut (),
        Grouped((gains, scales)): Grouped<&'p mut (G, S)>,
    ) {
        self.0.forward(&mut *device, (), &mut (), gains);
        self.1.forward(device, (), &mut (), scales);
    }
}
