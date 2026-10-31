//! A decoder run by its family's loops over its layers and its row tiles.
//!
//! - Attention is a decoder's only step that reads other tokens' rows: a layer attends once every
//!   row tile of the launch stored its queries, keys and values. Every other op reads only its own
//!   token, so the work from one attention to the next runs tile by tile in one pass.
//! - That work is a [`Tokenwise`] of the end of one part and the start of the next. It takes a
//!   tile of the part's attention output and of the part's input rows, the residual's other term,
//!   and gives the tile of the next part's input rows, having stored the tile's queries, keys and
//!   values for the next attention, its state. A part is a layer after its attention, or the
//!   embedding; the next part a layer up to its attention's store, or the head.
//! - The cut of a layer at its attention is private to the impls here, written once for each layer
//!   shape: the pre-norm layer of Qwen3, Llama and Mistral at every size and in every projection
//!   form, and the sandwich-norm layer of Gemma 4 (`sandwich`). It is generic in the layer's
//!   leaves, each of which takes the cut's steps its own way: the block-scaled output projection
//!   projects a tile over three steps, the dense one in the last of them; Llama's and Mistral's
//!   head norms, `()`, leave the heads as projected. One routine runs every cut's attention: the
//!   rows it reads from the pool stored before it, the rows it defers after it.
//! - [`Decoder`] runs the loops: the first layer's stores over the embedded tiles, then for each
//!   layer its attention and, tile by tile, the work up to the next store or the head; over
//!   interleaved collections in the order's periods (`interleave`). Two layers of an edge may
//!   attend in different groups.
//! - The DMA engine serves transfers in issue order, so weights issue one part ahead of their use
//!   and behind the small transfers of the work they overlap: the first down panel behind the
//!   output projection's rows, the first gate/up pair behind the exchange of its partial sums,
//!   the other down panels behind the gate/up pairs and the down's inputs, the next layer's query
//!   projection behind them and its key and value projections under its input norm. An
//!   attention's working set fills most of data memory, so only the output projection lands
//!   during one, once its last history is in flight. A form whose leaves land nothing ahead takes
//!   the same steps, doing nothing in them.

use crate::Begin;
use crate::Module;
use crate::load::Named;
use crate::models::decoder::attention::qwen3::{self, LmHead, QueryHeads, fp8};
use crate::models::decoder::attention::{
    ForCausalLM, ForSequenceClassification, Geometry, Head, KvHeads, Qwen2Layer,
};
use crate::models::decoder::form::BlockScaled;
use crate::models::decoder::launch;
use crate::models::decoder::{self, Form};
use crate::modules::attention::KeyValue;
use crate::modules::attention::{Attention, NormalizedQkv, Positioned, Projection};
use crate::modules::generic::collection::View;
use crate::modules::language::{Backbone, Call, Classifier, Selected};
use crate::modules::lora::States;
use crate::modules::residual::Residual;
use crate::ops::attention::full::paged::{self, Retained};
use crate::ops::embedding;
use crate::ops::index::{gather::Gather, scatter::Scatter};
use crate::ops::linear::query::group::{Group, TilePair, Tiles};
use crate::ops::linear::query::launch::{Launch, Next, Normalized};
use crate::ops::linear::query::{self, Tile};
use crate::ops::normalization::span::{Owned, Span, Stage, Tiled};
use crate::ops::normalization::{Blocked, RmsNorm};
use crate::ops::pointwise::{SwiGlu, add::Add};
use crate::ops::rotary::group::{self, ToDm};
use crate::ops::rotary::{Rotary, Row, SplitHalf};
use crate::ops::swiglu::{self, Mlp};
use crate::specialize::attention::Decode;
use crate::tensor::axis;
use crate::tensor::cache::Blocks;
use crate::tensor::live::{Fill, Live, Spill};
use crate::tensor::memory::Hbm;
use crate::tensor::parallel::Chip;
use crate::{Sequential, Then};
use core::marker::PhantomData;
use furiosa_opt_std::prelude::*;

mod interleave;
mod sandwich;

/// The work between two attentions on one row tile: the end of `A`, then the start of `B`.
pub struct Tokenwise<A, B>(pub A, pub B);

/// A decoder run by its family's loops over its layers and its row tiles.
pub struct Decoder<M>(pub M);

/// The catalog's layer of the size `Z` whose projections hold form `P`: a [`Layer`] of the
/// form's leaves.
type Catalog<P, Z> = qwen3::Layer<Z, P, bf16, Hbm>;
/// A pre-norm layer of the size `Z` over the leaves its form chooses: the query, key and value
/// projections `Q`, `K`, `V`, their head norms `QN`, `KN`, the attention `A`, the output projection
/// `O` and the MLP's gate/up and down projections `G`, `D`; the catalog's layer in every form.
type Layer<Z, Q, K, V, QN, KN, A, O, G, D> = Qwen2Layer<
    NormalizedQkv<Projection<Q, K, V>, QN, KN>,
    Rotary<Head>,
    Scatter,
    Gather,
    A,
    O,
    Mlp<Width<Z>, Intermediate<Z>, G, D>,
    Norm<Z>,
    Add,
>;
type Capacity<Z> = launch::Capacity<Z>;
type Reach<Z> = <Z as Decode>::Reach;
type Width<Z> = <Z as Geometry>::Hidden;
type Intermediate<Z> = <Z as Geometry>::Intermediate;
type Groups<Z> = <Z as decoder::Attention>::Groups;
/// A launch's rows of the hidden width in HBM: the embedded rows, a layer's input, the head's rows.
type Rows<Z> = HbmTensor<bf16, Chip, Blocked<Width<Z>, Capacity<Z>>>;
/// A launch's `Reach` rows of each of group `A`'s key/value heads and query groups: its queries,
/// or what its attention gives.
type QueryRows<A, Reach> = HbmTensor<
    bf16,
    Chip,
    m![
        { Symbol<Reach> },
        { Symbol<<A as decoder::Attention>::Heads> },
        { Symbol<<A as decoder::Attention>::Groups> },
        { Symbol<<A as decoder::Attention>::Head> }
    ],
>;
/// A launch's `Reach` rows of each of group `A`'s key/value heads: its keys or its values.
type KeyRows<A, Reach> = HbmTensor<
    bf16,
    Chip,
    m![
        { Symbol<Reach> },
        { Symbol<<A as decoder::Attention>::Heads> },
        { Symbol<<A as decoder::Attention>::Head> }
    ],
>;
/// A cache of group `A`'s key/value heads.
type Pool<A> = Blocks<<A as decoder::Attention>::Heads, <A as decoder::Attention>::Head>;
type Norm<Z> = <Z as Geometry>::Norm<Width<Z>, Hbm>;
/// A launch's call as its layers read it: the context its rotary table prepares and its token IDs.
type Launched<'c, 'a, Z> = Call<
    'c,
    launch::Context<
        'a,
        Capacity<Z>,
        Reach<Z>,
        <Z as Decode>::Sequences,
        <Z as Decode>::Slots,
        Groups<Z>,
        launch::Waves<Z>,
        <Z as Decode>::Windows,
    >,
    launch::Ids<Z>,
>;
/// A launch's attention work as its layers read it.
type Work<'a, Z> = paged::Context<
    'a,
    <Z as Decode>::Sequences,
    KvHeads,
    Groups<Z>,
    Head,
    launch::Waves<Z>,
    <Z as Decode>::Windows,
    Capacity<Z>,
>;
/// A row tile of a launch, as the work between two attentions passes it on.
type RowTile<Z> = Live<Owned<Width<Z>, Span>, Capacity<Z>>;
/// A launch's period of group `G`, a pair of tiles or its last single tile, as the value
/// projection's form stages it in `F`.
type Period<'n, G, Z, F> = Next<'n, G, Width<Z>, F, Capacity<Z>>;
/// A period's queries of `G` tiles, as every form's head norm leaves them for the rotation.
type Queries<G, Z> = query::Queries<G, QueryHeads<Z>, Head>;
/// A period's keys of `G` tiles, as their head norm leaves them, or its values, as projected.
type Pairs<G> = query::Pairs<G, KvHeads, Head>;
/// A launch's rotation: each slot's coefficients.
type Rotation<Z> = SplitHalf<Head, group::Rotation<<Z as Decode>::Slots, Row<Head>>>;
/// The landing of a period's rotation coefficients into the slices of its queries, for periods of
/// `G` tiles, with the rotation and the period index it lands them from.
type Landing<'r, G, Z> = (ToDm<G, QueryHeads<Z>>, (&'r Rotation<Z>, usize));
/// A span tile of the attended rows as the output projection's steps take it, with what its form
/// landed.
type Projecting<'o, 'r, Z, L> = (&'o mut L, Live<&'r QueryRows<Z, Reach<Z>>, Capacity<Z>>);

/// What a layer of group `A` stores of a launch's `Reach` rows: their queries, keys and values.
pub struct Stored<A: decoder::Attention, Reach: AxisName> {
    queries: QueryRows<A, Reach>,
    retained: (KeyRows<A, Reach>, KeyRows<A, Reach>),
}

/// The rows a layer of group `A` attends of a launch's `Reach` rows.
pub struct Attended<A: decoder::Attention, Reach: AxisName>(QueryRows<A, Reach>);

/// A tile of a layer's attention output, the tile of `E` rows of what it attended and the output
/// projection over them, `L` as its form lands it.
type Output<'p, A, Reach, E, L> = (Live<&'p Attended<A, Reach>, E>, &'p mut L);

/// A launch's places of the layers attending in group `A`: what one stores and attends, which the
/// group's next layer reuses once that attention read them.
pub struct Places<A: decoder::Attention, Reach: AxisName> {
    stored: Stored<A, Reach>,
    attended: Attended<A, Reach>,
}

impl<A: decoder::Attention, Reach: AxisName> Places<A, Reach> {
    pub fn new() -> Self {
        Self {
            stored: Stored {
                queries: HbmTensor::new(),
                retained: (HbmTensor::new(), HbmTensor::new()),
            },
            attended: Attended(HbmTensor::new()),
        }
    }
}

impl<A: decoder::Attention, Reach: AxisName> Default for Places<A, Reach> {
    fn default() -> Self {
        Self::new()
    }
}

// # The cut of a layer at its attention

/// A layer cut at its attention, which it attends in group `A` over what a call lends it, `C`:
/// the steps of the work between two attentions that run its leaves over a tile of rows, its
/// leaves running with `T`, and its attention once every tile stored its rows. The work's order is
/// [`Tokenwise`]'s; a layer shape gives the steps.
trait Cut<A: decoder::Attention, C, T> {
    /// A tile's rows of the hidden width.
    type Rows;
    /// A launch's rows: an envelope, a runtime count of them live.
    type Envelope: AxisName;
    /// The rows of the envelope one step of the work takes at once.
    type Tile: AxisName;
    /// The rows of a launch its attention stores and attends.
    type Reach: AxisName;
    /// A launch's rows of the hidden width in HBM, where a tile of them waits for an attention.
    type Home;
    /// Its front landed for a tile's rows.
    type Front;
    /// Its output projection as its form lands it at its attention's boundary.
    type Landed;
    /// A tile between its MLP's two steps.
    type Staged;

    /// The rows of `tile` from `home`.
    fn fill(
        &self,
        device: &mut Device,
        home: &Self::Home,
        tile: Live<(), Self::Envelope>,
    ) -> Self::Rows;

    /// A tile's `rows` into `home`.
    fn spill(
        &self,
        device: &mut Device,
        home: &mut Self::Home,
        rows: Live<&Self::Rows, Self::Envelope>,
    );

    /// Its front landed for the rows of `tile`.
    fn front(&self, device: &mut Device, tile: Live<(), Self::Envelope>) -> Self::Front;

    /// A tile's input `rows` through it up to its attention's store, its `front` landed: the input
    /// norm, then the tile's queries, keys and values, head-normed, rotated and stored in
    /// `stored`; the rows, for the residual.
    fn start(
        &self,
        device: &mut Device,
        call: C,
        front: Self::Front,
        rows: Live<Self::Rows, Self::Envelope>,
        state: (&mut Stored<A, Self::Reach>, &mut T),
    ) -> Live<Self::Rows, Self::Envelope>;

    /// Its attention over the `live` rows its `places` stored, into what they attend, over its
    /// `pool`; its output projection landed at the attention's boundary.
    fn attend(
        &self,
        device: &mut Device,
        call: C,
        pool: (&mut &mut Pool<A>, Live<(), Self::Envelope>),
        places: &mut Places<A, Self::Reach>,
    ) -> Self::Landed;

    /// A tile through it from its attention to its MLP's second step, given a tile `output` of its
    /// attention output and `input` of its input rows: the output projection, the attention
    /// residual and the MLP's first step on its norm.
    fn end(
        &self,
        device: &mut Device,
        output: Output<'_, A, Self::Reach, Self::Envelope, Self::Landed>,
        input: Live<Self::Rows, Self::Envelope>,
        adapters: &mut T,
    ) -> Self::Staged;

    /// A `staged` tile through its MLP's second step and the MLP residual; the rows.
    fn update(
        &self,
        device: &mut Device,
        staged: Self::Staged,
        adapters: &mut T,
    ) -> Live<Self::Rows, Self::Envelope>;
}

/// A layer's attention once every tile of a launch `stored` its rows' keys and values: its
/// cache `stage` stores the `live` rows `attend` reads from the `pool`, `attend` attends, then the
/// stage stores the rows it deferred, which attention read where the launch retained them.
fn attend_stored<W, R, WI, RI, A, Reach, E, D, O>(
    device: &mut Device,
    (stage, operands): (&KeyValue<W, R>, (&WI, &RI)),
    (pool, stored, live): (&mut &mut Pool<A>, &Stored<A, Reach>, Live<(), E>),
    attend: impl FnOnce(&mut Device, &Pool<A>) -> O,
) -> O
where
    A: decoder::Attention,
    Reach: AxisName,
    E: AxisName,
    KeyValue<W, R>: for<'p, 'w, 's, 'b> Module<
            (
                &'p (KeyRows<A, Reach>, KeyRows<A, Reach>),
                (&'w WI, &'w RI),
                Live<(), E>,
            ),
            (),
            &'s mut &'b mut Pool<A>,
            Output = D,
        > + for<'p, 'w, 's, 'b> Module<
            (D, &'p (KeyRows<A, Reach>, KeyRows<A, Reach>), &'w WI),
            (),
            &'s mut &'b mut Pool<A>,
            Output = (),
        >,
{
    let retained = &stored.retained;
    let deferred = stage.forward(&mut *device, (), &mut *pool, (retained, operands, live));
    let output = attend(&mut *device, &**pool);
    stage.forward(device, (), pool, (deferred, retained, operands.0));
    output
}

/// The forms a layer's input norm, query projection and head norms take for a tile's rows,
/// landed: the norm's gain, the query form and the two head norms' forms.
struct Front<G, Q, QN, KN> {
    gain: G,
    query: Q,
    query_norm: QN,
    key_norm: KN,
}

/// A tile between its MLP's two steps: the down projection's inputs `R` landed, and the rows of the
/// attention residual, which the MLP residual adds to.
struct Staged<R, X> {
    ready: R,
    attended: X,
}

/// The pre-norm layer, in every form whose leaves take each step. The value projection's form
/// `VB`, begun, begins the launch whose periods it stages in `F` and gives as `PP` (pairs of tiles)
/// and `PS` (single tiles); the key and query forms `KB`, `QB` give a period's heads `KHP`, `QHP`
/// (`KHS`, `QHS`), which the head norms' forms `KNB`, `QNB` leave as the rotation reads them. The
/// output projection lands as `OL`; the MLP's gate/up form `GB` gives `GO` and its gating `SO`,
/// whose down projection lands a weight panel as `DP`, stages the product as `DS`, lands its
/// inputs as `DL` and, with its last panel, as `DR`.
impl<
    'c,
    'a,
    Z,
    H,
    const E: usize,
    const C: usize,
    const L: usize,
    Q,
    K,
    V,
    QN,
    KN,
    A,
    O,
    G,
    D,
    VB,
    F,
    KB,
    QB,
    QNB,
    KNB,
    PP,
    PS,
    QHP,
    QHS,
    KHP,
    KHS,
    OL,
    GB,
    GO,
    SO,
    DP,
    DS,
    DL,
    DR,
    T,
> Cut<Z, Launched<'c, 'a, Z>, T> for Layer<Z, Q, K, V, QN, KN, A, O, G, D>
where
    T: States,
    Rows<Z>: Fill<Device, Live<(), Capacity<Z>>, Rows = Owned<Width<Z>, Span>>
        + for<'r> Spill<Device, &'r Owned<Width<Z>, Span>, Live<(), Capacity<Z>>>,
    Z: Geometry<Hidden = H, Norm<H, Hbm> = RmsNorm<H, E, Chip>>
        + decoder::Attention<Heads = KvHeads, Head = Head>
        + Decode
        + 'static,
    H: Tiled<Capacity = axis::Rows<C>, Last = axis::Rows<L>>,
    swiglu::Down<Intermediate<Z>>: AxisName,
    V: for<'x> Module<Begin<()>, Live<(), Capacity<Z>>, &'x mut (), Output = VB>,
    VB: for<'x> Module<
            Begin<()>,
            Live<(), Capacity<Z>>,
            &'x mut (),
            Output = Launch<Width<Z>, F, Capacity<Z>>,
        > + for<'n, 'x> Module<
            Begin<Period<'n, TilePair, Z, F>>,
            Live<(), Capacity<Z>>,
            &'x mut (),
            Output = (),
        > + for<'n, 'x> Module<
            Period<'n, TilePair, Z, F>,
            Live<(), Capacity<Z>>,
            &'x mut (),
            Output = PP,
        > + for<'n, 'x> Module<
            Period<'n, Tiles<1>, Z, F>,
            Live<(), Capacity<Z>>,
            &'x mut (),
            Output = PS,
        > + for<'r, 'x> Module<
            &'r PP,
            Live<(), Capacity<Z>>,
            &'x mut T::Value,
            Output = Pairs<TilePair>,
        > + for<'r, 'x> Module<
            &'r PS,
            Live<(), Capacity<Z>>,
            &'x mut T::Value,
            Output = Pairs<Tiles<1>>,
        >,
    K: for<'x> Module<Begin<()>, Live<(), Capacity<Z>>, &'x mut (), Output = KB>,
    KB: for<'r, 'x> Module<&'r PP, Live<(), Capacity<Z>>, &'x mut T::Key, Output = KHP>
        + for<'r, 'x> Module<&'r PS, Live<(), Capacity<Z>>, &'x mut T::Key, Output = KHS>,
    Q: for<'x> Module<Begin<()>, Live<(), Capacity<Z>>, &'x mut (), Output = QB>,
    QB: for<'r, 'x> Module<&'r PP, Live<(), Capacity<Z>>, &'x mut T::Query, Output = QHP>
        + for<'r, 'x> Module<&'r PS, Live<(), Capacity<Z>>, &'x mut T::Query, Output = QHS>,
    QN: for<'x> Module<Begin<()>, Live<(), Capacity<Z>>, &'x mut (), Output = QNB>,
    QNB: for<'x> Module<QHP, Live<(), Capacity<Z>>, &'x mut (), Output = Queries<TilePair, Z>>
        + for<'x> Module<QHS, Live<(), Capacity<Z>>, &'x mut (), Output = Queries<Tiles<1>, Z>>,
    KN: for<'x> Module<Begin<()>, Live<(), Capacity<Z>>, &'x mut (), Output = KNB>,
    KNB: for<'x> Module<KHP, Live<(), Capacity<Z>>, &'x mut (), Output = Pairs<TilePair>>
        + for<'x> Module<KHS, Live<(), Capacity<Z>>, &'x mut (), Output = Pairs<Tiles<1>>>,
    O: for<'x> Module<Begin<()>, (), &'x mut (), Output = OL>
        + for<'o, 'r, 'x> Module<
            Begin<Begin<Projecting<'o, 'r, Z, OL>>>,
            (),
            &'x mut (),
            Output = Begin<Projecting<'o, 'r, Z, OL>>,
        > + for<'o, 'r, 'x> Module<
            Begin<Projecting<'o, 'r, Z, OL>>,
            (),
            &'x mut (),
            Output = Projecting<'o, 'r, Z, OL>,
        > + for<'o, 'r, 'x> Module<
            Projecting<'o, 'r, Z, OL>,
            (),
            &'x mut T::Output,
            Output = Owned<Width<Z>, Span>,
        > + for<'g, 'x> Module<&'g mut OL, (), &'x mut (), Output = ()>,
    A: for<'q, 'h, 'w, 'b, 'o, 'g, 't, 'x> Module<
            (
                &'q QueryRows<Z, Reach<Z>>,
                &'h Retained<'h, Reach<Z>, KvHeads, Head>,
                &'w Work<'a, Z>,
                (&'b O, &'g mut OL),
                &'t mut QueryRows<Z, Reach<Z>>,
            ),
            (),
            &'x mut (),
            Output = (),
        >,
    G: for<'x> Module<Begin<()>, (), &'x mut (), Output = GB>
        + for<'r, 'x> Module<
            (GB, Live<&'r Owned<Width<Z>, Span>, Capacity<Z>>),
            (),
            &'x mut T::GateUp,
            Output = GO,
        >,
    SwiGlu: for<'x> Module<Live<GO, Capacity<Z>>, (), &'x mut (), Output = SO>,
    D: for<'x> Module<Begin<usize>, (), &'x mut (), Output = DP>
        + for<'x> Module<Begin<(Live<SO, Capacity<Z>>, DP)>, (), &'x mut (), Output = DS>
        + for<'x> Module<Live<(DS, DP), Capacity<Z>>, (), &'x mut (), Output = DL>
        + for<'x> Module<(DL, DP), (), &'x mut (), Output = DR>
        + for<'x> Module<Live<DR, Capacity<Z>>, (), &'x mut T::Down, Output = Owned<Width<Z>, Span>>,
{
    type Rows = Owned<Width<Z>, Span>;
    type Envelope = Capacity<Z>;
    type Tile = Span;
    type Reach = Reach<Z>;
    type Home = Rows<Z>;
    type Front = Front<<Stage as crate::Op<&'static Norm<Z>>>::Output, QB, QNB, KNB>;
    type Landed = OL;
    type Staged = Staged<DR, RowTile<Z>>;

    fn fill(
        &self,
        device: &mut Device,
        home: &Rows<Z>,
        tile: Live<(), Capacity<Z>>,
    ) -> Owned<Width<Z>, Span> {
        home.fill(device, tile)
    }

    fn spill(
        &self,
        device: &mut Device,
        home: &mut Rows<Z>,
        rows: Live<&Owned<Width<Z>, Span>, Capacity<Z>>,
    ) {
        home.spill(device, rows.value(), rows.with(()));
    }

    fn attend(
        &self,
        device: &mut Device,
        call: Launched<'c, 'a, Z>,
        (pool, live): (&mut &mut Pool<Z>, Live<(), Capacity<Z>>),
        places: &mut Places<Z, Reach<Z>>,
    ) -> OL {
        let Sequential(Residual(_, attention, _), _) = self;
        let Attention {
            cache: stage,
            attend,
            output,
            ..
        } = &attention.0;
        let context = &call.context.0;
        let Places { stored, attended } = places;
        let Stored { queries, retained } = &*stored;
        let operands = (&context.write, &context.read);
        attend_stored(
            device,
            (stage, operands),
            (pool, &*stored, live),
            |device, pool| {
                let history = Retained {
                    pool,
                    key: &retained.0,
                    value: &retained.1,
                    read: &stage.read,
                };
                // The output projection's weights land while the attention's last history is in flight.
                let mut projecting = output.forward(&mut *device, (), &mut (), Begin(()));
                let boundary = (output, &mut projecting);
                let rows = (queries, &history, &context.read, boundary, &mut attended.0);
                attend.forward(device, (), &mut (), rows);
                projecting
            },
        )
    }

    fn front(&self, device: &mut Device, tile: Live<(), Capacity<Z>>) -> Self::Front {
        let Sequential(Residual(norm, attention, _), _) = self;
        let Attention {
            qkv:
                Positioned {
                    projection:
                        NormalizedQkv {
                            projection: Projection { query, .. },
                            query_norm,
                            key_norm,
                        },
                    ..
                },
            ..
        } = &attention.0;
        let query = query.forward(&mut *device, tile, &mut (), Begin(()));
        let query_norm = query_norm.forward(&mut *device, tile, &mut (), Begin(()));
        let key_norm = key_norm.forward(&mut *device, tile, &mut (), Begin(()));
        let gain = crate::Op::forward(&Stage, &mut *device, &norm.0);
        Front {
            gain,
            query,
            query_norm,
            key_norm,
        }
    }

    fn start(
        &self,
        device: &mut Device,
        call: Launched<'c, 'a, Z>,
        front: Self::Front,
        rows: RowTile<Z>,
        (stored, adapters): (&mut Stored<Z, Reach<Z>>, &mut T),
    ) -> RowTile<Z> {
        let Sequential(Residual(_, attention, _), _) = self;
        let Attention {
            qkv:
                Positioned {
                    projection:
                        NormalizedQkv {
                            projection: Projection { key, value, .. },
                            ..
                        },
                    positions,
                },
            ..
        } = &attention.0;
        let tile = rows.with(());
        // The periods read the key and value projections after the queries', so they land here,
        // under the input norm, behind the exchange that ends the down projection.
        let key = key.forward(&mut *device, tile, &mut (), Begin(()));
        let value = value.forward(&mut *device, tile, &mut (), Begin(()));
        let mut launch = value.forward(&mut *device, tile, &mut (), Begin(()));
        let normalized = front.gain.forward(&mut *device, (), &mut (), &rows);
        // The rows the periods read lie apart from the launch, which the period loops carry.
        let mut home = Normalized::new();
        home.spill(&mut *device, normalized.value(), (tile, tile.rows()));
        let rotation = &call.context.0.positions;
        type Pair = <TilePair as Group>::Period;
        let pairs = tile.count::<Tile>() / 2;
        let most = <Capacity<Z>>::SIZE / Pair::SIZE;
        for _ in 0..(pairs + most - 1) / most {
            let begin = Begin(launch.period::<TilePair>(&home, 0));
            value.forward(&mut *device, tile, &mut (), begin);
        }
        for at in 0..pairs {
            let start = tile.tile::<Pair>(at).start();
            let landing = ToDm::<TilePair, QueryHeads<Z>>(PhantomData);
            let next = launch.period::<TilePair>(&home, at);
            let landed = (
                (landing, (rotation, start / Pair::SIZE)),
                value.forward(&mut *device, tile, &mut (), next),
            );
            period(
                device,
                positions,
                (&front, &key, &value),
                tile,
                landed,
                start,
                (stored, adapters),
            );
        }
        for _ in 0..tile.count::<Tile>() % 2 {
            let start = tile.tile::<Tile>(2 * pairs).start();
            let landing = ToDm::<Tiles<1>, QueryHeads<Z>>(PhantomData);
            let next = launch.period::<Tiles<1>>(&home, pairs);
            let landed = (
                (landing, (rotation, start / Tile::SIZE)),
                value.forward(&mut *device, tile, &mut (), next),
            );
            period(
                device,
                positions,
                (&front, &key, &value),
                tile,
                landed,
                start,
                (stored, adapters),
            );
        }
        rows
    }

    fn end(
        &self,
        device: &mut Device,
        (attended, landed): Output<'_, Z, Reach<Z>, Capacity<Z>, OL>,
        input: RowTile<Z>,
        adapters: &mut T,
    ) -> Self::Staged {
        let Sequential(Residual(_, attention, sum), Residual(norm, mlp, _)) = self;
        let Attention { output, .. } = &attention.0;
        let (gate_up, down) = mlp.0.parts();
        let tile = input.with(());
        let gain = crate::Op::forward(&Stage, &mut *device, &norm.0);
        let rows = attended.with(&attended.value().0);
        // The first down panel lands behind the output projection's rows, under its passes, the
        // first gate/up pair behind the exchange of its partial sums, which the sum reads.
        let begun = output.forward(
            &mut *device,
            (),
            &mut (),
            Begin(Begin((&mut *landed, rows))),
        );
        let first = down.forward(&mut *device, (), &mut (), Begin(0usize));
        let begun = output.forward(&mut *device, (), &mut (), begun);
        let ahead = gate_up.forward(&mut *device, (), &mut (), Begin(()));
        let projected = tile.with(output.forward(&mut *device, (), adapters.output(), begun));
        let attended = sum.forward(&mut *device, (), &mut (), (&input, &projected));
        let normalized = gain.forward(&mut *device, (), &mut (), &attended);
        let rows = tile.with(normalized.value());
        let projected = gate_up.forward(&mut *device, (), adapters.gate_up(), (ahead, rows));
        // The second down panel lands under the last gate/up pair and the gating, the third behind
        // the down projection's inputs, so the projection starts on its first two panels; the next
        // part's weights land behind the third.
        let second = down.forward(&mut *device, (), &mut (), Begin(1usize));
        let product = SwiGlu.forward(&mut *device, (), &mut (), tile.with(projected));
        let product = tile.with(product);
        let staged = down.forward(&mut *device, (), &mut (), Begin((product, first)));
        let landing = down.forward(&mut *device, (), &mut (), tile.with((staged, second)));
        let third = down.forward(&mut *device, (), &mut (), Begin(2usize));
        let ready = down.forward(&mut *device, (), &mut (), (landing, third));
        Staged { ready, attended }
    }

    fn update(&self, device: &mut Device, staged: Self::Staged, adapters: &mut T) -> RowTile<Z> {
        let Sequential(_, Residual(_, mlp, sum)) = self;
        let (_, down) = mlp.0.parts();
        let Staged { ready, attended } = staged;
        let tile = attended.with(());
        let fed = tile.with(down.forward(&mut *device, (), adapters.down(), tile.with(ready)));
        sum.forward(device, (), &mut (), (&attended, &fed))
    }
}

/// A period of `G` tiles of a tile's rows through a layer's projections, their forms begun in
/// `front`, `key` and `value` and running with `adapters`, stored at row `start` of `stored`, its
/// rows landed and its rotation's coefficients landed by `landing` from `index`: the queries
/// projected, normalized, rotated and stored while the keys and values project, then the keys
/// normalized, rotated and stored with the values.
fn period<Z, G, P, GN, V, K, Q, QN, KN, QH, KH, T: States>(
    device: &mut Device,
    positions: &Rotary<Head>,
    (front, key, value): (&Front<GN, Q, QN, KN>, &K, &V),
    tile: Live<(), Capacity<Z>>,
    ((landing, index), rows): (Landing<'_, G, Z>, P),
    start: usize,
    (stored, adapters): (&mut Stored<Z, Reach<Z>>, &mut T),
) where
    Z: decoder::Attention<Heads = KvHeads, Head = Head> + Decode + 'static,
    G: Group,
    V: for<'r, 'x> Module<&'r P, Live<(), Capacity<Z>>, &'x mut T::Value, Output = Pairs<G>>,
    K: for<'r, 'x> Module<&'r P, Live<(), Capacity<Z>>, &'x mut T::Key, Output = KH>,
    Q: for<'r, 'x> Module<&'r P, Live<(), Capacity<Z>>, &'x mut T::Query, Output = QH>,
    QN: for<'x> Module<QH, Live<(), Capacity<Z>>, &'x mut (), Output = Queries<G, Z>>,
    KN: for<'x> Module<KH, Live<(), Capacity<Z>>, &'x mut (), Output = Pairs<G>>,
{
    let Front {
        query,
        query_norm,
        key_norm,
        ..
    } = front;
    let Stored { queries, retained } = stored;
    let projected = query.forward(&mut *device, tile, adapters.query(), &rows);
    // Behind the period's rows, which the projection reads first.
    let coefficients = crate::Op::forward(&landing, &mut *device, index);
    let normalized = query_norm.forward(&mut *device, tile, &mut (), projected);
    let rotation = crate::Op::forward(&group::Stage, &mut *device, &coefficients);
    let rotated = positions.forward(&mut *device, tile, &mut (), (&normalized, &rotation));
    let projected = key.forward(&mut *device, tile, adapters.key(), &rows);
    queries.spill(&mut *device, &rotated, start);
    let values = value.forward(&mut *device, tile, adapters.value(), &rows);
    let normalized = key_norm.forward(&mut *device, tile, &mut (), projected);
    let heads = (&coefficients, PhantomData::<KvHeads>);
    let rotation = crate::Op::forward(&group::Stage, &mut *device, heads);
    let rotated = positions.forward(&mut *device, tile, &mut (), (&normalized, &rotation));
    retained.spill(&mut *device, (&rotated, &values), start);
}

/// A tile of the rows `home` holds through the first layer up to its attention's store, its
/// projections running with `adapters`; the tile's rows, the first layer's input.
impl<'l, 's, A, C, X, T, H, Y, E, Reach>
    Module<Live<(), E>, C, (&'s mut Stored<A, Reach>, &'s mut T)> for Tokenwise<&'l H, &'l Y>
where
    A: decoder::Attention,
    Reach: AxisName,
    Y: Cut<A, C, T, Rows = X, Envelope = E, Reach = Reach, Home = H>,
    E: AxisName,
{
    type Output = Live<X, E>;

    fn forward(
        &self,
        device: &mut Device,
        call: C,
        (stored, adapters): (&'s mut Stored<A, Reach>, &'s mut T),
        tile: Live<(), E>,
    ) -> Self::Output {
        let Self(home, layer) = *self;
        let front = layer.front(device, tile);
        let rows = tile.with(layer.fill(&mut *device, home, tile));
        layer.start(device, call, front, rows, (stored, adapters))
    }
}

/// A tile through a layer from its attention and the next layer up to its attention's store, each
/// layer's leaves running with its own slots and reading its own call; the next layer's input
/// rows. The layers may attend in different groups; the next layer's front lands while this one's
/// MLP projects.
impl<'l, 'p, 's, A, B, C, D, X, T, U, P, Q, E, RA, RB, L>
    Module<
        (Output<'p, A, RA, E, L>, Live<X, E>),
        (C, D),
        (&'s mut Stored<B, RB>, (&'s mut T, &'s mut U)),
    > for Tokenwise<&'l P, &'l Q>
where
    A: decoder::Attention,
    B: decoder::Attention,
    E: AxisName,
    RA: AxisName,
    RB: AxisName,
    P: Cut<A, C, T, Rows = X, Envelope = E, Reach = RA, Landed = L>,
    Q: Cut<B, D, U, Rows = X, Envelope = E, Reach = RB>,
{
    type Output = Live<X, E>;

    fn forward(
        &self,
        device: &mut Device,
        (_, call): (C, D),
        (stored, (adapters, following)): (&'s mut Stored<B, RB>, (&'s mut T, &'s mut U)),
        (output, input): (Output<'p, A, RA, E, L>, Live<X, E>),
    ) -> Self::Output {
        let Self(layer, next) = *self;
        let tile = input.with(());
        let staged = layer.end(device, output, input, adapters);
        let front = next.front(device, tile);
        let rows = layer.update(device, staged, adapters);
        next.start(device, call, front, rows, (stored, following))
    }
}

/// A tile through the last layer from its attention, its leaves running with `adapters`, and the
/// final norm; the head's rows.
impl<'l, 'p, 's, A, C, X, T, P, E, RA, L, N>
    Module<(Output<'p, A, RA, E, L>, Live<X, E>), C, &'s mut T>
    for Tokenwise<&'l P, &'l Named<"norm", N>>
where
    A: decoder::Attention,
    E: AxisName,
    RA: AxisName,
    P: Cut<A, C, T, Rows = X, Envelope = E, Reach = RA, Landed = L>,
    Named<"norm", N>: for<'r, 'x> Module<&'r Live<X, E>, (), &'x mut (), Output = Live<X, E>>,
{
    type Output = Live<X, E>;

    fn forward(
        &self,
        device: &mut Device,
        _: C,
        adapters: &'s mut T,
        (output, input): (Output<'p, A, RA, E, L>, Live<X, E>),
    ) -> Self::Output {
        let Self(layer, norm) = *self;
        let staged = layer.end(device, output, input, adapters);
        let rows = layer.update(device, staged, adapters);
        norm.forward(device, (), &mut (), &rows)
    }
}

/// A launch's rows through the catalog base model's leaves in the family's order, at any size and
/// in any projection form whose leaves run the cut: embedded, the first layer's stores over every
/// tile, then each layer attended and, tile by tile, its work up to the next layer's store or the
/// head; the head's rows and the rows the launch selects among them. Each layer's projections run
/// with the adapters `T` its slot holds. Generic in its env, so only the launches that run it
/// compile it.
impl<
    'm,
    'i,
    'a,
    's,
    'b,
    Z,
    P,
    T,
    const DEPTH: usize,
    H,
    const E: usize,
    const C: usize,
    const L: usize,
    OL,
    Env,
> Module<&'i mut launch::Input<'a, Z>, Env, launch::Body<'s, Z, [launch::Layer<'b, T>; DEPTH]>>
    for Decoder<&'m launch::Base<Z, P>>
where
    Z: Geometry<Hidden = H, Norm<H, Hbm> = RmsNorm<H, E, Chip>, Layers<()> = [(); DEPTH]>
        + decoder::Attention<Heads = KvHeads, Head = Head>
        + Decode
        + qwen3::Front
        + 'static,
    H: Tiled<Capacity = axis::Rows<C>, Last = axis::Rows<L>>,
    P: Form,
    T: States,
    swiglu::Down<Intermediate<Z>>: AxisName,
    Catalog<P, Z>: View<DEPTH, Layer = Catalog<P, Z>>
        + for<'c> Cut<
            Z,
            Launched<'c, 'a, Z>,
            T,
            Rows = Owned<Width<Z>, Span>,
            Envelope = Capacity<Z>,
            Reach = Reach<Z>,
            Home = Rows<Z>,
            Landed = OL,
        >,
    Rows<Z>: Fill<Device, Live<(), Capacity<Z>>, Rows = Owned<Width<Z>, Span>>
        + for<'r> Spill<Device, &'r Owned<Width<Z>, Span>, Live<(), Capacity<Z>>>,
{
    type Output = (Rows<Z>, &'a Selected<Capacity<Z>>);

    fn forward(
        &self,
        device: &mut Device,
        _: Env,
        body: launch::Body<'s, Z, [launch::Layer<'b, T>; DEPTH]>,
        input: &'i mut launch::Input<'a, Z>,
    ) -> Self::Output {
        let model = self.0;
        let live = Live::<(), Capacity<Z>>::read(&input.count);
        let context =
            (model.rotary_emb).forward(&mut *device, (), &mut (), (live, &mut input.cache));
        let call = Call {
            context: &context,
            tokens: &input.tokens,
        };
        let ((embedded, slots), ()) = body;
        let Then(Backbone { embedding, body }, norm) = &model.body;
        let launch = embedding::Launch {
            ids: call.tokens,
            count: live,
            output: &mut *embedded,
        };
        crate::Op::forward(&embedding.0, &mut *device, launch);
        let (layers, spans) = (&body.layers.0, live.count::<Span>());
        let mut places = Places::new();
        let mut home: Rows<Z> = HbmTensor::new();
        let first = layers.layer(0);
        let mut adapters = slots[0].1;
        for span in 0..spans {
            let tile = live.tile::<Span>(span);
            let tokenwise = Tokenwise(&*embedded, &first);
            let state = (&mut places.stored, &mut adapters);
            let rows = tokenwise.forward(&mut *device, call, state, tile);
            home.spill(&mut *device, rows.value(), tile);
        }
        for index in 0..DEPTH - 1 {
            let (layer, next) = (layers.layer(index), layers.layer(index + 1));
            let pool = (&mut slots[index].0.1, live);
            let mut projecting = layer.attend(&mut *device, call, pool, &mut places);
            let (mut adapters, mut following) = (slots[index].1, slots[index + 1].1);
            // In place, so the loop carries one state: a tile overwrites only its own rows, read
            // just before, and the attention has read every store.
            for span in 0..spans {
                let tile = live.tile::<Span>(span);
                let input = tile.with(home.fill(&mut *device, tile));
                let tokenwise = Tokenwise(&layer, &next);
                let output = (tile.with(&places.attended), &mut projecting);
                let state = (&mut places.stored, (&mut adapters, &mut following));
                let rows = tokenwise.forward(&mut *device, (call, call), state, (output, input));
                home.spill(&mut *device, rows.value(), tile);
            }
        }
        let last = layers.layer(DEPTH - 1);
        let pool = (&mut slots[DEPTH - 1].0.1, live);
        let mut projecting = last.attend(&mut *device, call, pool, &mut places);
        let mut adapters = slots[DEPTH - 1].1;
        let mut head: Rows<Z> = HbmTensor::new();
        for span in 0..spans {
            let tile = live.tile::<Span>(span);
            let input = tile.with(home.fill(&mut *device, tile));
            let output = (tile.with(&places.attended), &mut projecting);
            let tokenwise = Tokenwise(&last, norm);
            let rows = tokenwise.forward(&mut *device, call, &mut adapters, (output, input));
            head.spill(&mut *device, rows.value(), tile);
        }
        (head, call.context.1)
    }
}

/// The base model's loops over a launch of size `Z`: the head's rows and the rows it selects.
type Loops<'a, Z> = (Rows<Z>, &'a Selected<Capacity<Z>>);

/// A launch of the catalog's causal model: its base model's loops, then the selected rows' greedy
/// tokens.
impl<'i, 'a, 's, Z: Geometry + decoder::Attention + Decode + 'static, B, P: Form, L, Env>
    Module<&'i mut launch::Input<'a, Z>, Env, launch::State<'s, Z, L>>
    for ForCausalLM<Z, launch::Over<Z, B, P>, LmHead<Z, Hbm>, P>
where
    for<'m> Decoder<&'m launch::Over<Z, B, P>>:
        Module<&'i mut launch::Input<'a, Z>, Env, launch::Body<'s, Z, L>, Output = Loops<'a, Z>>,
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        state: launch::State<'s, Z, L>,
        input: &'i mut launch::Input<'a, Z>,
    ) {
        let (body, tokens) = state;
        let (head, selected) = Decoder(&self.model).forward(&mut *device, env, body, input);
        self.lm_head.read(device, &head, selected, tokens);
    }
}

/// A launch of the catalog's base model: its loops, then the selected rows' hidden values.
impl<'i, 'a, 's, Z: Geometry + decoder::Attention + Decode + 'static, B, P: Form, L, Env>
    Module<&'i mut launch::Input<'a, Z>, Env, launch::State<'s, Z, L, launch::Values<Z>>>
    for launch::Over<Z, B, P>
where
    for<'m> Decoder<&'m Self>:
        Module<&'i mut launch::Input<'a, Z>, Env, launch::Body<'s, Z, L>, Output = Loops<'a, Z>>,
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        state: launch::State<'s, Z, L, launch::Values<Z>>,
        input: &'i mut launch::Input<'a, Z>,
    ) {
        let (body, rows) = state;
        let (head, selected) = Decoder(self).forward(&mut *device, env, body, input);
        selected.hidden(device, &head, rows);
    }
}

/// A launch of the catalog's sequence classifier: its base model's loops, then the selected rows'
/// scores.
impl<'i, 'a, 's, Z: Geometry + decoder::Attention + Decode + 'static, B, P: Form, L, Env>
    Module<&'i mut launch::Input<'a, Z>, Env, launch::State<'s, Z, L, launch::Scores<Z>>>
    for ForSequenceClassification<Z, launch::Over<Z, B, P>, Classifier<Width<Z>>, P>
where
    for<'m> Decoder<&'m launch::Over<Z, B, P>>:
        Module<&'i mut launch::Input<'a, Z>, Env, launch::Body<'s, Z, L>, Output = Loops<'a, Z>>,
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        state: launch::State<'s, Z, L, launch::Scores<Z>>,
        input: &'i mut launch::Input<'a, Z>,
    ) {
        let (body, scores) = state;
        let (head, selected) = Decoder(&self.model).forward(&mut *device, env, body, input);
        self.score.read(device, &head, selected, scores);
    }
}

/// Block-scaled FP8 Qwen3-8B's launch runs its catalog model.
impl<'i, 'a, 's, L, Env>
    Module<
        &'i mut launch::Input<'a, qwen3::Size<"8B">>,
        Env,
        launch::State<'s, qwen3::Size<"8B">, L>,
    > for fp8::ForCausalLM
where
    launch::Model<qwen3::Size<"8B">, BlockScaled>: Module<
            &'i mut launch::Input<'a, qwen3::Size<"8B">>,
            Env,
            launch::State<'s, qwen3::Size<"8B">, L>,
            Output = (),
        >,
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        state: launch::State<'s, qwen3::Size<"8B">, L>,
        input: &'i mut launch::Input<'a, qwen3::Size<"8B">>,
    ) {
        self.model.forward(device, env, state, input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::decoder::attention::llama;
    use crate::models::decoder::form::{Affine, ChannelScaled};

    /// Each catalog root, of every size and projection form a registry row serves, runs by the
    /// decoder family's one loop on its launch's input and the state ending in the place its head
    /// writes: the causal model's tokens, the base model's hidden values, the classifier's scores.
    #[test]
    fn every_root_runs_by_the_loop() {
        fn runs<R, Z: Geometry + decoder::Attention + Decode, O>()
        where
            for<'i, 'a, 's, 'b> R: Module<
                    &'i mut launch::Input<'a, Z>,
                    (),
                    launch::State<'s, Z, <Z as Geometry>::Layers<launch::Layer<'b>>, O>,
                >,
        {
        }
        type Q<const N: &'static str> = qwen3::Size<N>;
        runs::<launch::Model<Q<"0.6B">, Affine>, Q<"0.6B">, launch::Tokens<Q<"0.6B">>>();
        runs::<launch::Base<Q<"0.6B">, Affine>, Q<"0.6B">, launch::Values<Q<"0.6B">>>();
        runs::<launch::Classifier<Q<"0.6B">, Affine>, Q<"0.6B">, launch::Scores<Q<"0.6B">>>();
        runs::<launch::Model<Q<"1.7B">, Affine>, Q<"1.7B">, launch::Tokens<Q<"1.7B">>>();
        runs::<launch::Base<Q<"1.7B">, Affine>, Q<"1.7B">, launch::Values<Q<"1.7B">>>();
        runs::<launch::Classifier<Q<"1.7B">, Affine>, Q<"1.7B">, launch::Scores<Q<"1.7B">>>();
        runs::<launch::Model<Q<"4B">, Affine>, Q<"4B">, launch::Tokens<Q<"4B">>>();
        runs::<launch::Base<Q<"4B">, Affine>, Q<"4B">, launch::Values<Q<"4B">>>();
        runs::<launch::Classifier<Q<"4B">, Affine>, Q<"4B">, launch::Scores<Q<"4B">>>();
        runs::<launch::Model<Q<"8B">, Affine>, Q<"8B">, launch::Tokens<Q<"8B">>>();
        runs::<launch::Base<Q<"8B">, Affine>, Q<"8B">, launch::Values<Q<"8B">>>();
        runs::<launch::Classifier<Q<"8B">, Affine>, Q<"8B">, launch::Scores<Q<"8B">>>();
        runs::<launch::Model<Q<"0.6B">, ChannelScaled>, Q<"0.6B">, launch::Tokens<Q<"0.6B">>>();
        runs::<launch::Model<Q<"1.7B">, ChannelScaled>, Q<"1.7B">, launch::Tokens<Q<"1.7B">>>();
        runs::<launch::Model<Q<"4B">, ChannelScaled>, Q<"4B">, launch::Tokens<Q<"4B">>>();
        runs::<fp8::ForCausalLM, Q<"8B">, launch::Tokens<Q<"8B">>>();
        type Llama = llama::Size<"8B">;
        runs::<launch::Model<Llama, Affine>, Llama, launch::Tokens<Llama>>();
        type Mistral = llama::Size<"Mistral-7B">;
        runs::<launch::Model<Mistral, Affine>, Mistral, launch::Tokens<Mistral>>();
    }
}
