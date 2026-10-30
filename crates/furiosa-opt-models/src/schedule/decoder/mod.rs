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
//! - The cut of a layer at its attention is private to the impls here, written once for the
//!   pre-norm layer of Qwen3, Llama and Mistral at every size and in every projection form. It is
//!   generic in the layer's leaves, each of which takes the cut's steps its own way: the
//!   block-scaled output projection projects a tile over three steps, the dense one in the last
//!   of them; Llama's and Mistral's head norms, `()`, leave the heads as projected.
//! - [`Decoder`] runs the loops: the first layer's stores over the embedded tiles, then for each
//!   layer its attention and, tile by tile, the work up to the next store or the head.
//! - The DMA engine serves transfers in issue order, so weights issue one part ahead of their use
//!   and behind the small transfers of the work they overlap: the MLP's first weights behind the
//!   output projection's rows, its down weights behind the gate/up pairs, the next layer's
//!   projections behind the down's inputs. An attention's working set fills most of data memory,
//!   so only the output projection lands during one, once its last history is in flight. A form
//!   whose leaves land nothing ahead takes the same steps, doing nothing in them.

use crate::Module;
use crate::load::Named;
use crate::models::decoder::Form;
use crate::models::decoder::attention::qwen3::{self, LmHead, QueryHeads, fp8};
use crate::models::decoder::attention::{
    ForCausalLM, ForSequenceClassification, Geometry, Head, KvHeads, Qwen2Layer,
};
use crate::models::decoder::form::BlockScaled;
use crate::models::decoder::launch;
use crate::modules::attention::{Attention, NormalizedQkv, Positioned, Projection};
use crate::modules::generic::collection::View;
use crate::modules::generic::schedule::{Begin, Fill, Spill};
use crate::modules::language::{Backbone, Call, Classifier, Selected};
use crate::modules::lora::States;
use crate::modules::residual::Residual;
use crate::ops::attention::full::paged::Retained;
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
use crate::tensor::live::Live;
use crate::tensor::memory::Hbm;
use crate::tensor::parallel::Chip;
use crate::{Sequential, Then};
use core::marker::PhantomData;
use furiosa_opt_std::prelude::*;

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
/// The catalog layer's output projection in form `P`.
type OutputProjection<P, Z> = <P as Form>::Output<<Z as Geometry>::Queries, Width<Z>, Hbm>;
type Capacity<Z> = launch::Capacity<Z>;
type Reach<Z> = <Z as Decode>::Reach;
type Width<Z> = <Z as Geometry>::Hidden;
type Intermediate<Z> = <Z as Geometry>::Intermediate;
type Groups<Z> = <Z as Geometry>::Groups;
/// A launch's rows of the hidden width in HBM: the embedded rows, a layer's input, the head's rows.
type Rows<Z> = HbmTensor<bf16, Chip, Blocked<Width<Z>, Capacity<Z>>>;
/// A launch's rows of each KV head and query group: its queries, or what its attention gives.
type QueryRows<Z> =
    HbmTensor<bf16, Chip, m![{ Symbol<Reach<Z>> }, KvHeads, { Symbol<Groups<Z>> }, Head]>;
/// A launch's rows of each KV head: its keys or its values.
type KeyRows<Z> = HbmTensor<bf16, Chip, m![{ Symbol<Reach<Z>> }, KvHeads, Head]>;
type Norm<Z> = <Z as Geometry>::Norm<Width<Z>, Hbm>;
/// The final norm, where the head starts.
type FinalNorm<Z> = Named<"norm", Norm<Z>>;
/// A launch's call as its layers read it: the context its rotary table prepares, its token IDs
/// and its live rows.
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
/// A period's rotation coefficients in the slices of its queries, for periods of `G` tiles.
type Coefficients<G, Z> =
    <ToDm<G, QueryHeads<Z>> as crate::Op<(&'static Rotation<Z>, usize)>>::Output;
/// A span tile of the attended rows as the output projection's steps take it, with what its form
/// landed.
type Projecting<'o, 'r, Z, L> = (&'o mut L, Live<&'r QueryRows<Z>, Capacity<Z>>);

/// What a layer's attention stores of a launch's tiles: their queries, keys and values.
pub struct Stored<Z: Geometry + Decode> {
    queries: QueryRows<Z>,
    retained: (KeyRows<Z>, KeyRows<Z>),
}

/// The rows a layer's attention attends of a launch.
pub struct Attended<Z: Geometry + Decode>(QueryRows<Z>);

/// A tile of a layer's attention output: the attended rows and the output projection over them,
/// `L` as its form lands it.
type Output<'p, Z, L> = (Live<&'p Attended<Z>, Capacity<Z>>, &'p mut L);

/// One layer's places of a launch: its input rows, what its attention stores and what it attends.
struct Places<Z: Geometry + Decode> {
    input: Rows<Z>,
    stored: Stored<Z>,
    attended: Attended<Z>,
}

impl<Z: Geometry + Decode> Places<Z> {
    fn new() -> Self {
        Self {
            input: HbmTensor::new(),
            stored: Stored {
                queries: HbmTensor::new(),
                retained: (HbmTensor::new(), HbmTensor::new()),
            },
            attended: Attended(HbmTensor::new()),
        }
    }
}

// # The cut of a pre-norm layer at its attention

/// A layer of the size `Z` cut at its attention: the steps of the work between two attentions
/// that run its leaves, each over one row tile. The work's order is [`Tokenwise`]'s; a layer shape
/// gives the steps.
trait Cut<Z: Geometry + Decode + 'static, A: States> {
    /// Its front landed for a tile's rows.
    type Front;
    /// Its output projection as its form lands it at its attention's boundary.
    type Landed;
    /// A tile between its MLP's two steps.
    type Staged;

    /// Its front landed for the rows of `tile`.
    fn front(&self, device: &mut Device, tile: Live<(), Capacity<Z>>) -> Self::Front;

    /// A tile's input `rows` through it up to its attention's store, its `front` landed: the input
    /// norm, then every period's queries, keys and values, head-normed, rotated and stored in
    /// `stored`; the rows, for the residual.
    fn start(
        &self,
        device: &mut Device,
        call: Launched<'_, '_, Z>,
        front: Self::Front,
        rows: RowTile<Z>,
        state: (&mut Stored<Z>, &mut A),
    ) -> RowTile<Z>;

    /// A tile through it from its attention to its MLP's second step, given a tile `output` of its
    /// attention output and `input` of its input rows: the output projection, the attention
    /// residual and the MLP's first step on its norm.
    fn end(
        &self,
        device: &mut Device,
        output: Output<'_, Z, Self::Landed>,
        input: RowTile<Z>,
        adapters: &mut A,
    ) -> Self::Staged;

    /// A `staged` tile through its MLP's second step, the down projection, and the MLP residual;
    /// the rows.
    fn update(&self, device: &mut Device, staged: Self::Staged, adapters: &mut A) -> RowTile<Z>;
}

/// The forms a layer's input norm, projections and head norms take for a tile's rows, landed: the
/// norm's gain, the value form, the launch of the tile's rows it stages, the key and query forms
/// and the two head norms' forms.
struct Front<G, V, L, K, Q, QN, KN> {
    gain: G,
    value: V,
    launch: L,
    key: K,
    query: Q,
    query_norm: QN,
    key_norm: KN,
}

/// A tile between its MLP's two steps: the down projection's inputs `R` landed, and the rows of the
/// attention residual, which the MLP residual adds to.
struct Staged<R, Z: Geometry + Decode> {
    ready: R,
    attended: RowTile<Z>,
}

/// The pre-norm layer, in every form whose leaves take each step. The value projection's form
/// `VB`, begun, begins the launch whose periods it stages in `F` and gives as `PP` (pairs of tiles)
/// and `PS` (single tiles); the key and query forms `KB`, `QB` give a period's heads `KHP`, `QHP`
/// (`KHS`, `QHS`), which the head norms' forms `KNB`, `QNB` leave as the rotation reads them. The
/// output projection lands as `OL`; the MLP's gate/up form `GB` gives `GO` and its gating `SO`,
/// whose down projection lands a weight panel as `DP`, stages the product as `DS` and lands its
/// inputs as `DR`.
impl<
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
    DR,
    T,
> Cut<Z, T> for Layer<Z, Q, K, V, QN, KN, A, O, G, D>
where
    T: States,
    Z: Geometry<KvHeads = KvHeads, Head = Head, Hidden = H, Norm<H, Hbm> = RmsNorm<H, E, Chip>>
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
        + for<'x> Module<Live<(DS, (DP, DP)), Capacity<Z>>, (), &'x mut (), Output = DR>
        + for<'x> Module<Live<DR, Capacity<Z>>, (), &'x mut T::Down, Output = Owned<Width<Z>, Span>>,
{
    type Front = Front<
        <Stage as crate::Op<&'static Norm<Z>>>::Output,
        VB,
        Launch<Width<Z>, F, Capacity<Z>>,
        KB,
        QB,
        QNB,
        KNB,
    >;
    type Landed = OL;
    type Staged = Staged<DR, Z>;

    fn front(&self, device: &mut Device, tile: Live<(), Capacity<Z>>) -> Self::Front {
        let Sequential(Residual(norm, attention, _), _) = self;
        let Attention {
            qkv:
                Positioned {
                    projection:
                        NormalizedQkv {
                            projection: Projection { query, key, value },
                            query_norm,
                            key_norm,
                        },
                    ..
                },
            ..
        } = &attention.0;
        let value = value.forward(&mut *device, tile, &mut (), Begin(()));
        let launch = value.forward(&mut *device, tile, &mut (), Begin(()));
        let key = key.forward(&mut *device, tile, &mut (), Begin(()));
        let query = query.forward(&mut *device, tile, &mut (), Begin(()));
        let query_norm = query_norm.forward(&mut *device, tile, &mut (), Begin(()));
        let key_norm = key_norm.forward(&mut *device, tile, &mut (), Begin(()));
        let gain = crate::Op::forward(&Stage, &mut *device, &norm.0);
        Front {
            gain,
            value,
            launch,
            key,
            query,
            query_norm,
            key_norm,
        }
    }

    fn start(
        &self,
        device: &mut Device,
        call: Launched<'_, '_, Z>,
        mut front: Self::Front,
        rows: RowTile<Z>,
        (stored, adapters): (&mut Stored<Z>, &mut T),
    ) -> RowTile<Z> {
        let Sequential(Residual(_, attention, _), _) = self;
        let Attention {
            qkv: Positioned { positions, .. },
            ..
        } = &attention.0;
        let tile = rows.with(());
        let normalized = front.gain.forward(&mut *device, (), &mut (), &rows);
        // The rows the periods read lie apart from the launch, which the period loops carry.
        let mut home = Normalized::new();
        home.spill(&mut *device, normalized.value(), (tile, tile.rows()));
        let rotation = &call.context.0.positions;
        type Pair = <TilePair as Group>::Period;
        let pairs = tile.count::<Tile>() / 2;
        let most = <Capacity<Z>>::SIZE / Pair::SIZE;
        for _ in 0..(pairs + most - 1) / most {
            let begin = Begin(front.launch.period::<TilePair>(&home, 0));
            front.value.forward(&mut *device, tile, &mut (), begin);
        }
        for at in 0..pairs {
            let start = tile.tile::<Pair>(at).start();
            let landing = ToDm::<TilePair, QueryHeads<Z>>(PhantomData);
            let index = (rotation, start / Pair::SIZE);
            let coefficients = crate::Op::forward(&landing, &mut *device, index);
            let next = front.launch.period::<TilePair>(&home, at);
            let landed = (
                coefficients,
                front.value.forward(&mut *device, tile, &mut (), next),
            );
            period(
                device,
                positions,
                &front,
                tile,
                landed,
                start,
                (stored, adapters),
            );
        }
        for _ in 0..tile.count::<Tile>() % 2 {
            let start = tile.tile::<Tile>(2 * pairs).start();
            let landing = ToDm::<Tiles<1>, QueryHeads<Z>>(PhantomData);
            let index = (rotation, start / Tile::SIZE);
            let coefficients = crate::Op::forward(&landing, &mut *device, index);
            let next = front.launch.period::<Tiles<1>>(&home, pairs);
            let landed = (
                coefficients,
                front.value.forward(&mut *device, tile, &mut (), next),
            );
            period(
                device,
                positions,
                &front,
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
        (attended, landed): Output<'_, Z, OL>,
        input: RowTile<Z>,
        adapters: &mut T,
    ) -> Self::Staged {
        let Sequential(Residual(_, attention, sum), Residual(norm, mlp, _)) = self;
        let Attention { output, .. } = &attention.0;
        let (gate_up, down) = mlp.0.parts();
        let tile = input.with(());
        let gain = crate::Op::forward(&Stage, &mut *device, &norm.0);
        let rows = attended.with(&attended.value().0);
        // The first gate/up pair lands behind the output projection's rows, under its passes.
        let begun = output.forward(
            &mut *device,
            (),
            &mut (),
            Begin(Begin((&mut *landed, rows))),
        );
        let ahead = gate_up.forward(&mut *device, (), &mut (), Begin(()));
        let begun = output.forward(&mut *device, (), &mut (), begun);
        let first = down.forward(&mut *device, (), &mut (), Begin(0usize));
        let projected = tile.with(output.forward(&mut *device, (), adapters.output(), begun));
        let attended = sum.forward(&mut *device, (), &mut (), (&input, &projected));
        let normalized = gain.forward(&mut *device, (), &mut (), &attended);
        let rows = tile.with(normalized.value());
        let projected = gate_up.forward(&mut *device, (), adapters.gate_up(), (ahead, rows));
        // The later down panels land under the last gate/up pair and the gating, the next part's
        // weights under the down projection once its inputs landed.
        let later = (
            down.forward(&mut *device, (), &mut (), Begin(1usize)),
            down.forward(&mut *device, (), &mut (), Begin(2usize)),
        );
        let product = SwiGlu.forward(&mut *device, (), &mut (), tile.with(projected));
        let product = tile.with(product);
        let staged = down.forward(&mut *device, (), &mut (), Begin((product, first)));
        let ready = down.forward(&mut *device, (), &mut (), tile.with((staged, later)));
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
/// `front` and running with `adapters`, stored at row `start` of `stored`, its rows landed and its
/// rotation's coefficients before them: the queries projected, normalized, rotated and stored
/// while the keys and values project, then the keys normalized, rotated and stored with the values.
fn period<Z, G, P, GN, V, LA, K, Q, QN, KN, QH, KH, T: States>(
    device: &mut Device,
    positions: &Rotary<Head>,
    front: &Front<GN, V, LA, K, Q, QN, KN>,
    tile: Live<(), Capacity<Z>>,
    (coefficients, rows): (Coefficients<G, Z>, P),
    start: usize,
    (stored, adapters): (&mut Stored<Z>, &mut T),
) where
    Z: Geometry<KvHeads = KvHeads, Head = Head> + Decode + 'static,
    G: Group,
    V: for<'r, 'x> Module<&'r P, Live<(), Capacity<Z>>, &'x mut T::Value, Output = Pairs<G>>,
    K: for<'r, 'x> Module<&'r P, Live<(), Capacity<Z>>, &'x mut T::Key, Output = KH>,
    Q: for<'r, 'x> Module<&'r P, Live<(), Capacity<Z>>, &'x mut T::Query, Output = QH>,
    QN: for<'x> Module<QH, Live<(), Capacity<Z>>, &'x mut (), Output = Queries<G, Z>>,
    KN: for<'x> Module<KH, Live<(), Capacity<Z>>, &'x mut (), Output = Pairs<G>>,
{
    let Front {
        query,
        key,
        value,
        query_norm,
        key_norm,
        ..
    } = front;
    let Stored { queries, retained } = stored;
    let projected = query.forward(&mut *device, tile, adapters.query(), &rows);
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

/// A tile of the embedded rows through the first layer up to its attention's store, its
/// projections running with `adapters`; the tile's rows, the first layer's input.
impl<'l, 'c, 'a, 's, Z: Geometry + Decode + 'static, T: States, X: Cut<Z, T>>
    Module<Live<(), Capacity<Z>>, Launched<'c, 'a, Z>, (&'s mut Stored<Z>, &'s mut T)>
    for Tokenwise<&'l Rows<Z>, &'l X>
where
    Rows<Z>: Fill<Device, Live<(), Capacity<Z>>, Rows = Owned<Width<Z>, Span>>,
{
    type Output = RowTile<Z>;

    fn forward(
        &self,
        device: &mut Device,
        call: Launched<'c, 'a, Z>,
        (stored, adapters): (&'s mut Stored<Z>, &'s mut T),
        tile: Live<(), Capacity<Z>>,
    ) -> Self::Output {
        let Self(embedded, layer) = *self;
        let front = layer.front(device, tile);
        let rows = tile.with(embedded.fill(&mut *device, tile));
        layer.start(device, call, front, rows, (stored, adapters))
    }
}

/// A tile through a layer from its attention and the next layer up to its attention's store, each
/// layer's projections running with its adapters; the next layer's input rows. The next layer's
/// front lands while this one's MLP projects.
impl<'l, 'c, 'a, 'p, 's, Z: Geometry + Decode + 'static, T: States, X: Cut<Z, T, Landed = L>, L>
    Module<
        (Output<'p, Z, L>, RowTile<Z>),
        Launched<'c, 'a, Z>,
        (&'s mut Stored<Z>, (&'s mut T, &'s mut T)),
    > for Tokenwise<&'l X, &'l X>
{
    type Output = RowTile<Z>;

    fn forward(
        &self,
        device: &mut Device,
        call: Launched<'c, 'a, Z>,
        (stored, (adapters, following)): (&'s mut Stored<Z>, (&'s mut T, &'s mut T)),
        (output, input): (Output<'p, Z, L>, RowTile<Z>),
    ) -> Self::Output {
        let Self(layer, next) = *self;
        let tile = input.with(());
        let staged = layer.end(device, output, input, adapters);
        let front = next.front(device, tile);
        let rows = layer.update(device, staged, adapters);
        next.start(device, call, front, rows, (stored, following))
    }
}

/// A tile through the last layer from its attention, its projections running with `adapters`, and
/// the final norm; the head's rows.
impl<'l, 'c, 'a, 'p, 's, Z: Geometry + Decode + 'static, T: States, X: Cut<Z, T, Landed = L>, L>
    Module<(Output<'p, Z, L>, RowTile<Z>), Launched<'c, 'a, Z>, &'s mut T>
    for Tokenwise<&'l X, &'l FinalNorm<Z>>
where
    FinalNorm<Z>: for<'r, 'x> Module<&'r RowTile<Z>, (), &'x mut (), Output = RowTile<Z>>,
{
    type Output = RowTile<Z>;

    fn forward(
        &self,
        device: &mut Device,
        _: Launched<'c, 'a, Z>,
        adapters: &'s mut T,
        (output, input): (Output<'p, Z, L>, RowTile<Z>),
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
    Z: Geometry<
            KvHeads = KvHeads,
            Head = Head,
            Hidden = H,
            Norm<H, Hbm> = RmsNorm<H, E, Chip>,
            Layers<()> = [(); DEPTH],
        > + Decode
        + qwen3::Front
        + 'static,
    H: Tiled<Capacity = axis::Rows<C>, Last = axis::Rows<L>>,
    P: Form,
    T: States,
    swiglu::Down<Intermediate<Z>>: AxisName,
    Catalog<P, Z>: View<DEPTH, Layer = Catalog<P, Z>> + Cut<Z, T, Landed = OL>,
    OutputProjection<P, Z>: for<'x> Module<Begin<()>, (), &'x mut (), Output = OL>
        + for<'d, 'g> Module<&'g mut OL, (), &'d mut (), Output = ()>,
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
            live,
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
        // A layer's attention once every tile stored its rows: the rows attended as items written
        // to the cache before, the deferred rows after.
        let attend = |device: &mut Device,
                      layer: &Catalog<P, Z>,
                      slot: &mut launch::Layer<'b, T>,
                      places: &mut Places<Z>| {
            let Sequential(Residual(_, attention, _), _) = layer;
            let Attention {
                cache: stage,
                attend,
                output,
                ..
            } = &attention.0;
            let (context, pool) = (&call.context.0, &mut slot.0.1);
            let operands = (&context.write, &context.read);
            let deferred = stage.forward(
                &mut *device,
                (),
                &mut *pool,
                (&places.stored.retained, operands, live),
            );
            let history = Retained {
                pool: &**pool,
                key: &places.stored.retained.0,
                value: &places.stored.retained.1,
                read: &stage.read,
            };
            let rows = (&places.stored.queries, &history, &context.read);
            // The output projection's weights land while the attention's last history is in flight.
            let mut projecting = output.forward(&mut *device, (), &mut (), Begin(()));
            let boundary = (&output.0, &mut projecting);
            let rows = (rows.0, rows.1, rows.2, boundary, &mut places.attended.0);
            attend.forward(&mut *device, (), &mut (), rows);
            let rows = (deferred, &places.stored.retained, &context.write);
            stage.forward(device, (), &mut *pool, rows);
            projecting
        };
        let mut places = Places::new();
        let first = layers.layer(0);
        let mut adapters = slots[0].1;
        for span in 0..spans {
            let tile = live.tile::<Span>(span);
            let tokenwise = Tokenwise(&*embedded, &first);
            let state = (&mut places.stored, &mut adapters);
            let rows = tokenwise.forward(&mut *device, call, state, tile);
            places.input.spill(&mut *device, rows.value(), tile);
        }
        for index in 0..DEPTH - 1 {
            let (layer, next) = (layers.layer(index), layers.layer(index + 1));
            let mut projecting = attend(&mut *device, &layer, &mut slots[index], &mut places);
            let (mut adapters, mut following) = (slots[index].1, slots[index + 1].1);
            // In place, so the loop carries one state: a tile overwrites only its own rows, read
            // just before, and the attention has read every store.
            for span in 0..spans {
                let tile = live.tile::<Span>(span);
                let input = tile.with(places.input.fill(&mut *device, tile));
                let tokenwise = Tokenwise(&layer, &next);
                let output = (tile.with(&places.attended), &mut projecting);
                let state = (&mut places.stored, (&mut adapters, &mut following));
                let rows = tokenwise.forward(&mut *device, call, state, (output, input));
                places.input.spill(&mut *device, rows.value(), tile);
            }
        }
        let last = layers.layer(DEPTH - 1);
        let mut projecting = attend(&mut *device, &last, &mut slots[DEPTH - 1], &mut places);
        let mut adapters = slots[DEPTH - 1].1;
        let mut head: Rows<Z> = HbmTensor::new();
        for span in 0..spans {
            let tile = live.tile::<Span>(span);
            let input = tile.with(places.input.fill(&mut *device, tile));
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
impl<'i, 'a, 's, Z: Geometry + Decode + 'static, B, P: Form, L, Env>
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
impl<'i, 'a, 's, Z: Geometry + Decode + 'static, B, P: Form, L, Env>
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
impl<'i, 'a, 's, Z: Geometry + Decode + 'static, B, P: Form, L, Env>
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
        fn runs<R, Z: Geometry + Decode, O>()
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
