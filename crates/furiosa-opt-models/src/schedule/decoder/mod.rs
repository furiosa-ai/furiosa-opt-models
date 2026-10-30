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
//! - The cut of a layer at its attention is private to the impls here, one for each layer shape the
//!   family runs: Qwen3's pre-norm layer in block-scaled FP8, and the pre-norm layer of Qwen3,
//!   Llama and Mistral in BF16 at every size ([`affine`]), whose query and key heads Llama's and
//!   Mistral's leave unnormed.
//! - [`Decoder`] runs the loops: the first layer's stores over the embedded tiles, then for each
//!   layer its attention and, tile by tile, the work up to the next store or the head.
//! - Weights land one part ahead of their use: the next layer's projections while this layer's MLP
//!   projects, the MLP's first weights while the output projection does; none during an attention,
//!   whose working set fills most of data memory.

use crate::Module;
use crate::load::Named;
use crate::models::decoder::attention::qwen3::{self, QueryHeads, Size, fp8, launch};
use crate::models::decoder::attention::{Geometry, Head, KvHeads};
use crate::models::decoder::{Form, form::BlockScaled};
use crate::modules::attention::{Attention, NormalizedQkv, Positioned, Projection};
use crate::modules::generic::collection::View;
use crate::modules::generic::schedule::{Begin, Fill, Spill};
use crate::modules::language::{Backbone, Call};
use crate::modules::residual::Residual;
use crate::ops::attention::full::{head, paged::Retained};
use crate::ops::embedding;
use crate::ops::linear::output::fp8::Landed;
use crate::ops::linear::query::group::{Group, TilePair, Tiles};
use crate::ops::linear::query::{Tile, launch::Normalized};
use crate::ops::normalization::Blocked;
use crate::ops::normalization::span::{Owned, Span, Stage};
use crate::ops::pointwise::SwiGlu;
use crate::ops::swiglu::spans::fp8::Staging;
use crate::specialize::attention::Decode;
use crate::tensor::live::Live;
use crate::tensor::memory::Hbm;
use crate::tensor::parallel::Chip;
use crate::{Either, Sequential, Then};
use core::task::Poll;
use furiosa_opt_std::prelude::*;

mod affine;

/// The work between two attentions on one row tile: the end of `A`, then the start of `B`.
pub struct Tokenwise<A, B>(pub A, pub B);

/// A decoder run by its family's loops over its layers and its row tiles.
pub struct Decoder<M>(pub M);

/// Qwen3-8B, the size of the block-scaled FP8 line; the type aliases below default to it.
type S = Size<"8B">;
/// The layer of the Qwen3 size `Z` whose projections hold form `P`: block-scaled FP8, or BF16.
type Layer<P = BlockScaled, Z = S> = qwen3::Layer<Z, P, bf16, Hbm>;
/// A layer as the stack of `DEPTH` layers of its model gives it.
type Viewed<P, Z, const DEPTH: usize> = <Layer<P, Z> as View<DEPTH>>::Layer;
type Capacity<Z = S> = launch::Capacity<Z>;
type Reach<Z = S> = <Z as Decode>::Reach;
type Width<Z = S> = <Z as Geometry>::Hidden;
type Intermediate = <S as Geometry>::Intermediate;
type Groups<Z = S> = <Z as Geometry>::Groups;
type Queries<Z = S> = <Z as Geometry>::Queries;
/// A launch's rows of the hidden width in HBM: the embedded rows, a layer's input, the head's rows.
type Rows<Z = S> = HbmTensor<bf16, Chip, Blocked<Width<Z>, Capacity<Z>>>;
type Norm<Z = S> = <Z as Geometry>::Norm<Width<Z>, Hbm>;
/// The final norm, where the head starts.
type FinalNorm<Z = S> = Named<"norm", Norm<Z>>;
/// A launch's call as its layers read it: the context its rotary table prepares, its token IDs
/// and its live rows.
type Launched<'c, 'a, Z = S> = Call<
    'c,
    launch::Context<
        'a,
        Capacity<Z>,
        Reach<Z>,
        <Z as Decode>::Sequences,
        <Z as Decode>::Slots,
        Groups<Z>,
    >,
    launch::Ids<Z>,
    Capacity<Z>,
>;
/// A row tile of a launch, as the work between two attentions passes it on.
type RowTile<Z = S> = Live<Owned<Width<Z>, Span>, Capacity<Z>>;

/// A launch's query rows, grouped by the key and value head they attend with.
type QueryRows<Z = S> =
    HbmTensor<bf16, Chip, m![{ Symbol<Reach<Z>> }, KvHeads, { Symbol<Groups<Z>> }, Head]>;
/// A launch's rows of each key or value head.
type HeadRows<Z = S> = HbmTensor<bf16, Chip, m![{ Symbol<Reach<Z>> }, KvHeads, Head]>;

/// What a layer's attention stores of a launch's tiles: their queries, keys and values.
pub struct Stored<Z: Geometry + Decode = S> {
    queries: QueryRows<Z>,
    retained: (HeadRows<Z>, HeadRows<Z>),
}

/// The rows a layer's attention attends of a launch.
pub struct Attended<Z: Geometry + Decode = S>(QueryRows<Z>);

/// A layer's block-scaled output projection, its weights landed while its attention's last
/// history was in flight.
type Projecting = Landed<Width, Queries, Span>;

/// A tile of a layer's attention output: the attended rows and the output projection over them,
/// `L` as its form lands it.
type Output<'p, Z = S, L = Projecting> = (Live<&'p Attended<Z>, Capacity<Z>>, &'p mut L);

/// One layer's places of a launch: its input rows, what its attention stores and what it attends.
struct Places<Z: Geometry + Decode = S> {
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

/// A layer's query, key and value projections, in block-scaled FP8.
type QueryProjection = <BlockScaled as Form>::Query<Width, QueryHeads<S>, Head, Hbm>;
type KeyProjection = <BlockScaled as Form>::Key<Width, KvHeads, Head, Hbm>;
type ValueProjection = <BlockScaled as Form>::Value<Width, KvHeads, Head, Hbm>;
/// The norm of each of a layer's query and key heads.
type HeadNorm = head::Stored<Head, { <S as Geometry>::EPSILON }>;

/// What a leaf `L` gives when begun for a tile of the launch's rows.
type Begun<L> = <L as Module<Begin<()>, Live<(), Capacity>, &'static mut ()>>::Output;

/// The forms a layer's input norm, projections and head norms take for a tile's rows, landed: the
/// norm's gain, the value form, the launch of the tile's rows it stages, the key and query forms
/// and the two head norms' forms.
struct Front {
    gain: <Stage as crate::Op<&'static <S as Geometry>::Norm<Width, Hbm>>>::Output,
    value: Begun<ValueProjection>,
    launch: Begun<Begun<ValueProjection>>,
    key: Begun<KeyProjection>,
    query: Begun<QueryProjection>,
    query_norm: Begun<HeadNorm>,
    key_norm: Begun<HeadNorm>,
}

/// A tile between its MLP's two steps: its product staged for the down projection, and the rows
/// of the attention residual, which the MLP residual adds to.
struct Staged {
    product: Staging<Width, Intermediate, Span>,
    attended: Live<Owned<Width, Span>, Capacity>,
}

/// `layer`'s front landed for the rows of `tile`.
fn front(device: &mut Device, layer: &Layer, tile: Live<(), Capacity>) -> Front {
    let Sequential(Residual(norm, attention, _), _) = layer;
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

/// A tile's input rows `rows` through `layer` up to its attention's store, its `front` landed: the
/// input norm, then every period's queries, keys and values, head-normed, rotated and stored in
/// `stored`; the rows, for the residual.
fn start(
    device: &mut Device,
    call: Launched<'_, '_>,
    layer: &Layer,
    front: Front,
    rows: Live<Owned<Width, Span>, Capacity>,
    stored: &mut Stored,
) -> Live<Owned<Width, Span>, Capacity> {
    let Sequential(Residual(_, attention, _), _) = layer;
    let Attention {
        qkv: Positioned { positions, .. },
        cache: stage,
        ..
    } = &attention.0;
    let Front {
        gain,
        value,
        mut launch,
        key,
        query,
        query_norm,
        key_norm,
    } = front;
    let Stored { queries, retained } = stored;
    let tile = rows.with(());
    let normalized = gain.forward(&mut *device, (), &mut (), &rows);
    // The rows the periods read lie apart from the launch, which the period loops carry.
    let mut home = Normalized::new();
    home.spill(&mut *device, normalized.value(), (tile, tile.rows()));
    let rotation = &call.context.0.positions;
    let mut project = |device: &mut Device, period: Live<Either<_, _>, Capacity>| {
        let at = period.with(());
        let rows = period.value();
        let projected = (
            query.forward(&mut *device, tile, &mut (), rows),
            key.forward(&mut *device, tile, &mut (), rows),
            value.forward(&mut *device, tile, &mut (), rows),
        );
        let normalized = (
            query_norm.forward(&mut *device, tile, &mut (), projected.0),
            key_norm.forward(&mut *device, tile, &mut (), projected.1),
        );
        let (query, key) =
            positions.forward(&mut *device, tile, &mut (), (normalized, &(rotation, at)));
        let stored = at.with((query, key, projected.2));
        let places = (&mut *queries, &mut *retained);
        stage.forward(device, (), &mut &mut (), (places, stored));
    };
    type Pair = <TilePair as Group>::Period;
    let pairs = tile.count::<Tile>() / 2;
    let most = <Capacity>::SIZE / Pair::SIZE;
    for _ in 0..(pairs + most - 1) / most {
        let begin = Begin(launch.period::<TilePair>(&home, 0));
        value.forward(&mut *device, tile, &mut (), begin);
    }
    for at in 0..pairs {
        let next = launch.period::<TilePair>(&home, at);
        let period = value.forward(&mut *device, tile, &mut (), next);
        project(
            &mut *device,
            tile.tile::<Pair>(at).with(Either::Left(period)),
        );
    }
    for _ in 0..tile.count::<Tile>() % 2 {
        let next = launch.period::<Tiles<1>>(&home, pairs);
        let period = value.forward(&mut *device, tile, &mut (), next);
        let single = tile.tile::<Tile>(2 * pairs);
        project(&mut *device, single.with(Either::Right(period)));
    }
    rows
}

/// A tile through `layer` from its attention to its MLP's second step, given a tile of its
/// attention output, the `attended` rows and the `landed` output projection, and `input` of its
/// input rows: the output projection, the attention residual and the MLP's first step on its norm.
fn end(
    device: &mut Device,
    layer: &Layer,
    (attended, landed): Output<'_>,
    input: Live<Owned<Width, Span>, Capacity>,
) -> Staged {
    let Sequential(Residual(_, attention, sum), Residual(norm, mlp, _)) = layer;
    let Attention { output, .. } = &attention.0;
    let (gate_up, down) = mlp.0.parts();
    let tile = input.with(());
    let gain = crate::Op::forward(&Stage, &mut *device, &norm.0);
    let ahead = gate_up.forward(&mut *device, (), &mut (), Begin(()));
    let mut step = |device: &mut Device| {
        output.forward(
            device,
            (),
            &mut (),
            (&mut *landed, &attended.value().0, tile),
        )
    };
    let _ = step(&mut *device);
    let _ = step(&mut *device);
    let projected = match step(&mut *device) {
        Poll::Ready(rows) => tile.with(rows),
        Poll::Pending => tile.with(Owned::new()),
    };
    let attended = sum.forward(&mut *device, (), &mut (), (&input, &projected));
    let normalized = gain.forward(&mut *device, (), &mut (), &attended);
    let rows = tile.with(normalized.value());
    let projected = gate_up.forward(&mut *device, (), &mut (), (ahead, rows));
    let product = SwiGlu.forward(&mut *device, (), &mut (), tile.with(projected));
    let staged = down.forward(&mut *device, (), &mut (), Begin(tile.with(product)));
    Staged {
        product: staged,
        attended,
    }
}

/// A `staged` tile through the second step of `layer`'s MLP, the down projection, and the MLP
/// residual; the rows.
fn update(
    device: &mut Device,
    layer: &Layer,
    staged: Staged,
) -> Live<Owned<Width, Span>, Capacity> {
    let Sequential(_, Residual(_, mlp, sum)) = layer;
    let (_, down) = mlp.0.parts();
    let Staged { product, attended } = staged;
    let tile = attended.with(());
    let fed = tile.with(down.forward(&mut *device, (), &mut (), tile.with(product)));
    sum.forward(device, (), &mut (), (&attended, &fed))
}

/// A tile of the embedded rows through the first layer up to its attention's store; the tile's
/// rows, the first layer's input.
impl<'l, 'c, 'a, 's> Module<Live<(), Capacity>, Launched<'c, 'a>, &'s mut Stored>
    for Tokenwise<&'l Rows, &'l Layer>
{
    type Output = Live<Owned<Width, Span>, Capacity>;

    fn forward(
        &self,
        device: &mut Device,
        call: Launched<'c, 'a>,
        stored: &'s mut Stored,
        tile: Live<(), Capacity>,
    ) -> Self::Output {
        let Self(embedded, layer) = *self;
        let front = front(device, layer, tile);
        let rows = tile.with(embedded.fill(&mut *device, tile));
        start(device, call, layer, front, rows, stored)
    }
}

/// A tile through a layer from its attention and the next layer up to its attention's store; the
/// next layer's input rows. The next layer's front lands while this one's MLP projects.
impl<'l, 'c, 'a, 'p, 's>
    Module<(Output<'p>, Live<Owned<Width, Span>, Capacity>), Launched<'c, 'a>, &'s mut Stored>
    for Tokenwise<&'l Layer, &'l Layer>
{
    type Output = Live<Owned<Width, Span>, Capacity>;

    fn forward(
        &self,
        device: &mut Device,
        call: Launched<'c, 'a>,
        stored: &'s mut Stored,
        (attended, input): (Output<'p>, Live<Owned<Width, Span>, Capacity>),
    ) -> Self::Output {
        let Self(layer, next) = *self;
        let tile = input.with(());
        let staged = end(device, layer, attended, input);
        let front = front(device, next, tile);
        let rows = update(device, layer, staged);
        start(device, call, next, front, rows, stored)
    }
}

/// A tile through the last layer from its attention and the final norm; the head's rows.
impl<'l, 'c, 'a, 'p, 's>
    Module<(Output<'p>, Live<Owned<Width, Span>, Capacity>), Launched<'c, 'a>, &'s mut ()>
    for Tokenwise<&'l Layer, &'l FinalNorm>
{
    type Output = Live<Owned<Width, Span>, Capacity>;

    fn forward(
        &self,
        device: &mut Device,
        _: Launched<'c, 'a>,
        _: &'s mut (),
        (attended, input): (Output<'p>, Live<Owned<Width, Span>, Capacity>),
    ) -> Self::Output {
        let Self(layer, norm) = *self;
        let staged = end(device, layer, attended, input);
        let rows = update(device, layer, staged);
        norm.forward(device, (), &mut (), &rows)
    }
}

/// A launch's rows through the catalog tree's leaves in the family's order: embedded, the first
/// layer's stores over every tile, then each layer attended and, tile by tile, its work up to the
/// next layer's store or the head, from whose rows the selected rows' tokens are read. Generic in
/// its env, so only the launches that run it compile it.
impl<'m, 'i, 'a, 's, 'b, Env> Module<&'i mut launch::Input<'a, S>, Env, launch::State<'s, 'b, S>>
    for Decoder<&'m launch::Model<S, BlockScaled>>
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        _: Env,
        state: launch::State<'s, 'b, S>,
        input: &'i mut launch::Input<'a, S>,
    ) {
        const LAYERS: usize = <S as Geometry>::LAYERS;
        let causal = self.0;
        let live = Live::<(), Capacity>::read(&input.count);
        let context =
            causal
                .model
                .rotary_emb
                .forward(&mut *device, (), &mut (), (live, &mut input.cache));
        let call = Call {
            context: &context,
            tokens: &input.tokens,
            live,
        };
        let (((embedded, slots), ()), tokens) = state;
        let Then(Backbone { embedding, body }, norm) = &causal.model.body;
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
                      layer: &Layer,
                      slot: &mut launch::Layer<'b>,
                      places: &mut Places| {
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
            let mut staged = crate::ops::linear::output::fp8::Stored::<Width, Queries>::empty();
            let boundary = (&output.0, &mut staged);
            let rows = (rows.0, rows.1, rows.2, boundary, &mut places.attended.0);
            attend.forward(&mut *device, (), &mut (), rows);
            let rows = (deferred, &places.stored.retained, &context.write);
            stage.forward(&mut *device, (), &mut *pool, rows);
            output.forward(device, (), &mut (), Begin(staged))
        };
        let mut places = Places::new();
        let first = layers.layer(0);
        for span in 0..spans {
            let tile = live.tile::<Span>(span);
            let tokenwise = Tokenwise(&*embedded, &first);
            let rows = tokenwise.forward(&mut *device, call, &mut places.stored, tile);
            places.input.spill(&mut *device, rows.value(), tile);
        }
        for index in 0..LAYERS - 1 {
            let (layer, next) = (layers.layer(index), layers.layer(index + 1));
            let mut projecting = attend(&mut *device, &layer, &mut slots[index], &mut places);
            let mut ahead = Places::new();
            for span in 0..spans {
                let tile = live.tile::<Span>(span);
                let input = tile.with(places.input.fill(&mut *device, tile));
                let tokenwise = Tokenwise(&layer, &next);
                let output = (tile.with(&places.attended), &mut projecting);
                let rows =
                    tokenwise.forward(&mut *device, call, &mut ahead.stored, (output, input));
                ahead.input.spill(&mut *device, rows.value(), tile);
            }
            places = ahead;
        }
        let last = layers.layer(LAYERS - 1);
        let mut projecting = attend(&mut *device, &last, &mut slots[LAYERS - 1], &mut places);
        let mut head: Rows = HbmTensor::new();
        for span in 0..spans {
            let tile = live.tile::<Span>(span);
            let input = tile.with(places.input.fill(&mut *device, tile));
            let output = (tile.with(&places.attended), &mut projecting);
            let rows = Tokenwise(&last, norm).forward(&mut *device, call, &mut (), (output, input));
            head.spill(&mut *device, rows.value(), tile);
        }
        causal.lm_head.read(device, &head, call.context.1, tokens);
    }
}

/// Block-scaled FP8 Qwen3-8B's launch runs its catalog model by the decoder family's loops.
impl<'i, 'a, 's, 'b, Env> Module<&'i mut launch::Input<'a, S>, Env, launch::State<'s, 'b, S>>
    for fp8::ForCausalLM
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        state: launch::State<'s, 'b, S>,
        input: &'i mut launch::Input<'a, S>,
    ) {
        Decoder(&self.model).forward(device, env, state, input)
    }
}
