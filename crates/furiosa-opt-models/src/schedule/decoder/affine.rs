//! The cut of a pre-norm layer at its attention in BF16, at every size of the families whose
//! catalog layer it is: Qwen3, Llama and Mistral.
//!
//! - The dense forms land nothing ahead: each contraction reads its weight slabs where they are
//!   stored. The output projection stages nothing at its attention's boundary and gives a tile's
//!   rows in its first step.
//! - A size's front names the norm of its query and key heads. Llama's and Mistral's have none,
//!   `()`, which leaves the heads as projected.

use super::{
    Capacity, Decoder, FinalNorm, Launched, Norm, Output, Places, RowTile, Rows, Stored, Tokenwise,
    Viewed, Width,
};
use crate::models::decoder::attention::qwen3::{self, QueryHeads, launch};
use crate::models::decoder::attention::{Geometry, Head, KvHeads};
use crate::models::decoder::{Form, form::Affine};
use crate::modules::attention::{Attention, NormalizedQkv, Positioned, Projection};
use crate::modules::generic::collection::View;
use crate::modules::generic::schedule::{Begin, Fill, Spill};
use crate::modules::language::{Backbone, Call};
use crate::modules::residual::Residual;
use crate::ops::attention::full::paged::Retained;
use crate::ops::embedding;
use crate::ops::linear::query::group::{Group, TilePair, Tiles};
use crate::ops::linear::query::{Tile, launch::Normalized, matrix};
use crate::ops::normalization::RmsNorm;
use crate::ops::normalization::span::{Owned, Span, Stage, Tiled};
use crate::ops::pointwise::SwiGlu;
use crate::ops::swiglu;
use crate::specialize::attention::Decode;
use crate::tensor::axis;
use crate::tensor::live::Live;
use crate::tensor::memory::Hbm;
use crate::tensor::parallel::Chip;
use crate::{Either, Module, Sequential, Then};
use core::task::Poll;
use furiosa_opt_std::prelude::*;

/// The BF16 layer of the size `Z`.
type Layer<Z> = super::Layer<Affine, Z>;
/// What a leaf `L` gives when begun for a tile of a launch of the size `Z`.
type Begun<L, Z> = <L as Module<Begin<()>, Live<(), Capacity<Z>>, &'static mut ()>>::Output;
type QueryProjection<Z> = <Affine as Form>::Query<Width<Z>, QueryHeads<Z>, Head, Hbm>;
type KeyProjection<Z> = <Affine as Form>::Key<Width<Z>, KvHeads, Head, Hbm>;
type ValueProjection<Z> = <Affine as Form>::Value<Width<Z>, KvHeads, Head, Hbm>;
/// The norms of a layer's query and key heads, as its size's front names them.
type QueryNorm<Z> = qwen3::QueryNorm<Z, Hbm>;
type KeyNorm<Z> = qwen3::KeyNorm<Z, Hbm>;
/// A period's rows, of a pair of tiles or of a single tile, as the dense projections read them.
type Period<Z> = Either<
    matrix::Period<Width<Z>, TilePair, Capacity<Z>>,
    matrix::Period<Width<Z>, Tiles<1>, Capacity<Z>>,
>;
/// The heads a begun projection `L` gives of a period.
type Heads<L, Z> =
    <L as Module<&'static Period<Z>, Live<(), Capacity<Z>>, &'static mut ()>>::Output;
/// A tile's product of its MLP's gate and up projections, staged for the down projection.
type Product<Z> = HbmTensor<bf16, Chip, m![Span, { Symbol<<Z as Geometry>::Intermediate> }]>;

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

/// A layer's front of the size `Z` landed, its head norms begun as `QN` and `KN`.
type Landed<Z, QN, KN> = Front<
    <Stage as crate::Op<&'static Norm<Z>>>::Output,
    Begun<ValueProjection<Z>, Z>,
    Begun<Begun<ValueProjection<Z>, Z>, Z>,
    Begun<KeyProjection<Z>, Z>,
    Begun<QueryProjection<Z>, Z>,
    QN,
    KN,
>;

/// A tile between its MLP's two steps: its product staged for the down projection, and the rows
/// of the attention residual, which the MLP residual adds to.
struct Staged<Z: Geometry + Decode> {
    product: Product<Z>,
    attended: RowTile<Z>,
}

/// `layer`'s front landed for the rows of `tile`.
fn front<Z, H, const E: usize, const C: usize, const L: usize, QN, KN>(
    device: &mut Device,
    layer: &Layer<Z>,
    tile: Live<(), Capacity<Z>>,
) -> Landed<Z, QN, KN>
where
    Z: Geometry<KvHeads = KvHeads, Head = Head, Hidden = H, Norm<H, Hbm> = RmsNorm<H, E, Chip>>
        + Decode
        + qwen3::Front
        + 'static,
    H: Tiled<Capacity = axis::Rows<C>, Last = axis::Rows<L>>,
    swiglu::Down<<Z as Geometry>::Intermediate>: AxisName,
    QueryNorm<Z>: for<'x> Module<Begin<()>, Live<(), Capacity<Z>>, &'x mut (), Output = QN>,
    KeyNorm<Z>: for<'x> Module<Begin<()>, Live<(), Capacity<Z>>, &'x mut (), Output = KN>,
{
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
fn start<Z, H, const E: usize, const C: usize, const L: usize, QN, KN>(
    device: &mut Device,
    call: Launched<'_, '_, Z>,
    layer: &Layer<Z>,
    front: Landed<Z, QN, KN>,
    rows: RowTile<Z>,
    stored: &mut Stored<Z>,
) -> RowTile<Z>
where
    Z: Geometry<KvHeads = KvHeads, Head = Head, Hidden = H, Norm<H, Hbm> = RmsNorm<H, E, Chip>>
        + Decode
        + qwen3::Front
        + 'static,
    H: Tiled<Capacity = axis::Rows<C>, Last = axis::Rows<L>>,
    swiglu::Down<<Z as Geometry>::Intermediate>: AxisName,
    QN: for<'x> Module<
            Heads<Begun<QueryProjection<Z>, Z>, Z>,
            Live<(), Capacity<Z>>,
            &'x mut (),
            Output = Heads<Begun<QueryProjection<Z>, Z>, Z>,
        >,
    KN: for<'x> Module<
            Heads<Begun<KeyProjection<Z>, Z>, Z>,
            Live<(), Capacity<Z>>,
            &'x mut (),
            Output = Heads<Begun<KeyProjection<Z>, Z>, Z>,
        >,
{
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
    let mut project = |device: &mut Device, period: Live<Period<Z>, Capacity<Z>>| {
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
    let most = <Capacity<Z>>::SIZE / Pair::SIZE;
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

/// A tile through `layer` from its attention to its MLP's second step, given a tile `output` of
/// its attention output and `input` of its input rows: the output projection, the attention
/// residual and the MLP's first step on its norm.
fn end<Z, H, const E: usize, const C: usize, const L: usize>(
    device: &mut Device,
    layer: &Layer<Z>,
    (attended, landed): Output<'_, Z, ()>,
    input: RowTile<Z>,
) -> Staged<Z>
where
    Z: Geometry<KvHeads = KvHeads, Head = Head, Hidden = H, Norm<H, Hbm> = RmsNorm<H, E, Chip>>
        + Decode
        + qwen3::Front
        + 'static,
    H: Tiled<Capacity = axis::Rows<C>, Last = axis::Rows<L>>,
    swiglu::Down<<Z as Geometry>::Intermediate>: AxisName,
{
    let Sequential(Residual(_, attention, sum), Residual(norm, mlp, _)) = layer;
    let Attention { output, .. } = &attention.0;
    let (gate_up, down) = mlp.0.parts();
    let tile = input.with(());
    let gain = crate::Op::forward(&Stage, &mut *device, &norm.0);
    let ahead = gate_up.forward(&mut *device, (), &mut (), Begin(()));
    let projected = match output.forward(
        &mut *device,
        (),
        &mut (),
        (&mut *landed, &attended.value().0, tile),
    ) {
        Poll::Ready(rows) => tile.with(rows),
        Poll::Pending => tile.with(Owned::new()),
    };
    let attended = sum.forward(&mut *device, (), &mut (), (&input, &projected));
    let normalized = gain.forward(&mut *device, (), &mut (), &attended);
    let rows = tile.with(normalized.value());
    let projected = gate_up.forward(&mut *device, (), &mut (), (ahead, rows));
    let product = SwiGlu.forward(&mut *device, (), &mut (), tile.with(projected));
    let product = down.forward(&mut *device, (), &mut (), Begin(tile.with(product)));
    Staged { product, attended }
}

/// A `staged` tile through the second step of `layer`'s MLP, the down projection, and the MLP
/// residual; the rows.
fn update<Z, H, const E: usize, const C: usize, const L: usize>(
    device: &mut Device,
    layer: &Layer<Z>,
    staged: Staged<Z>,
) -> RowTile<Z>
where
    Z: Geometry<KvHeads = KvHeads, Head = Head, Hidden = H, Norm<H, Hbm> = RmsNorm<H, E, Chip>>
        + Decode
        + qwen3::Front
        + 'static,
    H: Tiled<Capacity = axis::Rows<C>, Last = axis::Rows<L>>,
    swiglu::Down<<Z as Geometry>::Intermediate>: AxisName,
{
    let Sequential(_, Residual(_, mlp, sum)) = layer;
    let (_, down) = mlp.0.parts();
    let Staged { product, attended } = staged;
    let tile = attended.with(());
    let fed = tile.with(down.forward(&mut *device, (), &mut (), tile.with(product)));
    sum.forward(device, (), &mut (), (&attended, &fed))
}

/// A tile of the embedded rows through the first layer up to its attention's store.
impl<'l, 'c, 'a, 's, Z, H, const E: usize, const C: usize, const L: usize, QN, KN>
    Module<Live<(), Capacity<Z>>, Launched<'c, 'a, Z>, &'s mut Stored<Z>>
    for Tokenwise<&'l Rows<Z>, &'l Layer<Z>>
where
    Z: Geometry<KvHeads = KvHeads, Head = Head, Hidden = H, Norm<H, Hbm> = RmsNorm<H, E, Chip>>
        + Decode
        + qwen3::Front
        + 'static,
    H: Tiled<Capacity = axis::Rows<C>, Last = axis::Rows<L>>,
    swiglu::Down<<Z as Geometry>::Intermediate>: AxisName,
    QueryNorm<Z>: for<'x> Module<Begin<()>, Live<(), Capacity<Z>>, &'x mut (), Output = QN>,
    KeyNorm<Z>: for<'x> Module<Begin<()>, Live<(), Capacity<Z>>, &'x mut (), Output = KN>,
    QN: for<'x> Module<
            Heads<Begun<QueryProjection<Z>, Z>, Z>,
            Live<(), Capacity<Z>>,
            &'x mut (),
            Output = Heads<Begun<QueryProjection<Z>, Z>, Z>,
        >,
    KN: for<'x> Module<
            Heads<Begun<KeyProjection<Z>, Z>, Z>,
            Live<(), Capacity<Z>>,
            &'x mut (),
            Output = Heads<Begun<KeyProjection<Z>, Z>, Z>,
        >,
{
    type Output = RowTile<Z>;

    fn forward(
        &self,
        device: &mut Device,
        call: Launched<'c, 'a, Z>,
        stored: &'s mut Stored<Z>,
        tile: Live<(), Capacity<Z>>,
    ) -> Self::Output {
        let Self(embedded, layer) = *self;
        let front = front(device, layer, tile);
        let rows = tile.with(embedded.fill(&mut *device, tile));
        start(device, call, layer, front, rows, stored)
    }
}

/// A tile through a layer from its attention and the next layer up to its attention's store; the
/// next layer's input rows.
impl<'l, 'c, 'a, 'p, 's, Z, H, const E: usize, const C: usize, const L: usize, QN, KN>
    Module<(Output<'p, Z, ()>, RowTile<Z>), Launched<'c, 'a, Z>, &'s mut Stored<Z>>
    for Tokenwise<&'l Layer<Z>, &'l Layer<Z>>
where
    Z: Geometry<KvHeads = KvHeads, Head = Head, Hidden = H, Norm<H, Hbm> = RmsNorm<H, E, Chip>>
        + Decode
        + qwen3::Front
        + 'static,
    H: Tiled<Capacity = axis::Rows<C>, Last = axis::Rows<L>>,
    swiglu::Down<<Z as Geometry>::Intermediate>: AxisName,
    QueryNorm<Z>: for<'x> Module<Begin<()>, Live<(), Capacity<Z>>, &'x mut (), Output = QN>,
    KeyNorm<Z>: for<'x> Module<Begin<()>, Live<(), Capacity<Z>>, &'x mut (), Output = KN>,
    QN: for<'x> Module<
            Heads<Begun<QueryProjection<Z>, Z>, Z>,
            Live<(), Capacity<Z>>,
            &'x mut (),
            Output = Heads<Begun<QueryProjection<Z>, Z>, Z>,
        >,
    KN: for<'x> Module<
            Heads<Begun<KeyProjection<Z>, Z>, Z>,
            Live<(), Capacity<Z>>,
            &'x mut (),
            Output = Heads<Begun<KeyProjection<Z>, Z>, Z>,
        >,
{
    type Output = RowTile<Z>;

    fn forward(
        &self,
        device: &mut Device,
        call: Launched<'c, 'a, Z>,
        stored: &'s mut Stored<Z>,
        (output, input): (Output<'p, Z, ()>, RowTile<Z>),
    ) -> Self::Output {
        let Self(layer, next) = *self;
        let tile = input.with(());
        let staged = end(device, layer, output, input);
        let front = front(device, next, tile);
        let rows = update(device, layer, staged);
        start(device, call, next, front, rows, stored)
    }
}

/// A tile through the last layer from its attention and the final norm; the head's rows.
impl<'l, 'c, 'a, 'p, 's, Z, H, const E: usize, const C: usize, const L: usize>
    Module<(Output<'p, Z, ()>, RowTile<Z>), Launched<'c, 'a, Z>, &'s mut ()>
    for Tokenwise<&'l Layer<Z>, &'l FinalNorm<Z>>
where
    Z: Geometry<KvHeads = KvHeads, Head = Head, Hidden = H, Norm<H, Hbm> = RmsNorm<H, E, Chip>>
        + Decode
        + qwen3::Front
        + 'static,
    H: Tiled<Capacity = axis::Rows<C>, Last = axis::Rows<L>>,
    swiglu::Down<<Z as Geometry>::Intermediate>: AxisName,
{
    type Output = RowTile<Z>;

    fn forward(
        &self,
        device: &mut Device,
        _: Launched<'c, 'a, Z>,
        _: &'s mut (),
        (output, input): (Output<'p, Z, ()>, RowTile<Z>),
    ) -> Self::Output {
        let Self(layer, norm) = *self;
        let staged = end(device, layer, output, input);
        let rows = update(device, layer, staged);
        norm.forward(device, (), &mut (), &rows)
    }
}

/// A BF16 launch through its catalog model's leaves in the family's order, at any size of a family
/// whose catalog layer this is: embedded, the first layer's stores over every tile, then each layer
/// attended and, tile by tile, its work up to the next layer's store or the head, from whose rows
/// the selected rows' tokens are read. Generic in its env, so only the launches that run it
/// compile it.
impl<
    'm,
    'i,
    'a,
    's,
    'b,
    Z,
    const DEPTH: usize,
    H,
    const E: usize,
    const C: usize,
    const L: usize,
    Env,
> Module<&'i mut launch::Input<'a, Z>, Env, launch::State<'s, 'b, Z>>
    for Decoder<&'m launch::Model<Z, Affine>>
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
    <Z as Geometry>::Layers<launch::Layer<'b>>: AsMut<[launch::Layer<'b>]>,
    swiglu::Down<<Z as Geometry>::Intermediate>: AxisName,
    QueryNorm<Z>: View<DEPTH>,
    KeyNorm<Z>: View<DEPTH>,
    for<'l, 'c, 'x, 'y> Tokenwise<&'l Rows<Z>, &'l Viewed<Affine, Z, DEPTH>>:
        Module<Live<(), Capacity<Z>>, Launched<'c, 'x, Z>, &'y mut Stored<Z>, Output = RowTile<Z>>,
    for<'l, 'c, 'x, 'p, 'y> Tokenwise<&'l Viewed<Affine, Z, DEPTH>, &'l Viewed<Affine, Z, DEPTH>>:
        Module<
                (Output<'p, Z, ()>, RowTile<Z>),
                Launched<'c, 'x, Z>,
                &'y mut Stored<Z>,
                Output = RowTile<Z>,
            >,
    for<'l, 'c, 'x, 'p, 'y> Tokenwise<&'l Viewed<Affine, Z, DEPTH>, &'l FinalNorm<Z>>: Module<
            (Output<'p, Z, ()>, RowTile<Z>),
            Launched<'c, 'x, Z>,
            &'y mut (),
            Output = RowTile<Z>,
        >,
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        _: Env,
        state: launch::State<'s, 'b, Z>,
        input: &'i mut launch::Input<'a, Z>,
    ) {
        let causal = self.0;
        let live = Live::<(), Capacity<Z>>::read(&input.count);
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
        let slots = slots.as_mut();
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
                      layer: &Viewed<Affine, Z, DEPTH>,
                      slot: &mut launch::Layer<'_>,
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
            // The dense output projection stages nothing at the boundary.
            let mut staged = ();
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
        for index in 0..DEPTH - 1 {
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
        let last = layers.layer(DEPTH - 1);
        let mut projecting = attend(&mut *device, &last, &mut slots[DEPTH - 1], &mut places);
        let mut head: Rows<Z> = HbmTensor::new();
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

/// A BF16 launch runs its catalog model by the decoder family's loops.
impl<'i, 'a, 's, 'b, Z, const DEPTH: usize, H, const E: usize, const C: usize, const L: usize, Env>
    Module<&'i mut launch::Input<'a, Z>, Env, launch::State<'s, 'b, Z>> for launch::Model<Z, Affine>
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
    swiglu::Down<<Z as Geometry>::Intermediate>: AxisName,
    QueryNorm<Z>: View<DEPTH>,
    KeyNorm<Z>: View<DEPTH>,
    for<'m> Decoder<&'m Self>:
        Module<&'i mut launch::Input<'a, Z>, Env, launch::State<'s, 'b, Z>, Output = ()>,
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        state: launch::State<'s, 'b, Z>,
        input: &'i mut launch::Input<'a, Z>,
    ) {
        Decoder(self).forward(device, env, state, input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::decoder::attention::{llama, qwen3::Size};

    /// Each BF16 catalog launch of Qwen3, Llama and Mistral runs by the decoder family's loops on
    /// its launch's input and state.
    #[test]
    fn every_size_runs_by_the_loops() {
        fn runs<Z>()
        where
            for<'m, 'i, 'a, 's, 'b> Decoder<&'m launch::Model<Z, Affine>>:
                Module<&'i mut launch::Input<'a, Z>, (), launch::State<'s, 'b, Z>>,
        {
        }
        runs::<Size<"0.6B">>();
        runs::<Size<"1.7B">>();
        runs::<Size<"4B">>();
        runs::<Size<"8B">>();
        runs::<llama::Size<"8B">>();
        runs::<llama::Size<"Mistral-7B">>();
    }
}
