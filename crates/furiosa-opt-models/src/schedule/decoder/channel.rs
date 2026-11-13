//! The channel-scaled Llama front and MLP in the decoder's attention-to-attention schedule.
//! QKV consumes one shared activation operand; native down panels retain the 32-feature tail.

use super::{Decoder, Tokenwise};
use crate::models::decoder::attention::{
    Geometry, Head, KvHeads,
    llama::fp8,
    qwen3::{QueryHeads, launch},
};
use crate::models::decoder::form::TokenScaled;
use crate::modules::attention::{Attention, Positioned, Projection};
use crate::modules::generic::schedule::{Begin, Fill, Spill};
use crate::modules::language::{Backbone, Call};
use crate::modules::residual::Residual;
use crate::ops::attention::full::paged::Retained;
use crate::ops::embedding;
use crate::ops::linear::query::{
    self, Tile,
    fp8::channel as qkv,
    group::{Group, TilePair, Tiles},
    launch::Normalized,
};
use crate::ops::normalization::span::{Owned, Span, Stage};
use crate::ops::swiglu::spans::fp8::channel as mlp;
use crate::tensor::live::Live;
use crate::{Either, Module, Sequential, Then};
use core::{marker::PhantomData, task::Poll};
use furiosa_opt_std::prelude::*;

type S = fp8::Size;
type Width = super::Width<S>;
type Queries = super::Queries<S>;
type Intermediate = <S as Geometry>::Intermediate;
type Capacity = super::Capacity<S>;
type Layer = super::Layer<TokenScaled, S>;
type Rows = super::Rows<S>;
type Stored = super::Stored<S>;
type Places = super::Places<S>;
type FinalNorm = super::FinalNorm<S>;
type Launched<'c, 'a> = super::Launched<'c, 'a, S>;
type Projecting = crate::ops::linear::output::fp8::Landed<Width, Queries, Span, f32>;
type Output<'p> = super::Output<'p, S, Projecting>;

struct Front {
    gain: <Stage as crate::Op<&'static <S as Geometry>::Norm<Width, crate::tensor::memory::Hbm>>>::Output,
    value: qkv::Prepared<Width, KvHeads, Head>,
    key: qkv::Prepared<Width, KvHeads, Head>,
    query: qkv::Prepared<Width, QueryHeads<S>, Head>,
}
struct Staged {
    product: mlp::Staging<Width, Intermediate, Span>,
    attended: Live<Owned<Width, Span>, Capacity>,
}

fn front(device: &mut Device, layer: &Layer, _: Live<(), Capacity>) -> Front {
    let Sequential(Residual(norm, attention, _), _) = layer;
    let Projection { query, key, value } = &attention.0.qkv.projection.projection;
    let value = qkv::prepare(device, &value.0.weight);
    let key = qkv::prepare(device, &key.0.weight);
    let query = qkv::prepare(device, &query.0.weight);
    let gain = crate::Op::forward(&Stage, device, &norm.0);
    Front {
        gain,
        value,
        key,
        query,
    }
}

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
    let tile = rows.with(());
    let normalized = front.gain.forward(device, (), &mut (), &rows);
    let mut home = Normalized::<Width, Capacity>::new();
    home.spill(device, normalized.value(), (tile, tile.rows()));
    let rotation = &call.context.0.positions;
    let mut store = |device: &mut Device, at, query, key, value| {
        let (query, key) =
            positions.forward(device, tile, &mut (), ((query, key), &(rotation, at)));
        stage.forward(
            device,
            (),
            &mut &mut (),
            (
                (&mut stored.queries, &mut stored.retained),
                at.with((query, key, value)),
            ),
        );
    };
    let pairs = tile.count::<Tile>() / 2;
    if pairs > 0 {
        let staged = qkv::Staged::new(device, &home, tile.start() + tile.rows());
        let first = tile.start() / 256;
        let mut current = staged.pair(device, first);
        for pair in 0..pairs {
            // DMA the next pair before this contraction; both use the same resident QKV weights.
            let next = staged.pair(device, (first + pair + 1) % (Capacity::SIZE / 256));
            let operand = current.fill(device);
            let q: query::Queries<TilePair, QueryHeads<S>, Head> =
                current.project(device, &front.query, &operand);
            let k: query::Pairs<TilePair, KvHeads, Head> =
                current.project(device, &front.key, &operand);
            let v: query::Pairs<TilePair, KvHeads, Head> =
                current.project(device, &front.value, &operand);
            let at = tile.tile::<<TilePair as Group>::Period>(pair);
            store(
                device,
                at,
                Either::Left(q),
                Either::Left(k),
                Either::Left(v),
            );
            current = next;
        }
    }
    if tile.count::<Tile>() % 2 != 0 {
        let at = tile.tile::<Tile>(2 * pairs);
        let groups = at.rows().div_ceil(8);
        let input = crate::Op::forward(
            &query::fp8::ToDm::<Tiles<1>>(PhantomData),
            device,
            (&home, (at.start(), Tile::SIZE), groups),
        );
        let period = qkv::Period::quantize(device, &input, groups);
        let operand = period.fill(device);
        let q: query::Queries<Tiles<1>, QueryHeads<S>, Head> =
            period.project(device, &front.query, &operand);
        let k: query::Pairs<Tiles<1>, KvHeads, Head> = period.project(device, &front.key, &operand);
        let v: query::Pairs<Tiles<1>, KvHeads, Head> =
            period.project(device, &front.value, &operand);
        store(
            device,
            at,
            Either::Right(q),
            Either::Right(k),
            Either::Right(v),
        );
    }
    rows
}

/// A tile through `layer` from its attention to its MLP's second step, given a tile of its
/// attention output, the `attended` rows and the `landed` output projection, and `input` of its
/// input rows: the output projection, the attention residual and the MLP's first step on its norm.
fn end(
    device: &mut Device,
    call: Launched<'_, '_>,
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
    // Only pure item launches retain product scales in DM. Prompt and mixed launches keep
    // the HBM path, even for short spans; row count alone does not identify decode work.
    let decode = call.context.0.read.items().rows() == call.live.rows();
    let staged = down.forward(
        &mut *device,
        (),
        &mut (),
        Begin((tile.with(projected), decode)),
    );
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
        let staged = end(device, call, layer, attended, input);
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
        call: Launched<'c, 'a>,
        _: &'s mut (),
        (attended, input): (Output<'p>, Live<Owned<Width, Span>, Capacity>),
    ) -> Self::Output {
        let Self(layer, norm) = *self;
        let staged = end(device, call, layer, attended, input);
        let rows = update(device, layer, staged);
        norm.forward(device, (), &mut (), &rows)
    }
}

/// A launch's rows through the catalog tree's leaves in the family's order: embedded, the first
/// layer's stores over every tile, then each layer attended and, tile by tile, its work up to the
/// next layer's store or the head, from whose rows the selected rows' tokens are read. Generic in
/// its env, so only the launches that run it compile it.
impl<'m, 'i, 'a, 's, 'b, Env> Module<&'i mut launch::Input<'a, S>, Env, launch::State<'s, 'b, S>>
    for Decoder<&'m launch::Model<S, TokenScaled>>
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
        let layers = &body.layers.0;
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
            let mut staged =
                crate::ops::linear::output::fp8::channel::Stored::<Width, Queries>::empty();
            let boundary = (&output.0, &mut staged);
            let rows = (rows.0, rows.1, rows.2, boundary, &mut places.attended.0);
            attend.forward(&mut *device, (), &mut (), rows);
            let rows = (deferred, &places.stored.retained, &context.write);
            stage.forward(&mut *device, (), &mut *pool, rows);
            output.forward(device, (), &mut (), Begin(staged))
        };
        super::run::<S, _>(
            device,
            live,
            LAYERS,
            (
                |device, tile, stored| {
                    let first = layers.layer(0);
                    Tokenwise(&*embedded, &first).forward(device, call, stored, tile)
                },
                |device, index, places| {
                    let layer = layers.layer(index);
                    attend(device, &layer, &mut slots[index], places)
                },
                |device, index, output, input, stored| {
                    let (layer, next) = (layers.layer(index), layers.layer(index + 1));
                    Tokenwise(&layer, &next).forward(device, call, stored, (output, input))
                },
                |device, output, input| {
                    let last = layers.layer(LAYERS - 1);
                    Tokenwise(&last, norm).forward(device, call, &mut (), (output, input))
                },
                |device, head| causal.lm_head.read(device, head, call.context.1, tokens),
            ),
        );
    }
}

/// Native channel-scaled FP8 Llama 3.1's launch runs its catalog model by the decoder family's loops.
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
