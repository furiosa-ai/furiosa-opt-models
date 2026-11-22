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
use crate::ops::rotary::{self, group};
use crate::ops::swiglu::spans::fp8::channel as mlp;
use crate::tensor::cache::{self, Blocks};
use crate::tensor::live::Live;
use crate::{Module, Sequential, Then};
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
type Projecting = crate::ops::linear::output::fp8::Landed<Width, Queries, Span, f32, bf16>;
type FullPanel = mlp::FullPanel<Width>;
type Output<'p> = super::Output<'p, S, Projecting>;

struct Front {
    gain: <Stage as crate::Op<&'static <S as Geometry>::Norm<Width, crate::tensor::memory::Hbm>>>::Output,
    value: qkv::Prepared<Width, KvHeads, Head>,
    key: qkv::Prepared<Width, KvHeads, Head>,
    query: qkv::Prepared<Width, QueryHeads<S>, Head>,
}
struct Staged {
    product: mlp::Staging<Intermediate, Span>,
    second: FullPanel,
    attended: Live<Owned<Width, Span>, Capacity>,
}

fn front<const SPLIT_QUERY: bool>(device: &mut Device, layer: &Layer) -> Front {
    let Sequential(Residual(norm, attention, _), _) = layer;
    let Projection { query, key, value } = &attention.0.qkv.projection.projection;
    let value = qkv::prepare(device, &value.0.weight);
    let key = qkv::prepare(device, &key.0.weight);
    let query = qkv::prepare_weight::<_, _, _, SPLIT_QUERY>(device, &query.0.weight);
    let gain = crate::Op::forward(&Stage, device, &norm.0);
    Front {
        gain,
        value,
        key,
        query,
    }
}

fn short_decode(call: Launched<'_, '_>) -> bool {
    call.live.rows() <= 8 && call.context.0.read.items().rows() == call.live.rows()
}

/// Rotate Q immediately, spill it between K and V, then rotate and spill K with V.
fn store_period<G: Group>(
    device: &mut Device,
    positions: &rotary::Rotary<Head>,
    coefficients: &rotary::SplitHalf<
        Head,
        group::Coefficients<
            query::Cluster<QueryHeads<S>>,
            query::QuerySlices<G, QueryHeads<S>>,
            rotary::Row<Head>,
        >,
    >,
    stored: &mut Stored,
    start: usize,
    q: query::Queries<G, QueryHeads<S>, Head>,
    project_key: impl FnOnce(&mut Device) -> query::Pairs<G, KvHeads, Head>,
    project_value: impl FnOnce(&mut Device) -> query::Pairs<G, KvHeads, Head>,
) {
    let qr = crate::Op::forward(&group::Stage, device, coefficients);
    let q = crate::Op::forward(positions, device, (&q, &qr));
    let k = project_key(device);
    stored.queries.spill(device, &q, start);
    let v = project_value(device);
    let kr = crate::Op::forward(
        &group::Stage,
        device,
        (coefficients, PhantomData::<KvHeads>),
    );
    let k = crate::Op::forward(positions, device, (&k, &kr));
    stored.retained.spill(device, (&k, &v), start);
}

fn start(
    device: &mut Device,
    call: Launched<'_, '_>,
    layer: &Layer,
    front: Front,
    rows: Live<Owned<Width, Span>, Capacity>,
    (stored, residual): (&mut Stored, &mut Rows),
    projecting: &mut Projecting,
) {
    let Sequential(Residual(_, attention, _), _) = layer;
    let Attention {
        qkv: Positioned { positions, .. },
        output,
        ..
    } = &attention.0;
    let tile = rows.with(());
    let normalized = front.gain.normalize_span(device, &rows, short_decode(call));
    let mut home = Normalized::<Width, Capacity>::new();
    if tile.rows() > 8 && tile.rows() <= 64 {
        query::launch::spill_half_wave(device, &mut home, normalized.value(), tile.start());
    } else {
        home.spill(device, normalized.value(), (tile, tile.rows()));
    }
    let rotation = &call.context.0.positions;
    let last_span = tile.start() + tile.rows() == call.live.rows();
    let single = tile.count::<Tile>() % 2 != 0;
    let pairs = tile.count::<Tile>() / 2;
    if pairs > 0 {
        let staged = qkv::Staged::new(device, &home, tile.start() + tile.rows());
        let first = tile.start() / 256;
        let mut current = staged.pair(device, first);
        for pair in 0..pairs {
            // DMA the next pair before this contraction; both use the same resident QKV weights.
            let next = staged.pair(device, (first + pair + 1) % (Capacity::SIZE / 256));
            let at = tile.tile::<<TilePair as Group>::Period>(pair);
            let coefficients = crate::Op::forward(
                &group::ToDm::<TilePair, QueryHeads<S>>(PhantomData),
                device,
                (rotation, first + pair),
            );
            if !single && pair + 1 == pairs {
                super::rows::store::<S>(device, rows.value(), residual, tile);
            }
            let operand = current.fill(device);
            let q: query::Queries<TilePair, QueryHeads<S>, Head> =
                current.project_with_prefetch(device, &front.query, &operand, |device| {
                    if last_span && !single && pair + 1 == pairs {
                        output.forward(device, (), &mut (), &mut *projecting);
                    }
                });
            store_period::<TilePair>(
                device,
                positions,
                &coefficients,
                stored,
                at.start(),
                q,
                |device| current.project(device, &front.key, &operand),
                |device| current.project(device, &front.value, &operand),
            );
            current = next;
        }
    }
    if single {
        let at = tile.tile::<Tile>(2 * pairs);
        let groups = at.rows().div_ceil(8);
        let input = crate::Op::forward(
            &query::fp8::ToDm::<Tiles<1>, true>(PhantomData),
            device,
            (&home, (at.start(), Tile::SIZE), groups),
        );
        let coefficients = crate::Op::forward(
            &group::ToDm::<Tiles<1>, QueryHeads<S>>(PhantomData),
            device,
            (rotation, at.start() / Tile::SIZE),
        );
        super::rows::store::<S>(device, rows.value(), residual, tile);
        let period = qkv::Period::quantize(device, &input, groups);
        let operand = period.fill(device);
        let early = last_span && short_decode(call);
        let mut key_channels = DmTensor::new();
        let q: query::Queries<Tiles<1>, QueryHeads<S>, Head> =
            period.project_with_prefetch(device, &front.query, &operand, |device| {
                if last_span {
                    if early {
                        key_channels = period.channels(device, &front.key);
                    }
                    output.forward(device, (), &mut (), &mut *projecting);
                }
            });
        store_period::<Tiles<1>>(
            device,
            positions,
            &coefficients,
            stored,
            at.start(),
            q,
            |device| {
                if !early {
                    key_channels = period.channels(device, &front.key);
                }
                period.project_ready(device, &front.key, &operand, key_channels)
            },
            |device| period.project(device, &front.value, &operand),
        );
    }
}

/// A tile through `layer` from its attention to its MLP's second step, given a tile of its
/// attention output, the `attended` rows and the `landed` output projection, and `input` of its
/// input rows: the output projection, the attention residual and the MLP's first step on its norm.
fn end(
    device: &mut Device,
    call: Launched<'_, '_>,
    layer: &Layer,
    (attended, landed): Output<'_>,
    residual: &Rows,
) -> Staged {
    let Sequential(Residual(_, attention, sum), Residual(norm, mlp, _)) = layer;
    let Attention { output, .. } = &attention.0;
    let (gate_up, down) = mlp.0.parts();
    let tile = attended.with(());
    let gain = crate::Op::forward(&Stage, &mut *device, &norm.0);
    let decode = call.context.0.read.items().rows() == call.live.rows();
    let (input, ahead, projected) = if decode {
        output.0.decode_span(
            device,
            landed,
            &attended.value().0,
            tile,
            |device| tile.with(super::rows::load::<S>(device, residual, tile)),
            |device| gate_up.forward(device, (), &mut (), Begin(())),
        )
    } else {
        let input = output
            .0
            .begin_span(device, landed, &attended.value().0, tile, |device| {
                tile.with(super::rows::load::<S>(device, residual, tile))
            });
        let mut step = |device: &mut Device| {
            output.forward(
                device,
                (),
                &mut (),
                (&mut *landed, &attended.value().0, tile),
            )
        };
        let ahead = gate_up.forward(&mut *device, (), &mut (), Begin(()));
        let _ = step(&mut *device);
        let projected = match step(&mut *device) {
            Poll::Ready(rows) => rows,
            Poll::Pending => Owned::new(),
        };
        (input, ahead, projected)
    };
    let projected = tile.with(projected);
    let attended = sum.forward(&mut *device, (), &mut (), (&input, &projected));
    let normalized = gain.normalize_span(device, &attended, short_decode(call));
    let rows = tile.with(normalized.value());
    let (projected, second) =
        mlp::gate_up(device, gate_up, ahead, rows, short_decode(call), |device| {
            mlp::down::<_, _, 64>(device, down, 1)
        });
    // Pure item launches retain product scales in DM at any span size. The MLP also
    // retains them for spans of at most 256 rows, including prompt and mixed launches.
    let staged = down.forward(
        &mut *device,
        (),
        &mut (),
        Begin((tile.with(projected), decode)),
    );
    Staged {
        product: staged,
        second,
        attended,
    }
}

/// A `staged` tile through the second step of `layer`'s MLP, the down projection, and the MLP
/// residual; the rows.
fn update<F>(
    device: &mut Device,
    layer: &Layer,
    staged: Staged,
    first: &FullPanel,
    front: (
        impl FnOnce(&mut Device) -> F,
        impl FnOnce(&mut Device, &mut F),
    ),
) -> (Live<Owned<Width, Span>, Capacity>, F) {
    let Sequential(_, Residual(_, mlp, sum)) = layer;
    let (_, down) = mlp.0.parts();
    let Staged {
        product,
        second,
        attended,
    } = staged;
    let tile = attended.with(());
    let (fed, front) = mlp::project_down(device, down, (tile.with(product), first, second), front);
    (
        sum.forward(device, (), &mut (), (&attended, &tile.with(fed))),
        front,
    )
}

/// A tile of the embedded rows through the first layer up to its attention's store; the tile's
/// rows, the first layer's input.
impl<'l, 'c, 'a, 's>
    Module<Live<(), Capacity>, Launched<'c, 'a>, (&'s mut Stored, &'s mut Rows, &'s mut Projecting)>
    for Tokenwise<&'l Rows, &'l Layer>
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        call: Launched<'c, 'a>,
        (stored, residual, projecting): (&'s mut Stored, &'s mut Rows, &'s mut Projecting),
        tile: Live<(), Capacity>,
    ) -> Self::Output {
        let Self(embedded, layer) = *self;
        let front = front::<false>(device, layer);
        let rows = tile.with(embedded.fill(&mut *device, tile));
        start(
            device,
            call,
            layer,
            front,
            rows,
            (stored, residual),
            projecting,
        )
    }
}

/// A tile through a layer from its attention and the next layer up to its attention's store; the
/// next layer's input rows. The next layer's front lands while this one's MLP projects.
impl<'l, 'c, 'a, 'p, 's>
    Module<(Output<'p>, &'s mut Rows), Launched<'c, 'a>, (&'s mut Stored, &'s FullPanel)>
    for Tokenwise<&'l Layer, &'l Layer>
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        call: Launched<'c, 'a>,
        (stored, first): (&'s mut Stored, &'s FullPanel),
        (attended, input): (Output<'p>, &'s mut Rows),
    ) -> Self::Output {
        let Self(layer, next) = *self;
        let (attended, projecting) = attended;
        let staged = end(device, call, layer, (attended, &mut *projecting), input);
        let Sequential(Residual(_, attention, _), _) = next;
        let query = &attention.0.qkv.projection.projection.query.0.weight;
        let (rows, front) = update(
            device,
            layer,
            staged,
            first,
            (
                |device| front::<true>(device, next),
                |device, front| qkv::finish_query(device, query, &mut front.query),
            ),
        );
        start(device, call, next, front, rows, (stored, input), projecting)
    }
}

/// A tile through the last layer from its attention and the final norm; the head's rows.
impl<'l, 'c, 'a, 'p, 's> Module<(Output<'p>, &'s Rows), Launched<'c, 'a>, &'s FullPanel>
    for Tokenwise<&'l Layer, &'l FinalNorm>
{
    type Output = Live<Owned<Width, Span>, Capacity>;

    fn forward(
        &self,
        device: &mut Device,
        call: Launched<'c, 'a>,
        first: &'s FullPanel,
        (attended, input): (Output<'p>, &'s Rows),
    ) -> Self::Output {
        let Self(layer, norm) = *self;
        let staged = end(device, call, layer, attended, input);
        let (rows, ()) = update(device, layer, staged, first, (|_| (), |_, _| ()));
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
        let (((embedded, blocks), ()), tokens) = state;
        let range = blocks.tokens.len() / LAYERS;
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
                      mut pool: cache::Layer<&mut Blocks<KvHeads, Head>>,
                      places: &mut Places| {
            let Sequential(Residual(_, attention, _), Residual(_, mlp, _)) = layer;
            let (_, down) = mlp.0.parts();
            let Attention {
                cache: stage,
                attend,
                ..
            } = &attention.0;
            let context = &call.context.0;
            let operands = (&context.write, &context.read);
            let deferred = stage.forward(
                &mut *device,
                (),
                &mut pool,
                (&places.stored.retained, operands, live),
            );
            let history = Retained {
                pool: pool.read(),
                key: &places.stored.retained.0,
                value: &places.stored.retained.1,
                read: &stage.read,
            };
            let rows = (&places.stored.queries, &history, &context.read);
            let mut panel = FullPanel::new();
            let boundary = (down, &mut panel);
            let rows = (rows.0, rows.1, rows.2, boundary, &mut places.attended.0);
            attend.forward(&mut *device, (), &mut (), rows);
            let rows = (deferred, &places.stored.retained, &context.write);
            stage.forward(&mut *device, (), &mut pool, rows);
            panel
        };
        let Sequential(Residual(_, attention, _), _) = layers.layer(0);
        let empty = crate::ops::linear::output::fp8::channel::Stored::<Width, Queries>::empty();
        let mut projecting = attention
            .0
            .output
            .forward(device, (), &mut (), Begin(empty));
        super::run::<S, _, _>(
            device,
            live,
            LAYERS,
            &mut projecting,
            (
                |device, tile, stored, residual, projecting| {
                    let first = layers.layer(0);
                    Tokenwise(&*embedded, &first).forward(
                        device,
                        call,
                        (stored, residual, projecting),
                        tile,
                    )
                },
                |device, index, places| {
                    let layer = layers.layer(index);
                    attend(
                        device,
                        &layer,
                        cache::Layer {
                            blocks: &mut **blocks,
                            entry: (index) * range,
                        },
                        places,
                    )
                },
                |device, index, (attended, panel), input, stored, projecting| {
                    let (layer, next) = (layers.layer(index), layers.layer(index + 1));
                    Tokenwise(&layer, &next).forward(
                        device,
                        call,
                        (stored, &*panel),
                        ((attended, projecting), input),
                    )
                },
                |device, (attended, panel), input, projecting| {
                    let last = layers.layer(LAYERS - 1);
                    Tokenwise(&last, norm).forward(
                        device,
                        call,
                        &*panel,
                        ((attended, projecting), input),
                    )
                },
                |device, head| {
                    causal
                        .lm_head
                        .read_gathered(device, head, call.context.1, tokens)
                },
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
