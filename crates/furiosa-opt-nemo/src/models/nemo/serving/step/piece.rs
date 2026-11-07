//! One piece of the step's layer at a time, for chip probes that compile in seconds: each entry
//! runs the step's own code for its piece at layer 0, over the step's `Model`, `Input` and
//! `State`, and leaves what it computes where the host reads it back.
use super::*;
type Model = Recipe<'static>;
use crate::models::nemo::serving::attention::prompt::{FirstSpans, Segments};
use crate::models::nemo::serving::mlp::block::{Block, Blocks};
type DReduced = crate::models::nemo::serving::mlp::dpanel::Reduced;

/// Each wave's attended rows, as a decode batch's merge restores them.
pub type AttendedRows = HbmTensor<f32, Chips, m![Waves, Rows, Width]>;

/// What a piece produces besides `State`, or reads in the layer's own temporaries' place.
#[derive(DeviceSend)]
pub struct Out {
    /// Decode attention's output; the tails' decode-wave input.
    pub attended: AttendedRows,
    /// Each wave's post-attention residual: the tails' output, the MLP's input.
    pub residuals: HiddenRows,
    /// Each wave's quantized feed-forward activation: the tails' output, the MLP's input.
    pub activations: mlp::Input,
    /// The layer's output where `post` leaves it, so `state.hidden` stays its input across
    /// launches.
    pub layer: HiddenRows,
    /// The feed-forward's E4M3 gate·up product, the down projection's input (`mlp`, `post`).
    pub product: mlp::Product,
    /// `activations` K80-blocked (`dmlp`'s input).
    pub dactivations: mlp::block::Blocked,
    /// Each chip's own hidden half of `residuals` (`dmlp`'s input).
    pub dresiduals: mlp::dpanel::Owned,
    /// Each chip's own hidden half of the layer output (`dmlp`).
    pub dlayer: mlp::dpanel::Owned,
    /// Each chip's own hidden half of the post-attention residual (`dtail`).
    pub dresid: mlp::dpanel::Owned,
    /// Debug: the norm's row sums.
    pub dsums: HbmTensor<f32, Chips, m![Blocks, Hidden % 1280 / 640, { DReduced }, Block % 8, 8]>,
    /// Layer 0's D-sharded O weight, each chip's own half by every head (`dtail`).
    pub output_d: HbmTensor<f8e4m3, Chips, output::dshard::Stored>,
    /// The next layer's FP8 QKV input both chips gathered from the D-sharded layer output (`dpost`).
    pub dnext: dlayer::NextRows,
    /// `dactivations` in the absolute halves' own order (`dpost`'s MLP input).
    pub dmlpin: dlayer::NextRows,
    /// Each chip's index, the position of its own hidden half.
    pub chip: mlp::Count,
}

fn live(input: &Input) -> usize {
    input.live.extent::<m![Capacity]>()
}

/// The step's chunk geometry: (decode waves, prompt blocks, live waves).
fn shape(input: &Input) -> (usize, usize, usize) {
    let rows = live(input);
    let decode = input.prompt.start.extent::<m![Capacity]>() / 64;
    let blocks = input.prompt.blocks.extent::<m![Prompt / 256]>();
    (decode, blocks, rows.div_ceil(64))
}

/// Q/K/V of every live wave and their cache writes, as the step's layer 0 runs them.
#[device(chip = 2)]
pub fn qkv(
    device: &mut Device,
    model: &Model,
    (input, state, _out): (&Input, &mut State, &mut Out),
) {
    let (decode, blocks, _) = shape(input);
    let rows = input.live.extent::<m![Capacity]>();
    let layer = (&model.layers, 0);
    // As in the step: a decode-only step of three waves or more, and a chunk's two decode waves
    // or more, project in 256-row tiles, when the tile pass ends inside the capacity.
    let only = 1 - blocks.min(1);
    let fits = 1 - (4 * (blocks + (decode + 3) / 4) - mlp::Waves::SIZE).min(1);
    let tiled = (only * (decode - 2).min(1) + (1 - only) * (decode - 1).min(1)) * fits;
    let perwave = 1 - tiled;
    let mut weights = qkv::Weights::new();
    let mut first = qkv::Landed::new();
    for _ in 0..(decode * perwave).min(1) {
        Live::of(live(input)).land(device, wave(&state.hidden, 0), &mut first);
    }
    weights.land_norm(device, layer);
    // As in the step: with decode rows the projection weights land first, else in the chunk.
    for _ in 0..decode.min(1) {
        weights.land_pairs(device, layer);
        weights.land_query(device, layer);
    }
    weights.spread_scales(device);
    let first_wave = input::Wave::copy(device, input);
    for _ in 0..(decode * perwave).min(1) {
        let rotations = qkv::Rotations::land(device, &input.rotation, 0);
        let begun = qkv::begin(device, (&weights, weights.scales()), &first, &rotations);
        let quantized = input::quantize(device, (&begun.key, &begun.value));
        let (panels, key, value) = begun.split();
        first_wave.write(device, (&mut state.pool, 0), quantized, Live::of(rows));
        panels.store(device, (&rotations, 0), &mut state.queries);
        qkv::store_rows(
            device,
            (&key, &value),
            0,
            (&mut state.keys, &mut state.values),
        );
    }
    // As in the step: each later wave's rows land a wave ahead into one shared buffer.
    let later = perwave * (decode - decode.min(1));
    let mut ahead = qkv::Landed::new();
    for _ in 0..later.min(1) {
        qkv::land_into(device, wave(&state.hidden, 1), &mut ahead);
    }
    for at in 0..later {
        let rotations = qkv::Rotations::land(device, &input.rotation, at + 1);
        let entries = input::entries(device, input, at + 1);
        let begun = qkv::begin(device, (&weights, weights.scales()), &ahead, &rotations);
        for _ in 0..(later - at - 1).min(1) {
            qkv::land_into(device, wave(&state.hidden, at + 2), &mut ahead);
        }
        let quantized = input::quantize(device, (&begun.key, &begun.value));
        let (panels, key, value) = begun.split();
        input::write_wave(device, (&mut state.pool, 0), quantized, &entries);
        panels.store(device, (&rotations, at + 1), &mut state.queries);
        let kv = (&mut state.keys, &mut state.values);
        qkv::store_rows(device, (&key, &value), at + 1, kv);
    }
    qkv::tile::chunk(
        device,
        &mut weights,
        (layer, 1 - decode.min(1)),
        (
            &state.hidden,
            decode * perwave,
            blocks + tiled * (decode + 3) / 4,
        ),
        &input.rotation,
        &mut state.queries,
        qkv::tile::Cache {
            pool: &mut state.pool,
            entry: 0,
            offsets: &input.keys,
        },
    );
}

/// Q/K/V of a prompt-only chunk whose layer input is D-sharded (`out.dresiduals`, each chip's own
/// hidden half): each 256-row block's input norm over both halves (row sums exchanged), FP8 by
/// the QKV input scale, both halves all-gathered into `out.dactivations`, then `qkv`'s K/V and Q
/// commands over those rows.
#[device(chip = 2)]
pub fn dqkv(
    device: &mut Device,
    model: &Model,
    (input, state, out): (&Input, &mut State, &mut Out),
) {
    let (_, blocks, _) = shape(input);
    let layer = (&model.layers, 0);
    let mut weights = qkv::Weights::new();
    weights.land_norm(device, layer);
    weights.spread_scales(device);
    let gain = dlayer::gain(device, &model.layers.norm.at(0));
    let divisor = crate::ops::linear::Affine::new(
        (),
        weights
            .scales()
            .get::<mlp::dpanel::Cluster, DReduced>(device, 0),
    );
    for block in 0..blocks {
        let own = mlp::dpanel::load(device, &out.dresiduals, block);
        let mut slots = dlayer::Slots::new();
        dlayer::normalize::<4>(device, &own, (&gain, &divisor), &mut slots);
        dlayer::gather_send(device, &slots).place(
            device,
            &mut slots,
            (&mut out.dactivations, block),
        );
    }
    qkv::tile::chunk_gathered(
        device,
        &mut weights,
        (layer, 1),
        (&out.dactivations, 0, blocks),
        &input.rotation,
        &mut state.queries,
        qkv::tile::Cache {
            pool: &mut state.pool,
            entry: 0,
            offsets: &input.keys,
        },
    );
}

/// `dqkv` in the oracle's order: every block's input norm first, each chip keeping its FP8 half
/// in data memory, then each block's halves exchanged and placed into the chunk's activation
/// under the block before's commands (no HBM round trip of the gathered rows).
#[device(chip = 2)]
pub fn dqkv2(
    device: &mut Device,
    model: &Model,
    (input, state, out): (&Input, &mut State, &mut Out),
) {
    let (_, blocks, _) = shape(input);
    let layer = (&model.layers, 0);
    let mut weights = qkv::Weights::new();
    weights.land_norm(device, layer);
    weights.spread_scales(device);
    let gain = dlayer::gain(device, &model.layers.norm.at(0));
    let divisor = crate::ops::linear::Affine::new(
        (),
        weights
            .scales()
            .get::<mlp::dpanel::Cluster, DReduced>(device, 0),
    );
    let mut normalized = dlayer::Halves::new();
    dlayer::normalize_halves(
        device,
        (&gain, &divisor),
        &out.dresiduals,
        (&mut normalized, blocks),
    );
    qkv::tile::chunk_resident(
        device,
        &mut weights,
        (layer, 1),
        (&mut normalized, blocks),
        &input.rotation,
        &mut state.queries,
        qkv::tile::Cache {
            pool: &mut state.pool,
            entry: 0,
            offsets: &input.keys,
        },
    );
}

/// Debug: `dqkv`'s K/V and Q commands alone, over whatever `out.dactivations` holds (the QKV of a
/// layer whose previous layer already gathered its input).
#[device(chip = 2)]
pub fn qkv1(
    device: &mut Device,
    model: &Model,
    (input, state, out): (&Input, &mut State, &mut Out),
) {
    let (_, blocks, _) = shape(input);
    let layer = (&model.layers, 0);
    let mut weights = qkv::Weights::new();
    weights.land_norm(device, layer);
    weights.spread_scales(device);
    qkv::tile::chunk_gathered(
        device,
        &mut weights,
        (layer, 1),
        (&out.dactivations, 0, blocks),
        &input.rotation,
        &mut state.queries,
        qkv::tile::Cache {
            pool: &mut state.pool,
            entry: 0,
            offsets: &input.keys,
        },
    );
}

/// The first decode batch's attention over layer 0's pool pages through its few-row chain alone,
/// into `out.attended`: the batch must be a few-row batch.
#[device(chip = 2)]
pub fn few_attention(
    device: &mut Device,
    _model: &Model,
    (input, state, out): (&Input, &mut State, &mut Out),
) {
    let positions = attention::positions(device, &input.attention);
    let few = attention::few(&input.attention);
    let attended = attention::attend_few(
        device,
        &attention::History {
            pool: &state.pool,
            entry: 0,
            plan: &input.attention,
            batch: 0,
            rows: live(input).min(Rows::SIZE),
            positions: &positions,
            few: &few,
            next: 0,
        },
        &state.queries,
    );
    // SAFETY: the completed row-major owner flattens Heads/Groups/Head in wire order.
    let attended = unsafe { attended.reshape::<Chips, m![Rows, Width]>() };
    attended.view().to_hbm_view(
        &mut device.tdma,
        out.attended
            .view_mut()
            .tile::<m![Waves], 1, m![Waves = 1 #{!} Waves, Rows, Width]>(0),
    );
}

/// Every decode batch's attention over layer 0's pool pages into `out.attended`; the last batch
/// lands layer 1's Q/K/V weights as a middle layer's does.
#[device(chip = 2)]
pub fn decode_attention(
    device: &mut Device,
    model: &Model,
    (input, state, out): (&Input, &mut State, &mut Out),
) {
    let (decode, _, _) = shape(input);
    let mut weights = qkv::Weights::new();
    let positions = attention::positions(device, &input.attention);
    let few = attention::few(&input.attention);
    let feed = mlp::Model {
        layers: &model.layers,
        index: 0,
    };
    let mut first = mlp::weight::First::new(None);
    let mut buffers = attention::Buffers::new();
    for at in 0..decode {
        let merging = attention::attend(
            device,
            &attention::History {
                pool: &state.pool,
                entry: 0,
                plan: &input.attention,
                batch: at,
                rows: (live(input) - 64 * at).min(Rows::SIZE),
                positions: &positions,
                few: &few,
                next: (decode - at - 1).min(1),
            },
            (&state.queries, &mut state.partials),
            attention::Prefetch {
                gate: (feed.gate(), &mut first, 1 - (decode - at - 1).min(1)),
                lead: (&mut (), 0),
            },
            &mut buffers,
        );
        for _ in 0..1 - (decode - at - 1).min(1) {
            weights.prefetch(device, (&model.layers, 1), 1);
        }
        let attended = merging.finish(device);
        // SAFETY: the completed row-major owner flattens Heads/Groups/Head in wire order.
        let attended = unsafe { attended.reshape::<Chips, m![Rows, Width]>() };
        attended.view().to_hbm_view(
            &mut device.tdma,
            out.attended
                .view_mut()
                .tile::<m![Waves], 1, m![Waves = 1 #{!} Waves, Rows, Width]>(at),
        );
    }
}

/// The chunk's attention over layer 0's pool pages, span after span, into `state.prompt`.
#[device(chip = 2)]
pub fn prompt_attention(
    device: &mut Device,
    _model: &Model,
    (input, state, _out): (&Input, &mut State, &mut Out),
) {
    let start = input.prompt.start.extent::<m![Capacity]>();
    let segments = input.prompt.segments.extent::<m![Segments]>();
    let first = FirstSpans::of(&input.prompt);
    let landing = attention::prompt::land(device, (&state.pool, 0), (&input.prompt, &first), 1);
    attention::prompt::attend(
        device,
        (&state.pool, 0),
        (&input.prompt, (start, segments), &state.queries),
        landing,
        &mut state.prompt,
    );
}

/// Every live wave's O projection, TP2 exchange, residual and post-attention norm: decode waves
/// from `out.attended`, chunk waves from `state.prompt`, into `out.residuals` and
/// `out.activations`.
#[device(chip = 2)]
pub fn tails(
    device: &mut Device,
    model: &Model,
    (input, state, out): (&Input, &mut State, &mut Out),
) {
    let (decode, _, waves) = shape(input);
    let mut tail = Tail::new((&model.layers, 0), input.chip.extent::<m![2]>());
    tail.land_into(device);
    for at in 0..decode {
        let rows: Projected<f32> = out
            .attended
            .view()
            .tile::<m![Waves], 1, m![Waves = 1 # Waves, Rows, Width]>(at)
            .to_dm(&mut device.tdma);
        let hidden: Residual = wave(&state.hidden, at).to_dm(&mut device.tdma);
        let factors = tail.factors(device);
        let pending = tail.issue(
            device,
            &factors,
            (rows, hidden, at, Live::of(live(input) - 64 * at)),
        );
        // The step's decode completion: a wave's live groups only.
        tail.complete_live(
            device,
            &factors,
            (pending, |_: &mut Device| {}),
            (&mut out.residuals, &mut out.activations),
        );
    }
    let factors = tail.factors(device);
    for at in 0..waves - decode {
        let pending = tail.issue_prompt(device, &factors, (decode, decode + at), state);
        tail.complete(
            device,
            &factors,
            pending,
            (&mut out.residuals, &mut out.activations),
        );
    }
}

/// Gate, up and down of every live wave from `out.activations`, each wave's partials of both
/// chips added to its `out.residuals` rows into `state.hidden`.
#[device(chip = 2)]
pub fn mlp(
    device: &mut Device,
    model: &Model,
    (input, state, out): (&Input, &mut State, &mut Out),
) {
    let (_, _, waves) = shape(input);
    let model = mlp::Model {
        layers: &model.layers,
        index: 0,
    };
    let mut first = mlp::weight::First::new(None);
    model.land_first(device, &mut first);
    let scales = scale::Scales::land(device, model.layers.mlp_scales.at(0));
    let mut partials = mlp::Output::new();
    let pass = model.begin(
        device,
        (&out.activations, (&first.low, None), &scales),
        &mut out.product,
    );
    pass.sweep(
        device,
        (&out.activations, &mut out.product, &mut partials),
        (&out.residuals, &mut state.hidden, waves, live(input)),
        (&mut (), 0),
    );
}

/// The layer after attention as the step runs it: every decode wave's tail from `out.attended`,
/// then the chunk's tails interleaved with the feed-forward (each prompt wave's O projection and
/// exchange under the previous wave's gate and up), the down sweep into `out.layer`.
#[device(chip = 2)]
pub fn post(device: &mut Device, model: &Model, arguments: (&Input, &mut State, &mut Out)) {
    post_layer(device, model, arguments);
}

/// TP2's layer after attention and the next layer's Q/K/V over its output (layer 0's weights
/// for both): one launch across the layer boundary.
#[device(chip = 2)]
pub fn tpchain(
    device: &mut Device,
    model: &Model,
    (input, state, out): (&Input, &mut State, &mut Out),
) {
    post_layer(device, model, (input, &mut *state, &mut *out));
    let (decode, blocks, _) = shape(input);
    let layer = (&model.layers, 0);
    let mut weights = qkv::Weights::new();
    weights.land_norm(device, layer);
    weights.spread_scales(device);
    qkv::tile::chunk(
        device,
        &mut weights,
        (layer, 1),
        (&out.layer, decode, blocks),
        &input.rotation,
        &mut state.queries,
        qkv::tile::Cache {
            pool: &mut state.pool,
            entry: 0,
            offsets: &input.keys,
        },
    );
}

fn post_layer(
    device: &mut Device,
    model: &Model,
    (input, state, out): (&Input, &mut State, &mut Out),
) {
    let (decode, blocks, waves) = shape(input);
    let mut tail = Tail::new((&model.layers, 0), input.chip.extent::<m![2]>());
    let model = mlp::Model {
        layers: &model.layers,
        index: 0,
    };
    let mut first = mlp::weight::First::new(None);
    for at in 0..decode {
        let mut seeds = tail::Seeds::new();
        for _ in 0..1 - at.min(1) {
            tail.land(device, &mut seeds);
        }
        let rows: Projected<f32> = out
            .attended
            .view()
            .tile::<m![Waves], 1, m![Waves = 1 # Waves, Rows, Width]>(at)
            .to_dm(&mut device.tdma);
        for _ in 0..1 - at.min(1) {
            tail.spread(device, &seeds);
        }
        let hidden: Residual = wave(&state.hidden, at).to_dm(&mut device.tdma);
        let factors = tail.factors(device);
        let pending = tail.issue(
            device,
            &factors,
            (rows, hidden, at, Live::of(live(input) - 64 * at)),
        );
        for _ in 0..1 - at.min(1) {
            model.land_first(device, &mut first);
        }
        tail.complete_live(
            device,
            &factors,
            (pending, |_: &mut Device| {}),
            (&mut out.residuals, &mut out.activations),
        );
    }
    for _ in 0..blocks.min(1) * (1 - decode.min(1)) {
        tail.land_into(device);
        model.land_first(device, &mut first);
    }
    let mut partials = mlp::Output::new();
    // On a step with no decode rows the first two waves land their attended rows, then both
    // project and send before either completes.
    let pair = blocks.min(1) * (1 - decode.min(1)) * (waves - 1).min(1);
    for _ in 0..pair {
        let mut rows = (Projected::<f32>::new(), Projected::<f32>::new());
        tail.land_prompt_into(device, (0, 0), &state.prompt, &mut rows.0);
        tail.land_prompt_into(device, (0, 1), &state.prompt, &mut rows.1);
        let factors = tail.factors(device);
        let first = tail.issue_landed(device, &factors, (&rows.0, 0), state);
        let second = tail.issue_landed(device, &factors, (&rows.1, 1), state);
        tail.complete(
            device,
            &factors,
            first,
            (&mut out.residuals, &mut out.activations),
        );
        tail.complete(
            device,
            &factors,
            second,
            (&mut out.residuals, &mut out.activations),
        );
    }
    for _ in 0..blocks.min(1) - pair {
        let factors = tail.factors(device);
        let pending = tail.issue_prompt(device, &factors, (decode, decode), state);
        tail.complete(
            device,
            &factors,
            pending,
            (&mut out.residuals, &mut out.activations),
        );
    }
    let many = (decode - decode.min(1)).min(1);
    let fused = blocks.min(1) * (1 - many) * (waves - 4).min(1);
    let past = blocks.min(1) * many * ((waves - decode) / 4).min(1);
    // A fused chunk's wave 2 lands its attended rows ahead of the feed-forward's weights.
    let mut third = Projected::<f32>::new();
    for _ in 0..fused {
        tail.land_prompt_into(device, (decode, 2), &state.prompt, &mut third);
    }
    let pass = model.begin(
        device,
        (&out.activations, (&first.low, None), tail.scales()),
        &mut out.product,
    );
    // A step without a chunk: every wave's tail ran before its feed-forward.
    for _ in 0..1 - blocks.min(1) {
        pass.sweep(
            device,
            (&out.activations, &mut out.product, &mut partials),
            (&out.residuals, &mut out.layer, waves, live(input)),
            (&mut (), 0),
        );
    }
    // A chunk after one decode wave at most, of five waves or more: fused (`chunk`).
    for _ in 0..fused {
        let swept = pass.start(device, (&out.activations, &mut out.product), waves);
        let factors = tail.factors(device);
        let pending = tail.issue_landed(device, &factors, (&third, 2), state);
        tail.complete(
            device,
            &factors,
            pending,
            (&mut out.residuals, &mut out.activations),
        );
        let projection = tail.project_prompt(device, &factors, (decode, 3), state);
        exchange::chunk(
            device,
            (&tail, &factors, &pass, &state.prompt, projection),
            (&mut out.residuals, &mut out.activations, &mut out.product),
            (
                exchange::Hidden {
                    input: Some(&state.hidden),
                    output: &mut out.layer,
                },
                &mut partials,
            ),
            (swept, exchange::Second, decode, waves),
        );
    }
    // After two decode waves or more, a chunk of four waves or more: fused from its first wave.
    for _ in 0..past {
        let mut third = Projected::<f32>::new();
        tail.land_prompt_into(device, (decode, decode + 1), &state.prompt, &mut third);
        let swept = pass.start(device, (&out.activations, &mut out.product), waves);
        let factors = tail.factors(device);
        let pending = tail.issue_landed(device, &factors, (&third, decode + 1), state);
        tail.complete(
            device,
            &factors,
            pending,
            (&mut out.residuals, &mut out.activations),
        );
        let projection = tail.project_prompt(device, &factors, (decode, decode + 2), state);
        exchange::chunk(
            device,
            (&tail, &factors, &pass, &state.prompt, projection),
            (&mut out.residuals, &mut out.activations, &mut out.product),
            (
                exchange::Hidden {
                    input: Some(&state.hidden),
                    output: &mut out.layer,
                },
                &mut partials,
            ),
            (swept, exchange::Past(decode), decode, waves),
        );
    }
    // Any other chunk: each chunk wave's tail under the previous wave's gate/up.
    for _ in 0..blocks.min(1) - fused - past {
        for at in 0..decode - decode.min(1) {
            pass.project(device, &out.activations, (&mut out.product, at + 1));
        }
        serial(
            device,
            (&tail, &pass, state),
            out,
            (&mut partials, decode, waves),
        );
    }
}

/// The routed chunk's D-sharded layers, `input.layers` of them, as `Model::apply` runs them after
/// the decode rows' layers: prompt attention, the D-sharded tail and the next layer's Q/K/V.
#[device(chip = 2)]
pub fn chunk(
    device: &mut Device,
    model: &Model,
    (input, state, _out): (&Input, &mut State, &mut Out),
) {
    let shape = sharded::Shape {
        start: input.prompt.start.extent::<m![Capacity]>(),
        blocks: input.routed.blocks.extent::<m![Prompt / 256]>(),
        segments: input.routed.segments.extent::<m![Segments]>(),
        first: FirstSpans::of(&input.prompt),
        pages: input.pages.extent::<m![Pages]>(),
    };
    let depth = input.layers.extent::<m![Depth]>();
    crate::Module::forward(
        model,
        (
            device,
            sharded::Prompt {
                input,
                state,
                shape: &shape,
                layers: depth,
            },
        ),
    );
}

/// The step's layers of a decode-only step, `input.layers` of them over `state.hidden`: each wave's
/// Q/K/V, every batch's attention and tail, the feed-forward with the next layer's Q/K/V ahead. It
/// is `Model::apply` without the embedding, the prompt paths and the readout, kept in step with it.
#[device(chip = 2)]
pub fn layers(
    device: &mut Device,
    model: &Model,
    (input, state, _out): (&Input, &mut State, &mut Out),
) {
    let rows = input.live.extent::<m![Capacity]>();
    let waves = (rows + 63) / 64;
    let decode = input.prompt.start.extent::<m![Capacity]>() / 64;
    let pages = input.pages.extent::<m![Pages]>();
    let depth = input.layers.extent::<m![Depth]>();
    let alone = 1 - (waves - 1).min(1);
    let mut weights = qkv::Weights::new();
    let positions = attention::positions(device, &input.attention);
    let few = attention::few(&input.attention);
    let first_wave = input::Wave::copy(device, input);
    let mut rotations = qkv::Rotations::new();
    rotations.land_into(device, &input.rotation, 0);
    let mut pair_scale = qkv::tile::PairScale::new();
    for index in 0..depth {
        let layer = (&model.layers, index);
        let tiled = (decode - 2).min(1);
        let perwave = 1 - tiled;
        let carried = (index * (decode - 1).min(1) * perwave).min(1);
        let prefetched = (index * (decode - 1)).min(1);
        let fresh = 1 - carried;
        let mut first = qkv::Landed::new();
        for _ in 0..(fresh * perwave).min(1) {
            Live::of(rows).land(device, wave(&state.hidden, 0), &mut first);
        }
        for _ in 0..1 - index.min(1) {
            weights.land_weights(device, layer);
        }
        weights.spread_scales(device);
        for _ in 0..(index * (1 - prefetched)).min(1) {
            weights.land_last(device, layer);
        }
        let later = (perwave * (fresh * (decode - 1) + carried)).min(Waves::SIZE);
        let mut ahead = qkv::Landed::new();
        for _ in 0..later.min(1) {
            qkv::land_into(device, wave(&state.hidden, 1), &mut ahead);
        }
        for _ in 0..(fresh * perwave).min(1) {
            let begun = qkv::begin(device, (&weights, weights.scales()), &first, &rotations);
            let quantized = input::quantize(device, (&begun.key, &begun.value));
            let (panels, _, _) = begun.split();
            panels.store(device, (&rotations, 0), &mut state.queries);
            first_wave.write(
                device,
                (&mut state.pool, index * pages),
                quantized,
                Live::of(rows),
            );
        }
        for at in 0..later {
            let wave_at = at + 1;
            let rotations = qkv::Rotations::land(device, &input.rotation, wave_at);
            let entries = input::entries(device, input, wave_at);
            let begun = qkv::begin(device, (&weights, weights.scales()), &ahead, &rotations);
            for _ in 0..(later - at - 1).min(1) {
                qkv::land_into(device, wave(&state.hidden, wave_at + 1), &mut ahead);
            }
            let quantized = input::quantize(device, (&begun.key, &begun.value));
            let (panels, _, _) = begun.split();
            input::write_wave(
                device,
                (&mut state.pool, index * pages),
                quantized,
                &entries,
            );
            panels.store(device, (&rotations, wave_at), &mut state.queries);
        }
        qkv::tile::chunk_carried(
            device,
            &mut weights,
            (layer, 1 - decode.min(1)),
            (&state.hidden, 0, tiled * (decode + 3) / 4),
            (&mut pair_scale, 1 - index.min(1)),
            &input.rotation,
            &mut state.queries,
            qkv::tile::Cache {
                pool: &mut state.pool,
                entry: index * pages,
                offsets: &input.keys,
            },
        );
        let mut activations = mlp::Input::new();
        let mut residuals = HiddenRows::new();
        let mut tail = Tail::new(layer, input.chip.extent::<m![2]>());
        let feed = mlp::Model {
            layers: &model.layers,
            index,
        };
        let ahead = (decode - 1).min(1);
        let mut first = mlp::weight::First::new(Some(ahead));
        let late = 1 - ahead;
        decode_batches(
            device,
            &Batches {
                pool: (&state.pool, index * pages),
                plan: &input.attention,
                positions: &positions,
                few: &few,
                rows: (rows, decode),
            },
            (&state.queries, &mut state.partials, &state.hidden),
            (&mut tail, &feed, &mut first, late),
            Prefetched {
                weights: &mut weights,
                layer: (&model.layers, index + 1),
                count: (depth - index - 1).min(1),
                pairs: 1 - alone - tiled,
            },
            (&mut residuals, &mut activations),
        );
        let mut product = mlp::Product::new();
        let mut partials = mlp::Output::new();
        let pass = feed.begin(
            device,
            (
                &activations,
                (&first.low, Some((&mut first.high, ahead + late))),
                tail.scales(),
            ),
            &mut product,
        );
        pass.sweep(
            device,
            (&activations, &mut product, &mut partials),
            (&residuals, &mut state.hidden, waves, rows),
            (
                &mut Ahead {
                    input,
                    weights: &mut weights,
                    layer: (&model.layers, index + 1),
                    scales: scale::Scales::new(),
                    written: ((&mut state.pool, (index + 1) * pages), &mut state.queries),
                    first: qkv::Landed::new(),
                    count: (depth - index - 1).min(1),
                    issued: perwave,
                    pair_scale: (&mut pair_scale, tiled),
                },
                alone,
            ),
        );
    }
}

/// A chunk's feed-forward after its decode waves' gate/up, each wave's down after every gate/up:
/// a later wave's tail runs under each prompt wave's gate/up.
fn serial(
    device: &mut Device,
    (tail, pass, state): (&Tail<'_>, &mlp::Pass<'_>, &State),
    out: &mut Out,
    (partials, decode, waves): (&mut mlp::Output, usize, usize),
) {
    let projected = decode + 1 - decode.min(1);
    let rest = waves - projected;
    for _ in 0..rest.min(1) {
        let factors = tail.factors(device);
        for at in 0..rest - rest.min(1) {
            let wave = projected + at;
            let projection = tail.project_prompt(device, &factors, (decode, wave + 1), state);
            let activation = mlp::producer::Activation::land(device, &out.activations, wave);
            let pending = tail.send(device, projection, state);
            let contracted = pass.contract(device, &activation);
            let summed = tail.sum(device, pending);
            pass.store(device, contracted, (&mut out.product, wave));
            tail.normalize(
                device,
                &factors,
                summed,
                (&mut out.residuals, &mut out.activations),
            );
        }
        pass.project(device, &out.activations, (&mut out.product, waves - 1));
    }
    pass.finish(
        device,
        (&out.product, partials),
        (&out.residuals, &mut out.layer, waves),
    );
}

/// The step's end as it runs after the last layer: each live wave of `state.hidden` back into row
/// order, the selected rows' final norm and head projection into `state.logits`, then one greedy
/// token per counted row into `state.tokens`.
#[device(chip = 2)]
pub fn head(
    device: &mut Device,
    model: &Model,
    (norm, input, state, _out): (&Norm, &Input, &mut State, &mut Out),
) {
    let (_, _, waves) = shape(input);
    model.finish(device, (norm, input, waves), state);
}

/// Probe: wave 0's hidden rows landed with both hidden halves in the element (clusters split by
/// rows), each chip's own half (`Hidden / 2560 = chip`) copied out and the peer's own half received,
/// both stored to `out.layer` wave 0: the D-sharded layout's selections are element digits.
#[device(chip = 2)]
pub fn halves(
    device: &mut Device,
    _model: &Model,
    (_input, state, out): (&Input, &mut State, &mut Out),
) {
    type Split = m![Rows % 8 / 4];
    type Groups = m![Hidden % 2560 / 80, Rows / 8];
    type Both = DmTensor<bf16, Chips, Split, Groups, m![Hidden / 2560, Rows % 4, Hidden % 80]>;
    type Half = DmTensor<bf16, Chips, Split, Groups, m![Rows % 4, Hidden % 80]>;
    let rows: Both = wave(&state.hidden, 0).to_dm(&mut device.tdma);
    let own: Half = rows
        .view()
        .chip_slice::<m![Hidden / 2560], m![Rows % 4, Hidden % 80]>([0, 1])
        .to_dm(&mut device.tdma);
    let peer: Half = rows
        .view()
        .chip_shuffle([1, 0])
        .chip_slice::<m![Hidden / 2560], m![Rows % 4, Hidden % 80]>([0, 1])
        .to_dm(&mut device.tdma);
    own.view().to_hbm_view(
        &mut device.tdma,
        wave_mut(&mut out.layer, 0).tile::<m![Hidden / 2560], 1, m![
            Waves = 1 #{!} Waves,
            Hidden / 2560 = 1 #{!} 2,
            Hidden % 2560 / 80,
            Rows,
            Hidden % 80
        ]>(0),
    );
    peer.view().to_hbm_view(
        &mut device.tdma,
        wave_mut(&mut out.layer, 0).tile::<m![Hidden / 2560], 1, m![
            Waves = 1 #{!} Waves,
            Hidden / 2560 = 1 #{!} 2,
            Hidden % 2560 / 80,
            Rows,
            Hidden % 80
        ]>(1),
    );
}

/// The D-sharded O and residual of every 256-row block of the chunk's waves: each wave's own
/// heads quantized and staged, the peer's received and staged, the block projected once in the
/// gate/up's geometry (`output::oblock`), added to its own half of the layer input
/// (`out.dresiduals`) into `out.dresid`.
#[device(chip = 2)]
pub fn dtail(
    device: &mut Device,
    model: &Model,
    (input, state, out): (&Input, &mut State, &mut Out),
) {
    use output::oblock;
    let (_, _, waves) = shape(input);
    let projection = oblock::Projection::land(
        device,
        (
            (&model.layers, 0, out.chip.extent::<m![2]>()),
            model.layers.output_scales.at(0),
        ),
    );
    let factors = projection.factors(device);
    let rank = out.chip.extent::<m![2]>();
    let mut staged = oblock::Staged::new();
    for block in 0..waves.div_ceil(4) {
        #[unroll]
        for wave in 0..4 {
            let landed = oblock::land(device, &state.prompt, 4 * block + wave);
            let own = oblock::quantize(device, &landed, &factors);
            let peer = oblock::send(device, &own);
            oblock::store(device, &own, (&mut staged, 1 - rank, (block, wave)));
            oblock::store(device, &peer, (&mut staged, rank, (block, wave)));
        }
        let mut heads = oblock::Heads::new();
        oblock::stage(device, (&staged, block), &mut heads);
        let filled = oblock::fill(device, &heads);
        let projected = projection.project(device, (&filled, &factors));
        let sums = oblock::transpose(device, &projected);
        let hidden = mlp::dpanel::load(device, &out.dresiduals, block);
        let mut residual = mlp::dpanel::Own::new();
        dlayer::residual(device, (&sums, &hidden), &mut residual);
        mlp::dpanel::store(device, &residual, (&mut out.dresid, block));
    }
}

/// The D-sharded layer after attention over the chunk's 256-row blocks (`step::dlayer`): its
/// layer input `out.dresiduals` and output `out.dlayer` each chip's own hidden half; the
/// post-attention residual into `out.dresid` and the FP8 activation into `out.dactivations`.
#[device(chip = 2)]
pub fn dpost(
    device: &mut Device,
    model: &Model,
    (input, state, out): (&Input, &mut State, &mut Out),
) {
    dpost_layer(device, model, (input, &mut *state, &mut *out), &mut ());
    let (_, blocks, _) = shape(input);
    let chip = out.chip.extent::<m![2]>();
    for k in 0..blocks - blocks.min(1) {
        let peer = dlayer::send_own(device, &out.dnext, (1 + k, chip));
        peer.place_half(device, (&mut out.dnext, 1 + k, chip));
    }
}

/// `dpost` and the next layer's Q/K/V over the rows it gathered into `out.dnext` (layer 0's
/// weights for both): one launch across the layer boundary.
#[device(chip = 2)]
pub fn dchain(
    device: &mut Device,
    model: &Model,
    (input, state, out): (&Input, &mut State, &mut Out),
) {
    let layer = (&model.layers, 0);
    let mut weights = qkv::Weights::new();
    let mut fused = qkv::tile::fused::Weight::new();
    let mut scales = qkv::tile::fused::Seeds::new();
    let mut next = qkv::tile::NextFused {
        weights: &mut weights,
        fused: &mut fused,
        scales: &mut scales,
        layer,
    };
    dpost_layer(device, model, (input, &mut *state, &mut *out), &mut next);
    let (_, blocks, _) = shape(input);
    let chip = out.chip.extent::<m![2]>();
    weights.spread_scales(device);
    qkv::tile::chunk_fused(
        device,
        (&weights, (&fused, &scales)),
        (&mut out.dnext, chip, 0, blocks),
        &input.rotation,
        &mut state.queries,
        qkv::tile::Cache {
            pool: &mut state.pool,
            entry: 0,
            offsets: &input.keys,
        },
    );
}

fn dpost_layer<P: dlayer::Prefetch>(
    device: &mut Device,
    model: &Model,
    (input, state, out): (&Input, &mut State, &mut Out),
    prefetch: &mut P,
) {
    let (_, _, waves) = shape(input);
    let mut projection =
        output::oblock::Projection::with_scales(device, model.layers.output_scales.at(0));
    let factors = projection.factors(device);
    // Phase A projects with `output::oblock`; the per-wave weight stays unlanded.
    let weight = output::dshard::Weight::new();
    let divisor = weight.divisor::<output::dshard::Cluster, output::dshard::Landing>(device);
    let gain = dlayer::post_norm(device, &model.layers, 0);
    let scales = scale::Scales::land(device, model.layers.mlp_scales.at(0));
    let quantize = crate::ops::linear::Affine::new(
        (),
        scales.get::<mlp::dpanel::Cluster, mlp::dpanel::Reduced>(device, 0),
    );
    let next_gain = dlayer::input_norm(device, &model.layers, 0);
    let qkv_scales = scale::Scales::land(device, model.layers.qkv_scales.at(0));
    let next_quantize = crate::ops::linear::Affine::new(
        (),
        qkv_scales.get::<mlp::dpanel::Cluster, mlp::dpanel::Reduced>(device, 0),
    );
    let tail = dlayer::Tail {
        output: &weight,
        divisor: &divisor,
        gain: &gain,
        quantize: &quantize,
        next: (&next_gain, &next_quantize),
    };
    let mut heads = output::oblock::Heads::new();
    let rank = out.chip.extent::<m![2]>();
    let staged = dlayer::first_heads(
        device,
        &state.prompt,
        (&factors, &mut heads),
        (&mut projection, (&model.layers, 0)),
        rank,
    );
    dlayer::layer_seq(
        device,
        prefetch,
        (&tail, (&model.layers, 0), &scales),
        ((&projection, &factors), staged, &heads),
        (&state.prompt, &out.dresiduals),
        (&mut out.dresid, &mut out.dmlpin, &mut out.dlayer),
        (
            waves.div_ceil(4),
            &mut out.dnext,
            out.chip.extent::<m![2]>(),
        ),
    );
}

/// Debug: phase A alone (`dlayer::tail`) over every block.
#[device(chip = 2)]
pub fn dnorm(device: &mut Device, model: &Model, arguments: (&Input, &mut State, &mut Out)) {
    dnorm_with::<4, true>(device, model, arguments);
}

fn dnorm_with<const NORM: usize, const GATHER: bool>(
    device: &mut Device,
    model: &Model,
    (input, state, out): (&Input, &mut State, &mut Out),
) {
    let (_, _, waves) = shape(input);
    let mut weight = output::dshard::Weight::new();
    weight.land_from(device, (&out.output_d, model.layers.output_scales.at(0)));
    let divisor = weight.divisor::<output::dshard::Cluster, output::dshard::Landing>(device);
    let gain = dlayer::post_norm(device, &model.layers, 0);
    let scales = scale::Scales::land(device, model.layers.mlp_scales.at(0));
    let quantize = crate::ops::linear::Affine::new(
        (),
        scales.get::<mlp::dpanel::Cluster, mlp::dpanel::Reduced>(device, 0),
    );
    let tail = dlayer::Tail {
        output: &weight,
        divisor: &divisor,
        gain: &gain,
        quantize: &quantize,
        next: (&gain, &quantize),
    };
    for block in 0..waves.div_ceil(4) {
        dlayer::tail::<NORM, GATHER>(
            device,
            &tail,
            (&state.prompt, &out.dresiduals),
            (&mut out.dresid, &mut out.dactivations),
            block,
        );
    }
}

/// Debug: the norm's row sums local to a cluster's slice group (no swaps or exchanges).
#[device(chip = 2)]
pub fn dnorm_a(device: &mut Device, model: &Model, arguments: (&Input, &mut State, &mut Out)) {
    dnorm_with::<1, true>(device, model, arguments);
}

/// Debug: + the output groups' slice swap.
#[device(chip = 2)]
pub fn dnorm_b(device: &mut Device, model: &Model, arguments: (&Input, &mut State, &mut Out)) {
    dnorm_with::<2, true>(device, model, arguments);
}

/// Debug: + the cluster swap.
#[device(chip = 2)]
pub fn dnorm_c(device: &mut Device, model: &Model, arguments: (&Input, &mut State, &mut Out)) {
    dnorm_with::<3, true>(device, model, arguments);
}

/// Debug: the norm's row sums of each block of `out.dresiduals` (their square root's inverse
/// times the block into `out.dresid` when SCALE, else only the sums' first row group times one).
fn dsq_with<const STEP: usize>(
    device: &mut Device,
    model: &Model,
    (input, _state, out): (&Input, &mut State, &mut Out),
) {
    let (_, _, waves) = shape(input);
    let gain = dlayer::gain(device, &model.layers.post_norm.at(0));
    for block in 0..waves.div_ceil(4) {
        let residual = mlp::dpanel::load(device, &out.dresiduals, block);
        let mut result = mlp::dpanel::Own::new();
        if STEP == 0 {
            let sums = dlayer::squares(device, &residual);
            dlayer::scaled(device, (&sums, &residual), &mut result);
        } else if STEP == 2 {
            // squares alone, its sums' bytes stored as the block's first panel slice
            let sums = dlayer::squares(device, &residual);
            // SAFETY: debug store of the sums' bytes only.
            let bytes = unsafe {
                sums.view()
                    .reshape::<Chips, mlp::dpanel::Cluster, mlp::dpanel::Reduced, m![Block % 8, 8]>(
                    )
            };
            bytes.to_hbm_view(&mut device.tdma, out.dsums.view_mut().tile::<m![Blocks], 1, m![Blocks = 1 #{!} 16, Hidden % 1280 / 640, { DReduced }, Block % 8, 8]>(block));
            result
                .view_mut()
                .memset(const { bf16::from_f32(0.0) }, &mut device.sub);
        } else if STEP == 4 {
            let sums = dlayer::squares_with::<false>(device, &residual);
            dlayer::scaled(device, (&sums, &residual), &mut result);
        } else if STEP == 3 {
            // inverse alone, from zeroed sums
            let mut sums = dlayer::RowSums::new();
            sums.view_mut().memset(const { 1.0f32 }, &mut device.sub);
            dlayer::scaled(device, (&sums, &residual), &mut result);
        } else {
            dlayer::gained(device, (&gain, &residual), &mut result);
        }
        mlp::dpanel::store(device, &result, (&mut out.dresid, block));
    }
}

/// Debug: squares + inverse.
#[device(chip = 2)]
pub fn dsq(device: &mut Device, model: &Model, arguments: (&Input, &mut State, &mut Out)) {
    dsq_with::<0>(device, model, arguments);
}

/// Debug: gain VRF.
#[device(chip = 2)]
pub fn dgain(device: &mut Device, model: &Model, arguments: (&Input, &mut State, &mut Out)) {
    dsq_with::<1>(device, model, arguments);
}

/// Debug: squares alone.
#[device(chip = 2)]
pub fn dsq0(device: &mut Device, model: &Model, arguments: (&Input, &mut State, &mut Out)) {
    dsq_with::<2>(device, model, arguments);
}

/// Debug: inverse alone.
#[device(chip = 2)]
pub fn dsq1(device: &mut Device, model: &Model, arguments: (&Input, &mut State, &mut Out)) {
    dsq_with::<3>(device, model, arguments);
}

/// Debug: squares without the inter-slice reduction.
#[device(chip = 2)]
pub fn dsq2(device: &mut Device, model: &Model, arguments: (&Input, &mut State, &mut Out)) {
    dsq_with::<4>(device, model, arguments);
}
