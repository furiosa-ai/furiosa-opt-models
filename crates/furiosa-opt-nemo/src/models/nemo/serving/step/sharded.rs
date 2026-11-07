//! A step whose chunk holds a 256-row block or more: every layer's chunk after
//! attention D-sharded (`dlayer`), each later layer's Q/K/V one weight-stationary command a block
//! over the FP8 halves the layer before normalized (`qkv::tile::chunk_fused`).
//! - Layer 0's Q/K/V projects from the replicated hidden rows; trip `k` attends layer `k`, runs its
//!   D-sharded tail, whose down lands layer `k + 1`'s Q/K/V weights, then projects layer `k + 1`'s
//!   Q/K/V. The last trip lands its own layer's weights again and projects nothing, so the tail is
//!   one copy in the kernel.
//! - Each chip keeps its own hidden half between layers; the readout's rows are joined back once.
//! - The chunk's rows start at capacity row `start`, past the decode rows the TP2 layers ran.
use super::*;
use crate::Module;
use crate::models::nemo::serving::mlp::dpanel;
use crate::models::nemo::serving::output::{dshard, oblock};

/// The chunk's rows a D-sharded layer reads and writes besides the step's state: each chip's own
/// half of the layer input (also its output) and of the post-attention residual, the feed-forward's
/// FP8 input, and the FP8 halves the next layer's Q/K/V reads.
struct Rows {
    hidden: dpanel::Owned,
    residuals: dpanel::Owned,
    blocked: dlayer::NextRows,
    next: dlayer::NextRows,
}

/// The chunk's geometry: its first capacity row, its 256-row blocks, its prompt segments and
/// their first spans, and the pool pages a layer spans.
pub(crate) struct Shape {
    pub(crate) start: usize,
    pub(crate) blocks: usize,
    pub(crate) segments: usize,
    pub(crate) first: FirstSpans,
    pub(crate) pages: usize,
}

/// The chunk of a launch of `layers` layers, of `shape`.
pub(crate) struct Prompt<'a> {
    pub(crate) input: &'a Input,
    pub(crate) state: &'a mut State,
    pub(crate) shape: &'a Shape,
    pub(crate) layers: usize,
}

impl Module<(&mut Device, Prompt<'_>)> for Recipe<'_> {
    type Output = ();

    fn forward(&self, (device, prompt): (&mut Device, Prompt<'_>)) {
        let model = self;
        let (input, shape, layers) = (prompt.input, prompt.shape, prompt.layers);
        let wave = shape.start / 64;
        let state = &mut *prompt.state;
        let count = input.count.extent::<m![Selected]>();
        let chip = input.chip.extent::<m![2]>();
        let mut rows = Rows {
            hidden: dpanel::Owned::new(),
            residuals: dpanel::Owned::new(),
            blocked: dlayer::NextRows::new(),
            next: dlayer::NextRows::new(),
        };
        let mut weights = qkv::Weights::new();
        {
            let layer = (&model.layers, 0);
            weights.land_norm(device, layer);
            weights.spread_scales(device);
            qkv::tile::chunk(
                device,
                &mut weights,
                (layer, 1),
                (&state.hidden, wave, shape.blocks),
                &input.rotation,
                &mut state.queries,
                qkv::tile::Cache {
                    pool: &mut state.pool,
                    entry: 0,
                    offsets: &input.keys,
                },
            );
            dlayer::own(
                device,
                &state.hidden,
                (&mut rows.hidden, wave, shape.blocks),
            );
        }
        // A first span of history lands under the layer before's down; one holding the chunk's
        // own keys lands behind its layer's cache writes.
        let (narrow, wide) = attention::prompt::history(&input.prompt);
        let tables = (&input.prompt, &shape.first);
        let mut landing = attention::prompt::land(device, (&state.pool, 0), tables, 1);
        for index in 0..layers {
            let more = (layers - index - 1).min(1);
            let next = index + more;
            let loads = Loads::land(device, (model, index, next));
            let landed = attention::prompt::attend(
                device,
                (&state.pool, index * shape.pages),
                (&input.prompt, (shape.start, shape.segments), &state.queries),
                landing,
                &mut state.prompt,
            );
            let mut fused = qkv::tile::fused::Weight::new();
            let mut seeds = qkv::tile::fused::Seeds::new();
            let mut prefetch = Ahead {
                next: qkv::tile::NextFused {
                    weights: &mut weights,
                    fused: &mut fused,
                    scales: &mut seeds,
                    layer: (&model.layers, next),
                },
                landing: landed,
                span: (
                    (&state.pool, next * shape.pages),
                    tables,
                    (narrow, wide, more),
                ),
            };
            tail(
                device,
                (model, index, chip, loads),
                (&state.prompt, shape.blocks),
                &mut rows,
                &mut prefetch,
            );
            let landed = prefetch.landing;
            for _ in 0..more {
                weights.spread_scales(device);
                qkv::tile::chunk_fused(
                    device,
                    (&weights, (&fused, &seeds)),
                    (&mut rows.next, chip, wave, shape.blocks),
                    &input.rotation,
                    &mut state.queries,
                    qkv::tile::Cache {
                        pool: &mut state.pool,
                        entry: next * shape.pages,
                        offsets: &input.keys,
                    },
                );
            }
            landing = landed.land(
                device,
                (&state.pool, next * shape.pages),
                tables,
                (narrow, wide, more),
            );
        }
        for _ in 0..count.min(1) {
            dlayer::join(
                device,
                &rows.hidden,
                (&mut state.hidden, wave, shape.blocks),
            );
        }
    }
}

/// The next layer's Q/K/V weights, and its first span of history when `span`'s count is 1, landed
/// under the down.
struct Ahead<'a> {
    next: qkv::tile::NextFused<'a>,
    landing: attention::prompt::Landing,
    span: (
        (&'a attention::Pool, usize),
        (&'a attention::prompt::Plan, &'a FirstSpans),
        (usize, usize, usize),
    ),
}

impl dlayer::Prefetch for Ahead<'_> {
    fn land(&mut self, device: &mut Device) {
        self.next.land(device);
        let (cache, tables, counts) = self.span;
        self.landing.land_ahead(device, cache, tables, counts);
    }
}

/// Layer `index`'s tail weights and scales that do not wait for its attention, landed ahead of it
/// in the transfer queue's idle time while its first span decodes: behind the attention's last
/// store, they would delay phase O's first heads.
struct Loads {
    output: scale::Seeds<Chips>,
    gain: dlayer::PostNorm,
    mlp: scale::Seeds<Chips>,
    next_gain: dlayer::PostNorm,
    qkv: scale::Seeds<Chips>,
}

impl Loads {
    fn land(device: &mut Device, (model, index, next): (&Recipe<'_>, usize, usize)) -> Self {
        let layers = &model.layers;
        let seeds = |device: &mut Device, row| {
            let mut seeds = scale::Seeds::new();
            seeds.land_into(device, row);
            seeds
        };
        Self {
            output: seeds(device, layers.output_scales.at(index)),
            gain: dlayer::post_norm(device, layers, index),
            mlp: seeds(device, layers.mlp_scales.at(index)),
            next_gain: dlayer::input_norm(device, layers, next),
            qkv: seeds(device, layers.qkv_scales.at(next)),
        }
    }

    /// The scales spread over every slice, once the attention leaves the queues.
    fn spread(seeds: &scale::Seeds<Chips>, device: &mut Device) -> scale::Scales<Chips> {
        let mut scales = scale::Scales::new();
        scales.spread(device, seeds);
        scales
    }
}

/// Layer `index`'s D-sharded tail over the chunk's attended rows: O, residual, post-attention
/// norm, gate/up, down, residual, and layer `next`'s input norm into `rows.next`, landing what
/// `prefetch` holds under its down.
fn tail<P: dlayer::Prefetch>(
    device: &mut Device,
    (model, index, chip, loads): (&Recipe<'_>, usize, usize, Loads),
    (prompt, blocks): (&attention::prompt::Attended, usize),
    rows: &mut Rows,
    prefetch: &mut P,
) {
    let layers = &model.layers;
    let mut projection = oblock::Projection::of(Loads::spread(&loads.output, device));
    let factors = projection.factors(device);
    let (gain, next_gain) = (loads.gain, loads.next_gain);
    let scales = Loads::spread(&loads.mlp, device);
    let qkv_scales = Loads::spread(&loads.qkv, device);
    let weight = dshard::Weight::new();
    let divisor = weight.divisor::<dshard::Cluster, dshard::Landing>(device);
    let quantize = crate::ops::linear::Affine::new(
        (),
        scales.get::<dpanel::Cluster, dpanel::Reduced>(device, 0),
    );
    let next_quantize = crate::ops::linear::Affine::new(
        (),
        qkv_scales.get::<dpanel::Cluster, dpanel::Reduced>(device, 0),
    );
    let parts = dlayer::Tail {
        output: &weight,
        divisor: &divisor,
        gain: &gain,
        quantize: &quantize,
        next: (&next_gain, &next_quantize),
    };
    let mut heads = oblock::Heads::new();
    let staged = dlayer::first_heads(
        device,
        prompt,
        (&factors, &mut heads),
        (&mut projection, (layers, index)),
        chip,
    );
    // SAFETY: phase O reads every block of the layer input before phase D writes the output into
    // the same rows.
    let hidden: dpanel::Owned = unsafe { rows.hidden.view().alias() };
    dlayer::layer_seq(
        device,
        prefetch,
        (&parts, (layers, index), &scales),
        ((&projection, &factors), staged, &heads),
        (prompt, &hidden),
        (&mut rows.residuals, &mut rows.blocked, &mut rows.hidden),
        (blocks, &mut rows.next, chip),
    );
}
