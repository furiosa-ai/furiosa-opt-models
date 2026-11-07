// The row-parallel body: a slice holds four rows of one KV head's four query heads against every
// key of the span, so a span lands once a cluster and every softmax step stays inside a slice.
type Cluster = m![Heads / 2];
use crate::Module;
use crate::compose::{Tile, Tiles};
use crate::ops::attention::gqa::{PagedAttention, completion, value, visibility};
use crate::ops::layout::Dm;

type Space = Tile<Queries, Queries>;

impl completion::Accumulation for Space {
    type Groups = m![Queries % 4, Groups];
    type Channel = m![Channel];
    type Term = Term;
    type Terms = m![Term];
}

impl value::product::Mapping for Space {
    type Channel = m![Channel];
}

impl visibility::Mapping for Space {
    type Rows = m![Queries];
    type Groups = m![Groups];
    type Keys = m![Keys];
}

impl visibility::Causal for Space {
    type Time = m![Queries % 2, Groups, Queries % 4 / 2, Keys / 8];
    type Packet = m![Keys % 8];
    type Tags = m![Queries % 2, Groups, Queries % 4 / 2, Keys];
}
/// A KV head's 128 slices, four rows each.
type Slices = m![Heads % 2, Queries / 4];
/// A span spread over its head's slices, four consecutive keys a slice.
type Spread = m![Heads % 2, Keys / 4];
/// The spread as the pool's pages land: a page's sixteen tokens fill four slices.
type Landing = m![Heads % 2, Keys / 16, Page / 4];
/// A slice's keys and values of its head as the pool stores them, in the first of the landing
/// buffer's 32 parts, so the buffer keeps the other geometries' size.
pub type Landed = DmTensor<f8e4m3, Chip, Cluster, Landing, m![Part, Page % 4, Kv, Head]>;
/// A slice's decoded keys, eight channels of its four keys a run, as the scores stream them.
type Key = DmTensor<bf16, Chip, Cluster, Spread, m![Head / 8, Keys % 4, Head % 8]>;
/// A slice's decoded values, each followed by its channels of ones.
type Value = DmTensor<bf16, Chip, Cluster, Spread, m![Keys % 4, Channel]>;
/// A slice's rows of its KV head's query heads, each lane's queries contiguous.
type Query = DmTensor<bf16, Chip, Cluster, Slices, m![Queries % 2, Groups, Queries % 4 / 2, Head]>;
/// The slice's queries, eight a lane.
type QueryRegister =
    TrfTensor<bf16, Chip, Cluster, Slices, m![Queries % 2, Groups], m![Queries % 4 / 2, Head]>;
/// Every query's raw products with every key of the span.
type Scores = DmTensor<f32, Chip, Cluster, Slices, m![Queries % 2, Groups, Queries % 4 / 2, Keys]>;
/// How far each key lies past each row's last visible key: positive keys are absent.
type Excess = DmTensor<bf16, Chip, Cluster, Slices, m![Queries % 4, Keys]>;
type Tags = VrfTensor<i4, Chip, Cluster, Slices, m![Queries % 2, Groups, Queries % 4 / 2, Keys]>;
type Masked<'l> = VectorTensor<
    'l,
    { Tu::Main },
    stage::Narrow,
    f32,
    Chip,
    Cluster,
    Slices,
    m![Queries % 2, Groups, Queries % 4 / 2, Keys / 4],
    m![Keys % 4],
    Fresh,
    { stage::VeOrder::IntraFirst },
    stage::Standalone,
    { stage::Way::Way4 },
>;
type Peak = DmTensor<f32, Chip, Cluster, Slices, m![Queries % 2, Groups, Queries % 4 / 2, 1 # 2]>;
/// Every query's peak, as a vector operand.
type Base = VrfTensor<f32, Chip, Cluster, Slices, m![Queries % 2, Groups, Queries % 4 / 2, 1 # 2]>;
type Bound = VrfTensor<f32, Chip, Cluster, Slices, m![Queries % 4, 1 # 8]>;
/// The rows' limits as they land, eight copies a row.
type Limits = DmTensor<f32, Chip, Cluster, Slices, m![Queries % 4, Lane]>;
type Positions = DmTensor<f32, Chip, Cluster, Slices, m![Keys]>;
/// Each key's index within the span, four consecutive keys a slice as the span spreads.
type Spaced = DmTensor<f32, Chip, Cluster, Spread, m![Keys % 4]>;
/// The block's weights, each lane's queries contiguous so the register fills by one run a lane.
type Weights =
    DmTensor<bf16, Chip, Cluster, Slices, m![Queries % 2, Groups, Queries % 4 / 2, Keys]>;
/// The block's weights, eight queries a lane.
type WeightRegister =
    TrfTensor<bf16, Chip, Cluster, Slices, m![Queries % 2, Groups], m![Queries % 4 / 2, Keys]>;
/// Every query's value product over the span, the ones channels' its mass.
type Output = DmTensor<f32, Chip, Cluster, Slices, m![Queries % 4, Groups, Channel]>;
/// Every query's mass, as a vector operand.
type Operand = VrfTensor<f32, Chip, Cluster, Slices, m![Queries % 4, Groups, 1 # 8]>;
type Normalized = DmTensor<f32, Chip, Cluster, Slices, m![Queries % 4, Groups, Head]>;
/// Byte offset of each of a span's key pages within its layer's pool pages.
pub type Pages = HbmTensor<i32, Chip, m![Keys / 16]>;
/// Every block's value product over the spans attended so far.
type Products = DmTensor<f32, Chip, Cluster, Slices, m![QueryBlocks, Queries % 4, Groups, Channel]>;
/// The first span's peaks of every block.
type Bases = DmTensor<
    f32,
    Chip,
    Cluster,
    Slices,
    m![QueryBlocks, Queries % 2, Groups, Queries % 4 / 2, 1 # 2],
>;

/// The segments' spans in this geometry: positions `Keys * t..Keys * (t + 1)` of a segment's
/// sequence form its span `t`.
#[derive(DeviceSend)]
pub struct Tables {
    /// The spans segment `s` attends at entry `s`, 0 when it attends in the other geometry.
    pub spans: PerSegment,
    /// The leading spans whose every key each row of segment `s` sees, at entry `s`.
    pub full: PerSegment,
    /// The spans segment `s` reaches before its first row's span, at entry `s`: block `b` attends
    /// the spans up to `b` past them, the blocks before `t` less them end before span `t` begins.
    pub ahead: PerSegment,
    /// Segment `s`'s span `t` pages at entry `s * Spans + t`, read through the slice table as the
    /// job plan's are.
    pub pages: Slice<Pages>,
    /// The last key of its segment's span `t` that chunk row `r` sees at `[t][r]`, -1 when it sees
    /// none, in every lane.
    pub limits: HbmTensor<f32, Chip, m![Spans, Prompt, Lane]>,
    /// Each key's index within its span.
    pub positions: HbmTensor<f32, Chip, m![Keys]>,
}

/// The largest weight: a later span's key far above the first span's peak saturates here rather
/// than overflow the mass and the value product (e^70).
const CEILING: f32 = 2.5e30;
const ABSENT: TagGuard =
    TagGuard::matches([BitReq::One, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]);
const CLEARED: TagGuard =
    TagGuard::matches([BitReq::Zero, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]);
const SECOND_CLEARED: TagGuard =
    TagGuard::matches([BitReq::Ignore, BitReq::Zero, BitReq::Ignore, BitReq::Ignore]);

/// Gathers into `landed` the keys and values of the span whose pages `pages` lists.
pub fn land(
    context: &mut Device,
    (pool, entry): (&Pool, usize),
    pages: &Pages,
    landed: &mut Landed,
) {
    let first = landed
        .view_mut()
        .tile::<m![Part], 1, m![Part = 1 #{!} 32, Page % 4, Kv, Head]>(0);
    // SAFETY: the host plan bounds every page by the layer's pages, and each offset starts one.
    unsafe {
        pool.gather::<m![1], m![Page, Heads, Kv, Head]>()
            .entry(entry)
            .by_byte_offsets(pages)
            .unchecked()
    }
    .to_dm_view(&mut context.tdma, first);
}

/// The chunk's keys and values of the span being attended, the pair's queries and limits, and every
/// row's first-span peaks and product over the spans attended so far.
pub struct Sequence {
    landed: Landed,
    keys: Key,
    values: Value,
    positions: Positions,
    queries: Query,
    limits: Limits,
    bases: Bases,
    sums: Products,
}

impl Sequence {
    /// The sequence whose first span `landed` holds.
    pub fn new(landed: Landed) -> Self {
        Self {
            landed,
            keys: Key::new(),
            values: Value::new(),
            positions: Positions::new(),
            queries: Query::new(),
            limits: Limits::new(),
            bases: Bases::new(),
            sums: Products::new(),
        }
    }

    /// The landing buffer, for the other geometries' sequences.
    pub fn into_landed(self) -> Landed {
        self.landed
    }

    /// Gathers the span whose pages `pages` lists as a segment's first.
    pub fn land_first(
        &mut self,
        context: &mut Device,
        (pool, entry): (&Pool, usize),
        pages: &Pages,
    ) {
        land(context, (pool, entry), pages, &mut self.landed);
    }

    /// The `blocks` blocks of the segment at entry `segment` over its `spans` spans into `output`,
    /// the `full` leading spans seen whole by every row: each block attends the spans up to its
    /// own last row's, `ahead` spans past its index, and the blocks before span `t` less `ahead`
    /// skip span `t`, whose keys all lie past their rows. A small span has no half geometry to
    /// hand its last span to: `LAST` is false and the last-span count 0.
    pub fn attend_segment<const LAST: bool>(
        &mut self,
        context: &mut Device,
        (pool, entry): (&Pool, usize),
        (plan, (segment, start, first), queries): (&Tables, (usize, usize, usize), &qkv::Queries),
        ((spans, full, ahead, blocks), _): ((usize, usize, usize, usize), usize),
        output: &mut Attended,
    ) {
        const { assert!(!LAST) };
        for span in 0..spans {
            let from = span - span.min(ahead);
            // `min` gives the loop end a finite envelope; a plan's blocks never exceed it.
            let pairs = (blocks - from).min(QueryBlocks::SIZE);
            for at in 0..pairs {
                let block = from + at;
                self.attend(
                    context,
                    (pool, entry),
                    (plan, (segment, start, first), queries),
                    (span, spans, full, ahead),
                    (at, pairs, block, (block + ahead).min(spans - 1) - span),
                    output,
                );
            }
        }
    }

    /// Block `block`'s attended rows of both KV heads over span `span` of `spans` into `output`,
    /// the block the span's `at`-th of `pairs` and followed by `after` spans of its own, for the
    /// segment at entry `segment` whose first chunk row is `first`, the chunk's first capacity row
    /// `start`, the blocks before span `t` less `ahead` skipping it; the span's first block decodes
    /// the span's keys and values for the layer.
    pub fn attend(
        &mut self,
        context: &mut Device,
        (pool, entry): (&Pool, usize),
        (plan, (segment, start, first), queries): (&Tables, (usize, usize, usize), &qkv::Queries),
        (span, spans, full, ahead): (usize, usize, usize, usize),
        (at, pairs, block, after): (usize, usize, usize, usize),
        output: &mut Attended,
    ) {
        let row = first + Queries::SIZE * block;
        // Each span's decode writes the value channels alone, so the ones hold for the segment,
        // as do the keys' positions; the segment's first pair lands its own queries and limits,
        // every later pair's land under the pair before it.
        for _ in 0..(1 - at.min(1)) * (1 - span.min(1)) {
            self.values
                .view_mut()
                .memset(const { bf16::from_f32(1.0) }, &mut context.sub);
            positions(context, plan, &mut self.positions);
            query_into(context, queries, start + row, &mut self.queries);
            limits_into(context, plan, (span, row), &mut self.limits);
        }
        for _ in 0..1 - at.min(1) {
            keys(context, &self.landed, &mut self.keys);
            values(context, &self.landed, &mut self.values);
        }
        // Behind the block's own transfers, once this span's keys and values are decoded.
        for _ in 0..(1 - at.min(1)) * (spans - 1 - span).min(1) {
            let pages = &plan.pages[segment * Spans::SIZE + span + 1];
            land(context, (pool, entry), pages, &mut self.landed);
        }
        // Every row of block `block` sees every key of the `full + block` leading spans.
        let whole = (full + block - span.min(full + block)).min(1);
        // The pair after this one: the span's next block, or the next span's first block, whose
        // queries stay when it is this block again.
        let within = (pairs - 1 - at).min(1);
        let onward = (1 - within) * (spans - 1 - span).min(1);
        let span_next = span + 1 - within;
        let from_next = span_next - span_next.min(ahead);
        let block_next = within * (block + 1) + (1 - within) * from_next;
        let reload = within + onward * (block - from_next.min(block)).min(1);
        let next = first + Queries::SIZE * block_next;
        // The queries fill under the mask, which needs no scores; the next pair's inputs land
        // behind them, ahead of this block's row store on the transfer queue.
        let register = query_register(context, &self.queries);
        let limit = bound(context, &self.limits);
        for _ in 0..reload {
            query_into(context, queries, start + next, &mut self.queries);
        }
        for _ in 0..within + onward {
            limits_into(context, plan, (span_next, next), &mut self.limits);
        }
        // A pair whose every row sees every key needs no mask.
        let later = span.min(1);
        let place = (after, block, row);
        for _ in 0..(1 - whole) * (1 - later) {
            let tags = visibility(context, (&self.positions, &limit));
            self.block::<false>(context, (&register, Some(&tags)), place, output);
        }
        for _ in 0..(1 - whole) * later {
            let tags = visibility(context, (&self.positions, &limit));
            self.block::<true>(context, (&register, Some(&tags)), place, output);
        }
        for _ in 0..whole * (1 - later) {
            self.block::<false>(context, (&register, None), place, output);
        }
        for _ in 0..whole * later {
            self.block::<true>(context, (&register, None), place, output);
        }
    }

    /// The block's rows over the span, a `LATER` span's weights above the first span's peaks, into
    /// `output` from chunk row `row`, its queries in `register` and its visibility in `tags`, every
    /// key seen without them.
    fn block<const LATER: bool>(
        &mut self,
        context: &mut Device,
        (register, tags): (&QueryRegister, Option<&Tags>),
        (after, block, row): (usize, usize, usize),
        output: &mut Attended,
    ) {
        let mut raw = Scores::new();
        scores(context, (&self.keys, register), &mut raw);
        // A later span's weights stand above the first span's peaks.
        let top = if LATER {
            first_peaks(context, &self.bases, block)
        } else {
            let peak = match tags {
                Some(tags) => peaks(context, (&raw, tags)),
                None => whole_peaks(context, &raw),
            };
            for _ in 0..after.min(1) {
                record_base(context, &peak, &mut self.bases, block);
            }
            top(context, &peak)
        };
        let mut weights = Weights::new();
        match tags {
            Some(tags) => exponents(context, (&raw, tags, &top), &mut weights),
            None => whole_exponents(context, (&raw, &top), &mut weights),
        }
        drop(raw);
        drop(top);
        let register = weight_register(context, &weights);
        let mut product = Output::new();
        weighted(context, (&self.values, &register), &mut product);
        if LATER {
            self.add(context, &product, (block, after, row), output);
        } else {
            self.first(context, &product, (block, after, row), output);
        }
    }

    /// The first span's rows from chunk row `row`: alone, the product divided by the mass; with
    /// later spans to come, the product as it is.
    fn first(
        &mut self,
        context: &mut Device,
        product: &Output,
        (block, after, row): (usize, usize, usize),
        output: &mut Attended,
    ) {
        for _ in 0..1 - after.min(1) {
            normalize(context, product, rows_mut(output, row));
        }
        for _ in 0..after.min(1) {
            record_rows(context, product, &mut self.sums, block);
        }
    }

    /// A later span's product added to the earlier spans'; the last span, with `after` zero spans
    /// after it, divides the sum into the rows from chunk row `row`.
    fn add(
        &mut self,
        context: &mut Device,
        product: &Output,
        (block, after, row): (usize, usize, usize),
        output: &mut Attended,
    ) {
        let earlier = earlier_rows(context, &self.sums, block);
        let sum: Output = PagedAttention::<
            Heads,
            Groups,
            Head,
            { crate::models::nemo::SCORE_SCALE },
            Chip,
        >::default()
        .forward((
            &mut *context,
            Dm {
                tiles: Tiles::<Space>::over(Queries::SIZE),
                value: ((earlier.view(), product.view()),),
            },
        ))
        .value;
        for _ in 0..1 - after.min(1) {
            normalize(context, &sum, rows_mut(output, row));
        }
        for _ in 0..after.min(1) {
            record_rows(context, &sum, &mut self.sums, block);
        }
    }
}

/// The limits in span `span` of the block from chunk row `row` into `limits`, in the slices of
/// their rows.
fn limits_into(
    context: &mut Device,
    plan: &Tables,
    (span, row): (usize, usize),
    limits: &mut Limits,
) {
    let rows = plan
        .limits
        .view()
        .tile::<m![Spans], 1, m![Spans = 1 # Spans, Prompt, Lane]>(span)
        .tile::<m![Prompt], { Queries::SIZE }, m![Spans = 1 # Spans, { Block }, Lane]>(row);
    // SAFETY: the tile keeps the chunk's row stride; only its row axis is renamed.
    let rows = unsafe { rows.reshape::<Chip, m![Spans = 1 # Spans, Queries # Prompt, Lane]>() };
    rows.to_dm_view(&mut context.tdma, limits.view_mut());
}

/// Every key's index within the span into every slice: each slice of the spread lands its four
/// keys' and the switch gathers them past every slice of their head, so the table crosses from
/// memory once a head rather than once a slice.
fn positions(context: &mut Device, plan: &Tables, positions: &mut Positions) {
    let spaced: Spaced = plan.positions.view().to_dm(&mut context.tdma);
    context
        .main
        .begin(spaced.view())
        .fetch::<m![1], m![Keys % 4]>()
        .switch::<Slices, m![Keys / 4]>(SwitchConfig::Broadcast01 {
            slice1: 128,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![Keys / 4], m![Keys % 4 # 8]>()
        .commit_trim::<m![Keys % 4]>()
        .commit_view(positions.view_mut());
}

/// The rows' limits as a vector operand.
fn bound(context: &mut Device, limits: &Limits) -> Bound {
    // SAFETY: `Lane` and `1 # 8` lay out the same eight cells per row.
    let limit = unsafe {
        limits
            .view()
            .reshape::<Chip, Cluster, Slices, m![Queries % 4, 1 # 8]>()
    };
    context
        .sub
        .begin(limit)
        .fetch::<m![Queries % 4], m![1 # 8]>()
        .collect::<m![Queries % 4], m![1 # 8]>()
        .to_vrf()
}

/// The block's visibility of the span's keys, from the keys' positions and the rows' limits.
fn visibility(context: &mut Device, (positions, limit): (&Positions, &Bound)) -> Tags {
    let mut excess = Excess::new();
    mask(context, (positions, limit), &mut excess);
    tags(context, &excess)
}

/// Every key's excess over every row's limit into `excess`.
fn mask(context: &mut Device, (positions, limit): (&Positions, &Bound), excess: &mut Excess) {
    context
        .main
        .begin(positions.view())
        .fetch::<m![Queries % 4, Keys / 8], m![Keys % 8]>()
        .collect::<m![Queries % 4, Keys / 8], m![Keys % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Queries % 4, Keys / 4], m![Keys % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, limit)
        .vector_widen_concat::<m![Queries % 4, Keys / 8], m![Keys % 8]>()
        .vector_final()
        .cast::<bf16, m![Keys % 8 # 16]>()
        .commit_trim::<m![Keys % 8]>()
        .commit_view(excess.view_mut());
}

/// The block's visibility as execution tags: set where a row does not see the key.
fn tags(context: &mut Device, excess: &Excess) -> Tags {
    PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default()
        .forward((
            context,
            Dm::<Space, _> {
                tiles: Tiles::over(Queries::SIZE),
                value: excess,
            },
        ))
        .value
}

/// The spans' value product so far of block `block`.
fn earlier_rows(context: &mut Device, sums: &Products, block: usize) -> Output {
    let cells = sums.view().tile::<m![QueryBlocks], 1, m![
        QueryBlocks = 1 # QueryBlocks,
        Queries % 4,
        Groups,
        Channel
    ]>(block);
    context
        .main
        .begin(cells)
        .fetch::<m![Queries % 4, Groups, Channel / 8], m![Channel % 8]>()
        .collect::<m![Queries % 4, Groups, Channel / 8], m![Channel % 8]>()
        .commit_trim::<m![Channel % 8]>()
        .commit()
}

/// Keeps `product` as the spans' value product so far of block `block`.
fn record_rows(context: &mut Device, product: &Output, sums: &mut Products, block: usize) {
    let cells = sums.view_mut().tile::<m![QueryBlocks], 1, m![
        QueryBlocks = 1 #{!} QueryBlocks,
        Queries % 4,
        Groups,
        Channel
    ]>(block);
    context
        .main
        .begin(product.view())
        .fetch::<m![Queries % 4, Groups, Channel / 8], m![Channel % 8]>()
        .collect::<m![Queries % 4, Groups, Channel / 8], m![Channel % 8]>()
        .commit_trim::<m![Channel % 8]>()
        .commit_view(cells);
}

/// The rows of the block from chunk row `row`, to write.
fn rows_mut(
    output: &mut Attended,
    row: usize,
) -> HbmTensorViewMut<'_, f32, Chip, m![Queries #{!} Prompt, Heads, Groups, Head]> {
    let rows = output
        .view_mut()
        .tile::<m![Prompt], { Queries::SIZE }, m![{ BlockMut }, Heads, Groups, Head]>(row);
    // SAFETY: the block keeps the chunk's row stride; only its row axis is renamed.
    unsafe { rows.reshape::<Chip, m![Queries #{!} Prompt, Heads, Groups, Head]>() }
}

/// `product`'s value channels divided by its mass, stored into `rows`.
fn normalize(
    context: &mut Device,
    product: &Output,
    rows: HbmTensorViewMut<'_, f32, Chip, m![Queries #{!} Prompt, Heads, Groups, Head]>,
) {
    let ones = product
        .view()
        .tile::<m![Channel], 8, m![Queries % 4, Groups, Channel = 8 # Channel]>(Head::SIZE);
    // SAFETY: every ones channel's product is the row's mass, so the first stands for all eight.
    let mass = unsafe {
        ones.reshape::<Chip, Cluster, Slices, m![Queries % 4, Groups, [1 # 8] # Channel]>()
    };
    let total: Operand = context
        .sub
        .begin(mass)
        .fetch::<m![Queries % 4, Groups], m![1 # 8]>()
        .collect::<m![Queries % 4, Groups], m![1 # 8]>()
        .to_vrf();
    let values = product
        .view()
        .tile::<m![Channel], { Head::SIZE }, m![Queries % 4, Groups, Channel = 128 # Channel]>(0);
    // SAFETY: the value channels are the first channels in order; only the axis is renamed.
    let values = unsafe {
        values.reshape::<Chip, Cluster, Slices, m![Queries % 4, Groups, Head # Channel]>()
    };
    let product = context
        .main
        .begin(values)
        .fetch::<m![Queries % 4, Groups, Head / 8], m![Head % 8]>()
        .collect::<m![Queries % 4, Groups, Head / 8], m![Head % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Queries % 4, Groups, Head / 4], m![Head % 4]>();
    let attended: Normalized =
        PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default(
        )
        .forward(Dm::<Space, _> {
            tiles: Tiles::over(Queries::SIZE),
            value: (product, (&total,)),
        })
        .value
        .vector_widen_concat::<m![Queries % 4, Groups, Head / 8], m![Head % 8]>()
        .vector_final()
        .commit_trim::<m![Head % 8]>()
        .commit();
    attended.view().to_hbm_view(&mut context.tdma, rows);
}

/// Stores the first span's peaks of block `block`.
fn record_base(context: &mut Device, peak: &Peak, bases: &mut Bases, block: usize) {
    let cells = bases.view_mut().tile::<m![QueryBlocks], 1, m![
        QueryBlocks = 1 #{!} QueryBlocks,
        Queries % 2,
        Groups,
        Queries % 4 / 2,
        1 # 2
    ]>(block);
    context
        .main
        .begin(peak.view())
        .fetch::<m![Queries % 2, Groups / 2], m![Groups % 2, Queries % 4 / 2, 1 # 2]>()
        .collect::<m![Queries % 2, Groups / 2], m![Groups % 2, Queries % 4 / 2, 1 # 2]>()
        .commit_trim::<m![Groups % 2, Queries % 4 / 2, 1 # 2]>()
        .commit_view(cells);
}

/// The first span's peaks of block `block`, as a vector operand.
fn first_peaks(context: &mut Device, bases: &Bases, block: usize) -> Base {
    let cells = bases.view().tile::<m![QueryBlocks], 1, m![
        QueryBlocks = 1 # QueryBlocks,
        Queries % 2,
        Groups,
        Queries % 4 / 2,
        1 # 2
    ]>(block);
    context
        .sub
        .begin(cells)
        .fetch::<m![Queries % 2, Groups / 2], m![Groups % 2, Queries % 4 / 2, 1 # 2]>()
        .collect::<m![Queries % 2, Groups / 2], m![Groups % 2, Queries % 4 / 2, 1 # 2]>()
        .to_vrf()
}

/// The block's queries in the register file, eight a lane.
fn query_register(context: &mut Device, queries: &Query) -> QueryRegister {
    context
        .sub
        .begin(queries.view())
        .fetch::<m![Queries % 2, Groups, Queries % 4 / 2, Head / 16], m![Head % 16]>()
        .collect::<m![Queries % 2, Groups, Queries % 4 / 2, Head / 16], m![Head % 16]>()
        .to_trf()
}

/// The landed keys or values of the span, by key.
fn tokens(
    landed: &Landed,
    kv: usize,
) -> DmTensorView<'_, f8e4m3, Chip, Cluster, Spread, m![Part = 1 # 32, Keys % 4, Kv = 1 # 2, Head]>
{
    // SAFETY: a page's sixteen tokens are its keys in order.
    unsafe {
        landed
            .view()
            .reshape::<Chip, Cluster, Spread, m![Part, Keys % 4, Kv, Head]>()
    }
    .tile::<m![Part], 1, m![Part = 1 # 32, Keys % 4, Kv, Head]>(0)
    .tile::<m![Kv], 1, m![Part = 1 # 32, Keys % 4, Kv = 1 # 2, Head]>(kv)
}

/// Decoded keys with every non-finite cell cleared, since the scores read unwritten keys too; the
/// fetch widens each 8-bit float to f32 itself, so no decode table loads ahead of the pass.
fn keys(context: &mut Device, landed: &Landed, key: &mut Key) {
    context
        .main
        .begin(tokens(landed, 0))
        .fetch::<m![Keys % 4, Head / 8], m![Head % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Keys % 4, Head / 8], m![Head % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Comparison([
            Cmp::Less(f32::INFINITY),
            Cmp::Greater(f32::NEG_INFINITY),
            Cmp::False,
            Cmp::False,
        ]))
        .vector_logic(
            LogicBinaryOpF32::BitAnd,
            Branched::imm(CLEARED, 0.0f32).imm(SECOND_CLEARED, 0.0f32),
        )
        .vector_final()
        .cast::<bf16, m![Head % 8 # 16]>()
        .commit_trim::<m![Head % 8]>()
        .commit_view(key.view_mut());
}

/// Decoded values with every non-finite cell cleared: an unwritten page may hold NaN, and a zero
/// weight times NaN is NaN.
fn values(context: &mut Device, landed: &Landed, value: &mut Value) {
    let channels = value
        .view_mut()
        .tile::<m![Channel], { Head::SIZE }, m![Keys % 4, Channel = 128 #{!} Channel]>(0);
    // SAFETY: the value channels are the first channels in order; only the axis is renamed.
    let channels =
        unsafe { channels.reshape::<Chip, Cluster, Spread, m![Keys % 4, Head #{!} Channel]>() };
    context
        .main
        .begin(tokens(landed, 1))
        .fetch::<m![Keys % 4, Head / 8], m![Head % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Keys % 4, Head / 8], m![Head % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Comparison([
            Cmp::Less(f32::INFINITY),
            Cmp::Greater(f32::NEG_INFINITY),
            Cmp::False,
            Cmp::False,
        ]))
        .vector_logic(
            LogicBinaryOpF32::BitAnd,
            Branched::imm(CLEARED, 0.0f32).imm(SECOND_CLEARED, 0.0f32),
        )
        .vector_final()
        .cast::<bf16, m![Head % 8 # 16]>()
        .commit_trim::<m![Head % 8]>()
        .commit_view(channels);
}

/// The block's queries from capacity row `first` into `query`: a slice's four rows of its KV
/// head's query heads.
fn query_into(context: &mut Device, queries: &qkv::Queries, first: usize, query: &mut Query) {
    // SAFETY: a capacity row's query heads are its `Heads × Groups` heads in head order.
    let rows = unsafe {
        queries
            .view()
            .reshape::<Chip, m![Capacity, Heads, Groups, Head]>()
    }
    .tile::<m![Capacity], { Queries::SIZE }, m![{ CapacityBlock }, Heads, Groups, Head]>(first);
    // SAFETY: the block keeps the capacity stride; only its row axis is renamed.
    unsafe { rows.reshape::<Chip, m![Queries # Capacity, Heads, Groups, Head]>() }
        .to_dm_view(&mut context.tdma, query.view_mut());
}

/// Every query's raw products with every key of the span: the keys stream from the slices that
/// hold them past every slice of their head, four keys a packet, against the queries held in the
/// register.
fn scores(context: &mut Device, (key, queries): (&Key, &QueryRegister), scores: &mut Scores) {
    PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default()
        .forward((
            context,
            Dm::<Space, _> {
                tiles: Tiles::over(Queries::SIZE),
                value: (key.view(), queries, scores.view_mut()),
            },
        ));
}

/// The scores as the softmax passes read them: keys a row does not see at the lowest bf16.
fn masked<'l>(context: &'l mut Device, (scores, tags): (&'l Scores, &Tags)) -> Masked<'l> {
    let tagged = context
        .main
        .begin(scores.view())
        .fetch::<m![Queries % 2, Groups, Queries % 4 / 2, Keys / 8], m![Keys % 8]>()
        .collect::<m![Queries % 2, Groups, Queries % 4 / 2, Keys / 8], m![Keys % 8]>()
        .vector_init()
        .vector_tagged(tags);
    VectorTensor::<_, _, f32, _, _, _, _, _, _, _, _, _>::vector_logic_with_mode(
        tagged,
        LogicBinaryOpF32::BitAnd,
        BinaryArgMode::Mode11,
        Branched::imm(ABSENT, f32::from_bits(0xff7f_0000)),
    )
    .vector_narrow_split::<m![Queries % 2, Groups, Queries % 4 / 2, Keys / 4], m![Keys % 4]>()
}

/// Every query's scaled peak over the keys its row sees.
fn peaks(context: &mut Device, (scores, tags): (&Scores, &Tags)) -> Peak {
    PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default()
        .forward(Dm::<Space, _> {
            tiles: Tiles::over(Queries::SIZE),
            value: masked(context, (scores, tags)),
        })
        .value
        .vector_intra_slice_reduce::<Keys, m![Queries % 2, Groups, Queries % 4 / 2], m![1 # 4]>(
            IntraSliceReduceOpF32::Max,
        )
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 2]>()
        .commit()
}

/// Every query's scaled peak over every key of the span, which its row sees whole.
fn whole_peaks(context: &mut Device, scores: &Scores) -> Peak {
    let products = context
        .main
        .begin(scores.view())
        .fetch::<m![Queries % 2, Groups, Queries % 4 / 2, Keys / 8], m![Keys % 8]>()
        .collect::<m![Queries % 2, Groups, Queries % 4 / 2, Keys / 8], m![Keys % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Queries % 2, Groups, Queries % 4 / 2, Keys / 4], m![Keys % 4]>();
    PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default()
        .forward(Dm::<Space, _> {
            tiles: Tiles::over(Queries::SIZE),
            value: products,
        })
        .value
        .vector_intra_slice_reduce::<Keys, m![Queries % 2, Groups, Queries % 4 / 2], m![1 # 4]>(
            IntraSliceReduceOpF32::Max,
        )
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 2]>()
        .commit()
}

/// Every query's peak, as the exponent pass's operand.
fn top(context: &mut Device, peak: &Peak) -> Base {
    context
        .sub
        .begin(peak.view())
        .fetch::<m![Queries % 2, Groups / 2], m![Groups % 2, Queries % 4 / 2, 1 # 2]>()
        .collect::<m![Queries % 2, Groups / 2], m![Groups % 2, Queries % 4 / 2, 1 # 2]>()
        .to_vrf()
}

/// Unnormalized softmax weights into `weights`: e^(scale * product - top), zero where the row does
/// not see the key, saturating at the ceiling where a later span's key lies far above `top`.
fn exponents(
    context: &mut Device,
    (scores, tags, top): (&Scores, &Tags, &Base),
    weights: &mut Weights,
) {
    // As the top less the scaled product, negated.
    PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default()
        .forward(Dm::<Space, _> {
            tiles: Tiles::over(Queries::SIZE),
            value: (
                PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default()
                    .forward(Dm::<Space, _> {
                        tiles: Tiles::over(Queries::SIZE),
                        value: (masked(context, (scores, tags)), top),
                    })
                    .value
                    .vector_widen_concat::<m![Queries % 2, Groups, Queries % 4 / 2, Keys / 8], m![Keys % 8]>(),
                CEILING,
            ),
        })
        .value
        .vector_final()
        .cast::<bf16, m![Keys % 8 # 16]>()
        .commit_trim::<m![Keys % 8]>()
        .commit_view(weights.view_mut());
}

/// `exponents` of a span every row sees whole.
fn whole_exponents(context: &mut Device, (scores, top): (&Scores, &Base), weights: &mut Weights) {
    // As the top less the scaled product, negated.
    let scores = context
        .main
        .begin(scores.view())
        .fetch::<m![Queries % 2, Groups, Queries % 4 / 2, Keys / 8], m![Keys % 8]>()
        .collect::<m![Queries % 2, Groups, Queries % 4 / 2, Keys / 8], m![Keys % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Queries % 2, Groups, Queries % 4 / 2, Keys / 4], m![Keys % 4]>();
    PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default()
        .forward(Dm::<Space, _> {
            tiles: Tiles::over(Queries::SIZE),
            value: (
                PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default()
                    .forward(Dm::<Space, _> {
                        tiles: Tiles::over(Queries::SIZE),
                        value: (scores, top),
                    })
                    .value
                    .vector_widen_concat::<m![Queries % 2, Groups, Queries % 4 / 2, Keys / 8], m![Keys % 8]>(),
                CEILING,
            ),
        })
        .value
        .vector_final()
        .cast::<bf16, m![Keys % 8 # 16]>()
        .commit_trim::<m![Keys % 8]>()
        .commit_view(weights.view_mut());
}

/// The block's weights in the register file, eight queries a lane.
fn weight_register(context: &mut Device, weights: &Weights) -> WeightRegister {
    context
        .sub
        .begin(weights.view())
        .fetch::<m![Queries % 2, Groups, Queries % 4 / 2, Keys / 16], m![Keys % 16]>()
        .collect::<m![Queries % 2, Groups, Queries % 4 / 2, Keys / 16], m![Keys % 16]>()
        .to_trf()
}

/// The weighted values into `product`: every key's values stream from the slices that hold them
/// past every slice of their head, each packet against both halves of the register's queries.
fn weighted(
    context: &mut Device,
    (value, weights): (&Value, &WeightRegister),
    product: &mut Output,
) {
    PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default()
        .forward((
            context,
            Dm::<Space, _> {
                tiles: Tiles::over(Queries::SIZE),
                value: (value.view(), weights, (product.view_mut(),)),
            },
        ));
}
