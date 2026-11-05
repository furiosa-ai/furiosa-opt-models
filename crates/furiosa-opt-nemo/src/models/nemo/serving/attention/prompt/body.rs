type Cluster = m![Heads / 2];
use crate::Module;
use crate::compose::{Tile, Tiles};
use crate::ops::attention::gqa::{PagedAttention, completion, score, value, visibility};
use crate::ops::layout::Dm;

type Space = Tile<Queries, Queries>;

impl completion::Accumulation for Space {
    type Groups = m![Heads % 2, { Row }];
    type Channel = m![Channel];
    type Term = Term;
    type Terms = m![Term];
}

impl value::product::Mapping for Space {
    type Channel = m![Channel];
}

impl value::product::Partitioned for Space {
    type Row = Row;
    type RowGroup = RowGroup;
    type Span = m![Keys % { SLICE_KEYS }];
}

impl visibility::Mapping for Space {
    type Rows = m![Queries];
    type Groups = m![Groups];
    type Keys = m![Keys];
}

impl score::query::Mapping for Space {
    type Pair = RowPair;
    type RowGroup = RowGroup;
    type Span = m![Keys % { SLICE_KEYS }];
}

impl visibility::Causal for Space {
    type Time = m![Queries % { TILE_ROWS }, Keys % { SLICE_KEYS } / 8];
    type Packet = m![Keys % 8];
    type Tags = m![Queries % { TILE_ROWS }, Keys % { SLICE_KEYS }];
}
type Slices = m![Queries / { TILE_ROWS }, Keys / { SLICE_KEYS }];
type QuerySlices = m![Queries / { TILE_ROWS }, { RowGroup }, Groups];
/// A slice's keys of both heads.
type Key = DmTensor<bf16, Chip, Cluster, Slices, m![Heads % 2, Keys % { SLICE_KEYS }, Head]>;
type Transposed =
    DmTensor<f8e4m3, Chip, Cluster, Slices, m![Heads % 2, Head, Keys % { SLICE_KEYS }]>;
/// A slice's values of both heads, one 64-key run per channel, each head's value channels followed
/// by its channels of ones.
type Value = DmTensor<bf16, Chip, Cluster, Slices, m![Heads % 2, Channel, Keys % { SLICE_KEYS }]>;
type KeyRegister = TrfTensor<
    bf16,
    Chip,
    Cluster,
    Slices,
    m![Keys % 8],
    m![
        Keys % { SLICE_KEYS } / 16,
        Head / 16,
        Keys % 16 / 8,
        Head % 16
    ],
>;
type ValueRegister = TrfTensor<
    bf16,
    Chip,
    Cluster,
    Slices,
    m![Channel % 8],
    m![
        Channel / 16,
        Keys % { SLICE_KEYS } / 16,
        Channel % 16 / 8,
        Keys % 16
    ],
>;
/// A slice's first 16 keys of one head, the sample a whole first span's peaks come from.
type SampleRegister =
    TrfTensor<bf16, Chip, Cluster, Slices, m![Keys % 8], m![Head / 16, Keys % 16 / 8, Head % 16]>;
type Sample = DmTensor<bf16, Chip, Cluster, Slices, m![Groups, Queries % { TILE_ROWS }, Keys % 16]>;
type Query = DmTensor<bf16, Chip, Cluster, QuerySlices, m![{ Row }, Head]>;
type Scores = DmTensor<
    bf16,
    Chip,
    Cluster,
    Slices,
    m![Groups, Queries % { TILE_ROWS }, Keys % { SLICE_KEYS }],
>;
type GroupScores<'l> = DmTensorView<
    'l,
    bf16,
    Chip,
    Cluster,
    Slices,
    m![Groups = 1 # 4, Queries % { TILE_ROWS }, Keys % { SLICE_KEYS }],
>;
/// How far each key of the slice lies past each row's last visible key: positive keys are absent.
type Excess =
    DmTensor<bf16, Chip, Cluster, Slices, m![Queries % { TILE_ROWS }, Keys % { SLICE_KEYS }]>;
type Tags =
    VrfTensor<i4, Chip, Cluster, Slices, m![Queries % { TILE_ROWS }, Keys % { SLICE_KEYS }]>;
type Masked<'l> = VectorTensor<
    'l,
    { Tu::Main },
    stage::Narrow,
    f32,
    Chip,
    Cluster,
    Slices,
    m![Queries % { TILE_ROWS }, Keys % { SLICE_KEYS } / 4],
    m![Keys % 4],
    Fresh,
    { stage::VeOrder::IntraFirst },
    stage::Standalone,
    { stage::Way::Way4 },
>;
type Peak = DmTensor<f32, Chip, Cluster, Slices, m![Groups, Queries % { TILE_ROWS }, 1 # 2]>;
type Top = VrfTensor<f32, Chip, Cluster, Slices, m![Queries % { TILE_ROWS }, 1 # 2]>;
type Base = VrfTensor<f32, Chip, Cluster, Slices, m![Groups, Queries % { TILE_ROWS }, 1 # 2]>;
type Bound = VrfTensor<f32, Chip, Cluster, Slices, m![Queries % { TILE_ROWS }, 1 # 8]>;
type Limit = DmTensor<f32, Chip, Cluster, Slices, m![Queries % { TILE_ROWS }, 1 # 8]>;
type Positions = DmTensor<f32, Chip, Cluster, Slices, m![Keys % { SLICE_KEYS }]>;
type PositionRegister = VrfTensor<f32, Chip, Cluster, Slices, m![Keys % { SLICE_KEYS }]>;
/// Both heads' softmax weights of the block against the slice's keys.
type Weights = DmTensor<
    bf16,
    Chip,
    Cluster,
    Slices,
    m![
        Heads % 2,
        Groups,
        Queries % { TILE_ROWS },
        Keys % { SLICE_KEYS }
    ],
>;
type HeadWeightsMut<'l> = DmTensorViewMut<
    'l,
    bf16,
    Chip,
    Cluster,
    Slices,
    m![Heads % 2 = 1 #{!} 2, Groups, Queries % { TILE_ROWS }, Keys % { SLICE_KEYS }],
>;
/// Every row's softmax mass of both heads, as a vector operand.
type Operand = VrfTensor<f32, Chip, Cluster, QuerySlices, m![Heads % 2, { Row }, 1 # 8]>;
/// Every row's value product of both heads on its query slice, the ones channels' its mass.
type Output = DmTensor<f32, Chip, Cluster, QuerySlices, m![Heads % 2, { Row }, Channel]>;
type Normalized = DmTensor<f32, Chip, Cluster, QuerySlices, m![Heads % 2, { Row }, Head]>;
/// Byte offset of each of a span's key pages within its layer's pool pages.
pub type Pages = HbmTensor<i32, Chip, m![Keys / 16]>;
/// Every block and head's value product over the spans attended so far, on its query slices.
type Products =
    DmTensor<f32, Chip, Cluster, QuerySlices, m![QueryBlocks, Heads % 2, { Row }, Channel]>;
/// The first span's peaks of every block and head, in every key slice.
type Bases = DmTensor<
    f32,
    Chip,
    Cluster,
    Slices,
    m![QueryBlocks, Heads % 2, Groups, Queries % { TILE_ROWS }, 1 # 2],
>;

/// The segments' spans in this geometry: positions `Keys * t..Keys * (t + 1)` of a segment's
/// sequence form its span `t`.
#[derive(DeviceSend)]
pub struct Tables {
    /// The spans segment `s` attends at entry `s`, 0 when it attends in the other geometry.
    pub spans: PerSegment,
    /// The leading spans whose every key each row of segment `s` sees, at entry `s`.
    pub full: PerSegment,
    /// Segment `s`'s first position in blocks, rounded up, at entry `s`: block `b`'s last row lies
    /// in span `(b + ahead) / SPAN_BLOCKS`, so the blocks before `SPAN_BLOCKS * t - ahead` skip
    /// span `t`.
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

/// The blocks whose rows one span's keys cover; `Reach` counts them over every span, the
/// envelope of `ahead`.
const SPAN_BLOCKS: usize = Keys::SIZE / Queries::SIZE;
const _: () = assert!(Keys::SIZE % Queries::SIZE == 0 && Reach::SIZE == Spans::SIZE * SPAN_BLOCKS);
/// The largest weight: a later span's key far above the first span's peak saturates here rather
/// than overflow the mass and the value product (e^70).
const CEILING: f32 = 2.5e30;
const ABSENT: TagGuard =
    TagGuard::matches([BitReq::One, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]);
const CLEARED: TagGuard =
    TagGuard::matches([BitReq::Zero, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]);
const SECOND_CLEARED: TagGuard =
    TagGuard::matches([BitReq::Ignore, BitReq::Zero, BitReq::Ignore, BitReq::Ignore]);

/// The chunk's keys and values of the span being attended, and every row's first-span peaks and
/// product over the spans attended so far.
pub struct Sequence {
    landed: Landed,
    keys: Key,
    values: Value,
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
            bases: Bases::new(),
            sums: Products::new(),
        }
    }

    /// The landing buffer, for the other geometry's sequence.
    pub fn into_landed(self) -> Landed {
        self.landed
    }

    /// The landing buffer, the first span's peaks and the product so far, for the sequence that
    /// attends a segment's last span in another geometry.
    pub fn into_parts(self) -> (Landed, Bases, Products) {
        (self.landed, self.bases, self.sums)
    }

    /// The `blocks` blocks of the segment at entry `segment` over its `spans` spans into `output`,
    /// the `full` leading spans seen whole by every row, the segment's first position `ahead`
    /// blocks rounded up: each block attends the spans up to its own last row's, and the blocks
    /// before `SPAN_BLOCKS * t - ahead` skip span `t`, whose keys all lie past their rows. With
    /// `LAST`, the `last` (0 or 1) final spans are landed for the half geometry, which attends them.
    pub fn attend_segment<const LAST: bool>(
        &mut self,
        context: &mut Device,
        (pool, entry): (&Pool, usize),
        (plan, (segment, start, first), queries): (&Tables, (usize, usize, usize), &qkv::Queries),
        ((spans, full, ahead, blocks), last): ((usize, usize, usize, usize), usize),
        output: &mut Attended,
    ) {
        for span in 0..spans - last {
            let reach = span * SPAN_BLOCKS;
            let from = reach - reach.min(ahead);
            // `min` gives the loop end a finite envelope; a plan's blocks never exceed it.
            for at in 0..(blocks - from).min(QueryBlocks::SIZE) {
                let block = from + at;
                self.attend::<LAST>(
                    context,
                    (pool, entry),
                    (plan, (segment, start, first), queries),
                    (span, (spans, last), full),
                    (
                        at,
                        block,
                        ((block + ahead) / SPAN_BLOCKS).min(spans - 1) - span,
                    ),
                    output,
                );
            }
        }
    }

    /// Block `block`'s attended rows of both KV heads over span `span` of `spans` into `output`,
    /// the `full` leading spans seen whole by every row, the block the span's `at`-th and followed
    /// by `after` spans of its own, for the segment at entry `segment` whose first chunk row is
    /// `first`, the chunk's first capacity row `start`; the span's first block gathers the span's
    /// keys and values for the layer, with `LAST` the half geometry's `last` final spans in its
    /// layout.
    pub fn attend<const LAST: bool>(
        &mut self,
        context: &mut Device,
        (pool, entry): (&Pool, usize),
        (plan, (segment, start, first), queries): (&Tables, (usize, usize, usize), &qkv::Queries),
        (span, (spans, last), full): (usize, (usize, usize), usize),
        (at, block, after): (usize, usize, usize),
        output: &mut Attended,
    ) {
        // Each span's decode writes the value channels alone, so the ones hold for the segment.
        for _ in 0..(1 - at.min(1)) * (1 - span.min(1)) {
            self.values
                .view_mut()
                .tile::<m![Channel], 16, m![Heads % 2, Channel = 16 #{!} Channel, Keys % { SLICE_KEYS }]>(Head::SIZE)
                .memset(const { bf16::from_f32(1.0) }, &mut context.sub);
        }
        // The transpose needs no lookup table, so each decode's table loads under the pass before.
        for _ in 0..1 - at.min(1) {
            let transposed = transpose(context, &self.landed);
            keys(context, &self.landed, &mut self.keys);
            values(context, &transposed, &mut self.values);
        }
        let row = first + Queries::SIZE * block;
        // Queued ahead of the block's limits, so the first head's scores need not wait for them.
        let block_queries = [
            query(context, queries, (start + row, 0)),
            query(context, queries, (start + row, 1)),
        ];
        let limit = limits(context, plan, (span, row));
        let positions: Positions = plan.positions.to_dm(&mut context.tdma);
        // Behind the block's own transfers, once this span's keys and values are decoded.
        let next = (1 - at.min(1)) * (spans - 1 - span).min(1);
        if LAST {
            let halved = (last * (1 - (spans - last - 1 - span).min(1))).min(1);
            for _ in 0..next * (1 - halved) {
                let pages = &plan.pages[segment * Spans::SIZE + span + 1];
                land(context, (pool, entry), pages, &mut self.landed);
            }
            for _ in 0..next * halved {
                let pages = &plan.pages[segment * Spans::SIZE + span + 1];
                super::half::land_wide(context, (pool, entry), pages, &mut self.landed);
            }
        } else {
            for _ in 0..next {
                let pages = &plan.pages[segment * Spans::SIZE + span + 1];
                land(context, (pool, entry), pages, &mut self.landed);
            }
        }
        let (later, whole) = (span.min(1), (full - span.min(full)).min(1));
        let masking = Some((&positions, &limit));
        let place = (after, block, row);
        for _ in 0..(1 - later) * (1 - whole) {
            self.block::<false, false>(context, (&block_queries, masking), place, output);
        }
        for _ in 0..(1 - later) * whole {
            self.sampled(context, &block_queries, place, output);
        }
        for _ in 0..later * (1 - whole) {
            self.block::<true, false>(context, (&block_queries, masking), place, output);
        }
        for _ in 0..later * whole {
            self.block::<true, true>(context, (&block_queries, None), place, output);
        }
    }

    /// The block's rows of both KV heads over the span, a `LATER` span's weights above the first
    /// span's peaks, into `output` from chunk row `row`; `masking` holds the keys' positions and
    /// the rows' limits unless every row sees every key. A `WHOLE` later span's scores are its
    /// weights already.
    fn block<const LATER: bool, const WHOLE: bool>(
        &mut self,
        context: &mut Device,
        (queries, masking): (&[Query; 2], Option<(&Positions, &Limit)>),
        (after, block, row): (usize, usize, usize),
        output: &mut Attended,
    ) {
        // One product holds both heads and their mass, so each pass after the products comes once
        // a block.
        let mut weights = Weights::new();
        let mut product = Output::new();
        // A register takes half the tensor register file: the second head's keys take the first
        // head's key half once its scores are out, its values the first head's value half.
        let key = key_register(context, (&self.keys, 0));
        let base = if LATER {
            Some(first_peaks(context, &self.bases, (block, 0)))
        } else {
            None
        };
        let value = value_register(context, (&self.values, 0));
        let mut raw = Scores::new();
        if WHOLE {
            scores(
                context,
                (&queries[0], &key),
                &base,
                head_mut(&mut weights, 0),
            );
        } else {
            scores(context, (&queries[0], &key), &base, raw.view_mut());
        }
        drop(base);
        drop(key);
        // Both heads mask the block's rows alike, and the first head's scores give the shape. Sub
        // loads in source order: the second head's keys wait for these scores, so they go last.
        let tags = match masking {
            Some((positions, limit)) => {
                let mut excess = Excess::new();
                mask(context, &raw, (positions, limit), &mut excess);
                Some(tags(context, &excess))
            }
            None => None,
        };
        let key = key_register(context, (&self.keys, 1));
        if WHOLE {
        } else {
            self.weigh::<LATER>(context, (&raw, &tags), (after, block, 0), &mut weights);
        }
        drop(raw);
        weighted(context, (&weights, 0), value, &mut product);
        // The second head's base lands under the first head's product, its values once that ends.
        let base = if LATER {
            Some(first_peaks(context, &self.bases, (block, 1)))
        } else {
            None
        };
        let value = value_register(context, (&self.values, 1));
        let mut raw = Scores::new();
        if WHOLE {
            scores(
                context,
                (&queries[1], &key),
                &base,
                head_mut(&mut weights, 1),
            );
        } else {
            scores(context, (&queries[1], &key), &base, raw.view_mut());
        }
        drop(base);
        drop(key);
        if WHOLE {
        } else {
            self.weigh::<LATER>(context, (&raw, &tags), (after, block, 1), &mut weights);
        }
        drop(raw);
        drop(tags);
        weighted(context, (&weights, 1), value, &mut product);
        if LATER {
            self.add(context, &product, (block, after, row), output);
        } else {
            self.first(context, &product, (block, after, row), output);
        }
    }

    /// The block's rows of both KV heads over a first span every row sees whole, into `output`
    /// from chunk row `row`: each head's base is scored from its keys' sample ahead of its scores,
    /// which are then its weights already.
    fn sampled(
        &mut self,
        context: &mut Device,
        queries: &[Query; 2],
        (after, block, row): (usize, usize, usize),
        output: &mut Attended,
    ) {
        let mut weights = Weights::new();
        let mut product = Output::new();
        let (key, value, base) = self.sample(context, (&queries[0], 0), (after, block));
        scores(
            context,
            (&queries[0], &key),
            &base,
            head_mut(&mut weights, 0),
        );
        drop(base);
        drop(key);
        weighted(context, (&weights, 0), value, &mut product);
        let (key, value, base) = self.sample(context, (&queries[1], 1), (after, block));
        scores(
            context,
            (&queries[1], &key),
            &base,
            head_mut(&mut weights, 1),
        );
        drop(base);
        drop(key);
        weighted(context, (&weights, 1), value, &mut product);
        self.first(context, &product, (block, after, row), output);
    }

    /// Head `head`'s keys, values and base over a whole first span: the base is each row's peak
    /// over every slice's first 16 keys (kept for later spans); the keys take the sample's half.
    fn sample(
        &mut self,
        context: &mut Device,
        (query, head): (&Query, usize),
        (after, block): (usize, usize),
    ) -> (KeyRegister, ValueRegister, Option<Base>) {
        let sample: SampleRegister = context
            .sub
            .begin(
                self.keys
                    .view()
                    .tile::<m![Heads % 2], 1, m![Heads % 2 = 1 # 2, Keys % { SLICE_KEYS }, Head]>(
                        head,
                    )
                    .tile::<m![Keys % { SLICE_KEYS } / 16], 1, m![
                        Heads % 2 = 1 # 2,
                        Keys % { SLICE_KEYS } / 16 = 1 # { SLICE_KEYS / 16 },
                        Keys % 16,
                        Head
                    ]>(0),
            )
            .fetch::<m![Keys % 8, Head / 16, Keys % 16 / 8], m![Head % 16]>()
            .collect::<m![Keys % 8, Head / 16, Keys % 16 / 8], m![Head % 16]>()
            .to_trf();
        let raw = sample_scores(context, (query, &sample));
        drop(sample);
        let value = value_register(context, (&self.values, head));
        let key = key_register(context, (&self.keys, head));
        let peak = peaks(context, raw.view());
        for _ in 0..after.min(1) {
            record_base(context, &peak, &mut self.bases, (block, head));
        }
        let base: Base = context
            .sub
            .begin(peak.view())
            .fetch::<m![Groups, Queries % { TILE_ROWS } / 4], m![Queries % 4, 1 # 2]>()
            .collect::<m![Groups, Queries % { TILE_ROWS } / 4], m![Queries % 4, 1 # 2]>()
            .to_vrf();
        (key, value, Some(base))
    }

    /// Head `head`'s weights of the block's rows into `weights` from its raw scores: the first
    /// span's exponents above its own peaks (kept for later spans), a later span's cleared where
    /// a row does not see the key.
    fn weigh<const LATER: bool>(
        &mut self,
        context: &mut Device,
        (scores, tags): (&Scores, &Option<Tags>),
        (after, block, head): (usize, usize, usize),
        weights: &mut Weights,
    ) {
        if LATER {
            match tags {
                None => {}
                Some(tags) => clear(context, (scores, tags), (weights, head)),
            }
        } else {
            let peak = match tags {
                Some(_) => visible_peaks(context, (scores, tags)),
                None => {
                    let sampled = scores.view().tile::<m![Keys % { SLICE_KEYS } / 16], 1, m![
                        Groups,
                        Queries % { TILE_ROWS },
                        Keys % { SLICE_KEYS } / 16 = 1 # { SLICE_KEYS / 16 },
                        Keys % 16
                    ]>(0);
                    peaks(context, sampled)
                }
            };
            exponents(context, (scores, tags, &peak), (weights, head));
            for _ in 0..after.min(1) {
                record_base(context, &peak, &mut self.bases, (block, head));
            }
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

/// The limits in span `span` of the block from chunk row `row`, landed in every key slice of both
/// clusters.
fn limits(context: &mut Device, plan: &Tables, (span, row): (usize, usize)) -> Limit {
    let rows = plan
        .limits
        .view()
        .tile::<m![Spans], 1, m![Spans = 1 # Spans, Prompt, Lane]>(span)
        .tile::<m![Prompt], { Queries::SIZE }, m![Spans = 1 # Spans, { Block }, Lane]>(row);
    // SAFETY: the tile keeps the chunk's row stride; only its row axis is renamed.
    let rows = unsafe { rows.reshape::<Chip, m![Spans = 1 # Spans, Queries # Prompt, Lane]>() };
    let landed: DmTensor<f32, Chip, Cluster, Slices, m![Queries % { TILE_ROWS }, Lane]> =
        rows.to_dm(&mut context.tdma);
    // SAFETY: `Lane` and `1 # 8` lay out the same eight cells per row.
    unsafe { landed.reshape() }
}

/// Every key's excess over every row's limit into `excess`; the scores only give the stream its
/// shape, as each is cleared to zero first.
fn mask(
    context: &mut Device,
    scores: &Scores,
    (positions, limit): (&Positions, &Limit),
    excess: &mut Excess,
) {
    let positions: PositionRegister = context
        .sub
        .begin(positions.view())
        .fetch::<m![Keys % { SLICE_KEYS } / 8], m![Keys % 8]>()
        .collect::<m![Keys % { SLICE_KEYS } / 8], m![Keys % 8]>()
        .to_vrf();
    let limit: Bound = context
        .sub
        .begin(limit.view())
        .fetch::<m![Queries % { TILE_ROWS }], m![1 # 8]>()
        .collect::<m![Queries % { TILE_ROWS }], m![1 # 8]>()
        .to_vrf();
    context
        .main
        .begin(group(scores, 0))
        .fetch::<m![Queries % { TILE_ROWS }, Keys % { SLICE_KEYS } / 8], m![Keys % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Queries % { TILE_ROWS }, Keys % { SLICE_KEYS } / 8], m![Keys % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, Branched::imm(CLEARED, 0.0f32))
        .vector_narrow_split::<m![Queries % { TILE_ROWS }, Keys % { SLICE_KEYS } / 4], m![Keys % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, &limit)
        .vector_widen_concat::<m![Queries % { TILE_ROWS }, Keys % { SLICE_KEYS } / 8], m![Keys % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &positions)
        .vector_final()
        .cast::<bf16, m![Keys % 8 # 16]>()
        .commit_trim::<m![Keys % 8]>()
        .commit_view(excess.view_mut());
}

/// The spans' value product so far of block `block`.
fn earlier_rows(context: &mut Device, sums: &Products, block: usize) -> Output {
    let cells = sums
        .view()
        .tile::<m![QueryBlocks], 1, m![QueryBlocks = 1 # QueryBlocks, Heads % 2, { Row }, Channel]>(
            block,
        );
    context
        .main
        .begin(cells)
        .fetch::<m![Heads % 2, { Row }, Channel / 8], m![Channel % 8]>()
        .collect::<m![Heads % 2, { Row }, Channel / 8], m![Channel % 8]>()
        .commit_trim::<m![Channel % 8]>()
        .commit()
}

/// Keeps `product` as the spans' value product so far of block `block`.
fn record_rows(context: &mut Device, product: &Output, sums: &mut Products, block: usize) {
    let cells = sums
        .view_mut()
        .tile::<m![QueryBlocks], 1, m![QueryBlocks = 1 #{!} QueryBlocks, Heads % 2, { Row }, Channel]>(block);
    context
        .main
        .begin(product.view())
        .fetch::<m![Heads % 2, { Row }, Channel / 8], m![Channel % 8]>()
        .collect::<m![Heads % 2, { Row }, Channel / 8], m![Channel % 8]>()
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
        .tile::<m![Channel], 8, m![Heads % 2, { Row }, Channel = 8 # Channel]>(Head::SIZE);
    // SAFETY: every ones channel's product is the row's mass, so the first stands for all eight.
    let mass = unsafe {
        ones.reshape::<Chip, Cluster, QuerySlices, m![Heads % 2, { Row }, [1 # 8] # Channel]>()
    };
    let total: Operand = context
        .sub
        .begin(mass)
        .fetch::<m![Heads % 2, { Row }], m![1 # 8]>()
        .collect::<m![Heads % 2, { Row }], m![1 # 8]>()
        .to_vrf();
    let values = product
        .view()
        .tile::<m![Channel], { Head::SIZE }, m![Heads % 2, { Row }, Channel = 128 # Channel]>(0);
    // SAFETY: a head's value channels are its first channels in order; only the axis is renamed.
    let values = unsafe {
        values.reshape::<Chip, Cluster, QuerySlices, m![Heads % 2, { Row }, Head # Channel]>()
    };
    let product = context
        .main
        .begin(values)
        .fetch::<m![Heads % 2, { Row }, Head / 8], m![Head % 8]>()
        .collect::<m![Heads % 2, { Row }, Head / 8], m![Head % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Heads % 2, { Row }, Head / 4], m![Head % 4]>();
    let attended: Normalized =
        PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default(
        )
        .forward(Dm::<Space, _> {
            tiles: Tiles::over(Queries::SIZE),
            value: (product, (&total,)),
        })
        .value
        .vector_widen_concat::<m![Heads % 2, { Row }, Head / 8], m![Head % 8]>()
        .vector_final()
        .commit_trim::<m![Head % 8]>()
        .commit();
    attended.view().to_hbm_view(&mut context.tdma, rows);
}

/// Stores the first span's all-reduced peaks of block `block` and head `head`.
fn record_base(
    context: &mut Device,
    peak: &Peak,
    bases: &mut Bases,
    (block, head): (usize, usize),
) {
    let cells = bases
        .view_mut()
        .tile::<m![QueryBlocks], 1, m![
            QueryBlocks = 1 #{!} QueryBlocks,
            Heads % 2,
            Groups,
            Queries % { TILE_ROWS },
            1 # 2
        ]>(block)
        .tile::<m![Heads % 2], 1, m![
            QueryBlocks = 1 #{!} QueryBlocks,
            Heads % 2 = 1 #{!} 2,
            Groups,
            Queries % { TILE_ROWS },
            1 # 2
        ]>(head);
    context
        .main
        .begin(peak.view())
        .fetch::<m![Groups, Queries % { TILE_ROWS } / 4], m![Queries % 4, 1 # 2]>()
        .collect::<m![Groups, Queries % { TILE_ROWS } / 4], m![Queries % 4, 1 # 2]>()
        .commit_trim::<m![Queries % 4, 1 # 2]>()
        .commit_view(cells);
}

/// The first span's peaks of block `block` and head `head`, every group's, as a vector operand.
fn first_peaks(context: &mut Device, bases: &Bases, (block, head): (usize, usize)) -> Base {
    let cells = bases
        .view()
        .tile::<m![QueryBlocks], 1, m![
            QueryBlocks = 1 # QueryBlocks,
            Heads % 2,
            Groups,
            Queries % { TILE_ROWS },
            1 # 2
        ]>(block)
        .tile::<m![Heads % 2], 1, m![
            QueryBlocks = 1 # QueryBlocks,
            Heads % 2 = 1 # 2,
            Groups,
            Queries % { TILE_ROWS },
            1 # 2
        ]>(head);
    context
        .sub
        .begin(cells)
        .fetch::<m![Groups, Queries % { TILE_ROWS } / 4], m![Queries % 4, 1 # 2]>()
        .collect::<m![Groups, Queries % { TILE_ROWS } / 4], m![Queries % 4, 1 # 2]>()
        .to_vrf()
}

fn key_register(context: &mut Device, (key, head): (&Key, usize)) -> KeyRegister {
    context
        .sub
        .begin(
            key.view()
                .tile::<m![Heads % 2], 1, m![Heads % 2 = 1 # 2, Keys % { SLICE_KEYS }, Head]>(head),
        )
        .fetch::<m![
            Keys % 8,
            Keys % { SLICE_KEYS } / 16,
            Head / 16,
            Keys % 16 / 8
        ], m![Head % 16]>()
        .collect::<m![
            Keys % 8,
            Keys % { SLICE_KEYS } / 16,
            Head / 16,
            Keys % 16 / 8
        ], m![Head % 16]>()
        .to_trf()
}

fn value_register(context: &mut Device, (value, head): (&Value, usize)) -> ValueRegister {
    context
        .sub
        .begin(
            value
                .view()
                .tile::<m![Heads % 2], 1, m![Heads % 2 = 1 # 2, Channel, Keys % { SLICE_KEYS }]>(
                    head,
                ),
        )
        .fetch::<m![
            Channel % 8,
            Channel / 16,
            Keys % { SLICE_KEYS } / 16,
            Channel % 16 / 8
        ], m![Keys % 16]>()
        .collect::<m![
            Channel % 8,
            Channel / 16,
            Keys % { SLICE_KEYS } / 16,
            Channel % 16 / 8
        ], m![Keys % 16]>()
        .to_trf()
}

/// Decoded keys with every non-finite cell cleared, since the peaks read unwritten keys too; the
/// fetch widens each 8-bit float to f32 itself, so no decode table loads ahead of the pass.
fn keys(context: &mut Device, landed: &Landed, key: &mut Key) {
    context
        .main
        .begin(tokens(landed, 0))
        .fetch::<m![Heads % 2, Keys % { SLICE_KEYS }, Head / 8], m![Head % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Heads % 2, Keys % { SLICE_KEYS }, Head / 8], m![Head % 8]>()
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

/// The landed values, one 64-key run per channel.
fn transpose(context: &mut Device, landed: &Landed) -> Transposed {
    context
        .main
        .begin(tokens(landed, 1))
        .fetch::<m![Heads % 2, Head / 16, Keys % { SLICE_KEYS }], m![Head % 16]>()
        .collect::<m![Heads % 2, Head / 16, Keys % { SLICE_KEYS }], m![Head % 16 # 32]>()
        .transpose::<m![Heads % 2, Head / 16, Keys % { SLICE_KEYS } / 8, Head % 16], m![Keys % 8 # 32]>()
        .commit_trim::<m![Keys % 8]>()
        .commit()
}

/// Decoded values, one 64-key run per channel, with every non-finite cell cleared: an unwritten
/// page may hold NaN, and a zero weight times NaN is NaN.
fn values(context: &mut Device, transposed: &Transposed, value: &mut Value) {
    let channels = value
        .view_mut()
        .tile::<m![Channel], { Head::SIZE }, m![Heads % 2, Channel = 128 #{!} Channel, Keys % { SLICE_KEYS }]>(0);
    // SAFETY: a head's value channels are its first channels in order; only the axis is renamed.
    let channels = unsafe {
        channels.reshape::<Chip, Cluster, Slices, m![Heads % 2, Head #{!} Channel, Keys % { SLICE_KEYS }]>()
    };
    context
        .main
        .begin(transposed.view())
        .fetch::<m![Heads % 2, Head, Keys % { SLICE_KEYS } / 8], m![Keys % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Heads % 2, Head, Keys % { SLICE_KEYS } / 8], m![Keys % 8]>()
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
        .cast::<bf16, m![Keys % 8 # 16]>()
        .commit_trim::<m![Keys % 8]>()
        .commit_view(channels);
}

/// One block's queries of head `head`, four rows of one query head a slice.
fn query(context: &mut Device, queries: &qkv::Queries, (first, head): (usize, usize)) -> Query {
    // SAFETY: a capacity row's query heads are its `Heads × Groups` heads in head order.
    let rows = unsafe {
        queries
            .view()
            .reshape::<Chip, m![Capacity, Heads, Groups, Head]>()
    }
    .tile::<m![Capacity], { Queries::SIZE }, m![{ CapacityBlock }, Heads, Groups, Head]>(first);
    // SAFETY: the block keeps the capacity stride; only its row axis is renamed.
    unsafe { rows.reshape::<Chip, m![Queries # Capacity, Heads, Groups, Head]>() }
        .tile::<m![Heads % 2], 1, m![Queries # Capacity, Heads / 2, Heads % 2 = 1 # 2, Groups, Head]>(
            head,
        )
        .to_dm(&mut context.tdma)
}

/// Scaled scores of every query of the block against every key of the slice, or with `base` their
/// weights above it with every key still in, seen or not: each query broadcasts over its half's
/// 128 key slices and meets 64 keys there.
fn scores<E: M>(
    context: &mut Device,
    (query, key): (&Query, &KeyRegister),
    base: &Option<Base>,
    scores: DmTensorViewMut<'_, bf16, Chip, Cluster, Slices, E>,
) {
    type Sums = m![
        { RowPair },
        Keys % { SLICE_KEYS } / 16,
        { RowGroup },
        Groups,
        Keys % 16 / 8
    ];
    let products =
        PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default(
        )
        .forward((
            context,
            Dm::<Space, _> {
                tiles: Tiles::over(Queries::SIZE),
                value: (query.view(), key),
            },
        ))
        .value;
    match base {
        None => PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default()
            .forward(Dm::<Space, _> {
                tiles: Tiles::over(Queries::SIZE),
                value: products,
            })
            .value
            .vector_widen_concat::<m![{ Sums }, Queries % 2], m![Keys % 8]>()
            .vector_final()
            .commit_trim::<m![Keys % 8]>()
            .commit_cast::<bf16>()
            .commit_view(scores),
        // e^(scale * product - base), as the base less the scaled product, negated.
        Some(base) => PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default()
            .forward(Dm::<Space, _> {
                tiles: Tiles::over(Queries::SIZE),
                value: (
                    PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default()
                        .forward(Dm::<Space, _> {
                            tiles: Tiles::over(Queries::SIZE),
                            value: (products, base),
                        })
                        .value
                        .vector_widen_concat::<m![{ Sums }, Queries % 2], m![Keys % 8]>(),
                    CEILING,
                ),
            })
            .value
            .vector_final()
            .commit_trim::<m![Keys % 8]>()
            .commit_cast::<bf16>()
            .commit_view(scores),
    }
}

/// Scaled scores of every query of the block against the first 16 keys of the slice.
fn sample_scores(context: &mut Device, (query, sample): (&Query, &SampleRegister)) -> Sample {
    type Stream = m![{ RowPair }, { RowGroup }, Groups, Head / 16];
    type Steps = m![{ Stream }, Keys % 16 / 8];
    type Sums = m![{ RowPair }, { RowGroup }, Groups, Keys % 16 / 8];
    let products = context
        .main
        .begin(query.view())
        .fetch::<m![{ RowPair }, Head / 16], m![Queries % 2, Head % 16]>()
        .switch::<Slices, Stream>(SwitchConfig::Broadcast01 {
            slice1: Keys::SIZE / SLICE_KEYS,
            slice0: 1,
            time0: 8,
        })
        .collect::<m![{ Stream }, Queries % 2], m![Head % 16]>()
        .contract_outer::<Steps, m![Queries % 2, Head % 16], _, _, _>(sample)
        .contract_packet::<m![Queries % 2]>()
        .contract_time::<Sums>()
        .contract_lane::<m![{ Sums }, Queries % 2], m![Keys % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ Sums }, Queries % 2, Keys % 8 / 4], m![Keys % 4]>();
    PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default()
        .forward(Dm::<Space, _> {
            tiles: Tiles::over(Queries::SIZE),
            value: products,
        })
        .value
        .vector_widen_concat::<m![{ Sums }, Queries % 2], m![Keys % 8]>()
        .vector_final()
        .commit_trim::<m![Keys % 8]>()
        .commit_cast::<bf16>()
        .commit()
}

/// Head `head`'s weights to write.
fn head_mut(weights: &mut Weights, head: usize) -> HeadWeightsMut<'_> {
    weights
        .view_mut()
        .tile::<m![Heads % 2], 1, m![Heads % 2 = 1 #{!} 2, Groups, Queries % { TILE_ROWS }, Keys % { SLICE_KEYS }]>(head)
}

/// Head `head`'s weights of group `group` to write.
fn head_group(
    weights: &mut Weights,
    (head, group): (usize, usize),
) -> DmTensorViewMut<
    '_,
    bf16,
    Chip,
    Cluster,
    Slices,
    m![Heads % 2 = 1 #{!} 2, Groups = 1 #{!} 4, Queries % { TILE_ROWS }, Keys % { SLICE_KEYS }],
> {
    head_mut(weights, head)
        .tile::<m![Groups], 1, m![Heads % 2 = 1 #{!} 2, Groups = 1 #{!} 4, Queries % { TILE_ROWS }, Keys % { SLICE_KEYS }]>(group)
}

/// Group `group`'s scores.
fn group(scores: &Scores, group: usize) -> GroupScores<'_> {
    scores
        .view()
        .tile::<m![Groups], 1, m![Groups = 1 # 4, Queries % { TILE_ROWS }, Keys % { SLICE_KEYS }]>(
            group,
        )
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

/// Group `at`'s scores as the softmax passes read them: keys a row does not see at the lowest bf16,
/// where `tags` marks any.
fn masked<'l>(
    context: &'l mut Device,
    (scores, tags): (&'l Scores, &Option<Tags>),
    at: usize,
) -> Masked<'l> {
    let collected = context
        .main
        .begin(group(scores, at))
        .fetch::<m![Queries % { TILE_ROWS }, Keys % { SLICE_KEYS } / 8], m![Keys % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Queries % { TILE_ROWS }, Keys % { SLICE_KEYS } / 8], m![Keys % 8]>()
        .vector_init();
    match tags {
        None => collected
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Queries % { TILE_ROWS }, Keys % { SLICE_KEYS } / 4], m![Keys % 4]>(),
        Some(tags) => VectorTensor::<_, _, f32, _, _, _, _, _, _, _, _, _>::vector_logic_with_mode(
            collected.vector_tagged(tags),
            LogicBinaryOpF32::BitAnd,
            BinaryArgMode::Mode11,
            Branched::imm(ABSENT, f32::from_bits(0xff7f_0000)),
        )
        .vector_narrow_split::<m![Queries % { TILE_ROWS }, Keys % { SLICE_KEYS } / 4], m![Keys % 4]>(),
    }
}

/// `raw`'s weights into `weights`, zero where a row does not see the key.
fn clear(
    context: &mut Device,
    (raw, tags): (&Scores, &Tags),
    (weights, head): (&mut Weights, usize),
) {
    for at in 0..4 {
        let tagged = context
            .main
            .begin(group(raw, at))
            .fetch::<m![Queries % { TILE_ROWS }, Keys % { SLICE_KEYS } / 8], m![Keys % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Queries % { TILE_ROWS }, Keys % { SLICE_KEYS } / 8], m![Keys % 8]>()
            .vector_init()
            .vector_tagged(tags);
        VectorTensor::<_, _, f32, _, _, _, _, _, _, _, _, _>::vector_logic_with_mode(
            tagged,
            LogicBinaryOpF32::BitAnd,
            BinaryArgMode::Mode11,
            Branched::imm(ABSENT, 0.0f32),
        )
        .vector_final()
        .cast::<bf16, m![Keys % 8 # 16]>()
        .commit_trim::<m![Keys % 8]>()
        .commit_view(head_group(weights, (head, at)));
    }
}

/// Each row's peak over the first 16 keys of every slice of its half, seen or not: softmax is
/// the same above any shift, the exponents' ceiling bounds a key far above this one, and every
/// row sees the span's first key.
fn peaks<E: M>(
    context: &mut Device,
    sampled: DmTensorView<'_, bf16, Chip, Cluster, Slices, E>,
) -> Peak {
    let local: Peak = context
        .main
        .begin(sampled)
        .fetch::<m![Groups, Queries % { TILE_ROWS }, Keys % 16 / 8], m![Keys % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Groups, Queries % { TILE_ROWS }, Keys % 16 / 8], m![Keys % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Groups, Queries % { TILE_ROWS }, Keys % 16 / 4], m![Keys % 4]>()
        .vector_intra_slice_reduce::<Keys, m![Groups, Queries % { TILE_ROWS }], m![1 # 4]>(
            IntraSliceReduceOpF32::Max,
        )
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 2]>()
        .commit();
    context
        .main
        .begin(local.view())
        .fetch::<m![Groups, Queries % { TILE_ROWS } / 4], m![Queries % 4, 1 # 2]>()
        .collect::<m![Groups, Queries % { TILE_ROWS } / 4], m![Queries % 4, 1 # 2]>()
        .vector_init()
        .vector_inter_slice_reduce_span(InterSliceReduceOpF32::Max, KEY_SLICE_LEVELS)
        .vector_final()
        .commit_trim::<m![Queries % 4, 1 # 2]>()
        .commit()
}

/// Each row's peak over the keys it sees in every slice of its half: the keys past a row hold
/// other sequences' pages or unwritten cells, so they must not move its shift.
fn visible_peaks(context: &mut Device, (scores, tags): (&Scores, &Option<Tags>)) -> Peak {
    let mut local = Peak::new();
    #[unroll]
    for at in 0..4 {
        let cells = local
            .view_mut()
            .tile::<m![Groups], 1, m![Groups = 1 #{!} 4, Queries % { TILE_ROWS }, 1 # 2]>(at);
        masked(context, (scores, tags), at)
            .vector_intra_slice_reduce::<Keys, m![Queries % { TILE_ROWS }], m![1 # 4]>(
                IntraSliceReduceOpF32::Max,
            )
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .commit_trim::<m![1 # 2]>()
            .commit_view(cells);
    }
    context
        .main
        .begin(local.view())
        .fetch::<m![Groups, Queries % { TILE_ROWS } / 4], m![Queries % 4, 1 # 2]>()
        .collect::<m![Groups, Queries % { TILE_ROWS } / 4], m![Queries % 4, 1 # 2]>()
        .vector_init()
        .vector_inter_slice_reduce_span(InterSliceReduceOpF32::Max, KEY_SLICE_LEVELS)
        .vector_final()
        .commit_trim::<m![Queries % 4, 1 # 2]>()
        .commit()
}

/// Unnormalized softmax weights into `weights`: each the exponent of a score above `peak`.
fn exponents(
    context: &mut Device,
    (scores, tags, peak): (&Scores, &Option<Tags>, &Peak),
    (weights, head): (&mut Weights, usize),
) {
    // At eight bytes a row (a commit's least), every group's peaks fit the vector registers beside
    // the tags, so they all land before the first exponent pass.
    let tops = [
        top(context, peak, 0),
        top(context, peak, 1),
        top(context, peak, 2),
        top(context, peak, 3),
    ];
    #[unroll]
    for at in 0..4 {
        PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default()
            .forward(Dm::<Space, _> {
                tiles: Tiles::over(Queries::SIZE),
                value: (
                    masked(context, (scores, tags), at)
                        .vector_fp_binary(FpBinaryOp::SubF, &tops[at])
                        .vector_fp_unary(FpUnaryOp::Exp)
                        .vector_widen_concat::<m![Queries % { TILE_ROWS }, Keys % { SLICE_KEYS } / 8], m![Keys % 8]>(),
                    CEILING,
                ),
            })
            .value
            .vector_final()
            .cast::<bf16, m![Keys % 8 # 16]>()
            .commit_trim::<m![Keys % 8]>()
            .commit_view(head_group(weights, (head, at)));
    }
}

/// Group `at`'s all-reduced peaks, one a row, as the exponent pass's operand.
fn top(context: &mut Device, peak: &Peak, at: usize) -> Top {
    context
        .sub
        .begin(
            peak.view()
                .tile::<m![Groups], 1, m![Groups = 1 # 4, Queries % { TILE_ROWS }, 1 # 2]>(at),
        )
        .fetch::<m![Queries % { TILE_ROWS } / 4], m![Queries % 4, 1 # 2]>()
        .collect::<m![Queries % { TILE_ROWS } / 4], m![Queries % 4, 1 # 2]>()
        .to_vrf()
}

/// Head `head`'s weighted values into `product`, summed over the half's key slices onto each
/// query's own slice.
fn weighted(
    context: &mut Device,
    (weights, head): (&Weights, usize),
    value: ValueRegister,
    product: &mut Output,
) {
    PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default()
        .forward((context, Dm::<Space, _> {
            tiles: Tiles::over(Queries::SIZE),
            value: (
            weights
                .view()
                .tile::<m![Heads % 2], 1, m![Heads % 2 = 1 # 2, Groups, Queries % { TILE_ROWS }, Keys % { SLICE_KEYS }]>(
                    head,
                ),
            value,
            (product
                .view_mut()
                .tile::<m![Heads % 2], 1, m![Heads % 2 = 1 #{!} 2, { Row }, Channel]>(head),),
            ),
        }));
}
