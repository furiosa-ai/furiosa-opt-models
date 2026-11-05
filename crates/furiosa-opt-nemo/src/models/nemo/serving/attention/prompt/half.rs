// The landing of a geometry whose slice holds two 16-token pages in the first half of the 32 KiB
// every geometry's landing buffer holds: the half geometry shares the wide geometry's buffer.
/// A slice's keys of its cluster's two heads, each head's key and value adjacent as the pool
/// stores them.
pub type Raw = m![Keys % { SLICE_KEYS } / 16 # 4, Page, Heads % 2, Kv, Head];
pub type Landing = m![{ Copies }, Keys / { SLICE_KEYS }];
/// One gather entry lands in every copy.
pub type Landed = DmTensor<f8e4m3, Chip, Cluster, Landing, Raw>;

/// Gathers into `landed` the keys and values of the span whose pages `pages` lists.
pub fn land(
    context: &mut Device,
    (pool, entry): (&Pool, usize),
    pages: &Pages,
    landed: &mut Landed,
) {
    // SAFETY: the host plan bounds every page by the layer's pages, and each offset starts one.
    unsafe {
        pool.gather::<m![1], m![Page, Heads, Kv, Head]>()
            .entry(entry)
            .by_byte_offsets(pages)
            .unchecked()
    }
    .to_dm_view(&mut context.tdma, landed.view_mut());
}

/// Gathers into `landed`, a geometry's 32-KiB landing buffer, the keys and values of the wide span
/// whose pages `pages` lists, its first 1,024 keys in this geometry's layout and the rest in the
/// two pages a slice that layout leaves out.
pub fn land_wide<P: M, S: M, E: M>(
    context: &mut Device,
    (pool, entry): (&Pool, usize),
    pages: &HbmTensor<i32, Chip, P>,
    landed: &mut DmTensor<f8e4m3, Chip, Cluster, S, E>,
) {
    type WideKeys = super::wide::Keys;
    // SAFETY: every geometry's landing buffer holds 32 KiB in each of a cluster's 256 slices.
    let view = unsafe {
        landed.view_mut().reshape::<Chip, Cluster, m![{ Copies }, WideKeys % { Keys::SIZE } / { SLICE_KEYS }], m![
            WideKeys / { Keys::SIZE },
            WideKeys % { SLICE_KEYS } / 16,
            Page,
            Heads % 2,
            Kv,
            Head
        ]>()
    };
    // SAFETY: the host plan bounds every page by the layer's pages, and each offset starts one.
    unsafe {
        pool.gather::<m![1], m![Page, Heads, Kv, Head]>()
            .entry(entry)
            .by_byte_offsets(pages)
            .unchecked()
    }
    .to_dm_view(&mut context.tdma, view);
}

/// Both heads' keys or values of every slice, in token order.
fn tokens(
    landed: &Landed,
    kv: usize,
) -> DmTensorView<'_, f8e4m3, Chip, Cluster, Slices, m![Keys % { SLICE_KEYS }, Heads % 2, Kv = 1 # 2, Head]> {
    let live = landed
        .view()
        .unpad::<m![Keys % { SLICE_KEYS } / 16, Page, Heads % 2, Kv, Head]>();
    // SAFETY: every copy of a slice's keys serves the query group that holds it, and a slice's
    // pages of sixteen tokens are its keys in order.
    unsafe { live.reshape::<Chip, Cluster, Slices, m![Keys % { SLICE_KEYS }, Heads % 2, Kv, Head]>() }
        .tile::<m![Kv], 1, m![Keys % { SLICE_KEYS }, Heads % 2, Kv = 1 # 2, Head]>(kv)
}

/// The blocks of one wide span: a wide segment's blocks reach its spans in steps of these.
const WIDE_SPAN_BLOCKS: usize = super::wide::Keys::SIZE / Queries::SIZE;

impl Sequence {
    /// A wide segment's sequence, its last span landed in this geometry's layout.
    pub fn resume(sequence: super::wide::Sequence) -> Self {
        let (landed, bases, sums) = sequence.into_parts();
        // SAFETY: both geometries' landing buffers hold 32 KiB in each slice, and both lay a
        // block's 128-row tiles, key slices and query slices out alike.
        unsafe {
            Self {
                landed: landed.reshape(),
                keys: Key::new(),
                values: Value::new(),
                bases: bases.reshape(),
                sums: sums.reshape(),
            }
        }
    }

    /// The last of a wide segment's `spans` spans into `output` when `last` is 1, the span at
    /// most its first 1,024 keys: the blocks whose rows reach it add its product to the earlier
    /// spans' and divide. The segment's first chunk row is `first`, its first position `ahead`
    /// wide blocks rounded up, and the chunk's first capacity row `start`.
    pub fn attend_last(
        &mut self,
        context: &mut Device,
        (plan, (start, first), queries): (&super::wide::Tables, (usize, usize), &qkv::Queries),
        (spans, ahead, blocks, last): (usize, usize, usize, usize),
        output: &mut Attended,
    ) {
        for _ in 0..last {
            let span = spans - 1;
            let reach = span * WIDE_SPAN_BLOCKS;
            let from = reach - reach.min(ahead);
            // `min` gives the loop end a finite envelope; a plan's blocks never exceed it.
            for at in 0..(blocks - from).min(QueryBlocks::SIZE) {
                let block = from + at;
                for _ in 0..1 - at.min(1) {
                    self.values
                        .view_mut()
                        .tile::<m![Channel], 16, m![Heads % 2, Channel = 16 #{!} Channel, Keys % { SLICE_KEYS }]>(Head::SIZE)
                        .memset(const { bf16::from_f32(1.0) }, &mut context.sub);
                    let transposed = transpose(context, &self.landed);
                    keys(context, &self.landed, &mut self.keys);
                    values(context, &transposed, &mut self.values);
                }
                let row = first + Queries::SIZE * block;
                let block_queries = [
                    query(context, queries, (start + row, 0)),
                    query(context, queries, (start + row, 1)),
                ];
                let limit = last_limits(context, plan, (span, row));
                let positions = last_positions(context, plan);
                self.block::<true, false>(
                    context,
                    (&block_queries, Some((&positions, &limit))),
                    (0, block, row),
                    output,
                );
            }
        }
    }
}

/// The limits in wide span `span` of the block from chunk row `row`, landed in every key slice of
/// both clusters.
fn last_limits(context: &mut Device, plan: &super::wide::Tables, (span, row): (usize, usize)) -> Limit {
    type WideSpans = super::wide::Spans;
    let rows = plan
        .limits
        .view()
        .tile::<m![WideSpans], 1, m![WideSpans = 1 # WideSpans, Prompt, Lane]>(span)
        .tile::<m![Prompt], { Queries::SIZE }, m![WideSpans = 1 # WideSpans, { Block }, Lane]>(row);
    // SAFETY: the tile keeps the chunk's row stride; only its row axis is renamed.
    let rows = unsafe { rows.reshape::<Chip, m![WideSpans = 1 # WideSpans, Queries # Prompt, Lane]>() };
    let landed: DmTensor<f32, Chip, Cluster, Slices, m![Queries % { TILE_ROWS }, Lane]> =
        rows.to_dm(&mut context.tdma);
    // SAFETY: `Lane` and `1 # 8` lay out the same eight cells per row.
    unsafe { landed.reshape() }
}

/// The positions of a wide span's first 1,024 keys, each its index within the span.
fn last_positions(context: &mut Device, plan: &super::wide::Tables) -> Positions {
    type WideKeys = super::wide::Keys;
    let first = plan
        .positions
        .view()
        .tile::<m![WideKeys], { Keys::SIZE }, m![WideKeys = { Keys::SIZE } # WideKeys]>(0);
    // SAFETY: the tile keeps the table's order; only its key axis is renamed.
    unsafe { first.reshape::<Chip, m![Keys # WideKeys]>() }.to_dm(&mut context.tdma)
}
