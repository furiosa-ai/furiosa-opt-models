use super::*;
use crate::Module;
use crate::compose::{Tile, Tiles};
use crate::ops::attention::gqa::{PagedAttention, completion, reduction};
use crate::ops::layout::Dm;

pub(super) type Space = Tile<Rows, Rows>;

impl completion::Mapping for Space {
    const SPAN: usize = 3;
}

impl reduction::Mapping for Space {
    type Entries = m![Entries % 4];
    type Entry = Entries;
}

/// Each half of a PE's slices holds four consecutive rows, the second half 32 rows on: up to 32
/// rows fill one half of every PE, so its gather walks one data-memory network's records.
type C = m![Rows % 32 / 16];
type S = m![Rows % 16 / 4, Rows / 32, Rows % 4, Entries / 4];
type Gathered = DmTensor<f32, Chip, C, S, m![Entries % 4, Heads, Stat, Groups, 1 # 8]>;
type Records = DmTensor<f32, Chip, C, S, m![Entries % 4, Heads, Stat, Groups, Lane]>;
type Visibility = DmTensor<f32, Chip, C, S, m![Entries % 4]>;
type Stats<'l> =
    DmTensorView<'l, f32, Chip, C, S, m![Entries % 4, Heads, Stat = 1 # 2, Groups, 1 # 8]>;
type Values = DmTensor<bf16, Chip, C, S, m![Heads, Groups, Entries % 4, Head]>;
type Tags = VrfTensor<i4, Chip, C, S, m![Heads, Groups, Entries % 4, 1 # 8]>;
type Terms = DmTensor<f32, Chip, C, S, m![Heads, Groups, Entries % 4, 1 # 8]>;
type Scales = VrfTensor<f32, Chip, C, S, m![Heads, Groups, Entries % 4, 1 # 8]>;
type Sum = DmTensor<f32, Chip, C, S, m![Heads, Groups, 1 # 8]>;
/// Each row's peak and mass, laid out as a stored statistics record.
type RowStats = DmTensor<f32, Chip, C, S, m![Heads, Stat, Groups, 1 # 8]>;
type RowStat<'l> = DmTensorView<'l, f32, Chip, C, S, m![Heads, Stat = 1 # 2, Groups, 1 # 8]>;
type RowStatMut<'l> =
    DmTensorViewMut<'l, f32, Chip, C, S, m![Heads, Stat = 1 #{!} 2, Groups, 1 # 8]>;
type Operand = VrfTensor<f32, Chip, C, S, m![Heads, Groups, 1 # 8]>;
pub(super) type Weighted = DmTensor<f32, Chip, C, S, m![Heads, Groups, Head]>;
const CLEARED: TagGuard =
    TagGuard::matches([BitReq::Zero, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]);

/// Record `index` of every gathered entry: 0 the peak, 1 the mass.
fn stat(stats: &Gathered, index: usize) -> Stats<'_> {
    stats
        .view()
        .tile::<m![Stat], 1, m![Entries % 4, Heads, Stat = 1 # 2, Groups, 1 # 8]>(index)
}

/// Record `index` of each row's statistics: 0 the peak, 1 the mass.
fn row_stat(row_stats: &RowStats, index: usize) -> RowStat<'_> {
    row_stats
        .view()
        .tile::<m![Stat], 1, m![Heads, Stat = 1 # 2, Groups, 1 # 8]>(index)
}

fn row_stat_mut(row_stats: &mut RowStats, index: usize) -> RowStatMut<'_> {
    row_stats
        .view_mut()
        .tile::<m![Stat], 1, m![Heads, Stat = 1 #{!} 2, Groups, 1 # 8]>(index)
}

/// A merge pass's entries of its first `rows` rows: their visibility tags, statistics and outputs.
/// The later rows keep stale values; all their entries are invisible, so their merge yields zero.
fn gather(
    context: &mut Device,
    partials: &Partials,
    (statistics, outputs, visible): (&Merge, &Merge, &Visible),
    rows: usize,
) -> (Tags, Gathered, Values) {
    let entries = (rows * Entries::SIZE) as i32;
    // A gather of no rows still builds its descriptor lists, so a pass that reads none issues none.
    let mut visibility = Visibility::new();
    let mut records = Records::new();
    let mut values = Values::new();
    for _ in 0..rows.min(1) {
        visible
            .view()
            .to_dm_view(&mut context.tdma, visibility.view_mut());
        // SAFETY: the host addresses whole stored records.
        unsafe {
            partials
                .stats
                .gather::<m![Burst, Jobs], m![Heads, Stat, Groups, Lane]>()
                .by_byte_offsets(statistics)
                .sparse_prefix(entries)
                .unchecked()
        }
        .to_dm_view(&mut context.tdma, records.view_mut());
        // Issued ahead of the statistics passes, which do not read it, so it lands while they run.
        // SAFETY: the host addresses whole stored records.
        unsafe {
            partials
                .output
                .gather::<m![Burst, Jobs], m![Heads, Groups, Head]>()
                .by_byte_offsets(outputs)
                .sparse_prefix(entries)
                .unchecked()
        }
        .to_dm_view(&mut context.tdma, values.view_mut());
    }
    // SAFETY: `Lane` re-reads as the `1 # 8` vector lanes the statistics were committed with.
    let stats: Gathered = unsafe { records.reshape() };
    let tags: Tags = context
        .main
        .begin(visibility.view())
        .fetch::<m![Heads, Groups, Entries % 4], m![1 # 8]>()
        .collect::<m![Heads, Groups, Entries % 4], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Comparison([
            Cmp::Greater(0.0f32),
            Cmp::False,
            Cmp::False,
            Cmp::False,
        ]))
        .vector_log(&mut context.sub);
    (tags, stats, values)
}

/// Each row's peak over its visible entries, into record 0 of `row_stats`.
fn peak(context: &mut Device, stats: &Gathered, tags: &Tags, row_stats: &mut RowStats) {
    let attention =
        PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default(
        );
    let local: Sum = attention
        .forward((
            &mut *context,
            Dm::<Space, _> {
                tiles: Tiles::over(Rows::SIZE),
                value: (stat(stats, 0), tags),
            },
        ))
        .value;
    attention.forward((
        &mut *context,
        Dm::<Space, _> {
            tiles: Tiles::over(Rows::SIZE),
            value: ((local.view(), (), ()), row_stat_mut(row_stats, 0)),
        },
    ));
}

/// Each entry's weight against its row's peak; an invisible entry weighs zero.
fn scales(context: &mut Device, stats: &Gathered, tags: &Tags, row_stats: &RowStats) -> Scales {
    let attention =
        PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default(
        );
    let peak: Operand = attention
        .forward((
            &mut *context,
            Dm::<Space, _> {
                tiles: Tiles::over(Rows::SIZE),
                value: ((), row_stat(row_stats, 0)),
            },
        ))
        .value;
    attention
        .forward((
            &mut *context,
            Dm::<Space, _> {
                tiles: Tiles::over(Rows::SIZE),
                value: (stat(stats, 0), tags, &peak),
            },
        ))
        .value
}

/// Each row's weighted mass, into record 1 of `row_stats`; an invisible entry's mass is cleared,
/// since its stale record times a zero weight is NaN when the record is not finite.
fn mass(
    context: &mut Device,
    stats: &Gathered,
    tags: &Tags,
    scales: &Scales,
    row_stats: &mut RowStats,
) {
    let terms: Terms = context
        .main
        .begin(stat(stats, 1))
        .fetch::<m![Heads, Groups, Entries % 4], m![1 # 8]>()
        .collect::<m![Heads, Groups, Entries % 4], m![1 # 8]>()
        .vector_init()
        .vector_tagged(tags)
        .vector_logic(LogicBinaryOpF32::BitAnd, Branched::imm(CLEARED, 0.0f32))
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scales)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let attention =
        PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default(
        );
    let local: Sum = attention
        .forward((
            &mut *context,
            Dm::<Space, _> {
                tiles: Tiles::over(Rows::SIZE),
                value: (terms.view(), (), ()),
            },
        ))
        .value;
    attention.forward((
        &mut *context,
        Dm::<Space, _> {
            tiles: Tiles::over(Rows::SIZE),
            value: ((local.view(), ()), row_stat_mut(row_stats, 1)),
        },
    ));
}

/// Each slice's weighted sum of its entries' outputs, still to be summed over the row's slices.
/// Non-finite outputs are cleared first: only an invisible entry's stale record holds one, and
/// the visibility tags do not cover the output's head lanes.
fn weighted(context: &mut Device, values: &Values, scales: &Scales) -> Weighted {
    type Scaled = DmTensor<f32, Chip, C, S, m![Heads, Groups, Entries % 4, Head]>;
    let scaled: Scaled = context
        .main
        .begin(values.view())
        .fetch::<m![Heads, Groups, Entries % 4, Head / 8], m![Head % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Heads, Groups, Entries % 4, Head / 8], m![Head % 8]>()
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
        .vector_narrow_split::<m![Heads, Groups, Entries % 4, Head / 4], m![Head % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scales)
        .vector_widen_concat::<m![Heads, Groups, Entries % 4, Head / 8], m![Head % 8]>()
        .vector_final()
        .commit_trim::<m![Head % 8]>()
        .commit();
    PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default()
        .forward((
            &mut *context,
            Dm::<Space, _> {
                tiles: Tiles::over(Rows::SIZE),
                value: ((scaled.view(), ()),),
            },
        ))
        .value
}

/// Stores each row's statistics as its record of fold `fold`'s wave of the first entry.
fn record(context: &mut Device, row_stats: &RowStats, entry: &mut Statistics, fold: usize) {
    // SAFETY: `1 # 8` and `Lane` lay out the same eight cells per head and group, and a fold
    // wave's records are its rows in order.
    unsafe {
        let records = entry
            .view_mut()
            .reshape::<Chip, m![Burst, Rows, Heads, Stat, Groups, Lane]>();
        row_stats
            .view()
            .reshape::<Chip, C, S, m![Heads, Stat, Groups, Lane]>()
            .slice_tile::<m![Entries / 4], 1, m![Rows % 16 / 4, Rows / 32, Rows % 4, Entries / 4 = 1 # 8]>(0)
            .to_hbm_view(
                &mut context.tdma,
                records
                    .tile::<m![Burst], 1, m![Burst = 1 #{!} 8, Rows, Heads, Stat, Groups, Lane]>(
                        fold,
                    ),
            );
    }
}

/// Fold pass `fold`: each row's later jobs into one record of the fold entry, at the row's index,
/// for the final pass to read as one more partial.
pub(super) fn fold(
    context: &mut Device,
    (passes, entry): (&Passes, usize),
    partials: &mut Partials,
    fold: usize,
) {
    let (tags, stats, values) = gather(
        context,
        partials,
        (
            &passes.statistics[entry],
            &passes.outputs[entry],
            &passes.visible[entry],
        ),
        Rows::SIZE,
    );
    let mut row_stats = RowStats::new();
    peak(context, &stats, &tags, &mut row_stats);
    let scales = scales(context, &stats, &tags, &row_stats);
    mass(context, &stats, &tags, &scales, &mut row_stats);
    let local = weighted(context, &values, &scales);
    let output: DmTensor<bf16, Chip, C, S, m![Heads, Groups, Head]> =
        PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default(
        )
        .forward((
            &mut *context,
            Dm::<Space, _> {
                tiles: Tiles::over(Rows::SIZE),
                value: ((local.view(),), ()),
            },
        ))
        .value;
    record(context, &row_stats, &mut partials.stats[0], fold);
    // SAFETY: a fold wave's records are its rows in order.
    let records = unsafe {
        partials.output[0]
            .view_mut()
            .reshape::<Chip, m![Burst, Rows, Heads, Groups, Head]>()
    };
    output
        .view()
        .slice_tile::<m![Entries / 4], 1, m![Rows % 16 / 4, Rows / 32, Rows % 4, Entries / 4 = 1 # 8]>(0)
        .to_hbm_view(
            &mut context.tdma,
            records.tile::<m![Burst], 1, m![Burst = 1 #{!} 8, Rows, Heads, Groups, Head]>(fold),
        );
}

/// The final merge's entries, gathered and tagged; [`finish`] merges them.
pub struct Gathers(Tags, Gathered, Values);

pub(super) fn rows(
    context: &mut Device,
    (passes, entry): (&Passes, usize),
    partials: &Partials,
    rows: usize,
) -> Gathers {
    let (tags, stats, values) = gather(
        context,
        partials,
        (
            &passes.statistics[entry],
            &passes.outputs[entry],
            &passes.visible[entry],
        ),
        rows,
    );
    Gathers(tags, stats, values)
}

/// The batch's attended rows from its gathered final-merge entries.
pub(super) fn finish(
    context: &mut Device,
    Gathers(tags, stats, values): &Gathers,
    output: &mut Restored,
) {
    let mut row_stats = RowStats::new();
    peak(context, stats, tags, &mut row_stats);
    let scales = scales(context, stats, tags, &row_stats);
    mass(context, stats, tags, &scales, &mut row_stats);
    let attention =
        PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default(
        );
    let total: Operand = attention
        .forward((
            &mut *context,
            Dm::<Space, _> {
                tiles: Tiles::over(Rows::SIZE),
                value: (row_stat(&row_stats, 1), ()),
            },
        ))
        .value;
    let local = weighted(context, values, &scales);
    let result: Weighted = attention
        .forward((
            &mut *context,
            Dm::<Space, _> {
                tiles: Tiles::over(Rows::SIZE),
                value: (local.view(), &total),
            },
        ))
        .value;
    store(context, &result, output);
}

/// Each row's result from its first slice into `output`.
pub(super) fn store(context: &mut Device, result: &Weighted, output: &mut Restored) {
    result
        .view()
        .slice_tile::<m![Entries / 4], 1, m![Rows % 16 / 4, Rows / 32, Rows % 4, Entries / 4 = 1 # 8]>(0)
        .to_hbm_view(&mut context.tdma, output.view_mut());
}
