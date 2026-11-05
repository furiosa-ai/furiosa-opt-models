//! The final merge of a batch whose rows read at most `Runs` entries each: one entry a slice, a
//! row's eight on the eight slices `merge` gives a row's 32, so its gathers issue a quarter of the
//! descriptors. Its arithmetic is the merge's, each slice holding one entry instead of four.
use super::*;
use crate::Module;
use crate::compose::Tiles;
use crate::ops::attention::gqa::PagedAttention;
use crate::ops::layout::Dm;

/// Each half of a PE's slices holds four consecutive rows, the second half 32 rows on, as `merge`.
type C = m![Rows % 32 / 16];
type S = m![Rows % 16 / 4, Rows / 32, Rows % 4, Runs];
type Gathered = DmTensor<f32, Chip, C, S, m![Heads, Stat, Groups, 1 # 8]>;
type Visibility = DmTensor<f32, Chip, C, S, m![Lane]>;
type Records = DmTensor<f32, Chip, C, S, m![Heads, Stat, Groups, Lane]>;
type Stats<'l> = DmTensorView<'l, f32, Chip, C, S, m![Heads, Stat = 1 # 2, Groups, 1 # 8]>;
type Values = DmTensor<bf16, Chip, C, S, m![Heads, Groups, Head]>;
type Tags = VrfTensor<i4, Chip, C, S, m![Heads, Groups, 1 # 8]>;
type Terms = DmTensor<f32, Chip, C, S, m![Heads, Groups, 1 # 8]>;
type Scales = VrfTensor<f32, Chip, C, S, m![Heads, Groups, 1 # 8]>;
/// Each row's peak and mass, laid out as a stored statistics record.
type RowStats = DmTensor<f32, Chip, C, S, m![Heads, Stat, Groups, 1 # 8]>;
type RowStat<'l> = DmTensorView<'l, f32, Chip, C, S, m![Heads, Stat = 1 # 2, Groups, 1 # 8]>;
type RowStatMut<'l> =
    DmTensorViewMut<'l, f32, Chip, C, S, m![Heads, Stat = 1 #{!} 2, Groups, 1 # 8]>;
type Operand = VrfTensor<f32, Chip, C, S, m![Heads, Groups, 1 # 8]>;
type Weighted = DmTensor<f32, Chip, C, S, m![Heads, Groups, Head]>;
const ADMITTED: TagGuard =
    TagGuard::matches([BitReq::One, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]);

/// Record `index` of every gathered entry: 0 the peak, 1 the mass.
fn stat(stats: &Gathered, index: usize) -> Stats<'_> {
    stats
        .view()
        .tile::<m![Stat], 1, m![Heads, Stat = 1 # 2, Groups, 1 # 8]>(index)
}

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

/// A run pass's entries of its first `reach` rows: their visibility tags, statistics and outputs.
/// The later rows keep stale values; all their entries are invisible, so their merge yields zero.
fn gather(
    context: &mut Device,
    partials: &Partials,
    (statistics, outputs, visible): (&RunMerge, &RunMerge, &RunVisible),
    reach: usize,
) -> Gathers {
    // A row prefix skips whole rows' descriptors, so a batch's gathers follow its rows.
    let entries = (reach * Runs::SIZE) as i32;
    // A gather of no rows still builds its descriptor lists, so a pass that reads none issues none.
    let mut tags = Visibility::new();
    let mut records = Records::new();
    let mut values = Values::new();
    for _ in 0..reach.min(1) {
        visible
            .view()
            .to_dm_view(&mut context.tdma, tags.view_mut());
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
        // Issued ahead of the statistics passes, which do not read it, so it lands while they
        // run. SAFETY: the host addresses whole stored records.
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
    Gathers(tags, unsafe { records.reshape() }, values)
}

/// Each entry's visibility tag from its eight lanes: positive where the entry counts.
fn visibility(context: &mut Device, visible: &Visibility) -> Tags {
    // SAFETY: the host writes an entry's tag in all eight lanes.
    let visible = unsafe { visible.view().reshape::<Chip, C, S, m![1 # 8]>() };
    context
        .main
        .begin(visible)
        .fetch::<m![Heads, Groups], m![1 # 8]>()
        .collect::<m![Heads, Groups], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Comparison([
            Cmp::Greater(0.0f32),
            Cmp::False,
            Cmp::False,
            Cmp::False,
        ]))
        .vector_log(&mut context.sub)
}

/// Each row's peak over its visible entries, into record 0 of `row_stats`.
fn peak(context: &mut Device, stats: &Gathered, tags: &Tags, row_stats: &mut RowStats) {
    let local: Terms = context
        .main
        .begin(stat(stats, 0))
        .fetch::<m![Heads, Groups], m![1 # 8]>()
        .collect::<m![Heads, Groups], m![1 # 8]>()
        .vector_init()
        .vector_tagged(tags)
        .vector_logic(LogicBinaryOpF32::BitAnd, Branched::imm(CLEARED, 0.0f32))
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, Branched::imm(CLEARED, f32::NEG_INFINITY))
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    context
        .main
        .begin(local.view())
        .fetch::<m![Heads, Groups], m![1 # 8]>()
        .collect::<m![Heads, Groups], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce_span(InterSliceReduceOpF32::Max, 3)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Max, -1e30f32)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit_view(row_stat_mut(row_stats, 0));
}

/// Each entry's weight against its row's peak; an invisible entry weighs zero.
fn scales(context: &mut Device, stats: &Gathered, tags: &Tags, row_stats: &RowStats) -> Scales {
    let peak: Operand = context
        .sub
        .begin(row_stat(row_stats, 0))
        .fetch::<m![Heads, Groups], m![1 # 8]>()
        .collect::<m![Heads, Groups], m![1 # 8]>()
        .to_vrf();
    let scales: Terms = context
        .main
        .begin(stat(stats, 0))
        .fetch::<m![Heads, Groups], m![1 # 8]>()
        .collect::<m![Heads, Groups], m![1 # 8]>()
        .vector_init()
        .vector_tagged(tags)
        .vector_logic(LogicBinaryOpF32::BitAnd, Branched::imm(CLEARED, 0.0f32))
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(
            FpBinaryOp::SubF,
            Branched::imm(CLEARED, f32::INFINITY).rf(ADMITTED, &peak),
        )
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    context
        .sub
        .begin(scales.view())
        .fetch::<m![Heads, Groups], m![1 # 8]>()
        .collect::<m![Heads, Groups], m![1 # 8]>()
        .to_vrf()
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
    let attention =
        PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default(
        );
    let terms: Terms = attention
        .forward((
            &mut *context,
            Dm::<merge::Space, _> {
                tiles: Tiles::over(Rows::SIZE),
                value: (stat(stats, 1), (tags, scales)),
            },
        ))
        .value;
    attention.forward((
        &mut *context,
        Dm::<merge::Space, _> {
            tiles: Tiles::over(Rows::SIZE),
            value: ((terms.view(), ()), row_stat_mut(row_stats, 1)),
        },
    ));
}

/// Each slice's weighted output, still to be summed over the row's slices. Non-finite outputs are
/// cleared first: only an invisible entry's stale record holds one, and the visibility tags do not
/// cover the output's head lanes.
fn weighted(context: &mut Device, values: &Values, scales: &Scales) -> Weighted {
    PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default()
        .forward((
            &mut *context,
            Dm::<merge::Space, _> {
                tiles: Tiles::over(Rows::SIZE),
                value: (values.view(), scales),
            },
        ))
        .value
}

/// The run pass's entries, gathered; [`finish`] tags and merges them.
pub struct Gathers(Visibility, Gathered, Values);

pub(super) fn rows(
    context: &mut Device,
    (passes, entry): (&RunPasses, usize),
    partials: &Partials,
    reach: usize,
) -> Gathers {
    gather(
        context,
        partials,
        (
            &passes.statistics[entry],
            &passes.outputs[entry],
            &passes.visible[entry],
        ),
        reach,
    )
}

/// The batch's attended rows from its gathered run-pass entries.
pub(super) fn finish(
    context: &mut Device,
    Gathers(visible, stats, values): &Gathers,
    output: &mut Restored,
) {
    let tags = &visibility(context, visible);
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
            Dm::<merge::Space, _> {
                tiles: Tiles::over(Rows::SIZE),
                value: (row_stat(&row_stats, 1), ()),
            },
        ))
        .value;
    let local = weighted(context, values, &scales);
    let result: Weighted = attention
        .forward((
            &mut *context,
            Dm::<merge::Space, _> {
                tiles: Tiles::over(Rows::SIZE),
                value: (local.view(), &total),
            },
        ))
        .value;
    result
        .view()
        .slice_tile::<m![Runs], 1, m![Rows % 16 / 4, Rows / 32, Rows % 4, Runs = 1 # 8]>(0)
        .to_hbm_view(&mut context.tdma, output.view_mut());
}
