//! A batch whose waves each hold whole rows, attended on chip: block `b` of the batch's rows is 16
//! rows of at most [`ROW_JOBS`] jobs, row `16 b + 4 q + 2 c + h` in slot times `4 h .. 4 h + 4` of
//! PE `q` of cluster `c`. Wave `w` attends block `w`, except that a two-wave batch whose last block
//! is partial attends it in wave 0 and block 0 in wave 1.
//! The wave weighs each key against its row's peak over the row's jobs, so a row's attended output
//! is its jobs' weighted values summed over their masses: no job partial needs rescaling.
use super::*;

axes![Half = 2, Wave = 4];
/// The rows of a block.
pub(super) const ROWS: usize = Jobs::SIZE / ROW_JOBS;
const _: () = assert!(Half::SIZE * ROW_JOBS == 8 && Wave::SIZE * ROWS == Rows::SIZE);

type RowStat = DmTensor<f32, Chip, Cluster, Slices, m![Half, Groups, 1 # 8]>;
type RowOperand = VrfTensor<f32, Chip, Cluster, Slices, m![Half, Groups, 1 # 8]>;
type Attended = DmTensor<f32, Chip, Cluster, Slices, m![Half, Groups, Head]>;

/// Every row of `output` zero, for the waves to store theirs over.
pub(super) fn clear(context: &mut Device, output: &mut Restored) {
    let mut zero = merge::Weighted::new();
    zero.view_mut().memset(0.0f32, &mut context.sub);
    merge::store(context, &zero, output);
}

/// Each row's stat over its jobs' page-slice stats `local`: `Max` for its peak (at least the
/// lowest finite score), `Add` for its mass (at least 1e-20), as a register operand.
fn row(context: &mut Device, local: &Local, peak: bool) -> RowOperand {
    let (inter, intra, clip) = if peak {
        (
            InterSliceReduceOpF32::Max,
            IntraSliceReduceOpF32::Max,
            (ClipBinaryOpF32::Max, -1e30f32),
        )
    } else {
        (
            InterSliceReduceOpF32::Add,
            IntraSliceReduceOpF32::Add,
            (ClipBinaryOpF32::Add, 1e-20f32),
        )
    };
    // SAFETY: slot time `Jobs % 8` is `4 * Half + Jobs % 4`.
    let jobs = unsafe {
        local
            .view()
            .reshape::<Chip, Cluster, Slices, m![Half, Jobs % 4, Groups, 1 # 8]>()
    };
    let stat: RowStat = context
        .main
        .begin(jobs)
        .fetch::<m![Half, Groups, Jobs % 4], m![1 # 8]>()
        .collect::<m![Half, Groups, Jobs % 4], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce_span(inter, 4)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<Jobs, m![Half, Groups], m![1 # 4]>(intra)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(clip.0, clip.1)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    context
        .sub
        .begin(stat.view())
        .fetch::<m![Half, Groups], m![1 # 8]>()
        .collect::<m![Half, Groups], m![1 # 8]>()
        .to_vrf()
}

/// Each key's weight against its row's peak, into `weights`: an absent key sits at the lowest
/// bf16, so it weighs zero, and a row of idle jobs weighs nothing.
pub(super) fn weigh(
    context: &mut Device,
    (scores, local): (&Weights, &Local),
    weights: &mut Weights,
) {
    let top = row(context, local, true);
    // SAFETY: as `row`.
    let keys = unsafe {
        scores
            .view()
            .reshape::<Chip, Cluster, Slices, m![Half, Jobs % 4, Groups, Tokens / 16]>()
    };
    // SAFETY: as `row`.
    let into = unsafe {
        weights
            .view_mut()
            .reshape::<Chip, Cluster, Slices, m![Half, Jobs % 4, Groups, Tokens / 16]>()
    };
    context
        .main
        .begin(keys)
        .fetch::<m![Half, Groups, Jobs % 4, Tokens / 128], m![Tokens / 16 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Half, Groups, Jobs % 4, Tokens / 128], m![Tokens / 16 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Half, Groups, Jobs % 4, Tokens / 64], m![Tokens / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, &top)
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_widen_concat::<m![Half, Groups, Jobs % 4, Tokens / 128], m![Tokens / 16 % 8]>()
        .vector_final()
        .cast::<bf16, m![Tokens / 16 % 8 # 16]>()
        .commit_trim::<m![Tokens / 16 % 8]>()
        .commit_view(into);
}

/// A wave's rows, each its jobs' `weighted` values summed over their page-slice `masses`, into
/// block `block` of `output`.
pub(super) fn finish(
    context: &mut Device,
    (masses, weighted): (&Local, &Partial),
    block: usize,
    output: &mut Restored,
) {
    let total = row(context, masses, false);
    // SAFETY: as `row`.
    let jobs = unsafe {
        weighted
            .view()
            .reshape::<Chip, Cluster, Slices, m![Half, Jobs % 4, Groups, Head]>()
    };
    let rows: Attended = context
        .main
        .begin(jobs)
        .fetch::<m![Half, Groups, Head / 8, Jobs % 4], m![Head % 8]>()
        .collect::<m![Half, Groups, Head / 8, Jobs % 4], m![Head % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Half, Groups, Head / 8, Jobs % 4, Head % 8 / 4], m![Head % 4]>()
        .vector_intra_slice_reduce::<Jobs, m![Half, Groups, Head / 4], m![Head % 4]>(
            IntraSliceReduceOpF32::Add,
        )
        .vector_fp_div(&total)
        .vector_widen_concat::<m![Half, Groups, Head / 8], m![Head % 8]>()
        .vector_final()
        .commit_trim::<m![Head % 8]>()
        .commit();
    // SAFETY: block `b`'s row in half `h` of cluster `c`'s PE `q` is row `16 b + 4 q + 2 c + h`.
    let attended = unsafe {
        output
            .view_mut()
            .reshape::<Chip, m![Wave, Jobs % 32 / 8, Jobs / 32, Half, Heads, Groups, Head]>()
    };
    rows.view().slice_tile::<m![Page], 1, Stored>(0).to_hbm_view(
        &mut context.tdma,
        attended.tile::<m![Wave], 1, m![Wave = 1 #{!} 4, Jobs % 32 / 8, Jobs / 32, Half, Heads, Groups, Head]>(
            block,
        ),
    );
}
