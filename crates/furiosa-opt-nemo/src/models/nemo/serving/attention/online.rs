//! A batch whose rows each keep one job slot through its waves carries every row's softmax in
//! its slot: each wave folds its keys into the slot's peak, mass and weighted sum, and the last
//! wave leaves the row attended, with no partial records and no merge. A batch of at most 32 rows
//! may give each row two slots of one PE instead, slot times `j` and `j + 4`, each over half the
//! row's blocks: its last wave merges each pair there ([`pairs`]).
use super::*;
use crate::Module;
use crate::compose::{Tile, Tiles};
use crate::ops::attention::gqa::PagedAttention;
use crate::ops::layout::Dm;

/// A slot's softmax so far: its row's peak, the mass of its weights against that peak and the
/// weighted sum of its values against it.
pub(super) struct Running {
    peak: Local,
    mass: Local,
    sum: Partial,
}

impl Running {
    fn new() -> Self {
        Self {
            peak: Local::new(),
            mass: Local::new(),
            sum: Partial::new(),
        }
    }

    pub(super) fn peak(&self) -> &Local {
        &self.peak
    }
}

/// Each slot's carried peak, or at a run's start the lowest finite one, so the wave's factor on
/// what it carried is zero: the host marks a continuing job's present pages as [`CARRIED`].
pub(super) fn restart(context: &mut Device, old: &Running, present: Counts<'_>) -> Local {
    let carried = operand(context, &old.peak);
    context
        .main
        .begin(present)
        .fetch::<m![Jobs % 8, Groups], m![1 # 8]>()
        .collect::<m![Jobs % 8, Groups], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Comparison([
            Cmp::Greater(CARRIED / 2.0),
            Cmp::False,
            Cmp::False,
            Cmp::False,
        ]))
        .vector_logic(
            LogicBinaryOpF32::BitAnd,
            Branched::imm(CLEARED, 0.0f32).imm(POSITIVE, 0.0f32),
        )
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(
            FpBinaryOp::AddF,
            Branched::imm(CLEARED, -1e30f32).rf(POSITIVE, &carried),
        )
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

/// Each slot's softmax so far as wave `at`'s partial: its peak and mass into `records`, its
/// weighted sum rounded into `rounded`.
pub(super) fn record(
    context: &mut Device,
    new: &Running,
    (records, rounded): (&mut Records, &mut Rounded),
    at: usize,
) {
    statistics(context, new, records, at);
    block::settle(context, &new.sum, rounded, at);
}

/// A run batch's last wave: [`sum`] and [`record`] in one pass, its sum rounded straight into
/// `rounded`, since no later wave reads it.
pub(super) fn close(
    context: &mut Device,
    weighted: &Partial,
    (old, new): (&Running, &Running),
    scale: &Local,
    (records, rounded): (&mut Records, &mut Rounded),
    at: usize,
) {
    statistics(context, new, records, at);
    let factor = operand(context, scale);
    context
        .main
        .begin_interleaved::<Term, _, _, _, _, _>(old.sum.view(), weighted.view())
        .fetch::<m![Jobs % 8, Groups, Head / 8, Term], m![Head % 8]>()
        .collect::<m![Jobs % 8, Groups, Head / 8, Term], m![Head % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Term, m![Jobs % 8, Groups, Head / 8]>()
        .vector_narrow_split::<m![Jobs % 8, Groups, Head / 4], m![Head % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &factor, ())
        .vector_fp_zip(FpBinaryOp::AddF)
        .vector_widen_concat::<m![Jobs % 8, Groups, Head / 8], m![Head % 8]>()
        .vector_final()
        .cast::<bf16, m![Head % 8 # 16]>()
        .commit_trim::<m![Head % 8]>()
        .commit_view(
            rounded
                .view_mut()
                .tile::<m![Burst], 1, m![Burst = 1 #{!} 8, Jobs % 8, Groups, Head]>(at),
        );
}

/// Each slot's peak and mass into wave `at`'s records.
fn statistics(context: &mut Device, new: &Running, records: &mut Records, at: usize) {
    context
        .main
        .begin(new.peak.view())
        .fetch::<m![Jobs % 8, Groups], m![1 # 8]>()
        .collect::<m![Jobs % 8, Groups], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, 0.0f32)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit_view(block::stat(records, at, 0));
    context
        .main
        .begin(new.mass.view())
        .fetch::<m![Jobs % 8, Groups], m![1 # 8]>()
        .collect::<m![Jobs % 8, Groups], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, 0.0f32)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit_view(block::stat(records, at, 1));
}

/// The softmax every slot carries: an even wave reads `odd` and writes `even`, an odd wave the
/// other way round, and `scale` holds a wave's factor on what its slots carried.
pub(super) struct State {
    even: Running,
    odd: Running,
    scale: Local,
}

impl State {
    pub(super) fn new() -> Self {
        Self {
            even: Running::new(),
            odd: Running::new(),
            scale: Local::new(),
        }
    }

    /// What an even wave if `EVEN`, else an odd one, reads and writes, and its factor.
    pub(super) fn wave<const EVEN: bool>(&mut self) -> (&Running, &mut Running, &mut Local) {
        let Self { even, odd, scale } = self;
        if EVEN {
            (&*odd, even, scale)
        } else {
            (&*even, odd, scale)
        }
    }

    /// No keys yet in what a batch's first wave, an even one, reads: the lowest finite peak, no
    /// mass and a zero sum. Garbage from an earlier owner of the memory clears too.
    pub(super) fn clear(&mut self, context: &mut Device) {
        let Self { even, odd, .. } = self;
        context
            .main
            .begin(even.peak.view())
            .fetch::<m![Jobs % 8, Groups], m![1 # 8]>()
            .collect::<m![Jobs % 8, Groups], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_logic(LogicBinaryOpF32::BitAnd, Branched::imm(CLEARED, 0.0f32))
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::AddF, -1e30f32)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit_view(odd.peak.view_mut());
        context
            .main
            .begin(even.mass.view())
            .fetch::<m![Jobs % 8, Groups], m![1 # 8]>()
            .collect::<m![Jobs % 8, Groups], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_logic(LogicBinaryOpF32::BitAnd, Branched::imm(CLEARED, 0.0f32))
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit_view(odd.mass.view_mut());
        context
            .main
            .begin(even.sum.view())
            .fetch::<m![Jobs % 8, Groups, Head / 8], m![Head % 8]>()
            .collect::<m![Jobs % 8, Groups, Head / 8], m![Head % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_logic(LogicBinaryOpF32::BitAnd, Branched::imm(CLEARED, 0.0f32))
            .vector_final()
            .commit_trim::<m![Head % 8]>()
            .commit_view(odd.sum.view_mut());
    }
}

/// `tensor`'s cells as a vector-register operand.
fn operand(context: &mut Device, tensor: &Local) -> Scalar {
    context
        .sub
        .begin(tensor.view())
        .fetch::<m![Jobs % 8, Groups], m![1 # 8]>()
        .collect::<m![Jobs % 8, Groups], m![1 # 8]>()
        .to_vrf()
}

axes![Term = 2, Half = 2];

/// A paired batch's per-row cells, a row's two slots merged: slot time `j` of each PE.
type Merged = DmTensor<f32, Chip, Cluster, Slices, m![Jobs % 4, Groups, Lane]>;
type Shares = VrfTensor<f32, Chip, Cluster, Slices, m![Jobs % 4, Groups, 1 # 8]>;
const POSITIVE: TagGuard =
    TagGuard::matches([BitReq::One, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]);

/// Each slot's peak so far into `new`, the larger of its carried peak `carried` and this wave's
/// over the page slices' `local` peaks, and the factor from the carried peak to it into `scale`;
/// returns the new peak, which this wave's weights are taken against.
pub(super) fn peak(
    context: &mut Device,
    local: &Local,
    (carried, new): (&Local, &mut Running),
    scale: &mut Local,
) -> Scalar {
    PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default()
        .forward((
            context,
            Dm::<Tile<Jobs, Jobs>, _> {
                tiles: Tiles::over(Jobs::SIZE),
                value: (
                    (local.view(), carried.view()),
                    (&mut new.peak, scale.view_mut()),
                ),
            },
        ))
        .value
}

/// `tensor`'s cells, the eight vector lanes of a record's `1 # 8` named.
fn lanes(
    tensor: &Local,
) -> DmTensorView<'_, f32, Chip, Cluster, Slices, m![Jobs % 8, Groups, Lane]> {
    // SAFETY: `1 # 8` and `Lane` lay out the same eight cells per job and group.
    unsafe {
        tensor
            .view()
            .reshape::<Chip, Cluster, Slices, m![Jobs % 8, Groups, Lane]>()
    }
}

/// Each slot's mass so far into `new`: what it carried at the new peak plus this wave's over the
/// page slices' `local` masses.
pub(super) fn mass(
    context: &mut Device,
    local: &Local,
    (old, new): (&Running, &mut Running),
    scale: &Local,
) {
    let mut now = Local::new();
    block::job_mass(context, local, now.view_mut());
    let factor = operand(context, scale);
    // SAFETY: as `lanes`.
    let mass = unsafe {
        new.mass
            .view_mut()
            .reshape::<Chip, Cluster, Slices, m![Jobs % 8, Groups, Lane]>()
    };
    PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default()
        .forward((
            context,
            Dm::<Tile<Jobs, Jobs>, _> {
                tiles: Tiles::over(Jobs::SIZE),
                value: ((lanes(&old.mass), lanes(&now)), (&factor, mass)),
            },
        ));
}

/// Each slot's weighted sum so far into `new`: what it carried at the new peak plus this wave's
/// `weighted` values.
pub(super) fn sum(
    context: &mut Device,
    weighted: &Partial,
    (old, new): (&Running, &mut Running),
    scale: &Local,
) {
    let factor = operand(context, scale);
    PagedAttention::<Heads, Groups, Head, { crate::models::nemo::SCORE_SCALE }, Chip>::default()
        .forward((
            context,
            Dm::<Tile<Jobs, Jobs>, _> {
                tiles: Tiles::over(Jobs::SIZE),
                value: (
                    (old.sum.view(), weighted.view()),
                    &factor,
                    new.sum.view_mut(),
                ),
            },
        ));
}

/// The batch's last wave: each slot's sum as [`sum`] over its mass so far, its row attended,
/// stored as row `8 * (slot % 8) + slot / 8` of `output`, the row the plan deals to the slot (the
/// PEs read their slots' pages in turn, so a PE's slots hold rows eight apart); an idle slot's
/// row is zero.
pub(super) fn finish(
    context: &mut Device,
    weighted: &Partial,
    (old, new): (&Running, &mut Running),
    scale: &Local,
    output: &mut Restored,
) {
    let factor = operand(context, scale);
    let total: Scalar = context
        .main
        .begin(new.mass.view())
        .fetch::<m![Jobs % 8, Groups], m![1 # 8]>()
        .collect::<m![Jobs % 8, Groups], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, 1e-20f32)
        .vector_final()
        .to_vrf(&mut context.sub);
    context
        .main
        .begin_interleaved::<Term, _, _, _, _, _>(old.sum.view(), weighted.view())
        .fetch::<m![Jobs % 8, Groups, Head / 8, Term], m![Head % 8]>()
        .collect::<m![Jobs % 8, Groups, Head / 8, Term], m![Head % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Term, m![Jobs % 8, Groups, Head / 8]>()
        .vector_narrow_split::<m![Jobs % 8, Groups, Head / 4], m![Head % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &factor, ())
        .vector_fp_zip(FpBinaryOp::AddF)
        .vector_fp_div(&total)
        .vector_widen_concat::<m![Jobs % 8, Groups, Head / 8], m![Head % 8]>()
        .vector_final()
        .commit_trim::<m![Head % 8]>()
        .commit_view(new.sum.view_mut());
    store(context, &new.sum, output);
}

/// Each slot's row of `rows` as row `8 * (slot % 8) + slot / 8` of `output`.
fn store(context: &mut Device, rows: &Partial, output: &mut Restored) {
    // SAFETY: row `8 * (slot % 8) + slot / 8` of the attended rows holds slot `slot`'s row.
    let attended = unsafe {
        output
            .view_mut()
            .reshape::<Chip, m![Jobs % 8, Jobs / 8, Heads, Groups, Head]>()
    };
    rows.view()
        .slice_tile::<m![Page], 1, Stored>(0)
        .to_hbm_view(&mut context.tdma, attended);
}

/// A paired batch's last wave: each slot's sum as [`sum`], then each row's two slots merged, its
/// rows attended, stored as [`finish`] stores a slot's row: the first slot of a pair holds the
/// row, the second a zero row past the batch's 32. A vector pass reads one register operand for
/// both its streams, so each slot's sum and mass first take their own slot's factor.
pub(super) fn pairs(
    context: &mut Device,
    weighted: &Partial,
    (old, new): (&Running, &mut Running),
    scale: &Local,
    output: &mut Restored,
) {
    sum(context, weighted, (old, &mut *new), scale);
    // SAFETY: slot time `Jobs % 8` is `4 * Half + Jobs % 4`, and `1 # 8` and `Lane` lay out the
    // same eight cells.
    let peaks = unsafe {
        new.peak
            .view()
            .reshape::<Chip, Cluster, Slices, m![Half, Jobs % 4, Groups, Lane]>()
    };
    // The first half's peak less the second's.
    let gap: Merged = context
        .main
        .begin(peaks)
        .fetch::<m![Jobs % 4, Groups, Half], m![Lane]>()
        .collect::<m![Jobs % 4, Groups, Half], m![Lane]>()
        .vector_init()
        .vector_intra_slice_unzip::<Half, m![Jobs % 4, Groups]>()
        .vector_narrow_split::<m![Jobs % 4, Groups, Lane / 4], m![Lane % 4]>()
        .vector_fp_zip(FpBinaryOp::SubF)
        .vector_widen_concat::<m![Jobs % 4, Groups], m![Lane]>()
        .vector_final()
        .commit_trim::<m![Lane]>()
        .commit();
    // Each slot's factor to its pair's peak.
    let mut shares = Local::new();
    share::<true>(context, &gap, &mut shares);
    share::<false>(context, &gap, &mut shares);
    let factor = operand(context, &shares);
    let scaled: Local = context
        .main
        .begin(new.mass.view())
        .fetch::<m![Jobs % 8, Groups], m![1 # 8]>()
        .collect::<m![Jobs % 8, Groups], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &factor)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // SAFETY: as `peaks`.
    let masses = unsafe {
        scaled
            .view()
            .reshape::<Chip, Cluster, Slices, m![Half, Jobs % 4, Groups, Lane]>()
    };
    let mass: Merged = context
        .main
        .begin(masses)
        .fetch::<m![Jobs % 4, Groups, Half], m![Lane]>()
        .collect::<m![Jobs % 4, Groups, Half], m![Lane]>()
        .vector_init()
        .vector_intra_slice_unzip::<Half, m![Jobs % 4, Groups]>()
        .vector_narrow_split::<m![Jobs % 4, Groups, Lane / 4], m![Lane % 4]>()
        .vector_fp_zip(FpBinaryOp::AddF)
        .vector_widen_concat::<m![Jobs % 4, Groups], m![Lane]>()
        .vector_final()
        .commit_trim::<m![Lane]>()
        .commit();
    // SAFETY: lane 0 of each cell holds its value, which `1 # 8` reads.
    let total: Shares = context
        .main
        .begin(unsafe {
            mass.view()
                .reshape::<Chip, Cluster, Slices, m![Jobs % 4, Groups, 1 # 8]>()
        })
        .fetch::<m![Jobs % 4, Groups], m![1 # 8]>()
        .collect::<m![Jobs % 4, Groups], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, 1e-20f32)
        .vector_final()
        .to_vrf(&mut context.sub);
    let sums: Partial = context
        .main
        .begin(new.sum.view())
        .fetch::<m![Jobs % 8, Groups, Head / 8], m![Head % 8]>()
        .collect::<m![Jobs % 8, Groups, Head / 8], m![Head % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Jobs % 8, Groups, Head / 4], m![Head % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &factor)
        .vector_widen_concat::<m![Jobs % 8, Groups, Head / 8], m![Head % 8]>()
        .vector_final()
        .commit_trim::<m![Head % 8]>()
        .commit();
    // SAFETY: as `peaks`.
    let halves = unsafe {
        sums.view()
            .reshape::<Chip, Cluster, Slices, m![Half, Jobs % 4, Groups, Head]>()
    };
    let mut rows = Partial::new();
    context
        .main
        .begin(halves)
        .fetch::<m![Jobs % 4, Groups, Head / 8, Half], m![Head % 8]>()
        .collect::<m![Jobs % 4, Groups, Head / 8, Half], m![Head % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Half, m![Jobs % 4, Groups, Head / 8]>()
        .vector_narrow_split::<m![Jobs % 4, Groups, Head / 4], m![Head % 4]>()
        .vector_fp_zip(FpBinaryOp::AddF)
        .vector_fp_div(&total)
        .vector_widen_concat::<m![Jobs % 4, Groups, Head / 8], m![Head % 8]>()
        .vector_final()
        .commit_trim::<m![Head % 8]>()
        .commit_view(
            // SAFETY: as `peaks`.
            unsafe {
                rows.view_mut()
                    .reshape::<Chip, Cluster, Slices, m![Half, Jobs % 4, Groups, Head]>()
            }
            .tile::<m![Half], 1, m![Half = 1 #{!} 2, Jobs % 4, Groups, Head]>(0),
        );
    // SAFETY: as `peaks`.
    let second = unsafe {
        sums.view()
            .reshape::<Chip, Cluster, Slices, m![Half, Jobs % 4, Groups, Head]>()
    };
    context
        .main
        .begin(second.tile::<m![Half], 1, m![Half = 1 # 2, Jobs % 4, Groups, Head]>(1))
        .fetch::<m![Jobs % 4, Groups, Head / 8], m![Head % 8]>()
        .collect::<m![Jobs % 4, Groups, Head / 8], m![Head % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, Branched::imm(CLEARED, 0.0f32))
        .vector_final()
        .commit_trim::<m![Head % 8]>()
        .commit_view(
            // SAFETY: as `peaks`.
            unsafe {
                rows.view_mut()
                    .reshape::<Chip, Cluster, Slices, m![Half, Jobs % 4, Groups, Head]>()
            }
            .tile::<m![Half], 1, m![Half = 1 #{!} 2, Jobs % 4, Groups, Head]>(1),
        );
    store(context, &rows, output);
}

/// Each pair's first slot's factor to the pair's peak, `exp(min(gap, 0))`, if `FIRST`, else its
/// second's, `exp(min(-gap, 0))`, into that slot's cells of `shares`.
fn share<const FIRST: bool>(context: &mut Device, gap: &Merged, shares: &mut Local) {
    let past = if FIRST {
        Cmp::Greater(0.0f32)
    } else {
        Cmp::Less(0.0f32)
    };
    // SAFETY: slot time `Jobs % 8` is `4 * Half + Jobs % 4`, and `1 # 8` and `Lane` lay out the
    // same eight cells.
    let halves = unsafe {
        shares
            .view_mut()
            .reshape::<Chip, Cluster, Slices, m![Half, Jobs % 4, Groups, Lane]>()
    };
    context
        .main
        .begin(gap.view())
        .fetch::<m![Jobs % 4, Groups], m![Lane]>()
        .collect::<m![Jobs % 4, Groups], m![Lane]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Comparison([
            past,
            Cmp::False,
            Cmp::False,
            Cmp::False,
        ]))
        .vector_logic(LogicBinaryOpF32::BitAnd, Branched::imm(POSITIVE, 0.0f32))
        .vector_narrow_split::<m![Jobs % 4, Groups, Lane / 4], m![Lane % 4]>()
        .vector_fp_binary(
            FpBinaryOp::MulF(FpMulAlu::Mul0),
            if FIRST { 1.0f32 } else { -1.0f32 },
        )
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_widen_concat::<m![Jobs % 4, Groups], m![Lane]>()
        .vector_final()
        .commit_trim::<m![Lane]>()
        .commit_view(
            halves.tile::<m![Half], 1, m![Half = 1 #{!} 2, Jobs % 4, Groups, Lane]>(if FIRST {
                0
            } else {
                1
            }),
        );
}
