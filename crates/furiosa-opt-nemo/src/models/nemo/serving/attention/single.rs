//! A batch of one decode row whose jobs fill one wave, merged on chip: the wave runs over its jobs'
//! first pages only, then its job partials meet in data memory instead of through HBM, as the
//! oracle's B1 launch never writes a partial.
//!
//! - The wave stages the row's one query and its own counts, not a burst's.
//! - It computes each job's first [`SINGLE_PAGES`] pages ([`row`]), as many as the host gives one.
//! - Each PE merges its eight jobs in time; two transfers then land every PE's merged peak and
//!   mass, and output, of a head on one slice of the head pair's cluster.
//! - The eight merge there at the batch's [`Merging::finish`], so the step's O weight lands under
//!   that merge as under any batch's final merge.
//! - The host marks the batch with no job waves; its first wave's tables hold the row's jobs.
use super::*;

mod row;

/// The job stats of the wave's burst: 0 the peak, 1 the mass.
type Job<'l> = DmTensorView<
    'l,
    f32,
    Chip,
    Cluster,
    Slices,
    m![Burst = 1 # 8, Jobs % 8, Stat = 1 # 2, Groups, 1 # 8],
>;
/// Each PE's merged peak and mass: 0 the peak, 1 the mass.
type LocalStats = DmTensor<f32, Chip, Cluster, Slices, m![Stat, Groups, 1 # 8]>;
type LocalStat<'l> =
    DmTensorViewMut<'l, f32, Chip, Cluster, Slices, m![Stat = 1 #{!} 2, Groups, 1 # 8]>;
type LocalOperand = VrfTensor<f32, Chip, Cluster, Slices, m![Groups, 1 # 8]>;
type Terms = DmTensor<f32, Chip, Cluster, Slices, m![Groups, Jobs % 8, 1 # 8]>;
type Scales = VrfTensor<f32, Chip, Cluster, Slices, m![Groups, Jobs % 8, 1 # 8]>;
type Scaled = DmTensor<f32, Chip, Cluster, Slices, m![Groups, Jobs % 8, Head]>;
type Summed = DmTensor<f32, Chip, Cluster, Slices, m![Groups, Head]>;
/// One slice of each head ring, its PE's merged partial.
type Rings = m![Jobs % 32 / 8, Heads % 2, Heads / 2, Page = 1 # 16];
type Owner = m![Heads / 2];
/// A cluster's two heads, one slice each, every PE's partial in time.
type Duo = m![Heads % 2, 1 # 128];
type PartStats = DmTensor<f32, Chip, Owner, Duo, m![Jobs / 8, Stat, Groups, 1 # 8]>;
type Parts<'l> = DmTensorView<'l, f32, Chip, Owner, Duo, m![Jobs / 8, Stat = 1 # 2, Groups, 1 # 8]>;
type PartTerms = DmTensor<f32, Chip, Owner, Duo, m![Groups, Jobs / 8, 1 # 8]>;
type PartScales = VrfTensor<f32, Chip, Owner, Duo, m![Groups, Jobs / 8, 1 # 8]>;
type Products = DmTensor<f32, Chip, Owner, Duo, m![Jobs / 8, Groups, Head]>;
type PartScaled = DmTensor<f32, Chip, Owner, Duo, m![Groups, Jobs / 8, Head]>;
type Total = DmTensor<f32, Chip, Owner, Duo, m![Groups, 1 # 8]>;
type TotalOperand = VrfTensor<f32, Chip, Owner, Duo, m![Groups, 1 # 8]>;
type Merged = DmTensor<f32, Chip, Owner, Duo, m![Groups, Head]>;

/// Record `index` of the wave's job stats.
fn job(records: &Records, index: usize) -> Job<'_> {
    records
        .view()
        .tile::<m![Burst], 1, m![Burst = 1 # 8, Jobs % 8, Stat, Groups, 1 # 8]>(0)
        .tile::<m![Stat], 1, m![Burst = 1 # 8, Jobs % 8, Stat = 1 # 2, Groups, 1 # 8]>(index)
}

/// The row's query in every slot of the wave and the wave's counts, as a burst's first wave lands
/// its burst's.
pub(super) fn stage(
    context: &mut Device,
    (history, queries): (&History<'_>, &qkv::Queries),
    buffers: &mut Tables,
) {
    // SAFETY: a capacity row's query heads are its `Heads × Groups` heads in head order.
    unsafe {
        queries
            .view()
            .reshape::<Chip, m![Capacity, Heads, Groups, Head]>()
    }
    .tile::<m![Capacity], 1, m![Capacity = 1 # Capacity, Heads, Groups, Head]>(
        Rows::SIZE * history.batch,
    )
    .to_dm_view(
        &mut context.tdma,
        buffers
            .staged
            .view_mut()
            .tile::<m![Parity], 1, m![Parity = 1 #{!} 2, Burst, Groups, Head]>(history.batch % 2)
            .tile::<m![Burst], 1, m![Parity = 1 #{!} 2, Burst = 1 #{!} 8, Groups, Head]>(0),
    );
    history.plan.counts[history.batch * (Blocks::SIZE / Burst::SIZE)]
        .view()
        .tile::<m![Burst], 1, m![Page, Burst = 1 # 8, Jobs, Lane]>(0)
        .to_dm_view(
            &mut context.tdma,
            buffers
                .landed
                .view_mut()
                .tile::<m![Parity], 1, m![Parity = 1 #{!} 2, Burst, Jobs % 8, Lane]>(
                    history.batch % 2,
                )
                .tile::<m![Burst], 1, m![Parity = 1 #{!} 2, Burst = 1 #{!} 8, Jobs % 8, Lane]>(0),
        );
}

fn local_stat(stats: &mut LocalStats, index: usize) -> LocalStat<'_> {
    stats
        .view_mut()
        .tile::<m![Stat], 1, m![Stat = 1 #{!} 2, Groups, 1 # 8]>(index)
}

fn part(stats: &PartStats, index: usize) -> Parts<'_> {
    stats
        .view()
        .tile::<m![Stat], 1, m![Jobs / 8, Stat = 1 # 2, Groups, 1 # 8]>(index)
}

/// Every PE's merged partial of each head on one slice of the head pair's cluster, the PEs in
/// time: [`finish`] merges them.
pub(super) struct Collected {
    stats: PartStats,
    products: Products,
}

impl Collected {
    pub(super) fn new() -> Self {
        Self {
            stats: PartStats::new(),
            products: Products::new(),
        }
    }
}

/// Each PE's merge of the wave's partials into `collected`, and `output` zero until [`finish`]
/// stores the row.
pub(super) fn collect(
    context: &mut Device,
    (records, rounded): (&Records, &Rounded),
    collected: &mut Collected,
    output: &mut Restored,
) {
    // Each PE's eight jobs.
    let mut local = LocalStats::new();
    local_peak(context, records, local_stat(&mut local, 0));
    let scales = local_scales(context, records, &local);
    local_mass(context, records, &scales, local_stat(&mut local, 1));
    let product = local_product(context, rounded, &scales);
    local
        .view()
        .slice_tile::<m![Page], 1, Rings>(0)
        .to_dm_view(&mut context.tdma, collected.stats.view_mut());
    product
        .view()
        .slice_tile::<m![Page], 1, Rings>(0)
        .to_dm_view(&mut context.tdma, collected.products.view_mut());
    // After the transfers: its cluster acquire would hold the per-PE passes' issue.
    let mut zero = merge::Weighted::new();
    zero.view_mut().memset(0.0f32, &mut context.sub);
    merge::store(context, &zero, output);
}

/// The row of the collected partials into `output`.
pub(super) fn finish(context: &mut Device, collected: &Collected, output: &mut Restored) {
    let top = part_peak(context, part(&collected.stats, 0));
    let scales = part_scales(context, part(&collected.stats, 0), &top);
    let total = part_mass(context, part(&collected.stats, 1), &scales);
    let result = part_product(context, (&collected.products, &scales), &total);
    result.view().to_hbm_view(
        &mut context.tdma,
        output
            .view_mut()
            .tile::<m![Rows], 1, m![Rows = 1 #{!} Rows, Heads, Groups, Head]>(0),
    );
}

fn local_peak(context: &mut Device, records: &Records, peak: LocalStat<'_>) {
    context
        .main
        .begin(job(records, 0))
        .fetch::<m![Groups, Jobs % 8], m![1 # 8]>()
        .collect::<m![Groups, Jobs % 8], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<Jobs, m![Groups], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit_view(peak);
}

/// Each job's weight against its PE's peak: an empty job's peak sits at -1e30, so it weighs zero.
fn local_scales(context: &mut Device, records: &Records, local: &LocalStats) -> Scales {
    let top: LocalOperand = context
        .sub
        .begin(
            local
                .view()
                .tile::<m![Stat], 1, m![Stat = 1 # 2, Groups, 1 # 8]>(0),
        )
        .fetch::<m![Groups], m![1 # 8]>()
        .collect::<m![Groups], m![1 # 8]>()
        .to_vrf();
    let scales: Terms = context
        .main
        .begin(job(records, 0))
        .fetch::<m![Groups, Jobs % 8], m![1 # 8]>()
        .collect::<m![Groups, Jobs % 8], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, &top)
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    context
        .sub
        .begin(scales.view())
        .fetch::<m![Groups, Jobs % 8], m![1 # 8]>()
        .collect::<m![Groups, Jobs % 8], m![1 # 8]>()
        .to_vrf()
}

fn local_mass(context: &mut Device, records: &Records, scales: &Scales, mass: LocalStat<'_>) {
    let terms: Terms = context
        .main
        .begin(job(records, 1))
        .fetch::<m![Groups, Jobs % 8], m![1 # 8]>()
        .collect::<m![Groups, Jobs % 8], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scales)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    context
        .main
        .begin(terms.view())
        .fetch::<m![Groups, Jobs % 8], m![1 # 8]>()
        .collect::<m![Groups, Jobs % 8], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<Jobs, m![Groups], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit_view(mass);
}

/// Each PE's jobs' outputs weighed and summed.
fn local_product(context: &mut Device, rounded: &Rounded, scales: &Scales) -> Summed {
    let scaled: Scaled = context
        .main
        .begin(
            rounded
                .view()
                .tile::<m![Burst], 1, m![Burst = 1 # 8, Jobs % 8, Groups, Head]>(0),
        )
        .fetch::<m![Groups, Jobs % 8, Head / 8], m![Head % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Groups, Jobs % 8, Head / 8], m![Head % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Groups, Jobs % 8, Head / 4], m![Head % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scales)
        .vector_widen_concat::<m![Groups, Jobs % 8, Head / 8], m![Head % 8]>()
        .vector_final()
        .commit_trim::<m![Head % 8]>()
        .commit();
    context
        .main
        .begin(scaled.view())
        .fetch::<m![Groups, Head / 8, Jobs % 8], m![Head % 8]>()
        .collect::<m![Groups, Head / 8, Jobs % 8], m![Head % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Groups, Head / 8, Jobs % 8, Head % 8 / 4], m![Head % 4]>()
        .vector_intra_slice_reduce::<Jobs, m![Groups, Head / 4], m![Head % 4]>(
            IntraSliceReduceOpF32::Add,
        )
        .vector_widen_concat::<m![Groups, Head / 8], m![Head % 8]>()
        .vector_final()
        .commit_trim::<m![Head % 8]>()
        .commit()
}

fn part_peak(context: &mut Device, peaks: Parts<'_>) -> Total {
    context
        .main
        .begin(peaks)
        .fetch::<m![Groups, Jobs / 8], m![1 # 8]>()
        .collect::<m![Groups, Jobs / 8], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<Jobs, m![Groups], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

fn part_scales(context: &mut Device, peaks: Parts<'_>, top: &Total) -> PartScales {
    let top: TotalOperand = context
        .sub
        .begin(top.view())
        .fetch::<m![Groups], m![1 # 8]>()
        .collect::<m![Groups], m![1 # 8]>()
        .to_vrf();
    let scales: PartTerms = context
        .main
        .begin(peaks)
        .fetch::<m![Groups, Jobs / 8], m![1 # 8]>()
        .collect::<m![Groups, Jobs / 8], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, &top)
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    context
        .sub
        .begin(scales.view())
        .fetch::<m![Groups, Jobs / 8], m![1 # 8]>()
        .collect::<m![Groups, Jobs / 8], m![1 # 8]>()
        .to_vrf()
}

fn part_mass(context: &mut Device, masses: Parts<'_>, scales: &PartScales) -> Total {
    let terms: PartTerms = context
        .main
        .begin(masses)
        .fetch::<m![Groups, Jobs / 8], m![1 # 8]>()
        .collect::<m![Groups, Jobs / 8], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scales)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    context
        .main
        .begin(terms.view())
        .fetch::<m![Groups, Jobs / 8], m![1 # 8]>()
        .collect::<m![Groups, Jobs / 8], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<Jobs, m![Groups], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, 1e-20f32)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

/// The PEs' outputs weighed, summed and divided by the row's mass.
fn part_product(
    context: &mut Device,
    (products, scales): (&Products, &PartScales),
    total: &Total,
) -> Merged {
    let scaled: PartScaled = context
        .main
        .begin(products.view())
        .fetch::<m![Groups, Jobs / 8, Head / 8], m![Head % 8]>()
        .collect::<m![Groups, Jobs / 8, Head / 8], m![Head % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Groups, Jobs / 8, Head / 4], m![Head % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scales)
        .vector_widen_concat::<m![Groups, Jobs / 8, Head / 8], m![Head % 8]>()
        .vector_final()
        .commit_trim::<m![Head % 8]>()
        .commit();
    let total: TotalOperand = context
        .sub
        .begin(total.view())
        .fetch::<m![Groups], m![1 # 8]>()
        .collect::<m![Groups], m![1 # 8]>()
        .to_vrf();
    let summed: Merged = context
        .main
        .begin(scaled.view())
        .fetch::<m![Groups, Head / 8, Jobs / 8], m![Head % 8]>()
        .collect::<m![Groups, Head / 8, Jobs / 8], m![Head % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Groups, Head / 8, Jobs / 8, Head % 8 / 4], m![Head % 4]>()
        .vector_intra_slice_reduce::<Jobs, m![Groups, Head / 4], m![Head % 4]>(
            IntraSliceReduceOpF32::Add,
        )
        .vector_widen_concat::<m![Groups, Head / 8], m![Head % 8]>()
        .vector_final()
        .commit_trim::<m![Head % 8]>()
        .commit();
    context
        .main
        .begin(summed.view())
        .fetch::<m![Groups, Head / 8], m![Head % 8]>()
        .collect::<m![Groups, Head / 8], m![Head % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Groups, Head / 4], m![Head % 4]>()
        .vector_fp_div(&total)
        .vector_widen_concat::<m![Groups, Head / 8], m![Head % 8]>()
        .vector_final()
        .commit_trim::<m![Head % 8]>()
        .commit()
}
