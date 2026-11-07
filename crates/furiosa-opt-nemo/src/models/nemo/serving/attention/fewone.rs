//! One decode row's attention with its keys spread over every slice of the head pair's cluster:
//! a slice holds one head, one page token and every eighth page of the row, so each Main command
//! works on 16 keys a slice and the softmax reduces across 128 slices in the vector engine.
//!
//! - Cluster `c` holds heads `2 c` and `2 c + 1`; page `p` of the row lands on page group
//!   `p % 8`, at time `p / 8`.
//! - The host gives the row's page offsets at the start of page table 1, its length first in
//!   length table 1, and each slice's page times less its count of present pages as merge
//!   visibility 0.
use super::*;
use crate::models::nemo::serving::Rows as WaveRows;

axes![Keys = 2048];

type Cluster1 = m![Heads / 2];
/// A cluster's slices: a head of the pair, a page group, a page token.
type Ring = m![Heads % 2, Keys % 128 / 16, Page];
type RowPages = HbmTensor<i32, Chip, m![Keys / 16]>;
type RowLength = HbmTensor<i32, Chip, m![1]>;
type Raw = DmTensor<f8e4m3, Chip, Cluster1, Ring, m![Keys / 128, Kv, Head]>;
type Decoded = DmTensor<bf16, Chip, Cluster1, Ring, m![Keys / 128, Head]>;
type Query = DmTensor<bf16, Chip, Cluster1, m![Heads % 2, 1 # 128], m![Groups, Head]>;
type Lanes = TrfTensor<bf16, Chip, Cluster1, Ring, m![Groups], m![Head]>;
type Scores = DmTensor<f32, Chip, Cluster1, Ring, m![Groups, Keys / 128]>;
type Weights = DmTensor<bf16, Chip, Cluster1, Ring, m![Groups, Keys / 128]>;
type Probability = TrfTensor<bf16, Chip, Cluster1, Ring, m![Groups], m![Keys / 128]>;
type Local = DmTensor<f32, Chip, Cluster1, Ring, m![Groups, 1 # 8]>;
type Operand = VrfTensor<f32, Chip, Cluster1, Ring, m![Groups, 1 # 8]>;
type Beyond = DmTensor<f32, Chip, Cluster1, Ring, m![Keys / 128]>;
type Partial = DmTensor<f32, Chip, Cluster1, Ring, m![Groups, Head]>;

/// The row's page offsets and length.
fn tables(history: &History<'_>) -> (RowPages, RowLength) {
    // SAFETY: the host's page table 1 starts with the row's pages, length table 1 with its length.
    unsafe {
        (
            history.plan.pages[1]
                .view()
                .reshape::<Chip, m![Keys / 16 # 2048]>()
                .alias::<Chip, m![Keys / 16]>(),
            history.few[0]
                .view()
                .reshape::<Chip, m![1 # 64]>()
                .alias::<Chip, m![1]>(),
        )
    }
}

/// Each slice's page times past its count of present pages: a key is absent where it is >= 0.
fn stage(context: &mut Device, plan: &Plan, beyond: &mut Beyond) {
    // SAFETY: the host gives page group `g`'s token-`t` times `[g][t][k]` as merge visibility 0.
    let table = unsafe {
        plan.merge.visible[0]
            .view()
            .reshape::<Chip, m![Keys % 128 / 16, Page, Keys / 128]>()
    };
    table.to_dm_view(&mut context.tdma, beyond.view_mut());
}

/// The row's query heads, one slice of each head's ring.
fn query(context: &mut Device, queries: &qkv::Queries, query: &mut Query) {
    // SAFETY: a capacity row's query heads are its `Heads × Groups` heads in head order.
    unsafe {
        queries
            .view()
            .reshape::<Chip, m![Capacity, Heads, Groups, Head]>()
    }
    .tile::<m![Capacity], 1, m![Capacity = 1 # Capacity, Heads, Groups, Head]>(0)
    .to_dm_view(&mut context.tdma, query.view_mut());
}

fn lanes(context: &mut Device, query: &Query) -> Lanes {
    context
        .sub
        .begin(query.view())
        .fetch::<m![Groups], m![Head]>()
        .switch::<Ring, m![Groups]>(SwitchConfig::CustomBroadcast { ring_size: 128 })
        .collect::<m![Groups, Head / 16], m![Head % 16]>()
        .to_trf()
}

fn gather(
    context: &mut Device,
    (pool, entry): (&Pool, usize),
    (pages, length): &(RowPages, RowLength),
    raw: &mut Raw,
) {
    // SAFETY: as `attention::gather`.
    unsafe {
        pool.gather::<m![1], m![Page, Heads, Kv, Head]>()
            .entry(entry)
            .by_byte_offsets(pages)
            .prefix::<m![Keys], _>(length, 0)
            .unchecked()
    }
    .to_dm_view(&mut context.tdma, raw.view_mut());
}

fn tokens(
    raw: &Raw,
    kv: usize,
) -> DmTensorView<'_, f8e4m3, Chip, Cluster1, Ring, m![Keys / 128, Kv = 1 # 2, Head]> {
    raw.view()
        .tile::<m![Kv], 1, m![Keys / 128, Kv = 1 # 2, Head]>(kv)
}

fn keys(context: &mut Device, raw: &Raw) -> Decoded {
    context
        .main
        .begin(tokens(raw, 0))
        .fetch::<m![Keys / 128, Head / 8], m![Head % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Keys / 128, Head / 8], m![Head % 8]>()
        .cast::<bf16, m![Head % 8 # 16]>()
        .commit_trim::<m![Head % 8]>()
        .commit()
}

fn values(context: &mut Device, raw: &Raw) -> Decoded {
    context
        .main
        .begin(tokens(raw, 1))
        .fetch::<m![Keys / 128, Head / 8], m![Head % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Keys / 128, Head / 8], m![Head % 8]>()
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
        .commit()
}

fn scores(context: &mut Device, key: &Decoded, query: &Lanes) -> Scores {
    context
        .main
        .begin(key.view())
        .fetch::<m![Keys / 128, Head / 16], m![Head % 16]>()
        .collect::<m![Keys / 128, Head / 16], m![Head % 16]>()
        .contract_outer::<m![Keys / 128, Head / 32], m![Head % 32], _, _, _>(query)
        .contract_packet::<m![1]>()
        .contract_time::<m![Keys / 128]>()
        .contract_lane::<m![Keys / 128], m![Groups # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_scale(block::SCALE)
        .vector_final()
        .transpose::<m![Keys / 256, Groups], m![Keys / 128 % 2 # 8]>()
        .commit_trim::<m![Keys / 128 % 2]>()
        .commit()
}

/// Scaled scores, absent keys at the lowest bf16.
fn mask(context: &mut Device, scores: &Scores, beyond: &Beyond) -> Weights {
    let tags: VrfTensor<i4, Chip, Cluster1, Ring, m![Groups, Keys / 128]> = context
        .main
        .begin(beyond.view())
        .fetch::<m![Groups, Keys / 128 / 8], m![Keys / 128 % 8]>()
        .collect::<m![Groups, Keys / 128 / 8], m![Keys / 128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Comparison([
            Cmp::Greater(-0.5f32),
            Cmp::False,
            Cmp::False,
            Cmp::False,
        ]))
        .vector_log(&mut context.sub);
    context
        .main
        .begin(scores.view())
        .fetch::<m![Groups, Keys / 128 / 8], m![Keys / 128 % 8]>()
        .collect::<m![Groups, Keys / 128 / 8], m![Keys / 128 % 8]>()
        .vector_init()
        .vector_tagged(&tags)
        .vector_logic_with_mode(
            LogicBinaryOpF32::BitAnd,
            BinaryArgMode::Mode11,
            Branched::imm(block::ABSENT, f32::from_bits(0xff7f_0000)),
        )
        .vector_final()
        .cast::<bf16, m![Keys / 128 % 8 # 16]>()
        .commit_trim::<m![Keys / 128 % 8]>()
        .commit()
}

fn local(context: &mut Device, scores: &Weights, peak: bool) -> Local {
    let op = if peak {
        IntraSliceReduceOpF32::Max
    } else {
        IntraSliceReduceOpF32::Add
    };
    context
        .main
        .begin(scores.view())
        .fetch::<m![Groups, Keys / 128 / 8], m![Keys / 128 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Groups, Keys / 128 / 8], m![Keys / 128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Groups, Keys / 128 / 4], m![Keys / 128 % 4]>()
        .vector_intra_slice_reduce::<Keys, m![Groups], m![1 # 4]>(op)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

/// Each head's stat over its 128 slices' stats, into the register file: the peak at least the
/// lowest finite score, the mass at least 1e-20.
fn row(context: &mut Device, local: &Local, peak: bool) -> Operand {
    let (inter, clip) = if peak {
        (InterSliceReduceOpF32::Max, (ClipBinaryOpF32::Max, -1e30f32))
    } else {
        (InterSliceReduceOpF32::Add, (ClipBinaryOpF32::Add, 1e-20f32))
    };
    let stat: Local = context
        .main
        .begin(local.view())
        .fetch::<m![Groups], m![1 # 8]>()
        .collect::<m![Groups], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce_span(inter, 7)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(clip.0, clip.1)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    context
        .sub
        .begin(stat.view())
        .fetch::<m![Groups], m![1 # 8]>()
        .collect::<m![Groups], m![1 # 8]>()
        .to_vrf()
}

fn weigh(context: &mut Device, scores: &Weights, top: &Operand) -> Weights {
    context
        .main
        .begin(scores.view())
        .fetch::<m![Groups, Keys / 128 / 8], m![Keys / 128 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Groups, Keys / 128 / 8], m![Keys / 128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Groups, Keys / 128 / 4], m![Keys / 128 % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, top)
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_widen_concat::<m![Groups, Keys / 128 / 8], m![Keys / 128 % 8]>()
        .vector_final()
        .cast::<bf16, m![Keys / 128 % 8 # 16]>()
        .commit_trim::<m![Keys / 128 % 8]>()
        .commit()
}

fn fill(context: &mut Device, weights: &Weights) -> Probability {
    context
        .sub
        .begin(weights.view())
        .fetch::<m![Groups], m![Keys / 128]>()
        .collect::<m![Groups], m![Keys / 128]>()
        .to_trf()
}

fn weighted(context: &mut Device, value: &Decoded, weights: &Probability) -> Partial {
    context
        .main
        .begin(value.view())
        .fetch::<m![Head / 32, Keys / 128], m![Head % 32]>()
        .collect::<m![Head / 32, Keys / 128, Head % 32 / 16], m![Head % 16]>()
        .contract_outer::<m![Head / 32, Keys / 128], m![Head % 32], _, _, _>(weights)
        .contract_packet::<m![Head % 32]>()
        .contract_time::<m![Head / 32]>()
        .contract_lane::<m![Head / 32, Groups, Head % 32 / 8], m![Head % 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_inter_slice_reduce_span(InterSliceReduceOpF32::Add, 7)
        .vector_final()
        .commit_trim::<m![Head % 8]>()
        .commit()
}

fn finish(context: &mut Device, (masses, weighted): (&Local, &Partial), output: &mut Restored) {
    let total = row(context, masses, false);
    let rows: Partial = context
        .main
        .begin(weighted.view())
        .fetch::<m![Groups, Head / 8], m![Head % 8]>()
        .collect::<m![Groups, Head / 8], m![Head % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Groups, Head / 4], m![Head % 4]>()
        .vector_fp_div(&total)
        .vector_widen_concat::<m![Groups, Head / 8], m![Head % 8]>()
        .vector_final()
        .commit_trim::<m![Head % 8]>()
        .commit();
    // SAFETY: a row's width is its heads, groups and head channels in turn.
    let attended = unsafe {
        output
            .view_mut()
            .reshape::<Chip, m![WaveRows, Heads, Groups, Head]>()
    };
    rows.view()
        .slice_tile::<m![Keys % 128 / 16], 1, m![Heads % 2, Keys % 128 / 16 = 1 # 8, Page]>(0)
        .slice_tile::<m![Page], 1, m![Heads % 2, Keys % 128 / 16 = 1 # 8, Page = 1 # 16]>(0)
        .to_hbm_view(
            &mut context.tdma,
            attended.tile::<m![WaveRows], 1, m![WaveRows = 1 #{!} 64, Heads, Groups, Head]>(0),
        );
}

/// The first eight rows of `output`, by head of the pair and row.
type Zeros = DmTensor<f32, Chip, Cluster1, m![Heads % 2, WaveRows % 8, 1 # 16], m![Groups, Head]>;

/// The first eight rows of `output` zero, for the live rows to store theirs over: the step's tail
/// reads a batch's rows in groups of eight, and an unwritten row of the group is any bytes.
pub(super) fn clear(context: &mut Device, output: &mut Restored) {
    let mut zero = Zeros::new();
    zero.view_mut().memset(0.0f32, &mut context.sub);
    // SAFETY: a row's width is its heads, groups and head channels in turn.
    let rows = unsafe {
        output
            .view_mut()
            .reshape::<Chip, m![WaveRows, Heads, Groups, Head]>()
    };
    zero.view().to_hbm_view(
        &mut context.tdma,
        rows.tile::<m![WaveRows / 8], 1, m![WaveRows / 8 = 1 #{!} 8, WaveRows % 8, Heads, Groups, Head]>(0),
    );
}

/// Capacity row 0's attention into `output`'s row 0, its next seven rows zero, the mask's counts
/// and then `under` issued behind the page gather.
pub fn attend(
    context: &mut Device,
    history: &History<'_>,
    queries: &qkv::Queries,
    output: &mut Restored,
    under: impl FnOnce(&mut Device),
) {
    let mut staged = Query::new();
    query(context, queries, &mut staged);
    let pages = tables(history);
    let mut raw = Raw::new();
    gather(context, (history.pool, history.entry), &pages, &mut raw);
    let mut beyond = Beyond::new();
    stage(context, history.plan, &mut beyond);
    clear(context, output);
    under(context);
    let lanes = lanes(context, &staged);
    let key = keys(context, &raw);
    let scores = scores(context, &key, &lanes);
    drop(key);
    let scores = mask(context, &scores, &beyond);
    let peak = local(context, &scores, true);
    let top = row(context, &peak, true);
    let weights = weigh(context, &scores, &top);
    let masses = local(context, &weights, false);
    let probabilities = fill(context, &weights);
    let value = values(context, &raw);
    drop(raw);
    let weighted = weighted(context, &value, &probabilities);
    finish(context, (&masses, &weighted), output);
}
