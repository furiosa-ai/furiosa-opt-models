//! Up to four decode rows' attention, each row's keys spread over 32 slices of the head pair's
//! cluster: PE `r` holds row `r`, a slice one head, one page token and every other page, so a
//! page lands on one PE of each cluster, and
//! each Main command works on 64 keys a slice and the softmax reduces across 32 slices.
//!
//! - Cluster `c` holds heads `2 c` and `2 c + 1`; page `p` of row `r` lands on page group `p % 2`
//!   of the row's slices, at time `p / 2`.
//! - The host gives row `r`'s page offsets at `r * 128` of page table 1, its length at `r` of
//!   length table 1 (0 past the live rows), and each slice's page times less its count of present
//!   pages, `[r][g][t][k]`, at the start of the first burst's counts.
use super::*;
use crate::models::nemo::serving::Rows as WaveRows;

axes![Keys = 2048, Four = 4];

type Cluster1 = m![Heads / 2];
/// A cluster's slices: a head of the pair, a page group, a page token.
type Ring = m![Four, Heads % 2, Keys % 32 / 16, Page];
type RowPages = HbmTensor<i32, Chip, m![Four, Keys / 16]>;
type RowLength = HbmTensor<i32, Chip, m![Four]>;
type Raw = DmTensor<f8e4m3, Chip, Cluster1, Ring, m![Keys / 32, Kv, Head]>;
type Decoded = DmTensor<bf16, Chip, Cluster1, Ring, m![Keys / 32, Head]>;
type Query = DmTensor<bf16, Chip, Cluster1, m![Four, Heads % 2, 1 # 32], m![Groups, Head]>;
type Lanes = TrfTensor<bf16, Chip, Cluster1, Ring, m![Groups], m![Head]>;
type Scores = DmTensor<f32, Chip, Cluster1, Ring, m![Groups, Keys / 32]>;
type Weights = DmTensor<bf16, Chip, Cluster1, Ring, m![Groups, Keys / 32]>;
type Probability = TrfTensor<bf16, Chip, Cluster1, Ring, m![Groups], m![Keys / 32]>;
type Local = DmTensor<f32, Chip, Cluster1, Ring, m![Groups, 1 # 8]>;
type Operand = VrfTensor<f32, Chip, Cluster1, Ring, m![Groups, 1 # 8]>;
type Beyond = DmTensor<f32, Chip, Cluster1, Ring, m![Keys / 32]>;
type Partial = DmTensor<f32, Chip, Cluster1, Ring, m![Groups, Head]>;

/// The row's page offsets and length.
fn tables(history: &History<'_>) -> (RowPages, RowLength) {
    // SAFETY: the host's page table 1 starts with the row's pages, length table 1 with its length.
    unsafe {
        (
            history.plan.pages[1]
                .view()
                .reshape::<Chip, m![Four # 16, Keys / 16]>()
                .alias::<Chip, m![Four, Keys / 16]>(),
            history.few[0]
                .view()
                .reshape::<Chip, m![Four # 64]>()
                .alias::<Chip, m![Four]>(),
        )
    }
}

/// Each slice's page times past its count of present pages: a key is absent where it is >= 0.
fn stage(context: &mut Device, plan: &Plan, beyond: &mut Beyond) {
    // SAFETY: the host gives row `r`'s page group `g`'s token-`t` times `[r][g][t][k]` at the start
    // of the first burst's counts.
    let table = unsafe {
        plan.counts[0]
            .view()
            .reshape::<Chip, m![Four # 32, Keys % 32 / 16, Page, Keys / 32]>()
            .alias::<Chip, m![Four, Keys % 32 / 16, Page, Keys / 32]>()
    };
    table
        .view()
        .to_dm_view(&mut context.tdma, beyond.view_mut());
}

/// The rows' query heads, one slice of each row's head ring.
fn query(context: &mut Device, queries: &qkv::Queries, query: &mut Query) {
    // SAFETY: a capacity row's query heads are its `Heads × Groups` heads in head order.
    unsafe {
        queries
            .view()
            .reshape::<Chip, m![Capacity / 4, Four, Heads, Groups, Head]>()
    }
    .tile::<m![Capacity / 4], 1, m![Capacity / 4 = 1 # 1024, Four, Heads, Groups, Head]>(0)
    .to_dm_view(&mut context.tdma, query.view_mut());
}

fn lanes(context: &mut Device, query: &Query) -> Lanes {
    context
        .sub
        .begin(query.view())
        .fetch::<m![Groups], m![Head]>()
        .switch::<Ring, m![Groups]>(SwitchConfig::CustomBroadcast { ring_size: 32 })
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
) -> DmTensorView<'_, f8e4m3, Chip, Cluster1, Ring, m![Keys / 32, Kv = 1 # 2, Head]> {
    raw.view()
        .tile::<m![Kv], 1, m![Keys / 32, Kv = 1 # 2, Head]>(kv)
}

fn keys(context: &mut Device, raw: &Raw) -> Decoded {
    context
        .main
        .begin(tokens(raw, 0))
        .fetch::<m![Keys / 32, Head / 8], m![Head % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Keys / 32, Head / 8], m![Head % 8]>()
        .cast::<bf16, m![Head % 8 # 16]>()
        .commit_trim::<m![Head % 8]>()
        .commit()
}

fn values(context: &mut Device, raw: &Raw) -> Decoded {
    context
        .main
        .begin(tokens(raw, 1))
        .fetch::<m![Keys / 32, Head / 8], m![Head % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Keys / 32, Head / 8], m![Head % 8]>()
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
        .fetch::<m![Keys / 32, Head / 16], m![Head % 16]>()
        .collect::<m![Keys / 32, Head / 16], m![Head % 16]>()
        .contract_outer::<m![Keys / 32, Head / 32], m![Head % 32], _, _, _>(query)
        .contract_packet::<m![1]>()
        .contract_time::<m![Keys / 32]>()
        .contract_lane::<m![Keys / 32], m![Groups # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_scale(block::SCALE)
        .vector_final()
        .transpose::<m![Keys / 64, Groups], m![Keys / 32 % 2 # 8]>()
        .commit_trim::<m![Keys / 32 % 2]>()
        .commit()
}

/// Scaled scores, absent keys at the lowest bf16.
fn mask(context: &mut Device, scores: &Scores, beyond: &Beyond) -> Weights {
    let tags: VrfTensor<i4, Chip, Cluster1, Ring, m![Groups, Keys / 32]> = context
        .main
        .begin(beyond.view())
        .fetch::<m![Groups, Keys / 256], m![Keys / 32 % 8]>()
        .collect::<m![Groups, Keys / 256], m![Keys / 32 % 8]>()
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
        .fetch::<m![Groups, Keys / 256], m![Keys / 32 % 8]>()
        .collect::<m![Groups, Keys / 256], m![Keys / 32 % 8]>()
        .vector_init()
        .vector_tagged(&tags)
        .vector_logic_with_mode(
            LogicBinaryOpF32::BitAnd,
            BinaryArgMode::Mode11,
            Branched::imm(block::ABSENT, f32::from_bits(0xff7f_0000)),
        )
        .vector_final()
        .cast::<bf16, m![Keys / 32 % 8 # 16]>()
        .commit_trim::<m![Keys / 32 % 8]>()
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
        .fetch::<m![Groups, Keys / 256], m![Keys / 32 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Groups, Keys / 256], m![Keys / 32 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Groups, Keys / 32], m![Keys / 32 % 4]>()
        .vector_intra_slice_reduce::<Keys, m![Groups], m![1 # 4]>(op)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

/// Each row's stat of a head over its 32 slices' stats, into the register file: the peak at least the
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
        .vector_inter_slice_reduce_span(inter, 5)
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
        .fetch::<m![Groups, Keys / 256], m![Keys / 32 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Groups, Keys / 256], m![Keys / 32 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Groups, Keys / 32], m![Keys / 32 % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, top)
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_widen_concat::<m![Groups, Keys / 256], m![Keys / 32 % 8]>()
        .vector_final()
        .cast::<bf16, m![Keys / 32 % 8 # 16]>()
        .commit_trim::<m![Keys / 32 % 8]>()
        .commit()
}

fn fill(context: &mut Device, weights: &Weights) -> Probability {
    context
        .sub
        .begin(weights.view())
        .fetch::<m![Groups, Keys / 512], m![Keys / 32 % 16]>()
        .collect::<m![Groups, Keys / 512], m![Keys / 32 % 16]>()
        .to_trf()
}

fn weighted(context: &mut Device, value: &Decoded, weights: &Probability) -> Partial {
    context
        .main
        .begin(value.view())
        .fetch::<m![Head / 32, Keys / 32], m![Head % 32]>()
        .collect::<m![Head / 32, Keys / 32, Head % 32 / 16], m![Head % 16]>()
        .contract_outer::<m![Head / 32, Keys / 32], m![Head % 32], _, _, _>(weights)
        .contract_packet::<m![Head % 32]>()
        .contract_time::<m![Head / 32]>()
        .contract_lane::<m![Head / 32, Groups, Head % 32 / 8], m![Head % 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_inter_slice_reduce_span(InterSliceReduceOpF32::Add, 5)
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
            .reshape::<Chip, m![WaveRows / 4, Four, Heads, Groups, Head]>()
    };
    rows.view()
        .slice_tile::<m![Keys % 32 / 16], 1, m![Four, Heads % 2, Keys % 32 / 16 = 1 # 2, Page]>(0)
        .slice_tile::<m![Page], 1, m![Four, Heads % 2, Keys % 32 / 16 = 1 # 2, Page = 1 # 16]>(0)
        .to_hbm_view(
            &mut context.tdma,
            attended
                .tile::<m![WaveRows / 4], 1, m![WaveRows / 4 = 1 #{!} 16, Four, Heads, Groups, Head]>(0),
        );
}

/// Capacity rows 0..4's attention into `output`'s rows 0..4, its rows up to 8 zero, the mask's
/// counts and then `under` issued behind the page gather.
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
    super::fewone::clear(context, output);
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
