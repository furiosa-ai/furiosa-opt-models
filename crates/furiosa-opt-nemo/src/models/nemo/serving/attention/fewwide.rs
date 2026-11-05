//! A batch of at most 32 whole rows in waves of 16 (the [`rowwise`] mapping), in parts of half a
//! wave: part `2 w + h` holds wave `w`'s rows of slot times `4 h .. 4 h + 4`, its pages land in a
//! gather of its own, and its whole attention runs on Main while the next part's pages land. An
//! odd last part is packed: rows `8 (2 w) + 2 q + c`, one a PE in slot times `0 .. 4`, so a wave
//! of eight rows keeps every PE's chain as short as a half's.
use super::*;

axes![Half = 2, Wave = 4, Octet = 8];

type HalfPages = HbmTensor<i32, Chip, m![Jobs / 8, Jobs % 4, Tokens / 16]>;
type HalfLengths = HbmTensor<i32, Chip, m![Jobs / 8, Jobs % 4]>;
/// A part's pages, by slot time `Jobs % 4`.
type Raw = DmTensor<f8e4m3, Chip, Cluster, Slices, m![Jobs % 4, Tokens / 16, Kv, Head]>;
type Pages<'l> =
    DmTensorView<'l, f8e4m3, Chip, Cluster, Slices, m![Jobs % 4, Tokens / 16, Kv, Head]>;
/// The first half of a record page buffer, slot times `0 .. 4`, as a part's pages.
type Padded = m![Jobs % 4 # 8, Tokens / 16, Kv, Head];
type Decoded<'l> = DmTensorView<'l, bf16, Chip, Cluster, Slices, m![Jobs % 4, Tokens / 16, Head]>;
/// A part's decoded keys or values in the first half of the record's rounded outputs.
type Widened = m![Jobs % 4 # 8, Tokens / 16, Head];
type Query = DmTensor<
    bf16,
    Chip,
    Cluster,
    m![Jobs % 32 / 8, Heads % 2, Heads / 2, 1 # 16],
    m![Groups, Head],
>;
type Lanes = TrfTensor<bf16, Chip, Cluster, Slices, m![Groups], m![Head]>;
type Scores = DmTensor<f32, Chip, Cluster, Slices, m![Jobs % 4, Groups, Tokens / 16]>;
type Weights = DmTensor<bf16, Chip, Cluster, Slices, m![Jobs % 4, Groups, Tokens / 16]>;
type Probability = TrfTensor<bf16, Chip, Cluster, Slices, m![Groups], m![Jobs % 4, Tokens / 16]>;
type Local = DmTensor<f32, Chip, Cluster, Slices, m![Jobs % 4, Groups, 1 # 8]>;
type Scalar = VrfTensor<f32, Chip, Cluster, Slices, m![Jobs % 4, Groups, 1 # 8]>;
type RowOperand = VrfTensor<f32, Chip, Cluster, Slices, m![Groups, 1 # 8]>;
type Partial = DmTensor<f32, Chip, Cluster, Slices, m![Jobs % 4, Groups, Head]>;
type Attended = DmTensor<f32, Chip, Cluster, Slices, m![Groups, Head]>;
type HalfCounts<'l> = DmTensorView<
    'l,
    f32,
    Chip,
    Cluster,
    Slices,
    m![Parity = 1 # 2, Burst = 1 # 8, Half = 1 # 2, Jobs % 4, 1 # 8],
>;

/// Part `part`'s page offsets and lengths: the host lays them out at the start of tables
/// `1 + part`, PE-major.
fn tables(history: &History<'_>, part: usize) -> (HalfPages, HalfLengths) {
    // SAFETY: the first 32 jobs of tables `1 + part` are that part's `[p][j]`; nothing writes them.
    unsafe {
        (
            history.plan.pages[1 + part]
                .view()
                .reshape::<Chip, m![Jobs / 8 # 16, Jobs % 4, Tokens / 16]>()
                .alias::<Chip, m![Jobs / 8, Jobs % 4, Tokens / 16]>(),
            history.few[part]
                .view()
                .reshape::<Chip, m![Jobs / 8 # 16, Jobs % 4]>()
                .alias::<Chip, m![Jobs / 8, Jobs % 4]>(),
        )
    }
}

/// Part `$part`'s pages into the data-memory view `$raw`.
macro_rules! gather {
    ($context:expr, ($pool:expr, $entry:expr), $history:expr, $part:expr, $raw:expr) => {{
        let (pages, lengths) = tables($history, $part);
        // SAFETY: as `attention::gather`.
        unsafe {
            $pool
                .gather::<m![1], m![Page, Heads, Kv, Head]>()
                .entry($entry)
                .by_byte_offsets(&pages)
                .prefix::<m![Tokens], _>(&lengths, 0)
                .unchecked()
        }
        .to_dm_view(&mut $context.tdma, $raw)
    }};
}

/// Part `part`'s pages into the first half of the record page buffer `odd`.
fn land(
    context: &mut Device,
    (pool, entry): (&Pool, usize),
    (history, part): (&History<'_>, usize),
    odd: &mut super::Raw,
) {
    // SAFETY: slot times `0 .. 4` of `odd` are its first half in every slice.
    let raw = unsafe { odd.view_mut().reshape::<Chip, Cluster, Slices, Padded>() };
    gather!(context, (pool, entry), history, part, raw);
}

/// Part `part`'s pages into `raw`.
fn gather(
    context: &mut Device,
    (pool, entry): (&Pool, usize),
    (history, part): (&History<'_>, usize),
    raw: &mut Raw,
) {
    gather!(context, (pool, entry), history, part, raw.view_mut());
}

/// The pages `land` put in `odd`.
fn first_half(odd: &super::Raw) -> Pages<'_> {
    // SAFETY: as `land`.
    unsafe { odd.view().reshape::<Chip, Cluster, Slices, Padded>() }.unpad::<m![
        Jobs % 4,
        Tokens / 16,
        Kv,
        Head
    ]>()
}

/// Part `part`'s rows' query heads, one slice of each head ring.
fn stage(
    context: &mut Device,
    queries: &qkv::Queries,
    (part, packed): (usize, bool),
    query: &mut Query,
) {
    let (wave, half) = (part / 2, part % 2);
    // SAFETY: a capacity row's query heads are its `Heads × Groups` heads in head order, and row
    // `16 w + 4 q + 2 c + h` is half `h` of cluster `c`'s PE `q` in wave `w`.
    let rows = unsafe {
        queries
            .view()
            .reshape::<Chip, m![Capacity, Heads, Groups, Head]>()
    };
    if packed {
        // SAFETY: packed row `8 (2 w) + 2 q + c` is cluster `c`'s PE `q`.
        unsafe {
            rows.tile::<m![Capacity / 8], 1, m![Capacity / 8 = 1 # 512, Capacity % 8, Heads, Groups, Head]>(2 * wave)
                .reshape::<Chip, m![Jobs % 32 / 8 = 1 # 512, Jobs % 32 / 8, Jobs / 32, Heads, Groups, Head]>()
        }
        .to_dm_view(&mut context.tdma, query.view_mut());
        return;
    }
    let rows = unsafe {
        rows.tile::<m![Capacity / 16], 1, m![Capacity / 16 = 1 # 256, Capacity % 16, Heads, Groups, Head]>(wave)
            .reshape::<Chip, m![Jobs % 32 / 8 = 1 # 256, Jobs % 32 / 8, Jobs / 32, Half, Heads, Groups, Head]>()
    };
    rows.tile::<m![Half], 1, m![Jobs % 32 / 8 = 1 # 256, Jobs % 32 / 8, Jobs / 32, Half = 1 # 2, Heads, Groups, Head]>(half)
        .to_dm_view(&mut context.tdma, query.view_mut());
}

fn lanes(context: &mut Device, query: &Query) -> Lanes {
    context
        .sub
        .begin(query.view())
        .fetch::<m![Groups], m![Head]>()
        .switch::<Slices, m![Groups]>(SwitchConfig::CustomBroadcast { ring_size: 16 })
        .collect::<m![Groups, Head / 16], m![Head % 16]>()
        .to_trf()
}

fn tokens<'l>(
    raw: &Pages<'l>,
    kv: usize,
) -> DmTensorView<'l, f8e4m3, Chip, Cluster, Slices, m![Jobs % 4, Tokens / 16, Kv = 1 # 2, Head]> {
    raw.tile::<m![Kv], 1, m![Jobs % 4, Tokens / 16, Kv = 1 # 2, Head]>(kv)
}

fn widened(
    rounded: &mut super::Rounded,
) -> DmTensorViewMut<'_, bf16, Chip, Cluster, Slices, Widened> {
    // SAFETY: a part's decoded tokens fill the first half of each slice of the rounded outputs,
    // which a few-row batch leaves to its chain.
    unsafe {
        rounded
            .view_mut()
            .reshape::<Chip, Cluster, Slices, Widened>()
    }
}

fn decoded(rounded: &super::Rounded) -> Decoded<'_> {
    // SAFETY: as `widened`.
    unsafe { rounded.view().reshape::<Chip, Cluster, Slices, Widened>() }.unpad::<m![
        Jobs % 4,
        Tokens / 16,
        Head
    ]>()
}

fn keys(context: &mut Device, raw: &Pages<'_>, rounded: &mut super::Rounded) {
    context
        .main
        .begin(tokens(raw, 0))
        .fetch::<m![Jobs % 4, Tokens / 16, Head / 8], m![Head % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Jobs % 4, Tokens / 16, Head / 8], m![Head % 8]>()
        .cast::<bf16, m![Head % 8 # 16]>()
        .commit_trim::<m![Head % 8]>()
        .commit_view(widened(rounded));
}

fn values(context: &mut Device, raw: &Pages<'_>, rounded: &mut super::Rounded) {
    context
        .main
        .begin(tokens(raw, 1))
        .fetch::<m![Jobs % 4, Tokens / 16, Head / 8], m![Head % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Jobs % 4, Tokens / 16, Head / 8], m![Head % 8]>()
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
        .commit_view(widened(rounded));
}

fn scores(context: &mut Device, key: Decoded<'_>, query: &Lanes) -> Scores {
    context
        .main
        .begin(key)
        .fetch::<m![Jobs % 4, Tokens / 16, Head / 16], m![Head % 16]>()
        .collect::<m![Jobs % 4, Tokens / 16, Head / 16], m![Head % 16]>()
        .contract_outer::<m![Jobs % 4, Tokens / 16, Head / 32], m![Head % 32], _, _, _>(query)
        .contract_packet::<m![1]>()
        .contract_time::<m![Jobs % 4, Tokens / 16]>()
        .contract_lane::<m![Jobs % 4, Tokens / 16], m![Groups # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_scale(block::SCALE)
        .vector_final()
        .transpose::<m![Jobs % 4, Tokens / 32, Groups], m![Tokens / 16 % 2 # 8]>()
        .commit_trim::<m![Tokens / 16 % 2]>()
        .commit()
}

fn mask(
    context: &mut Device,
    scores: &Scores,
    (positions, counts): (&Positions, HalfCounts<'_>),
) -> Weights {
    let count: Scalar = context
        .sub
        .begin(counts)
        .fetch::<m![Jobs % 4, Groups], m![1 # 8]>()
        .collect::<m![Jobs % 4, Groups], m![1 # 8]>()
        .to_vrf();
    let beyond: Scores = context
        .main
        .begin(positions.view())
        .fetch::<m![Jobs % 4, Groups, Tokens / 128], m![Tokens / 16 % 8]>()
        .collect::<m![Jobs % 4, Groups, Tokens / 128], m![Tokens / 16 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Jobs % 4, Groups, Tokens / 64], m![Tokens / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, &count)
        .vector_widen_concat::<m![Jobs % 4, Groups, Tokens / 128], m![Tokens / 16 % 8]>()
        .vector_final()
        .commit_trim::<m![Tokens / 16 % 8]>()
        .commit();
    let tags: VrfTensor<i4, Chip, Cluster, Slices, m![Jobs % 4, Groups, Tokens / 16]> = context
        .main
        .begin(beyond.view())
        .fetch::<m![Jobs % 4, Groups, Tokens / 128], m![Tokens / 16 % 8]>()
        .collect::<m![Jobs % 4, Groups, Tokens / 128], m![Tokens / 16 % 8]>()
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
        .fetch::<m![Jobs % 4, Groups, Tokens / 128], m![Tokens / 16 % 8]>()
        .collect::<m![Jobs % 4, Groups, Tokens / 128], m![Tokens / 16 % 8]>()
        .vector_init()
        .vector_tagged(&tags)
        .vector_logic_with_mode(
            LogicBinaryOpF32::BitAnd,
            BinaryArgMode::Mode11,
            Branched::imm(block::ABSENT, f32::from_bits(0xff7f_0000)),
        )
        .vector_final()
        .cast::<bf16, m![Tokens / 16 % 8 # 16]>()
        .commit_trim::<m![Tokens / 16 % 8]>()
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
        .fetch::<m![Jobs % 4, Groups, Tokens / 128], m![Tokens / 16 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Jobs % 4, Groups, Tokens / 128], m![Tokens / 16 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Jobs % 4, Groups, Tokens / 64], m![Tokens / 16 % 4]>()
        .vector_intra_slice_reduce::<Tokens, m![Jobs % 4, Groups], m![1 # 4]>(op)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

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
    let stat: DmTensor<f32, Chip, Cluster, Slices, m![Groups, 1 # 8]> = context
        .main
        .begin(local.view())
        .fetch::<m![Groups, Jobs % 4], m![1 # 8]>()
        .collect::<m![Groups, Jobs % 4], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce_span(inter, 4)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<Jobs, m![Groups], m![1 # 4]>(intra)
        .vector_widen_pad::<m![1 # 8]>()
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

fn weigh(context: &mut Device, scores: &Weights, top: &RowOperand) -> Weights {
    context
        .main
        .begin(scores.view())
        .fetch::<m![Groups, Jobs % 4, Tokens / 128], m![Tokens / 16 % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Groups, Jobs % 4, Tokens / 128], m![Tokens / 16 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Groups, Jobs % 4, Tokens / 64], m![Tokens / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, top)
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_widen_concat::<m![Groups, Jobs % 4, Tokens / 128], m![Tokens / 16 % 8]>()
        .vector_final()
        .cast::<bf16, m![Tokens / 16 % 8 # 16]>()
        .commit_trim::<m![Tokens / 16 % 8]>()
        .commit()
}

fn fill(context: &mut Device, weights: &Weights) -> Probability {
    context
        .sub
        .begin(weights.view())
        .fetch::<m![Groups, Jobs % 4, Tokens / 256], m![Tokens / 16 % 16]>()
        .collect::<m![Groups, Jobs % 4, Tokens / 256], m![Tokens / 16 % 16]>()
        .to_trf()
}

fn weighted(context: &mut Device, value: Decoded<'_>, weights: &Probability) -> Partial {
    context
        .main
        .begin(value)
        .fetch::<m![Jobs % 4, Head / 32, Tokens / 16], m![Head % 32]>()
        .collect::<m![Jobs % 4, Head / 32, Tokens / 16, Head % 32 / 16], m![Head % 16]>()
        .contract_outer::<m![Jobs % 4, Head / 32, Tokens / 16], m![Head % 32], _, _, _>(weights)
        .contract_packet::<m![Head % 32]>()
        .contract_time::<m![Jobs % 4, Head / 32]>()
        .contract_lane::<m![Jobs % 4, Head / 32, Groups, Head % 32 / 8], m![Head % 8]>(
            LaneMode::Sequential,
        )
        .vector_init()
        .vector_inter_slice_reduce_span(InterSliceReduceOpF32::Add, 4)
        .vector_final()
        .commit_trim::<m![Head % 8]>()
        .commit()
}

fn finish(context: &mut Device, (masses, weighted): (&Local, &Partial)) -> Attended {
    let total = row(context, masses, false);
    context
        .main
        .begin(weighted.view())
        .fetch::<m![Groups, Head / 8, Jobs % 4], m![Head % 8]>()
        .collect::<m![Groups, Head / 8, Jobs % 4], m![Head % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Groups, Head / 8, Jobs % 4, Head % 8 / 4], m![Head % 4]>()
        .vector_intra_slice_reduce::<Jobs, m![Groups, Head / 4], m![Head % 4]>(
            IntraSliceReduceOpF32::Add,
        )
        .vector_fp_div(&total)
        .vector_widen_concat::<m![Groups, Head / 8], m![Head % 8]>()
        .vector_final()
        .commit_trim::<m![Head % 8]>()
        .commit()
}

fn store(
    context: &mut Device,
    rows: &Attended,
    (part, packed): (usize, bool),
    output: &mut Restored,
) {
    let (wave, half) = (part / 2, part % 2);
    if packed {
        // SAFETY: packed row `8 (2 w) + 2 q + c` is cluster `c`'s PE `q`, and a row's width is its
        // heads, groups and head channels in turn.
        let attended = unsafe {
            output
                .view_mut()
                .reshape::<Chip, m![Octet, Jobs % 32 / 8, Jobs / 32, Heads, Groups, Head]>()
        };
        rows.view().slice_tile::<m![Page], 1, Stored>(0).to_hbm_view(
            &mut context.tdma,
            attended
                .tile::<m![Octet], 1, m![Octet = 1 #{!} 8, Jobs % 32 / 8, Jobs / 32, Heads, Groups, Head]>(2 * wave),
        );
        return;
    }
    // SAFETY: the row in half `h` of cluster `c`'s PE `q` is row `4 q + 2 c + h` of its wave, and a row's
    // width is its heads, groups and head channels in turn.
    let attended = unsafe {
        output
            .view_mut()
            .reshape::<Chip, m![Wave, Jobs % 32 / 8, Jobs / 32, Half, Heads, Groups, Head]>()
    };
    rows.view().slice_tile::<m![Page], 1, Stored>(0).to_hbm_view(
        &mut context.tdma,
        attended
            .tile::<m![Wave], 1, m![Wave = 1 #{!} 4, Jobs % 32 / 8, Jobs / 32, Half, Heads, Groups, Head]>(wave)
            .tile::<m![Half], 1, m![Wave = 1 #{!} 4, Jobs % 32 / 8, Jobs / 32, Half = 1 #{!} 2, Heads, Groups, Head]>(half),
    );
}

/// One part's attended rows from its landed pages, ready to store.
fn attend_half(
    context: &mut Device,
    (raw, lanes, positions, landed): (Pages<'_>, &Lanes, &Positions, &Landed),
    rounded: &mut super::Rounded,
    (part, packed): (usize, bool),
) -> Attended {
    let (wave, half) = (part / 2, part % 2);
    // SAFETY: the four head rings of a PE hold the same counts, `Lane` lays out `1 # 8`, and slot
    // time `Jobs % 8` is `4 * Half + Jobs % 4`.
    let counts = unsafe {
        landed
            .view()
            .reshape::<Chip, Cluster, Slices, m![Parity, Burst, Half, Jobs % 4, 1 # 8]>()
    }
    .tile::<m![Parity], 1, m![Parity = 1 # 2, Burst, Half, Jobs % 4, 1 # 8]>(0)
    .tile::<m![Burst], 1, m![Parity = 1 # 2, Burst = 1 # 8, Half, Jobs % 4, 1 # 8]>(wave)
    .tile::<m![Half], 1, m![Parity = 1 # 2, Burst = 1 # 8, Half = 1 # 2, Jobs % 4, 1 # 8]>(
        if packed { 0 } else { half },
    );
    keys(context, &raw, rounded);
    let scores = scores(context, decoded(rounded), lanes);
    let scores = mask(context, &scores, (positions, counts));
    let peak = local(context, &scores, true);
    let top = row(context, &peak, true);
    let weights = weigh(context, &scores, &top);
    let masses = local(context, &weights, false);
    let probabilities = fill(context, &weights);
    values(context, &raw, rounded);
    let weighted = weighted(context, decoded(rounded), &probabilities);
    finish(context, (&masses, &weighted))
}

/// The first batch's rows in `parts` parts of half a wave (`2 ..= 4`, at most `8 parts` whole
/// rows), into `output`'s rows `0 .. 8 parts`, `under` issued behind the last page gather. Parts
/// 0 and 2 land in the first half of the record page buffer `odd`, parts 1 and 3 in a buffer of
/// their own, so a part's gather lands beside the chain it runs under; three parts pack part 2.
///
/// - The first two parts' queries land ahead of the first gather, and the counts and later parts'
///   tables right behind it: a chain reads those only once its pages are in.
/// - Part `p + 1`'s gather issues ahead of part `p`'s chain and part `p + 2`'s right after it, as
///   a gather's issue waits for its length load behind the previous gather.
/// - Part `p`'s store issues after part `p + 1`'s chain, ahead of part `p + 3`'s gather and of
///   the last chain: a chain whose buffers reuse a stored part's waits for that store, so no
///   store queues behind the gather that chain also waits for, and only the last store follows
///   the last chain.
/// - One body serves every part count, so the record buffers have one writer.
pub fn attend(
    context: &mut Device,
    (history, queries): (&History<'_>, &qkv::Queries),
    (odd, rounded): (&mut super::Raw, &mut super::Rounded),
    (output, parts): (&mut Restored, usize),
    under: &mut impl FnMut(&mut Device),
) {
    let (pool, plan, positions) = (
        (history.pool, history.entry),
        history.plan,
        history.positions,
    );
    let (third, fourth) = (parts / 3, parts / 4);
    let packed = third * (1 - fourth);
    let mut landed = Landed::new();
    let mut counts = |context: &mut Device, wave: usize| {
        plan.counts[0]
            .view()
            .tile::<m![Burst], 1, m![Page, Burst = 1 # 8, Jobs, Lane]>(wave)
            .to_dm_view(
                &mut context.tdma,
                landed
                    .view_mut()
                    .tile::<m![Parity], 1, m![Parity = 1 #{!} 2, Burst, Jobs % 8, Lane]>(0)
                    .tile::<m![Burst], 1, m![Parity = 1 #{!} 2, Burst = 1 #{!} 8, Jobs % 8, Lane]>(
                        wave,
                    ),
            );
    };
    let mut query = (Query::new(), Query::new(), Query::new(), Query::new());
    stage(context, queries, (0, false), &mut query.0);
    stage(context, queries, (1, false), &mut query.1);
    let mut spare = Raw::new();
    land(context, pool, (history, 0), odd);
    counts(context, 0);
    for _ in 0..third {
        counts(context, 1);
    }
    for _ in 0..packed {
        stage(context, queries, (2, true), &mut query.2);
    }
    for _ in 0..fourth {
        stage(context, queries, (2, false), &mut query.2);
        stage(context, queries, (3, false), &mut query.3);
    }
    let (first, second) = (lanes(context, &query.0), lanes(context, &query.1));
    gather(context, pool, (history, 1), &mut spare);
    for _ in 0..1 - third {
        under(context);
    }
    // Part `p + 2`'s lanes fill the TRF after part `p`'s chain frees its own: the TRF holds two
    // parts' lanes and one part's probabilities.
    let rows = attend_half(
        context,
        (first_half(odd), &first, positions, &landed),
        rounded,
        (0, false),
    );
    for _ in 0..third {
        land(context, pool, (history, 2), odd);
        for _ in 0..1 - fourth {
            under(context);
        }
    }
    let later = attend_half(
        context,
        (spare.view(), &second, positions, &landed),
        rounded,
        (1, false),
    );
    store(context, &rows, (0, false), output);
    for _ in 0..fourth {
        gather(context, pool, (history, 3), &mut spare);
        under(context);
    }
    for _ in 0..third {
        let lanes = lanes(context, &query.2);
        let rows = attend_half(
            context,
            (first_half(odd), &lanes, positions, &landed),
            rounded,
            (2, false),
        );
        store(context, &later, (1, false), output);
        for _ in 0..packed {
            store(context, &rows, (2, true), output);
        }
        for _ in 0..fourth {
            store(context, &rows, (2, false), output);
        }
    }
    for _ in 0..1 - third {
        store(context, &later, (1, false), output);
    }
    for _ in 0..fourth {
        let lanes = lanes(context, &query.3);
        let rows = attend_half(
            context,
            (spare.view(), &lanes, positions, &landed),
            rounded,
            (3, false),
        );
        store(context, &rows, (3, false), output);
    }
}
