//! Attention of a prompt chunk whose rows see at most 2,048 keys each, in passes of 1,024 rows:
//! a pass's eight 128-row tiles each hold their own 2,048 keys on 32 slices, so a short history
//! fills the slices with rows instead of absent keys.
//!
//! - A cluster owns two KV heads, attended one after the other; tile `t` of a pass is the chunk's
//!   rows `128 * t..` of the pass, and every tile lands the chunk's keys.
//! - Softmax peaks are all-reduced over a tile's 32 slices; the value product and the mass are
//!   reduced onto the tile's query slices, where the product is divided by the mass.
use super::*;
use prompt::{Attended, Prompt};

axes![Keys = 2048, Queries = 1024];

type Cluster = m![Heads / 2];
type Slices = m![Queries / 128, Keys / 64];
type QuerySlices = m![Queries / 128, Queries % 128 / 16, Groups];
type RawElement = m![Keys % 64 / 16, Page, Heads % 2, Kv, Head];
/// A slice's 64 keys of its cluster's two heads, each head's key and value adjacent as the pool
/// stores them; one gather entry lands in every tile.
type Landed = DmTensor<f8e4m3, Chip, Cluster, m![8, Keys / 64], RawElement>;
type Key = DmTensor<bf16, Chip, Cluster, Slices, m![Keys % 64, Head]>;
type Transposed = DmTensor<f8e4m3, Chip, Cluster, Slices, m![Head, Keys % 64]>;
type Value = DmTensor<bf16, Chip, Cluster, Slices, m![Head, Keys % 64]>;
type KeyRegister = TrfTensor<
    bf16,
    Chip,
    Cluster,
    Slices,
    m![Keys % 8],
    m![Keys % 64 / 16, Head / 16, Keys % 16 / 8, Head % 16],
>;
type ValueRegister = TrfTensor<
    bf16,
    Chip,
    Cluster,
    Slices,
    m![Head % 8],
    m![Head / 16, Keys % 64 / 16, Head % 16 / 8, Keys % 16],
>;
type Query = DmTensor<bf16, Chip, Cluster, QuerySlices, m![Queries % 16, Head]>;
type Scores = DmTensor<bf16, Chip, Cluster, Slices, m![Groups, Queries % 128, Keys % 64]>;
type GroupScores<'l> =
    DmTensorView<'l, bf16, Chip, Cluster, Slices, m![Groups = 1 # 4, Queries % 128, Keys % 64]>;
/// How far each key of the slice lies past each row's last visible key: positive keys are absent.
type Excess = DmTensor<bf16, Chip, Cluster, Slices, m![Queries % 128, Keys % 64]>;
type Tags = VrfTensor<i4, Chip, Cluster, Slices, m![Queries % 128, Keys % 64]>;
type Masked<'l> = VectorTensor<
    'l,
    { Tu::Main },
    stage::Narrow,
    f32,
    Chip,
    Cluster,
    Slices,
    m![Queries % 128, Keys % 64 / 4],
    m![Keys % 4],
    Fresh,
    { stage::VeOrder::IntraFirst },
    stage::Standalone,
    { stage::Way::Way4 },
>;
type Peak = DmTensor<f32, Chip, Cluster, Slices, m![Groups, Queries % 128, 1 # 2]>;
type Top = VrfTensor<f32, Chip, Cluster, Slices, m![Queries % 128, 1 # 2]>;
type Bound = VrfTensor<f32, Chip, Cluster, Slices, m![Queries % 128, 1 # 8]>;
type Limit = DmTensor<f32, Chip, Cluster, Slices, m![Queries % 128, 1 # 8]>;
type Positions = DmTensor<f32, Chip, Cluster, Slices, m![Keys % 64]>;
type PositionRegister = VrfTensor<f32, Chip, Cluster, Slices, m![Keys % 64]>;
type Ones = DmTensor<bf16, Chip, Cluster, Slices, m![Keys % 64]>;
type OnesRegister = TrfTensor<bf16, Chip, Cluster, Slices, m![1], m![Keys % 64]>;
/// Every row's softmax mass on its query slice.
type Statistic = DmTensor<f32, Chip, Cluster, QuerySlices, m![Queries % 16, 1 # 8]>;
type Operand = VrfTensor<f32, Chip, Cluster, QuerySlices, m![Queries % 16, 1 # 8]>;
type Output = DmTensor<f32, Chip, Cluster, QuerySlices, m![Queries % 16, Head]>;

/// Byte offset of each of the chunk's 128 key pages within its layer's pool pages.
pub type Pages = HbmTensor<i32, Chip, m![Keys / 16]>;

/// The short path of one step's prompt chunk: the rows `start..start + rows` of the prompt plan.
#[derive(DeviceSend)]
pub struct Plan {
    /// The chunk's 1,024-row passes, 0 when the chunk attends span after span instead.
    pub passes: Count,
    /// The chunk's key pages, the same for every tile.
    pub pages: Pages,
    /// The last key of its tile's keys that chunk row `r` sees, in every lane.
    pub limits: HbmTensor<f32, Chip, m![Prompt, Lane]>,
    /// Each key's index within its tile's keys.
    pub positions: HbmTensor<f32, Chip, m![Keys]>,
}

const CEILING: f32 = 2.5e30;
const ABSENT: TagGuard =
    TagGuard::matches([BitReq::One, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]);
const CLEARED: TagGuard =
    TagGuard::matches([BitReq::Zero, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]);
const SECOND_CLEARED: TagGuard =
    TagGuard::matches([BitReq::Ignore, BitReq::Zero, BitReq::Ignore, BitReq::Ignore]);

/// Pass `pass`'s attended rows of both KV heads into `output`, the chunk starting at capacity
/// row `start`.
pub fn attend(
    context: &mut Device,
    (pool, entry): (&Pool, usize),
    (plan, queries, start): (&Plan, &qkv::Queries, usize),
    pass: usize,
    output: &mut Attended,
) {
    let mut landed = Landed::new();
    // SAFETY: the host plan bounds every page by the layer's pages, and each offset starts one.
    unsafe {
        pool.gather::<m![1], m![Page, Heads, Kv, Head]>()
            .entry(entry)
            .by_byte_offsets(&plan.pages)
            .unchecked()
    }
    .to_dm_view(&mut context.tdma, landed.view_mut());
    let mut keys = [Key::new(), Key::new()];
    let mut values = [Value::new(), Value::new()];
    #[unroll]
    for head in 0..2 {
        decode_keys(context, &landed, head, &mut keys[head]);
        decode_values(context, &landed, head, &mut values[head]);
    }
    let first = start + 1024 * pass;
    let pass_queries = [
        query(context, queries, (first, 0)),
        query(context, queries, (first, 1)),
    ];
    let limit = limits(context, plan, pass);
    let positions: Positions = plan.positions.to_dm(&mut context.tdma);
    // A register takes half the tensor register file: the second head's keys take the first
    // head's key half once its scores are out, its values follow the first head's rows.
    let key = key_register(context, &keys[0]);
    let value = value_register(context, &values[0]);
    let first_scores = scores(context, &pass_queries[0], &key);
    drop(key);
    let key = key_register(context, &keys[1]);
    let mut excess = Excess::new();
    mask(context, &first_scores, (&positions, &limit), &mut excess);
    let tags = tags(context, &excess);
    let (mass, product) = weigh(context, (&first_scores, &tags), value);
    normalize(context, (&mass, &product), rows_mut(output, (pass, 0)));
    let value = value_register(context, &values[1]);
    let second_scores = scores(context, &pass_queries[1], &key);
    drop(key);
    let (mass, product) = weigh(context, (&second_scores, &tags), value);
    normalize(context, (&mass, &product), rows_mut(output, (pass, 1)));
}

fn weigh(
    context: &mut Device,
    (scores, tags): (&Scores, &Tags),
    value: ValueRegister,
) -> (Statistic, Output) {
    let mut weights = Scores::new();
    let peak = peaks(context, (scores, tags));
    exponents(context, (scores, tags, &peak), &mut weights);
    product(context, &weights, value)
}

/// Pass `pass`'s limits, landed in every key slice of its tiles in both clusters.
fn limits(context: &mut Device, plan: &Plan, pass: usize) -> Limit {
    let rows = plan
        .limits
        .view()
        .tile::<m![Prompt], 1024, m![Prompt = 1024 # Prompt, Lane]>(1024 * pass);
    // SAFETY: the tile keeps the chunk's row stride; only its row axis is renamed.
    let rows = unsafe { rows.reshape::<Chip, m![Queries # Prompt, Lane]>() };
    let landed: DmTensor<f32, Chip, Cluster, Slices, m![Queries % 128, Lane]> =
        rows.to_dm(&mut context.tdma);
    // SAFETY: `Lane` and `1 # 8` lay out the same eight cells per row.
    unsafe { landed.reshape() }
}

fn mask(
    context: &mut Device,
    scores: &Scores,
    (positions, limit): (&Positions, &Limit),
    excess: &mut Excess,
) {
    let positions: PositionRegister = context
        .sub
        .begin(positions.view())
        .fetch::<m![Keys % 64 / 8], m![Keys % 8]>()
        .collect::<m![Keys % 64 / 8], m![Keys % 8]>()
        .to_vrf();
    let limit: Bound = context
        .sub
        .begin(limit.view())
        .fetch::<m![Queries % 128], m![1 # 8]>()
        .collect::<m![Queries % 128], m![1 # 8]>()
        .to_vrf();
    context
        .main
        .begin(group(scores, 0))
        .fetch::<m![Queries % 128, Keys % 64 / 8], m![Keys % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Queries % 128, Keys % 64 / 8], m![Keys % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, Branched::imm(CLEARED, 0.0f32))
        .vector_narrow_split::<m![Queries % 128, Keys % 64 / 4], m![Keys % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, &limit)
        .vector_widen_concat::<m![Queries % 128, Keys % 64 / 8], m![Keys % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &positions)
        .vector_final()
        .cast::<bf16, m![Keys % 8 # 16]>()
        .commit_trim::<m![Keys % 8]>()
        .commit_view(excess.view_mut());
}

/// Pass `pass`'s rows of head `head` to write.
fn rows_mut(
    output: &mut Attended,
    (pass, head): (usize, usize),
) -> HbmTensorViewMut<
    '_,
    f32,
    Chip,
    m![Queries #{!} Prompt, Heads / 2, Heads % 2 = 1 #{!} 2, Groups, Head],
> {
    let rows = output
        .view_mut()
        .tile::<m![Prompt], 1024, m![Prompt = 1024 #{!} Prompt, Heads, Groups, Head]>(1024 * pass);
    // SAFETY: the pass keeps the chunk's row stride; only its row axis is renamed.
    unsafe { rows.reshape::<Chip, m![Queries #{!} Prompt, Heads, Groups, Head]>() }.tile::<
        m![Heads % 2],
        1,
        m![Queries #{!} Prompt, Heads / 2, Heads % 2 = 1 #{!} 2, Groups, Head],
    >(head)
}

fn operand(context: &mut Device, statistic: &Statistic) -> Operand {
    context
        .sub
        .begin(statistic.view())
        .fetch::<m![Queries % 16], m![1 # 8]>()
        .collect::<m![Queries % 16], m![1 # 8]>()
        .to_vrf()
}

/// `product` divided by `mass`, stored into `rows`.
fn normalize(
    context: &mut Device,
    (mass, product): (&Statistic, &Output),
    rows: HbmTensorViewMut<
        '_,
        f32,
        Chip,
        m![Queries #{!} Prompt, Heads / 2, Heads % 2 = 1 #{!} 2, Groups, Head],
    >,
) {
    let total = operand(context, mass);
    let attended: Output = context
        .main
        .begin(product.view())
        .fetch::<m![Queries % 16, Head / 8], m![Head % 8]>()
        .collect::<m![Queries % 16, Head / 8], m![Head % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Queries % 16, Head / 4], m![Head % 4]>()
        .vector_fp_div(&total)
        .vector_widen_concat::<m![Queries % 16, Head / 8], m![Head % 8]>()
        .vector_final()
        .commit_trim::<m![Head % 8]>()
        .commit();
    attended.view().to_hbm_view(&mut context.tdma, rows);
}

fn key_register(context: &mut Device, key: &Key) -> KeyRegister {
    context
        .sub
        .begin(key.view())
        .fetch::<m![Keys % 8, Keys % 64 / 16, Head / 16, Keys % 16 / 8], m![Head % 16]>()
        .collect::<m![Keys % 8, Keys % 64 / 16, Head / 16, Keys % 16 / 8], m![Head % 16]>()
        .to_trf()
}

fn value_register(context: &mut Device, value: &Value) -> ValueRegister {
    context
        .sub
        .begin(value.view())
        .fetch::<m![Head % 8, Head / 16, Keys % 64 / 16, Head % 16 / 8], m![Keys % 16]>()
        .collect::<m![Head % 8, Head / 16, Keys % 64 / 16, Head % 16 / 8], m![Keys % 16]>()
        .to_trf()
}

/// Head `head`'s keys or values of every slice, in token order.
fn tokens(
    landed: &Landed,
    (kv, head): (usize, usize),
) -> DmTensorView<
    '_,
    f8e4m3,
    Chip,
    Cluster,
    Slices,
    m![Keys % 64, Heads % 2 = 1 # 2, Kv = 1 # 2, Head],
> {
    // SAFETY: every tile's copy of a slice's keys serves that tile, and a slice's four pages of
    // sixteen tokens are its 64 keys in order.
    unsafe {
        landed
            .view()
            .reshape::<Chip, Cluster, Slices, m![Keys % 64, Heads % 2, Kv, Head]>()
    }
    .tile::<m![Kv], 1, m![Keys % 64, Heads % 2, Kv = 1 # 2, Head]>(kv)
    .tile::<m![Heads % 2], 1, m![Keys % 64, Heads % 2 = 1 # 2, Kv = 1 # 2, Head]>(head)
}

fn decode_keys(context: &mut Device, landed: &Landed, head: usize, key: &mut Key) {
    context
        .main
        .begin(tokens(landed, (0, head)))
        .fetch::<m![Keys % 64, Head / 16], m![Head % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Keys % 64, Head / 16], m![Head % 16]>()
        .commit_trim::<m![Head % 16]>()
        .commit_view(key.view_mut());
}

/// Decoded values, one 64-key run per channel, with every non-finite cell cleared: an unwritten
/// page may hold NaN, and a zero weight times NaN is NaN.
fn decode_values(context: &mut Device, landed: &Landed, head: usize, value: &mut Value) {
    let transposed: Transposed = context
        .main
        .begin(tokens(landed, (1, head)))
        .fetch::<m![Head / 16, Keys % 64], m![Head % 16]>()
        .collect::<m![Head / 16, Keys % 64], m![Head % 16 # 32]>()
        .transpose::<m![Head / 16, Keys % 64 / 8, Head % 16], m![Keys % 8 # 32]>()
        .commit_trim::<m![Keys % 8]>()
        .commit();
    context
        .main
        .begin(transposed.view())
        .fetch::<m![Head, Keys % 64 / 8], m![Keys % 8]>()
        .fetch_table_lookup::<bf16>()
        .fetch_cast::<f32>()
        .collect::<m![Head, Keys % 64 / 8], m![Keys % 8]>()
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
        .commit_view(value.view_mut());
}

/// The pass's queries of head `head`, sixteen rows of one query head a slice.
fn query(context: &mut Device, queries: &qkv::Queries, (first, head): (usize, usize)) -> Query {
    // SAFETY: a capacity row's query heads are its `Heads × Groups` heads in head order.
    let rows = unsafe {
        queries
            .view()
            .reshape::<Chip, m![Capacity, Heads, Groups, Head]>()
    }
    .tile::<m![Capacity], 1024, m![Capacity = 1024 # Capacity, Heads, Groups, Head]>(first);
    // SAFETY: the pass keeps the capacity stride; only its row axis is renamed.
    unsafe { rows.reshape::<Chip, m![Queries # Capacity, Heads, Groups, Head]>() }
        .tile::<m![Heads % 2], 1, m![Queries # Capacity, Heads / 2, Heads % 2 = 1 # 2, Groups, Head]>(
            head,
        )
        .to_dm(&mut context.tdma)
}

/// Scaled scores of every query of a tile against every key of the slice: each query broadcasts
/// over its tile's 32 key slices and meets 64 keys there.
fn scores(context: &mut Device, query: &Query, key: &KeyRegister) -> Scores {
    type Stream = m![
        Queries % 16 / 2,
        Keys % 64 / 16,
        Queries % 128 / 16,
        Groups,
        Head / 16
    ];
    type Steps = m![{ Stream }, Keys % 16 / 8];
    type Sums = m![
        Queries % 16 / 2,
        Keys % 64 / 16,
        Queries % 128 / 16,
        Groups,
        Keys % 16 / 8
    ];
    context
        .main
        .begin(query.view())
        .fetch::<m![Queries % 16 / 2, Keys % 64 / 16, Head / 16], m![Queries % 2, Head % 16]>()
        .switch::<Slices, Stream>(SwitchConfig::Broadcast01 {
            slice1: 32,
            slice0: 1,
            time0: 8,
        })
        .collect::<m![{ Stream }, Queries % 2], m![Head % 16]>()
        .contract_outer::<Steps, m![Queries % 2, Head % 16], _, _, _>(key)
        .contract_packet::<m![Queries % 2]>()
        .contract_time::<Sums>()
        .contract_lane::<m![{ Sums }, Queries % 2], m![Keys % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ Sums }, Queries % 2, Keys % 8 / 4], m![Keys % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.08838834764831845f32)
        .vector_widen_concat::<m![{ Sums }, Queries % 2], m![Keys % 8]>()
        .vector_final()
        .commit_trim::<m![Keys % 8]>()
        .commit_cast::<bf16>()
        .commit()
}

fn group(scores: &Scores, group: usize) -> GroupScores<'_> {
    scores
        .view()
        .tile::<m![Groups], 1, m![Groups = 1 # 4, Queries % 128, Keys % 64]>(group)
}

/// The pass's visibility as execution tags: set where a row does not see the key.
fn tags(context: &mut Device, excess: &Excess) -> Tags {
    context
        .main
        .begin(excess.view())
        .fetch::<m![Queries % 128, Keys % 64 / 8], m![Keys % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Queries % 128, Keys % 64 / 8], m![Keys % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Comparison([
            Cmp::Greater(0.0f32),
            Cmp::False,
            Cmp::False,
            Cmp::False,
        ]))
        .vector_log(&mut context.sub)
}

/// Group `at`'s scores as the softmax passes read them: keys a row does not see at the lowest bf16.
fn masked<'l>(
    context: &'l mut Device,
    (scores, tags): (&'l Scores, &Tags),
    at: usize,
) -> Masked<'l> {
    let tagged = context
        .main
        .begin(group(scores, at))
        .fetch::<m![Queries % 128, Keys % 64 / 8], m![Keys % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Queries % 128, Keys % 64 / 8], m![Keys % 8]>()
        .vector_init()
        .vector_tagged(tags);
    VectorTensor::<_, _, f32, _, _, _, _, _, _, _, _, _>::vector_logic_with_mode(
        tagged,
        LogicBinaryOpF32::BitAnd,
        BinaryArgMode::Mode11,
        Branched::imm(ABSENT, f32::from_bits(0xff7f_0000)),
    )
    .vector_narrow_split::<m![Queries % 128, Keys % 64 / 4], m![Keys % 4]>()
}

/// Each row's peak over its tile's keys, all-reduced over the tile's 32 key slices.
fn peaks(context: &mut Device, (scores, tags): (&Scores, &Tags)) -> Peak {
    let mut local = Peak::new();
    for at in 0..4 {
        masked(context, (scores, tags), at)
            .vector_intra_slice_reduce::<Keys, m![Queries % 128], m![1 # 4]>(
                IntraSliceReduceOpF32::Max,
            )
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .commit_trim::<m![1 # 2]>()
            .commit_view(
                local
                    .view_mut()
                    .tile::<m![Groups], 1, m![Groups = 1 #{!} 4, Queries % 128, 1 # 2]>(at),
            );
    }
    context
        .main
        .begin(local.view())
        .fetch::<m![Groups, Queries % 128 / 4], m![Queries % 4, 1 # 2]>()
        .collect::<m![Groups, Queries % 128 / 4], m![Queries % 4, 1 # 2]>()
        .vector_init()
        .vector_inter_slice_reduce_span(InterSliceReduceOpF32::Max, 5)
        .vector_final()
        .commit_trim::<m![Queries % 4, 1 # 2]>()
        .commit()
}

/// Unnormalized softmax weights into `weights`: each the exponent of a score above `peak`.
fn exponents(
    context: &mut Device,
    (scores, tags, peak): (&Scores, &Tags, &Peak),
    weights: &mut Scores,
) {
    let tops = [
        top(context, peak, 0),
        top(context, peak, 1),
        top(context, peak, 2),
        top(context, peak, 3),
    ];
    #[unroll]
    for at in 0..4 {
        masked(context, (scores, tags), at)
            .vector_fp_binary(FpBinaryOp::SubF, &tops[at])
            .vector_fp_unary(FpUnaryOp::Exp)
            .vector_widen_concat::<m![Queries % 128, Keys % 64 / 8], m![Keys % 8]>()
            .vector_clip(ClipBinaryOpF32::Min, CEILING)
            .vector_final()
            .cast::<bf16, m![Keys % 8 # 16]>()
            .commit_trim::<m![Keys % 8]>()
            .commit_view(
                weights
                    .view_mut()
                    .tile::<m![Groups], 1, m![Groups = 1 #{!} 4, Queries % 128, Keys % 64]>(at),
            );
    }
}

fn top(context: &mut Device, peak: &Peak, at: usize) -> Top {
    context
        .sub
        .begin(
            peak.view()
                .tile::<m![Groups], 1, m![Groups = 1 # 4, Queries % 128, 1 # 2]>(at),
        )
        .fetch::<m![Queries % 128 / 4], m![Queries % 4, 1 # 2]>()
        .collect::<m![Queries % 128 / 4], m![Queries % 4, 1 # 2]>()
        .to_vrf()
}

/// Each query's softmax mass and its weighted values, both summed over its tile's key slices onto
/// the query's own slice; the mass is the weights' contraction with ones.
fn product(context: &mut Device, weights: &Scores, value: ValueRegister) -> (Statistic, Output) {
    // A row of the weights with its bits cleared before the one is added, so even a non-finite
    // cell gives a one in every key of the slice.
    let ones: Ones = context
        .main
        .begin(group(weights, 0).tile::<m![Queries % 128], 1, m![
            Groups = 1 # 4,
            Queries % 128 = 1 # 128,
            Keys % 64
        ]>(0))
        .fetch::<m![Keys % 64 / 8], m![Keys % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Keys % 64 / 8], m![Keys % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)
        .vector_clip(ClipBinaryOpF32::Add, 1.0f32)
        .vector_final()
        .cast::<bf16, m![Keys % 8 # 16]>()
        .commit_trim::<m![Keys % 8]>()
        .commit();
    type Fetch = m![
        Queries % 16,
        Head / 16,
        Queries % 128 / 16,
        Groups / 2,
        Keys % 64 / 16
    ];
    type Steps = m![{ Fetch }, Head % 16 / 8];
    type Sums = m![
        Queries % 16,
        Head / 16,
        Queries % 128 / 16,
        Groups / 2,
        Head % 16 / 8
    ];
    let product: Output = context
        .main
        .begin(weights.view())
        .fetch::<Fetch, m![Groups % 2, Keys % 16]>()
        .collect::<m![{ Fetch }, Groups % 2], m![Keys % 16]>()
        .contract_outer::<Steps, m![Groups % 2, Keys % 16], _, _, _>(&value)
        .contract_packet::<m![Groups % 2]>()
        .contract_time::<Sums>()
        .contract_lane::<m![{ Sums }, Groups % 2], m![Head % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<QuerySlices, m![Queries % 16, Head / 8]>(
            InterSliceReduceOpF32::Add,
        )
        .vector_final()
        .commit_trim::<m![Head % 8]>()
        .commit();
    drop(value);
    let ones: OnesRegister = context
        .sub
        .begin(ones.view())
        .fetch::<m![Keys % 64 / 16], m![Keys % 16]>()
        .collect::<m![Keys % 64 / 16], m![Keys % 16]>()
        .to_trf();
    type Rows = m![Queries % 128 / 16, Groups, Queries % 16];
    let mass: Statistic = context
        .main
        .begin(weights.view())
        .fetch::<m![{ Rows }, Keys % 64 / 16], m![Keys % 16]>()
        .collect::<m![{ Rows }, Keys % 64 / 16], m![Keys % 16]>()
        .contract_outer::<m![{ Rows }, Keys % 64 / 32], m![Keys % 32], _, _, _>(&ones)
        .contract_packet::<m![1]>()
        .contract_time::<Rows>()
        .contract_lane::<Rows, m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<QuerySlices, m![Queries % 16]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    (mass, product)
}
