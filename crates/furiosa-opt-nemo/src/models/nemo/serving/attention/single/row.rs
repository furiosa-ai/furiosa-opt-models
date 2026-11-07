//! A one-row batch's wave over the first [`SINGLE_PAGES`] pages of each job slot: the host makes
//! the row's jobs no longer, so the wave's Main work is that share of a burst wave's.
//!
//! The scores and weights hold the groups outermost, so the probabilities' register fill reads two
//! jobs' pages as one flit.
use furiosa_opt_std::prelude::*;

use super::job;
use crate::models::nemo::serving::attention::{Tables, *};

type Pages = m![Tokens / 16 % 8];
type Token<'l> = DmTensorView<
    'l,
    f8e4m3,
    Chip,
    Cluster,
    Slices,
    m![Jobs % 8, Tokens / 128 = 1 # 4, { Pages }, Kv = 1 # 2, Head],
>;
type Decoded = DmTensor<bf16, Chip, Cluster, Slices, m![Jobs % 8, { Pages }, Head]>;
type Scores = DmTensor<f32, Chip, Cluster, Slices, m![Groups, Jobs % 8, { Pages }]>;
type Weights = DmTensor<bf16, Chip, Cluster, Slices, m![Groups, Jobs % 8, { Pages }]>;
type Probability = TrfTensor<bf16, Chip, Cluster, Slices, m![Groups], m![Jobs % 8, { Pages }]>;
type Indices<'l> =
    DmTensorView<'l, f32, Chip, Cluster, Slices, m![Tokens / 128 = 1 # 4, { Pages }]>;
type Operand = VrfTensor<f32, Chip, Cluster, Slices, m![Groups, Jobs % 8, 1 # 8]>;

const _: () = assert!(Pages::SIZE == SINGLE_PAGES);

impl<L: Lead> Batch<'_, '_, L> {
    /// The one-row batch's wave over `raw`'s first pages, its partials kept in `buffers` for the
    /// on-chip merge; the layer's first gate half lands under it.
    pub(in crate::models::nemo::serving::attention) fn attend_row(
        &mut self,
        context: &mut Device,
        raw: &Raw,
        buffers: &mut Tables,
    ) {
        for _ in 0..self.prefetch.gate.2 {
            let (gate, landed, _) = &mut self.prefetch.gate;
            gate.land_first(context, landed);
        }
        let key = keys(context, raw);
        let parity = self.history.batch % 2;
        let lanes = lanes(context, &buffers.staged, (parity, 0));
        let scores = scores(context, &key, &lanes);
        let positions = self
            .history
            .positions
            .view()
            .tile::<m![Tokens / 128], 1, m![Tokens / 128 = 1 # 4, { Pages }]>(0);
        let scores = mask(
            context,
            &scores,
            (positions, counts(&buffers.landed, (parity, 0))),
        );
        peak(context, &scores, &mut buffers.records);
        let weights = weights(context, &scores, &buffers.records);
        mass(context, &weights, &mut buffers.records);
        let probabilities = fill(context, &weights);
        let value = values(context, raw);
        let weighted = weighted(context, &value, &probabilities);
        block::settle(context, &weighted, &mut buffers.rounded, 0);
    }
}

/// The first pages' tokens of one of the two records.
fn tokens(raw: &Raw, kv: usize) -> Token<'_> {
    raw.view()
        .tile::<m![Tokens / 128], 1, m![Jobs % 8, Tokens / 128 = 1 # 4, { Pages }, Kv, Head]>(0)
        .tile::<m![Kv], 1, m![Jobs % 8, Tokens / 128 = 1 # 4, { Pages }, Kv = 1 # 2, Head]>(kv)
}

/// Widened through f32: the A0 fetch converts an 8-bit float only to f32.
fn keys(context: &mut Device, raw: &Raw) -> Decoded {
    context
        .main
        .begin(tokens(raw, 0))
        .fetch::<m![Jobs % 8, { Pages }, Head / 8], m![Head % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Jobs % 8, { Pages }, Head / 8], m![Head % 8]>()
        .cast::<bf16, m![Head % 8 # 16]>()
        .commit_trim::<m![Head % 8]>()
        .commit()
}

/// Decoded values with every non-finite cell cleared, as a burst wave's.
fn values(context: &mut Device, raw: &Raw) -> Decoded {
    context
        .main
        .begin(tokens(raw, 1))
        .fetch::<m![Jobs % 8, { Pages }, Head / 8], m![Head % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Jobs % 8, { Pages }, Head / 8], m![Head % 8]>()
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
        .fetch::<m![Jobs % 8, { Pages }, Head / 16], m![Head % 16]>()
        .collect::<m![Jobs % 8, { Pages }, Head / 16], m![Head % 16]>()
        .contract_outer::<m![Jobs % 8, { Pages }, Head / 32], m![Head % 32], _, _, _>(query)
        .contract_packet::<m![1]>()
        .contract_time::<m![Jobs % 8, { Pages }]>()
        .contract_lane::<m![Jobs % 8, { Pages }], m![Groups # 8]>(LaneMode::Interleaved)
        .transpose::<m![Jobs % 8, Tokens / 32 % 4, Groups], m![Tokens / 16 % 2 # 8]>()
        .commit_trim::<m![Tokens / 16 % 2]>()
        .commit()
}

/// Scaled scores, absent keys at the lowest bf16, as [`block::mask`].
fn mask(
    context: &mut Device,
    scores: &Scores,
    (positions, counts): (Indices<'_>, Counts<'_>),
) -> Weights {
    let count: Operand = context
        .sub
        .begin(counts)
        .fetch::<m![Groups, Jobs % 8], m![1 # 8]>()
        .collect::<m![Groups, Jobs % 8], m![1 # 8]>()
        .to_vrf();
    let beyond: Scores = context
        .main
        .begin(positions)
        .fetch::<m![Groups, Jobs % 8], m![{ Pages }]>()
        .collect::<m![Groups, Jobs % 8], m![{ Pages }]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Groups, Jobs % 8, Tokens / 64 % 2], m![Tokens / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, &count)
        .vector_widen_concat::<m![Groups, Jobs % 8], m![{ Pages }]>()
        .vector_final()
        .commit_trim::<m![{ Pages }]>()
        .commit();
    let tags: VrfTensor<i4, Chip, Cluster, Slices, m![Groups, Jobs % 8, { Pages }]> = context
        .main
        .begin(beyond.view())
        .fetch::<m![Groups, Jobs % 8], m![{ Pages }]>()
        .collect::<m![Groups, Jobs % 8], m![{ Pages }]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Comparison([
            Cmp::Greater(-0.5f32),
            Cmp::False,
            Cmp::False,
            Cmp::False,
        ]))
        .vector_log(&mut context.sub);
    let scaled: Scores = context
        .main
        .begin(scores.view())
        .fetch::<m![Groups, Jobs % 8], m![{ Pages }]>()
        .collect::<m![Groups, Jobs % 8], m![{ Pages }]>()
        .vector_init()
        .vector_tagged(&tags)
        .vector_scale(Branched::imm(
            TagGuard::matches([BitReq::Zero, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]),
            block::SCALE,
        ))
        .vector_final()
        .commit_trim::<m![{ Pages }]>()
        .commit();
    context
        .main
        .begin(scaled.view())
        .fetch::<m![Groups, Jobs % 8], m![{ Pages }]>()
        .collect::<m![Groups, Jobs % 8], m![{ Pages }]>()
        .vector_init()
        .vector_tagged(&tags)
        .vector_logic_with_mode(
            LogicBinaryOpF32::BitAnd,
            BinaryArgMode::Mode11,
            Branched::imm(block::ABSENT, f32::from_bits(0xff7f_0000)),
        )
        .vector_final()
        .cast::<bf16, m![{ Pages } # 16]>()
        .commit_trim::<m![{ Pages }]>()
        .commit()
}

/// Each job's peak over its pages, into record 0 of the wave.
fn peak(context: &mut Device, scores: &Weights, records: &mut Records) {
    let local: Local = context
        .main
        .begin(scores.view())
        .fetch::<m![Groups, Jobs % 8], m![{ Pages }]>()
        .fetch_cast::<f32>()
        .collect::<m![Groups, Jobs % 8], m![{ Pages }]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Groups, Jobs % 8, Tokens / 64 % 2], m![Tokens / 16 % 4]>()
        .vector_intra_slice_reduce::<Tokens, m![Groups, Jobs % 8], m![1 # 4]>(
            IntraSliceReduceOpF32::Max,
        )
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    block::job_peak(context, &local, block::stat(records, 0, 0));
}

fn weights(context: &mut Device, scores: &Weights, records: &Records) -> Weights {
    let top: Operand = context
        .sub
        .begin(job(records, 0))
        .fetch::<m![Groups, Jobs % 8], m![1 # 8]>()
        .collect::<m![Groups, Jobs % 8], m![1 # 8]>()
        .to_vrf();
    context
        .main
        .begin(scores.view())
        .fetch::<m![Groups, Jobs % 8], m![{ Pages }]>()
        .fetch_cast::<f32>()
        .collect::<m![Groups, Jobs % 8], m![{ Pages }]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Groups, Jobs % 8, Tokens / 64 % 2], m![Tokens / 16 % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, &top)
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_widen_concat::<m![Groups, Jobs % 8], m![{ Pages }]>()
        .vector_final()
        .cast::<bf16, m![{ Pages } # 16]>()
        .commit_trim::<m![{ Pages }]>()
        .commit()
}

/// Each job's weight mass over its pages, into record 1 of the wave.
fn mass(context: &mut Device, weights: &Weights, records: &mut Records) {
    let local: Local = context
        .main
        .begin(weights.view())
        .fetch::<m![Groups, Jobs % 8], m![{ Pages }]>()
        .fetch_cast::<f32>()
        .collect::<m![Groups, Jobs % 8], m![{ Pages }]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Groups, Jobs % 8, Tokens / 64 % 2], m![Tokens / 16 % 4]>()
        .vector_intra_slice_reduce::<Tokens, m![Groups, Jobs % 8], m![1 # 4]>(
            IntraSliceReduceOpF32::Add,
        )
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    block::job_mass(context, &local, block::stat(records, 0, 1));
}

/// Two jobs' pages a flit.
fn fill(context: &mut Device, weights: &Weights) -> Probability {
    context
        .sub
        .begin(weights.view())
        .fetch::<m![Groups, Jobs % 8 / 2], m![Jobs % 2, { Pages }]>()
        .collect::<m![Groups, Jobs % 8 / 2], m![Jobs % 2, { Pages }]>()
        .to_trf()
}

fn weighted(context: &mut Device, value: &Decoded, weights: &Probability) -> Partial {
    context
        .main
        .begin(value.view())
        .fetch::<m![Jobs % 8, Head / 32, { Pages }], m![Head % 32]>()
        .collect::<m![Jobs % 8, Head / 32, { Pages }, Head % 32 / 16], m![Head % 16]>()
        .contract_outer::<m![Jobs % 8, Head / 32, { Pages }], m![Head % 32], _, _, _>(weights)
        .contract_packet::<m![Head % 32]>()
        .contract_time::<m![Jobs % 8, Head / 32]>()
        .contract_lane::<m![Jobs % 8, Head / 32, Groups, Head % 32 / 8], m![Head % 8]>(
            LaneMode::Sequential,
        )
        .vector_init()
        .vector_inter_slice_reduce_span(InterSliceReduceOpF32::Add, 4)
        .vector_final()
        .commit_trim::<m![Head % 8]>()
        .commit()
}
