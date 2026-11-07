use super::*;

pub(super) fn contract<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    (weight, lanes): (
        &Panel<Hidden, Intermediate>,
        &Lanes<Hidden, Intermediate, Rows>,
    ),
    (output, term, wave): (&mut Both<Intermediate, Rows>, usize, usize),
    rows: usize,
) {
    context.main.begin(weight.view())
        .fetch::<m![Rows % 16 / 8, { Out<Intermediate> } / 4, { Out<Intermediate> } % 4 / 2,
            Rows / (Rows / 4), { K<Hidden> } / 32],
            m![{ Out<Intermediate> } % 2, { K<Hidden> } % 32]>()
        .collect::<m![Rows % 16 / 8, { Out<Intermediate> } / 2, Rows / (Rows / 4),
            { K<Hidden> } / 32, { Out<Intermediate> } % 2], m![{ K<Hidden> } % 32]>()
        .contract_outer::<m![Rows % 16 / 8, { Out<Intermediate> } / 2, Rows / (Rows / 4),
            { K<Hidden> } / 32, Rows % (Rows / 4) / (Rows / 8)],
            m![{ Out<Intermediate> } % 2, { K<Hidden> } % 32], _, _, _>(lanes)
        .contract_packet::<m![{ Out<Intermediate> } % 2]>()
        .contract_time::<m![Rows % 16 / 8, { Out<Intermediate> } / 2, Rows / (Rows / 8)]>()
        .contract_lane::<m![Rows % 16 / 8, { Out<Intermediate> } / 2, Rows / (Rows / 8),
            { Out<Intermediate> } % 2], m![Rows % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<Slice<Intermediate, Rows>,
            m![Rows % 16 / 8, { Out<Intermediate> }]>(InterSliceReduceOpF32::Add)
        .vector_final().cast::<bf16, m![Rows % 8 # 16]>()
        .transpose::<m![Rows % 16 / 8, { Out<Intermediate> } / 4, Rows % 8],
            m![{ Out<Intermediate> } % 4 # 16]>().commit_trim::<m![{ Out<Intermediate> } % 4]>()
        .commit_prefix::<m![Rows % 16 / 8], _, _>(output.view_mut()
            .tile::<m![Term], 1,
                m![Rows % (Rows / 8), Term = 1 #{!} 2, { Out<Intermediate> }]>(term)
            .tile::<m![Rows % (Rows / 8) / 16], 1,
                m![Rows % (Rows / 8) / 16 = 1 #{!} (Rows / 128), Rows % 16,
                    Term = 1 #{!} 2, { Out<Intermediate> }]>(wave),
            (rows + 63) / 64 - 2 * wave);
}

pub(super) fn contract_both<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    (weight, lanes): (
        &GateUp<Hidden, Intermediate>,
        &Lanes<Hidden, Intermediate, Rows>,
    ),
    (output, wave): (&mut Both<Intermediate, Rows>, usize),
    rows: usize,
) {
    context.main.begin(weight.view())
        .fetch::<m![Rows % 16 / 8, Term, { Out<Intermediate> } / 4,
            { Out<Intermediate> } % 4 / 2, Rows / (Rows / 4), { K<Hidden> } / 32],
            m![{ Out<Intermediate> } % 2, { K<Hidden> } % 32]>()
        .collect::<m![Rows % 16 / 8, Term, { Out<Intermediate> } / 2, Rows / (Rows / 4),
            { K<Hidden> } / 32, { Out<Intermediate> } % 2], m![{ K<Hidden> } % 32]>()
        .contract_outer::<m![Rows % 16 / 8, Term, { Out<Intermediate> } / 2,
            Rows / (Rows / 4), { K<Hidden> } / 32, Rows % (Rows / 4) / (Rows / 8)],
            m![{ Out<Intermediate> } % 2, { K<Hidden> } % 32], _, _, _>(lanes)
        .contract_packet::<m![{ Out<Intermediate> } % 2]>()
        .contract_time::<m![Rows % 16 / 8, Term, { Out<Intermediate> } / 2, Rows / (Rows / 8)]>()
        .contract_lane::<m![Rows % 16 / 8, Term, { Out<Intermediate> } / 2,
            Rows / (Rows / 8), { Out<Intermediate> } % 2], m![Rows % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<Slice<Intermediate, Rows>,
            m![Rows % 16 / 8, Term, { Out<Intermediate> }]>(InterSliceReduceOpF32::Add)
        .vector_final().cast::<bf16, m![Rows % 8 # 16]>()
        .transpose::<m![Rows % 16 / 8, Term, { Out<Intermediate> } / 4, Rows % 8],
            m![{ Out<Intermediate> } % 4 # 16]>().commit_trim::<m![{ Out<Intermediate> } % 4]>()
        .commit_prefix::<m![Rows % 16 / 8], _, _>(output.view_mut()
            .tile::<m![Rows % (Rows / 8) / 16], 1,
                m![Rows % (Rows / 8) / 16 = 1 #{!} (Rows / 128), Rows % 16,
                    Term, { Out<Intermediate> }]>(wave), (rows + 63) / 64 - 2 * wave);
}

pub(super) fn add_both<Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    (left, right): (&Both<Intermediate, Rows>, &Both<Intermediate, Rows>),
) -> Both<Intermediate, Rows> {
    context.main.begin_interleaved::<Side, _, _, _, _, _>(left.view(), right.view())
        .fetch::<m![Rows % (Rows / 8), Term, Side, { Out<Intermediate> } / 8],
            m![{ Out<Intermediate> } % 8]>().fetch_cast::<f32>()
        .collect::<m![Rows % (Rows / 8), Term, Side, { Out<Intermediate> } / 8],
            m![{ Out<Intermediate> } % 8]>()
        .vector_init().vector_intra_slice_unzip::<Side,
            m![Rows % (Rows / 8), Term, { Out<Intermediate> } / 8]>()
        .vector_clip_zip(ClipBinaryOpF32::Add).vector_final()
        .cast::<bf16, m![{ Out<Intermediate> } % 8 # 16]>()
        .commit_trim::<m![{ Out<Intermediate> } % 8]>().commit()
}

fn side<Intermediate: AxisName, Rows: AxisName>(
    both: &Both<Intermediate, Rows>,
    term: usize,
) -> DmTensorView<
    '_,
    bf16,
    Chip,
    Cluster<Intermediate>,
    Slice<Intermediate, Rows>,
    m![Rows % (Rows / 8), Term = 1 # 2, { Out<Intermediate> }],
> {
    both.view()
        .tile::<m![Term], 1, m![Rows % (Rows / 8), Term = 1 # 2, { Out<Intermediate> }]>(term)
}

pub(super) fn scaled<Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    (left, right, term): (&Both<Intermediate, Rows>, &Both<Intermediate, Rows>, usize),
    (row, channel): (
        &RowScale<Intermediate, Rows>,
        &ChannelScale<Intermediate, Rows>,
    ),
    rows: usize,
) -> Projection<Intermediate, Rows> {
    let live = (rows + 63) / 64 * 8;
    let mut sum: DmTensor<
        f32,
        Chip,
        Cluster<Intermediate>,
        Slice<Intermediate, Rows>,
        m![Rows % (Rows / 8), { Out<Intermediate> }],
    > = DmTensor::new();
    context.main.begin_interleaved::<Side, _, _, _, _, _>(
        side::<Intermediate, Rows>(left, term), side::<Intermediate, Rows>(right, term))
        .fetch::<m![Rows % (Rows / 8), Side, { Out<Intermediate> } / 8],
            m![{ Out<Intermediate> } % 8]>().fetch_cast::<f32>()
        .collect::<m![Rows % (Rows / 8), Side, { Out<Intermediate> } / 8],
            m![{ Out<Intermediate> } % 8]>()
        .vector_init().vector_intra_slice_unzip::<Side,
            m![Rows % (Rows / 8), { Out<Intermediate> } / 8]>()
        .vector_clip_zip(ClipBinaryOpF32::Add).vector_final()
        .commit_trim::<m![{ Out<Intermediate> } % 8]>()
        .commit_prefix::<m![Rows % (Rows / 8)], _, _>(sum.view_mut(), live);
    let mut output: Projection<Intermediate, Rows> = DmTensor::new();
    context.main.begin(sum.view())
        .fetch::<m![Rows % (Rows / 8), { Out<Intermediate> } / 8], m![{ Out<Intermediate> } % 8]>()
        .collect::<m![Rows % (Rows / 8), { Out<Intermediate> } / 8], m![{ Out<Intermediate> } % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Rows % (Rows / 8), { Out<Intermediate> } / 4],
            m![{ Out<Intermediate> } % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), row)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), channel)
        .vector_widen_concat::<m![Rows % (Rows / 8), { Out<Intermediate> } / 8],
            m![{ Out<Intermediate> } % 8]>()
        .vector_final().cast::<bf16, m![{ Out<Intermediate> } % 8 # 16]>()
        .commit_trim::<m![{ Out<Intermediate> } % 8]>()
        .commit_prefix::<m![Rows % (Rows / 8)], _, _>(output.view_mut(), live);
    output
}
