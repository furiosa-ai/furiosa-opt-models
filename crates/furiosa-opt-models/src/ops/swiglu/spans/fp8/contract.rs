use super::*;
use crate::Op;

/// A panel's contraction with a wave's TRF lanes into the wave's rows of the term's sums, the
/// wave's first live row groups alone.
pub(super) struct Contract;

impl<'a, Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>
    Op<(
        (
            &'a Panel<Hidden, Intermediate>,
            &'a Lanes<Hidden, Intermediate, Rows>,
        ),
        (&'a mut Both<Intermediate, Rows>, usize, usize),
        usize,
    )> for Contract
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        ((weight, lanes), (output, term, wave), rows): (
            (
                &'a Panel<Hidden, Intermediate>,
                &'a Lanes<Hidden, Intermediate, Rows>,
            ),
            (&'a mut Both<Intermediate, Rows>, usize, usize),
            usize,
        ),
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
}

/// A gate/up panel pair's contraction with a wave's TRF lanes into the wave's rows of both terms'
/// sums.
impl<'a, Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>
    Op<(
        (
            &'a GateUp<Hidden, Intermediate>,
            &'a Lanes<Hidden, Intermediate, Rows>,
        ),
        (&'a mut Both<Intermediate, Rows>, usize),
        usize,
    )> for Contract
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        ((weight, lanes), (output, wave), rows): (
            (
                &'a GateUp<Hidden, Intermediate>,
                &'a Lanes<Hidden, Intermediate, Rows>,
            ),
            (&'a mut Both<Intermediate, Rows>, usize),
            usize,
        ),
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
            .contract_time::<
                m![Rows % 16 / 8, Term, { Out<Intermediate> } / 2, Rows / (Rows / 8)]
            >()
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
}

/// A term's two partial sums added in FP32 and scaled by the row and channel scales, the first
/// `rows` rows' row groups.
pub(super) struct Combine;

/// The two partial sums with the term they hold, and the row and channel scales.
type Terms<'a, Intermediate, Rows> = (
    (
        &'a Both<Intermediate, Rows>,
        &'a Both<Intermediate, Rows>,
        usize,
    ),
    (
        &'a RowScale<Intermediate, Rows>,
        &'a ChannelScale<Intermediate, Rows>,
    ),
    usize,
);

impl<'a, Intermediate: AxisName, Rows: AxisName> Op<Terms<'a, Intermediate, Rows>> for Combine {
    type Output = Projection<Intermediate, Rows>;

    fn forward(
        &self,
        context: &mut Device,
        ((left, right, term), (row, channel), rows): Terms<'a, Intermediate, Rows>,
    ) -> Self::Output {
        let side = |both: &'a Both<Intermediate, Rows>, term: usize| {
            both.view()
                .tile::<m![Term], 1, m![Rows % (Rows / 8), Term = 1 # 2, { Out<Intermediate> }]>(
                    term,
                )
        };
        let live = (rows + 63) / 64 * 8;
        let mut sum: DmTensor<
            f32,
            Chip,
            Cluster<Intermediate>,
            Slice<Intermediate, Rows>,
            m![Rows % (Rows / 8), { Out<Intermediate> }],
        > = DmTensor::new();
        context.main.begin_interleaved::<Side, _, _, _, _, _>(
            side(left, term), side(right, term))
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
            .fetch::<
                m![Rows % (Rows / 8), { Out<Intermediate> } / 8],
                m![{ Out<Intermediate> } % 8],
            >()
            .collect::<
                m![Rows % (Rows / 8), { Out<Intermediate> } / 8],
                m![{ Out<Intermediate> } % 8],
            >()
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
}
