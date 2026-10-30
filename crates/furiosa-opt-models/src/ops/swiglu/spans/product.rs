use super::*;
use crate::Op;
use crate::ops::pointwise::SwiGlu;
use crate::ops::vector::{Silu as _, Zip};
use crate::tensor::live::Live;

/// Products in the order the down projection's staging reads them: each slice's three panels of
/// 64 features, its rows by wave, half and lane.
pub(super) type Halves<Intermediate, Rows> = DmTensor<
    bf16,
    Chip,
    Cluster<Intermediate>,
    Slice<Intermediate, Rows>,
    m![
        { Out<Intermediate> } / 64,
        Rows % (Rows / 8) / 16,
        Rows % 16 / 8,
        Rows % 8,
        Intermediate % 64
    ],
>;

/// `bf16(bf16(silu(gate)) * up)` of a span's gate and up projections, the live rows' row groups,
/// in the order the down projection stages them.
impl<Intermediate: AxisName, Rows: AxisName, Capacity: AxisName>
    Op<
        Live<
            (
                Projection<Intermediate, Rows>,
                Projection<Intermediate, Rows>,
            ),
            Capacity,
        >,
    > for SwiGlu
{
    type Output = Halves<Intermediate, Rows>;

    fn forward(
        &self,
        context: &mut Device,
        projected: Live<
            (
                Projection<Intermediate, Rows>,
                Projection<Intermediate, Rows>,
            ),
            Capacity,
        >,
    ) -> Self::Output {
        let rows = projected.rows();
        let (gate, up) = projected.into_value();
        let product = Multiply.forward(&mut *context, ((&gate, &up), rows));
        let mut output = Halves::<Intermediate, Rows>::new();
        context
            .main
            .begin(product.view())
            .fetch::<m![
                { Out<Intermediate> } / 64,
                Rows % (Rows / 8) / 16,
                Rows % 16 / 8,
                Rows % 8,
                Intermediate % 64 / 16
            ], m![Intermediate % 16]>()
            .collect::<m![
                { Out<Intermediate> } / 64,
                Rows % (Rows / 8) / 16,
                Rows % 16 / 8,
                Rows % 8,
                Intermediate % 64 / 16
            ], m![Intermediate % 16]>()
            .commit_trim::<m![Intermediate % 16]>()
            .commit_view(output.view_mut());
        output
    }
}

/// SiLU of the gate times the up projection, the first `rows` rows' row groups.
pub(crate) struct Multiply;

impl<'a, Intermediate: AxisName, Rows: AxisName>
    Op<(
        (
            &'a Projection<Intermediate, Rows>,
            &'a Projection<Intermediate, Rows>,
        ),
        usize,
    )> for Multiply
{
    type Output = Projection<Intermediate, Rows>;

    fn forward(
        &self,
        context: &mut Device,
        ((gate, up), rows): (
            (
                &'a Projection<Intermediate, Rows>,
                &'a Projection<Intermediate, Rows>,
            ),
            usize,
        ),
    ) -> Self::Output {
        let groups = (rows + 63) / 64 * 8;
        let mut activated: Projection<Intermediate, Rows> = DmTensor::new();
        context.main.begin(gate.view())
            .fetch::<m![Rows % (Rows / 8), { Out<Intermediate> } / 8], m![Intermediate % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Rows % (Rows / 8), { Out<Intermediate> } / 8], m![Intermediate % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<
                m![Rows % (Rows / 8), { Out<Intermediate> } / 4],
                m![Intermediate % 4],
            >()
            .vector_silu()
            .vector_widen_concat::<
                m![Rows % (Rows / 8), { Out<Intermediate> } / 8],
                m![Intermediate % 8],
            >()
            .vector_final().cast::<bf16, m![Intermediate % 8 # 16]>()
            .commit_trim::<m![Intermediate % 8]>()
            .commit_prefix::<m![Rows % (Rows / 8)], _, _>(activated.view_mut(), groups);
        let mut output: Projection<Intermediate, Rows> = DmTensor::new();
        context.main
            .begin_interleaved::<Term, _, _, _, _, _>(activated.view(), up.view())
            .fetch::<
                m![Rows % (Rows / 8), Term, { Out<Intermediate> } / 8],
                m![Intermediate % 8],
            >()
            .fetch_cast::<f32>()
            .collect::<
                m![Rows % (Rows / 8), Term, { Out<Intermediate> } / 8],
                m![Intermediate % 8],
            >()
            .vector_init()
            .vector_zip::<
                Term,
                m![Rows % (Rows / 8), { Out<Intermediate> } / 8],
                m![Rows % (Rows / 8), { Out<Intermediate> } / 4],
                m![Intermediate % 4],
            >(FpBinaryOp::MulF(FpMulAlu::Fma))
            .vector_final().cast::<bf16, m![Intermediate % 8 # 16]>()
            .commit_trim::<m![Intermediate % 8]>()
            .commit_prefix::<m![Rows % (Rows / 8)], _, _>(output.view_mut(), groups);
        output
    }
}
