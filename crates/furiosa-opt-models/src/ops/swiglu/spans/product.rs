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
        let mut output = Projection::new();
        multiply::<Intermediate, m![Rows % (Rows / 8)], _, _, _>(
            context,
            (gate.view(), up.view()),
            output.view_mut(),
            rows.div_ceil(64) * 8,
        );
        output
    }
}

/// The two BF16 rounding boundaries of SwiGLU, independent of its physical row ownership.
fn multiply<Intermediate: AxisName, Row: M, S: M, Destination: M, E: M>(
    context: &mut Device,
    (gate, up): (
        DmTensorView<'_, bf16, Chip, Cluster<Intermediate>, S, m![{Row}, {Out<Intermediate>}]>,
        DmTensorView<'_, bf16, Chip, Cluster<Intermediate>, S, m![{Row}, {Out<Intermediate>}]>,
    ),
    output: DmTensorViewMut<'_, bf16, Chip, Cluster<Intermediate>, Destination, E>,
    groups: usize,
) {
    let mut activated: DmTensor<
        bf16,
        Chip,
        Cluster<Intermediate>,
        S,
        m![{Row}, {Out<Intermediate>}],
    > = DmTensor::new();
    context
        .main
        .begin(gate)
        .fetch::<m![{Row}, { Out<Intermediate> } / 8], m![Intermediate % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![{Row}, { Out<Intermediate> } / 8], m![Intermediate % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{Row}, { Out<Intermediate> } / 4], m![Intermediate % 4]>()
        .vector_silu()
        .vector_widen_concat::<m![{Row}, { Out<Intermediate> } / 8], m![Intermediate % 8]>()
        .vector_final()
        .cast::<bf16, m![Intermediate % 8 # 16]>()
        .commit_trim::<m![Intermediate % 8]>()
        .commit_prefix::<m![{ Row }], _, _>(activated.view_mut(), groups);
    context.main
            .begin_interleaved::<Term, _, _, _, _, _>(activated.view(), up)
            .fetch::<
                m![{Row}, Term, { Out<Intermediate> } / 8],
                m![Intermediate % 8],
            >()
            .fetch_cast::<f32>()
            .collect::<
                m![{Row}, Term, { Out<Intermediate> } / 8],
                m![Intermediate % 8],
            >()
            .vector_init()
            .vector_zip::<
                Term,
                m![{Row}, { Out<Intermediate> } / 8],
                m![{Row}, { Out<Intermediate> } / 4],
                m![Intermediate % 4],
            >(FpBinaryOp::MulF(FpMulAlu::Fma))
            .vector_final().cast::<bf16, m![Intermediate % 8 # 16]>()
            .commit_trim::<m![Intermediate % 8]>()
            .commit_prefix::<m![{Row}], _, _>(output, groups);
}

/// Short native FP8 spans keep only their live eight-row owners and the first eight local rows.
pub(super) fn short<'a, I: AxisName, R: AxisName, const G: usize>(
    context: &mut Device,
    gate: &'a Projection<I, R>,
    up: &'a Projection<I, R>,
) -> Projection<I, R> {
    type S<I, R, const G: usize> = m![{Blocks<I>}, R / (R / 8) = {G} # 8];
    let input = |value: &'a Projection<I, R>| {
        value
            .view()
            .slice_tile::<m![R / (R / 8)], G, S<I, R, G>>(0)
            .tile::<m![R % (R / 8)], 8, m![R % (R / 8) = 8 # (R / 8), {Out<I>}]>(0)
            .unpad::<m![R % (R / 8) = 8, {Out<I>}]>()
    };
    let mut output = Projection::<I, R>::new();
    let into = output
        .view_mut()
        .slice_tile::<m![R / (R / 8)], G, m![{Blocks<I>}, R / (R / 8) = {G} #{!} 8]>(0)
        .tile::<m![R % (R / 8)], 8, m![R % (R / 8) = 8 #{!} (R / 8), {Out<I>}]>(0);
    multiply::<I, m![R % (R / 8) = 8], _, _, _>(context, (input(gate), input(up)), into, 8);
    output
}
