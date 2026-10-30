use super::*;
use crate::Op;

/// The row scales, gathered into the slices that rescale the partial sums.
pub(super) struct Gather;

impl<'a, Hidden: AxisName, QWidth: AxisName, Capacity: AxisName>
    Op<&'a ReplicatedScales<Hidden, QWidth, Capacity>> for Gather
{
    type Output = Scales<Hidden, QWidth, Capacity>;

    fn forward(
        &self,
        context: &mut Device,
        scales: &'a ReplicatedScales<Hidden, QWidth, Capacity>,
    ) -> Self::Output {
        // SAFETY: the same slices; a quarter's replicas are named by the slices they reach.
        let quarters = unsafe {
            scales.view().reshape::<Chip, Cluster<QWidth>,
                m![Hidden / 1024, Capacity / (Capacity / 4), Hidden % 256 / 16],
                m![Capacity % (Capacity / 4)]>()
        };
        let mut gathered = Scales::new();
        context.main.begin(quarters)
            .fetch::<m![Capacity % (Capacity / 4) / 16], m![Capacity % 16]>()
            .switch::<
                m![Hidden / 16],
                m![Capacity % (Capacity / 4) / 16, Capacity / (Capacity / 4)],
            >(
                SwitchConfig::Broadcast1 { slice1: 4, slice0: 16 })
            .collect::<m![Capacity % (Capacity / 4) / 16, Capacity / (Capacity / 4)],
                m![Capacity % 16]>()
            .commit_trim::<m![Capacity % 16]>()
            .commit_view(gathered.view_mut());
        gathered
    }
}

/// The partial sums of the first `waves` row groups times their row and channel scales.
pub(super) struct Rescale;

/// The partial sums, the row and channel scales, and the row groups a [`Rescale`] writes.
type Operands<'a, Hidden, QWidth, Capacity> = (
    &'a tiles::Partial<Hidden, QWidth, Capacity>,
    (
        &'a Scales<Hidden, QWidth, Capacity>,
        &'a Channels<Hidden, QWidth>,
    ),
    usize,
);

impl<'a, Hidden: AxisName, QWidth: AxisName, Capacity: AxisName>
    Op<Operands<'a, Hidden, QWidth, Capacity>> for Rescale
{
    type Output = tiles::Partial<Hidden, QWidth, Capacity>;

    fn forward(
        &self,
        context: &mut Device,
        (partial, (scales, channels), waves): Operands<'a, Hidden, QWidth, Capacity>,
    ) -> Self::Output {
        let row: VrfTensor<
            f32,
            Chip,
            Cluster<QWidth>,
            m![Hidden / 16],
            m![Capacity / (Capacity / 4), Capacity % (Capacity / 4)],
        > = context
            .sub
            .begin(scales.view())
            .fetch::<
                m![Capacity / (Capacity / 4), Capacity % (Capacity / 4) / 8],
                m![Capacity % 8],
            >()
            .fetch_cast::<f32>()
            .collect::<
                m![Capacity / (Capacity / 4), Capacity % (Capacity / 4) / 8],
                m![Capacity % 8],
            >()
            .to_vrf();
        let channel: VrfTensor<f32, Chip, Cluster<QWidth>, m![Hidden / 16], m![Hidden % 16]> =
            context
                .sub
                .begin(channels.view())
                .fetch::<m![Hidden % 16 / 8], m![Hidden % 8]>()
                .collect::<m![Hidden % 16 / 8], m![Hidden % 8]>()
                .to_vrf();
        let mut output = tiles::Partial::new();
        context
            .main
            .begin(partial.view())
            .fetch::<m![
                Capacity % (Capacity / 4) / 8,
                Capacity / (Capacity / 4),
                Capacity % 8,
                Hidden % 16 / 8
            ], m![Hidden % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![
                Capacity % (Capacity / 4) / 8,
                Capacity / (Capacity / 4),
                Capacity % 8,
                Hidden % 16 / 8
            ], m![Hidden % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![
                Capacity % (Capacity / 4) / 8,
                Capacity / (Capacity / 4),
                Capacity % 8,
                Hidden % 16 / 4
            ], m![Hidden % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &row)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &channel)
            .vector_widen_concat::<m![
                Capacity % (Capacity / 4) / 8,
                Capacity / (Capacity / 4),
                Capacity % 8,
                Hidden % 16 / 8
            ], m![Hidden % 8]>()
            .vector_final()
            .cast::<bf16, m![Hidden % 8 # 16]>()
            .commit_trim::<m![Hidden % 8]>()
            .commit_prefix::<m![Capacity % (Capacity / 4) / 8], _, _>(output.view_mut(), waves);
        output
    }
}
