use super::*;
use crate::Op;

/// The staged weight resident in the TRF.
pub(super) struct Stage;

impl<Hidden: AxisName, QWidth: AxisName> Op<&Weight<Hidden, QWidth>> for Stage {
    type Output = Resident<Hidden, QWidth>;

    fn forward(&self, context: &mut Device, weight: &Weight<Hidden, QWidth>) -> Self::Output {
        context
            .sub
            .begin(weight.view())
            .fetch::<m![Hidden % 8, Hidden % 256 / 8, QWidth % 128 / 32], m![QWidth % 32]>()
            .collect::<m![Hidden % 8, Hidden % 256 / 8, QWidth % 128 / 32], m![QWidth % 32]>()
            .to_trf()
    }
}

/// The contraction of `ROWS` rows at row group `offset` into the partial sums.
pub(super) struct Pass<const ROWS: usize>;

/// A [`Pass`]'s resident weight and quantized rows, and the partial sums with the row group.
type Operands<'a, Hidden, QWidth, Capacity> = (
    (
        &'a Resident<Hidden, QWidth>,
        &'a Source<Hidden, QWidth, Capacity>,
    ),
    (&'a mut tiles::Partial<Hidden, QWidth, Capacity>, usize),
);

impl<'a, Hidden: AxisName, QWidth: AxisName, Capacity: AxisName>
    Op<Operands<'a, Hidden, QWidth, Capacity>> for Pass<8>
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        ((weight, input), (output, offset)): Operands<'a, Hidden, QWidth, Capacity>,
    ) {
        context.main.begin(input.view().tile::<m![Capacity % (Capacity / 4) / 8], 1,
            m![Capacity % (Capacity / 4) / 8 = 1 # (Capacity / 32), Capacity % 8, QWidth % 128]>(
            offset))
            .fetch::<m![Capacity % 8, QWidth % 128 / 64], m![QWidth % 64]>()
            .switch::<
                WeightSlices<Hidden, QWidth>,
                m![Capacity % 8, QWidth % 128 / 64, Capacity / (Capacity / 4)],
            >(
                SwitchConfig::Broadcast1 { slice1: 4, slice0: 16 })
            .collect::<m![Capacity % 8, QWidth % 128 / 64,
                Capacity / (Capacity / 4), QWidth % 64 / 32], m![QWidth % 32]>()
            .contract_outer::<m![Capacity % 8, QWidth % 128 / 64,
                Capacity / (Capacity / 4), Hidden % 256 / 8], m![QWidth % 64], _, _, _>(weight)
            .contract_packet::<m![1]>()
            .contract_time::<m![Capacity % 8, Capacity / (Capacity / 4), Hidden % 256 / 8]>()
            .contract_lane::<m![Capacity % 8, Capacity / (Capacity / 4), Hidden % 256 / 8],
                m![Hidden % 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<m![Hidden / 16],
                m![Capacity % 8, Capacity / (Capacity / 4), Hidden % 16 / 8]>(
                InterSliceReduceOpF32::Add)
            .vector_final().cast::<bf16, m![Hidden % 8 # 16]>()
            .commit_trim::<m![Hidden % 8]>()
            .commit_view(output.view_mut().tile::<m![Capacity % (Capacity / 4) / 8], 1,
                m![Capacity % (Capacity / 4) / 8 = 1 #{!} (Capacity / 32),
                    Capacity / (Capacity / 4), Capacity % 8, Hidden % 16]>(offset));
    }
}

impl<'a, Hidden: AxisName, QWidth: AxisName, Capacity: AxisName>
    Op<Operands<'a, Hidden, QWidth, Capacity>> for Pass<32>
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        ((weight, input), (output, offset)): Operands<'a, Hidden, QWidth, Capacity>,
    ) {
        context.main.begin(input.view().tile::<m![Capacity % (Capacity / 4) / 32], 1,
            m![Capacity % (Capacity / 4) / 32 = 1 # 4, Capacity % 32, QWidth % 128]>(offset))
            .fetch::<m![Capacity % 32, QWidth % 128 / 64], m![QWidth % 64]>()
            .switch::<
                WeightSlices<Hidden, QWidth>,
                m![Capacity % 32, QWidth % 128 / 64, Capacity / (Capacity / 4)],
            >(
                SwitchConfig::Broadcast1 { slice1: 4, slice0: 16 })
            .collect::<m![Capacity % 32, QWidth % 128 / 64,
                Capacity / (Capacity / 4), QWidth % 64 / 32], m![QWidth % 32]>()
            .contract_outer::<m![Capacity % 32, QWidth % 128 / 64,
                Capacity / (Capacity / 4), Hidden % 256 / 8], m![QWidth % 64], _, _, _>(weight)
            .contract_packet::<m![1]>()
            .contract_time::<m![Capacity % 32, Capacity / (Capacity / 4), Hidden % 256 / 8]>()
            .contract_lane::<m![Capacity % 32, Capacity / (Capacity / 4), Hidden % 256 / 8],
                m![Hidden % 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<m![Hidden / 16],
                m![Capacity % 32, Capacity / (Capacity / 4), Hidden % 16 / 8]>(
                InterSliceReduceOpF32::Add)
            .vector_final().cast::<bf16, m![Hidden % 8 # 16]>()
            .commit_trim::<m![Hidden % 8]>()
            .commit_view(output.view_mut().tile::<m![Capacity % (Capacity / 4) / 32], 1,
                m![Capacity % (Capacity / 4) / 32 = 1 #{!} 4, Capacity % 32 / 8,
                    Capacity / (Capacity / 4), Capacity % 8, Hidden % 16]>(offset));
    }
}
