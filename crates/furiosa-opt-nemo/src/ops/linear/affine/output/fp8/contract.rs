use super::*;

pub(super) fn resident<Hidden: AxisName, QWidth: AxisName>(
    context: &mut Device,
    weight: &Weight8<Hidden, QWidth>,
) -> Resident<Hidden, QWidth> {
    context
        .sub
        .begin(weight.view())
        .fetch::<m![Hidden % 8, Hidden % 256 / 8, QWidth % 128 / 32], m![QWidth % 32]>()
        .collect::<m![Hidden % 8, Hidden % 256 / 8, QWidth % 128 / 32], m![QWidth % 32]>()
        .to_trf()
}

pub(super) fn contract<Hidden: AxisName, QWidth: AxisName, Capacity: AxisName>(
    context: &mut Device,
    (weight, input): (
        &Resident<Hidden, QWidth>,
        &Source8<Hidden, QWidth, Capacity>,
    ),
    (output, offset): (&mut waves::Partial<Hidden, QWidth, Capacity>, usize),
) {
    context.main.begin(input.view().tile::<m![Capacity % (Capacity / 4) / 8], 1,
        m![Capacity % (Capacity / 4) / 8 = 1 # (Capacity / 32), Capacity % 8, QWidth % 128]>(
        offset))
        .fetch::<m![Capacity % 8, QWidth % 128 / 64], m![QWidth % 64]>()
        .switch::<Resident8<Hidden, QWidth>, m![Capacity % 8, QWidth % 128 / 64, Capacity / (Capacity / 4)]>(
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

pub(super) fn contract_wave<Hidden: AxisName, QWidth: AxisName, Capacity: AxisName>(
    context: &mut Device,
    (weight, input): (
        &Resident<Hidden, QWidth>,
        &Source8<Hidden, QWidth, Capacity>,
    ),
    (output, offset): (&mut waves::Partial<Hidden, QWidth, Capacity>, usize),
) {
    context.main.begin(input.view().tile::<m![Capacity % (Capacity / 4) / 32], 1,
        m![Capacity % (Capacity / 4) / 32 = 1 # 4, Capacity % 32, QWidth % 128]>(offset))
        .fetch::<m![Capacity % 32, QWidth % 128 / 64], m![QWidth % 64]>()
        .switch::<Resident8<Hidden, QWidth>, m![Capacity % 32, QWidth % 128 / 64, Capacity / (Capacity / 4)]>(
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
