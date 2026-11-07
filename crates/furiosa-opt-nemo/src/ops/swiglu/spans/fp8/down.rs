use super::*;

pub(super) fn panel_scale<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    scales: &Scale<Hidden, Intermediate, Rows>,
) -> PanelScale<Hidden, Intermediate, Rows> {
    context
        .sub
        .begin(scales.view())
        .fetch::<m![{ Groups<Rows> }], m![{ RowLanes<Rows> }]>()
        .fetch_cast::<f32>()
        .collect::<m![{ Groups<Rows> }], m![{ RowLanes<Rows> }]>()
        .to_vrf()
}

fn down_lanes<
    Hidden: AxisName,
    Intermediate: AxisName,
    Rows: AxisName,
    Fill: M,
    const W: usize,
    Element: M,
>(
    context: &mut Device,
    product: DmTensorView<
        '_,
        f8e4m3,
        Chip,
        Cluster<Intermediate>,
        DownSlice<Hidden, Intermediate>,
        Element,
    >,
    quarters: usize,
) -> Operand<Hidden, Intermediate, Rows, Fill, W> {
    context
        .sub
        .begin(product)
        .fetch::<m![{ RowLanes<Rows> }, { Fill } % { W } / 32,
            { Fill } / { W }, Rows % 32 / 8, Intermediate % 64 / 32], m![Intermediate % 32]>(
        )
        .collect::<m![{ RowLanes<Rows> }, { Fill } % { W } / 32,
            { Fill } / { W }, Rows % 32 / 8, Intermediate % 64 / 32], m![Intermediate % 32]>(
        )
        .to_trf_prefix::<m![{ Fill } % { W } / 32], _, _>(quarters)
}

fn project<
    Hidden: AxisName,
    Intermediate: AxisName,
    Rows: AxisName,
    Fill: M,
    const W: usize,
    Output: M,
>(
    context: &mut Device,
    (weight, lanes): (
        &Down<Hidden, Intermediate>,
        &Operand<Hidden, Intermediate, Rows, Fill, W>,
    ),
    (output, quarters): (
        DmTensorViewMut<'_, bf16, Chip, Cluster<Intermediate>, m![Hidden / 16], Output>,
        usize,
    ),
) {
    context
        .main
        .begin(weight.view())
        .fetch::<m![
            { Fill } % { W } / 32,
            Hidden % 16 / 4,
            { Fill } / { W },
            Hidden % 512 / 32,
            Hidden % 4,
            Intermediate % 64 / 32
        ], m![Hidden % 32 / 16, Intermediate % 32]>()
        .collect::<m![
            { Fill } % { W } / 32,
            Hidden % 16 / 4,
            { Fill } / { W },
            Hidden % 512 / 32,
            Hidden % 4,
            Intermediate % 64 / 32,
            Hidden % 32 / 16
        ], m![Intermediate % 32]>()
        .contract_outer::<m![
            { Fill } % { W } / 32,
            Hidden % 16 / 4,
            { Fill } / { W },
            Hidden % 512 / 32,
            Hidden % 4,
            Intermediate % 64 / 32,
            Rows % 32 / 8
        ], m![Hidden % 32 / 16, Intermediate % 32], _, _, _>(lanes)
        .contract_packet::<m![Hidden % 32 / 16]>()
        .contract_time::<m![
            { Fill } % { W } / 32,
            Hidden % 16 / 4,
            { Fill } / { W },
            Hidden % 512 / 32,
            Hidden % 4,
            Rows % 32 / 8
        ]>()
        .contract_lane::<m![
            { Fill } % { W } / 32,
            Hidden % 16 / 4,
            { Fill } / { W },
            Hidden % 512 / 32,
            Hidden % 4,
            Rows % 32 / 8,
            Hidden % 32 / 16
        ], m![{ RowLanes<Rows> } # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<m![Hidden / 16], m![
            { Fill } % { W } / 32,
            Hidden % 16 / 4,
            { Fill } / { W },
            Hidden % 4,
            Rows % 32 / 8
        ]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![{ RowLanes<Rows> } # 16]>()
        .transpose::<m![
            { Fill } % { W } / 32,
            Hidden % 16 / 4,
            { Fill } / { W },
            Rows % 32
        ], m![Hidden % 4 # 16]>()
        .commit_trim::<m![Hidden % 4]>()
        .commit_prefix::<m![{ Fill } % { W } / 32], _, _>(output, quarters);
}

pub(super) fn project_live<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    (weight, product): (
        &Down<Hidden, Intermediate>,
        &Activation<Hidden, Intermediate, Rows>,
    ),
    (live, rows): (usize, usize),
) -> Partial<Hidden, Intermediate, Rows> {
    let mut output: Partial<Hidden, Intermediate, Rows> = DmTensor::new();
    let quarters = (rows + 31) / 32;
    let mut lanes = down_lanes::<Hidden, Intermediate, Rows, m![Rows = { WAVE }], WAVE, _>(
        context,
        product
            .view()
            .tile::<m![Rows], WAVE, m![Rows = { WAVE } # Rows, Intermediate % 64]>(0)
            .unpad::<m![Rows = { WAVE }, Intermediate % 64]>(),
        quarters,
    );
    for pair in 0..(live + 1) / 2 {
        let upcoming = (pair + 1) % (Rows::SIZE / WAVE);
        let next = down_lanes::<Hidden, Intermediate, Rows, m![Rows = { WAVE }], WAVE, _>(
            context,
            product
                .view()
                .tile::<m![Rows], WAVE, m![Rows = { WAVE } # Rows, Intermediate % 64]>(
                    upcoming * WAVE,
                )
                .unpad::<m![Rows = { WAVE }, Intermediate % 64]>(),
            quarters - WAVE / 32 * (pair + 1),
        );
        project::<Hidden, Intermediate, Rows, m![Rows = { WAVE }], WAVE, _>(
            context,
            (weight, &lanes),
            (
                output
                    .view_mut()
                    .tile::<m![Rows], WAVE, m![Rows = { WAVE } #{!} Rows, Hidden % 16]>(
                        pair * WAVE,
                    ),
                quarters - WAVE / 32 * pair,
            ),
        );
        lanes = next;
    }
    output
}

pub(super) type ChannelScales<Hidden, Intermediate> =
    VrfTensor<f32, Chip, Cluster<Intermediate>, m![Hidden / 16], m![Hidden % 16]>;

pub(super) fn rescale<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    partial: &Partial<Hidden, Intermediate, Rows>,
    (row, channel): (
        &PanelScale<Hidden, Intermediate, Rows>,
        &ChannelScales<Hidden, Intermediate>,
    ),
    groups: usize,
) -> Partial<Hidden, Intermediate, Rows> {
    let mut output: Partial<Hidden, Intermediate, Rows> = DmTensor::new();
    context.main.begin(partial.view())
        .fetch::<m![{ Groups<Rows> }, { RowLanes<Rows> }, Hidden % 16 / 8], m![Hidden % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![{ Groups<Rows> }, { RowLanes<Rows> }, Hidden % 16 / 8], m![Hidden % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ Groups<Rows> }, { RowLanes<Rows> }, Hidden % 16 / 4], m![Hidden % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), row)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), channel)
        .vector_widen_concat::<m![{ Groups<Rows> }, { RowLanes<Rows> }, Hidden % 16 / 8], m![Hidden % 8]>()
        .vector_final().cast::<bf16, m![Hidden % 8 # 16]>().commit_trim::<m![Hidden % 8]>()
        .commit_prefix::<m![{ Groups<Rows> }], _, _>(output.view_mut(), groups);
    output
}

pub(super) fn sum<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    (left, right): (
        &Partial<Hidden, Intermediate, Rows>,
        &Partial<Hidden, Intermediate, Rows>,
    ),
    groups: usize,
) -> Partial<Hidden, Intermediate, Rows> {
    let mut output: Partial<Hidden, Intermediate, Rows> = DmTensor::new();
    context.main.begin_interleaved::<Term, _, _, _, _, _>(left.view(), right.view())
        .fetch::<m![{ Groups<Rows> }, Term, { RowLanes<Rows> }, Hidden % 16 / 8], m![Hidden % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![{ Groups<Rows> }, Term, { RowLanes<Rows> }, Hidden % 16 / 8], m![Hidden % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Term, m![{ Groups<Rows> }, { RowLanes<Rows> }, Hidden % 16 / 8]>()
        .vector_clip_zip(ClipBinaryOpF32::Add).vector_final().cast::<bf16, m![Hidden % 8 # 16]>()
        .commit_trim::<m![Hidden % 8]>()
        .commit_prefix::<m![{ Groups<Rows> }], _, _>(output.view_mut(), groups);
    output
}
