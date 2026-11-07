use super::*;

pub(super) fn channel_scale<Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    scale: &Channels<Intermediate, Rows>,
) -> ChannelScale<Intermediate, Rows> {
    context
        .sub
        .begin(scale.view())
        .fetch::<m![{ Out<Intermediate> } / 8], m![{ Out<Intermediate> } % 8]>()
        .collect::<m![{ Out<Intermediate> } / 8], m![{ Out<Intermediate> } % 8]>()
        .to_vrf()
}

pub(super) fn input_scales<Hidden: AxisName, Rows: AxisName>(
    context: &mut Device,
    rows: &Owned<Hidden, Rows>,
) -> Scales<Rows> {
    let mut scales: DmTensor<bf16, Chip, m![Rows / (Rows / 2)], m![1 # 256], Labels<Rows>> =
        DmTensor::new();
    context
        .main
        .begin(rows.view())
        .fetch::<m![Rows % (Rows / 2), Hidden % 16 / 8], m![Hidden % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Rows % (Rows / 2), Hidden % 16 / 8], m![Hidden % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Rows % (Rows / 2), Hidden % 16 / 4], m![Hidden % 4]>()
        .vector_fp_binary_with_mode(
            FpBinaryOp::MulF(FpMulAlu::Mul0),
            BinaryArgMode::Mode00,
            0.0f32,
        )
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_intra_slice_reduce::<Hidden, m![Rows % (Rows / 2)], m![1 # 4]>(
            IntraSliceReduceOpF32::Max,
        )
        .vector_fp_div(LARGEST)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, SMALLEST)
        .vector_inter_slice_reduce::<m![1 # 256], m![Rows % (Rows / 2)]>(InterSliceReduceOpF32::Max)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Rows % (Rows / 2) / 4], m![Rows % 4 # 16]>()
        .commit_trim::<m![Rows % 4]>()
        .commit_view(scales.view_mut());
    let mut stored: Scales<Rows> = HbmTensor::new();
    // SAFETY: the stored labels preserve the row order consumed by each gate/up wave.
    let labelled = unsafe {
        stored.view_mut().reshape::<Chip, m![
            Rows % (Rows / 8) / 8,
            Rows / (Rows / 4),
            Rows % (Rows / 4) / (Rows / 8),
            Rows % 8
        ]>()
    };
    scales.view().to_hbm_view(&mut context.tdma, labelled);
    stored
}

pub(super) fn row_scales<Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    scales: &Scales<Rows>,
) -> DmTensor<bf16, Chip, Cluster<Intermediate>, Slice<Intermediate, Rows>, m![Rows % (Rows / 8)]> {
    scales.to_dm(&mut context.tdma)
}

pub(super) fn row_scale<Intermediate: AxisName, Rows: AxisName, S: M>(
    context: &mut Device,
    scales: DmTensorView<'_, bf16, Chip, Cluster<Intermediate>, S, m![Rows % (Rows / 8)]>,
) -> VrfTensor<f32, Chip, Cluster<Intermediate>, S, m![Rows % (Rows / 8)]> {
    context
        .sub
        .begin(scales)
        .fetch::<m![Rows % (Rows / 8) / 8], m![Rows % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Rows % (Rows / 8) / 8], m![Rows % 8]>()
        .to_vrf()
}

type Wave<Hidden, Intermediate, Rows> = DmTensor<
    bf16,
    Chip,
    Cluster<Intermediate>,
    m![Hidden % 1024 / 32, Rows / (Rows / 8)],
    m![
        Hidden / 1024,
        Rows % (Rows / 8) / 16 = 1,
        Rows % 16,
        Hidden % 32
    ],
>;

pub(super) fn waves<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    input: &gate::Normalized<Hidden, Rows, Chip>,
    (live, rows): (usize, usize),
) -> [Wave<Hidden, Intermediate, Rows>; 4] {
    // SAFETY: wave, half, group and lane digits preserve the stored rows' coordinates.
    let view = unsafe {
        input.view().reshape::<Chip, m![
            Rows % (Rows / 8) / 16,
            Rows % 16 / 8,
            Rows / (Rows / 8),
            Hidden / 32,
            Rows % 8,
            Hidden % 32
        ]>()
    };
    let mut waves = [Wave::new(), Wave::new(), Wave::new(), Wave::new()];
    if rows <= 8 {
        view.tile::<m![Rows % (Rows / 8) / 16], 1, m![Rows % (Rows / 8) / 16 = 1 # (Rows / 128),
            Rows % 16 / 8, Rows / (Rows / 8), Hidden / 32, Rows % 8, Hidden % 32]>(0)
            .tile::<m![Rows % 16 / 8], 1, m![Rows % (Rows / 8) / 16 = 1 # (Rows / 128),
                Rows % 16 / 8 = 1 # 2, Rows / (Rows / 8), Hidden / 32, Rows % 8, Hidden % 32]>(0)
            .tile::<m![Rows / (Rows / 8)], 1, m![Rows % (Rows / 8) / 16 = 1 # (Rows / 128),
                Rows % 16 / 8 = 1 # 2, Rows / (Rows / 8) = 1 # 8, Hidden / 32, Rows % 8,
                Hidden % 32]>(0)
            .to_dm_view(&mut context.tdma, waves[0].view_mut()
                .slice_tile::<m![Rows / (Rows / 8)], 1,
                    m![Hidden % 1024 / 32, Rows / (Rows / 8) = 1 #{!} 8]>(0)
                .tile::<m![Rows % 16 / 8], 1, m![Hidden / 1024, Rows % (Rows / 8) / 16 = 1,
                    Rows % 16 / 8 = 1 #{!} 2, Rows % 8, Hidden % 32]>(0));
    } else {
        land::<Hidden, Intermediate, Rows>(context, &view, (&mut waves[0], 0));
        if live > 1 {
            land::<Hidden, Intermediate, Rows>(context, &view, (&mut waves[1], 1));
            if live > 2 {
                land::<Hidden, Intermediate, Rows>(context, &view, (&mut waves[2], 2));
                land::<Hidden, Intermediate, Rows>(context, &view, (&mut waves[3], 3));
            }
        }
    }
    waves
}

fn land<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    stored: &HbmTensorView<
        '_,
        bf16,
        Chip,
        m![
            Rows % (Rows / 8) / 16,
            Rows % 16 / 8,
            Rows / (Rows / 8),
            Hidden / 32,
            Rows % 8,
            Hidden % 32
        ],
    >,
    (output, wave): (&mut Wave<Hidden, Intermediate, Rows>, usize),
) {
    stored.tile::<m![Rows % (Rows / 8) / 16], 1,
        m![Rows % (Rows / 8) / 16 = 1 # (Rows / 128), Rows % 16 / 8,
            Rows / (Rows / 8), Hidden / 32, Rows % 8, Hidden % 32]>(wave)
        .to_dm_view(&mut context.tdma, output.view_mut());
}

type WaveScale<Hidden, Intermediate, Rows> = VrfTensor<
    f32,
    Chip,
    Cluster<Intermediate>,
    m![Hidden % 1024 / 32, Rows / (Rows / 8)],
    m![Rows % 16],
>;

pub(super) fn wave_scale<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    scales: &DmTensor<
        bf16,
        Chip,
        Cluster<Intermediate>,
        m![Hidden % 1024 / 32, Rows / (Rows / 8)],
        m![Rows % (Rows / 8)],
    >,
    wave: usize,
) -> WaveScale<Hidden, Intermediate, Rows> {
    context.sub.begin(scales.view().tile::<m![Rows % (Rows / 8) / 16], 1,
        m![Rows % (Rows / 8) / 16 = 1 # (Rows / 128), Rows % 16]>(wave))
        .fetch::<m![Rows % 16 / 8], m![Rows % 8]>().fetch_cast::<f32>()
        .collect::<m![Rows % 16 / 8], m![Rows % 8]>().to_vrf()
}

pub(super) fn quantize<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    input: &Wave<Hidden, Intermediate, Rows>,
    (scale, wave): (&WaveScale<Hidden, Intermediate, Rows>, usize),
    output: &mut Input<Hidden, Intermediate, Rows>,
) {
    context.main.begin(input.view())
        .fetch::<m![Hidden / 1024, Rows % 16, Hidden % 32 / 8], m![Hidden % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Hidden / 1024, Rows % 16, Hidden % 32 / 8], m![Hidden % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Hidden / 1024, Rows % 16, Hidden % 32 / 4], m![Hidden % 4]>()
        .vector_fp_div(scale)
        .vector_widen_concat::<m![Hidden / 1024, Rows % 16, Hidden % 32 / 8], m![Hidden % 8]>()
        .vector_final().cast::<f8e4m3, m![Hidden % 8 # 32]>().commit_trim::<m![Hidden % 8]>()
        .commit_view(output.view_mut().tile::<m![Rows % (Rows / 8) / 16], 1,
            m![Hidden / 1024, Rows % (Rows / 8) / 16 = 1 #{!} (Rows / 128),
                Rows % 16, Hidden % 32]>(wave));
}

pub(super) fn lanes<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    input: &Input<Hidden, Intermediate, Rows>,
    (panel, wave): (usize, usize),
) -> Lanes<Hidden, Intermediate, Rows> {
    context.sub.begin(input.view()
        .tile::<m![Hidden / 1024], 1, m![Hidden / 1024 = 1 # 4, Rows % (Rows / 8), Hidden % 32]>(panel)
        .tile::<m![Rows % (Rows / 8) / 16], 1, m![Hidden / 1024 = 1 # 4,
            Rows % (Rows / 8) / 16 = 1 # (Rows / 128), Rows % 16, Hidden % 32]>(wave))
        .fetch::<m![Rows % 8, Rows % 16 / 8], m![Hidden % 32]>()
        .switch::<Partition<Hidden, Intermediate>,
            m![Rows % 8, Rows % 16 / 8, Hidden % 128 / 32, Rows / (Rows / 8)]>(
            SwitchConfig::TransposedBroadcast1 { slice1: 8, slice0: 32 })
        .collect::<m![Rows % 8, Rows % 16 / 8, Hidden % 128 / 32, Rows / (Rows / 8)],
            m![Hidden % 32]>().to_trf()
}
