use super::*;

pub(super) type Staged<Intermediate, Rows> = HbmTensor<
    f8e4m3,
    Chip,
    m![
        Intermediate / 6144,
        Intermediate % 6144 / 192,
        Rows % (Rows / 8) / 16,
        Rows % 16 / 8,
        Rows / (Rows / 8),
        Rows % 8,
        Intermediate % 64
    ],
>;
pub(super) type StagedScales<Intermediate, Rows> = HbmTensor<
    bf16,
    Chip,
    m![
        Intermediate / 6144,
        Panels,
        Rows % (Rows / 8) / 16,
        Rows % 16 / 8,
        Rows / (Rows / 8),
        Rows % 8
    ],
>;

pub(super) fn stage_panel<Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    quantized: &Spread<Intermediate, Rows, f8e4m3>,
    panel: usize,
    stored: &mut Staged<Intermediate, Rows>,
) {
    quantized
        .view()
        .tile::<m![Panels], 1, m![Panels = 1 # 3, Rows % (Rows / 8), Intermediate % 64]>(panel)
        .to_hbm_view(&mut context.tdma, stored.view_mut());
}

pub(super) fn copy_panel<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    stored: &Staged<Intermediate, Rows>,
    quantized: &mut Activation<Hidden, Intermediate, Rows>,
) {
    // SAFETY: the store's row digits are the span's rows in order.
    let rows = unsafe {
        stored.view().reshape::<Chip, m![
            Intermediate / 6144,
            Intermediate % 6144 / 192,
            Rows,
            Intermediate % 64
        ]>()
    };
    rows.to_dm_view(&mut context.tdma, quantized.view_mut());
}

pub(super) fn stage_scales<Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    groups: &SpreadScale<Intermediate, Rows>,
    stored: &mut StagedScales<Intermediate, Rows>,
) {
    groups
        .view()
        .slice_tile::<m![Copies], 1, m![Rows / (Rows / 8), Copies = 1 # 32]>(0)
        .to_hbm_view(&mut context.tdma, stored.view_mut());
}

pub(super) fn copy_scales<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    stored: &StagedScales<Intermediate, Rows>,
    panel: usize,
    scales: &mut Scale<Hidden, Intermediate, Rows>,
) {
    // SAFETY: the store's row digits are the span's rows in order.
    let rows = unsafe {
        stored
            .view()
            .reshape::<Chip, m![Intermediate / 6144, Panels, Rows]>()
    };
    rows.tile::<m![Panels], 1, m![Intermediate / 6144, Panels = 1 # 3, Rows]>(panel)
        .to_dm_view(&mut context.tdma, scales.view_mut());
}

pub(super) type Stores<Intermediate, Rows> = m![
    Intermediate / 6144,
    Intermediate % 6144 / 64,
    Rows % (Rows / 8) / 16,
    Rows % 16 / 8,
    Rows / (Rows / 8),
    Rows % 8,
    Intermediate % 64
];

pub(super) fn store_waves<Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    packed: &Halves<Intermediate, Rows>,
    rows: usize,
    stored: &mut HbmTensor<bf16, Chip, Stores<Intermediate, Rows>>,
) {
    if rows > 128 {
        packed.view().tile::<m![Rows % (Rows / 8) / 16], 2,
            m![Intermediate % 192 / 64, Rows % (Rows / 8) / 16 = 2 # (Rows / 128),
                Rows % 16 / 8, Rows % 8, Intermediate % 64]>(0)
            .to_hbm_view(&mut context.tdma, stored.view_mut()
                .tile::<m![Rows % (Rows / 8) / 16], 2, m![Intermediate / 6144,
                    Intermediate % 6144 / 64, Rows % (Rows / 8) / 16 = 2 #{!} (Rows / 128),
                    Rows % 16 / 8, Rows / (Rows / 8), Rows % 8, Intermediate % 64]>(0));
    } else {
        let packed = packed.view().tile::<m![Rows % (Rows / 8) / 16], 1,
            m![Intermediate % 192 / 64, Rows % (Rows / 8) / 16 = 1 # (Rows / 128),
                Rows % 16 / 8, Rows % 8, Intermediate % 64]>(0);
        let wave = stored
            .view_mut()
            .tile::<m![Rows % (Rows / 8) / 16], 1, m![Intermediate / 6144, Intermediate % 6144 / 64,
                Rows % (Rows / 8) / 16 = 1 #{!} (Rows / 128), Rows % 16 / 8,
                Rows / (Rows / 8), Rows % 8, Intermediate % 64]>(0);
        if rows > 64 {
            packed.to_hbm_view(&mut context.tdma, wave);
        } else {
            let packed = packed.tile::<m![Rows % 16 / 8], 1,
                m![Intermediate % 192 / 64, Rows % (Rows / 8) / 16 = 1 # (Rows / 128),
                    Rows % 16 / 8 = 1 # 2, Rows % 8, Intermediate % 64]>(0);
            let half = wave
                .tile::<m![Rows % 16 / 8], 1, m![Intermediate / 6144, Intermediate % 6144 / 64,
                    Rows % (Rows / 8) / 16 = 1 #{!} (Rows / 128), Rows % 16 / 8 = 1 #{!} 2,
                    Rows / (Rows / 8), Rows % 8, Intermediate % 64]>(0);
            if rows > 8 {
                packed.to_hbm_view(&mut context.tdma, half);
            } else {
                packed.slice_tile::<m![Rows / (Rows / 8)], 1,
                    m![Intermediate % 6144 / 192, Rows / (Rows / 8) = 1 # 8]>(0)
                    .to_hbm_view(&mut context.tdma, half.tile::<m![Rows / (Rows / 8)], 1,
                        m![Intermediate / 6144, Intermediate % 6144 / 64,
                            Rows % (Rows / 8) / 16 = 1 #{!} (Rows / 128),
                            Rows % 16 / 8 = 1 #{!} 2, Rows / (Rows / 8) = 1 #{!} 8,
                            Rows % 8, Intermediate % 64]>(0));
            }
        }
    }
}

pub(super) fn load_product<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    product: &HbmTensor<bf16, Chip, Chunks<Intermediate, Rows>>,
    panel: usize,
    (live, rows): (usize, usize),
) -> Product<Hidden, Intermediate, Rows> {
    let mut output = Product::<Hidden, Intermediate, Rows>::new();
    // SAFETY: the same order with each chunk split into its slice and panel.
    let chunks = unsafe {
        product
            .view()
            .reshape::<Chip, ChunkSlices<Intermediate, Rows>>()
    };
    let panel =
        chunks.tile::<m![Intermediate % 192 / 64], 1, ChunkPanel<Intermediate, Rows>>(panel);
    if live > 2 {
    } else if rows > 128 {
        load_prefix::<Hidden, Intermediate, Rows, 256>(context, panel, &mut output);
    } else if rows > 64 {
        load_prefix::<Hidden, Intermediate, Rows, 128>(context, panel, &mut output);
    } else if rows > 8 {
        load_prefix::<Hidden, Intermediate, Rows, 64>(context, panel, &mut output);
    } else {
        load_prefix::<Hidden, Intermediate, Rows, 8>(context, panel, &mut output);
    }
    output
}

fn load_prefix<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName, const LEN: usize>(
    context: &mut Device,
    panel: HbmTensorView<'_, bf16, Chip, ChunkPanel<Intermediate, Rows>>,
    output: &mut Product<Hidden, Intermediate, Rows>,
) {
    panel
        .tile::<m![Rows], LEN, m![Intermediate / 6144, Intermediate % 6144 / 192,
        Intermediate % 192 / 64 = 1 # 3, Rows = { LEN } # Rows, Intermediate % 64]>(0)
        .to_dm_view(
            &mut context.tdma,
            output
                .view_mut()
                .tile::<m![Rows], LEN, m![Rows = { LEN } #{!} Rows, Intermediate % 64]>(0),
        );
}
