use super::*;
use crate::Op;
use core::marker::PhantomData;

pub(super) type Staged<Intermediate, Rows> = HbmTensor<
    f8e4m3,
    Chip,
    m![
        { Cluster<Intermediate> },
        { Blocks<Intermediate> },
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
        { Cluster<Intermediate> },
        Panels,
        Rows % (Rows / 8) / 16,
        Rows % 16 / 8,
        Rows / (Rows / 8),
        Rows % 8
    ],
>;

/// Panel `panel` of the quantized products, stored in HBM in row order.
pub(super) struct StagePanel;

impl<'a, Intermediate: AxisName, Rows: AxisName>
    Op<(
        &'a Spread<Intermediate, Rows, f8e4m3>,
        usize,
        &'a mut Staged<Intermediate, Rows>,
    )> for StagePanel
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (quantized, panel, stored): (
            &'a Spread<Intermediate, Rows, f8e4m3>,
            usize,
            &'a mut Staged<Intermediate, Rows>,
        ),
    ) {
        quantized
            .view()
            .tile::<m![Panels], 1, m![Panels = 1 # Panels, Rows % (Rows / 8), Intermediate % 64]>(
                panel,
            )
            .to_hbm_view(&mut context.tdma, stored.view_mut());
    }
}

/// A stored panel, written as the down projection's input.
pub(super) struct CopyPanel;

impl<'a, Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>
    Op<(
        &'a Staged<Intermediate, Rows>,
        &'a mut Activation<Hidden, Intermediate, Rows>,
    )> for CopyPanel
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (stored, quantized): (
            &'a Staged<Intermediate, Rows>,
            &'a mut Activation<Hidden, Intermediate, Rows>,
        ),
    ) {
        // SAFETY: the store's row digits are the span's rows in order.
        let rows = unsafe {
            stored.view().reshape::<Chip, m![
                { Cluster<Intermediate> },
                { Blocks<Intermediate> },
                Rows,
                Intermediate % 64
            ]>()
        };
        rows.to_dm_view(&mut context.tdma, quantized.view_mut());
    }
}

/// The row groups' scales, stored in HBM in row order.
pub(super) struct StageScales;

impl<'a, Intermediate: AxisName, Rows: AxisName>
    Op<(
        &'a SpreadScale<Intermediate, Rows>,
        &'a mut StagedScales<Intermediate, Rows>,
    )> for StageScales
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (groups, stored): (
            &'a SpreadScale<Intermediate, Rows>,
            &'a mut StagedScales<Intermediate, Rows>,
        ),
    ) {
        groups
            .view()
            .slice_tile::<m![Copies], 1, m![Rows / (Rows / 8), Copies = 1 # Copies]>(0)
            .to_hbm_view(&mut context.tdma, stored.view_mut());
    }
}

/// Panel `panel`'s stored scales, written as the down projection's row scales.
pub(super) struct CopyScales;

impl<'a, Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>
    Op<(
        &'a StagedScales<Intermediate, Rows>,
        usize,
        &'a mut Scale<Hidden, Intermediate, Rows>,
    )> for CopyScales
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (stored, panel, scales): (
            &'a StagedScales<Intermediate, Rows>,
            usize,
            &'a mut Scale<Hidden, Intermediate, Rows>,
        ),
    ) {
        // SAFETY: the store's row digits are the span's rows in order.
        let rows = unsafe {
            stored
                .view()
                .reshape::<Chip, m![{ Cluster<Intermediate> }, Panels, Rows]>()
        };
        rows.tile::<m![Panels], 1, m![{ Cluster<Intermediate> }, Panels = 1 # Panels, Rows]>(panel)
            .to_dm_view(&mut context.tdma, scales.view_mut());
    }
}

pub(super) type Stores<Intermediate, Rows> = m![
    { Cluster<Intermediate> },
    Intermediate % (Intermediate / 2) / 64,
    Rows % (Rows / 8) / 16,
    Rows % 16 / 8,
    Rows / (Rows / 8),
    Rows % 8,
    Intermediate % 64
];

/// The gated products of the first `rows` rows, stored in HBM by waves.
pub(super) struct StoreTiles;

impl<'a, Intermediate: AxisName, Rows: AxisName>
    Op<(
        &'a Halves<Intermediate, Rows>,
        usize,
        &'a mut HbmTensor<bf16, Chip, Stores<Intermediate, Rows>>,
    )> for StoreTiles
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (packed, rows, stored): (
            &'a Halves<Intermediate, Rows>,
            usize,
            &'a mut HbmTensor<bf16, Chip, Stores<Intermediate, Rows>>,
        ),
    ) {
        // SAFETY: split the contiguous owner/panel digit without moving any row or feature.
        let packed = packed.view();
        let stored = unsafe {
            stored.view_mut().reshape::<Chip, ProductStore<
                Intermediate,
                Rows,
                m![{Out<Intermediate>} / 64],
                m![Intermediate % 64],
            >>()
        };
        store_rows(context, packed, stored, rows);
    }
}

/// A product store's owner, optional panel digit, canonical row digits and local features.
/// The panel digit is one for a native-width owner, or three for Qwen's packed 64-feature bands.
pub(super) type ProductStore<I, R, Outer, Inner> = m![
    {Cluster<I>}, {Blocks<I>}, {Outer},
    R % (R / 8) / 16, R % 16 / 8, R / (R / 8), R % 8, {Inner}
];

/// Panel `panel` of the stored products of a span of at most two waves, written as the down
/// projection's input: the first rows' power-of-two prefix.
pub(super) struct LoadProduct<Hidden>(pub(super) PhantomData<Hidden>);

impl<'a, Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>
    Op<(
        &'a HbmTensor<bf16, Chip, Chunks<Intermediate, Rows>>,
        usize,
        (usize, usize),
    )> for LoadProduct<Hidden>
{
    type Output = Product<Hidden, Intermediate, Rows>;

    fn forward(
        &self,
        context: &mut Device,
        (product, panel, (live, rows)): (
            &'a HbmTensor<bf16, Chip, Chunks<Intermediate, Rows>>,
            usize,
            (usize, usize),
        ),
    ) -> Self::Output {
        let mut output = Product::<Hidden, Intermediate, Rows>::new();
        // SAFETY: the same order with each chunk split into its slice and panel.
        let chunks = unsafe {
            product
                .view()
                .reshape::<Chip, ChunkSlices<Intermediate, Rows>>()
        };
        let panel =
            chunks.tile::<m![{ Out<Intermediate> } / 64], 1, ChunkPanel<Intermediate, Rows>>(panel);
        if live <= 2 {
            // SAFETY: grouping the leading owner digits preserves their physical strides.
            let panel = unsafe {
                panel.reshape::<Chip, m![
                {m![{Cluster<Intermediate>}, {Blocks<Intermediate>}, {Out<Intermediate>} / 64 = 1 # ({Out<Intermediate>} / 64)]},
                Rows, Intermediate % 64]>()
            };
            load_rows(context, panel, output.view_mut(), rows);
        }
        output
    }
}

/// Qwen's live-row transfer windows, shared by packed panels and native-width products.
pub(super) fn store_rows<Intermediate: AxisName, Rows: AxisName, Outer: M, Inner: M>(
    context: &mut Device,
    packed: DmTensorView<
        '_,
        bf16,
        Chip,
        Cluster<Intermediate>,
        Slice<Intermediate, Rows>,
        m![
            { Outer },
            Rows % (Rows / 8) / 16,
            Rows % 16 / 8,
            Rows % 8,
            { Inner }
        ],
    >,
    stored: HbmTensorViewMut<'_, bf16, Chip, ProductStore<Intermediate, Rows, Outer, Inner>>,
    rows: usize,
) {
    if rows > 128 {
        packed.tile::<m![Rows % (Rows / 8) / 16], 2,
                m![{Outer}, Rows % (Rows / 8) / 16 = 2 # (Rows / 128),
                    Rows % 16 / 8, Rows % 8, {Inner}]>(0)
                .to_hbm_view(&mut context.tdma, stored
                    .tile::<m![Rows % (Rows / 8) / 16], 2, m![{ Cluster<Intermediate> },
                        {Blocks<Intermediate>}, {Outer},
                        Rows % (Rows / 8) / 16 = 2 #{!} (Rows / 128), Rows % 16 / 8,
                        Rows / (Rows / 8), Rows % 8, {Inner}]>(0));
    } else {
        let packed = packed.tile::<m![Rows % (Rows / 8) / 16], 1,
                m![{Outer}, Rows % (Rows / 8) / 16 = 1 # (Rows / 128),
                    Rows % 16 / 8, Rows % 8, {Inner}]>(0);
        let wave = stored.tile::<m![Rows % (Rows / 8) / 16], 1, m![
            { Cluster<Intermediate> },
            {Blocks<Intermediate>}, {Outer},
            Rows % (Rows / 8) / 16 = 1 #{!} (Rows / 128),
            Rows % 16 / 8,
            Rows / (Rows / 8),
            Rows % 8,
            {Inner}
        ]>(0);
        if rows > 64 {
            packed.to_hbm_view(&mut context.tdma, wave);
        } else {
            let packed = packed
                .tile::<m![Rows % 16 / 8], 1, m![{Outer}, Rows % (Rows / 8) / 16 = 1 # (Rows / 128),
                        Rows % 16 / 8 = 1 # 2, Rows % 8, {Inner}]>(0);
            let half = wave.tile::<m![Rows % 16 / 8], 1, m![{ Cluster<Intermediate> },
                        {Blocks<Intermediate>}, {Outer},
                        Rows % (Rows / 8) / 16 = 1 #{!} (Rows / 128), Rows % 16 / 8 = 1 #{!} 2,
                        Rows / (Rows / 8), Rows % 8, {Inner}]>(0);
            if rows > 8 {
                packed.to_hbm_view(&mut context.tdma, half);
            } else {
                packed.slice_tile::<m![Rows / (Rows / 8)], 1,
                        m![{ Blocks<Intermediate> }, Rows / (Rows / 8) = 1 # 8]>(0)
                        .to_hbm_view(&mut context.tdma, half.tile::<m![Rows / (Rows / 8)], 1,
                            m![{ Cluster<Intermediate> }, {Blocks<Intermediate>}, {Outer},
                                Rows % (Rows / 8) / 16 = 1 #{!} (Rows / 128),
                                Rows % 16 / 8 = 1 #{!} 2, Rows / (Rows / 8) = 1 #{!} 8,
                                Rows % 8, {Inner}]>(0));
            }
        }
    }
}
/// Load only the selected power-of-two row prefix, preserving source padding and destination
/// ownership. Feature axes may differ in stride (e.g. the native 32-wide tail of a 224-wide owner).
pub(super) fn load_rows<
    R: AxisName,
    D: MaterializableScalar,
    Before: M,
    From: M,
    C: M,
    S: M,
    To: M,
>(
    device: &mut Device,
    source: HbmTensorView<'_, D, Chip, m![{ Before }, R, { From }]>,
    output: DmTensorViewMut<'_, D, Chip, C, S, m![R, { To }]>,
    rows: usize,
) {
    if rows > 128 {
        load_prefix::<R, 256, _, _, _, _, _, _>(device, source, output);
    } else if rows > 64 {
        load_prefix::<R, 128, _, _, _, _, _, _>(device, source, output);
    } else if rows > 8 {
        load_prefix::<R, 64, _, _, _, _, _, _>(device, source, output);
    } else {
        load_prefix::<R, 8, _, _, _, _, _, _>(device, source, output);
    }
}
pub(super) fn load_prefix<
    R: AxisName,
    const N: usize,
    D: MaterializableScalar,
    Before: M,
    From: M,
    C: M,
    S: M,
    To: M,
>(
    device: &mut Device,
    source: HbmTensorView<'_, D, Chip, m![{ Before }, R, { From }]>,
    output: DmTensorViewMut<'_, D, Chip, C, S, m![R, { To }]>,
) {
    source
        .tile::<m![R], N, m![{Before}, R = {N} # R, {From}]>(0)
        .to_dm_view(
            &mut device.tdma,
            output.tile::<m![R], N, m![R = {N} #{!} R, {To}]>(0),
        );
}
