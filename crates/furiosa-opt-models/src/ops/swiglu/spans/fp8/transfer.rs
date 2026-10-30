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
        if rows > 128 {
            packed.view().tile::<m![Rows % (Rows / 8) / 16], 2,
                m![{ Out<Intermediate> } / 64, Rows % (Rows / 8) / 16 = 2 # (Rows / 128),
                    Rows % 16 / 8, Rows % 8, Intermediate % 64]>(0)
                .to_hbm_view(&mut context.tdma, stored.view_mut()
                    .tile::<m![Rows % (Rows / 8) / 16], 2, m![{ Cluster<Intermediate> },
                        Intermediate % (Intermediate / 2) / 64,
                        Rows % (Rows / 8) / 16 = 2 #{!} (Rows / 128), Rows % 16 / 8,
                        Rows / (Rows / 8), Rows % 8, Intermediate % 64]>(0));
        } else {
            let packed = packed.view().tile::<m![Rows % (Rows / 8) / 16], 1,
                m![{ Out<Intermediate> } / 64, Rows % (Rows / 8) / 16 = 1 # (Rows / 128),
                    Rows % 16 / 8, Rows % 8, Intermediate % 64]>(0);
            let wave = stored.view_mut().tile::<m![Rows % (Rows / 8) / 16], 1, m![
                { Cluster<Intermediate> },
                Intermediate % (Intermediate / 2) / 64,
                Rows % (Rows / 8) / 16 = 1 #{!} (Rows / 128),
                Rows % 16 / 8,
                Rows / (Rows / 8),
                Rows % 8,
                Intermediate % 64
            ]>(0);
            if rows > 64 {
                packed.to_hbm_view(&mut context.tdma, wave);
            } else {
                let packed = packed.tile::<m![Rows % 16 / 8], 1,
                    m![{ Out<Intermediate> } / 64, Rows % (Rows / 8) / 16 = 1 # (Rows / 128),
                        Rows % 16 / 8 = 1 # 2, Rows % 8, Intermediate % 64]>(0);
                let half =
                    wave.tile::<m![Rows % 16 / 8], 1, m![{ Cluster<Intermediate> },
                        Intermediate % (Intermediate / 2) / 64,
                        Rows % (Rows / 8) / 16 = 1 #{!} (Rows / 128), Rows % 16 / 8 = 1 #{!} 2,
                        Rows / (Rows / 8), Rows % 8, Intermediate % 64]>(0);
                if rows > 8 {
                    packed.to_hbm_view(&mut context.tdma, half);
                } else {
                    packed.slice_tile::<m![Rows / (Rows / 8)], 1,
                        m![{ Blocks<Intermediate> }, Rows / (Rows / 8) = 1 # 8]>(0)
                        .to_hbm_view(&mut context.tdma, half.tile::<m![Rows / (Rows / 8)], 1,
                            m![{ Cluster<Intermediate> }, Intermediate % (Intermediate / 2) / 64,
                                Rows % (Rows / 8) / 16 = 1 #{!} (Rows / 128),
                                Rows % 16 / 8 = 1 #{!} 2, Rows / (Rows / 8) = 1 #{!} 8,
                                Rows % 8, Intermediate % 64]>(0));
                }
            }
        }
    }
}

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
        if live > 2 {
        } else if rows > 128 {
            Prefix::<256>.forward(&mut *context, (panel, &mut output));
        } else if rows > 64 {
            Prefix::<128>.forward(&mut *context, (panel, &mut output));
        } else if rows > 8 {
            Prefix::<64>.forward(&mut *context, (panel, &mut output));
        } else {
            Prefix::<8>.forward(&mut *context, (panel, &mut output));
        }
        output
    }
}

/// The first `LEN` rows of a stored product panel, written in the down projection's input.
struct Prefix<const LEN: usize>;

impl<'p, 'a, Hidden: AxisName, Intermediate: AxisName, Rows: AxisName, const LEN: usize>
    Op<(
        HbmTensorView<'p, bf16, Chip, ChunkPanel<Intermediate, Rows>>,
        &'a mut Product<Hidden, Intermediate, Rows>,
    )> for Prefix<LEN>
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (panel, output): (
            HbmTensorView<'p, bf16, Chip, ChunkPanel<Intermediate, Rows>>,
            &'a mut Product<Hidden, Intermediate, Rows>,
        ),
    ) {
        panel
            .tile::<m![Rows], LEN, m![{ Cluster<Intermediate> }, { Blocks<Intermediate> },
            { Out<Intermediate> } / 64 = 1 # ({ Out<Intermediate> } / 64), Rows = { LEN } # Rows,
            Intermediate % 64]>(0)
            .to_dm_view(
                &mut context.tdma,
                output
                    .view_mut()
                    .tile::<m![Rows], LEN, m![Rows = { LEN } #{!} Rows, Intermediate % 64]>(0),
            );
    }
}
