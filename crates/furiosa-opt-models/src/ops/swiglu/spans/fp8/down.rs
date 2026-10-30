use super::*;
use crate::Op;
use crate::modules::generic::schedule::Begin;
use crate::ops::normalization::span;
use crate::tensor::axis;
use crate::tensor::live::Live;
use core::marker::PhantomData;

/// A down panel's row scales, fetched into the VRF of every feature block.
pub(super) struct FetchScales;

impl<'a, Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>
    Op<&'a Scale<Hidden, Intermediate, Rows>> for FetchScales
{
    type Output = PanelScale<Hidden, Intermediate, Rows>;

    fn forward(
        &self,
        context: &mut Device,
        scales: &'a Scale<Hidden, Intermediate, Rows>,
    ) -> Self::Output {
        context
            .sub
            .begin(scales.view())
            .fetch::<m![{ Groups<Rows> }], m![{ RowLanes<Rows> }]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Groups<Rows> }], m![{ RowLanes<Rows> }]>()
            .to_vrf()
    }
}

/// The down projection of the live waves' products into the first `rows` rows' partial sums, a
/// pair of waves' TRF fill one pair ahead.
pub(super) struct Project;

impl<'a, Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>
    Op<(
        (
            &'a Down<Hidden, Intermediate>,
            &'a Activation<Hidden, Intermediate, Rows>,
        ),
        (usize, usize),
    )> for Project
{
    type Output = Partial<Hidden, Intermediate, Rows>;

    fn forward(
        &self,
        context: &mut Device,
        ((weight, product), (live, rows)): (
            (
                &'a Down<Hidden, Intermediate>,
                &'a Activation<Hidden, Intermediate, Rows>,
            ),
            (usize, usize),
        ),
    ) -> Self::Output {
        const { assert!(Hidden::SIZE / 16 == 256, "written for 16-output slices") };
        type Batch<Rows> = m![Rows = { WAVE }];
        let down_lanes = |context: &mut Device,
                          product: DmTensorView<
            '_,
            f8e4m3,
            Chip,
            Cluster<Intermediate>,
            DownSlice<Hidden, Intermediate>,
            m![Rows = { WAVE }, Intermediate % 64],
        >,
                          quarters: usize|
         -> Operand<Hidden, Intermediate, Rows, Batch<Rows>, WAVE> {
            context
                .sub
                .begin(product)
                .fetch::<m![
                    { RowLanes<Rows> },
                    { Batch<Rows> } % { WAVE } / 32,
                    { Batch<Rows> } / { WAVE },
                    Rows % 32 / 8,
                    Intermediate % 64 / 32
                ], m![Intermediate % 32]>()
                .collect::<m![
                    { RowLanes<Rows> },
                    { Batch<Rows> } % { WAVE } / 32,
                    { Batch<Rows> } / { WAVE },
                    Rows % 32 / 8,
                    Intermediate % 64 / 32
                ], m![Intermediate % 32]>()
                .to_trf_prefix::<m![{ Batch<Rows> } % { WAVE } / 32], _, _>(quarters)
        };
        let project = |context: &mut Device,
                       (weight, lanes): (
            &Down<Hidden, Intermediate>,
            &Operand<Hidden, Intermediate, Rows, Batch<Rows>, WAVE>,
        ),
                       (output, quarters): (
            DmTensorViewMut<
                '_,
                bf16,
                Chip,
                Cluster<Intermediate>,
                m![Hidden / 16],
                m![Rows = { WAVE } #{!} Rows, Hidden % 16],
            >,
            usize,
        )| {
            context
                .main
                .begin(weight.view())
                .fetch::<m![
                    { Batch<Rows> } % { WAVE } / 32,
                    Hidden % 16 / 4,
                    { Batch<Rows> } / { WAVE },
                    Hidden % (Hidden / 8) / 32,
                    Hidden % 4,
                    Intermediate % 64 / 32
                ], m![Hidden % 32 / 16, Intermediate % 32]>()
                .collect::<m![
                    { Batch<Rows> } % { WAVE } / 32,
                    Hidden % 16 / 4,
                    { Batch<Rows> } / { WAVE },
                    Hidden % (Hidden / 8) / 32,
                    Hidden % 4,
                    Intermediate % 64 / 32,
                    Hidden % 32 / 16
                ], m![Intermediate % 32]>()
                .contract_outer::<m![
                    { Batch<Rows> } % { WAVE } / 32,
                    Hidden % 16 / 4,
                    { Batch<Rows> } / { WAVE },
                    Hidden % (Hidden / 8) / 32,
                    Hidden % 4,
                    Intermediate % 64 / 32,
                    Rows % 32 / 8
                ], m![Hidden % 32 / 16, Intermediate % 32], _, _, _>(lanes)
                .contract_packet::<m![Hidden % 32 / 16]>()
                .contract_time::<m![
                    { Batch<Rows> } % { WAVE } / 32,
                    Hidden % 16 / 4,
                    { Batch<Rows> } / { WAVE },
                    Hidden % (Hidden / 8) / 32,
                    Hidden % 4,
                    Rows % 32 / 8
                ]>()
                .contract_lane::<m![
                    { Batch<Rows> } % { WAVE } / 32,
                    Hidden % 16 / 4,
                    { Batch<Rows> } / { WAVE },
                    Hidden % (Hidden / 8) / 32,
                    Hidden % 4,
                    Rows % 32 / 8,
                    Hidden % 32 / 16
                ], m![{ RowLanes<Rows> } # 8]>(LaneMode::Interleaved)
                .vector_init()
                .vector_inter_slice_reduce::<m![Hidden / 16], m![
                    { Batch<Rows> } % { WAVE } / 32,
                    Hidden % 16 / 4,
                    { Batch<Rows> } / { WAVE },
                    Hidden % 4,
                    Rows % 32 / 8
                ]>(InterSliceReduceOpF32::Add)
                .vector_final()
                .cast::<bf16, m![{ RowLanes<Rows> } # 16]>()
                .transpose::<m![
                    { Batch<Rows> } % { WAVE } / 32,
                    Hidden % 16 / 4,
                    { Batch<Rows> } / { WAVE },
                    Rows % 32
                ], m![Hidden % 4 # 16]>()
                .commit_trim::<m![Hidden % 4]>()
                .commit_prefix::<m![{ Batch<Rows> } % { WAVE } / 32], _, _>(output, quarters);
        };
        let mut output: Partial<Hidden, Intermediate, Rows> = DmTensor::new();
        let quarters = (rows + 31) / 32;
        let mut lanes = down_lanes(
            context,
            product
                .view()
                .tile::<m![Rows], WAVE, m![Rows = { WAVE } # Rows, Intermediate % 64]>(0)
                .unpad::<m![Rows = { WAVE }, Intermediate % 64]>(),
            quarters,
        );
        for pair in 0..(live + 1) / 2 {
            let upcoming = (pair + 1) % (Rows::SIZE / WAVE);
            let next = down_lanes(
                context,
                product
                    .view()
                    .tile::<m![Rows], WAVE, m![Rows = { WAVE } # Rows, Intermediate % 64]>(
                        upcoming * WAVE,
                    )
                    .unpad::<m![Rows = { WAVE }, Intermediate % 64]>(),
                quarters - WAVE / 32 * (pair + 1),
            );
            project(
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
}

pub(super) type ChannelScales<Hidden, Intermediate> =
    VrfTensor<f32, Chip, Cluster<Intermediate>, m![Hidden / 16], m![Hidden % 16]>;

/// The partial sums of the first `groups` row groups times their row and channel scales.
pub(super) struct Rescale<Rows>(pub(super) PhantomData<Rows>);

/// The partial sums, the row and channel scales and the row groups a [`Rescale`] writes.
type Scaled<'a, Hidden, Intermediate, Rows> = (
    &'a Partial<Hidden, Intermediate, Rows>,
    (
        &'a PanelScale<Hidden, Intermediate, Rows>,
        &'a ChannelScales<Hidden, Intermediate>,
    ),
    usize,
);

impl<'a, Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>
    Op<Scaled<'a, Hidden, Intermediate, Rows>> for Rescale<Rows>
{
    type Output = Partial<Hidden, Intermediate, Rows>;

    fn forward(
        &self,
        context: &mut Device,
        (partial, (row, channel), groups): Scaled<'a, Hidden, Intermediate, Rows>,
    ) -> Self::Output {
        let mut output: Partial<Hidden, Intermediate, Rows> = DmTensor::new();
        context.main.begin(partial.view())
            .fetch::<m![{ Groups<Rows> }, { RowLanes<Rows> }, Hidden % 16 / 8], m![Hidden % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Groups<Rows> }, { RowLanes<Rows> }, Hidden % 16 / 8], m![Hidden % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<
                m![{ Groups<Rows> }, { RowLanes<Rows> }, Hidden % 16 / 4],
                m![Hidden % 4],
            >()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), row)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), channel)
            .vector_widen_concat::<
                m![{ Groups<Rows> }, { RowLanes<Rows> }, Hidden % 16 / 8],
                m![Hidden % 8],
            >()
            .vector_final().cast::<bf16, m![Hidden % 8 # 16]>().commit_trim::<m![Hidden % 8]>()
            .commit_prefix::<m![{ Groups<Rows> }], _, _>(output.view_mut(), groups);
        output
    }
}

/// A down panel's input as it lands: a small product's panel as it is, or a larger product's
/// quantized panel and its row scales.
type Landed<H, I, Rows> = (
    Product<H, I, Rows>,
    Activation<H, I, Rows>,
    Scale<H, I, Rows>,
);

/// What a tile's product leaves the down projection between the node's two steps, all of it in
/// HBM but the first down panel: the product of at most two waves as it is, of more waves its
/// three quantized panels and their row scales.
pub struct Staging<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName> {
    products: HbmTensor<bf16, Chip, Stores<Intermediate, Rows>>,
    panels: (
        Staged<Intermediate, Rows>,
        Staged<Intermediate, Rows>,
        Staged<Intermediate, Rows>,
    ),
    scales: StagedScales<Intermediate, Rows>,
    first: Down<Hidden, Intermediate>,
}

/// A tile's product staged for the down projection: a span of more than two waves quantized by
/// each row group's scales over all the slices of its panels (the vector path's scale floor), a
/// smaller one kept in BF16 for its panels' block quantization.
impl<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName, Capacity: AxisName>
    Op<Begin<Live<Halves<Intermediate, Rows>, Capacity>>> for DownWeights<Hidden, Intermediate>
{
    type Output = Staging<Hidden, Intermediate, Rows>;

    fn forward(
        &self,
        context: &mut Device,
        Begin(product): Begin<Live<Halves<Intermediate, Rows>, Capacity>>,
    ) -> Self::Output {
        const { assert!(<m![{ Out<Intermediate> } / 64]>::SIZE == Panels::SIZE) };
        let (live, rows) = (product.count::<axis::Rows<128>>(), product.rows());
        let product = product.into_value();
        let mut products = HbmTensor::<bf16, Chip, Stores<Intermediate, Rows>>::new();
        let (mut h0, mut h1, mut h2) = (Staged::new(), Staged::new(), Staged::new());
        let mut scales = StagedScales::<Intermediate, Rows>::new();
        if live <= 2 {
            StoreTiles.forward(&mut *context, (&product, rows, &mut products));
        } else {
            let mut chunks = ChunkMax::<Intermediate, Rows, Slice<Intermediate, Rows>>::new();
            PanelMaxima::<0>.forward(&mut *context, (&product, &mut chunks));
            PanelMaxima::<1>.forward(&mut *context, (&product, &mut chunks));
            PanelMaxima::<2>.forward(&mut *context, (&product, &mut chunks));
            let mut groups = SpreadScale::<Intermediate, Rows>::new();
            let broadcast = GroupScales.forward(&mut *context, (&chunks, &mut groups));
            let mut quantized = Spread::<Intermediate, Rows, f8e4m3>::new();
            QuantizeSpread::<0>.forward(&mut *context, (&product, &broadcast, &mut quantized));
            QuantizeSpread::<1>.forward(&mut *context, (&product, &broadcast, &mut quantized));
            QuantizeSpread::<2>.forward(&mut *context, (&product, &broadcast, &mut quantized));
            StagePanel.forward(&mut *context, (&quantized, 0, &mut h0));
            StagePanel.forward(&mut *context, (&quantized, 1, &mut h1));
            StagePanel.forward(&mut *context, (&quantized, 2, &mut h2));
            StageScales.forward(&mut *context, (&groups, &mut scales));
        }
        Staging {
            products,
            panels: (h0, h1, h2),
            scales,
            first: self.forward(&mut *context, Begin(0)),
        }
    }
}

/// The down projection of a staged product: each of the three panels quantized, projected and
/// rescaled in turn while the next weight panel lands, the panels' sums added, and each cluster's
/// row half of the sums exchanged into the span's rows.
impl<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName, Capacity: AxisName>
    Op<Live<Staging<Hidden, Intermediate, Rows>, Capacity>> for DownWeights<Hidden, Intermediate>
{
    type Output = Owned<Hidden, Rows>;

    #[allow(clippy::too_many_lines)]
    fn forward(
        &self,
        context: &mut Device,
        staging: Live<Staging<Hidden, Intermediate, Rows>, Capacity>,
    ) -> Self::Output {
        const { assert!(<m![{ Out<Intermediate> } / 64]>::SIZE == Panels::SIZE) };
        let (live, rows) = (staging.count::<axis::Rows<128>>(), staging.rows());
        let Staging {
            products,
            panels: (h0, h1, h2),
            scales,
            first,
        } = staging.into_value();
        // SAFETY: the store's row digits are the span's rows in order.
        let products: HbmTensor<bf16, Chip, Chunks<Intermediate, Rows>> =
            unsafe { products.reshape() };
        let down: DmTensor<f32, Chip, Cluster<Intermediate>, m![Hidden / 16], m![Hidden % 16]> =
            self.1.to_dm(&mut context.tdma);
        let channels: ChannelScales<Hidden, Intermediate> = context
            .sub
            .begin(down.view())
            .fetch::<m![Hidden % 16 / 8], m![Hidden % 8]>()
            .collect::<m![Hidden % 16 / 8], m![Hidden % 8]>()
            .to_vrf();
        let groups = (rows + 7) / 8;
        // Every panel's input lands ahead of the projections, in the order they read it with their
        // weight panels: a product of at most two waves as it is, of more its quantized panel.
        let landed = |context: &mut Device,
                      (panel, staged): (usize, &Staged<Intermediate, Rows>)| {
            let mut product = Product::<Hidden, Intermediate, Rows>::new();
            let mut activation = Activation::<Hidden, Intermediate, Rows>::new();
            let mut scale = Scale::<Hidden, Intermediate, Rows>::new();
            if live <= 2 {
                product = LoadProduct::<Hidden>(PhantomData)
                    .forward(&mut *context, (&products, panel, (live, rows)));
            } else {
                CopyPanel.forward(&mut *context, (staged, &mut activation));
                CopyScales.forward(&mut *context, (&scales, panel, &mut scale));
            }
            (product, activation, scale)
        };
        let project = |context: &mut Device,
                       (weight, (product, mut activation, mut scale)): (
            &Down<Hidden, Intermediate>,
            Landed<Hidden, Intermediate, Rows>,
        )|
         -> Partial<Hidden, Intermediate, Rows> {
            if live <= 2 {
                QuantizeProduct.forward(
                    &mut *context,
                    (&product, rows, (&mut activation, &mut scale)),
                );
            }
            let projected = Project.forward(&mut *context, ((weight, &activation), (live, rows)));
            let row = FetchScales.forward(&mut *context, &scale);
            Rescale::<Rows>(PhantomData)
                .forward(&mut *context, (&projected, (&row, &channels), groups))
        };
        let add = |context: &mut Device,
                   (left, right): (
            &Partial<Hidden, Intermediate, Rows>,
            &Partial<Hidden, Intermediate, Rows>,
        )|
         -> Partial<Hidden, Intermediate, Rows> {
            let mut sums = Partial::<Hidden, Intermediate, Rows>::new();
            Add::of::<
                m![{ Groups<Rows> }],
                m![{ RowLanes<Rows> }, Hidden % 16 / 8],
                m![Hidden % 8],
                Term,
                _,
                _,
                _,
                _,
            >(&mut *context, (left, right))
            .commit_prefix::<m![{ Groups<Rows> }], _, _>(sums.view_mut(), groups);
            sums
        };
        let (input0, input1) = (landed(context, (0, &h0)), landed(context, (1, &h1)));
        let second = self.forward(&mut *context, Begin(1));
        let input2 = landed(context, (2, &h2));
        let d0 = project(context, (&first, input0));
        let third = self.forward(&mut *context, Begin(2));
        let d1 = project(context, (&second, input1));
        let sums = add(context, (&d0, &d1));
        let d2 = project(context, (&third, input2));
        let partial = add(context, (&sums, &d2));
        let mut result: DmTensor<
            bf16,
            Chip,
            Cluster<Intermediate>,
            m![Hidden / 16],
            m![Rows % (Rows / 2), Hidden % 16],
        > = DmTensor::new();
        if rows <= 8 {
            let live_rows = || {
                partial.view().tile::<m![Rows % (Rows / 2)], 8,
                    m![Rows / (Rows / 2), Rows % (Rows / 2) = 8 # (Rows / 2), Hidden % 16]>(0)
            };
            let half = |context: &mut Device, indices| {
                live_rows().asymmetric_cluster_slice::<
                        m![Rows / (Rows / 2)],
                        m![Rows % (Rows / 2) = 8, Hidden % 16],
                    >(&mut context.sub, indices)
            };
            let (send, own) = (half(context, &[1, 0]), half(context, &[0, 1]));
            let exchanged = send.view().cluster_swap().to_dm(&mut context.tdma);
            Add::of::<m![Rows % (Rows / 2) = 8, Hidden % 16 / 8], m![1], m![Hidden % 8], Term, _, _, _, _>(&mut *context, (&own, &exchanged)).commit_prefix::<m![Rows % (Rows / 2) = 8], _, _>(result.view_mut().tile::<m![Rows % (Rows / 2)], 8,
                    m![Rows % (Rows / 2) = 8 #{!} (Rows / 2), Hidden % 16]>(0), rows);
        } else {
            let half = |context: &mut Device, indices| {
                partial.view().asymmetric_cluster_slice::<
                        m![Rows / (Rows / 2)],
                        m![Rows % (Rows / 2), Hidden % 16],
                    >(&mut context.sub, indices)
            };
            let (send, own) = (half(context, &[1, 0]), half(context, &[0, 1]));
            let exchanged = send.view().cluster_swap().to_dm(&mut context.tdma);
            let reach = span::reach::<Rows>(rows);
            Add::of::<
                m![Rows % (Rows / 2) / 8],
                m![Rows % 8, Hidden % 16 / 8],
                m![Hidden % 8],
                Term,
                _,
                _,
                _,
                _,
            >(&mut *context, (&own, &exchanged))
            .commit_prefix::<m![Rows % (Rows / 2) / 8], _, _>(result.view_mut(), reach.div_ceil(8));
        }
        // SAFETY: extraction and exchange give cluster c the complete sum for row half c.
        unsafe { result.reshape() }
    }
}
