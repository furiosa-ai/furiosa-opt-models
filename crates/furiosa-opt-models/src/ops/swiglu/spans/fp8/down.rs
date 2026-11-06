use super::*;
use crate::Begin;
use crate::Op;
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

/// What a tile's product leaves the down projection between its two steps, all of it in
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

/// A [`Staging`] but its first down panel.
type Spilled<Intermediate, Rows> = (
    HbmTensor<bf16, Chip, Stores<Intermediate, Rows>>,
    (
        Staged<Intermediate, Rows>,
        Staged<Intermediate, Rows>,
        Staged<Intermediate, Rows>,
    ),
    StagedScales<Intermediate, Rows>,
);

/// A tile's product staged for the down projection, then the first down panel landed.
impl<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName, Capacity: AxisName>
    Op<Begin<Live<Halves<Intermediate, Rows>, Capacity>>> for DownWeights<Hidden, Intermediate>
{
    type Output = Staging<Hidden, Intermediate, Rows>;

    fn forward(
        &self,
        context: &mut Device,
        Begin(product): Begin<Live<Halves<Intermediate, Rows>, Capacity>>,
    ) -> Self::Output {
        let (products, panels, scales) = spill(&mut *context, product);
        let first = self.forward(&mut *context, Begin(0));
        Staging {
            products,
            panels,
            scales,
            first,
        }
    }
}

/// A tile's product staged for the down projection whose first down panel landed beforehand.
impl<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName, Capacity: AxisName>
    Op<
        Begin<(
            Live<Halves<Intermediate, Rows>, Capacity>,
            Down<Hidden, Intermediate>,
        )>,
    > for DownWeights<Hidden, Intermediate>
{
    type Output = Staging<Hidden, Intermediate, Rows>;

    fn forward(
        &self,
        context: &mut Device,
        Begin((product, first)): Begin<(
            Live<Halves<Intermediate, Rows>, Capacity>,
            Down<Hidden, Intermediate>,
        )>,
    ) -> Self::Output {
        let (products, panels, scales) = spill(context, product);
        Staging {
            products,
            panels,
            scales,
            first,
        }
    }
}

/// The down channel scales in data memory.
type Channels<Hidden, Intermediate> =
    DmTensor<f32, Chip, Cluster<Intermediate>, m![Hidden / 16], m![Hidden % 16]>;

/// A staged product's down projection with its weight panels, the third as `W`, their inputs in
/// panel order and its channel scales landed: once the third has, what is left is compute and the
/// exchange of the row halves.
pub struct Ready<
    Hidden: AxisName,
    Intermediate: AxisName,
    Rows: AxisName,
    W = Down<Hidden, Intermediate>,
> {
    weights: (Down<Hidden, Intermediate>, Down<Hidden, Intermediate>, W),
    inputs: (
        Landed<Hidden, Intermediate, Rows>,
        Landed<Hidden, Intermediate, Rows>,
        Landed<Hidden, Intermediate, Rows>,
    ),
    channels: Channels<Hidden, Intermediate>,
}

/// The down projection of a staged product, its second and third down panels landing between its
/// first panels' work, and its row halves summed once exchanged.
impl<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName, Capacity: AxisName>
    Op<Live<Staging<Hidden, Intermediate, Rows>, Capacity>> for DownWeights<Hidden, Intermediate>
{
    type Output = Owned<Hidden, Rows>;

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
        let channels = self.1.to_dm(&mut context.tdma);
        // Every panel's input lands ahead of the projections, in the order they read it with their
        // weight panels.
        let mut input0 = loaded(&mut *context, &products, 0, (live, rows));
        copied(&mut *context, (&h0, &scales), 0, live, &mut input0);
        let mut input1 = loaded(&mut *context, &products, 1, (live, rows));
        copied(&mut *context, (&h1, &scales), 1, live, &mut input1);
        let second = self.forward(&mut *context, Begin(1));
        let mut input2 = loaded(&mut *context, &products, 2, (live, rows));
        copied(&mut *context, (&h2, &scales), 2, live, &mut input2);
        let third = |context: &mut Device| self.forward(context, Begin(2));
        let inputs = (input0, input1, input2);
        let exchange = projected(
            &mut *context,
            ((first, second), third),
            (inputs, channels),
            (live, rows),
        );
        exchange.summed(context)
    }
}

/// A larger product's third quantized panel and its row scales, which the third input copies once
/// the third weight panel is issued, and the product's waves.
type Pending<Intermediate, Rows> = (
    Staged<Intermediate, Rows>,
    StagedScales<Intermediate, Rows>,
    usize,
);

/// A staged product's down panels' inputs and channel scales landed, its second weight panel
/// landed beforehand and its third still to land, and with it a larger product's third input:
/// that copy's cluster sync would hold back the third panel's transfer behind it.
impl<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName, Capacity: AxisName>
    Op<
        Live<
            (
                Staging<Hidden, Intermediate, Rows>,
                Down<Hidden, Intermediate>,
            ),
            Capacity,
        >,
    > for DownWeights<Hidden, Intermediate>
{
    type Output = (
        Ready<Hidden, Intermediate, Rows, ()>,
        Pending<Intermediate, Rows>,
    );

    fn forward(
        &self,
        context: &mut Device,
        staged: Live<
            (
                Staging<Hidden, Intermediate, Rows>,
                Down<Hidden, Intermediate>,
            ),
            Capacity,
        >,
    ) -> Self::Output {
        const { assert!(<m![{ Out<Intermediate> } / 64]>::SIZE == Panels::SIZE) };
        let (live, rows) = (staged.count::<axis::Rows<128>>(), staged.rows());
        let (staging, second) = staged.into_value();
        let Staging {
            products,
            panels: (h0, h1, h2),
            scales,
            first,
        } = staging;
        // SAFETY: the store's row digits are the span's rows in order.
        let products: HbmTensor<bf16, Chip, Chunks<Intermediate, Rows>> =
            unsafe { products.reshape() };
        let channels = self.1.to_dm(&mut context.tdma);
        let mut input0 = loaded(&mut *context, &products, 0, (live, rows));
        copied(&mut *context, (&h0, &scales), 0, live, &mut input0);
        let mut input1 = loaded(&mut *context, &products, 1, (live, rows));
        copied(&mut *context, (&h1, &scales), 1, live, &mut input1);
        let input2 = loaded(&mut *context, &products, 2, (live, rows));
        let ready = Ready {
            weights: (first, second, ()),
            inputs: (input0, input1, input2),
            channels,
        };
        (ready, (h2, scales, live))
    }
}

/// A down projection whose inputs landed, with its third weight panel, landed behind them, and a
/// larger product's third input copied behind that panel.
impl<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>
    Op<(
        (
            Ready<Hidden, Intermediate, Rows, ()>,
            Pending<Intermediate, Rows>,
        ),
        Down<Hidden, Intermediate>,
    )> for DownWeights<Hidden, Intermediate>
{
    type Output = Ready<Hidden, Intermediate, Rows>;

    fn forward(
        &self,
        context: &mut Device,
        ((ready, (staged, scales, live)), third): (
            (
                Ready<Hidden, Intermediate, Rows, ()>,
                Pending<Intermediate, Rows>,
            ),
            Down<Hidden, Intermediate>,
        ),
    ) -> Self::Output {
        let Ready {
            weights: (first, second, ()),
            inputs: (input0, input1, mut input2),
            channels,
        } = ready;
        copied(context, (&staged, &scales), 2, live, &mut input2);
        Ready {
            weights: (first, second, third),
            inputs: (input0, input1, input2),
            channels,
        }
    }
}

/// The down projection of a staged product whose weights and inputs landed, up to the exchange of
/// its row halves.
impl<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName, Capacity: AxisName>
    Op<Begin<Live<Ready<Hidden, Intermediate, Rows>, Capacity>>>
    for DownWeights<Hidden, Intermediate>
{
    type Output = Exchange<Hidden, Intermediate, Rows>;

    fn forward(
        &self,
        context: &mut Device,
        Begin(ready): Begin<Live<Ready<Hidden, Intermediate, Rows>, Capacity>>,
    ) -> Self::Output {
        let (live, rows) = (ready.count::<axis::Rows<128>>(), ready.rows());
        let Ready {
            weights: (first, second, third),
            inputs,
            channels,
        } = ready.into_value();
        let third = |_: &mut Device| third;
        projected(
            context,
            ((first, second), third),
            (inputs, channels),
            (live, rows),
        )
    }
}

/// An exchange's row halves summed: the span's rows.
impl<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName, Capacity: AxisName>
    Op<Live<Exchange<Hidden, Intermediate, Rows>, Capacity>> for DownWeights<Hidden, Intermediate>
{
    type Output = Owned<Hidden, Rows>;

    fn forward(
        &self,
        context: &mut Device,
        exchange: Live<Exchange<Hidden, Intermediate, Rows>, Capacity>,
    ) -> Self::Output {
        exchange.into_value().summed(context)
    }
}

/// A cluster half's first eight rows of a span's sums in each slice of their features.
type Eight<Hidden, Intermediate, Rows> = DmTensor<
    bf16,
    Chip,
    Cluster<Intermediate>,
    m![Hidden / 16],
    m![Rows % (Rows / 2) = 8, Hidden % 16],
>;

/// A cluster half's rows of a span's sums in each slice of their features.
type Half<Hidden, Intermediate, Rows> = DmTensor<
    bf16,
    Chip,
    Cluster<Intermediate>,
    m![Hidden / 16],
    m![Rows % (Rows / 2), Hidden % 16],
>;

/// A down projection's summed partial sums for the first `rows` rows of a span, each cluster's row
/// half of them sent to the other cluster: the first eight rows of each half when the span holds
/// no more, into the first tensor of `sent`, else every row, into the second.
pub struct Exchange<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName> {
    partial: Partial<Hidden, Intermediate, Rows>,
    sent: (
        Eight<Hidden, Intermediate, Rows>,
        Half<Hidden, Intermediate, Rows>,
    ),
    rows: usize,
}

impl<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>
    Exchange<Hidden, Intermediate, Rows>
{
    /// The sums of the first `rows` rows of a span, each cluster's row half sent to the other.
    fn new(
        context: &mut Device,
        partial: Partial<Hidden, Intermediate, Rows>,
        rows: usize,
    ) -> Self {
        let mut exchange = Self {
            partial,
            sent: (DmTensor::new(), DmTensor::new()),
            rows,
        };
        if rows <= 8 {
            let send = exchange.eight(context, &[1, 0]);
            let sent = exchange.sent.0.view_mut();
            send.view()
                .cluster_swap()
                .to_dm_view(&mut context.tdma, sent);
        } else {
            let send = exchange.half(context, &[1, 0]);
            let sent = exchange.sent.1.view_mut();
            send.view()
                .cluster_swap()
                .to_dm_view(&mut context.tdma, sent);
        }
        exchange
    }

    /// The span's rows: each cluster's own row half of the sums added to the half the other sent.
    fn summed(self, context: &mut Device) -> Owned<Hidden, Rows> {
        let mut result: Half<Hidden, Intermediate, Rows> = DmTensor::new();
        if self.rows <= 8 {
            let own = self.eight(context, &[0, 1]);
            Add::of::<m![Rows % (Rows / 2) = 8, Hidden % 16 / 8], m![1], m![Hidden % 8], Term, _, _, _, _>(&mut *context, (&own, &self.sent.0)).commit_prefix::<m![Rows % (Rows / 2) = 8], _, _>(result.view_mut().tile::<m![Rows % (Rows / 2)], 8,
                    m![Rows % (Rows / 2) = 8 #{!} (Rows / 2), Hidden % 16]>(0), self.rows);
        } else {
            let own = self.half(context, &[0, 1]);
            let reach = span::reach::<Rows>(self.rows);
            Add::of::<
                m![Rows % (Rows / 2) / 8],
                m![Rows % 8, Hidden % 16 / 8],
                m![Hidden % 8],
                Term,
                _,
                _,
                _,
                _,
            >(&mut *context, (&own, &self.sent.1))
            .commit_prefix::<m![Rows % (Rows / 2) / 8], _, _>(result.view_mut(), reach.div_ceil(8));
        }
        // SAFETY: extraction and exchange give cluster c the complete sum for row half c.
        unsafe { result.reshape() }
    }

    /// Each cluster's half of the first eight rows of the sums, from the cluster `indices` names.
    fn eight(&self, context: &mut Device, indices: &[usize]) -> Eight<Hidden, Intermediate, Rows> {
        self.partial.view().tile::<m![Rows % (Rows / 2)], 8,
            m![Rows / (Rows / 2), Rows % (Rows / 2) = 8 # (Rows / 2), Hidden % 16]>(0)
            .asymmetric_cluster_slice::<
                m![Rows / (Rows / 2)],
                m![Rows % (Rows / 2) = 8, Hidden % 16],
            >(&mut context.sub, indices)
    }

    /// Each cluster's half of every row of the sums, from the cluster `indices` names.
    fn half(&self, context: &mut Device, indices: &[usize]) -> Half<Hidden, Intermediate, Rows> {
        self.partial
            .view()
            .asymmetric_cluster_slice::<m![Rows / (Rows / 2)], m![Rows % (Rows / 2), Hidden % 16]>(
                &mut context.sub,
                indices,
            )
    }
}

/// A tile's product staged for the down projection: a span of more than two waves quantized by
/// each row group's scales over all the slices of its panels (the vector path's scale floor), a
/// smaller one kept in BF16 for its panels' block quantization.
fn spill<Intermediate: AxisName, Rows: AxisName, Capacity: AxisName>(
    context: &mut Device,
    product: Live<Halves<Intermediate, Rows>, Capacity>,
) -> Spilled<Intermediate, Rows> {
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
    (products, (h0, h1, h2), scales)
}

/// Panel `panel`'s input as it lands: a product of at most two waves as it is; a larger
/// product's panel and row scales are [`copied`] into it.
fn loaded<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    products: &HbmTensor<bf16, Chip, Chunks<Intermediate, Rows>>,
    panel: usize,
    (live, rows): (usize, usize),
) -> Landed<Hidden, Intermediate, Rows> {
    let mut product = Product::<Hidden, Intermediate, Rows>::new();
    let activation = Activation::<Hidden, Intermediate, Rows>::new();
    let scale = Scale::<Hidden, Intermediate, Rows>::new();
    if live <= 2 {
        product = LoadProduct::<Hidden>(PhantomData)
            .forward(&mut *context, (products, panel, (live, rows)));
    }
    (product, activation, scale)
}

/// A product of more than two waves: its quantized panel `panel` and its row scales copied into
/// the panel's input.
fn copied<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    (staged, scales): (&Staged<Intermediate, Rows>, &StagedScales<Intermediate, Rows>),
    panel: usize,
    live: usize,
    (_, activation, scale): &mut Landed<Hidden, Intermediate, Rows>,
) {
    if live > 2 {
        CopyPanel.forward(&mut *context, (staged, activation));
        CopyScales.forward(&mut *context, (scales, panel, scale));
    }
}

/// The down projection of a staged product from its first two weight panels, the third taken
/// from `third` once the first panel's work is issued, and its landed `inputs`: each panel
/// quantized, projected and rescaled in turn, the panels' sums added, and each cluster's row half
/// of the sums sent to the other cluster.
#[allow(clippy::too_many_lines)]
fn projected<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    ((first, second), third): (
        (Down<Hidden, Intermediate>, Down<Hidden, Intermediate>),
        impl FnOnce(&mut Device) -> Down<Hidden, Intermediate>,
    ),
    ((input0, input1, input2), down): (
        (
            Landed<Hidden, Intermediate, Rows>,
            Landed<Hidden, Intermediate, Rows>,
            Landed<Hidden, Intermediate, Rows>,
        ),
        Channels<Hidden, Intermediate>,
    ),
    (live, rows): (usize, usize),
) -> Exchange<Hidden, Intermediate, Rows> {
    let channels: ChannelScales<Hidden, Intermediate> = context
        .sub
        .begin(down.view())
        .fetch::<m![Hidden % 16 / 8], m![Hidden % 8]>()
        .collect::<m![Hidden % 16 / 8], m![Hidden % 8]>()
        .to_vrf();
    let groups = (rows + 7) / 8;
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
        Rescale::<Rows>(PhantomData).forward(&mut *context, (&projected, (&row, &channels), groups))
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
    let d0 = project(context, (&first, input0));
    let third = third(&mut *context);
    let d1 = project(context, (&second, input1));
    let sums = add(context, (&d0, &d1));
    let d2 = project(context, (&third, input2));
    let partial = add(context, (&sums, &d2));
    Exchange::new(context, partial, rows)
}
