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

impl<'a, Hidden: AxisName, Intermediate: AxisName, Rows: AxisName, const B: usize>
    Op<(
        (
            &'a Down<Hidden, Intermediate, B>,
            &'a Activation<Hidden, Intermediate, Rows, B>,
        ),
        (usize, usize),
    )> for Project
{
    type Output = Partial<Hidden, Intermediate, Rows>;

    fn forward(
        &self,
        context: &mut Device,
        ((weight, product), (_, rows)): (
            (
                &'a Down<Hidden, Intermediate, B>,
                &'a Activation<Hidden, Intermediate, Rows, B>,
            ),
            (usize, usize),
        ),
    ) -> Self::Output {
        project::<Hidden, Intermediate, Rows, B, WAVE, 4>(
            context,
            &weight.view(),
            &product.view(),
            rows,
            None,
        )
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
        let partial = panels(
            context,
            &first,
            rows,
            (
                |context, panel| landed(context, (panel, [&h0, &h1, &h2][panel])),
                (
                    |context| self.forward(context, Begin(1)),
                    |context| self.forward(context, Begin(2)),
                ),
                |context, weight, input| project(context, (weight, input)),
            ),
            (|_| (), |_, (), partial| partial),
        );
        finish(context, &partial, rows)
    }
}

/// Full-token scales remain in DM; each contraction wave fetches its own row window.
pub(super) type TokenScales<H, I, R> = DmTensor<f32, Chip, Cluster<I>, m![H / 16], m![R]>;
type FusedScales<'a, H, I, R> = (
    &'a DmTensorView<'a, f32, Chip, Cluster<I>, m![H / 16], m![R]>,
    &'a ChannelScales<H, I>,
);

/// The native channel-scaled path uses eight or sixteen rows when possible, then Qwen's
/// 32-row coverage. A three-group transpose cannot gather its 24 columns on this compiler.
pub(super) fn trimmed<H: AxisName, I: AxisName, R: AxisName, const B: usize>(
    context: &mut Device,
    weight: &DmTensorView<
        '_,
        f8e4m3,
        Chip,
        Cluster<I>,
        DownSlice<H, I>,
        m![H % (H / 8), I % { B }],
    >,
    product: &DmTensorView<'_, f8e4m3, Chip, Cluster<I>, DownSlice<H, I>, m![R, I % { B }]>,
    rows: usize,
    scales: FusedScales<'_, H, I, R>,
) -> Partial<H, I, R> {
    if rows <= 32 {
        match rows.div_ceil(8) {
            1 => project::<H, I, R, B, 32, 1>(context, weight, product, rows, Some(scales)),
            2 => project::<H, I, R, B, 32, 2>(context, weight, product, rows, Some(scales)),
            _ => project::<H, I, R, B, 32, 4>(context, weight, product, rows, Some(scales)),
        }
    } else {
        project::<H, I, R, B, WAVE, 4>(context, weight, product, rows, Some(scales))
    }
}
fn project<
    Hidden: AxisName,
    Intermediate: AxisName,
    Rows: AxisName,
    const B: usize,
    const W: usize,
    const G: usize,
>(
    context: &mut Device,
    weight: &DmTensorView<
        '_,
        f8e4m3,
        Chip,
        Cluster<Intermediate>,
        DownSlice<Hidden, Intermediate>,
        m![Hidden % (Hidden / 8), Intermediate % { B }],
    >,
    product: &DmTensorView<
        '_,
        f8e4m3,
        Chip,
        Cluster<Intermediate>,
        DownSlice<Hidden, Intermediate>,
        m![Rows, Intermediate % { B }],
    >,
    rows: usize,
    scales: Option<FusedScales<'_, Hidden, Intermediate, Rows>>,
) -> Partial<Hidden, Intermediate, Rows> {
    const { assert!(Hidden::SIZE / 16 == 256, "written for 16-output slices") };
    type Batch<Rows, const W: usize> = m![Rows = { W }];
    let down_lanes = |context: &mut Device,
                      product: DmTensorView<
        '_,
        f8e4m3,
        Chip,
        Cluster<Intermediate>,
        DownSlice<Hidden, Intermediate>,
        m![Rows = { W }, Intermediate % { B }],
    >,
                      quarters: usize|
     -> Operand<Hidden, Intermediate, Rows, Batch<Rows, W>, W, B, G> {
        // SAFETY: the selected W-row window starts at zero in its local coordinates; replacing
        // its resize with the equivalent row residue preserves every element and byte stride.
        let product = unsafe {
            product.reshape::<Chip, Cluster<Intermediate>, DownSlice<Hidden, Intermediate>,
                m![{ Batch<Rows, W> } % { W } / 32, DownGroups, Rows % 8, Intermediate % {B}]>()
        };
        let stream = context
            .sub
            .begin(product.tile::<m![DownGroups], G,
                m![{ Batch<Rows, W> } % { W } / 32, DownGroups = {G} # 4, Rows % 8, Intermediate % {B}]>(0)
                .unpad::<m![{ Batch<Rows, W> } % { W } / 32, DownGroups = {G}, Rows % 8, Intermediate % {B}]>())
            .fetch::<m![
                { RowLanes<Rows> },
                { Batch<Rows, W> } % { W } / 32,
                { Batch<Rows, W> } / { W },
                DownGroups = {G},
                Intermediate % {B} / 32
            ], m![Intermediate % 32]>()
            .collect::<m![
                { RowLanes<Rows> },
                { Batch<Rows, W> } % { W } / 32,
                { Batch<Rows, W> } / { W },
                DownGroups = {G},
                Intermediate % {B} / 32
            ], m![Intermediate % 32]>();
        if const { W == 32 } {
            stream.to_trf()
        } else {
            stream.to_trf_prefix::<m![{ Batch<Rows, W> } % { W } / 32], _, _>(quarters)
        }
    };
    let project = |context: &mut Device,
                   (weight, lanes): (
        &DmTensorView<
            '_,
            f8e4m3,
            Chip,
            Cluster<Intermediate>,
            DownSlice<Hidden, Intermediate>,
            m![Hidden % (Hidden / 8), Intermediate % { B }],
        >,
        &Operand<Hidden, Intermediate, Rows, Batch<Rows, W>, W, B, G>,
    ),
                   (output, quarters, at): (
        DmTensorViewMut<
            '_,
            bf16,
            Chip,
            Cluster<Intermediate>,
            m![Hidden / 16],
            m![Rows = { W } #{!} Rows, Hidden % 16],
        >,
        usize,
        usize,
    )| {
        let contracted = context
            .main
            .begin(weight.clone())
            .fetch::<m![
                { Batch<Rows, W> } % { W } / 32,
                Hidden % 16 / 4,
                { Batch<Rows, W> } / { W },
                Hidden % (Hidden / 8) / 32,
                Hidden % 4,
                Intermediate % {B} / 32
            ], m![Hidden % 32 / 16, Intermediate % 32]>()
            .collect::<m![
                { Batch<Rows, W> } % { W } / 32,
                Hidden % 16 / 4,
                { Batch<Rows, W> } / { W },
                Hidden % (Hidden / 8) / 32,
                Hidden % 4,
                Intermediate % {B} / 32,
                Hidden % 32 / 16
            ], m![Intermediate % 32]>()
            .contract_outer::<m![
                { Batch<Rows, W> } % { W } / 32,
                Hidden % 16 / 4,
                { Batch<Rows, W> } / { W },
                Hidden % (Hidden / 8) / 32,
                Hidden % 4,
                Intermediate % {B} / 32,
                DownGroups = {G}
            ], m![Hidden % 32 / 16, Intermediate % 32], _, _, _>(lanes)
            .contract_packet::<m![Hidden % 32 / 16]>()
            .contract_time::<m![
                { Batch<Rows, W> } % { W } / 32,
                Hidden % 16 / 4,
                { Batch<Rows, W> } / { W },
                Hidden % (Hidden / 8) / 32,
                Hidden % 4,
                DownGroups = {G}
            ]>()
            .contract_lane::<m![
                { Batch<Rows, W> } % { W } / 32,
                Hidden % 16 / 4,
                { Batch<Rows, W> } / { W },
                Hidden % (Hidden / 8) / 32,
                Hidden % 4,
                DownGroups = {G},
                Hidden % 32 / 16
            ], m![{ RowLanes<Rows> } # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<m![Hidden / 16], m![
                { Batch<Rows, W> } % { W } / 32,
                Hidden % 16 / 4,
                { Batch<Rows, W> } / { W },
                Hidden % 4,
                DownGroups = {G}
            ]>(InterSliceReduceOpF32::Add);
        let contracted = if let Some((row, channel)) = scales {
            // SAFETY: name the selected wave's row digits as the contraction does. The
            // window retains canonical row order, including the second 256-row wave.
            let row = unsafe {
                row.clone().tile::<m![Rows], W, m![Rows = {W} # Rows]>(at)
                    .unpad::<m![Rows = {W}]>()
                    .reshape::<Chip, Cluster<Intermediate>, m![Hidden / 16],
                        m![{Batch<Rows, W>} % {W} / 32, DownGroups, Rows % 8]>()
            };
            let row = context.sub.begin(row.tile::<m![DownGroups], G,
                    m![{Batch<Rows, W>} % {W} / 32, DownGroups = {G} # 4, Rows % 8]>(0))
                .fetch::<m![{Batch<Rows, W>} % {W} / 32, DownGroups = {G}], m![Rows % 8]>()
                .collect::<m![{Batch<Rows, W>} % {W} / 32, DownGroups = {G}], m![Rows % 8]>()
                .to_vrf::<m![{Batch<Rows, W>} % {W} / 32, DownGroups = {G}, Rows % 8]>();
            // Scale the complete FP32 slice sum before its first BF16 rounding. The
            // channel path has no unscaled partial in DM; Qwen retains its old boundary.
            contracted
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![
                    {Batch<Rows, W>} % {W} / 32, Hidden % 16 / 4,
                    {Batch<Rows, W>} / {W}, Hidden % 4, DownGroups = {G}, Rows % 8 / 4
                ], m![Rows % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &row)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), channel)
                .vector_widen_concat::<m![
                    {Batch<Rows, W>} % {W} / 32, Hidden % 16 / 4,
                    {Batch<Rows, W>} / {W}, Hidden % 4, DownGroups = {G}
                ], m![{RowLanes<Rows>} # 8]>()
                .vector_final()
        } else {
            contracted.vector_final()
        };
        let contracted = contracted
            .cast::<bf16, m![{ RowLanes<Rows> } # 16]>()
            .transpose::<m![
                { Batch<Rows, W> } % { W } / 32,
                Hidden % 16 / 4,
                { Batch<Rows, W> } / { W },
                DownGroups = {G}, Rows % 8
            ], m![Hidden % 4 # 16]>()
            .commit_trim::<m![Hidden % 4]>();
        // SAFETY: DownGroups names the four adjacent eight-row groups within each 32-row
        // unit. Keeping it separate prevents the 24-row case from merging a resized digit
        // into the lane coordinate; the physical order and strides are unchanged.
        let output = unsafe {
            output.reshape::<Chip, Cluster<Intermediate>, m![Hidden / 16],
                m![{ Batch<Rows, W> } % { W } / 32 #{!} (Rows / 32), DownGroups, Rows % 8, Hidden % 16]>()
        };
        if const { W == 32 } {
            contracted.commit_view(output.tile::<m![DownGroups], G,
                m![{ Batch<Rows, W> } % { W } / 32 #{!} (Rows / 32), DownGroups = {G} #{!} 4, Rows % 8, Hidden % 16]>(0));
        } else {
            contracted.commit_prefix::<m![{ Batch<Rows, W> } % { W } / 32], _, _>(output, quarters);
        }
    };
    let mut output: Partial<Hidden, Intermediate, Rows> = DmTensor::new();
    let quarters = if const { W == 32 } {
        rows.div_ceil(8)
    } else {
        rows.div_ceil(32)
    };
    let mut lanes = down_lanes(
        context,
        product
            .clone()
            .tile::<m![Rows], W, m![Rows = { W } # Rows, Intermediate % {B}]>(0)
            .unpad::<m![Rows = { W }, Intermediate % { B }]>(),
        quarters,
    );
    for pair in 0..rows.div_ceil(W) {
        let upcoming = (pair + 1) % (Rows::SIZE / W);
        if const { W == 32 } {
            project(
                context,
                (weight, &lanes),
                (
                    output
                        .view_mut()
                        .tile::<m![Rows], W, m![Rows = {W} #{!} Rows, Hidden % 16]>(0),
                    quarters,
                    0,
                ),
            );
        } else {
            let next = down_lanes(
                context,
                product
                    .clone()
                    .tile::<m![Rows], W, m![Rows = { W } # Rows, Intermediate % {B}]>(upcoming * W)
                    .unpad::<m![Rows = { W }, Intermediate % { B }]>(),
                quarters - W / 32 * (pair + 1),
            );
            project(
                context,
                (weight, &lanes),
                (
                    output
                        .view_mut()
                        .tile::<m![Rows], W, m![Rows = { W } #{!} Rows, Hidden % 16]>(pair * W),
                    quarters - W / 32 * pair,
                    pair * W,
                ),
            );
            lanes = next;
        }
    }
    output
}
/// Three full input panels in Qwen's load/compute order, followed by an optional native tail.
/// Tail preparation overlaps the third full contraction. Every addition rounds to BF16 in DM.
pub(super) fn panels<H: AxisName, I: AxisName, R: AxisName, W, Input, Tail>(
    device: &mut Device,
    first: &W,
    rows: usize,
    (mut load, (second, third), mut project): (
        impl FnMut(&mut Device, usize) -> Input,
        (impl FnOnce(&mut Device) -> W, impl FnOnce(&mut Device) -> W),
        impl FnMut(&mut Device, &W, Input) -> Partial<H, I, R>,
    ),
    (prepare_tail, finish_tail): (
        impl FnOnce(&mut Device) -> Tail,
        impl FnOnce(&mut Device, Tail, Partial<H, I, R>) -> Partial<H, I, R>,
    ),
) -> Partial<H, I, R> {
    let (input0, input1) = (load(device, 0), load(device, 1));
    let second = second(device);
    let input2 = load(device, 2);
    let a = project(device, first, input0);
    let third = third(device);
    let b = project(device, &second, input1);
    let sum = add(device, &a, &b, rows);
    let tail = prepare_tail(device);
    let c = project(device, &third, input2);
    let sum = add(device, &sum, &c, rows);
    finish_tail(device, tail, sum)
}
pub(super) fn add<H: AxisName, I: AxisName, R: AxisName>(
    device: &mut Device,
    left: &Partial<H, I, R>,
    right: &Partial<H, I, R>,
    rows: usize,
) -> Partial<H, I, R> {
    let mut sum = Partial::new();
    Add::of::<m![{Groups<R>}], m![{RowLanes<R>}, H % 16 / 8], m![H % 8], Term, _, _, _, _>(
        device,
        (left, right),
    )
    .commit_prefix::<m![{Groups<R>}], _, _>(sum.view_mut(), rows.div_ceil(8));
    sum
}
/// Sum the input-half contributions and return each cluster's own half of the span.
pub(super) fn finish<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    partial: &Partial<Hidden, Intermediate, Rows>,
    rows: usize,
) -> Owned<Hidden, Rows> {
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
