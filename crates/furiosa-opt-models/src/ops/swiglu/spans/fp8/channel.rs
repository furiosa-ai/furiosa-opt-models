//! Channel-scaled checkpoint storage for token-dynamic FP8 MLPs.
//! Down weights are packed into contiguous panels at load, with the native 32-feature tail
//! stored separately. Runtime DMA never strides across a complete 224-feature owner.

use super::*;
use crate::ops::vector::{self, Quantization};
use crate::tensor::form::fp8::{Channel, Rows as StoredRows};
use crate::tensor::live::Live;

type InputQuantizer<H, R> = Quantization<
    m![R % (R / 2), H % 16 / 8],
    m![R % (R / 2), H % 16 / 4],
    H,
    m![R % (R / 2)],
    m![1 # 256],
    m![R % (R / 2) / 4],
    m![R % 4],
>;

type TokenScales<I, R> = DmTensor<f32, Chip, Cluster<I>, m![1 # 32, R / (R / 8)], m![R % (R / 8)]>;
type ProductCodes<I, R> =
    DmTensor<f8e4m3, Chip, Cluster<I>, Slice<I, R>, m![R % (R / 8), {Out<I>}]>;
type ProductQuantizer<I, R> = Quantization<
    m![R % (R / 8), {Out<I>} / 8],
    m![R % (R / 8), {Out<I>} / 4],
    I,
    m![R % (R / 8)],
    Slice<I, R>,
    m![R % (R / 8) / 4],
    m![R % 4],
>;

pub type GateUpWeights<H, I> = StoredRows<
    m![H],
    m![Term, I],
    HbmTensor<f8e4m3, Chip, Blocked<H, I>>,
    HbmTensor<f32, Chip, m![Term, I]>,
    Channel,
>;
pub type Stored<H, const B: usize, const N: usize> = m![
    Panels = {N}, {Cluster<Inputs<B>>}, H / (H / 8), {Blocks<Inputs<B>>},
    H % (H / 8), {Symbol<Inputs<B>>} % {B}
];
pub type DownWeights<H, I> = StoredRows<
    m![I],
    m![H],
    (
        HbmTensor<f8e4m3, Chip, Stored<H, 64, 3>>,
        HbmTensor<f8e4m3, Chip, Stored<H, 32, 1>>,
    ),
    HbmTensor<f32, Chip, m![H]>,
    Channel,
>;

/// The packed inputs of one panel: 64 intermediate owners, each contributing B features.
/// A panel coordinate is not the original intermediate index: owner b selects b*224+offset.
#[derive(Debug, Clone)]
pub struct Inputs<const B: usize>;
impl<const B: usize> AxisName for Inputs<B> {
    const NAME: Ident = Ident::new("PanelInputs");
    const SIZE: usize = 64 * B;
}

impl<'a, H: AxisName, I: AxisName> Load<Projections<'a, 2>> for GateUpWeights<H, I> {
    fn load(cx: &mut Context<'_>, projections: Projections<'a, 2>) -> Result<Self, LoadError> {
        let [(gate, gates), (up, ups)] = projections.read(cx, |from, cx| {
            cx.field("weight", "weight", |cx| {
                fp8::channels::<m![H], m![I]>(from, cx)
            })
        })?;
        let codes = HostTensor::<f8e4m3, m![{ Chip }, { m![Term, I, H] }]>::from_vec(
            [gate, up].concat().repeat(Chip::SIZE),
        );
        let scales = HostTensor::<f32, m![{ Chip }, { m![Term, I] }]>::from_vec(
            [gates, ups].concat().repeat(Chip::SIZE),
        );
        Ok(Self::new(
            cx.field("0", "", |cx| Load::load(cx, codes))?,
            cx.field("1", "", |cx| Load::load(cx, scales))?,
        ))
    }
}

/// Checkpoint output-major codes, reordered without rounding into contiguous DMA panels.
fn packed_down<H: AxisName, I: AxisName, const B: usize, const N: usize>(
    codes: &[f8e4m3],
    offset: usize,
) -> HostTensor<f8e4m3, m![{Chip}, {Stored<H, B, N>}]> {
    const {
        assert!(I::SIZE == 64 * 224 && H::SIZE.is_multiple_of(8));
    }
    let mut packed = Vec::with_capacity(N * 64 * H::SIZE * B);
    for panel in 0..N {
        for cluster in 0..2 {
            for output_owner in 0..8 {
                for input_owner in 0..32 {
                    for output in 0..H::SIZE / 8 {
                        let row = output_owner * (H::SIZE / 8) + output;
                        let input = (cluster * 32 + input_owner) * 224 + offset + panel * B;
                        packed.extend_from_slice(
                            &codes[row * I::SIZE + input..row * I::SIZE + input + B],
                        );
                    }
                }
            }
        }
    }
    HostTensor::from_vec(packed.repeat(Chip::SIZE))
}

impl<'a, H: AxisName, I: AxisName> Load<Projections<'a, 1>> for DownWeights<H, I> {
    fn load(cx: &mut Context<'_>, projections: Projections<'a, 1>) -> Result<Self, LoadError> {
        let [(codes, scales)] = projections.read(cx, |from, cx| {
            cx.field("weight", "weight", |cx| {
                fp8::channels::<m![I], m![H]>(from, cx)
            })
        })?;
        let panels = cx.field("0", "", |cx| {
            Ok((
                cx.field("0", "", |cx| {
                    Load::load(cx, packed_down::<H, I, 64, 3>(&codes, 0))
                })?,
                cx.field("1", "", |cx| {
                    Load::load(cx, packed_down::<H, I, 32, 1>(&codes, 192))
                })?,
            ))
        })?;
        let scales = HostTensor::<f32, m![{ Chip }, H]>::from_vec(scales.repeat(Chip::SIZE));
        Ok(Self::new(
            panels,
            cx.field("1", "", |cx| Load::load(cx, scales))?,
        ))
    }
}

impl<H: AxisName, I: AxisName> Op<Begin<usize>> for GateUpWeights<H, I> {
    type Output = GateUp<H, I>;
    fn forward(&self, device: &mut Device, Begin(panel): Begin<usize>) -> Self::Output {
        stage_gate_up(device, &self.0, panel)
    }
}

impl<'a, H: AxisName, I: AxisName, R: AxisName, C: AxisName>
    Op<(GateUp<H, I>, Live<&'a Owned<H, R>, C>)> for GateUpWeights<H, I>
{
    type Output = (Projection<I, R>, Projection<I, R>);
    fn forward(
        &self,
        device: &mut Device,
        (first, normalized): (GateUp<H, I>, Live<&'a Owned<H, R>, C>),
    ) -> Self::Output {
        compute::forward(
            device,
            (&self.0, &self.1),
            first,
            normalized,
            true,
            |device, input, rows| input_scales(device, input, rows),
        )
    }
}

impl<H: AxisName, I: AxisName> Op<Begin<()>> for GateUpWeights<H, I> {
    type Output = GateUp<H, I>;
    fn forward(&self, device: &mut Device, _: Begin<()>) -> Self::Output {
        stage_gate_up(device, &self.0, 0)
    }
}

type StoredProducts<I, R> = m![{Cluster<I>}, {Blocks<I>}, R, {Out<I>}];
type StoredTokens<I, R> = HbmTensor<f32, Chip, m![{Cluster<I>}, R]>;
type PackedPanel<R, const B: usize> = HbmTensor<
    f8e4m3,
    Chip,
    m![{Cluster<Inputs<B>>}, {Blocks<Inputs<B>>}, R, {Symbol<Inputs<B>>} % {B}],
>;

/// The span's product and complete-token scales, with the first weight panel prefetched.
pub struct Staging<H: AxisName, I: AxisName, R: AxisName> {
    products: HbmTensor<bf16, Chip, StoredProducts<I, R>>,
    panels: (
        PackedPanel<R, 64>,
        PackedPanel<R, 64>,
        PackedPanel<R, 64>,
        PackedPanel<R, 32>,
    ),
    scales: StoredTokens<I, R>,
    first: Down<H, Inputs<64>>,
}

impl<H: AxisName, I: AxisName, R: AxisName, C: AxisName>
    Op<Begin<Live<(Projection<I, R>, Projection<I, R>), C>>> for DownWeights<H, I>
{
    type Output = Staging<H, I, R>;
    fn forward(
        &self,
        device: &mut Device,
        Begin(projected): Begin<Live<(Projection<I, R>, Projection<I, R>), C>>,
    ) -> Self::Output {
        let rows = projected.rows();
        let (gate, up) = projected.into_value();
        use super::super::product;
        let product = if rows <= 8 {
            product::short::<I, R, 1>(device, &gate, &up)
        } else if rows <= 16 {
            product::short::<I, R, 2>(device, &gate, &up)
        } else if rows <= 24 {
            product::short::<I, R, 3>(device, &gate, &up)
        } else if rows <= 32 {
            product::short::<I, R, 4>(device, &gate, &up)
        } else {
            product::Multiply.forward(device, ((&gate, &up), rows))
        };
        let factors = product_scales(device, &product, rows);
        // SAFETY: the row digits match the same physical gate/up output permutation.
        let scales = unsafe {
            factors.view().reshape::<Chip, Cluster<I>,
            m![1 # 32, R % 64 / 8], m![R / 128, R % 128 / 64, R % 8]>()
        };
        let mut stored_scales = StoredTokens::<I, R>::new();
        scales.to_hbm_view(&mut device.tdma, stored_scales.view_mut());
        let mut products = HbmTensor::new();
        let mut panels = (
            PackedPanel::new(),
            PackedPanel::new(),
            PackedPanel::new(),
            PackedPanel::new(),
        );
        if rows <= 256 {
            // Preserve Qwen's small-product route: redistribute BF16, then quantize in the
            // down owners. Its divisor is still the maximum over the entire 14336-feature row.
            // SAFETY: retain the same wave/half/lane permutation as Qwen; a native owner
            // has one feature run rather than three uniform panels.
            let values = unsafe {
                product.view().reshape::<Chip, Cluster<I>, Slice<I, R>,
                    m![1, R % (R / 8) / 16, R % 16 / 8, R % 8, {Out<I>}]>()
            };
            let stored = unsafe {
                products
                    .view_mut()
                    .reshape::<Chip, ProductStore<I, R, m![1], m![{Out<I>}]>>()
            };
            store_rows(device, values, stored, rows);
        } else {
            let codes = product_codes(device, &product, &factors, rows);
            panels = (
                stage_codes(device, &codes, 0),
                stage_codes(device, &codes, 1),
                stage_codes(device, &codes, 2),
                stage_codes(device, &codes, 3),
            );
        }
        Staging {
            products,
            panels,
            scales: stored_scales,
            first: down::<H, I, 64>(device, self, 0),
        }
    }
}

impl<H: AxisName, I: AxisName, R: AxisName, C: AxisName> Op<Live<Staging<H, I, R>, C>>
    for DownWeights<H, I>
{
    type Output = Owned<H, R>;
    fn forward(&self, device: &mut Device, staging: Live<Staging<H, I, R>, C>) -> Self::Output {
        let rows = staging.rows();
        let Staging {
            products,
            panels: (p0, p1, p2, p3),
            scales,
            first,
        } = staging.into_value();
        let row: DmTensor<f32, Chip, Cluster<I>, m![H / 16], m![R]> =
            scales.to_dm(&mut device.tdma);
        let row: PanelScale<H, I, R> = device
            .sub
            .begin(row.view())
            .fetch::<m![{Groups<R>}], m![{RowLanes<R>}]>()
            .collect::<m![{Groups<R>}], m![{RowLanes<R>}]>()
            .to_vrf();
        let channel: DmTensor<f32, Chip, Cluster<I>, m![H / 16], m![H % 16]> =
            self.1.to_dm(&mut device.tdma);
        let channel: down::ChannelScales<H, I> = device
            .sub
            .begin(channel.view())
            .fetch::<m![H % 16 / 8], m![H % 8]>()
            .collect::<m![H % 16 / 8], m![H % 8]>()
            .to_vrf();
        let sum = down::panels(
            device,
            first,
            rows,
            (
                |device, panel| {
                    load_product(
                        device,
                        &products,
                        [&p0, &p1, &p2][panel],
                        &scales,
                        panel,
                        rows,
                    )
                },
                |device, panel| down::<H, I, 64>(device, self, panel),
                |device, weight, input| project_panel(device, weight, &input, &row, &channel, rows),
            ),
            (
                |device| {
                    (
                        down::<H, I, 32>(device, self, 3),
                        load_product(device, &products, &p3, &scales, 3, rows),
                    )
                },
                |device, (weight, input), sum| {
                    let tail = project_panel(device, &weight, &input, &row, &channel, rows);
                    down::add(device, &sum, &tail, rows)
                },
            ),
        );
        down::finish(device, &sum, rows)
    }
}

fn input_scales<H: AxisName, R: AxisName>(
    device: &mut Device,
    input: &Owned<H, R>,
    rows: usize,
) -> HbmTensor<f32, Chip, m![R]> {
    // Commit the live prefix in canonical row order; the existing HBM store performs the
    // gate/up row permutation. A runtime prefix cannot commit directly into Labels<R>.
    let mut scales: DmTensor<f32, Chip, m![R / (R / 2)], m![1 # 256], m![R % (R / 2)]> =
        DmTensor::new();
    InputQuantizer::<H, R>::scales_f32(&mut device.main, input.view())
        .commit_prefix::<m![R % (R / 2) / 4], _, _>(
            scales.view_mut(),
            rows.min(R::SIZE / 2).div_ceil(4),
        );
    let mut stored = HbmTensor::<f32, Chip, m![R]>::new();
    // SAFETY: the same canonical-row to gate/up-wave permutation as InputScales, preserving
    // FP32 instead of rounding the agreed quantizer's divisor to BF16.
    let labels = unsafe {
        stored
            .view_mut()
            .reshape::<Chip, m![R % (R / 8) / 8, R / (R / 4), R % (R / 4) / (R / 8), R % 8]>()
    };
    scales.view().to_hbm_view(&mut device.tdma, labels);
    stored
}
/// Maxima cover all 224 features of each owner, all 32 owners and both clusters. In particular,
/// the last 32 features participate in the same token scale as the first three panels.
fn product_scales<I: AxisName, R: AxisName>(
    device: &mut Device,
    product: &Projection<I, R>,
    rows: usize,
) -> TokenScales<I, R> {
    let local_rows = rows.div_ceil(64) * 8;
    let mut local: DmTensor<f32, Chip, Cluster<I>, Slice<I, R>, m![R % (R / 8)]> = DmTensor::new();
    ProductQuantizer::<I, R>::scales_f32(&mut device.main, product.view())
        .commit_prefix::<m![R % (R / 8) / 4], _, _>(local.view_mut(), local_rows / 4);
    // VRU combines the innermost slice axis. Permute only these local maxima, rather than the
    // 224-feature products, so the 32 intermediate owners become the inner slice axis.
    let ordered: DmTensor<f32, Chip, Cluster<I>, m![R / (R / 8), {Blocks<I>}], m![R % (R / 8)]> =
        local.to_dm(&mut device.tdma);
    let mut reduced: DmTensor<f32, Chip, Cluster<I>, m![R / (R / 8), 1 # 32], m![R % (R / 8)]> =
        DmTensor::new();
    device
        .main
        .begin(ordered.view())
        .fetch::<m![R % (R / 8) / 8], m![R % 8]>()
        .collect::<m![R % (R / 8) / 8], m![R % 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![R / (R / 8), 1 # 32], m![R % (R / 8) / 8]>(
            InterSliceReduceOpF32::Max,
        )
        .vector_final()
        .commit_trim::<m![R % 8]>()
        .commit_prefix::<m![R % (R / 8) / 8], _, _>(reduced.view_mut(), local_rows / 8);
    let local: TokenScales<I, R> = reduced.to_dm(&mut device.tdma);
    vector::cluster_max(device, &local, local_rows)
}
fn product_codes<I: AxisName, R: AxisName>(
    device: &mut Device,
    product: &Projection<I, R>,
    scales: &TokenScales<I, R>,
    rows: usize,
) -> ProductCodes<I, R> {
    // TokenScales holds one owner plus slice padding. Broadcast the scale to every feature
    // owner before quantization; reinterpreting that padding would read unwritten scales.
    let broadcast: DmTensor<f32, Chip, Cluster<I>, Slice<I, R>, m![R % (R / 8)]> =
        scales.to_dm(&mut device.tdma);
    let scales: VrfTensor<f32, Chip, Cluster<I>, Slice<I, R>, m![R % (R / 8)]> = device
        .sub
        .begin(broadcast.view())
        .fetch::<m![R % (R / 8) / 8], m![R % 8]>()
        .collect::<m![R % (R / 8) / 8], m![R % 8]>()
        .to_vrf();
    let mut codes = ProductCodes::<I, R>::new();
    ProductQuantizer::<I, R>::rows(&mut device.main, product.view(), &scales)
        .commit_prefix::<m![R % (R / 8)], _, _>(codes.view_mut(), rows.div_ceil(64) * 8);
    codes
}
/// One of the first three 64-feature panels, or the final native 32-feature panel, of every
/// intermediate owner. Each panel is staged once and reused across the span's live waves.
pub(crate) fn down<H: AxisName, I: AxisName, const B: usize>(
    device: &mut Device,
    weights: &DownWeights<H, I>,
    panel: usize,
) -> Down<H, Inputs<B>, B> {
    const {
        assert!(B == 64 || B == 32);
    }
    let mut output = Down::<H, Inputs<B>, B>::new();
    if const { B == 64 } {
        weights.0.0.view()
            .tile::<m![Panels = 3], 1, m![Panels = 1 # 3, {Cluster<Inputs<64>>}, H / (H / 8), {Blocks<Inputs<64>>}, H % (H / 8), {Symbol<Inputs<64>>} % 64]>(panel)
            .to_dm_view(&mut device.tdma, output.view_mut());
    } else {
        weights
            .0
            .1
            .view()
            .to_dm_view(&mut device.tdma, output.view_mut());
    }
    output
}
fn stage_codes<I: AxisName, R: AxisName, const B: usize>(
    device: &mut Device,
    codes: &ProductCodes<I, R>,
    panel: usize,
) -> PackedPanel<R, B> {
    let band = codes
        .view()
        .tile::<Out<I>, B, m![R % (R / 8), {Out<I>} = {B} # (I / 64)]>(64 * panel);
    // SAFETY: give the selected band its panel-input coordinate, retaining the source stride.
    let band = unsafe {
        band.reshape::<Chip, Cluster<Inputs<B>>,
            m![{Blocks<Inputs<B>>}, R / (R / 8)],
            m![R % (R / 8), {Symbol<Inputs<B>>} % {B} # (I / 64)]>()
    };
    // Pack locally before row redistribution: DMA reads consecutive B-feature rows.
    let mut packed: DmTensor<
        f8e4m3,
        Chip,
        Cluster<Inputs<B>>,
        m![{Blocks<Inputs<B>>}, R / (R / 8)],
        m![R % (R / 8), {Symbol<Inputs<B>>} % {B}],
    > = DmTensor::new();
    device
        .main
        .begin(band)
        .fetch::<m![R % (R / 8), {Symbol<Inputs<B>>} % {B} / 32], m![{Symbol<Inputs<B>>} % 32]>()
        .collect::<m![R % (R / 8), {Symbol<Inputs<B>>} % {B} / 32], m![{Symbol<Inputs<B>>} % 32]>()
        .commit_trim::<m![{Symbol<Inputs<B>>} % 32]>()
        .commit_view(packed.view_mut());
    // SAFETY: only row labels and equal-sized owner coordinates change; the packed stride is B.
    let band = unsafe {
        packed.view().reshape::<Chip, Cluster<Inputs<B>>,
        m![{Blocks<Inputs<B>>}, R % 64 / 8],
        m![R / 128, R % 128 / 64, R % 8, {Symbol<Inputs<B>>} % {B}]>()
    };
    let mut stored = PackedPanel::new();
    band.to_hbm_view(&mut device.tdma, stored.view_mut());
    stored
}
fn load_product<H: AxisName, I: AxisName, R: AxisName, const B: usize>(
    device: &mut Device,
    products: &HbmTensor<bf16, Chip, StoredProducts<I, R>>,
    packed: &PackedPanel<R, B>,
    scales: &StoredTokens<I, R>,
    panel: usize,
    rows: usize,
) -> Activation<H, Inputs<B>, R, B> {
    if rows > 256 {
        return packed.to_dm(&mut device.tdma);
    }
    let band = products
        .view()
        .tile::<Out<I>, B, m![{Cluster<I>}, {Blocks<I>}, R, {Out<I>} = {B} # (I / 64)]>(64 * panel);
    // SAFETY: packed-panel coordinates preserve each intermediate owner's selected band.
    let band = unsafe {
        band.reshape::<Chip,
        m![{m![{Cluster<Inputs<B>>}, {Blocks<Inputs<B>>}]}, R, {Symbol<Inputs<B>>} % {B} # (I / 64)]>()
    };
    let mut values: DmTensor<
        bf16,
        Chip,
        Cluster<Inputs<B>>,
        DownSlice<H, Inputs<B>>,
        m![R, {Symbol<Inputs<B>>} % {B}],
    > = DmTensor::new();
    load_rows(device, band, values.view_mut(), rows);
    let scales = unsafe { scales.view().reshape::<Chip, m![{Cluster<Inputs<B>>}, R]>() };
    let mut factors: DmTensor<f32, Chip, Cluster<Inputs<B>>, DownSlice<H, Inputs<B>>, m![R, 1]> =
        DmTensor::new();
    // SAFETY: the unit feature digit does not change token-scale storage.
    let source = unsafe { scales.reshape::<Chip, m![{Cluster<Inputs<B>>}, R, 1]>() };
    load_rows(device, source, factors.view_mut(), rows);
    let scales: DmTensor<f32, Chip, Cluster<Inputs<B>>, DownSlice<H, Inputs<B>>, m![R]> =
        unsafe { factors.reshape() };
    let scale: VrfTensor<f32, Chip, Cluster<Inputs<B>>, DownSlice<H, Inputs<B>>, m![R]> = device
        .sub
        .begin(scales.view())
        .fetch::<m![R / 8], m![R % 8]>()
        .collect::<m![R / 8], m![R % 8]>()
        .to_vrf();
    type Quantizer<R, const B: usize> = Quantization<
        m![R, {Symbol<Inputs<B>>} % {B} / 8],
        m![R, {Symbol<Inputs<B>>} % {B} / 4],
        Inputs<B>,
        m![R],
        m![1 # 256],
        m![R / 4],
        m![R % 4],
    >;
    let mut codes = Activation::new();
    Quantizer::<R, B>::rows(&mut device.main, values.view(), &scale)
        .commit_prefix::<m![R], _, _>(codes.view_mut(), rows.next_multiple_of(8));
    codes
}
fn project_panel<H: AxisName, I: AxisName, R: AxisName, const B: usize>(
    device: &mut Device,
    weight: &Down<H, Inputs<B>, B>,
    input: &Activation<H, Inputs<B>, R, B>,
    row: &PanelScale<H, I, R>,
    channel: &down::ChannelScales<H, I>,
    rows: usize,
) -> Partial<H, I, R> {
    let partial = down::trimmed(device, weight, input, rows);
    // SAFETY: both cluster axes name the same intermediate input halves; outputs are unchanged.
    let partial: Partial<H, I, R> = unsafe { partial.reshape() };
    down::Rescale::<R>(core::marker::PhantomData)
        .forward(device, (&partial, (row, channel), rows.div_ceil(8)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::axis;
    use crate::tensor::live::{self, Live};
    use crate::tensor::scalar::Value;
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;

    axes![H = 4096, I = 14336];
    type R = axis::Rows<512>;
    type Input = HbmTensor<f8e4m3, Chip, m![R, I]>;
    type Output = HbmTensor<bf16, Chip, m![{Cluster<I>}, R, H]>;

    #[device(chip = 1)]
    fn staged_products(
        device: &mut Device,
        weight: &DownWeights<H, I>,
        gate: &HbmTensor<bf16, Chip, m![R, I]>,
        up: &HbmTensor<bf16, Chip, m![R, I]>,
        count: &Value,
    ) -> (PackedPanel<R, 64>, PackedPanel<R, 32>, StoredTokens<I, R>) {
        let place = |device: &mut Device, input: &HbmTensor<bf16, Chip, m![R, I]>| {
            let mut output = Projection::<I, R>::new();
            // The projection's row labels interleave eight-row groups across slices.
            let rows = unsafe {
                output.view_mut().reshape::<Chip, Cluster<I>,
                    m![{Blocks<I>}, R % 64 / 8],
                    m![R / 128, R % 128 / 64, R % 8, {Out<I>}]>()
            };
            input.view().to_dm_view(&mut device.tdma, rows);
            output
        };
        let gate = place(device, gate);
        let up = place(device, up);
        let live = Live::<(), R>::read(count);
        let staged = weight.forward(device, Begin(live.with((gate, up))));
        let first = load_product::<H, I, R, 64>(
            device,
            &staged.products,
            &staged.panels.0,
            &staged.scales,
            0,
            live.rows(),
        );
        let tail = load_product::<H, I, R, 32>(
            device,
            &staged.products,
            &staged.panels.3,
            &staged.scales,
            3,
            live.rows(),
        );
        let mut first_stored = PackedPanel::<R, 64>::new();
        first
            .view()
            .slice_tile::<m![H / (H / 8)], 1, m![H / (H / 8) = 1 # 8, {Blocks<Inputs<64>>}]>(0)
            .to_hbm_view(&mut device.tdma, first_stored.view_mut());
        let mut tail_stored = PackedPanel::<R, 32>::new();
        tail.view()
            .slice_tile::<m![H / (H / 8)], 1, m![H / (H / 8) = 1 # 8, {Blocks<Inputs<32>>}]>(0)
            .to_hbm_view(&mut device.tdma, tail_stored.view_mut());
        (first_stored, tail_stored, staged.scales)
    }

    /// Both transfer routes quantize panels with one token scale, including a maximum in the tail.
    #[tokio::test(flavor = "current_thread")]
    async fn products_preserve_token_scales_and_live_rows() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(staged_products.topology()).unwrap();
        let codes = HostTensor::<f8e4m3, m![H, I]>::from_vec(vec![
            f8e4m3::from_f32(0.0);
            H::SIZE * I::SIZE
        ])
        .into_vec();
        let full = packed_down::<H, I, 64, 3>(&codes, 0)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let tail = packed_down::<H, I, 32, 1>(&codes, 192)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let codes = (full, tail);
        let scales = HostTensor::<f32, m![H]>::from_vec(vec![1.0; H::SIZE])
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let weight = DownWeights::new(codes, scales);
        let value = |input: usize| {
            if input == I::SIZE - 1 {
                448.0
            } else {
                (input % 11 + 1) as f32
            }
        };
        for rows in [
            1, 8, 9, 16, 17, 24, 25, 32, 33, 64, 65, 128, 129, 256, 257, 512,
        ] {
            let tensor = |gate: bool| {
                HostTensor::<bf16, m![R, I]>::from_vec((0..R::SIZE * I::SIZE).map(|at| {
                    let row = at / I::SIZE;
                    bf16::from_f32(if row >= rows {
                        f32::NAN
                    } else if gate {
                        32.0
                    } else {
                        value(at % I::SIZE) * (1.0 + (row % 7) as f32 / 8.0) / 32.0
                    })
                }))
            };
            let gate = tensor(true).to_hbm(&mut device.pdma).await.unwrap();
            let up = tensor(false).to_hbm(&mut device.pdma).await.unwrap();
            let count = live::tests::count(rows)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let (first, tail, scales) =
                launch(staged_products, (&mut device, &weight, &gate, &up, &count))
                    .await
                    .unwrap();
            let scales = scales
                .to_host::<m![{Cluster<I>}, R]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            for cluster in 0..2 {
                for row in 0..rows {
                    assert_eq!(
                        scales[cluster * R::SIZE + row],
                        1.0 + (row % 7) as f32 / 8.0,
                        "rows={rows} cluster={cluster} row={row}"
                    );
                }
            }
            let first = first.to_host::<m![{Cluster<Inputs<64>>}, {Blocks<Inputs<64>>}, R, {Symbol<Inputs<64>>} % 64]>(&mut device.pdma).await.unwrap().into_vec();
            let tail = tail.to_host::<m![{Cluster<Inputs<32>>}, {Blocks<Inputs<32>>}, R, {Symbol<Inputs<32>>} % 32]>(&mut device.pdma).await.unwrap().into_vec();
            for (codes, offset, width) in [(first, 0, 64), (tail, 192, 32)] {
                for owner in 0..64 {
                    for row in 0..rows {
                        for feature in 0..width {
                            let at = (owner * R::SIZE + row) * width + feature;
                            assert_eq!(
                                codes[at].to_f32(),
                                value(owner * 224 + offset + feature),
                                "rows={rows} owner={owner} row={row} feature={feature} offset={offset}"
                            );
                        }
                    }
                }
            }
        }
    }

    fn project<const B: usize>(
        device: &mut Device,
        weight: &DownWeights<H, I>,
        input: &Input,
        rows: usize,
        panel: usize,
    ) -> Output {
        let weight = down::<H, I, B>(device, weight, panel);
        // SAFETY: split the complete intermediate width into the same cluster and 224-feature
        // owner runs as the weights; the selected B-feature bands retain their original strides.
        let input = unsafe {
            input
                .view()
                .reshape::<Chip, m![R, {Cluster<I>}, {Blocks<I>}, {Out<I>}]>()
        };
        let input = input
            .tile::<Out<I>, B, m![R, {Cluster<I>}, {Blocks<I>}, {Out<I>} = {B} # (I / 64)]>(
                64 * panel,
            );
        let input = unsafe {
            input.reshape::<Chip, m![R, {Cluster<Inputs<B>>}, {Blocks<Inputs<B>>}, {Symbol<Inputs<B>>} % {B} # (I / 64)]>()
        };
        let input: Activation<H, Inputs<B>, R, B> = input.to_dm(&mut device.tdma);
        let partial = super::super::down::trimmed(device, &weight, &input, rows);
        // SAFETY: the packed-panel and full-intermediate axes name the same input-half clusters.
        let partial: Partial<H, I, R> = unsafe { partial.reshape() };
        partial.to_hbm(&mut device.tdma)
    }

    #[device(chip = 1)]
    fn panels(
        device: &mut Device,
        weight: &DownWeights<H, I>,
        input: &Input,
        count: &Value,
    ) -> (Output, Output, Output, Output) {
        let rows = Live::<(), R>::read(count).rows();
        (
            project::<64>(device, weight, input, rows, 0),
            project::<64>(device, weight, input, rows, 1),
            project::<64>(device, weight, input, rows, 2),
            project::<32>(device, weight, input, rows, 3),
        )
    }

    /// The fourth panel is precisely features 192..224 of every 224-feature owner; dense dots
    /// span all 32 input slices per cluster, with poisoned rows beyond the live prefix.
    #[tokio::test(flavor = "current_thread")]
    async fn native_down_tail_uses_each_owners_last_32_features() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(panels.topology()).unwrap();
        fn coefficient(output: usize, input: usize) -> f32 {
            ((output % 5 + input % 7) % 9 + 1) as f32 / 8.0
        }
        let codes = HostTensor::<f8e4m3, m![H, I]>::from_vec(
            (0..H::SIZE * I::SIZE)
                .map(|at| f8e4m3::from_f32(coefficient(at / I::SIZE, at % I::SIZE))),
        )
        .into_vec();
        let full = packed_down::<H, I, 64, 3>(&codes, 0)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let tail = packed_down::<H, I, 32, 1>(&codes, 192)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let codes = (full, tail);
        let scales = HostTensor::<f32, m![H]>::from_vec(vec![1.0; H::SIZE])
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let weight = DownWeights::new(codes, scales);
        for rows in [1, 8, 9, 16, 17, 24, 25, 32, 33, 257, 512] {
            let input =
                HostTensor::<f8e4m3, m![R, I]>::from_vec((0..R::SIZE * I::SIZE).map(|at| {
                    f8e4m3::from_f32(if at / I::SIZE < rows {
                        (at % I::SIZE % 17 + 1) as f32 / 16.0
                    } else {
                        f32::NAN
                    })
                }))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let count = live::tests::count(rows)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let outputs = launch(panels, (&mut device, &weight, &input, &count))
                .await
                .unwrap();
            let outputs = [outputs.0, outputs.1, outputs.2, outputs.3];
            for (output, (offset, width)) in
                outputs
                    .into_iter()
                    .zip([(0, 64), (64, 64), (128, 64), (192, 32)])
            {
                let actual = output
                    .to_host::<m![{Cluster<I>}, R, H]>(&mut device.pdma)
                    .await
                    .unwrap()
                    .into_vec();
                for cluster in 0..2 {
                    let reference: Vec<_> = (0..5)
                        .map(|out| {
                            let dot: f32 = (0..32)
                                .flat_map(|owner| {
                                    (offset..offset + width).map(move |feature| {
                                        cluster * (I::SIZE / 2) + owner * 224 + feature
                                    })
                                })
                                .map(|i| {
                                    coefficient(out, i)
                                        * f8e4m3::from_f32((i % 17 + 1) as f32 / 16.0).to_f32()
                                })
                                .sum();
                            bf16::from_f32(dot).to_f32()
                        })
                        .collect();
                    for row in 0..rows {
                        for out in 0..H::SIZE {
                            assert_eq!(
                                actual[(cluster * R::SIZE + row) * H::SIZE + out].to_f32(),
                                reference[out % 5],
                                "rows={rows} panel-offset={offset} cluster={cluster} row={row} out={out}"
                            );
                        }
                    }
                }
            }
        }
    }
}
