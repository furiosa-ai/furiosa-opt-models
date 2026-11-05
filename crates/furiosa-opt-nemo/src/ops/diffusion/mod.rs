//! Image patches and adaptive conditioning over typed device tensors.
pub mod ddim;
pub mod flow_match;
pub mod guidance;
pub mod packed;
pub mod sequence;

use crate::Module;
use crate::ops::layout::axes::Single;
use crate::ops::weight::Optional;
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

/// Contiguous f32 feature rows.
pub type Rows<Width, Count> = HbmTensor<f32, m![1], m![Count, Width]>;

/// Contiguous channel-first f32 image.
pub type Image<Channels, Height, Width> = HbmTensor<f32, m![1], m![Channels, Height, Width]>;

/// One row of conditioning features.
pub type Vector<Width> = Rows<Width, Single>;

/// Adaptive normalization shift followed by scale.
pub type Modulation<Width> = (Vector<Width>, Vector<Width>);

/// Attention and feed-forward modulation, each paired with its residual gate.
pub type Gates<Width> = (
    (Modulation<Width>, Vector<Width>),
    (Modulation<Width>, Vector<Width>),
);

/// Number of nonoverlapping patches in a row-major image grid.
#[derive(Clone, Debug)]
pub struct Grid<Height, Width, PatchHeight, PatchWidth>(
    PhantomData<(Height, Width, PatchHeight, PatchWidth)>,
);

impl<H: AxisName, W: AxisName, PH: AxisName, PW: AxisName> AxisName for Grid<H, W, PH, PW> {
    const NAME: Ident = Ident::new("Patches");

    const SIZE: usize = {
        assert!(PH::SIZE > 0 && PW::SIZE > 0);
        assert!(H::SIZE % PH::SIZE == 0 && W::SIZE % PW::SIZE == 0);
        (H::SIZE / PH::SIZE) * (W::SIZE / PW::SIZE)
    };
}

/// Project nonoverlapping image patches into feature rows in row-major grid order.
#[derive(DeviceSend)]
pub struct Patch<
    Channels: AxisName,
    Width: AxisName,
    Height: AxisName,
    Columns: AxisName,
    D: Scalar = f32,
    Bias = HbmTensor<D, m![1], m![Width]>,
> {
    /// Projection coefficients ordered by output feature, channel, patch row and column.
    pub weight: HbmTensor<D, m![1], m![Width, Channels, Height, Columns]>,
    /// The additive bias per output feature, or `()` for none.
    pub bias: Bias,
}

impl<
    C: AxisName,
    O: AxisName,
    PH: AxisName,
    PW: AxisName,
    H: AxisName,
    W: AxisName,
    B: Optional<HbmTensor<f32, m![1], m![O]>>,
> Module<(&mut Device, &Image<C, H, W>)> for Patch<C, O, PH, PW, f32, B>
{
    type Output = Rows<O, Grid<H, W, PH, PW>>;

    fn forward(&self, _: (&mut Device, &Image<C, H, W>)) -> Self::Output {
        todo!(
            "NPU nonoverlapping image patch projection with optional bias in row-major grid order"
        )
    }
}

/// Two-dimensional sine/cosine patch positions, with half the features per coordinate.
#[derive(DeviceSend)]
pub struct Positions<Width, PatchHeight, PatchWidth>(
    pub PhantomData<(Width, PatchHeight, PatchWidth)>,
);

impl<Width, PatchHeight, PatchWidth> Default for Positions<Width, PatchHeight, PatchWidth> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<C: AxisName, O: AxisName, PH: AxisName, PW: AxisName, H: AxisName, W: AxisName>
    Module<(&mut Device, &Image<C, H, W>)> for Positions<O, PH, PW>
{
    type Output = Rows<O, Grid<H, W, PH, PW>>;

    fn forward(&self, _: (&mut Device, &Image<C, H, W>)) -> Self::Output {
        const { assert!(O::SIZE % 4 == 0) };
        todo!("NPU two-dimensional sine/cosine positions with half the channels per coordinate")
    }
}

/// Timestep cosine/sine embedding with period 10000 and a half-width frequency divisor.
#[derive(DeviceSend)]
pub struct Time<Width>(pub PhantomData<Width>);

impl<Width> Default for Time<Width> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<W: AxisName> Module<(&mut Device, &HbmTensor<f32, m![1], m![1]>)> for Time<W> {
    type Output = Vector<W>;

    fn forward(&self, _: (&mut Device, &HbmTensor<f32, m![1], m![1]>)) -> Self::Output {
        const { assert!(W::SIZE % 2 == 0) };
        todo!(
            "NPU timestep cosine/sine embedding with period 10000 and half-width frequency divisor"
        )
    }
}

/// Split attention shift/scale/gate followed by feed-forward shift/scale/gate.
#[derive(DeviceSend)]
pub struct SplitGates<Width>(pub PhantomData<Width>);

impl<Width> Default for SplitGates<Width> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<W: AxisName, Packed: AxisName> Module<(&mut Device, &Vector<Packed>)> for SplitGates<W> {
    type Output = Gates<W>;

    fn forward(&self, _: (&mut Device, &Vector<Packed>)) -> Self::Output {
        const { assert!(Packed::SIZE == 6 * W::SIZE) };
        todo!("NPU split of attention and feedforward shift, scale, and residual gate")
    }
}

/// The order in which a conditioning payload packs a modulation's shift and scale.
pub trait ModulationOrder {}

/// Shift, then scale.
pub struct ShiftFirst;

/// Scale, then shift.
pub struct ScaleFirst;

impl ModulationOrder for ShiftFirst {}

impl ModulationOrder for ScaleFirst {}

/// Split conditioning packed in `Order` into `(shift, scale)`.
#[derive(DeviceSend)]
pub struct SplitModulation<Width, Order = ShiftFirst>(pub PhantomData<(Width, Order)>);

impl<Width, Order> Default for SplitModulation<Width, Order> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<W: AxisName, Packed: AxisName, Order: ModulationOrder> Module<(&mut Device, &Vector<Packed>)>
    for SplitModulation<W, Order>
{
    type Output = Modulation<W>;

    fn forward(&self, _: (&mut Device, &Vector<Packed>)) -> Self::Output {
        const { assert!(Packed::SIZE == 2 * W::SIZE) };
        todo!(
            "NPU split of adaptive normalization into (shift, scale), respecting SCALE_FIRST input order"
        )
    }
}

/// Split conditioning into normalization shift, scale and residual gate.
#[derive(DeviceSend)]
pub struct SplitGate<Width>(pub PhantomData<Width>);

impl<Width> Default for SplitGate<Width> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<W: AxisName, Packed: AxisName> Module<(&mut Device, &Vector<Packed>)> for SplitGate<W> {
    type Output = (Modulation<W>, Vector<W>);

    fn forward(&self, _: (&mut Device, &Vector<Packed>)) -> Self::Output {
        const { assert!(Packed::SIZE == 3 * W::SIZE) };
        todo!("NPU split of normalization shift, scale, and residual gate")
    }
}

/// Apply affine-free layer norm, then multiply by `1 + scale` and add shift.
#[derive(DeviceSend)]
pub struct AdaptiveNorm {
    /// Stabilizer added to the normalization variance.
    pub epsilon: HbmTensor<f32, m![1], m![1]>,
}

impl<W: AxisName, R: AxisName> Module<(&mut Device, (&Rows<W, R>, &Modulation<W>))>
    for AdaptiveNorm
{
    type Output = Rows<W, R>;

    fn forward(&self, _: (&mut Device, (&Rows<W, R>, &Modulation<W>))) -> Self::Output {
        todo!("NPU affine-free layer norm followed by (1 + scale) multiplication and shift")
    }
}

/// Add a residual multiplied by a broadcast scalar or per-channel gate.
#[derive(Default, DeviceSend)]
pub struct GatedAdd;

impl<W: AxisName, R: AxisName> Module<(&mut Device, (&Rows<W, R>, &Rows<W, R>, &Vector<W>))>
    for GatedAdd
{
    type Output = Rows<W, R>;

    fn forward(&self, _: (&mut Device, (&Rows<W, R>, &Rows<W, R>, &Vector<W>))) -> Self::Output {
        todo!("NPU residual addition with a broadcast per-channel gate")
    }
}

impl<D: Scalar, W: AxisName, R: AxisName>
    Module<(
        &mut Device,
        (
            &HbmTensor<D, m![1], m![R, W]>,
            &HbmTensor<D, m![1], m![R, W]>,
            &HbmTensor<f32, m![1], m![1]>,
        ),
    )> for GatedAdd
{
    type Output = HbmTensor<D, m![1], m![R, W]>;

    fn forward(
        &self,
        _: (
            &mut Device,
            (
                &HbmTensor<D, m![1], m![R, W]>,
                &HbmTensor<D, m![1], m![R, W]>,
                &HbmTensor<f32, m![1], m![1]>,
            ),
        ),
    ) -> Self::Output {
        todo!("NPU residual addition with a broadcast scalar gate")
    }
}

/// Patch row, patch column, then channel.
pub struct ChannelLast;

/// Channel, patch row, then patch column.
pub struct ChannelFirst;

/// Restore CHW images from row-major patches whose payload is packed in `Order`.
#[derive(DeviceSend)]
pub struct Unpatch<Channels, Height, Width, PatchHeight, PatchWidth, Order = ChannelLast>(
    pub PhantomData<(Channels, Height, Width, PatchHeight, PatchWidth, Order)>,
);

impl<Channels, Height, Width, PatchHeight, PatchWidth, Order> Default
    for Unpatch<Channels, Height, Width, PatchHeight, PatchWidth, Order>
{
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<C: AxisName, H: AxisName, W: AxisName, PH: AxisName, PW: AxisName, Packed: AxisName>
    Module<(&mut Device, &Rows<Packed, Grid<H, W, PH, PW>>)>
    for Unpatch<C, H, W, PH, PW, ChannelLast>
{
    type Output = Image<C, H, W>;

    fn forward(&self, _: (&mut Device, &Rows<Packed, Grid<H, W, PH, PW>>)) -> Self::Output {
        const { assert!(Packed::SIZE == C::SIZE * PH::SIZE * PW::SIZE) };
        todo!("NPU row-major patches to CHW; payload is patch-height, patch-width, channel")
    }
}

/// The inverse of [`Pack`]'s permuting copy.
impl<C: AxisName, H: AxisName, W: AxisName, PH: AxisName, PW: AxisName, Packed: AxisName>
    Module<(&mut Device, &Rows<Packed, Grid<H, W, PH, PW>>)>
    for Unpatch<C, H, W, PH, PW, ChannelFirst>
{
    type Output = Image<C, H, W>;

    fn forward(
        &self,
        (device, rows): (&mut Device, &Rows<Packed, Grid<H, W, PH, PW>>),
    ) -> Self::Output {
        const { assert!(Packed::SIZE == C::SIZE * PH::SIZE * PW::SIZE) };
        let mut image: Self::Output = HbmTensor::new();
        // SAFETY: as for [`Pack`].
        unsafe {
            rows.view()
                .reshape::<m![1], m![H / PH, W / PW, C, H % PH, W % PW]>()
        }
        .to_hbm_view(&mut device.tdma, image.view_mut());
        image
    }
}

/// Convert decoded image values from [-1, 1] into bounded [0, 1] samples.
#[derive(DeviceSend)]
pub struct Denormalize;

impl<S: M + 'static> Module<(&mut Device, &HbmTensor<f32, m![1], S>)> for Denormalize {
    type Output = HbmTensor<f32, m![1], S>;

    fn forward(&self, (device, input): (&mut Device, &HbmTensor<f32, m![1], S>)) -> Self::Output {
        crate::ops::pointwise::Unary::rows::<_, _, { crate::ops::pointwise::Unary::Denormalize }, _>(
            device, input,
        )
    }
}

/// Rearrange CHW images into nonoverlapping row-major patches whose payload is packed in `Order`.
#[derive(DeviceSend)]
pub struct Pack<Width, PatchHeight, PatchWidth, Order = ChannelLast>(
    pub std::marker::PhantomData<(Width, PatchHeight, PatchWidth, Order)>,
);

impl<C: AxisName, H: AxisName, W: AxisName, O: AxisName, PH: AxisName, PW: AxisName>
    Module<(&mut Device, &Image<C, H, W>)> for Pack<O, PH, PW, ChannelLast>
{
    type Output = Rows<O, Grid<H, W, PH, PW>>;

    fn forward(&self, _: (&mut Device, &Image<C, H, W>)) -> Self::Output {
        const { assert!(O::SIZE == C::SIZE * PH::SIZE * PW::SIZE) };
        todo!("NPU CHW to row-major patches; payload is patch-height, patch-width, channel")
    }
}

/// One permuting copy: patch `(h, w)` of channel `c` at row `h * (W / PW) + w`, feature
/// `(c * PH + ph) * PW + pw`.
impl<C: AxisName, H: AxisName, W: AxisName, O: AxisName, PH: AxisName, PW: AxisName>
    Module<(&mut Device, &Image<C, H, W>)> for Pack<O, PH, PW, ChannelFirst>
{
    type Output = Rows<O, Grid<H, W, PH, PW>>;

    fn forward(&self, (device, image): (&mut Device, &Image<C, H, W>)) -> Self::Output {
        const { assert!(O::SIZE == C::SIZE * PH::SIZE * PW::SIZE) };
        let mut rows: Self::Output = HbmTensor::new();
        image.view().to_hbm_view(
            &mut device.tdma,
            // SAFETY: a patch row is its grid row then column, its features channel, patch row
            // and patch column, each in row-major order.
            unsafe {
                rows.view_mut()
                    .reshape::<m![1], m![H / PH, W / PW, C, H % PH, W % PW]>()
            },
        );
        rows
    }
}

/// Apply elementwise `input * scale + shift` with scalar broadcast.
#[derive(DeviceSend)]
pub struct Rescale {
    /// Multiplicative scalar.
    pub scale: HbmTensor<f32, m![1], m![1]>,
    /// Additive scalar.
    pub shift: HbmTensor<f32, m![1], m![1]>,
}

impl<S: M> Module<(&mut Device, &HbmTensor<f32, m![1], S>)> for Rescale {
    type Output = HbmTensor<f32, m![1], S>;

    fn forward(&self, _: (&mut Device, &HbmTensor<f32, m![1], S>)) -> Self::Output {
        todo!("NPU elementwise input * scale + shift with scalar broadcast")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use furiosa_opt_std::runtime::launch;

    axes![
        Channels = 24,
        Height = 8,
        Width = 12,
        Features = 96,
        Two = 2
    ];

    type Patches = Grid<Height, Width, Two, Two>;

    #[device(chip = 1)]
    fn packed(
        device: &mut Device,
        image: &Image<Channels, Height, Width>,
    ) -> Rows<Features, Patches> {
        Pack::<Features, Two, Two, ChannelFirst>(PhantomData).forward((device, image))
    }

    /// Row `(h, w)` feature `(c, ph, pw)` of the packed patches is pixel `(c, 2h + ph, 2w + pw)`.
    #[tokio::test(flavor = "current_thread")]
    async fn packs_channel_first_patches() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let (c, h, w) = (Channels::SIZE, Height::SIZE, Width::SIZE);
        let pixels: Vec<f32> = (0..c * h * w).map(|index| index as f32).collect();
        let image = HostTensor::<f32, m![Channels, Height, Width]>::from_vec(pixels)
            .to_hbm::<m![1], m![Channels, Height, Width]>(&mut device.pdma)
            .await
            .unwrap();
        let rows = launch(packed, (&mut device, &image))
            .await
            .unwrap()
            .to_host::<m![Patches, Features]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (index, got) in rows.iter().enumerate() {
            let (patch, feature) = (index / Features::SIZE, index % Features::SIZE);
            let (row, column) = (patch / (w / 2), patch % (w / 2));
            let (channel, dy, dx) = (feature / 4, feature / 2 % 2, feature % 2);
            let pixel = (channel * h + 2 * row + dy) * w + 2 * column + dx;
            assert_eq!(*got, pixel as f32, "patch {patch} feature {feature}");
        }
    }

    #[device(chip = 1)]
    fn unpacked(
        device: &mut Device,
        rows: &Rows<Features, Patches>,
    ) -> Image<Channels, Height, Width> {
        Unpatch::<Channels, Height, Width, Two, Two, ChannelFirst>(PhantomData)
            .forward((device, rows))
    }

    /// Unpatching channel-first patches restores the image they were packed from.
    #[tokio::test(flavor = "current_thread")]
    async fn unpacks_what_it_packs() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let count = Channels::SIZE * Height::SIZE * Width::SIZE;
        let pixels: Vec<f32> = (0..count)
            .map(|index| (index * 7 % 1009) as f32 - 3.5)
            .collect();
        let image = HostTensor::<f32, m![Channels, Height, Width]>::from_vec(pixels.clone())
            .to_hbm::<m![1], m![Channels, Height, Width]>(&mut device.pdma)
            .await
            .unwrap();
        let rows = launch(packed, (&mut device, &image)).await.unwrap();
        let restored = launch(unpacked, (&mut device, &rows))
            .await
            .unwrap()
            .to_host::<m![Channels, Height, Width]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        assert_eq!(restored, pixels);
    }
}
