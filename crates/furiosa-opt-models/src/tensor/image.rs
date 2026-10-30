//! FP32 feature rows and channel-first images, the patch grid an image is cut into, and the
//! permuting copies between an image and its patch rows.

use crate::Module;
use crate::tensor::axis::Single;
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

/// Contiguous f32 feature rows.
pub type Rows<Width, Count> = HbmTensor<f32, m![1], m![Count, Width]>;

/// Contiguous channel-first f32 image.
pub type Image<Channels, Height, Width> = HbmTensor<f32, m![1], m![Channels, Height, Width]>;

/// One row of conditioning features.
pub type Vector<Width> = Rows<Width, Single>;

/// Number of nonoverlapping patches in a row-major image grid.
#[derive(Clone, Debug)]
pub struct Grid<Height, Width, PatchHeight, PatchWidth>(
    PhantomData<(Height, Width, PatchHeight, PatchWidth)>,
);

impl<Height: AxisName, Width: AxisName, PatchHeight: AxisName, PatchWidth: AxisName> AxisName
    for Grid<Height, Width, PatchHeight, PatchWidth>
{
    const NAME: Ident = Ident::new("Patches");

    const SIZE: usize = {
        assert!(PatchHeight::SIZE > 0 && PatchWidth::SIZE > 0);
        assert!(Height::SIZE % PatchHeight::SIZE == 0 && Width::SIZE % PatchWidth::SIZE == 0);
        (Height::SIZE / PatchHeight::SIZE) * (Width::SIZE / PatchWidth::SIZE)
    };
}

/// Patch row, patch column, then channel.
pub struct ChannelLast;

/// Channel, patch row, then patch column.
pub struct ChannelFirst;

/// Rearrange CHW images into nonoverlapping row-major patches whose payload is packed in `Order`.
#[derive(DeviceSend)]
pub struct Pack<Width, PatchHeight, PatchWidth, Order = ChannelLast>(
    pub PhantomData<(Width, PatchHeight, PatchWidth, Order)>,
);

impl<
    C: AxisName,
    H: AxisName,
    W: AxisName,
    Width: AxisName,
    PatchHeight: AxisName,
    PatchWidth: AxisName,
    Env,
> Module<&Image<C, H, W>, Env, &mut ()> for Pack<Width, PatchHeight, PatchWidth, ChannelLast>
{
    type Output = Rows<Width, Grid<H, W, PatchHeight, PatchWidth>>;

    fn forward(&self, _: &mut Device, _: Env, _: &mut (), _: &Image<C, H, W>) -> Self::Output {
        const { assert!(Width::SIZE == C::SIZE * PatchHeight::SIZE * PatchWidth::SIZE) };
        todo!("NPU CHW to row-major patches; payload is patch-height, patch-width, channel")
    }
}

/// One permuting copy: patch `(h, w)` of channel `c` at row `h * (W / PatchWidth) + w`, feature
/// `(c * PatchHeight + ph) * PatchWidth + pw`.
impl<
    C: AxisName,
    H: AxisName,
    W: AxisName,
    Width: AxisName,
    PatchHeight: AxisName,
    PatchWidth: AxisName,
    Env,
> Module<&Image<C, H, W>, Env, &mut ()> for Pack<Width, PatchHeight, PatchWidth, ChannelFirst>
{
    type Output = Rows<Width, Grid<H, W, PatchHeight, PatchWidth>>;

    fn forward(
        &self,
        device: &mut Device,
        _: Env,
        _: &mut (),
        image: &Image<C, H, W>,
    ) -> Self::Output {
        const { assert!(Width::SIZE == C::SIZE * PatchHeight::SIZE * PatchWidth::SIZE) };
        let mut rows: Self::Output = HbmTensor::new();
        image.view().to_hbm_view(
            &mut device.tdma,
            // SAFETY: a patch row is its grid row then column, its features channel, patch row
            // and patch column, each in row-major order.
            unsafe {
                rows.view_mut().reshape::<m![1], m![
                    H / PatchHeight,
                    W / PatchWidth,
                    C,
                    H % PatchHeight,
                    W % PatchWidth
                ]>()
            },
        );
        rows
    }
}

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

impl<
    Channels: AxisName,
    Height: AxisName,
    Width: AxisName,
    PatchHeight: AxisName,
    PatchWidth: AxisName,
    Packed: AxisName,
    Env,
> Module<&Rows<Packed, Grid<Height, Width, PatchHeight, PatchWidth>>, Env, &mut ()>
    for Unpatch<Channels, Height, Width, PatchHeight, PatchWidth, ChannelLast>
{
    type Output = Image<Channels, Height, Width>;

    fn forward(
        &self,
        _: &mut Device,
        _: Env,
        _: &mut (),
        _: &Rows<Packed, Grid<Height, Width, PatchHeight, PatchWidth>>,
    ) -> Self::Output {
        const { assert!(Packed::SIZE == Channels::SIZE * PatchHeight::SIZE * PatchWidth::SIZE) };
        todo!("NPU row-major patches to CHW; payload is patch-height, patch-width, channel")
    }
}

/// The inverse of [`Pack`]'s permuting copy.
impl<
    Channels: AxisName,
    Height: AxisName,
    Width: AxisName,
    PatchHeight: AxisName,
    PatchWidth: AxisName,
    Packed: AxisName,
    Env,
> Module<&Rows<Packed, Grid<Height, Width, PatchHeight, PatchWidth>>, Env, &mut ()>
    for Unpatch<Channels, Height, Width, PatchHeight, PatchWidth, ChannelFirst>
{
    type Output = Image<Channels, Height, Width>;

    fn forward(
        &self,
        device: &mut Device,
        _: Env,
        _: &mut (),
        rows: &Rows<Packed, Grid<Height, Width, PatchHeight, PatchWidth>>,
    ) -> Self::Output {
        const { assert!(Packed::SIZE == Channels::SIZE * PatchHeight::SIZE * PatchWidth::SIZE) };
        let mut image: Self::Output = HbmTensor::new();
        // SAFETY: as for [`Pack`].
        unsafe {
            rows.view().reshape::<m![1], m![
                Height / PatchHeight,
                Width / PatchWidth,
                Channels,
                Height % PatchHeight,
                Width % PatchWidth
            ]>()
        }
        .to_hbm_view(&mut device.tdma, image.view_mut());
        image
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::DEVICE;
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
        Pack::<Features, Two, Two, ChannelFirst>(PhantomData).forward(device, (), &mut (), image)
    }

    /// Row `(h, w)` feature `(c, ph, pw)` of the packed patches is pixel `(c, 2h + ph, 2w + pw)`.
    #[tokio::test(flavor = "current_thread")]
    async fn packs_channel_first_patches() {
        let _lease = DEVICE.lock().await;
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
        Unpatch::<Channels, Height, Width, Two, Two, ChannelFirst>(PhantomData).forward(
            device,
            (),
            &mut (),
            rows,
        )
    }

    /// Unpatching channel-first patches restores the image they were packed from.
    #[tokio::test(flavor = "current_thread")]
    async fn unpacks_what_it_packs() {
        let _lease = DEVICE.lock().await;
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
