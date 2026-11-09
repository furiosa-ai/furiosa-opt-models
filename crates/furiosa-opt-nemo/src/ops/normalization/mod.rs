//! Per-row normalization with learned feature scales and FP32 statistics.

mod kernel;
pub(crate) mod span;
pub mod staging;

use furiosa_opt_std::prelude::*;

use crate::compose::Space;
use crate::ops::layout::{Dm, axes::Single};
use crate::{Module, ops::weight::Weight};

/// Feature slices retain every row in both clusters for the projections that consume them.
pub type Cluster = m![2];

/// Smallest multiple of 16 dividing Width into at most 256 hardware slices.
#[derive(Clone, Debug)]
pub struct Chunk<Width>(core::marker::PhantomData<Width>);

impl<Width: AxisName> AxisName for Chunk<Width> {
    const NAME: Ident = Ident::new("Chunk");

    const SIZE: usize = {
        assert!(Width::SIZE > 0 && Width::SIZE.is_multiple_of(16));
        let mut chunk = 16;
        while Width::SIZE / chunk > 256 || Width::SIZE % chunk != 0 {
            chunk += 16;
        }
        chunk
    };
}

/// Feature chunks broadcast across the 256 hardware slices.
pub type FeatureSlices<Width> = m![Width / ({ Symbol<Chunk<Width>> }) # 256];

/// Each row's contiguous features within one chunk.
pub type Element<Width, Rows> = m![Rows, Width % ({ Symbol<Chunk<Width>> })];

/// Hidden rows share the feature placement of the following projections.
pub type Spread<Width, Rows, Chips = m![1]> =
    DmTensor<bf16, Chips, Cluster, FeatureSlices<Width>, Element<Width, Rows>>;

/// FP32 bits of the epsilon an [`RmsNorm`] adds when its type names none.
pub const EPSILON: usize = 1e-6_f32.to_bits() as usize;

/// BF16 rows times learned weights, divided by sqrt(mean(x²) + epsilon).
/// EPSILON stores the FP32 bit pattern in a usize const parameter.
#[derive(DeviceSend)]
pub struct RmsNorm<
    Width: AxisName,
    const EPSILON: usize = { self::EPSILON },
    Chips: M = m![1],
    Parameter = Weight<m![Width], Chips>,
> {
    weight: Parameter,
    features: core::marker::PhantomData<(Width, Chips)>,
}

impl<Width: AxisName, const EPSILON: usize, Chips: M, Parameter>
    RmsNorm<Width, EPSILON, Chips, Parameter>
{
    /// Retain one learned BF16 scale per feature on the same chip mapping as the input.
    pub const fn new(weight: Parameter) -> Self {
        Self {
            weight,
            features: core::marker::PhantomData,
        }
    }

    /// Borrow the retained per-feature scale.
    pub fn weight(&self) -> &Parameter {
        &self.weight
    }

    pub fn weight_mut(&mut self) -> &mut Parameter {
        &mut self.weight
    }
}

/// A norm's learned per-feature scale, in a form the norm reads into each feature slice.
pub trait Learned<Width: AxisName, Chips: M> {
    fn lanes(&self, device: &mut Device) -> kernel::Features<Width, Chips>;
}

/// A norm gain landed in the feature slices of the rows it scales.
pub type Landed<Width, Chips = m![1]> =
    DmTensor<bf16, Chips, Cluster, FeatureSlices<Width>, m![Width % ({ Symbol<Chunk<Width>> })]>;

impl<Width: AxisName, Chips: M> Learned<Width, Chips> for Landed<Width, Chips> {
    fn lanes(&self, device: &mut Device) -> kernel::Features<Width, Chips> {
        device
            .sub
            .begin(self.view())
            .fetch::<m![Width % ({ Symbol<Chunk<Width>> }) / 8], m![Width % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Width % ({ Symbol<Chunk<Width>> }) / 8], m![Width % 8]>()
            .to_vrf()
    }
}

/// The retained HBM gain lands in the slices each time a norm reads it.
impl<Width: AxisName, Chips: M> Learned<Width, Chips> for Weight<m![Width], Chips> {
    fn lanes(&self, device: &mut Device) -> kernel::Features<Width, Chips> {
        let landed: Landed<Width, Chips> = self.tensor().to_dm(&mut device.tdma);
        landed.lanes(device)
    }
}

impl<Width: AxisName, const EPSILON: usize, Shape: M, Chips: M>
    RmsNorm<Width, EPSILON, Chips, Weight<Shape, Chips>>
{
    /// The same norm with its retained gain landed once in the DM mapping its forward reads.
    pub fn stage<Cl: M, S: M, E: M>(
        &self,
        device: &mut Device,
    ) -> RmsNorm<Width, EPSILON, Chips, DmTensor<bf16, Chips, Cl, S, E>> {
        RmsNorm::new(self.weight.tensor().to_dm(&mut device.tdma))
    }
}

impl<'d, 'x, Width: AxisName, Rows: AxisName, const EPSILON: usize, Chips: M, G>
    Module<(&'d mut Device, &'x Spread<Width, Rows, Chips>)> for RmsNorm<Width, EPSILON, Chips, G>
where
    G: Learned<Width, Chips>,
{
    type Output = Spread<Width, Rows, Chips>;

    fn forward(
        &self,
        (device, input): (&'d mut Device, &'x Spread<Width, Rows, Chips>),
    ) -> Self::Output {
        kernel::rms_norm::<Width, m![Rows], EPSILON, Chips, _>(device, input.view(), &self.weight)
            .commit()
    }
}

/// Every row of the envelope is normalized alone, so the live rows land what they land in a
/// launch of their own.
impl<'d, 'x, Width: AxisName, S: Space, const EPSILON: usize, Chips: M, G>
    Module<(&'d mut Device, &'x Dm<S, Spread<Width, S::Envelope, Chips>>)>
    for RmsNorm<Width, EPSILON, Chips, G>
where
    G: Learned<Width, Chips>,
{
    type Output = Dm<S, Spread<Width, S::Envelope, Chips>>;

    fn forward(
        &self,
        (device, input): (&'d mut Device, &'x Dm<S, Spread<Width, S::Envelope, Chips>>),
    ) -> Self::Output {
        Dm {
            value: self.forward((device, &input.value)),
            tiles: input.tiles,
        }
    }
}

impl<Width: AxisName, const EPSILON: usize, Chips: M>
    Module<(&mut Device, &HbmTensor<bf16, Chips, m![Width]>)> for RmsNorm<Width, EPSILON, Chips>
{
    type Output = HbmTensor<bf16, Chips, m![Width]>;

    fn forward(
        &self,
        (device, input): (&mut Device, &HbmTensor<bf16, Chips, m![Width]>),
    ) -> Self::Output {
        let input: Spread<Width, Single, Chips> = input.to_dm(&mut device.tdma);
        self.forward((&mut *device, &input))
            .to_hbm(&mut device.tdma)
    }
}

impl<Width: AxisName, Rows: AxisName, const EPSILON: usize, Chips: M>
    Module<(&mut Device, &HbmTensor<bf16, Chips, m![Rows, Width]>)>
    for RmsNorm<Width, EPSILON, Chips>
{
    type Output = HbmTensor<bf16, Chips, m![Rows, Width]>;

    fn forward(
        &self,
        (device, input): (&mut Device, &HbmTensor<bf16, Chips, m![Rows, Width]>),
    ) -> Self::Output {
        let input: Spread<Width, Rows, Chips> = input.to_dm(&mut device.tdma);
        self.forward((&mut *device, &input))
            .to_hbm(&mut device.tdma)
    }
}

/// `input * weight / sqrt(mean(input^2) + epsilon)` over each of the `P` head rows of every
/// slice, in two passes: one sums the squares, the other scales, with the square root taken on
/// the way into the divisor register. `EPSILON` holds epsilon's FP32 bits.
impl<'d, 'x, T: Space, S: M, P: M, Head: AxisName, const EPSILON: usize>
    Module<(
        &'d mut Device,
        &'x Dm<T, DmTensor<bf16, m![1], m![1 # 2], S, m![{ P }, Head]>>,
    )> for RmsNorm<Head, EPSILON>
{
    type Output = Dm<T, DmTensor<bf16, m![1], m![1 # 2], S, m![{ P }, Head]>>;

    fn forward(
        &self,
        (device, input): (
            &'d mut Device,
            &'x Dm<T, DmTensor<bf16, m![1], m![1 # 2], S, m![{ P }, Head]>>,
        ),
    ) -> Self::Output {
        Dm {
            value: kernel::heads::<S, P, Head, EPSILON>(device, &input.value, self.weight.tensor()),
            tiles: input.tiles,
        }
    }
}
