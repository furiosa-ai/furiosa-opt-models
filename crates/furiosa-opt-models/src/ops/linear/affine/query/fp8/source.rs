use super::{HeadScales, HeadSquares, Squared, Stored, ValueScale, Weight, Weights};
use crate::Op;
use crate::modules::generic::schedule::Begin;
use crate::ops::linear::Affine;
use crate::tensor::parallel::Chip;
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

pub(crate) struct Panel<'a, Hidden: AxisName, Heads: AxisName, Head: AxisName>(
    pub(crate) &'a Stored<Hidden, Heads, Head>,
);

impl<Hidden: AxisName, Heads: AxisName, Head: AxisName> Op<&mut Weight<Hidden, Heads, Head>>
    for Panel<'_, Hidden, Heads, Head>
{
    type Output = ();

    fn forward(&self, device: &mut Device, slot: &mut Weight<Hidden, Heads, Head>) {
        self.0.view().to_dm_view(&mut device.tdma, slot.view_mut());
    }
}

pub(crate) struct Scales<'a, Hidden: AxisName, Heads: AxisName, Head: AxisName> {
    pub(crate) stored: &'a HbmTensor<f32, Chip, m![Heads, Head]>,
    pub(crate) hidden: PhantomData<Hidden>,
}

impl<Hidden: AxisName, Heads: AxisName, Head: AxisName> Op<&mut ValueScale<Hidden, Heads, Head>>
    for Scales<'_, Hidden, Heads, Head>
{
    type Output = ();

    fn forward(&self, device: &mut Device, slot: &mut ValueScale<Hidden, Heads, Head>) {
        self.stored
            .view()
            .to_dm_view(&mut device.tdma, slot.view_mut());
    }
}

/// One half of the query weight's head dimension, written into its half of the resident weight.
pub(crate) struct Half<'a, Hidden: AxisName, Heads: AxisName, Head: AxisName, const HALF: usize>(
    pub(crate) &'a Stored<Hidden, Heads, Head>,
);

impl<Hidden: AxisName, Heads: AxisName, Head: AxisName, const HALF: usize>
    Op<&mut Affine<Weight<Hidden, Heads, Head>, ()>> for Half<'_, Hidden, Heads, Head, HALF>
{
    type Output = ();

    fn forward(&self, context: &mut Device, output: &mut Affine<Weight<Hidden, Heads, Head>, ()>) {
        const { assert!(Head::SIZE == 128 && HALF < 2) };
        self.0
            .view()
            .tile::<m![Head / 64], 1, m![
                Heads / (Heads / 2), Heads % (Heads / 2) / (Heads / 4),
                Heads % (Heads / 4) / (Heads / 8), Hidden / 64, Head / 64 = 1 # 2,
                Heads % (Heads / 8), Head % 64, Hidden % 64
            ]>(HALF)
            .to_dm_view(
                &mut context.tdma,
                output.weight.view_mut().tile::<m![Head / 64], 1, m![
                    Head / 64 = 1 #{!} 2, Heads % (Heads / 8), Head % 64, Hidden % 64
                ]>(HALF),
            );
    }
}

/// The query projection written to DM: each half of its head dimension in turn, its heads'
/// squared scales copied beside it for each period to write into its output's slices.
impl<Hidden: AxisName, Heads: AxisName, Head: AxisName> Op<Begin<()>>
    for Affine<Weights<Hidden, Heads, Head, HeadSquares<Heads>>, ()>
{
    type Output = Affine<(Weight<Hidden, Heads, Head>, Squared<Heads>), ()>;

    fn forward(&self, device: &mut Device, _: Begin<()>) -> Self::Output {
        let mut output = Affine::new(DmTensor::new(), ());
        Half::<_, _, _, 0>(&self.weight.0).forward(&mut *device, &mut output);
        Half::<_, _, _, 1>(&self.weight.0).forward(&mut *device, &mut output);
        let squares = self.weight.1.0.to_hbm(&mut device.tdma);
        Affine::new((output.weight, squares), ())
    }
}

/// The key projection written to DM, its heads' squared scales copied beside it.
impl<Hidden: AxisName, Heads: AxisName, Head: AxisName> Op<Begin<()>>
    for Affine<(Weights<Hidden, Heads, Head, HeadSquares<Heads>>, ()), ()>
{
    type Output = Affine<((Weight<Hidden, Heads, Head>, Squared<Heads>), ()), ()>;

    fn forward(&self, device: &mut Device, _: Begin<()>) -> Self::Output {
        let mut weight = DmTensor::new();
        Panel(&self.weight.0.0).forward(&mut *device, &mut weight);
        let squares = self.weight.0.1.0.to_hbm(&mut device.tdma);
        Affine::new(((weight, squares), ()), ())
    }
}

/// The value projection written to DM with its head scales.
impl<Hidden: AxisName, Heads: AxisName, Head: AxisName> Op<Begin<()>>
    for Affine<Weights<Hidden, Heads, Head, HeadScales<Heads, Head>>, ()>
{
    type Output = Affine<(Weight<Hidden, Heads, Head>, ValueScale<Hidden, Heads, Head>), ()>;

    fn forward(&self, device: &mut Device, _: Begin<()>) -> Self::Output {
        let (stored, scales) = (&self.weight.0, &self.weight.1);
        let (mut weight, mut scale) = (DmTensor::new(), DmTensor::new());
        Panel(stored).forward(&mut *device, &mut weight);
        Scales {
            stored: scales,
            hidden: PhantomData,
        }
        .forward(device, &mut scale);
        Affine::new((weight, scale), ())
    }
}
