//! Row grouping and explicit copies between device tensor mappings.

pub mod axes;

use core::marker::PhantomData;

use furiosa_opt_std::prelude::*;

use crate::Module;
use axes::Lanes;

pub(crate) type BroadcastClusters = m![1 # 2];
pub(crate) type BroadcastSlices = m![1 # 256];

/// The live rows of a launch over an `R`-row envelope, in packets of lanes.
pub type Live<R> = crate::compose::Tiles<Lanes<R>>;

/// A launch's rows on chip and the live ones among them: a leaf computes the live rows of `value`
/// and hands the same `tiles` on with its result; a row past them is padding nothing reads.
pub struct Dm<S, X> {
    pub value: X,
    pub tiles: crate::compose::Tiles<S>,
}

/// A launch's rows in HBM and the live domain its consumer computes.
pub struct Hbm<S, X> {
    pub value: X,
    pub tiles: crate::compose::Tiles<S>,
}

/// Packet-padded rows retain the logical lane count for short batches.
pub type Packets<R> = m![R / ({ Symbol<Lanes<R>> }), R % ({ Symbol<Lanes<R>> }) # 8];

/// Row groups and lane coordinates for projections sharing eight-lane vector packets.
pub struct Rows<R>(PhantomData<R>);

/// One cluster for one lane group, both clusters when the rows span several groups.
#[derive(Clone, Debug)]
pub struct Clusters<R>(PhantomData<R>);

impl<R: AxisName> AxisName for Clusters<R> {
    const NAME: Ident = Ident::new("Clusters");
    const SIZE: usize = if Lanes::<R>::SIZE < R::SIZE { 2 } else { 1 };
}

impl<R: AxisName> Rows<R> {
    pub type Groups = m![R / ({ Symbol<Lanes<R>> })];
    pub type Lanes = m![R % ({ Symbol<Lanes<R>> })];
    pub type Spatial = m![R / ({ Symbol<Lanes<R>> }) % ({ Symbol<Clusters<R>> })];
    pub type Serial = m![R / ({ Symbol<Lanes<R>> }) / ({ Symbol<Clusters<R>> })];
    pub type Selected = m![{ Self::Groups } = 1 # ({ Self::Groups })];
    pub type SelectedMut = m![{ Self::Groups } = 1 #{!} ({ Self::Groups })];
    pub type Elements<Inner: M> = m![{ Self::Selected }, { Self::Lanes }, { Inner }];
    pub type GroupMut<Inner: M> = m![{ Self::SelectedMut }, { Self::Lanes }, { Inner }];

    pub fn group<'a, D: Scalar, Chips: M, S: M, I: M>(
        rows: DmTensorView<'a, D, m![1], Chips, S, m![R, { I }]>,
        at: usize,
    ) -> DmTensorView<'a, D, m![1], Chips, S, Self::Elements<I>> {
        rows.tile::<Self::Groups, 1, Self::Elements<I>>(at)
    }

    pub fn selected_mut<'a, D: Scalar, Chips: M, S: M, I: M>(
        rows: DmTensorViewMut<'a, D, m![1], Chips, S, m![{ Self::Groups }, { I }]>,
        at: usize,
    ) -> DmTensorViewMut<'a, D, m![1], Chips, S, m![{ Self::SelectedMut }, { I }]> {
        rows.tile::<Self::Groups, 1, _>(at)
    }
}

/// Generic DM copy into the mapping selected by the consumer; unlike embedding Reorder,
/// the destination cluster, slice and element mappings are all caller-selected.
pub struct Reorder<C, S, E>(PhantomData<(C, S, E)>);

impl<C, S, E> Default for Reorder<C, S, E> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<'d, D: Scalar, Chip: M, IC: M, IS: M, IE: M, Chips: M, S: M, E: M>
    Module<(&'d mut Device, DmTensor<D, Chip, IC, IS, IE>)> for Reorder<Chips, S, E>
{
    type Output = DmTensor<D, Chip, Chips, S, E>;

    fn forward(
        &self,
        (device, input): (&'d mut Device, DmTensor<D, Chip, IC, IS, IE>),
    ) -> Self::Output {
        let mut output = DmTensor::new();
        input.view().to_dm_view(&mut device.tdma, output.view_mut());
        output
    }
}

/// Reinterpret contiguous axes as Output: by value shares storage, by reference copies HBM.
#[derive(DeviceSend)]
pub struct Reshape<Output>(PhantomData<Output>);

impl<Output> Default for Reshape<Output> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<D: Scalar, Chips: M, Input: M, Output: M> Module<(&mut Device, HbmTensor<D, Chips, Input>)>
    for Reshape<Output>
{
    type Output = HbmTensor<D, Chips, Output>;

    fn forward(&self, (_, input): (&mut Device, HbmTensor<D, Chips, Input>)) -> Self::Output {
        // SAFETY: only contiguous element coordinates change; reshape checks the size.
        unsafe { input.reshape() }
    }
}

impl<D: MaterializableScalar, Chip: M, Input: M, Output: M>
    Module<(&mut Device, &HbmTensor<D, Chip, Input>)> for Reshape<Output>
{
    type Output = HbmTensor<D, Chip, Output>;

    fn forward(&self, (device, input): (&mut Device, &HbmTensor<D, Chip, Input>)) -> Self::Output {
        let mut output = HbmTensor::new();
        // SAFETY: contiguous coordinates change without changing size or chip placement.
        unsafe { input.view().reshape::<Chip, Output>() }
            .to_hbm_view(&mut device.tdma, output.view_mut());
        output
    }
}

/// Copy the same elements into `Output`'s storage, which pads them differently.
#[derive(DeviceSend)]
pub struct Relayout<Output>(PhantomData<Output>);

impl<Output> Default for Relayout<Output> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<D: MaterializableScalar, Chip: M, Input: M, Output: M>
    Module<(&mut Device, &HbmTensor<D, Chip, Input>)> for Relayout<Output>
{
    type Output = HbmTensor<D, Chip, Output>;

    fn forward(&self, (device, input): (&mut Device, &HbmTensor<D, Chip, Input>)) -> Self::Output {
        input.to_hbm(&mut device.tdma)
    }
}

/// Overwrite a caller's tensor with a tensor of the same mapping, in one copy.
#[derive(Default, DeviceSend)]
pub struct Overwrite;

impl<'o, D: MaterializableScalar, Chip: M, E: M>
    Module<(
        &mut Device,
        (HbmTensor<D, Chip, E>, &'o mut HbmTensor<D, Chip, E>),
    )> for Overwrite
{
    type Output = ();

    fn forward(
        &self,
        (device, (source, destination)): (
            &mut Device,
            (HbmTensor<D, Chip, E>, &'o mut HbmTensor<D, Chip, E>),
        ),
    ) {
        source
            .view()
            .to_hbm_view(&mut device.tdma, destination.view_mut());
    }
}

/// Elements in a 256-bit DMA packet.
#[derive(Clone, Debug)]
pub(crate) struct Packet<D>(PhantomData<D>);

impl<D: Scalar> AxisName for Packet<D> {
    const NAME: Ident = Ident::new("Packet");
    const SIZE: usize = 256 / D::BITS;
}

/// Split each row's consecutive feature columns into two owned tensors.
#[derive(DeviceSend)]
pub struct Split<Left, Right>(pub PhantomData<(Left, Right)>);

impl<D: Scalar, Rows: AxisName, Width: AxisName, L: AxisName, R: AxisName>
    Module<(&mut Device, &HbmTensor<D, m![1], m![Rows, Width]>)> for Split<L, R>
{
    type Output = (
        HbmTensor<D, m![1], m![Rows, L]>,
        HbmTensor<D, m![1], m![Rows, R]>,
    );

    fn forward(&self, _: (&mut Device, &HbmTensor<D, m![1], m![Rows, Width]>)) -> Self::Output {
        const { assert!(Width::SIZE == L::SIZE + R::SIZE) };
        todo!("NPU split of each row into its first L and remaining R feature columns")
    }
}
