//! Row grouping and explicit copies between device tensor mappings.

use crate::load::Load;
use core::marker::PhantomData;

use furiosa_opt_std::prelude::*;

use super::axis::Lanes;
use crate::Module;

pub(crate) type BroadcastClusters = m![1 # 2];
pub(crate) type BroadcastSlices = m![1 # 256];

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

impl<
    D: Scalar,
    Chip: M,
    InputCluster: M,
    InputSlices: M,
    InputElements: M,
    Chips: M,
    S: M,
    E: M,
    Env,
> Module<DmTensor<D, Chip, InputCluster, InputSlices, InputElements>, Env, &mut ()>
    for Reorder<Chips, S, E>
{
    type Output = DmTensor<D, Chip, Chips, S, E>;

    fn forward(
        &self,
        device: &mut Device,
        _: Env,
        _: &mut (),
        input: DmTensor<D, Chip, InputCluster, InputSlices, InputElements>,
    ) -> Self::Output {
        let mut output = DmTensor::new();
        input.view().to_dm_view(&mut device.tdma, output.view_mut());
        output
    }
}

/// Reinterpret contiguous axes as Output: by value shares storage, by reference copies HBM.
#[derive(DeviceSend, Load)]
pub struct Reshape<Output>(PhantomData<Output>);

impl<Output> Default for Reshape<Output> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<D: Scalar, Chips: M, Input: M, Output: M, Env> Module<HbmTensor<D, Chips, Input>, Env, &mut ()>
    for Reshape<Output>
{
    type Output = HbmTensor<D, Chips, Output>;

    fn forward(
        &self,
        _: &mut Device,
        _: Env,
        _: &mut (),
        input: HbmTensor<D, Chips, Input>,
    ) -> Self::Output {
        // SAFETY: only contiguous element coordinates change; reshape checks the size.
        unsafe { input.reshape() }
    }
}

impl<D: MaterializableScalar, Chip: M, Input: M, Output: M, Env>
    Module<&HbmTensor<D, Chip, Input>, Env, &mut ()> for Reshape<Output>
{
    type Output = HbmTensor<D, Chip, Output>;

    fn forward(
        &self,
        device: &mut Device,
        _: Env,
        _: &mut (),
        input: &HbmTensor<D, Chip, Input>,
    ) -> Self::Output {
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

impl<D: MaterializableScalar, Chip: M, Input: M, Output: M, Env>
    Module<&HbmTensor<D, Chip, Input>, Env, &mut ()> for Relayout<Output>
{
    type Output = HbmTensor<D, Chip, Output>;

    fn forward(
        &self,
        device: &mut Device,
        _: Env,
        _: &mut (),
        input: &HbmTensor<D, Chip, Input>,
    ) -> Self::Output {
        input.to_hbm(&mut device.tdma)
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

impl<
    D: Scalar,
    Rows: AxisName,
    Width: AxisName,
    Left: AxisName,
    Right: AxisName,
    Env,
    Chips: M + 'static,
> Module<&HbmTensor<D, Chips, m![Rows, Width]>, Env, &mut ()> for Split<Left, Right>
{
    type Output = (
        HbmTensor<D, Chips, m![Rows, Left]>,
        HbmTensor<D, Chips, m![Rows, Right]>,
    );

    fn forward(
        &self,
        _: &mut Device,
        _: Env,
        _: &mut (),
        _: &HbmTensor<D, Chips, m![Rows, Width]>,
    ) -> Self::Output {
        const { assert!(Width::SIZE == Left::SIZE + Right::SIZE) };
        todo!("NPU split of each row into its first L and remaining R feature columns")
    }
}
