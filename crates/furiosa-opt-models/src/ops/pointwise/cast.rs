//! Convert tensor elements while preserving their logical row and feature axes.

use crate::Op;
use crate::load::Load;
use crate::tensor::mapping;
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

/// A narrow float format, by the largest finite magnitude its quantizers scale a group to.
pub(crate) trait Finite {
    const LARGEST: f32;
}

impl Finite for f8e4m3 {
    const LARGEST: f32 = 448.0;
}

impl Finite for f4e2m1 {
    const LARGEST: f32 = 6.0;
}

/// Convert elements to `D` while retaining every logical and physical axis.
#[derive(DeviceSend, Load)]
pub struct Convert<D>(PhantomData<D>);

impl<D> Default for Convert<D> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<In: Scalar, Out: Scalar, Width: AxisName, Rows: AxisName>
    Op<&HbmTensor<In, m![1], m![Rows, Width]>> for Convert<Out>
{
    type Output = HbmTensor<Out, m![1], m![Rows, Width]>;

    fn forward(&self, _: &mut Device, _: &HbmTensor<In, m![1], m![Rows, Width]>) -> Self::Output {
        todo!("NPU element conversion, with identity conversion elided")
    }
}

// An owned identity cast forwards the allocation, without a device instruction or copy.
impl<D: Scalar, Chips: M, K: M, S: M, E: M> Op<DmTensor<D, Chips, K, S, E>> for Convert<D> {
    type Output = DmTensor<D, Chips, K, S, E>;

    fn forward(&self, _: &mut Device, input: Self::Output) -> Self::Output {
        input
    }
}

impl<Chips: M, K: M, S: M, E: M + 'static> Op<DmTensor<f8e4m3, Chips, K, S, E>> for Convert<bf16> {
    type Output = DmTensor<bf16, Chips, K, S, E>;

    fn forward(
        &self,
        device: &mut Device,
        input: DmTensor<f8e4m3, Chips, K, S, E>,
    ) -> Self::Output {
        convert(device, input)
    }
}

impl<Chips: M, K: M, S: M, E: M + 'static> Op<DmTensor<bf16, Chips, K, S, E>> for Convert<f8e4m3> {
    type Output = DmTensor<f8e4m3, Chips, K, S, E>;

    fn forward(&self, device: &mut Device, input: DmTensor<bf16, Chips, K, S, E>) -> Self::Output {
        convert(device, input)
    }
}

pub(crate) fn convert<In, Out, Chips: M, K: M, S: M, E: M + 'static>(
    device: &mut Device,
    input: DmTensor<In, Chips, K, S, E>,
) -> DmTensor<Out, Chips, K, S, E>
where
    In: MaterializableScalar + FetchCast<f32>,
    Out: MaterializableScalar,
    f32: CastEngineCast<Out>,
{
    type Width<E> = Symbol<super::Flat<E>>;
    type Packet<D> = Symbol<mapping::Packet<D>>;
    // SAFETY: flatten the physical element order within each unchanged DM slice.
    let input = unsafe { input.reshape::<Chips, K, S, m![{ Width<E> }]>() };
    let output: DmTensor<Out, Chips, K, S, m![{ Width<E> }]> = device
        .main
        .begin(input.view())
        .fetch::<m![{ Width<E> } / ({ Packet<In> })], m![{ Width<E> } % ({ Packet<In> })]>()
        .fetch_cast::<f32>()
        .collect::<m![{ Width<E> } / 8], m![{ Width<E> } % 8]>()
        .cast::<Out, m![{ Width<E> } % 8 # ({ Packet<Out> })]>()
        .commit_trim::<m![{ Width<E> } % 8]>()
        .commit();
    // SAFETY: conversion preserves each element's physical position; restore its labels.
    unsafe { output.reshape() }
}
