//! Convert tensor elements while preserving their logical row and feature axes.

use crate::Module;
use furiosa_opt_std::prelude::*;

/// Convert elements to `D` while retaining every logical and physical axis.
#[derive(DeviceSend)]
pub struct Convert<D>(std::marker::PhantomData<D>);

impl<D> Default for Convert<D> {
    fn default() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<In: Scalar, Out: Scalar, Width: AxisName, Rows: AxisName>
    Module<(&mut Device, &HbmTensor<In, m![1], m![Rows, Width]>)> for Convert<Out>
{
    type Output = HbmTensor<Out, m![1], m![Rows, Width]>;

    fn forward(&self, _: (&mut Device, &HbmTensor<In, m![1], m![Rows, Width]>)) -> Self::Output {
        todo!("NPU element conversion, with identity conversion elided")
    }
}

// An owned identity cast forwards the allocation, without a device instruction or copy.
impl<D: Scalar, Chips: M, K: M, S: M, E: M> Module<(&mut Device, DmTensor<D, Chips, K, S, E>)>
    for Convert<D>
{
    type Output = DmTensor<D, Chips, K, S, E>;

    fn forward(&self, (_, input): (&mut Device, Self::Output)) -> Self::Output {
        input
    }
}

impl<Chips: M, K: M, S: M, E: M + 'static> Module<(&mut Device, DmTensor<f8e4m3, Chips, K, S, E>)>
    for Convert<bf16>
{
    type Output = DmTensor<bf16, Chips, K, S, E>;

    fn forward(
        &self,
        (device, input): (&mut Device, DmTensor<f8e4m3, Chips, K, S, E>),
    ) -> Self::Output {
        convert(device, input)
    }
}

impl<Chips: M, K: M, S: M, E: M + 'static> Module<(&mut Device, DmTensor<bf16, Chips, K, S, E>)>
    for Convert<f8e4m3>
{
    type Output = DmTensor<f8e4m3, Chips, K, S, E>;

    fn forward(
        &self,
        (device, input): (&mut Device, DmTensor<bf16, Chips, K, S, E>),
    ) -> Self::Output {
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
    type Width<E> = Symbol<crate::ops::pointwise::Flat<E>>;
    type Packet<D> = Symbol<crate::ops::layout::Packet<D>>;
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
