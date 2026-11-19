//! Checkpoint conversions and compositions over existing video-decoder numerical leaves.

use core::borrow::Borrow;
use core::marker::PhantomData;

use super::{Chips, EPSILON, HeadWidth, Hidden, Rotated};
use crate::load::{Checkpoint, Children, Context, Load, LoadError};
use crate::ops::attention::full;
use crate::ops::linear::{self, Unrounded};
use crate::ops::normalization::LayerNorm;
use crate::ops::pointwise::gated;
use crate::tensor::axis::Single;
use crate::tensor::memory::residency::Residency;
use crate::tensor::memory::{Hbm, Memory};
use crate::{Module, Op};
use furiosa_opt_std::prelude::*;

/// A source `[Out, In]` matrix and `[Out]` offset, rounded as the preserved Rust loader did.
#[derive(DeviceSend, Load, Residency)]
#[load(tree)]
#[residency(children(T, B))]
pub struct Dense<In: AxisName, Out: AxisName, C: M, T, B> {
    pub weight: T,
    pub bias: B,
    shape: PhantomData<fn() -> (In, Out, C)>,
}

impl<I, In: AxisName, Out: AxisName, C: M, T, B> !Op<I> for Dense<In, Out, C, T, B> {}

type View<'a, In, Out, C> =
    linear::Affine<&'a HbmTensor<bf16, C, m![Out, In]>, &'a HbmTensor<f32, C, m![Out]>>;

impl<In: AxisName, Out: AxisName, C: M, T, B> Dense<In, Out, C, T, B>
where
    T: Borrow<HbmTensor<bf16, C, m![Out, In]>>,
    B: Borrow<HbmTensor<f32, C, m![Out]>>,
{
    /// Lend the exact ordinary affine form used by both row-major and head-major projections.
    pub fn view(&self) -> View<'_, In, Out, C> {
        linear::Affine::new(self.weight.borrow(), self.bias.borrow())
    }
}

impl<I, O, E, In: AxisName, Out: AxisName, C: M, T, B> Module<I, E, &mut ()>
    for Dense<In, Out, C, T, B>
where
    T: Borrow<HbmTensor<bf16, C, m![Out, In]>>,
    B: Borrow<HbmTensor<f32, C, m![Out]>>,
    for<'a> View<'a, In, Out, C>: Op<I, Output = O>,
{
    type Output = O;

    fn forward(&self, device: &mut Device, _: E, _: &mut (), input: I) -> O {
        Op::forward(&self.view(), device, input)
    }
}

impl<'a, In: AxisName, Out: AxisName, C: M + 'static, T, B> Load<&'a Checkpoint>
    for Dense<In, Out, C, T, B>
where
    T: Load<HostTensor<bf16, m![{ C }, { m![Out, In] }]>>,
    B: Load<HostTensor<f32, m![{ C }, { m![Out] }]>>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let (weight, bias) = Self::images(from, cx)?;
            let weight = cx.field("weight", "weight", |cx| T::load(cx, weight))?;
            let bias = cx.field("bias", "bias", |cx| B::load(cx, bias))?;
            Ok(Self {
                weight,
                bias,
                shape: PhantomData,
            })
        })
    }
}

impl<'t, 's, In: AxisName, Out: AxisName, C: M + 'static, T, B>
    Load<(&'t mut Dense<In, Out, C, T, B>, &'s Checkpoint)> for &'t mut Dense<In, Out, C, T, B>
where
    for<'u> &'u mut T: Load<(&'u mut T, HostTensor<bf16, m![{ C }, { m![Out, In] }]>)>,
    for<'u> &'u mut B: Load<(&'u mut B, HostTensor<f32, m![{ C }, { m![Out] }]>)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut Dense<In, Out, C, T, B>, &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.field("weight", "weight", |cx| {
                cx.rule(
                    |cx| Dense::<In, Out, C, T, B>::weight_image(from, cx),
                    |cx, host| <&mut T as Load<_>>::load(cx, (&mut into.weight, host)).map(drop),
                )
            })?;
            cx.field("bias", "bias", |cx| {
                cx.rule(
                    |cx| Dense::<In, Out, C, T, B>::bias_image(from, cx),
                    |cx, host| <&mut B as Load<_>>::load(cx, (&mut into.bias, host)).map(drop),
                )
            })
            .map(drop)
        })?;
        Ok(into)
    }
}

impl<In: AxisName, Out: AxisName, C: M + 'static, T, B> Dense<In, Out, C, T, B> {
    fn images(
        source: &Checkpoint,
        context: &mut Context<'_>,
    ) -> Result<
        (
            HostTensor<bf16, m![{ C }, { m![Out, In] }]>,
            HostTensor<f32, m![{ C }, { m![Out] }]>,
        ),
        LoadError,
    > {
        let mut children = Children::default();
        let weight =
            children.ask(context.field("weight", "weight", |cx| Self::weight_image(source, cx)))?;
        let bias =
            children.ask(context.field("bias", "bias", |cx| Self::bias_image(source, cx)))?;
        children.node(weight.zip(bias))
    }

    fn weight_image(
        source: &Checkpoint,
        context: &mut Context<'_>,
    ) -> Result<HostTensor<bf16, m![{ C }, { m![Out, In] }]>, LoadError> {
        let weight = rounded(source, context, &[Out::SIZE, In::SIZE])?;
        let weight: Vec<bf16> = weight.into_iter().map(bf16::from_f32).collect();
        Ok(HostTensor::from_vec(weight.repeat(C::SIZE)))
    }

    fn bias_image(
        source: &Checkpoint,
        context: &mut Context<'_>,
    ) -> Result<HostTensor<f32, m![{ C }, { m![Out] }]>, LoadError> {
        rounded(source, context, &[Out::SIZE])
            .map(|bias| HostTensor::from_vec(bias.repeat(C::SIZE)))
    }
}

/// The same converted affine parameters, evaluated by public `Unrounded` into FP32 rows.
#[derive(DeviceSend, Load, Residency)]
#[residency(children(T, B))]
pub struct Readout<In: AxisName, Out: AxisName, C: M, T, B>(pub Dense<In, Out, C, T, B>);

impl<I, In: AxisName, Out: AxisName, C: M, T, B> !Op<I> for Readout<In, Out, C, T, B> {}

impl<In: AxisName, Out: AxisName, R: AxisName, C: M, T, B, E>
    Module<&HbmTensor<bf16, C, m![R, In]>, E, &mut ()> for Readout<In, Out, C, T, B>
where
    T: Borrow<HbmTensor<bf16, C, m![Out, In]>>,
    B: Borrow<HbmTensor<f32, C, m![Out]>>,
{
    type Output = HbmTensor<f32, C, m![R, Out]>;

    fn forward(
        &self,
        device: &mut Device,
        _: E,
        _: &mut (),
        input: &HbmTensor<bf16, C, m![R, In]>,
    ) -> Self::Output {
        let readout: Unrounded<In, Out, C, _, _> = Unrounded(self.0.view(), PhantomData);
        Op::forward(&readout, device, input)
    }
}

/// Learned per-feature scale, stored as the shared FP32 residual add's single-row gate.
#[derive(DeviceSend, Load, Residency)]
#[load(tree)]
#[residency(children(T))]
pub struct Scale<C: M, T> {
    pub weight: T,
    shape: PhantomData<fn() -> C>,
}

impl<I, C: M, T> !Op<I> for Scale<C, T> {}

impl<C: M, T, E> Module<(&super::Residual<C>, &super::Normalized<C>), E, &mut ()> for Scale<C, T>
where
    T: Borrow<HbmTensor<f32, C, Gate>>,
{
    type Output = super::Residual<C>;

    fn forward(
        &self,
        device: &mut Device,
        _: E,
        _: &mut (),
        (input, branch): (&super::Residual<C>, &super::Normalized<C>),
    ) -> Self::Output {
        Op::forward(&gated::Add, device, (input, branch, self.weight.borrow()))
    }
}

impl<'a, C: M + 'static, T> Load<&'a Checkpoint> for Scale<C, T>
where
    T: Load<HostTensor<f32, m![{ C }, { Gate }]>>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let host = gate::<C>(from, cx)?;
            cx.field("weight", "", |cx| T::load(cx, host))
                .map(|weight| Self {
                    weight,
                    shape: PhantomData,
                })
        })
    }
}

impl<'t, 's, C: M + 'static, T> Load<(&'t mut Scale<C, T>, &'s Checkpoint)> for &'t mut Scale<C, T>
where
    for<'u> &'u mut T: Load<(&'u mut T, HostTensor<f32, m![{ C }, { Gate }]>)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut Scale<C, T>, &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.rule(
                |cx| gate::<C>(from, cx),
                |cx, host| {
                    cx.field("weight", "", |cx| {
                        <&mut T as Load<_>>::load(cx, (&mut into.weight, host)).map(drop)
                    })
                },
            )
            .map(drop)
        })?;
        Ok(into)
    }
}

/// Generated unit head gain and `1e-5` epsilon, kept in the same storage as learned parameters.
pub type HeadNorm<C = Chips, X = Hbm> = UnitHeadNorm<
    C,
    <X as Memory>::Tensor<f32, C, m![HeadWidth]>,
    <X as Memory>::Tensor<f32, C, m![1]>,
>;

#[derive(DeviceSend, Load, Residency)]
#[load(tree)]
#[residency(children(T, E))]
pub struct UnitHeadNorm<C: M, T, E> {
    pub norm: full::HeadNorm<HeadWidth, C, T, E, EPSILON, Rotated>,
}

impl<I, C: M, T, E> !Op<I> for UnitHeadNorm<C, T, E> {}

impl<I, O, C: M, T, E, Env> Module<I, Env, &mut ()> for UnitHeadNorm<C, T, E>
where
    for<'n> full::Division<&'n full::HeadNorm<HeadWidth, C, T, E, EPSILON, Rotated>>:
        Op<I, Output = O>,
{
    type Output = O;

    fn forward(&self, device: &mut Device, _: Env, _: &mut (), input: I) -> O {
        Op::forward(&full::Division(&self.norm), device, input)
    }
}

impl<'a, C: M + 'static, T, E> Load<&'a Checkpoint> for UnitHeadNorm<C, T, E>
where
    T: Load<HostTensor<f32, m![{ C }, { m![HeadWidth] }]>>,
    E: Load<HostTensor<f32, m![{ C }, { m![1] }]>>,
{
    fn load(context: &mut Context<'_>, _: &'a Checkpoint) -> Result<Self, LoadError> {
        let weight = context.field("norm.weight", "", |cx| {
            T::load(
                cx,
                HostTensor::from_vec(vec![1.0; C::SIZE * HeadWidth::SIZE]),
            )
        })?;
        let epsilon = context.field("norm.epsilon", "", |cx| {
            E::load(
                cx,
                HostTensor::from_vec(vec![f32::from_bits(EPSILON as u32); C::SIZE]),
            )
        })?;
        Ok(Self {
            norm: full::HeadNorm {
                weight,
                epsilon,
                head: PhantomData,
            },
        })
    }
}

/// Neither the unit gain nor epsilon has a checkpoint tensor or an update target.
impl<'t, 's, C: M, T, E> Load<(&'t mut UnitHeadNorm<C, T, E>, &'s Checkpoint)>
    for &'t mut UnitHeadNorm<C, T, E>
{
    fn load(
        _: &mut Context<'_>,
        (into, _): (&'t mut UnitHeadNorm<C, T, E>, &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        Ok(into)
    }
}

/// The final LayerNorm, with source BF16 rounding and a replicated generated epsilon.
#[derive(DeviceSend, Load, Residency)]
#[load(tree)]
#[residency(children(T, E))]
pub struct FinalNorm<C: M, T, E> {
    pub norm: LayerNorm<T, E, EPSILON>,
    shape: PhantomData<fn() -> C>,
}

impl<I, C: M, T, E> !Op<I> for FinalNorm<C, T, E> {}

impl<C: M, T, E, Env> Module<&super::Residual<C>, Env, &mut ()> for FinalNorm<C, T, E>
where
    T: Borrow<HbmTensor<bf16, C, m![Hidden]>>,
    E: Borrow<HbmTensor<f32, C, m![1]>>,
{
    type Output = super::Normalized<C>;

    fn forward(
        &self,
        device: &mut Device,
        _: Env,
        _: &mut (),
        input: &super::Residual<C>,
    ) -> Self::Output {
        let weight: &HbmTensor<bf16, C, m![Hidden]> = self.norm.weight.borrow();
        let bias: &HbmTensor<bf16, C, m![Hidden]> = self.norm.bias.borrow();
        let epsilon: &HbmTensor<f32, C, m![1]> = self.norm.epsilon.borrow();
        let norm: LayerNorm<_, _, EPSILON> = LayerNorm::new(weight, bias, epsilon);
        Op::forward(&norm, device, input)
    }
}

impl<'a, C: M + 'static, T, E> Load<&'a Checkpoint> for FinalNorm<C, T, E>
where
    T: Load<HostTensor<bf16, m![{ C }, { m![Hidden] }]>>,
    E: Load<HostTensor<f32, m![{ C }, { m![1] }]>>,
{
    fn load(context: &mut Context<'_>, source: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(source, |cx| {
            let (weight, bias) = norm_images::<C>(source, cx)?;
            let weight = cx.field("norm.weight", "weight", |cx| T::load(cx, weight))?;
            let bias = cx.field("norm.bias", "bias", |cx| T::load(cx, bias))?;
            let epsilon = cx.field("norm.epsilon", "", |cx| {
                E::load(
                    cx,
                    HostTensor::from_vec(vec![f32::from_bits(EPSILON as u32); C::SIZE]),
                )
            })?;
            Ok(Self {
                norm: LayerNorm::new(weight, bias, epsilon),
                shape: PhantomData,
            })
        })
    }
}

impl<'t, 's, C: M + 'static, T, E> Load<(&'t mut FinalNorm<C, T, E>, &'s Checkpoint)>
    for &'t mut FinalNorm<C, T, E>
where
    for<'u> &'u mut T: Load<(&'u mut T, HostTensor<bf16, m![{ C }, { m![Hidden] }]>)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut FinalNorm<C, T, E>, &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.field("norm.weight", "weight", |cx| {
                cx.rule(
                    |cx| norm_parameter::<C>(from, cx),
                    |cx, host| {
                        <&mut T as Load<_>>::load(cx, (&mut into.norm.weight, host)).map(drop)
                    },
                )
            })?;
            cx.field("norm.bias", "bias", |cx| {
                cx.rule(
                    |cx| norm_parameter::<C>(from, cx),
                    |cx, host| <&mut T as Load<_>>::load(cx, (&mut into.norm.bias, host)).map(drop),
                )
            })
            .map(drop)
        })?;
        Ok(into)
    }
}

/// A feature scale represented as one row for the public gated residual add.
pub type Gate = m![Single, Hidden];

fn gate<C: M>(
    source: &Checkpoint,
    context: &mut Context<'_>,
) -> Result<HostTensor<f32, m![{ C }, { Gate }]>, LoadError> {
    rounded(source, context, &[Hidden::SIZE])
        .map(|values| HostTensor::from_vec(values.repeat(C::SIZE)))
}

fn norm_images<C: M>(
    source: &Checkpoint,
    context: &mut Context<'_>,
) -> Result<
    (
        HostTensor<bf16, m![{ C }, { m![Hidden] }]>,
        HostTensor<bf16, m![{ C }, { m![Hidden] }]>,
    ),
    LoadError,
> {
    let mut children = Children::default();
    let weight =
        children.ask(context.field("weight", "weight", |cx| norm_parameter::<C>(source, cx)))?;
    let bias = children.ask(context.field("bias", "bias", |cx| norm_parameter::<C>(source, cx)))?;
    children.node(weight.zip(bias))
}

fn norm_parameter<C: M>(
    source: &Checkpoint,
    context: &mut Context<'_>,
) -> Result<HostTensor<bf16, m![{ C }, { m![Hidden] }]>, LoadError> {
    let values = rounded(source, context, &[Hidden::SIZE])?;
    let values: Vec<bf16> = values.into_iter().map(bf16::from_f32).collect();
    Ok(HostTensor::from_vec(values.repeat(C::SIZE)))
}

/// Exact shapes and the preserved `Entry::rounded`: source FP32 → BF16 → FP32.
pub(super) fn rounded(
    source: &Checkpoint,
    context: &mut Context<'_>,
    shape: &[usize],
) -> Result<Vec<f32>, LoadError> {
    context
        .floats(source, shape)
        .map(|values| values.into_iter().map(round).collect())
}

pub(super) fn round(value: f32) -> f32 {
    bf16::from_f32(value).to_f32()
}
