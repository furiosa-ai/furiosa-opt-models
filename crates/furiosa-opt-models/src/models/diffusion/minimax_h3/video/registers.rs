//! Strict four-register checkpoint conversion around the public register-scatter leaf.

use core::borrow::Borrow;

use super::{Hidden, TOKENS};
use crate::load::{Checkpoint, Context, Load, LoadError};
use crate::ops::index::scatter::{self, Run};
use crate::tensor::memory::residency::Residency;
use crate::{Module, Op};
use furiosa_opt_std::prelude::*;

/// Four BF16-rounded learned rows and one zero class row, written after the latent tokens.
/// The shared scatter leaf owns execution; this adapter narrows its permissive source contract.
#[derive(DeviceSend, Load, Residency)]
#[load(tree)]
#[residency(children(T))]
pub struct Registers<C: M, T>(pub scatter::Registers<Hidden, TOKENS, 5, C, Rounded<T>>);

impl<I, C: M, T> !Op<I> for Registers<C, T> {}

impl<I, O, C: M, T, E> Module<I, E, &mut ()> for Registers<C, T>
where
    scatter::Registers<Hidden, TOKENS, 5, C, Rounded<T>>: Op<I, Output = O>,
{
    type Output = O;

    fn forward(&self, device: &mut Device, _: E, _: &mut (), input: I) -> O {
        Op::forward(&self.0, device, input)
    }
}

impl<'a, C: M + 'static, T> Load<&'a Checkpoint> for Registers<C, T>
where
    T: Load<HostTensor<f32, m![{ C }, { Layout }]>>,
{
    fn load(context: &mut Context<'_>, source: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(source, |cx| {
            shape(source, cx)?;
            scatter::Registers::load(cx, source).map(Self)
        })
    }
}

impl<'t, 's, C: M + 'static, T> Load<(&'t mut Registers<C, T>, &'s Checkpoint)>
    for &'t mut Registers<C, T>
where
    for<'u> &'u mut T: Load<(&'u mut T, HostTensor<f32, m![{ C }, { Layout }]>)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, source): (&'t mut Registers<C, T>, &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        context.checkpoint(source, |cx| {
            // Let the shared update rule record an absent source and defer its write.
            match shape(source, cx) {
                Ok(()) | Err(LoadError::Missing(_)) => {}
                Err(error) => return Err(error),
            }
            <&mut scatter::Registers<Hidden, TOKENS, 5, C, Rounded<T>> as Load<_>>::load(
                cx,
                (&mut into.0, source),
            )
            .map(drop)
        })?;
        Ok(into)
    }
}

/// A storage adapter that applies `Entry::rounded` to the shared loader's prepared FP32 rows.
/// It retains no second copy and borrows the underlying HBM tensor for the scatter operation.
#[derive(DeviceSend, Load, Residency)]
#[load(tree)]
#[residency(children(T))]
pub struct Rounded<T>(pub T);

impl<C: M, S: M, T> Borrow<HbmTensor<f32, C, S>> for Rounded<T>
where
    T: Borrow<HbmTensor<f32, C, S>>,
{
    fn borrow(&self) -> &HbmTensor<f32, C, S> {
        self.0.borrow()
    }
}

impl<E: M, T> Load<HostTensor<f32, E>> for Rounded<T>
where
    T: Load<HostTensor<f32, E>>,
{
    fn load(context: &mut Context<'_>, host: HostTensor<f32, E>) -> Result<Self, LoadError> {
        T::load(context, round(host)).map(Self)
    }
}

impl<'t, E: M, T> Load<(&'t mut Rounded<T>, HostTensor<f32, E>)> for &'t mut Rounded<T>
where
    for<'u> &'u mut T: Load<(&'u mut T, HostTensor<f32, E>)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, host): (&'t mut Rounded<T>, HostTensor<f32, E>),
    ) -> Result<Self, LoadError> {
        <&mut T as Load<_>>::load(context, (&mut into.0, round(host)))?;
        Ok(into)
    }
}

/// Four learned register rows followed by a zero class row, in hidden-feature order.
pub type Layout = m![{ Symbol<Run<5>> }, Hidden];

fn shape(source: &Checkpoint, context: &mut Context<'_>) -> Result<(), LoadError> {
    let (name, (_, stored)) = context.header(source)?;
    let expected = [1, 4, Hidden::SIZE];
    if stored != expected.as_slice() {
        return Err(LoadError::InvalidData {
            name,
            reason: format!("shape {stored:?}, video registers hold {expected:?}"),
        });
    }
    Ok(())
}

fn round<E: M>(host: HostTensor<f32, E>) -> HostTensor<f32, E> {
    HostTensor::from_vec(host.into_vec().into_iter().map(super::parameters::round))
}
