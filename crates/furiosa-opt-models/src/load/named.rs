//! A generic child named in both paths.

use super::{Checkpoint, Context, Footprint, Load, LoadError};
use crate::Module;
use furiosa_opt_std::prelude::{Device, DeviceSend};

/// `M` under the checkpoint and tree segment `NAME`; it computes and launches as `M`.
#[derive(Clone, DeviceSend)]
pub struct Named<const NAME: &'static str, M>(pub M);

/// A composition, not an op.
impl<I, const NAME: &'static str, M> !crate::Op<I> for Named<NAME, M> {}

impl<I, E, S, const NAME: &'static str, M: Module<I, E, S>> Module<I, E, S> for Named<NAME, M> {
    type Output = M::Output;

    fn forward(&self, device: &mut Device, env: E, state: S, input: I) -> M::Output {
        self.0.forward(device, env, state, input)
    }
}

impl<const NAME: &'static str, M: Footprint> Footprint for Named<NAME, M> {
    const SIZE: usize = M::SIZE;
}

impl<'a, const NAME: &'static str, T: Load<&'a Checkpoint>> Load<&'a Checkpoint>
    for Named<NAME, T>
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.field(NAME, NAME, |cx| T::load(cx, from)).map(Named)
        })
    }
}

impl<'t, 's, const NAME: &'static str, T> Load<(&'t mut Named<NAME, T>, &'s Checkpoint)>
    for &'t mut Named<NAME, T>
where
    for<'u> &'u mut T: Load<(&'u mut T, &'s Checkpoint)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut Named<NAME, T>, &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.field(NAME, NAME, |cx| {
                <&mut T as Load<_>>::load(cx, (&mut into.0, from))
            })
        })?;
        Ok(into)
    }
}

impl<'a, const NAME: &'static str, T: Load<&'a U>, U> Load<&'a Named<NAME, U>> for Named<NAME, T> {
    fn load(context: &mut Context<'_>, from: &'a Named<NAME, U>) -> Result<Self, LoadError> {
        context
            .field(NAME, NAME, |cx| T::load(cx, &from.0))
            .map(Named)
    }
}

impl<'t, const NAME: &'static str, T> Load<(&'t mut Named<NAME, T>, Named<NAME, T>)>
    for &'t mut Named<NAME, T>
where
    for<'u> &'u mut T: Load<(&'u mut T, T)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut Named<NAME, T>, Named<NAME, T>),
    ) -> Result<Self, LoadError> {
        context.field(NAME, NAME, |cx| {
            <&mut T as Load<_>>::load(cx, (&mut into.0, from.0))
        })?;
        Ok(into)
    }
}
