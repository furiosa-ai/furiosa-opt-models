//! Deterministic Euler integration of a velocity field along an explicit sigma schedule.
use crate::Module;
use furiosa_opt_std::prelude::*;
use std::{borrow::Borrow, marker::PhantomData};

pub type Values<S> = HbmTensor<f32, m![1], S>;

#[derive(DeviceSend)]
pub struct Timestep {
    pub current: Values<m![1]>,
    pub next: Values<m![1]>,
}

#[derive(DeviceSend)]
pub struct Update<S>(pub PhantomData<S>);

impl<S: M, X: Borrow<Values<S>>, Y: Borrow<Values<S>>, T: Borrow<Timestep>>
    Module<(&mut Device, (X, Y, T))> for Update<S>
{
    type Output = Values<S>;

    fn forward(&self, _: (&mut Device, (X, Y, T))) -> Self::Output {
        todo!("NPU Euler sample + (next_sigma - current_sigma) * velocity")
    }
}

#[derive(DeviceSend)]
pub struct Context<S: M> {
    pub sample: Values<S>,
    pub prediction: Values<S>,
    pub timestep: Timestep,
}

/// The sample, the prediction and the timestep a launch's context holds.
impl<S: M> Module<(&mut Device, &mut Context<S>)> for Update<S> {
    type Output = Values<S>;

    fn forward(&self, (device, context): (&mut Device, &mut Context<S>)) -> Values<S> {
        self.forward((
            device,
            (&context.sample, &context.prediction, &context.timestep),
        ))
    }
}
