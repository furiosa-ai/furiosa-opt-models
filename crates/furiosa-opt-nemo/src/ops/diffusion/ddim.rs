//! Deterministic DDIM updates from an epsilon prediction and explicit timestep indices.
use crate::Module;
use furiosa_opt_std::prelude::*;
use std::{borrow::Borrow, marker::PhantomData};

pub type Values<S> = HbmTensor<f32, m![1], S>;
type Coefficient = Values<m![1 # 8]>;

/// Current training index and the previous index; a negative previous index ends the trajectory.
#[derive(DeviceSend)]
pub struct Timestep {
    pub current: HbmTensor<i32, m![1], m![1]>,
    pub previous: HbmTensor<i32, m![1], m![1]>,
}

#[derive(DeviceSend)]
pub struct Update<S, Training: AxisName> {
    pub(crate) alpha: Values<m![Training]>,
    pub(crate) terminal: Coefficient,
    pub(crate) limit: Coefficient,
    pub(crate) shape: PhantomData<S>,
}

impl<S, Training: AxisName> Update<S, Training> {
    pub fn new(alpha: Values<m![Training]>, terminal: Coefficient, limit: Coefficient) -> Self {
        Self {
            alpha,
            terminal,
            limit,
            shape: PhantomData,
        }
    }
}

impl<S: M, T: AxisName, X: Borrow<Values<S>>, Y: Borrow<Values<S>>, J: Borrow<Timestep>>
    Module<(&mut Device, (X, Y, J))> for Update<S, T>
{
    type Output = Values<S>;

    fn forward(&self, _: (&mut Device, (X, Y, J))) -> Self::Output {
        todo!(
            "NPU alpha lookup (negative previous uses terminal); clean=clamp((sample-sqrt(1-alpha)*epsilon)/sqrt(alpha),-limit,limit); previous=sqrt(alpha_previous)*clean+sqrt(1-alpha_previous)*epsilon"
        )
    }
}

#[derive(DeviceSend)]
pub struct Context<S: M> {
    pub sample: Values<S>,
    pub prediction: Values<S>,
    pub timestep: Timestep,
}

/// The sample, the prediction and the timestep a launch's context holds.
impl<S: M, T: AxisName> Module<(&mut Device, &mut Context<S>)> for Update<S, T> {
    type Output = Values<S>;

    fn forward(&self, (device, context): (&mut Device, &mut Context<S>)) -> Values<S> {
        self.forward((
            device,
            (&context.sample, &context.prediction, &context.timestep),
        ))
    }
}
