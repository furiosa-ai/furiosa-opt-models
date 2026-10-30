//! The projection around spatial attention, into the attention's layout and back to the input's
//! shape.

use crate::Module;
use furiosa_opt_std::prelude::{Device, DeviceSend};

/// Projects the input into the body's layout, runs the body and restores the input's shape.
#[derive(DeviceSend)]
pub struct Projection<Project, Body, Restore> {
    pub project: Project,
    pub body: Body,
    pub restore: Restore,
}

/// A composition, not an op.
impl<X, Project, Body, Restore> !crate::Op<&X> for Projection<Project, Body, Restore> {}

/// The body takes the projected rows; the state is its slot.
impl<X, Y, Z, S, O, BodyState, Project, Body, Restore, Env: Copy> Module<&X, Env, BodyState>
    for Projection<Project, Body, Restore>
where
    Project: for<'d, 'x> Module<&'x X, Env, &'d mut (), Output = (Y, S)>,
    Body: Module<Y, Env, BodyState, Output = Z>,
    Restore: for<'d, 'z> Module<(&'z Z, S), Env, &'d mut (), Output = O>,
{
    type Output = O;

    fn forward(&self, device: &mut Device, env: Env, state: BodyState, input: &X) -> O {
        let (hidden, shape) = self.project.forward(&mut *device, env, &mut (), input);
        let hidden = self.body.forward(&mut *device, env, state, hidden);
        self.restore.forward(device, env, &mut (), (&hidden, shape))
    }
}
