//! Ways a residual merges its branch output back into the input: a projected shortcut, a gate or
//! a scale.

use crate::Module;
use crate::load::Load;
use furiosa_opt_std::prelude::{Device, DeviceSend};

/// Adds the branch output to a projection of the input.
#[derive(DeviceSend, Load)]
pub struct Shortcut<Project, Add>(pub Project, pub Add);

impl<X, Y, Z, O, Project, Add, Env: Copy> Module<(&X, &Y), Env, &mut ()> for Shortcut<Project, Add>
where
    Project: for<'d, 'x> Module<&'x X, Env, &'d mut (), Output = Z>,
    Add: for<'d, 'z, 'y> Module<(&'z Z, &'y Y), Env, &'d mut (), Output = O>,
{
    type Output = O;

    fn forward(&self, device: &mut Device, env: Env, _: &mut (), (input, branch): (&X, &Y)) -> O {
        let shortcut = self.0.forward(&mut *device, env, &mut (), input);
        self.1.forward(device, env, &mut (), (&shortcut, branch))
    }
}

/// Adds the branch output, masked to the live rows and scaled by the activated gate, to the input.
#[derive(DeviceSend, Load)]
pub struct Gated<Weight, Activation, Mask, Add> {
    #[load(rename = "")]
    pub gate: Weight,
    #[load(rename = "")]
    pub activation: Activation,
    #[load(rename = "")]
    pub mask: Mask,
    #[load(rename = "")]
    pub add: Add,
}

/// The live rows the mask keeps are the env.
impl<'e, X, Y, R, Weight, G, Activation, Mask, Add> Module<(&X, &Y), &'e R, &mut ()>
    for Gated<Weight, Activation, Mask, Add>
where
    Activation: for<'d, 'w> Module<&'w Weight, &'e R, &'d mut (), Output = G>,
    Mask: for<'d, 'y> Module<(&'y Y, &'e R), &'e R, &'d mut (), Output = Y>,
    Add: for<'d, 'x, 'y, 'g> Module<(&'x X, &'y Y, &'g G), &'e R, &'d mut (), Output = X>,
{
    type Output = X;

    fn forward(
        &self,
        device: &mut Device,
        rows: &'e R,
        _: &mut (),
        (input, branch): (&X, &Y),
    ) -> X {
        let branch = self
            .mask
            .forward(&mut *device, rows, &mut (), (branch, rows));
        let gate = self
            .activation
            .forward(&mut *device, rows, &mut (), &self.gate);
        self.add
            .forward(device, rows, &mut (), (input, &branch, &gate))
    }
}

/// Adds the branch output, scaled by the activated weight, to the input.
#[derive(DeviceSend, Load)]
pub struct Scaled<Weight, Activation, Add> {
    #[load(rename = "")]
    pub weight: Weight,
    #[load(rename = "")]
    pub activation: Activation,
    #[load(rename = "")]
    pub add: Add,
}

impl<X, Y, Weight, G, Activation, Add, Env: Copy> Module<(&X, &Y), Env, &mut ()>
    for Scaled<Weight, Activation, Add>
where
    Activation: for<'d, 'w> Module<&'w Weight, Env, &'d mut (), Output = G>,
    Add: for<'d, 'x, 'y, 'g> Module<(&'x X, &'y Y, &'g G), Env, &'d mut (), Output = X>,
{
    type Output = X;

    fn forward(&self, device: &mut Device, env: Env, _: &mut (), (input, branch): (&X, &Y)) -> X {
        let gate = self
            .activation
            .forward(&mut *device, env, &mut (), &self.weight);
        self.add
            .forward(device, env, &mut (), (input, branch, &gate))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modules::residual::Residual;
    use crate::tests::DEVICE;
    use furiosa_opt_std::prelude::Topology;
    use std::cell::RefCell;

    /// Each module records itself in the log it holds, so a test reads the order of the calls.
    struct Scale<'t>(i32, &'t RefCell<Vec<i32>>);

    impl<E> Module<&i32, E, &mut ()> for Scale<'_> {
        type Output = i32;

        fn forward(&self, _: &mut Device, _: E, _: &mut (), input: &i32) -> i32 {
            self.1.borrow_mut().push(self.0);
            input * self.0
        }
    }

    struct Mask<'t>(&'t RefCell<Vec<i32>>);

    impl<E> Module<(&i32, &i32), E, &mut ()> for Mask<'_> {
        type Output = i32;

        fn forward(&self, _: &mut Device, _: E, _: &mut (), (input, rows): (&i32, &i32)) -> i32 {
            self.0.borrow_mut().push(10);
            input * rows
        }
    }

    struct Add<'t>(&'t RefCell<Vec<i32>>);

    impl<E> Module<(&i32, &i32, &i32), E, &mut ()> for Add<'_> {
        type Output = i32;

        fn forward(
            &self,
            _: &mut Device,
            _: E,
            _: &mut (),
            (input, branch, gate): (&i32, &i32, &i32),
        ) -> i32 {
            self.0.borrow_mut().push(20);
            input + branch * gate
        }
    }

    /// Row masking precedes gate activation, and both apply only to the residual branch.
    #[test]
    fn masks_branch_before_gate() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let trace = &RefCell::new(Vec::new());
        let graph = Residual(
            Scale(2, trace),
            Scale(3, trace),
            Gated {
                gate: 1,
                activation: Scale(2, trace),
                mask: Mask(trace),
                add: Add(trace),
            },
        );
        let output = graph.forward(device, &3, &mut (), 7);
        assert_eq!(output, 259);
        assert_eq!(*trace.borrow(), [2, 3, 10, 2, 20]);
    }
}
