//! Generic combinators: they route operands and outputs between child modules and hold no model
//! constants.

use crate::Module;
use crate::load::Load;
pub use choice::Either;
pub use collection::{Stacked, View};
pub use each::Each;
pub use fork::Fork;
use furiosa_opt_std::prelude::{Device, DeviceSend};
pub use interleave::{Branch, Interleave};
pub(crate) use layers::Collection;
pub use layers::{Layers, Repeat, Rolled, Unrolled};
pub(crate) use order::Order;
pub use sequential::{Sequential, Then};

/// Returns the input unchanged.
#[derive(Clone, Default, DeviceSend, Load)]
pub struct Identity;

/// A combinator, not an op.
impl<X> !crate::Op<X> for Identity {}

impl<X, Env> Module<X, Env, &mut ()> for Identity {
    type Output = X;

    fn forward(&self, _: &mut Device, _: Env, _: &mut (), input: X) -> X {
        input
    }
}

/// Returns the first element of an input pair.
#[derive(Clone, Default, DeviceSend, Load)]
pub struct First;

impl<X, Y, Env> Module<(X, Y), Env, &mut ()> for First {
    type Output = X;

    fn forward(&self, _: &mut Device, _: Env, _: &mut (), (first, _): (X, Y)) -> X {
        first
    }
}

impl<'s, A, B, X, Y, AState, BState, Env: Copy> Module<(X, Y), Env, &'s mut (AState, BState)>
    for (A, B)
where
    A: Module<X, Env, &'s mut AState>,
    B: Module<Y, Env, &'s mut BState>,
{
    type Output = (A::Output, B::Output);

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (first, second): &'s mut (AState, BState),
        (x, y): (X, Y),
    ) -> Self::Output {
        let x = self.0.forward(&mut *device, env, first, x);
        (x, self.1.forward(device, env, second, y))
    }
}

impl<'s, A, B, C, X, Y, Z, AState, BState, CState, Env: Copy>
    Module<(X, Y, Z), Env, &'s mut (AState, BState, CState)> for (A, B, C)
where
    A: Module<X, Env, &'s mut AState>,
    B: Module<Y, Env, &'s mut BState>,
    C: Module<Z, Env, &'s mut CState>,
{
    type Output = (A::Output, B::Output, C::Output);

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (first, second, third): &'s mut (AState, BState, CState),
        (x, y, z): (X, Y, Z),
    ) -> Self::Output {
        let u = self.0.forward(&mut *device, env, first, x);
        let v = self.1.forward(&mut *device, env, second, y);
        (u, v, self.2.forward(device, env, third, z))
    }
}

mod sequential;

mod fork;

mod choice;

mod each;

mod layers;

pub mod collection;

mod interleave;

mod order;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::DEVICE;
    use furiosa_opt_std::prelude::Topology;
    use std::cell::RefCell;

    /// A module that multiplies its input and records its factor in the trace its env lends.
    pub(super) struct Scale(pub(super) i32);

    impl Module<i32, &RefCell<Vec<i32>>, &mut ()> for Scale {
        type Output = i32;

        fn forward(
            &self,
            _: &mut Device,
            trace: &RefCell<Vec<i32>>,
            _: &mut (),
            input: i32,
        ) -> i32 {
            trace.borrow_mut().push(self.0);
            input * self.0
        }
    }

    impl Module<&i32, &RefCell<Vec<i32>>, &mut ()> for Scale {
        type Output = i32;

        fn forward(
            &self,
            device: &mut Device,
            trace: &RefCell<Vec<i32>>,
            state: &mut (),
            input: &i32,
        ) -> i32 {
            self.forward(device, trace, state, *input)
        }
    }

    /// A step whose state is its history: it appends its input and returns the history's sum.
    pub(super) struct Step;

    impl<E> Module<i32, E, &mut Vec<i32>> for Step {
        type Output = i32;

        fn forward(&self, _: &mut Device, _: E, history: &mut Vec<i32>, input: i32) -> i32 {
            history.push(input);
            history.iter().sum()
        }
    }

    /// A module that writes its value into the destination its input carries and returns it.
    pub(super) struct Write;

    impl<'x, E> Module<(i32, &'x mut i32), E, &mut ()> for Write {
        type Output = &'x mut i32;

        fn forward(
            &self,
            _: &mut Device,
            _: E,
            _: &mut (),
            (value, into): (i32, &'x mut i32),
        ) -> &'x mut i32 {
            *into = value;
            into
        }
    }

    /// A product gives each child its own slot of a state tuple of different kinds, which keeps
    /// what an earlier call left there.
    #[test]
    fn threads_each_childs_state() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let pair = (Step, Identity);
        let mut state = (vec![10], ());
        assert_eq!(pair.forward(device, (), &mut state, (1, 7)), (11, 7));
        assert_eq!(state, (vec![10, 1], ()));
    }

    /// A destination in a product's input reaches the child that writes it, and comes back as its
    /// output.
    #[test]
    fn routes_destinations_in_the_input() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let mut output = 0;
        let mut state = (Vec::new(), ());
        let (sum, written) = (Step, Write).forward(device, (), &mut state, (3, (5, &mut output)));
        assert_eq!((sum, *written), (3, 5));
        assert_eq!(output, 5);
    }
}
