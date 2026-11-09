//! Modules applied one after another: each takes its predecessor's output, moved or lent.

use crate::Module;
use crate::load::Load;
use furiosa_opt_std::prelude::{Device, DeviceSend};

/// Applies A, then passes its output by value to B.
#[derive(Clone, Copy, DeviceSend, Load)]
pub struct Sequential<A, B>(pub A, pub B);

// --8<-- [start:sequential]
impl<'s, A, B, X, AState, BState, Env: Copy> Module<X, Env, &'s mut (AState, BState)>
    for Sequential<A, B>
where
    A: Module<X, Env, &'s mut AState>,
    B: Module<A::Output, Env, &'s mut BState>,
{
    type Output = B::Output;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (first, second): &'s mut (AState, BState),
        input: X,
    ) -> Self::Output {
        let output = self.0.forward(&mut *device, env, first, input);
        self.1.forward(device, env, second, output)
    }
}
// --8<-- [end:sequential]

/// Applies A, then lends its output to B without transferring ownership.
#[derive(Clone, DeviceSend, Load)]
pub struct Then<A, B>(pub A, pub B);

impl<A, B> Then<A, B> {
    pub const fn first(&self) -> &A {
        &self.0
    }

    pub const fn second(&self) -> &B {
        &self.1
    }
}

impl<'s, I, Y, Z, A, B, AState, BState, Env: Copy> Module<I, Env, &'s mut (AState, BState)>
    for Then<A, B>
where
    A: Module<I, Env, &'s mut AState, Output = Y>,
    B: for<'y> Module<&'y Y, Env, &'s mut BState, Output = Z>,
{
    type Output = Z;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (first, second): &'s mut (AState, BState),
        input: I,
    ) -> Z {
        let hidden = self.0.forward(&mut *device, env, first, input);
        self.1.forward(device, env, second, &hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::super::Identity;
    use super::super::tests::{Scale, Step, Write};
    use super::*;
    use crate::tests::DEVICE;
    use furiosa_opt_std::prelude::Topology;
    use std::cell::RefCell;

    /// A step whose output borrows the cache it appended to.
    struct Append;

    impl<'s, E> Module<i32, E, &'s mut Vec<i32>> for Append {
        type Output = &'s [i32];

        fn forward(&self, _: &mut Device, _: E, cache: &'s mut Vec<i32>, input: i32) -> &'s [i32] {
            cache.push(input);
            cache
        }
    }

    /// A sequence's output may borrow the state its last child wrote, past the call.
    const _: for<'s> fn(&mut Device, &'s mut ((), Vec<i32>)) -> &'s [i32] =
        |device, state| Sequential(Identity, Append).forward(device, (), state, 1);

    /// A sequence runs its modules in declared order, each on its predecessor's output.
    #[test]
    fn preserves_composition_order() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let trace = &RefCell::new(Vec::new());
        let graph = Sequential((Scale(2), Scale(3)), (Identity, Scale(5)));
        let state = &mut (((), ()), ((), ()));
        assert_eq!(graph.forward(device, trace, state, (7, 11)), (14, 165));
        assert_eq!(*trace.borrow(), [2, 3, 5]);
    }

    /// A sequence gives each child its own slot of a state tuple, and a later call observes the
    /// slots an earlier one updated.
    #[test]
    fn threads_each_childs_state() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let chain = Sequential(Step, Sequential(Identity, Step));
        let mut state = (Vec::new(), ((), Vec::new()));
        assert_eq!(chain.forward(device, (), &mut state, 1), 1);
        assert_eq!(chain.forward(device, (), &mut state, 2), 4);
        assert_eq!(state, (vec![1, 2], ((), vec![1, 3])));
    }

    /// A destination in a sequence's input reaches the child that writes it, and comes back as its
    /// output.
    #[test]
    fn routes_destinations_in_the_input() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let mut output = 0;
        let written =
            Sequential(Identity, Write).forward(device, (), &mut ((), ()), (9, &mut output));
        assert_eq!(*written, 9);
        assert_eq!(output, 9);
    }
}
