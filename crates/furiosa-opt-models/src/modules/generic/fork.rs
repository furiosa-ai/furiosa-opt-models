//! One borrowed input lent to each module of a pair or triple.

use crate::Module;
use crate::load::Load;
use furiosa_opt_std::prelude::{Device, DeviceSend};

/// Lends one borrowed input to each module of a pair or triple, returning their outputs in order.
#[derive(Clone, DeviceSend, Load)]
pub struct Fork<T>(pub T);

/// A composition, not an op.
impl<X, T> !crate::Op<&X> for Fork<T> {}

impl<'x, X, O, A, B, S, Env: Copy> Module<&'x X, Env, S> for Fork<(A, B)>
where
    (A, B): Module<(&'x X, &'x X), Env, S, Output = O>,
{
    type Output = O;

    fn forward(&self, device: &mut Device, env: Env, state: S, input: &'x X) -> O {
        self.0.forward(device, env, state, (input, input))
    }
}

impl<'x, X, O, A, B, C, S, Env: Copy> Module<&'x X, Env, S> for Fork<(A, B, C)>
where
    (A, B, C): Module<(&'x X, &'x X, &'x X), Env, S, Output = O>,
{
    type Output = O;

    fn forward(&self, device: &mut Device, env: Env, state: S, input: &'x X) -> O {
        self.0.forward(device, env, state, (input, input, input))
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::Scale;
    use super::*;
    use crate::tests::DEVICE;
    use furiosa_opt_std::prelude::Topology;
    use std::cell::RefCell;

    /// A fork lends one input to every module, in module order.
    #[test]
    fn forks_in_module_order() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let trace = &RefCell::new(Vec::new());
        assert_eq!(
            Fork((Scale(2), Scale(3))).forward(device, trace, &mut ((), ()), &7),
            (14, 21)
        );
        let triple = Fork((Scale(5), Scale(7), Scale(11)));
        assert_eq!(
            triple.forward(device, trace, &mut ((), (), ()), &2),
            (10, 14, 22)
        );
        assert_eq!(*trace.borrow(), [2, 3, 5, 7, 11]);
    }
}
