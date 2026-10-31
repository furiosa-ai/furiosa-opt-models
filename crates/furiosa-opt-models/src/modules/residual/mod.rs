//! Residual connections distinguished by normalization placement and summed branches.

use crate::Module;
use crate::load::Load;
use furiosa_opt_std::prelude::*;
pub use merge::{Gated, Scaled, Shortcut};

mod merge;

/// `x + branch(norm(x))`; state belongs to the branch, weights to the modules.
#[derive(DeviceSend, Load)]
pub struct Residual<Norm, Branch, Add>(pub Norm, pub Branch, pub Add);

/// A composition, not an op.
impl<X, Norm, Branch, Add> !crate::Op<X> for Residual<Norm, Branch, Add> {}

impl<X, Y, Z, O, S, Norm, Branch, Add, Env: Copy> Module<X, Env, S> for Residual<Norm, Branch, Add>
where
    Norm: for<'d, 'x> Module<&'x X, Env, &'d mut (), Output = Y>,
    Branch: for<'y> Module<&'y Y, Env, S, Output = Z>,
    Add: for<'d, 'x, 'z> Module<(&'x X, &'z Z), Env, &'d mut (), Output = O>,
{
    type Output = O;

    fn forward(&self, device: &mut Device, env: Env, state: S, input: X) -> O {
        let normalized = self.0.forward(&mut *device, env, &mut (), &input);
        let branch = self.1.forward(&mut *device, env, state, &normalized);
        self.2.forward(device, env, &mut (), (&input, &branch))
    }
}

/// Without a norm the branch reads the input itself.
impl<X, Y, O, S, Branch, Add, Env: Copy> Module<X, Env, S> for Residual<(), Branch, Add>
where
    Branch: for<'x> Module<&'x X, Env, S, Output = Y>,
    Add: for<'d, 'x, 'y> Module<(&'x X, &'y Y), Env, &'d mut (), Output = O>,
{
    type Output = O;

    fn forward(&self, device: &mut Device, env: Env, state: S, input: X) -> O {
        let branch = self.1.forward(&mut *device, env, state, &input);
        self.2.forward(device, env, &mut (), (&input, &branch))
    }
}

/// Lends the merge the outputs of its parts, a `Fork` of branches over one input or a pair of
/// modules over a pair of inputs: `Sum<Fork<(A, B)>, Add>` computes `a(x) + b(x)`.
#[derive(Clone, DeviceSend, Load)]
pub struct Sum<Parts, Merge>(pub Parts, pub Merge);

/// The state holds the parts' slot, then the merge's.
impl<'s, X, Y, Z, O, PartsState, MergeState, Parts, Merge, Env: Copy>
    Module<X, Env, &'s mut (PartsState, MergeState)> for Sum<Parts, Merge>
where
    Parts: Module<X, Env, &'s mut PartsState, Output = (Y, Z)>,
    Merge: for<'y> Module<(&'y Y, &'y Z), Env, &'s mut MergeState, Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (parts, merge): &'s mut (PartsState, MergeState),
        input: X,
    ) -> O {
        let outputs = self.0.forward(&mut *device, env, parts, input);
        self.1.forward(device, env, merge, (&outputs.0, &outputs.1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::DEVICE;
    use crate::{Fork, Then};
    use furiosa_opt_std::prelude::Topology;
    use std::cell::RefCell;

    struct Scale(i32);

    impl Module<&i32, &RefCell<Vec<i32>>, &mut ()> for Scale {
        type Output = i32;

        fn forward(
            &self,
            _: &mut Device,
            trace: &RefCell<Vec<i32>>,
            _: &mut (),
            input: &i32,
        ) -> i32 {
            trace.borrow_mut().push(self.0);
            input * self.0
        }
    }

    struct Plus;

    impl Module<(&i32, &i32), &RefCell<Vec<i32>>, &mut ()> for Plus {
        type Output = i32;

        fn forward(
            &self,
            _: &mut Device,
            trace: &RefCell<Vec<i32>>,
            _: &mut (),
            (input, branch): (&i32, &i32),
        ) -> i32 {
            trace.borrow_mut().push(0);
            input + branch
        }
    }

    struct Difference;

    impl Module<(&i32, &i32), &RefCell<Vec<i32>>, &mut ()> for Difference {
        type Output = i32;

        fn forward(
            &self,
            _: &mut Device,
            trace: &RefCell<Vec<i32>>,
            _: &mut (),
            (first, second): (&i32, &i32),
        ) -> i32 {
            trace.borrow_mut().push(0);
            first - second
        }
    }

    /// A pre-norm residual retains the original input and evaluates norm, branch, then sum.
    #[test]
    fn preserves_pre_norm_order() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let trace = &RefCell::new(Vec::new());
        let graph = Residual(Scale(2), Scale(3), Plus);
        let output = graph.forward(device, trace, &mut (), 7);
        assert_eq!(output, 49);
        assert_eq!(*trace.borrow(), [2, 3, 0]);
    }

    /// A post-norm residual normalizes the sum rather than the branch or original input.
    #[test]
    fn normalizes_residual_sum() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let trace = &RefCell::new(Vec::new());
        let graph = Then(Residual((), Scale(3), Plus), Scale(2));
        let output = graph.forward(device, trace, &mut ((), ()), 7);
        assert_eq!(output, 56);
        assert_eq!(*trace.borrow(), [3, 0, 2]);
    }

    /// A sum runs its parts before the merge, which reads their outputs in part order.
    #[test]
    fn merges_parts_in_order() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let graph = Sum(Fork((Scale(2), Scale(3))), Difference);
        let trace = &RefCell::new(Vec::new());
        let output = graph.forward(device, trace, &mut (((), ()), ()), &7);
        assert_eq!(output, -7);
        assert_eq!(*trace.borrow(), [2, 3, 0]);
    }
}
