//! Residual connections distinguished by normalization placement and summed branches.

use crate::Module;
use crate::load::Load;
use crate::modules::generic::schedule::{IntoStream, RowWise, Slot, Stream};
use core::task::Poll;
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

/// A residual's node, the upstream of its add: once the branch took the last input tile, a poll
/// normalizes the next into the branch's slot and gives the add it with the branch's next output.
impl<X, Y, E: Copy, N, Branch, Add> Stream for Residual<(N, E), Branch, Add>
where
    N: for<'a, 'x> Module<&'x X, E, &'a mut (), Output = Y>,
    Branch: Stream + AsMut<Slot<Y>>,
    Add: Stream<Item = X>,
{
    type Item = (Option<X>, Option<Branch::Item>);

    fn poll_next(&mut self, device: &mut Device) -> Poll<Option<Self::Item>> {
        let Self((norm, env), branch, upstream) = self;
        let mut input = None;
        if branch.as_mut().is_empty() {
            match upstream.poll_next(device) {
                Poll::Ready(Some(tile)) => {
                    let normalized = norm.forward(&mut *device, *env, &mut (), &tile);
                    branch.as_mut().put(Poll::Ready(Some(normalized)));
                    input = Some(tile);
                }
                ended => branch.as_mut().put(ended.map(|_| None)),
            }
        }
        match (input, branch.poll_next(device)) {
            (input, Poll::Ready(Some(output))) => Poll::Ready(Some((input, Some(output)))),
            (Some(input), _) => Poll::Ready(Some((Some(input), None))),
            (None, Poll::Pending) => Poll::Pending,
            (None, Poll::Ready(None)) => Poll::Ready(None),
        }
    }
}

/// A residual in a stack is fed through its upstream's slot.
impl<T, N, E, B, U: AsMut<Slot<T>>> AsMut<Slot<T>> for Residual<(N, E), B, U> {
    fn as_mut(&mut self) -> &mut Slot<T> {
        self.2.as_mut()
    }
}

/// The add's stream over the residual's: the branch streams over its slot on the residual's state,
/// as the residual's `forward` lends it, and the add, an op, on a unit.
impl<U, X, Y, E: Copy, S, Norm, Branch, Add, BranchStream, AddStream> IntoStream<U, E, S>
    for Residual<Norm, Branch, Add>
where
    U: Stream<Item = X>,
    Norm: for<'a, 'x> Module<&'x X, E, &'a mut (), Output = Y>,
    Branch: IntoStream<Slot<Y>, E, S, Stream = BranchStream>,
    BranchStream: Stream + AsMut<Slot<Y>>,
    Add:
        for<'u> IntoStream<Residual<(Norm, E), BranchStream, U>, E, &'u mut (), Stream = AddStream>,
    AddStream: Stream,
{
    type Stream = AddStream;

    fn into_stream(self, env: E, state: S, upstream: U) -> Self::Stream {
        let Self(norm, branch, add) = self;
        let branch = branch.into_stream(env, state, Slot::new());
        // The add has no slot in the state and its stream keeps none, so a unit lent to the call
        // alone serves it.
        add.into_stream(env, &mut (), Residual((norm, env), branch, upstream))
    }
}

/// A borrowed residual streams as the residual of its borrowed parts.
impl<'m, U: Stream, E, S, Norm, Branch, Add> IntoStream<U, E, S> for &'m Residual<Norm, Branch, Add>
where
    Residual<&'m Norm, &'m Branch, &'m Add>: IntoStream<U, E, S>,
{
    type Stream = <Residual<&'m Norm, &'m Branch, &'m Add> as IntoStream<U, E, S>>::Stream;

    fn into_stream(self, env: E, state: S, upstream: U) -> Self::Stream {
        Residual(&self.0, &self.1, &self.2).into_stream(env, state, upstream)
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

/// A sum streams one `forward` a tile: its merge reads every part's output of the same tile.
impl<'m, 's, U: Stream, E, S, P, M> IntoStream<U, E, &'s mut S> for &'m Sum<P, M>
where
    RowWise<'s, &'m Sum<P, M>, E, S, U>: Stream,
{
    type Stream = RowWise<'s, &'m Sum<P, M>, E, S, U>;

    fn into_stream(self, env: E, state: &'s mut S, upstream: U) -> Self::Stream {
        RowWise::new(self, env, state, upstream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modules::generic::schedule::tests::{Indices, Launch, Number, Record, drain};
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

    /// Doubles each number.
    struct Double;

    impl<E> Module<&Number, E, &mut ()> for Double {
        type Output = Number;

        fn forward(&self, _: &mut Device, _: E, _: &mut (), at: &Number) -> Number {
            at.with(2 * at.value())
        }
    }

    impl<E> Module<(&Number, &Number), E, &mut ()> for Plus {
        type Output = Number;

        fn forward(
            &self,
            _: &mut Device,
            _: E,
            _: &mut (),
            (at, branch): (&Number, &Number),
        ) -> Number {
            at.with(at.value() + branch.value())
        }
    }

    /// A row-wise branch gives its number with its input's.
    impl<E> Module<&(Option<Number>, Option<Number>), E, &mut ()> for Plus {
        type Output = Number;

        fn forward(
            &self,
            device: &mut Device,
            env: E,
            unit: &mut (),
            pair: &(Option<Number>, Option<Number>),
        ) -> Number {
            let (Some(at), Some(branch)) = pair else {
                panic!("a row-wise branch gives a number for each input number");
            };
            self.forward(device, env, unit, (at, branch))
        }
    }

    /// A residual lends its add a unit for the call alone, so the add's node holds a leaked one.
    impl<'m, 's, E: Copy, U> IntoStream<U, E, &'s mut ()> for &'m Plus
    where
        U: Stream<Item = (Option<Number>, Option<Number>)>,
    {
        type Stream = RowWise<'static, &'m Plus, E, (), U>;

        fn into_stream(self, env: E, _: &'s mut (), upstream: U) -> Self::Stream {
            RowWise::new(self, env, Box::leak(Box::new(())), upstream)
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

    /// A streamed residual lends its branch the slot its call lends and gives what its call
    /// returns.
    #[test]
    fn streams_the_branch_on_the_residual_slot() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let live = Launch::over(3);
        let residual = Residual(Double, Record, Plus);
        let (mut streamed, mut called) = (Vec::new(), Vec::new());
        let numbers = drain(
            device,
            (&residual).into_stream((), &mut &mut streamed, Indices::new(live)),
        );
        let returned: Vec<usize> = (0..live.rows())
            .map(|at| {
                let at = live.subrange(at, 1).with(at);
                residual
                    .forward(device, (), &mut &mut called, at)
                    .into_value()
            })
            .collect();
        assert_eq!(numbers, returned);
        assert_eq!(streamed, called);
        assert_eq!(called, [0, 2, 4]);
    }
}
