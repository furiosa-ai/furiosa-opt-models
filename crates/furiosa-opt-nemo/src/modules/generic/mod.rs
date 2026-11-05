//! Generic combinators: they route operands and outputs between child modules and hold no model
//! constants.

use crate::Module;
use furiosa_opt_std::prelude::DeviceSend;

/// Returns the input unchanged.
#[derive(Clone, Default, DeviceSend)]
pub struct Identity;

impl<'d, D, X> Module<(&'d mut D, X)> for Identity {
    type Output = X;

    fn forward(&self, (_, input): (&'d mut D, X)) -> X {
        input
    }
}

/// Returns the first element of an input pair.
#[derive(Clone, Default, DeviceSend)]
pub struct First;

impl<D, X, Y> Module<(&mut D, (X, Y))> for First {
    type Output = X;

    fn forward(&self, (_, (first, _)): (&mut D, (X, Y))) -> X {
        first
    }
}

/// Applies A, then passes its output by value to B.
#[derive(Clone, DeviceSend)]
pub struct Sequential<A, B>(pub A, pub B);

// --8<-- [start:sequential]
impl<'d, D, A, B, X, Y, Z> Module<(&'d mut D, X)> for Sequential<A, B>
where
    A: for<'a> Module<(&'a mut D, X), Output = Y>,
    B: for<'a> Module<(&'a mut D, Y), Output = Z>,
{
    type Output = Z;

    fn forward(&self, (device, input): (&'d mut D, X)) -> Z {
        let output = self.0.forward((&mut *device, input));
        self.1.forward((&mut *device, output))
    }
}
// --8<-- [end:sequential]

/// Applies A, then lends its output to B without transferring ownership.
#[derive(DeviceSend)]
pub struct Then<A, B>(pub A, pub B);

impl<D, I, Y, Z, A, B> Module<(&mut D, I)> for Then<A, B>
where
    A: for<'d> Module<(&'d mut D, I), Output = Y>,
    B: for<'d, 'y> Module<(&'d mut D, &'y Y), Output = Z>,
{
    type Output = Z;

    fn forward(&self, (device, input): (&mut D, I)) -> Z {
        let hidden = self.0.forward((&mut *device, input));
        self.1.forward((device, &hidden))
    }
}

/// Applies A and B to the same borrowed input, returning both outputs.
#[derive(Clone, DeviceSend)]
pub struct Fork<A, B>(pub A, pub B);

impl<'d, 'x, D, A, B, X, U, V> Module<(&'d mut D, &'x X)> for Fork<A, B>
where
    A: for<'a> Module<(&'a mut D, &'x X), Output = U>,
    B: for<'a> Module<(&'a mut D, &'x X), Output = V>,
{
    type Output = (U, V);

    fn forward(&self, (device, input): (&'d mut D, &'x X)) -> Self::Output {
        let x = self.0.forward((&mut *device, input));
        let y = self.1.forward((&mut *device, input));
        (x, y)
    }
}

/// Applies A and B to separate inputs, returning both outputs.
#[derive(Clone, DeviceSend)]
pub struct Product<A, B>(pub A, pub B);

impl<'d, D, A, B, X, Y, U, V> Module<(&'d mut D, (X, Y))> for Product<A, B>
where
    A: for<'a> Module<(&'a mut D, X), Output = U>,
    B: for<'a> Module<(&'a mut D, Y), Output = V>,
{
    type Output = (U, V);

    fn forward(&self, input: (&'d mut D, (X, Y))) -> Self::Output {
        (&self.0, &self.1).forward(input)
    }
}

/// Lends A and B the two inputs of a borrowed pair.
impl<'d, 'x, D, A, B, X, Y, U, V> Module<(&'d mut D, &'x (X, Y))> for Product<A, B>
where
    A: for<'a> Module<(&'a mut D, &'x X), Output = U>,
    B: for<'a> Module<(&'a mut D, &'x Y), Output = V>,
{
    type Output = (U, V);

    fn forward(&self, (device, (x, y)): (&'d mut D, &'x (X, Y))) -> Self::Output {
        (&self.0, &self.1).forward((device, (x, y)))
    }
}

/// Lends A and B the two inputs of an exclusively borrowed pair.
impl<'d, 'x, D, A, B, X, Y, U, V> Module<(&'d mut D, &'x mut (X, Y))> for Product<A, B>
where
    A: for<'a> Module<(&'a mut D, &'x mut X), Output = U>,
    B: for<'a> Module<(&'a mut D, &'x mut Y), Output = V>,
{
    type Output = (U, V);

    fn forward(&self, (device, (x, y)): (&'d mut D, &'x mut (X, Y))) -> Self::Output {
        (&self.0, &self.1).forward((device, (x, y)))
    }
}

impl<'d, D, A, B, X, Y, U, V> Module<(&'d mut D, (X, Y))> for (A, B)
where
    A: for<'a> Module<(&'a mut D, X), Output = U>,
    B: for<'a> Module<(&'a mut D, Y), Output = V>,
{
    type Output = (U, V);

    fn forward(&self, (device, (x, y)): (&'d mut D, (X, Y))) -> Self::Output {
        let x = self.0.forward((&mut *device, x));
        let y = self.1.forward((&mut *device, y));
        (x, y)
    }
}

impl<'d, D, A, B, C, X, Y, Z, U, V, W> Module<(&'d mut D, (X, Y, Z))> for (A, B, C)
where
    A: for<'a> Module<(&'a mut D, X), Output = U>,
    B: for<'a> Module<(&'a mut D, Y), Output = V>,
    C: for<'a> Module<(&'a mut D, Z), Output = W>,
{
    type Output = (U, V, W);

    fn forward(&self, (device, (x, y, z)): (&'d mut D, (X, Y, Z))) -> Self::Output {
        let u = self.0.forward((&mut *device, x));
        let v = self.1.forward((&mut *device, y));
        let w = self.2.forward((&mut *device, z));
        (u, v, w)
    }
}

mod choice;
pub use choice::Either;

mod each;
pub use each::Each;

mod layers;
pub use layers::{Contexts, Layers, Repeat};

pub mod collection;
pub use collection::{Stacked, View};

mod interleave;
pub use interleave::{Branch, Interleave};

mod schedule;
pub(crate) use schedule::{Entry, Schedule};

mod context;
pub use context::{Borrow, Chain, ProductWithContexts};

#[cfg(test)]
mod tests {
    use super::*;

    struct Scale(i32);

    impl Module<(&mut Vec<i32>, i32)> for Scale {
        type Output = i32;

        fn forward(&self, (trace, input): (&mut Vec<i32>, i32)) -> i32 {
            trace.push(self.0);
            input * self.0
        }
    }

    /// Either inputs run only the matching child and return its common output.
    #[test]
    fn selects_common_output() {
        let mut trace = Vec::new();
        let graph = (Scale(2), Scale(3));
        let input: Either<i32, i32> = Either::Right(7);
        let output: i32 = graph.forward((&mut trace, input));
        assert_eq!(output, 21);
        assert_eq!(trace, [3]);
    }

    /// Composition evaluates only the selected or present module, in declared sequence order.
    #[test]
    fn preserves_composition_order() {
        let mut trace = Vec::new();
        let graph = Sequential((Scale(2), Scale(3)), (Identity, Scale(5)));
        assert_eq!(graph.forward((&mut trace, (7, 11))), (14, 165));
        assert_eq!(trace, [2, 3, 5]);
        let selected: Either<Scale, Scale> = Either::Right(Scale(13));
        assert_eq!(selected.forward((&mut trace, 2)), 26);
        assert_eq!(trace, [2, 3, 5, 13]);
        let absent: Option<Scale> = None;
        let present = Some(Scale(7));
        assert_eq!(absent.forward((&mut trace, 3)), None);
        assert_eq!(trace, [2, 3, 5, 13]);
        assert_eq!(present.forward((&mut trace, 3)), Some(21));
        assert_eq!(trace, [2, 3, 5, 13, 7]);
    }
}
