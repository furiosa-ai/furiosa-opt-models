//! Combinators that give each child its own part of the invocation context.

use crate::{First, Identity, Module};

impl<'c, D, X, Y, Z, C, E, A, B> Module<(&mut D, (X, &'c mut (C, E)))> for (A, B, First)
where
    A: for<'d> Module<(&'d mut D, (X, &'c mut C)), Output = Y>,
    B: for<'d> Module<(&'d mut D, (Y, &'c mut E)), Output = Z>,
{
    type Output = Z;

    fn forward(&self, (device, (input, context)): (&mut D, (X, &'c mut (C, E)))) -> Z {
        self.forward((device, (input, (&mut context.0, &mut context.1))))
    }
}

impl<'c, D, X, Y, Z, C, E, A, B> Module<(&mut D, (X, &'c (C, E)))> for (A, B, First)
where
    A: for<'d> Module<(&'d mut D, (X, &'c C)), Output = Y>,
    B: for<'d> Module<(&'d mut D, (Y, &'c E)), Output = Z>,
{
    type Output = Z;

    fn forward(&self, (device, (input, context)): (&mut D, (X, &'c (C, E)))) -> Z {
        self.forward((device, (input, (&context.0, &context.1))))
    }
}

impl<D, X, C, P, O, A, B> Module<(&mut D, (X, &mut C, &P))> for (A, B, First)
where
    Self:
        for<'d, 'c, 'p, 's> Module<(&'d mut D, (X, &'s mut ((&'c mut C, &'p P), ()))), Output = O>,
{
    type Output = O;

    fn forward(&self, (device, (input, cache, context)): (&mut D, (X, &mut C, &P))) -> O {
        self.forward((device, (input, &mut ((cache, context), ()))))
    }
}

impl<D, X, Y, Z, C, E, A, B> Module<(&mut D, (X, (C, E)))> for (A, B, First)
where
    A: for<'a> Module<(&'a mut D, (X, C)), Output = Y>,
    B: for<'a> Module<(&'a mut D, (Y, E)), Output = Z>,
{
    type Output = Z;

    fn forward(&self, (device, (input, (first, second))): (&mut D, (X, (C, E)))) -> Z {
        let hidden = self.0.forward((&mut *device, (input, first)));
        self.1.forward((device, (hidden, second)))
    }
}

impl<D, X, Y, U, V, C, K, A, B> Module<(&mut D, ((X, Y), (C, K)))> for (A, B, Identity)
where
    A: for<'d> Module<(&'d mut D, (X, C)), Output = U>,
    B: for<'d> Module<(&'d mut D, (Y, K)), Output = V>,
{
    type Output = (U, V);

    fn forward(
        &self,
        (device, ((x, y), (first, second))): (&mut D, ((X, Y), (C, K))),
    ) -> Self::Output {
        let x = self.0.forward((&mut *device, (x, first)));
        let y = self.1.forward((device, (y, second)));
        (x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Affine(i32);

    impl Module<(&mut Vec<i32>, (i32, i32))> for Affine {
        type Output = i32;

        fn forward(&self, (trace, (input, context)): (&mut Vec<i32>, (i32, i32))) -> i32 {
            trace.push(context);
            input * self.0 + context
        }
    }

    /// Contextual composition lends each stage only its own invocation operands.
    #[test]
    fn preserves_stage_contexts() {
        let mut trace = Vec::new();
        let graph = (Affine(2), Affine(3), First);
        let output = graph.forward((&mut trace, (7, (11, 13))));
        assert_eq!(output, 88);
        assert_eq!(trace, [11, 13]);
    }

    impl Module<(&mut Vec<i32>, (i32, &i32))> for Affine {
        type Output = i32;

        fn forward(&self, (trace, (input, context)): (&mut Vec<i32>, (i32, &i32))) -> i32 {
            self.forward((trace, (input, *context)))
        }
    }

    /// A borrowed context pair lends each stage its own part, in stage order.
    #[test]
    fn splits_borrowed_contexts() {
        let mut trace = Vec::new();
        let output = (Affine(2), Affine(3), First).forward((&mut trace, (7, &(11, 13))));
        assert_eq!(output, 88);
        assert_eq!(trace, [11, 13]);
    }

    /// Independent contextual calls retain their input pairing and evaluation order.
    #[test]
    fn preserves_independent_contexts() {
        let mut trace = Vec::new();
        let graph = (Affine(2), Affine(3), Identity);
        let output = graph.forward((&mut trace, ((7, 11), (13, 17))));
        assert_eq!(output, (27, 50));
        assert_eq!(trace, [13, 17]);
    }
}
