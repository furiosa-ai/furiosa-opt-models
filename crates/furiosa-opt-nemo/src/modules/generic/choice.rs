//! A choice between two inputs or two modules; a module choice is fixed when its parameters load.

use crate::Module;

/// A choice of value or module; a module choice is fixed when its parameters load.
#[derive(Clone)]
pub enum Either<A, B> {
    /// Selects the first alternative.
    Left(A),
    /// Selects the second alternative.
    Right(B),
}

impl<D, I, J, O, A, B> Module<(&mut D, Either<I, J>)> for (A, B)
where
    A: for<'a> Module<(&'a mut D, I), Output = O>,
    B: for<'a> Module<(&'a mut D, J), Output = O>,
{
    type Output = O;

    fn forward(&self, (device, input): (&mut D, Either<I, J>)) -> Self::Output {
        match input {
            Either::Left(input) => self.0.forward((device, input)),
            Either::Right(input) => self.1.forward((device, input)),
        }
    }
}

impl<D, I, O, A, B> Module<(&mut D, I)> for Either<A, B>
where
    A: for<'d> Module<(&'d mut D, I), Output = O>,
    B: for<'d> Module<(&'d mut D, I), Output = O>,
{
    type Output = O;

    fn forward(&self, input: (&mut D, I)) -> O {
        match self {
            Self::Left(module) => module.forward(input),
            Self::Right(module) => module.forward(input),
        }
    }
}
