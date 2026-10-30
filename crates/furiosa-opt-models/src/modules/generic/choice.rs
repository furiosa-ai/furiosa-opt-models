//! A choice between two inputs or two modules; a module choice is fixed when its parameters load.

use crate::Module;
use furiosa_opt_std::prelude::Device;

/// A choice of value or module; a module choice is fixed when its parameters load.
#[derive(Clone)]
pub enum Either<A, B> {
    /// Selects the first alternative.
    Left(A),
    /// Selects the second alternative.
    Right(B),
}

/// A composition, not an op.
impl<I, A, B> !crate::Op<I> for Either<A, B> {}

impl<'s, I, J, O, A, B, AState, BState, Env: Copy>
    Module<Either<I, J>, Env, &'s mut (AState, BState)> for (A, B)
where
    A: Module<I, Env, &'s mut AState, Output = O>,
    B: Module<J, Env, &'s mut BState, Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (first, second): &'s mut (AState, BState),
        input: Either<I, J>,
    ) -> Self::Output {
        match input {
            Either::Left(input) => self.0.forward(device, env, first, input),
            Either::Right(input) => self.1.forward(device, env, second, input),
        }
    }
}

impl<I, E, S, O, A, B> Module<I, E, S> for Either<A, B>
where
    A: Module<I, E, S, Output = O>,
    B: Module<I, E, S, Output = O>,
{
    type Output = O;

    fn forward(&self, device: &mut Device, env: E, state: S, input: I) -> O {
        match self {
            Self::Left(module) => module.forward(device, env, state, input),
            Self::Right(module) => module.forward(device, env, state, input),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::Scale;
    use super::*;
    use crate::tests::DEVICE;
    use furiosa_opt_std::prelude::Topology;
    use std::cell::RefCell;

    /// A choice runs only the matching child and returns its output: a pair on an `Either`
    /// input, and an `Either` of modules.
    #[test]
    fn selects_common_output() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let trace = &RefCell::new(Vec::new());
        let input: Either<i32, i32> = Either::Right(7);
        let output: i32 = (Scale(2), Scale(3)).forward(device, trace, &mut ((), ()), input);
        assert_eq!(output, 21);
        let selected: Either<Scale, Scale> = Either::Left(Scale(13));
        assert_eq!(selected.forward(device, trace, &mut (), 2), 26);
        assert_eq!(*trace.borrow(), [3, 13]);
    }
}
