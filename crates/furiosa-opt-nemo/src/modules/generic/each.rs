//! One module applied independently to every input.

use crate::Module;
use furiosa_opt_std::prelude::DeviceSend;

/// Apply one function independently to every input.
#[derive(Clone, DeviceSend)]
pub struct Each<M>(pub M);

impl<D, I, O, M, const N: usize> Module<(&mut D, [I; N])> for Each<M>
where
    M: for<'d> Module<(&'d mut D, I), Output = O>,
{
    type Output = [O; N];

    fn forward(&self, (device, inputs): (&mut D, [I; N])) -> Self::Output {
        inputs.map(|input| self.0.forward((&mut *device, input)))
    }
}

impl<D, I, O, E: Clone, M> Module<(&mut D, &Result<I, E>)> for Each<M>
where
    M: for<'d, 'i> Module<(&'d mut D, &'i I), Output = O>,
{
    type Output = Result<O, E>;

    fn forward(&self, (device, input): (&mut D, &Result<I, E>)) -> Self::Output {
        match input {
            Ok(input) => Ok(self.0.forward((device, input))),
            Err(error) => Err(error.clone()),
        }
    }
}

impl<D, I, C, O, E: Clone, M> Module<(&mut D, (&Result<I, E>, C))> for Each<M>
where
    M: for<'d, 'i> Module<(&'d mut D, (&'i I, C)), Output = O>,
{
    type Output = Result<O, E>;

    fn forward(&self, (device, (input, context)): (&mut D, (&Result<I, E>, C))) -> Self::Output {
        match input {
            Ok(input) => Ok(self.0.forward((device, (input, context)))),
            Err(error) => Err(error.clone()),
        }
    }
}
