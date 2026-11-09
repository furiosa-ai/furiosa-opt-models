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

// --8<-- [end:sequential]

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

mod layers;

pub mod collection;

mod context;
