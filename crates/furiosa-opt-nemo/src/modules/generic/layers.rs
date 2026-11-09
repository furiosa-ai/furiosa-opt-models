//! Layer collections applied in order, with one shared context, one context per layer, or one
//! body repeated per context.

use crate::{DeviceModule, Module};
use furiosa_opt_std::prelude::{Device, unroll};

/// Apply each layer in order with the same borrowed context.
impl<X, C: Copy, L, const N: usize> Module<(&mut Device, (X, C))> for [L; N]
where
    L: for<'x> DeviceModule<(&'x X, C), Output = X>,
{
    type Output = X;

    fn forward(&self, (device, (mut hidden, context)): (&mut Device, (X, C))) -> X {
        if N > 0 {
            #[unroll]
            for index in 0..N {
                hidden = self[index].forward((&mut *device, (&hidden, context)));
            }
        }
        hidden
    }
}
