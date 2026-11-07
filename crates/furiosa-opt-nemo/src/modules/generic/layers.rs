//! Layer collections applied in order, with one shared context, one context per layer, or one
//! body repeated per context.

use crate::{DeviceModule, Module};
use furiosa_opt_std::prelude::{Device, DeviceSend, unroll};

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

/// A fixed collection consumes one context for each layer, in order.
#[derive(DeviceSend)]
pub struct Layers<Modules> {
    pub layers: Modules,
}

impl<D, X, L, P, const N: usize> Module<(&mut D, (X, P))> for Layers<[L; N]>
where
    P: Contexts<L, D, X>,
{
    type Output = X;

    fn forward(&self, (device, (mut hidden, mut contexts)): (&mut D, (X, P))) -> X {
        if N > 0 {
            for index in 0..N {
                hidden = contexts.apply(&self.layers[index], &mut *device, hidden, index);
            }
        }
        hidden
    }
}

/// The context each layer of a collection reads, found by the layer's index in the collection.
pub trait Contexts<L, D, X> {
    /// Applies the selected layer to `hidden` with the context at its collection index.
    fn apply(&mut self, layer: &L, device: &mut D, hidden: X, index: usize) -> X;
}

/// Each layer reads its own context.
impl<L, D, X, C, const M: usize> Contexts<L, D, X> for &mut [C; M]
where
    L: for<'d, 'x, 'c> Module<(&'d mut D, (&'x X, &'c mut C)), Output = X>,
{
    fn apply(&mut self, layer: &L, device: &mut D, hidden: X, index: usize) -> X {
        layer.forward((device, (&hidden, &mut self[index])))
    }
}

/// Each layer reads its own context and the context all layers share.
impl<'p, L, D, X, C, P, const M: usize> Contexts<L, D, X> for (&mut [C; M], &'p P)
where
    L: for<'d, 'x, 'c> Module<(&'d mut D, (&'x X, &'c mut C, &'p P)), Output = X>,
{
    fn apply(&mut self, layer: &L, device: &mut D, hidden: X, index: usize) -> X {
        layer.forward((device, (&hidden, &mut self.0[index], self.1)))
    }
}

/// An absent context skips the collection's layers.
impl<L, D, X> Contexts<L, D, X> for () {
    fn apply(&mut self, _: &L, _: &mut D, hidden: X, _: usize) -> X {
        hidden
    }
}

/// Reuse the same parameters for each supplied context.
#[derive(DeviceSend)]
pub struct Repeat<Body> {
    pub body: Body,
}

impl<D, X, C, Body, const N: usize> Module<(&mut D, (X, &mut [C; N]))> for Repeat<Body>
where
    Body: for<'d, 'c> Module<(&'d mut D, (X, &'c mut C)), Output = X>,
{
    type Output = X;

    fn forward(&self, (device, (mut hidden, contexts)): (&mut D, (X, &mut [C; N]))) -> X {
        if N > 0 {
            #[unroll]
            for index in 0..N {
                hidden = self
                    .body
                    .forward((&mut *device, (hidden, &mut contexts[index])));
            }
        }
        hidden
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Repeated parameters consume independent iteration contexts in order.
    #[test]
    fn preserves_iteration_states() {
        struct Accumulate;
        impl Module<(&mut (), (&i32, &mut i32))> for Accumulate {
            type Output = i32;

            fn forward(&self, (_, (hidden, state)): (&mut (), (&i32, &mut i32))) -> i32 {
                *state += hidden;
                *state
            }
        }

        let graph = Repeat {
            body: Layers {
                layers: [Accumulate, Accumulate],
            },
        };
        let mut states = [[1, 10], [100, 1000]];
        let contexts = &mut states;
        let output = graph.forward((&mut (), (2, contexts)));
        assert_eq!(output, 1113);
        assert_eq!(states, [[3, 13], [113, 1113]]);
    }
}
