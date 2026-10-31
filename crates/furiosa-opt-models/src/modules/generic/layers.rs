//! Layers applied in order: one fold over a collection of layers and the state slot of each, its
//! loop rolled or unrolled.
//!
//! - A collection holds a fixed number of layers and applies the one at an index: `[L; N]` gives
//!   each layer its own parameters, `Repeat` one body to every layer.
//! - The call's state is one slot a layer, `&mut [S; N]`; each layer takes the hidden value with
//!   its own slot and returns the next hidden value.
//! - The loop kind is a kernel difference: [`Rolled`] runs one copy of a layer's code every trip,
//!   [`Unrolled`] places every layer's code before scheduling.

use crate::Module;
use crate::load::Load;
use furiosa_opt_std::prelude::{Device, DeviceSend, unroll};

/// Apply each layer in order with its own state slot, unrolled.
impl<'s, X, LayerState, L, const N: usize, Env: Copy> Module<X, Env, &'s mut [LayerState; N]>
    for [L; N]
where
    L: Module<X, Env, &'s mut LayerState, Output = X>,
{
    type Output = X;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        state: &'s mut [LayerState; N],
        hidden: X,
    ) -> X {
        Layers {
            layers: self,
            kind: Unrolled,
        }
        .forward(device, env, state, hidden)
    }
}

/// The layers of a collection applied in order, each with its own state slot; the kind `K` rolls
/// or unrolls the loop.
#[derive(DeviceSend, Load)]
pub struct Layers<C, K = Rolled> {
    #[load(rename = "")]
    pub layers: C,
    #[load(rename = "")]
    pub kind: K,
}

/// A device loop over the layers: one copy of a layer's code runs every trip.
#[derive(Clone, Copy, Default, DeviceSend, Load)]
pub struct Rolled;

/// Every layer's code in place: the loop unrolls before scheduling.
#[derive(Clone, Copy, Default, DeviceSend, Load)]
pub struct Unrolled;

impl<'s, X, LayerState, C, const N: usize, Env: Copy> Module<X, Env, &'s mut [LayerState; N]>
    for Layers<C, Rolled>
where
    C: Collection<X, Env, &'s mut LayerState, Output = X>,
{
    type Output = X;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        state: &'s mut [LayerState; N],
        mut hidden: X,
    ) -> X {
        const { assert!(C::COUNT == N, "one state slot for each layer") };
        if N > 0 {
            for (index, slot) in state.iter_mut().enumerate() {
                hidden = self.layers.apply(&mut *device, env, slot, hidden, index);
            }
        }
        hidden
    }
}

impl<'s, X, LayerState, C, const N: usize, Env: Copy> Module<X, Env, &'s mut [LayerState; N]>
    for Layers<C, Unrolled>
where
    C: Collection<X, Env, &'s mut LayerState, Output = X>,
{
    type Output = X;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        state: &'s mut [LayerState; N],
        mut hidden: X,
    ) -> X {
        const { assert!(C::COUNT == N, "one state slot for each layer") };
        if N > 0 {
            #[unroll]
            for (index, slot) in state.iter_mut().enumerate() {
                hidden = self.layers.apply(&mut *device, env, slot, hidden, index);
            }
        }
        hidden
    }
}

/// A fixed number of layers, each taking the call's env and its own state slot; a trait, as an
/// array, a repeated body and stacked views hold layers differently and `Layers` takes any.
pub(crate) trait Collection<X, Env, S> {
    /// What a layer returns for hidden `X`: the next hidden value, of `X`'s type in a loop.
    type Output;

    /// How many layers the collection holds.
    const COUNT: usize;

    /// Applies the layer at `index` to `hidden` with that layer's state slot.
    fn apply(
        &self,
        device: &mut Device,
        env: Env,
        state: S,
        hidden: X,
        index: usize,
    ) -> Self::Output;
}

/// Each layer has its own parameters.
impl<X, Env, S, L: Module<X, Env, S>, const N: usize> Collection<X, Env, S> for [L; N] {
    type Output = L::Output;

    const COUNT: usize = N;

    fn apply(&self, device: &mut Device, env: Env, state: S, hidden: X, index: usize) -> L::Output {
        self[index].forward(device, env, state, hidden)
    }
}

/// A borrowed collection applies its layers in place.
impl<X, Env, S, C: Collection<X, Env, S>> Collection<X, Env, S> for &C {
    type Output = C::Output;

    const COUNT: usize = C::COUNT;

    fn apply(&self, device: &mut Device, env: Env, state: S, hidden: X, index: usize) -> C::Output {
        (**self).apply(device, env, state, hidden, index)
    }
}

/// One body for each of `N` layers.
#[derive(DeviceSend)]
pub struct Repeat<Body, const N: usize> {
    pub body: Body,
}

impl<X, Env, S, Body: Module<X, Env, S>, const N: usize> Collection<X, Env, S> for Repeat<Body, N> {
    type Output = Body::Output;

    const COUNT: usize = N;

    fn apply(&self, device: &mut Device, env: Env, state: S, hidden: X, _: usize) -> Body::Output {
        self.body.forward(device, env, state, hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::Step;
    use super::*;
    use crate::tests::DEVICE;
    use furiosa_opt_std::prelude::Topology;

    /// An attention that appends its input to the cache view its layer is lent.
    struct Attend;

    impl<'v, 'c, E> Module<i32, E, &'v mut &'c mut Vec<i32>> for Attend {
        type Output = i32;

        fn forward(&self, _: &mut Device, _: E, cache: &'v mut &'c mut Vec<i32>, x: i32) -> i32 {
            cache.push(x);
            cache.iter().sum()
        }
    }

    struct Double;

    impl<E> Module<i32, E, &mut ()> for Double {
        type Output = i32;

        fn forward(&self, _: &mut Device, _: E, _: &mut (), x: i32) -> i32 {
            2 * x
        }
    }

    /// An array of layers, and rolled layers, thread the hidden value through each layer and that
    /// layer's own state.
    #[test]
    fn gives_each_layer_its_state() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let stack = [Step, Step];
        let mut state = [Vec::new(), Vec::new()];
        assert_eq!(stack.forward(device, (), &mut state, 1), 1);
        assert_eq!(stack.forward(device, (), &mut state, 2), 4);
        assert_eq!(state, [vec![1, 2], vec![1, 3]]);
        let rolled = Layers {
            layers: [Step, Step],
            kind: Rolled,
        };
        let mut state = [Vec::new(), Vec::new()];
        assert_eq!(rolled.forward(device, (), &mut state, 1), 1);
        assert_eq!(rolled.forward(device, (), &mut state, 2), 4);
        assert_eq!(state, [vec![1, 2], vec![1, 3]]);
    }

    /// Repeated parameters consume independent iteration states in order.
    #[test]
    fn preserves_iteration_states() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        struct Accumulate;
        impl<Env> Module<i32, Env, &mut i32> for Accumulate {
            type Output = i32;

            fn forward(&self, _: &mut Device, _: Env, state: &mut i32, hidden: i32) -> i32 {
                *state += hidden;
                *state
            }
        }

        let graph = Layers {
            layers: Repeat::<_, 2> {
                body: Layers {
                    layers: [Accumulate, Accumulate],
                    kind: Rolled,
                },
            },
            kind: Unrolled,
        };
        let mut states = [[1, 10], [100, 1000]];
        let output = graph.forward(device, (), &mut states, 2);
        assert_eq!(output, 1113);
        assert_eq!(states, [[3, 13], [113, 1113]]);
    }

    /// A decoder layer of an attention and a stateless MLP composite writes only its own layer's
    /// cache, and a call over another backing leaves the first one's caches as they were.
    #[test]
    fn gives_each_layer_its_cache() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let layer = || crate::Sequential(Attend, crate::Sequential(Double, Double));
        let layers = Layers {
            layers: [layer(), layer()],
            kind: Rolled,
        };
        let mut first = [vec![1], vec![2]];
        let mut state = first.each_mut().map(|cache| (cache, Default::default()));
        assert_eq!(layers.forward(device, (), &mut state, 1), 40);
        let mut second = [Vec::new(), Vec::new()];
        let mut state = second.each_mut().map(|cache| (cache, Default::default()));
        assert_eq!(layers.forward(device, (), &mut state, 1), 16);
        assert_eq!(first, [vec![1, 1], vec![2, 8]]);
        assert_eq!(second, [vec![1], vec![4]]);
    }
}
