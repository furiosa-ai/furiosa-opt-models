//! Layers applied in order: one fold over a collection of layers and the state slot of each, its
//! loop rolled or unrolled.
//!
//! - A collection holds a fixed number of layers and applies the one at an index: `[L; N]` gives
//!   each layer its own parameters, `Repeat` one body to every layer.
//! - The call's state is one slot a layer, `&mut [S; N]`; each layer takes the hidden value with
//!   its own slot and returns the next hidden value.
//! - The loop kind is a kernel difference: [`Rolled`] runs one copy of a layer's code every trip,
//!   [`Unrolled`] places every layer's code before scheduling.

use super::collection::{Collected, Stack, Stacked, View};
use super::schedule::{IntoStream, Slot, Stream};
use crate::Module;
use crate::load::Load;
use core::iter::Zip;
use core::slice;
use core::task::Poll;
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

// TODO: the device translator refuses this chain: it cannot assign a field of an array element
// through `&mut self`, as each poll does to a layer's slot.
/// The layers chained over the upstream, ending with the last: a poll polls, in order, each layer
/// whose last output the next one took, so a tile passes every layer before the next arrives.
impl<T, U: Stream<Item = T>, C: Stream<Item = T> + AsMut<Slot<T>>, const N: usize> Stream
    for ([C; N], U)
{
    type Item = T;

    fn poll_next(&mut self, device: &mut Device) -> Poll<Option<T>> {
        let mut polled = if self.0[0].as_mut().is_empty() {
            self.1.poll_next(device)
        } else {
            Poll::Pending
        };
        #[unroll]
        for index in 0..N {
            self.0[index].as_mut().put(polled);
            polled = if index + 1 == N || self.0[index + 1].as_mut().is_empty() {
                self.0[index].poll_next(device)
            } else {
                Poll::Pending
            };
        }
        polled
    }
}

impl<'m, 's, U: Stream, E: Copy, S, L, C, const N: usize> IntoStream<U, E, &'s mut [S; N]>
    for &'m [L; N]
where
    &'m L: IntoStream<Slot<U::Item>, E, &'s mut S, Stream = C>,
    C: Stream<Item = U::Item> + AsMut<Slot<U::Item>>,
{
    type Stream = ([C; N], U);

    fn into_stream(self, env: E, slots: &'s mut [S; N], upstream: U) -> Self::Stream {
        (start(self, env, slots), upstream)
    }
}

/// Every layer's code in place: the chain of the collection's layers.
impl<'m, U: Stream, E, S, C> IntoStream<U, E, S> for &'m Layers<C, Unrolled>
where
    &'m C: IntoStream<U, E, S>,
{
    type Stream = <&'m C as IntoStream<U, E, S>>::Stream;

    fn into_stream(self, env: E, state: S, upstream: U) -> Self::Stream {
        (&self.layers).into_stream(env, state, upstream)
    }
}

/// What a rolled stack starts its next layer from: the layers not yet started, each with its state
/// slot, and the call's env.
trait Refill<T> {
    type Stream: Stream<Item = T> + AsMut<Slot<T>>;

    /// The next layer's stream on its state slot, if the stack has one more layer.
    fn start(&mut self) -> Option<Self::Stream>;
}

/// An array's layers, borrowed.
impl<'m, 's, T, E: Copy, S, L, C> Refill<T> for (Zip<slice::Iter<'m, L>, slice::IterMut<'s, S>>, E)
where
    &'m L: IntoStream<Slot<T>, E, &'s mut S, Stream = C>,
    C: Stream<Item = T> + AsMut<Slot<T>>,
{
    type Stream = C;

    fn start(&mut self) -> Option<C> {
        let (layers, env) = self;
        layers
            .next()
            .map(|(layer, slot)| layer.into_stream(*env, slot, Slot::new()))
    }
}

/// A stack's layers, each a view built when it starts, which streams itself; the index is the next
/// layer's.
impl<'m, 's, T, E: Copy, S, L: View<N>, C, const N: usize> Refill<T>
    for (&'m Stacked<L, N>, slice::IterMut<'s, S>, E, usize)
where
    L::Layer: IntoStream<Slot<T>, E, &'s mut S, Stream = C>,
    C: Stream<Item = T> + AsMut<Slot<T>>,
{
    type Stream = C;

    fn start(&mut self) -> Option<C> {
        let (stack, slots, env, index) = self;
        let slot = slots.next()?;
        *index += 1;
        Some(stack.layer(*index - 1).into_stream(*env, slot, Slot::new()))
    }
}

/// Any number of layers, two live at a time on their own state slots; no layers is the upstream.
/// A layer may give one tile before the layer before it is done; it waits for the next layer.
impl<'m, 's, U: Stream, E: Copy, S, L, C, const N: usize> IntoStream<U, E, &'s mut [S; N]>
    for &'m Layers<[L; N], Rolled>
where
    &'m L: IntoStream<Slot<U::Item>, E, &'s mut S, Stream = C>,
    C: Stream<Item = U::Item> + AsMut<Slot<U::Item>>,
{
    type Stream = Layers<
        (
            (Option<C>, Option<C>),
            (Zip<slice::Iter<'m, L>, slice::IterMut<'s, S>>, E),
            (usize, usize),
            Slot<U::Item>,
            U,
        ),
        Rolled,
    >;

    fn into_stream(self, env: E, slots: &'s mut [S; N], upstream: U) -> Self::Stream {
        Layers::start((self.layers.iter().zip(slots.iter_mut()), env), N, upstream)
    }
}

/// A stack of views rolls as an array does: each layer's view is built when it starts.
impl<'m, 's, U: Stream, E: Copy, S, L: View<N>, C, const N: usize> IntoStream<U, E, &'s mut [S; N]>
    for &'m Layers<Collected<Stack, L, N>, Rolled>
where
    L::Layer: IntoStream<Slot<U::Item>, E, &'s mut S, Stream = C>,
    C: Stream<Item = U::Item> + AsMut<Slot<U::Item>>,
{
    type Stream = Layers<
        (
            (Option<C>, Option<C>),
            (&'m Stacked<L, N>, slice::IterMut<'s, S>, E, usize),
            (usize, usize),
            Slot<U::Item>,
            U,
        ),
        Rolled,
    >;

    fn into_stream(self, env: E, slots: &'s mut [S; N], upstream: U) -> Self::Stream {
        Layers::start((&self.layers.0, slots.iter_mut(), env, 0), N, upstream)
    }
}

/// A rolled stack's node: its two live layers, what it starts the next from, the index of the
/// first live layer with the number of layers, the tile waiting for the next layer to start, and
/// its upstream.
impl<T, R, C, U> Layers<((Option<C>, Option<C>), R, (usize, usize), Slot<T>, U), Rolled> {
    /// The node with the stack's first two layers started.
    fn start(mut refill: R, count: usize, upstream: U) -> Self
    where
        R: Refill<T, Stream = C>,
    {
        let live = (refill.start(), refill.start());
        Self {
            layers: (live, refill, (0, count), Slot::new(), upstream),
            kind: Rolled,
        }
    }
}

impl<T, U: Stream<Item = T>, R: Refill<T, Stream = C>, C: Stream<Item = T> + AsMut<Slot<T>>> Stream
    for Layers<((Option<C>, Option<C>), R, (usize, usize), Slot<T>, U), Rolled>
{
    type Item = T;

    fn poll_next(&mut self, device: &mut Device) -> Poll<Option<T>> {
        let ((even, odd), refill, (first, count), waiting, upstream) = &mut self.layers;
        if *count == 0 {
            return upstream.poll_next(device);
        }
        let (earlier, later) = if *first % 2 == 0 {
            (even, odd)
        } else {
            (odd, even)
        };
        // A slot takes the next result of what feeds it only once its layer took the last one.
        let input_empty = match earlier {
            Some(layer) => layer.as_mut().is_empty(),
            None => false,
        };
        let output_empty = match later {
            Some(layer) => layer.as_mut().is_empty(),
            None => true,
        };
        if *first == 0 && input_empty {
            let input = upstream.poll_next(device);
            if let Some(layer) = earlier {
                layer.as_mut().put(input);
            }
        }
        let mut given = Poll::Pending;
        if output_empty {
            given = match earlier {
                Some(layer) => layer.poll_next(device),
                None => Poll::Ready(None),
            };
        }
        // The earlier layer is done: the later one's input ends and it leads, and layer
        // `first + 2` takes the earlier one's place, starting on the tile waiting for it, and is
        // the next layer polled.
        let refilled = matches!(given, Poll::Ready(None));
        if refilled {
            let Some(layer) = later.as_mut() else {
                return Poll::Ready(None);
            };
            layer.as_mut().put(Poll::Ready(None));
            *earlier = refill.start();
            *first += 1;
            if let Some(started) = earlier {
                started.as_mut().put(waiting.poll_next(device));
            }
        }
        // The next layer is layer `first + 1`: its tiles leave the stack if it is the last one,
        // and wait for the layer after it otherwise.
        let last = *first + 2 == *count;
        let next = if refilled { earlier } else { later };
        let Some(next) = next else {
            return if refilled { Poll::Pending } else { given };
        };
        if let Poll::Ready(Some(tile)) = given {
            next.as_mut().put(Poll::Ready(Some(tile)));
        }
        if !last && !waiting.is_empty() {
            return Poll::Pending;
        }
        match next.poll_next(device) {
            Poll::Ready(Some(tile)) if last => Poll::Ready(Some(tile)),
            Poll::Ready(Some(tile)) => {
                waiting.put(Poll::Ready(Some(tile)));
                Poll::Pending
            }
            _ => Poll::Pending,
        }
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

/// Starts each layer of a chain on its own state slot, over its [`Slot`].
pub(super) fn start<'m, 's, T, E: Copy, S, L, const N: usize>(
    layers: &'m [L; N],
    env: E,
    slots: &'s mut [S; N],
) -> [<&'m L as IntoStream<Slot<T>, E, &'s mut S>>::Stream; N]
where
    &'m L: IntoStream<Slot<T>, E, &'s mut S>,
{
    let mut index = 0;
    slots.each_mut().map(|slot| {
        index += 1;
        (&layers[index - 1]).into_stream(env, slot, Slot::new())
    })
}

#[cfg(test)]
mod tests {
    use super::super::schedule::tests::{Collect, Indices, Launch, Number, Record, drain};
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

    /// A layer that takes its whole input, as [`Collect`] does, after a first poll that takes none.
    struct Idle;

    struct IdleNode<S> {
        node: S,
        polled: usize,
    }

    impl<S: Stream> Stream for IdleNode<S> {
        type Item = S::Item;

        fn poll_next(&mut self, device: &mut Device) -> Poll<Option<S::Item>> {
            if self.polled < 1 {
                self.polled += 1;
                return Poll::Pending;
            }
            self.node.poll_next(device)
        }
    }

    impl<T, S: AsMut<Slot<T>>> AsMut<Slot<T>> for IdleNode<S> {
        fn as_mut(&mut self) -> &mut Slot<T> {
            self.node.as_mut()
        }
    }

    impl<'m, 's, U: Stream<Item = Number>> IntoStream<U, Launch, &'s mut ()> for &'m Idle {
        type Stream = IdleNode<<&'static Collect as IntoStream<U, Launch, &'s mut ()>>::Stream>;

        fn into_stream(self, live: Launch, unit: &'s mut (), upstream: U) -> Self::Stream {
            IdleNode {
                node: (&Collect).into_stream(live, unit, upstream),
                polled: 0,
            }
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

    /// A streamed rolled stack writes each layer's own slot: each layer, a sequence, records its
    /// input in the history its slot views.
    #[test]
    fn streams_each_layer_on_its_slot() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let live = Launch::over(3);
        let layers = Layers {
            layers: [
                crate::Sequential(Record, Collect),
                crate::Sequential(Record, Collect),
            ],
            kind: Rolled,
        };
        let mut histories = [Vec::new(), Vec::new()];
        let mut slots = histories.each_mut().map(|history| (history, ()));
        let stream = (&layers).into_stream(live, &mut slots, Indices::new(live));
        assert_eq!(drain(device, stream), [2, 3, 4]);
        assert_eq!(histories, [[0, 1, 2], [1, 2, 3]]);
    }

    /// A rolled stack of row-wise layers over one tile, as a call's rows stream, passes the tile
    /// through every layer in order and ends: each layer records its input and gives the next.
    #[test]
    fn streams_a_tile_through_every_rowwise_layer() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let live = Launch::over(1);
        let layers = Layers {
            layers: [Record, Record, Record, Record],
            kind: Rolled,
        };
        let mut histories = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut slots = histories.each_mut();
        let source = Indices::new(live);
        let numbers = drain(device, (&layers).into_stream(live, &mut slots, source));
        assert_eq!(numbers, [4]);
        assert_eq!(histories, [[0], [1], [2], [3]]);
    }

    /// A rolled stack waits for a layer that does not take a tile yet: every tile reaches every
    /// layer and the stack ends.
    #[test]
    fn waits_for_a_layer_to_take_its_tile() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let live = Launch::over(3);
        let layers = Layers {
            layers: [Idle, Idle, Idle],
            kind: Rolled,
        };
        let source = Indices::new(live);
        let numbers = drain(device, (&layers).into_stream(live, &mut [(); 3], source));
        assert_eq!(numbers, [0, 1, 2]);
    }

    /// A rolled stack of no layers gives its upstream's tiles.
    #[test]
    fn streams_no_layers_as_its_upstream() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let live = Launch::over(3);
        let layers = Layers {
            layers: [Idle; 0],
            kind: Rolled,
        };
        let source = Indices::new(live);
        let numbers = drain(device, (&layers).into_stream(live, &mut [(); 0], source));
        assert_eq!(numbers, [0, 1, 2]);
    }

    /// A rolled stack of layers that each take every tile before they give one, as an attention
    /// does, passes every tile through every layer in order and ends.
    #[test]
    fn streams_every_tile_through_every_whole_layer() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let live = Launch::over(3);
        let layers = Layers {
            layers: [
                crate::Sequential(Record, Collect),
                crate::Sequential(Record, Collect),
                crate::Sequential(Record, Collect),
                crate::Sequential(Record, Collect),
            ],
            kind: Rolled,
        };
        let mut histories = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut slots = histories.each_mut().map(|history| (history, ()));
        let source = Indices::new(live);
        let numbers = drain(device, (&layers).into_stream(live, &mut slots, source));
        assert_eq!(numbers, [4, 5, 6]);
        assert_eq!(histories, [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]);
    }

    /// A streamed rolled stack of views writes each layer's own slot as one of an array does: each
    /// layer is a view built when it starts.
    #[test]
    fn streams_each_view_on_its_slot() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let live = Launch::over(3);
        let layers = Layers {
            layers: Collected::<Stack, crate::Sequential<Record, Collect>, 2>(Stacked {
                storage: ((), ()),
            }),
            kind: Rolled,
        };
        let mut histories = [Vec::new(), Vec::new()];
        let mut slots = histories.each_mut().map(|history| (history, ()));
        let source = Indices::new(live);
        let numbers = drain(device, (&layers).into_stream(live, &mut slots, source));
        assert_eq!(numbers, [2, 3, 4]);
        assert_eq!(histories, [[0, 1, 2], [1, 2, 3]]);
    }
}
