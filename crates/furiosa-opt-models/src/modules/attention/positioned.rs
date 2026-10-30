//! A projection stage that positions the queries and keys it projects.

use super::Front;
use crate::Module;
use crate::load::Load;
use crate::modules::generic::schedule::{IntoStream, Slot, Stream};
use crate::tensor::live::Live;
use core::task::Poll;
use furiosa_opt_std::prelude::{AxisName, Device, DeviceSend};

/// Positions queries and keys after projection; values keep their original representation.
#[derive(DeviceSend, Load)]
pub struct Positioned<Projection, Position> {
    #[load(rename = "")]
    pub projection: Projection,
    #[load(rename = "")]
    pub positions: Position,
}

/// The state is the projection's slot.
impl<'x, Input, P, Q, K, V, PositionedQuery, PositionedKey, S, Projection, Position, Env: Copy>
    Module<(Input, &'x P), Env, S> for Positioned<Projection, Position>
where
    Projection: Module<Input, Env, S, Output = (Q, K, V)>,
    Position:
        for<'s> Module<((Q, K), &'x P), Env, &'s mut (), Output = (PositionedQuery, PositionedKey)>,
{
    type Output = (PositionedQuery, PositionedKey, V);

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        state: S,
        (input, positions): (Input, &'x P),
    ) -> Self::Output {
        let (query, key, value) = self.projection.forward(&mut *device, env, state, input);
        let (query, key) = self
            .positions
            .forward(device, env, &mut (), ((query, key), positions));
        (query, key, value)
    }
}

/// Positions stream over tiles given with their positions: the projection streams over the
/// tiles, and each tile it projects is positioned by the positions given with the last tile it
/// took, with its own rows. The node keeps the positions of a tile waiting in the projection's
/// slot until the projection takes it.
impl<U, X, P, E: Copy, S, Projection, Position, C> IntoStream<U, E, S>
    for Positioned<Projection, Position>
where
    U: Stream<Item = (X, P)>,
    Projection: IntoStream<Slot<X>, E, S, Stream = C>,
    Front<Positioned<C, Position>, P, E, Option<P>, U>: Stream,
{
    type Stream = Front<Positioned<C, Position>, P, E, Option<P>, U>;

    fn into_stream(self, env: E, state: S, upstream: U) -> Self::Stream {
        let positioned = Positioned {
            projection: self.projection.into_stream(env, state, Slot::new()),
            positions: self.positions,
        };
        Front::new(positioned, env, None, upstream)
    }
}

/// The projection takes the next tile once it took the last one; the projected tiles, any number
/// a tile, are positioned by the positions given with the last tile it took.
impl<U, X, P: Copy, E: Copy, C, Position, Q, K, V, RQ, RK, R> Stream
    for Front<Positioned<C, Position>, P, E, Option<P>, U>
where
    U: Stream<Item = (X, P)>,
    R: AxisName,
    C: Stream<Item = Live<(Q, K, V), R>> + AsMut<Slot<X>>,
    Position: for<'p, 's> Module<((Q, K), &'p (P, Live<(), R>)), E, &'s mut (), Output = (RQ, RK)>,
{
    type Item = Live<(RQ, RK, V), R>;

    fn poll_next(&mut self, device: &mut Device) -> Poll<Option<Self::Item>> {
        let Self {
            module:
                Positioned {
                    projection,
                    positions,
                },
            held,
            env,
            state: waiting,
            upstream,
        } = self;
        if projection.as_mut().is_empty() {
            match upstream.poll_next(&mut *device) {
                Poll::Ready(Some((input, at))) => {
                    projection.as_mut().put(Poll::Ready(Some(input)));
                    *waiting = Some(at);
                }
                ended => projection.as_mut().put(ended.map(|_| None)),
            }
        }
        let polled = projection.poll_next(&mut *device);
        if projection.as_mut().is_empty()
            && let Some(at) = waiting.take()
        {
            *held = Some(at);
        }
        match (polled, *held) {
            (Poll::Ready(Some(tile)), Some(at)) => {
                let rows = tile.with(());
                let (query, key, value) = tile.into_value();
                let (query, key) =
                    positions.forward(device, *env, &mut (), ((query, key), &(at, rows)));
                Poll::Ready(Some(rows.with((query, key, value))))
            }
            (Poll::Ready(None), _) => Poll::Ready(None),
            _ => Poll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modules::generic::schedule::{RowWise, Slot, Stream};
    use crate::tests::DEVICE;
    use core::task::Poll;
    use furiosa_opt_std::prelude::Topology;

    struct Project;

    impl Module<&([f64; 2], [f64; 2], i32), (), &mut ()> for Project {
        type Output = ([f64; 2], [f64; 2], i32);

        fn forward(&self, _: &mut Device, _: (), _: &mut (), input: &Self::Output) -> Self::Output {
            *input
        }
    }

    struct Rotate;

    impl Module<(([f64; 2], [f64; 2]), &[f64; 2]), (), &mut ()> for Rotate {
        type Output = ([f64; 2], [f64; 2]);

        fn forward(
            &self,
            _: &mut Device,
            _: (),
            _: &mut (),
            ((query, key), &[cos, sin]): (([f64; 2], [f64; 2]), &[f64; 2]),
        ) -> Self::Output {
            let rotate = |[low, high]: [f64; 2]| [low * cos - high * sin, low * sin + high * cos];
            (rotate(query), rotate(key))
        }
    }

    impl Module<&(&([f64; 2], [f64; 2], i32), &[f64; 2]), (), &mut ()> for Positioned<Project, Rotate> {
        type Output = ([f64; 2], [f64; 2], i32);

        fn forward(
            &self,
            device: &mut Device,
            _: (),
            state: &mut (),
            input: &(&([f64; 2], [f64; 2], i32), &[f64; 2]),
        ) -> Self::Output {
            self.forward(device, (), state, *input)
        }
    }

    /// Prepared rotation changes queries and keys, never values, in either execution form.
    #[test]
    fn uses_prepared_rotation() {
        let _lease = DEVICE.blocking_lock();
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let positioned = Positioned {
            projection: Project,
            positions: Rotate,
        };
        let input = ([1.0, 2.0], [3.0, -4.0], 17);
        let outputs = [0.3_f64, 1.1].map(|angle| {
            let rotation = [angle.cos(), angle.sin()];
            let sequential = positioned.forward(&mut device, (), &mut (), (&input, &rotation));
            let mut state = ();
            let mut slot = Slot::new();
            slot.put(Poll::Ready(Some((&input, &rotation))));
            let mut stream = RowWise::new(&positioned, (), &mut state, slot);
            assert_eq!(stream.poll_next(&mut device), Poll::Ready(Some(sequential)));
            sequential
        });
        assert_ne!(outputs[0].0, outputs[1].0);
        assert_ne!(outputs[0].1, outputs[1].1);
        assert_eq!((outputs[0].2, outputs[1].2), (17, 17));
    }
}
