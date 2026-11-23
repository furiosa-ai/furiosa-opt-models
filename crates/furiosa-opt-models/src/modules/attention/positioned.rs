//! A projection stage that positions the queries and keys it projects.

use crate::Module;
use crate::load::Load;
use furiosa_opt_std::prelude::{Device, DeviceSend};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::DEVICE;
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

    /// Prepared rotation changes queries and keys, never values.
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
            positioned.forward(&mut device, (), &mut (), (&input, &rotation))
        });
        assert_ne!(outputs[0].0, outputs[1].0);
        assert_ne!(outputs[0].1, outputs[1].1);
        assert_eq!((outputs[0].2, outputs[1].2), (17, 17));
    }
}
