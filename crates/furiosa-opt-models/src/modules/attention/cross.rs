//! Cross-attention: queries projected from the input attend over keys and values from a source.

use super::{Attention, Source, Stored};
use crate::Module;
use furiosa_opt_std::prelude::{Device, DeviceSend};

/// Projects the input and applies this call's positions to the projection.
#[derive(DeviceSend)]
pub struct PositionedProjection<Projection, Position> {
    pub projection: Projection,
    pub position: Position,
}

/// The state is the projection's slot.
impl<'x, 'p, X, P, Y, Z, S, Projection, Position, Env: Copy> Module<&(&'x X, &'p P), Env, S>
    for PositionedProjection<Projection, Position>
where
    Projection: for<'t> Module<&'t X, Env, S, Output = Y>,
    Position: for<'a, 'b, 's> Module<(&'a Y, &'b P), Env, &'s mut (), Output = Z>,
{
    type Output = Z;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        state: S,
        (input, positions): &(&'x X, &'p P),
    ) -> Z {
        let projected = self.projection.forward(&mut *device, env, state, input);
        self.position
            .forward(device, env, &mut (), (&projected, positions))
    }
}

/// Projects keys and values and positions only the keys.
#[derive(DeviceSend)]
pub struct PositionedKeyValue<Projection, Position> {
    pub projection: Projection,
    pub position: Position,
}

/// The state is the projection's slot.
impl<X, P, K, V, R, S, Projection, Position, Env: Copy> Module<&(X, P), Env, S>
    for PositionedKeyValue<Projection, Position>
where
    Projection: for<'x> Module<&'x X, Env, S, Output = (K, V)>,
    Position: for<'a, 'b, 's> Module<(&'a K, &'b P), Env, &'s mut (), Output = R>,
{
    type Output = (R, V);

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        state: S,
        (input, positions): &(X, P),
    ) -> Self::Output {
        let (key, value) = self.projection.forward(&mut *device, env, state, input);
        let key = self
            .position
            .forward(device, env, &mut (), (&key, positions));
        (key, value)
    }
}

/// Cross attention whose queries take this call's positions; stored source keys keep their own.
#[derive(DeviceSend)]
pub struct PositionedCross<Attention>(pub Attention);

/// A composition, not an op.
impl<X, Attention> !crate::Op<&X> for PositionedCross<Attention> {}

/// The env holds the query positions, then the cross attention's env; the state is the cross
/// attention's.
impl<'e, X, P, E: Copy, S, O, Attention> Module<&X, (&'e P, E), S> for PositionedCross<Attention>
where
    Attention: for<'x, 'q> Module<&'q (&'x X, &'e P), E, S, Output = O>,
{
    type Output = O;

    fn forward(&self, device: &mut Device, (positions, env): (&'e P, E), state: S, input: &X) -> O {
        self.0.forward(device, env, state, &(input, positions))
    }
}

/// Attention whose queries come from the input and keys and values from stored source memory: its
/// projection projects only the queries.
pub type Cross<Q, K, W, R, A, P> = Attention<Q, Stored<K, W, R>, A, P>;

/// A source attention reads the source rows and what its attend reads beside them from the env.
impl<'x, 'e, X, E, C: Copy, S, Qkv, Projection, Attend, Project, Out> Module<&'x X, (&'e E, C), S>
    for Attention<Qkv, Source<Projection>, Attend, Project>
where
    Self: Module<((&'x X, (&'e E, C)), ()), (&'e E, C), S, Output = Out>,
{
    type Output = Out;

    fn forward(&self, device: &mut Device, env: (&'e E, C), state: S, input: &'x X) -> Out {
        self.forward(device, env, state, ((input, env), ()))
    }
}

/// A stored attention reads this call's access of its layer's memory from the env: a write of the
/// source rows or a read, beside what its attend reads, or a borrowed reader its parent reads.
impl<'x, X, E: Copy, S, Qkv, Projection, Write, Read, Attend, Project, Out> Module<&'x X, E, S>
    for Attention<Qkv, Stored<Projection, Write, Read>, Attend, Project>
where
    Self: Module<((&'x X, E), ()), E, S, Output = Out>,
{
    type Output = Out;

    fn forward(&self, device: &mut Device, env: E, state: S, input: &'x X) -> Out {
        self.forward(device, env, state, ((input, env), ()))
    }
}
