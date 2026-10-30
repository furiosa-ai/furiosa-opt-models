//! Joint attention: two streams' queries, keys and values attend together, then split per stream.

use super::Context;
use crate::Module;
use furiosa_opt_std::prelude::{Device, DeviceSend};

/// Joins projected QKV from two streams, attends jointly and projects each split output.
#[derive(DeviceSend)]
pub struct Joint<LeftQkv, RightQkv, Join, Position, Attend, Split, LeftOutput, RightOutput> {
    pub left: LeftQkv,
    pub right: RightQkv,
    pub join: Join,
    pub position: Position,
    pub attend: Attend,
    pub split: Split,
    pub outputs: (LeftOutput, RightOutput),
}

/// The env is the attention's context, whose positions position the joined queries and keys; the
/// state holds each stream's projection slot, then the output projections' slots.
impl<
    's,
    'e,
    X,
    Y,
    C: Copy,
    Left,
    Right,
    J,
    P,
    A,
    B,
    L,
    R,
    O,
    LeftQkvState,
    RightQkvState,
    LeftOutputState,
    RightOutputState,
    LeftQkv,
    RightQkv,
    Join,
    Position,
    Attend,
    Split,
    LeftOutput,
    RightOutput,
>
    Module<
        &(X, Y),
        &'e Context<C, (), ()>,
        &'s mut (
            LeftQkvState,
            RightQkvState,
            (LeftOutputState, RightOutputState),
        ),
    > for Joint<LeftQkv, RightQkv, Join, Position, Attend, Split, LeftOutput, RightOutput>
where
    LeftQkv: for<'x> Module<&'x X, &'e Context<C, (), ()>, &'s mut LeftQkvState, Output = Left>,
    RightQkv: for<'x> Module<&'x Y, &'e Context<C, (), ()>, &'s mut RightQkvState, Output = Right>,
    Join: for<'a, 'b, 'd> Module<
            (&'a Left, &'b Right),
            &'e Context<C, (), ()>,
            &'d mut (),
            Output = J,
        >,
    Position: for<'d> Module<(J, C), &'e Context<C, (), ()>, &'d mut (), Output = P>,
    Attend: for<'x, 'd> Module<&'x P, &'e Context<C, (), ()>, &'d mut (), Output = O>,
    Split: for<'x, 'd> Module<&'x O, &'e Context<C, (), ()>, &'d mut (), Output = (A, B)>,
    LeftOutput: for<'x> Module<&'x A, &'e Context<C, (), ()>, &'s mut LeftOutputState, Output = L>,
    RightOutput:
        for<'x> Module<&'x B, &'e Context<C, (), ()>, &'s mut RightOutputState, Output = R>,
{
    type Output = (L, R);

    fn forward(
        &self,
        device: &mut Device,
        env: &'e Context<C, (), ()>,
        (left_state, right_state, (left_output, right_output)): &'s mut (
            LeftQkvState,
            RightQkvState,
            (LeftOutputState, RightOutputState),
        ),
        (left, right): &(X, Y),
    ) -> Self::Output {
        let left = self.left.forward(&mut *device, env, left_state, left);
        let right = self.right.forward(&mut *device, env, right_state, right);
        let joined = self
            .join
            .forward(&mut *device, env, &mut (), (&left, &right));
        let positioned = self
            .position
            .forward(&mut *device, env, &mut (), (joined, env.positions));
        let attended = self.attend.forward(&mut *device, env, &mut (), &positioned);
        let (left, right) = self.split.forward(&mut *device, env, &mut (), &attended);
        let left = self
            .outputs
            .0
            .forward(&mut *device, env, left_output, &left);
        let right = self.outputs.1.forward(device, env, right_output, &right);
        (left, right)
    }
}
