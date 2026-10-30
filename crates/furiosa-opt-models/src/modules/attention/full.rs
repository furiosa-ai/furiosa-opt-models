//! Attention over this call's queries, keys and values alone, without a cache.

use super::Context;
use crate::Module;
use crate::load::Load;
use furiosa_opt_std::prelude::{Device, DeviceSend};

/// Attends to Q/K/V from this call only, without a cache, then projects the attended rows with
/// `Project` (`()` leaves them); each child reads under the names its family gives it.
#[derive(DeviceSend, Load)]
pub struct Full<Qkv, Attend, Project> {
    #[load(rename = "")]
    pub qkv: Qkv,
    #[load(rename = "")]
    pub attend: Attend,
    #[load(rename = "")]
    pub output: Project,
}

/// The env is the attention's context, positions for the projection and a mask, bias or visibility
/// for the attend; the state holds the projection's slot, then the output projection's.
impl<'e, 's, X, P: Copy, R: Copy, Q, K, V, Y, Z, QkvState, ProjectState, Qkv, Attend, Project>
    Module<&X, &'e Context<P, (), R>, &'s mut (QkvState, ProjectState)>
    for Full<Qkv, Attend, Project>
where
    Qkv: for<'x> Module<(&'x X, P), &'e Context<P, (), R>, &'s mut QkvState, Output = (Q, K, V)>,
    Attend:
        for<'b, 'd> Module<(&'b Q, &'b K, &'b V, R), &'e Context<P, (), R>, &'d mut (), Output = Y>,
    Project: Projection<Y, &'e Context<P, (), R>, &'s mut ProjectState, Output = Z>,
{
    type Output = Z;

    fn forward(
        &self,
        device: &mut Device,
        context: &'e Context<P, (), R>,
        (qkv, output): &'s mut (QkvState, ProjectState),
        input: &X,
    ) -> Self::Output {
        let (query, keys, values) =
            self.qkv
                .forward(&mut *device, context, qkv, (input, context.positions));
        let hidden = self.attend.forward(
            &mut *device,
            context,
            &mut (),
            (&query, &keys, &values, context.read),
        );
        self.output.project(device, context, output, hidden)
    }
}

/// Without a context the attend reads Q/K/V alone.
impl<'s, X, Q, K, V, Y, Z, QkvState, ProjectState, Qkv, Attend, Project>
    Module<&X, (), &'s mut (QkvState, ProjectState)> for Full<Qkv, Attend, Project>
where
    Qkv: for<'x> Module<&'x X, (), &'s mut QkvState, Output = (Q, K, V)>,
    Attend: for<'b, 'd> Module<(&'b Q, &'b K, &'b V), (), &'d mut (), Output = Y>,
    Project: for<'x> Module<&'x Y, (), &'s mut ProjectState, Output = Z>,
{
    type Output = Z;

    fn forward(
        &self,
        device: &mut Device,
        env: (),
        (qkv, output): &'s mut (QkvState, ProjectState),
        input: &X,
    ) -> Self::Output {
        let (query, keys, values) = self.qkv.forward(&mut *device, env, qkv, input);
        let hidden = self
            .attend
            .forward(&mut *device, env, &mut (), (&query, &keys, &values));
        self.output.forward(device, env, output, &hidden)
    }
}

/// The output projection of the attended rows, or `()` for none: a block that projects them with
/// other rows projects them itself.
trait Projection<Y, E, S> {
    type Output;

    fn project(&self, device: &mut Device, env: E, state: S, hidden: Y) -> Self::Output;
}

impl<Y, E> Projection<Y, E, &mut ()> for () {
    type Output = Y;

    fn project(&self, _: &mut Device, _: E, _: &mut (), hidden: Y) -> Y {
        hidden
    }
}

impl<Y, Z, E, S, P: for<'x> Module<&'x Y, E, S, Output = Z>> Projection<Y, E, S> for P {
    type Output = Z;

    fn project(&self, device: &mut Device, env: E, state: S, hidden: Y) -> Z {
        self.forward(device, env, state, &hidden)
    }
}
