//! Structures that expand the hidden value into streams, run layers on them and reduce them back.

use crate::Module;
use furiosa_opt_std::prelude::{Device, DeviceSend};

/// Merges the hyper-connection streams into the layer input, runs it and expands the result.
#[derive(DeviceSend)]
pub struct HyperConnection<Mixer, Merge, Expand> {
    pub mixer: Mixer,
    pub merge: Merge,
    pub expand: Expand,
}

/// The state is the layer's slot.
impl<X, R, P, Q, S, Mixer, Merge, Expand, Env: Copy> Module<(X, R, P, Q), Env, S>
    for HyperConnection<Mixer, Merge, Expand>
where
    Merge: for<'b, 'd> Module<(&'b X, &'b R, &'b P, &'b Q), Env, &'d mut (), Output = X>,
    Mixer: Module<X, Env, S, Output = X>,
    Expand: for<'x, 'd> Module<&'x X, Env, &'d mut (), Output = (R, P, Q)>,
{
    type Output = (X, R, P, Q);

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        state: S,
        values: (X, R, P, Q),
    ) -> Self::Output {
        let hidden = self.merge.forward(
            &mut *device,
            env,
            &mut (),
            (&values.0, &values.1, &values.2, &values.3),
        );
        let hidden = self.mixer.forward(&mut *device, env, state, hidden);
        let (residual, post, mix) = self.expand.forward(device, env, &mut (), &hidden);
        (hidden, residual, post, mix)
    }
}

/// Expands the hidden value into hyper-connection streams, runs the body and returns the hidden.
#[derive(DeviceSend)]
pub struct HyperBody<Expand, Body> {
    pub expand: Expand,
    pub body: Body,
}

/// A composition, not an op.
impl<X, Expand, Body> !crate::Op<X> for HyperBody<Expand, Body> {}

/// The state is the body's slot.
impl<X, R, P, Q, S, Expand, Body, Env: Copy> Module<X, Env, S> for HyperBody<Expand, Body>
where
    Expand: for<'x, 'd> Module<&'x X, Env, &'d mut (), Output = (R, P, Q)>,
    Body: Module<(X, R, P, Q), Env, S, Output = (X, R, P, Q)>,
{
    type Output = X;

    fn forward(&self, device: &mut Device, env: Env, state: S, hidden: X) -> Self::Output {
        let (residual, post, mix) = self.expand.forward(&mut *device, env, &mut (), &hidden);
        let (hidden, _, _, _) =
            self.body
                .forward(device, env, state, (hidden, residual, post, mix));
        hidden
    }
}

/// Expands the hidden value into parallel streams, runs the body and reduces the streams back.
#[derive(DeviceSend)]
pub struct StreamBody<Expand, Body, Reduce> {
    pub expand: Expand,
    pub body: Body,
    pub reduce: Reduce,
}

/// The tokens every stream layer embeds are the input's; the state is the body's slot.
impl<'t, X, T, S, Streams, Expand, Body, Reduce, Env: Copy> Module<(X, &'t T), Env, S>
    for StreamBody<Expand, Body, Reduce>
where
    Expand: for<'x, 'd> Module<&'x X, Env, &'d mut (), Output = Streams>,
    Body: Module<(Streams, &'t T), Env, S, Output = (Streams, &'t T)>,
    Reduce: for<'x, 'd> Module<&'x Streams, Env, &'d mut (), Output = X>,
{
    type Output = X;

    fn forward(&self, device: &mut Device, env: Env, state: S, (hidden, tokens): (X, &'t T)) -> X {
        let streams = self.expand.forward(&mut *device, env, &mut (), &hidden);
        let (streams, _) = self
            .body
            .forward(&mut *device, env, state, (streams, tokens));
        self.reduce.forward(device, env, &mut (), &streams)
    }
}

/// Runs the layer on the reduced streams and mixes its output and the token embedding back in.
#[derive(DeviceSend)]
pub struct StreamLayer<Reduce, Embedding, Compute, Mix> {
    pub reduce: Reduce,
    pub embedding: Embedding,
    pub module: Compute,
    pub mix: Mix,
}

/// The state is the layer's slot.
impl<'t, X, T, Y, S, Streams, Reduce, Embedding, Compute, Mix, Env: Copy>
    Module<(Streams, &'t T), Env, S> for StreamLayer<Reduce, Embedding, Compute, Mix>
where
    Reduce: for<'x, 'd> Module<&'x Streams, Env, &'d mut (), Output = X>,
    Embedding: for<'d> Module<&'t T, Env, &'d mut (), Output = Y>,
    Compute: Module<X, Env, S, Output = X>,
    Mix: for<'b, 'd> Module<(&'b Streams, &'b X, &'b Y), Env, &'d mut (), Output = Streams>,
{
    type Output = (Streams, &'t T);

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        state: S,
        (streams, tokens): (Streams, &'t T),
    ) -> Self::Output {
        let embedding = self.embedding.forward(&mut *device, env, &mut (), tokens);
        let hidden = self.reduce.forward(&mut *device, env, &mut (), &streams);
        let hidden = self.module.forward(&mut *device, env, state, hidden);
        let streams = self
            .mix
            .forward(device, env, &mut (), (&streams, &hidden, &embedding));
        (streams, tokens)
    }
}
