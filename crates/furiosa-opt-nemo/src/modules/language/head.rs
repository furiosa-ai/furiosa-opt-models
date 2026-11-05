//! Heads that read a body's output: readouts, poolers, and retrieval, sparse and masked heads.

use crate::{Binary, DeviceModule, Module, Sequential, Then, Unary};
use furiosa_opt_std::prelude::{Device, DeviceSend};

/// Applies a head to borrowed body output and returns only the head output.
#[derive(Clone, DeviceSend)]
pub struct Readout<Body, Head>(pub Sequential<Body, Head>);

impl<Body, Head> Readout<Body, Head> {
    pub const fn new(body: Body, head: Head) -> Self {
        Self(Sequential(body, head))
    }

    pub const fn body(&self) -> &Body {
        &self.0.0
    }

    pub const fn head(&self) -> &Head {
        &self.0.1
    }

    pub const fn body_mut(&mut self) -> &mut Body {
        &mut self.0.0
    }

    pub fn into_parts(self) -> (Body, Head) {
        (self.0.0, self.0.1)
    }
}

impl<B, H> Readout<B, H> {
    /// The body whose output the head reads, so a family names it through its catalog binding.
    pub type Body = B;
}

impl<D, I, X, O, Body, Head> Module<(&mut D, I)> for Readout<Body, Head>
where
    Body: for<'d> Module<(&'d mut D, I), Output = X>,
    Head: for<'d, 'x> Module<(&'d mut D, &'x X), Output = O>,
{
    type Output = O;

    fn forward(&self, (device, input): (&mut D, I)) -> Self::Output {
        Then(self.body(), self.head()).forward((device, input))
    }
}

/// Returns both body features and the head output computed from those features.
#[derive(Clone, DeviceSend)]
pub struct FeatureHead<Body, Head> {
    pub body: Body,
    pub head: Head,
}

impl<D, I, X, O, Body, Head> Module<(&mut D, I)> for FeatureHead<Body, Head>
where
    Body: for<'d> Module<(&'d mut D, I), Output = X>,
    Head: for<'d, 'x> Module<(&'d mut D, &'x X), Output = O>,
{
    type Output = (X, O);

    fn forward(&self, (device, input): (&mut D, I)) -> Self::Output {
        let hidden = self.body.forward((&mut *device, input));
        let output = self.head.forward((device, &hidden));
        (hidden, output)
    }
}

/// Runs a body, then passes its hidden output and the caller's pooling context to a head.
#[derive(Clone, DeviceSend)]
pub struct Pooler<Body, Head> {
    pub body: Body,
    pub head: Head,
}

impl<I, C, X, O, Body, Head> Module<(&mut Device, (I, C))> for Pooler<Body, Head>
where
    Body: DeviceModule<I, Output = X>,
    Head: for<'b> DeviceModule<(&'b X, C), Output = O>,
{
    type Output = O;

    fn forward(&self, (device, (input, context)): (&mut Device, (I, C))) -> Self::Output {
        let hidden = self.body.forward((&mut *device, input));
        self.head.forward((device, (&hidden, context)))
    }
}

impl<I, C, X, O, Body, Head> Module<(&mut Device, &(I, C))> for Pooler<Body, Head>
where
    Body: Unary<I, Output = X>,
    Head: Binary<X, C, Output = O>,
{
    type Output = O;

    fn forward(&self, (device, (input, context)): (&mut Device, &(I, C))) -> O {
        let hidden = self.body.forward((&mut *device, input));
        self.head.forward((device, (&hidden, context)))
    }
}

impl<I, C, X, O, Body, Head> Module<(&mut Device, &mut (I, C))> for Pooler<Body, Head>
where
    Body: for<'i> DeviceModule<&'i mut I, Output = X>,
    Head: Binary<X, C, Output = O>,
{
    type Output = O;

    fn forward(&self, (device, (input, context)): (&mut Device, &mut (I, C))) -> O {
        let hidden = self.body.forward((&mut *device, input));
        self.head.forward((device, (&hidden, &*context)))
    }
}

/// Pools hidden rows using the supplied context, then computes sequence scores.
pub type SequenceHead<Pool, Score> = Then<Pool, Score>;

/// Produces dense, token and sparse retrieval outputs, each with its own context.
#[derive(Clone, DeviceSend)]
pub struct RetrievalHead<Dense, Tokens, Sparse> {
    pub dense: Dense,
    pub tokens: Tokens,
    pub sparse: Sparse,
}

impl<'s, X, C, K, S, Y, Z, O, Dense, Tokens, Sparse> Module<(&mut Device, (&X, &'s (C, K, S)))>
    for RetrievalHead<Dense, Tokens, Sparse>
where
    Dense: for<'b> DeviceModule<(&'b X, &'s C), Output = Y>,
    Tokens: for<'b> DeviceModule<(&'b X, &'s K), Output = Z>,
    Sparse: for<'b> DeviceModule<(&'b X, &'s S), Output = O>,
{
    type Output = (Y, Z, O);

    fn forward(
        &self,
        (device, (hidden, contexts)): (&mut Device, (&X, &'s (C, K, S))),
    ) -> Self::Output {
        let dense = self.dense.forward((&mut *device, (hidden, &contexts.0)));
        let tokens = self.tokens.forward((&mut *device, (hidden, &contexts.1)));
        let sparse = self.sparse.forward((device, (hidden, &contexts.2)));
        (dense, tokens, sparse)
    }
}

/// Projects hidden features, applies an activation and normalization, then computes logits.
pub type MaskedHead<Projection, Activation, Norm, Head> =
    Then<Projection, Then<Activation, Then<Norm, Head>>>;

/// Projects hidden features to token logits and pools them with the supplied sparse context.
#[derive(DeviceSend)]
pub struct SparseHead<Projection, Pool> {
    pub projection: Projection,
    pub pool: Pool,
}

impl<X, C, Y, O, Projection, Pool> Module<(&mut Device, (&X, C))> for SparseHead<Projection, Pool>
where
    Projection: Unary<X, Output = Y>,
    Pool: for<'b> DeviceModule<(&'b Y, C), Output = O>,
{
    type Output = O;

    fn forward(&self, (device, (hidden, context)): (&mut Device, (&X, C))) -> Self::Output {
        let logits = self.projection.forward((&mut *device, hidden));
        self.pool.forward((device, (&logits, context)))
    }
}

/// Selects hidden rows with a stored offset operand.
#[derive(DeviceSend)]
pub struct SelectRow<Selection, Offset> {
    pub select: Selection,
    pub offset: Offset,
}

impl<X, Y, S, O> Module<(&mut Device, &X)> for SelectRow<S, O>
where
    S: Binary<X, O, Output = Y>,
{
    type Output = Y;

    fn forward(&self, (device, input): (&mut Device, &X)) -> Y {
        self.select.forward((device, (input, &self.offset)))
    }
}
