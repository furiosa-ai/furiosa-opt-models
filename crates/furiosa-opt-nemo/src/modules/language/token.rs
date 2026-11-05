//! Bodies that carry the token IDs beside the hidden values, for layers that read them.

use crate::{Binary, DeviceModule, Module, Unary};
use furiosa_opt_std::prelude::{Device, DeviceSend};

/// Embeds tokens and retains their borrowed IDs alongside the body's invocation context.
#[derive(DeviceSend)]
pub struct TokenBackbone<Embedding, Body> {
    pub embedding: Embedding,
    pub body: Body,
}

impl<'t, T, C, X, Embedding, Body> Module<(&mut Device, (&'t T, C))>
    for TokenBackbone<Embedding, Body>
where
    Embedding: DeviceModule<&'t T, Output = X>,
    Body: DeviceModule<(X, (&'t T, C)), Output = X>,
{
    type Output = X;

    fn forward(&self, (device, (tokens, context)): (&mut Device, (&'t T, C))) -> Self::Output {
        let hidden = self.embedding.forward((&mut *device, tokens));
        self.body.forward((device, (hidden, (tokens, context))))
    }
}

impl<'t, T, I, C, X, Embedding, Body> Module<(&mut Device, (&'t T, I, C))>
    for TokenBackbone<Embedding, Body>
where
    Embedding: DeviceModule<(&'t T, I), Output = X>,
    Body: DeviceModule<(X, (&'t T, C)), Output = X>,
{
    type Output = X;

    fn forward(&self, (device, (tokens, media, context)): (&mut Device, (&'t T, I, C))) -> X {
        let hidden = self.embedding.forward((&mut *device, (tokens, media)));
        self.body.forward((device, (hidden, (tokens, context))))
    }
}

/// Runs a body carrying token IDs with hidden values and returns only the hidden output.
#[derive(DeviceSend)]
pub struct TokenBody<Body> {
    pub body: Body,
}

impl<'t, T, C, X, Body> Module<(&mut Device, (X, (&'t T, C)))> for TokenBody<Body>
where
    Body: DeviceModule<((X, &'t T), C), Output = (X, &'t T)>,
{
    type Output = X;

    fn forward(
        &self,
        (device, (hidden, (tokens, context))): (&mut Device, (X, (&'t T, C))),
    ) -> Self::Output {
        let (hidden, _) = self.body.forward((device, ((hidden, tokens), context)));
        hidden
    }
}

/// Adds projected token features before computation while preserving token IDs in the output.
#[derive(DeviceSend)]
pub struct TokenCondition<Embedding, Project, Compute, Add> {
    pub embedding: Embedding,
    pub project: Project,
    pub module: Compute,
    pub add: Add,
}

impl<'t, T, C, X, Y, Z, Embedding, Project, Compute, Add> Module<(&mut Device, (&(X, &'t T), C))>
    for TokenCondition<Embedding, Project, Compute, Add>
where
    Embedding: DeviceModule<&'t T, Output = Y>,
    Project: Unary<Y, Output = Z>,
    Add: Binary<X, Z, Output = X>,
    Compute: for<'b> DeviceModule<(&'b X, C), Output = X>,
{
    type Output = (X, &'t T);

    fn forward(
        &self,
        (device, ((hidden, tokens), context)): (&mut Device, (&(X, &'t T), C)),
    ) -> Self::Output {
        let embedding = self.embedding.forward((&mut *device, tokens));
        let embedding = self.project.forward((&mut *device, &embedding));
        let hidden = self.add.forward((&mut *device, (hidden, &embedding)));
        (self.module.forward((device, (&hidden, context))), tokens)
    }
}
