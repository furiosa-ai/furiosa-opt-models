//! Input embeddings that add segment, position, condition or n-gram features to token embeddings.

use crate::{Binary, DeviceModule, Module, Unary};
use furiosa_opt_std::prelude::{Device, DeviceSend};

/// Sums token, segment and position embeddings in a shared hidden space, then normalizes.
#[derive(DeviceSend)]
pub struct Embeddings<Token, Position, Segment, Add, Norm> {
    pub token: Token,
    pub position: Position,
    pub segment: Segment,
    pub add: Add,
    pub norm: Norm,
}

impl<I, J, K, X, O, T, P, S, A, N> Module<(&mut Device, (I, J, K))> for Embeddings<T, P, S, A, N>
where
    T: DeviceModule<I, Output = X>,
    P: DeviceModule<J, Output = X>,
    S: DeviceModule<K, Output = X>,
    A: Binary<X, X, Output = X>,
    N: Unary<X, Output = O>,
{
    type Output = O;

    fn forward(&self, (device, (tokens, positions, segments)): (&mut Device, (I, J, K))) -> O {
        let tokens = self.token.forward((&mut *device, tokens));
        let positions = self.position.forward((&mut *device, positions));
        let segments = self.segment.forward((&mut *device, segments));
        let hidden = self.add.forward((&mut *device, (&tokens, &segments)));
        let hidden = self.add.forward((&mut *device, (&hidden, &positions)));
        self.norm.forward((device, &hidden))
    }
}

/// Adds a separately embedded condition to the input embedding.
#[derive(DeviceSend)]
pub struct ConditionedEmbedding<Embedding, Condition, Add> {
    pub embedding: Embedding,
    pub condition: Condition,
    pub add: Add,
}

impl<I, J, X, Y, O, Embedding, Condition, Add> Module<(&mut Device, &(I, J))>
    for ConditionedEmbedding<Embedding, Condition, Add>
where
    Embedding: Unary<I, Output = X>,
    Condition: Unary<J, Output = Y>,
    Add: Binary<X, Y, Output = O>,
{
    type Output = O;

    fn forward(&self, (device, (input, condition)): (&mut Device, &(I, J))) -> O {
        let hidden = self.embedding.forward((&mut *device, input));
        let condition = self.condition.forward((&mut *device, condition));
        self.add.forward((device, (&hidden, &condition)))
    }
}

impl<I, J, X, Y, O, Embedding, Condition, Add> Module<(&mut Device, (I, J))>
    for ConditionedEmbedding<Embedding, Condition, Add>
where
    Embedding: DeviceModule<I, Output = X>,
    Condition: DeviceModule<J, Output = Y>,
    Add: Binary<X, Y, Output = O>,
{
    type Output = O;

    fn forward(&self, (device, (input, context)): (&mut Device, (I, J))) -> Self::Output {
        let hidden = self.embedding.forward((&mut *device, input));
        let condition = self.condition.forward((&mut *device, context));
        self.add.forward((device, (&hidden, &condition)))
    }
}

/// Adds history-dependent n-gram features to the current token embeddings.
#[derive(DeviceSend)]
pub struct NgramEmbedding<Embedding, Ngram, Add> {
    pub embedding: Embedding,
    pub ngram: Ngram,
    pub add: Add,
}

impl<'t, T, C, X, Y, O, Embedding, Ngram, Add> Module<(&mut Device, (&'t T, C))>
    for NgramEmbedding<Embedding, Ngram, Add>
where
    Embedding: DeviceModule<&'t T, Output = X>,
    Ngram: DeviceModule<(&'t T, C), Output = Y>,
    Add: Binary<X, Y, Output = O>,
{
    type Output = O;

    fn forward(&self, (device, (tokens, history)): (&mut Device, (&'t T, C))) -> Self::Output {
        let hidden = self.embedding.forward((&mut *device, tokens));
        let ngram = self.ngram.forward((&mut *device, (tokens, history)));
        self.add.forward((device, (&hidden, &ngram)))
    }
}
