//! Input embeddings that add segment, position, condition or n-gram features to token embeddings.

use crate::Module;
use crate::load::Load;
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

impl<I, J, K, X, O, Token, Position, Segment, Add, Norm, Env: Copy> Module<(I, J, K), Env, &mut ()>
    for Embeddings<Token, Position, Segment, Add, Norm>
where
    Token: for<'s> Module<I, Env, &'s mut (), Output = X>,
    Position: for<'s> Module<J, Env, &'s mut (), Output = X>,
    Segment: for<'s> Module<K, Env, &'s mut (), Output = X>,
    Add: for<'a, 'b, 's> Module<(&'a X, &'b X), Env, &'s mut (), Output = X>,
    Norm: for<'x, 's> Module<&'x X, Env, &'s mut (), Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        (tokens, positions, segments): (I, J, K),
    ) -> O {
        let tokens = self.token.forward(&mut *device, env, &mut (), tokens);
        let positions = self.position.forward(&mut *device, env, &mut (), positions);
        let segments = self.segment.forward(&mut *device, env, &mut (), segments);
        let hidden = self
            .add
            .forward(&mut *device, env, &mut (), (&tokens, &segments));
        let hidden = self
            .add
            .forward(&mut *device, env, &mut (), (&hidden, &positions));
        self.norm.forward(device, env, &mut (), &hidden)
    }
}

/// Adds a separately embedded condition to the input embedding; its children read under their own
/// names, as a combinator's do.
#[derive(DeviceSend, Load)]
pub struct ConditionedEmbedding<Embedding, Condition, Add> {
    #[load(rename = "")]
    pub embedding: Embedding,
    #[load(rename = "")]
    pub condition: Condition,
    #[load(rename = "")]
    pub add: Add,
}

/// The state holds the embedding's slot, then the condition's.
impl<'s, I, J, X, Y, O, EmbeddingState, ConditionState, Embedding, Condition, Add, Env: Copy>
    Module<&(I, J), Env, &'s mut (EmbeddingState, ConditionState)>
    for ConditionedEmbedding<Embedding, Condition, Add>
where
    Embedding: for<'x> Module<&'x I, Env, &'s mut EmbeddingState, Output = X>,
    Condition: for<'x> Module<&'x J, Env, &'s mut ConditionState, Output = Y>,
    Add: for<'a, 'b, 'd> Module<(&'a X, &'b Y), Env, &'d mut (), Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (embedding, conditioning): &'s mut (EmbeddingState, ConditionState),
        (input, condition): &(I, J),
    ) -> O {
        let hidden = self.embedding.forward(&mut *device, env, embedding, input);
        let condition = self
            .condition
            .forward(&mut *device, env, conditioning, condition);
        self.add
            .forward(device, env, &mut (), (&hidden, &condition))
    }
}

impl<'s, I, J, X, Y, O, EmbeddingState, ConditionState, Embedding, Condition, Add, Env: Copy>
    Module<(I, J), Env, &'s mut (EmbeddingState, ConditionState)>
    for ConditionedEmbedding<Embedding, Condition, Add>
where
    Embedding: Module<I, Env, &'s mut EmbeddingState, Output = X>,
    Condition: Module<J, Env, &'s mut ConditionState, Output = Y>,
    Add: for<'a, 'b, 'd> Module<(&'a X, &'b Y), Env, &'d mut (), Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (embedding, conditioning): &'s mut (EmbeddingState, ConditionState),
        (input, context): (I, J),
    ) -> Self::Output {
        let hidden = self.embedding.forward(&mut *device, env, embedding, input);
        let condition = self
            .condition
            .forward(&mut *device, env, conditioning, context);
        self.add
            .forward(device, env, &mut (), (&hidden, &condition))
    }
}

/// Adds history-dependent n-gram features to the current token embeddings.
#[derive(DeviceSend)]
pub struct NgramEmbedding<Embedding, Ngram, Add> {
    pub embedding: Embedding,
    pub ngram: Ngram,
    pub add: Add,
}

/// The state holds the embedding's slot, then the n-gram's, which holds the token history.
impl<'s, 'x, T, X, Y, O, EmbeddingState, NgramState, Embedding, Ngram, Add, Env: Copy>
    Module<&'x T, Env, &'s mut (EmbeddingState, NgramState)>
    for NgramEmbedding<Embedding, Ngram, Add>
where
    Embedding: Module<&'x T, Env, &'s mut EmbeddingState, Output = X>,
    Ngram: Module<&'x T, Env, &'s mut NgramState, Output = Y>,
    Add: for<'a, 'b, 'd> Module<(&'a X, &'b Y), Env, &'d mut (), Output = O>,
{
    type Output = O;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (embedding, ngram): &'s mut (EmbeddingState, NgramState),
        tokens: &'x T,
    ) -> Self::Output {
        let hidden = self.embedding.forward(&mut *device, env, embedding, tokens);
        let ngram = self.ngram.forward(&mut *device, env, ngram, tokens);
        self.add.forward(device, env, &mut (), (&hidden, &ngram))
    }
}
