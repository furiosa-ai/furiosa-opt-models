//! Self-conditioning: a previous prediction, when present, feeds back into the embedded input.

use crate::Module;
use furiosa_opt_std::prelude::{Device, DeviceSend};

/// Adds a signal from the previous prediction when present; leaves unconditioned input unchanged.
#[derive(DeviceSend)]
pub struct SelfConditioning<Norm, Signal, Add, Projection> {
    pub norm: Norm,
    pub signal: Signal,
    pub add: Add,
    pub output: Projection,
}

/// The state is the signal's slot.
impl<'p, X, Y, Z, S, Norm, Signal, Add, Projection, Env: Copy> Module<(X, Option<&'p X>), Env, S>
    for SelfConditioning<Norm, Signal, Add, Projection>
where
    Norm: for<'x, 's> Module<&'x X, Env, &'s mut (), Output = Y>,
    Signal: for<'x> Module<&'x Y, Env, S, Output = Z>,
    Add: for<'a, 'b, 's> Module<(&'a X, &'b Z), Env, &'s mut (), Output = X>,
    Projection: for<'x, 's> Module<&'x X, Env, &'s mut (), Output = X>,
{
    type Output = X;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        state: S,
        (hidden, previous): (X, Option<&'p X>),
    ) -> Self::Output {
        match previous {
            None => hidden,
            Some(previous) => {
                let previous = self.norm.forward(&mut *device, env, &mut (), previous);
                let signal = self.signal.forward(&mut *device, env, state, &previous);
                let hidden = self
                    .add
                    .forward(&mut *device, env, &mut (), (&hidden, &signal));
                self.output.forward(device, env, &mut (), &hidden)
            }
        }
    }
}

/// Embeds the input and supplies an optional previous prediction to self-conditioning.
#[derive(DeviceSend)]
pub struct ConditionedEmbedding<Embedding, Conditioning> {
    pub embedding: Embedding,
    pub conditioning: Conditioning,
}

/// The state holds the embedding's slot, then the self-conditioning's.
impl<'s, 'x, I, X, EmbeddingState, ConditioningState, Embedding, Conditioning, Env: Copy>
    Module<(I, Option<&'x X>), Env, &'s mut (EmbeddingState, ConditioningState)>
    for ConditionedEmbedding<Embedding, Conditioning>
where
    Embedding: Module<I, Env, &'s mut EmbeddingState, Output = X>,
    Conditioning: Module<(X, Option<&'x X>), Env, &'s mut ConditioningState, Output = X>,
{
    type Output = X;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (embedding, conditioning): &'s mut (EmbeddingState, ConditioningState),
        (input, previous): (I, Option<&'x X>),
    ) -> Self::Output {
        let hidden = self.embedding.forward(&mut *device, env, embedding, input);
        self.conditioning
            .forward(device, env, conditioning, (hidden, previous))
    }
}
