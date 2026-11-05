//! Language model structure: embeddings, context-bearing bodies, final normalization and heads.

mod backbone;
mod embedding;
mod head;
mod indexed;
mod token;

pub use backbone::{Backbone, Embedded};
pub use embedding::{ConditionedEmbedding, Embeddings, NgramEmbedding};
pub use head::{
    FeatureHead, MaskedHead, Pooler, Readout, RetrievalHead, SelectRow, SequenceHead, SparseHead,
};
pub use indexed::Indexed;
pub use token::{TokenBackbone, TokenBody, TokenCondition};

use crate::{Branch, DeviceModule, Interleave, Layers, Module, Unary};
use furiosa_opt_std::prelude::{Device, DeviceSend, unroll};

/// Reuses one body with independent contexts and a separate normalization per iteration.
#[derive(DeviceSend)]
pub struct Loop<Body, Norms> {
    pub body: Body,
    pub norms: Norms,
}

impl<X, C, Body, Norm, const N: usize> Module<(&mut Device, (X, &mut [C; N]))>
    for Loop<Body, [Norm; N]>
where
    Body: for<'c> DeviceModule<(X, &'c mut C), Output = X>,
    Norm: Unary<X, Output = X>,
{
    type Output = X;

    fn forward(&self, (device, (mut hidden, contexts)): (&mut Device, (X, &mut [C; N]))) -> X {
        if N > 0 {
            #[unroll]
            #[expect(
                clippy::needless_range_loop,
                reason = "device unrolling requires index ranges"
            )]
            for index in 0..N {
                hidden = self
                    .body
                    .forward((&mut *device, (hidden, &mut contexts[index])));
                hidden = self.norms[index].forward((&mut *device, &hidden));
            }
        }
        hidden
    }
}

/// Normalizes the body's final hidden values before any readout.
/// A struct, not `Then`: checkpoints name this norm `norm` beside the backbone, not inside it.
#[derive(Clone, DeviceSend)]
pub struct Normalized<Body, Norm> {
    pub body: Body,
    pub norm: Norm,
}

impl<I, X, Y, Body, Norm> Module<(&mut Device, I)> for Normalized<Body, Norm>
where
    Body: DeviceModule<I, Output = X>,
    Norm: Unary<X, Output = Y>,
{
    type Output = Y;

    fn forward(&self, (device, input): (&mut Device, I)) -> Self::Output {
        let hidden = self.body.forward((&mut *device, input));
        self.norm.forward((device, &hidden))
    }
}

/// Embedding and context-bearing body followed by final normalization and a language head.
pub type CausalLm<Embedding, Body, Norm, Head> =
    Readout<Normalized<Backbone<Embedding, Body>, Norm>, Head>;

/// Embeds the input, runs the body and returns its features with the head's output.
pub type Features<Embedding, Body, Head> = FeatureHead<Backbone<Embedding, Body>, Head>;

/// A causal language readout over a collection with one context per layer.
pub type Decoder<Embedding, Collection, Norm, Head> =
    CausalLm<Embedding, Layers<Collection>, Norm, Head>;

/// A causal language readout over attention and recurrent layers in a fixed shared order.
pub type Hybrid<Embedding, Attention, Recurrent, Norm, Head, const ORDER: &'static [Branch]> =
    CausalLm<Embedding, Interleave<Attention, Recurrent, ORDER>, Norm, Head>;

/// Embeds and runs layers before applying a head to their final hidden output.
pub type Encoder<Embedding, Collection, Head> =
    Readout<Backbone<Embedding, Layers<Collection>>, Head>;
