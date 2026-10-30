//! Language model structure: embeddings, context-bearing bodies, final normalization and heads.

pub use backbone::Backbone;
pub use embedding::{ConditionedEmbedding, Embeddings, NgramEmbedding};
pub use head::{
    Classifier, FeatureHead, Greedy, MaskedHead, Pooler, Readout, RetrievalHead, RowWise,
    SelectRow, Selected, SequenceHead, SparseHead,
};

use crate::load::Named;
use crate::tensor::{live::Live, parallel::Chip, scalar::Value};
use crate::{Branch, Collection, Interleave, Layers, Module, Then};
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

mod backbone;
mod embedding;
mod head;

/// Device operands of one packed prefix of at most `R` rows on the chips `Chips`: its tokens, its
/// live row count, and the cache operands that hold its logical history.
#[derive(DeviceSend)]
pub struct Input<R, Tokens, Cache, Chips: M = Chip> {
    pub(crate) tokens: Tokens,
    pub(crate) count: Value<Chips>,
    pub(crate) cache: Cache,
    pub(crate) rows: PhantomData<R>,
}

impl<R, Tokens, Cache, Chips: M> Input<R, Tokens, Cache, Chips> {
    pub const fn new(tokens: Tokens, count: Value<Chips>, cache: Cache) -> Self {
        Self {
            tokens,
            count,
            cache,
            rows: PhantomData,
        }
    }
}

/// What a call's leaves read: the context its preparation computed, its tokens and its live rows.
pub struct Call<'c, C, T, R> {
    pub(crate) context: &'c C,
    pub(crate) tokens: &'c T,
    pub(crate) live: Live<(), R>,
}

impl<C, T, R> Clone for Call<'_, C, T, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<C, T, R> Copy for Call<'_, C, T, R> {}

impl<C, T, R> From<Call<'_, C, T, R>> for Live<(), R> {
    fn from(call: Call<'_, C, T, R>) -> Self {
        call.live
    }
}

/// The iterations of a loop: one body for every iteration, each followed by its own norm.
#[derive(DeviceSend)]
pub struct Loop<Body, Norms> {
    pub body: Body,
    pub norms: Norms,
}

/// An iteration runs the body with its state slot, then the iteration's norm.
impl<X, S, Body, Norm, const N: usize, Env: Copy> Collection<X, Env, S> for Loop<Body, [Norm; N]>
where
    Body: Module<X, Env, S, Output = X>,
    Norm: for<'d, 'x> Module<&'x X, Env, &'d mut (), Output = X>,
{
    type Output = X;

    const COUNT: usize = N;

    fn apply(&self, device: &mut Device, env: Env, state: S, hidden: X, index: usize) -> X {
        let hidden = self.body.forward(&mut *device, env, state, hidden);
        self.norms[index].forward(device, env, &mut (), &hidden)
    }
}

/// Normalizes the body's final hidden values before any readout; the norm is its checkpoint's
/// `norm`.
pub type Normalized<Body, Norm> = Then<Body, Named<"norm", Norm>>;

impl<Body, Norm> Normalized<Body, Norm> {
    pub const fn new(body: Body, norm: Norm) -> Self {
        Then(body, Named(norm))
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
    CausalLm<Embedding, Interleave<(Attention, Recurrent), ORDER>, Norm, Head>;

/// Embeds and runs layers before applying a head to their final hidden output.
pub type Encoder<Embedding, Collection, Head> =
    Readout<Backbone<Embedding, Layers<Collection>>, Head>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Unrolled;
    use crate::tests::DEVICE;
    use furiosa_opt_std::prelude::Topology;

    struct Step;

    impl<Env> Module<i32, Env, &mut i32> for Step {
        type Output = i32;

        fn forward(&self, _: &mut Device, _: Env, state: &mut i32, hidden: i32) -> i32 {
            *state += 1;
            hidden + *state
        }
    }

    struct Scale(i32);

    impl<Env> Module<&i32, Env, &mut ()> for Scale {
        type Output = i32;

        fn forward(&self, _: &mut Device, _: Env, _: &mut (), hidden: &i32) -> i32 {
            hidden * self.0
        }
    }

    /// Each iteration runs the shared body with its own state slot, then that iteration's norm.
    #[test]
    fn normalizes_each_iteration() {
        let _lease = DEVICE.blocking_lock();
        let device = &mut Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let iterations = Layers {
            layers: Loop {
                body: Step,
                norms: [Scale(2), Scale(3)],
            },
            kind: Unrolled,
        };
        let mut state = [10, 20];
        assert_eq!(iterations.forward(device, (), &mut state, 1), 135);
        assert_eq!(state, [11, 21]);
    }
}
