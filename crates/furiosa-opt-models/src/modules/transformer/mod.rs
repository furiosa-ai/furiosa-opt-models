//! Transformer layers distinguished by norm placement around attention and feed-forward branches.

pub use encoder::Encoder;

use super::residual::{Residual, Sum};
use crate::load::Named;
use crate::{Fork, Module, Sequential, Then};
use furiosa_opt_std::prelude::{Device, DeviceSend};

mod encoder;

/// Adds each branch to the residual after normalizing that branch's input.
pub type PreNorm<Attention, FeedForward, Norm, Add> = Sequential<
    Residual<Named<"input_layernorm", Norm>, Named<"self_attn", Attention>, Add>,
    Residual<Named<"post_attention_layernorm", Norm>, Named<"mlp", FeedForward>, Add>,
>;

impl<Attention, FeedForward, Norm, Add> PreNorm<Attention, FeedForward, Norm, Add> {
    /// `norms` and `sums` hold the attention's, then the feed-forward's.
    pub fn new(
        attention: Attention,
        feedforward: FeedForward,
        norms: [Norm; 2],
        sums: [Add; 2],
    ) -> Self {
        let ([a, f], [sa, sf]) = (norms, sums);
        Sequential(
            Residual(Named(a), Named(attention), sa),
            Residual(Named(f), Named(feedforward), sf),
        )
    }
}

/// Mask-conditioned pre-norm residuals whose attention does not read the mask.
pub type RoutedNorm<AttentionNorm, Attention, FeedForwardNorm, FeedForward, Add> =
    Routed<AttentionNorm, Unmasked<Attention>, FeedForwardNorm, FeedForward, Add>;

/// Runs attention without the routing mask the layer reads.
#[derive(DeviceSend)]
pub struct Unmasked<Attention>(pub Attention);

/// A composition, not an op.
impl<X, Attention> !crate::Op<&X> for Unmasked<Attention> {}

impl<'e, X, M, E, S, O, Attention> Module<&X, (&'e M, E), S> for Unmasked<Attention>
where
    Attention: for<'x> Module<&'x X, E, S, Output = O>,
{
    type Output = O;

    fn forward(&self, device: &mut Device, (_, env): (&'e M, E), state: S, hidden: &X) -> O {
        self.0.forward(device, env, state, hidden)
    }
}

/// Pre-norm residuals routed by the mask beside the attention's env; each norm leads its branch, as
/// a routed norm is a composite, and each child reads under the names its family gives it.
pub type Routed<AttentionNorm, Attention, FeedForwardNorm, FeedForward, Add> = Sequential<
    Residual<(), Then<AttentionNorm, Attention>, Add>,
    Residual<(), Then<FeedForwardNorm, FeedForward>, Add>,
>;

/// Splits the rows by the routing mask the env holds, runs one child on each part and merges the
/// parts back in row order; a side whose rows are absent runs on an empty part.
pub type Partition<Split, Left, Right, Merge> = Sequential<Split, Sum<(Left, Right), Merge>>;

/// Normalizes each branch's output before adding it to the residual.
pub type BranchNorm<Attention, FeedForward, Norm, Add> = Sequential<
    Residual<(), Then<Named<"self_attn", Attention>, Named<"post_attention_layernorm", Norm>>, Add>,
    Residual<(), Then<Named<"mlp", FeedForward>, Named<"post_feedforward_layernorm", Norm>>, Add>,
>;

impl<Attention, FeedForward, Norm, Add> BranchNorm<Attention, FeedForward, Norm, Add> {
    /// `norms` and `sums` hold the attention's, then the feed-forward's.
    pub fn new(
        attention: Attention,
        feedforward: FeedForward,
        norms: [Norm; 2],
        sums: [Add; 2],
    ) -> Self {
        let ([a, f], [sa, sf]) = (norms, sums);
        Sequential(
            Residual((), Then(Named(attention), Named(a)), sa),
            Residual((), Then(Named(feedforward), Named(f)), sf),
        )
    }
}

/// Normalizes before and after each branch, then adds the branch output to the residual.
pub type SandwichNorm<Attention, FeedForward, Norm, Add> = Sequential<
    Residual<
        Named<"input_layernorm", Norm>,
        Then<Named<"self_attn", Attention>, Named<"post_attention_layernorm", Norm>>,
        Add,
    >,
    Residual<
        Named<"pre_feedforward_layernorm", Norm>,
        Then<Named<"mlp", FeedForward>, Named<"post_feedforward_layernorm", Norm>>,
        Add,
    >,
>;

impl<Attention, FeedForward, Norm, Add> SandwichNorm<Attention, FeedForward, Norm, Add> {
    /// `norms` holds each branch's norms before and after it, and `sums` its sum; the attention's
    /// come first.
    pub fn new(
        attention: Attention,
        feedforward: FeedForward,
        norms: [[Norm; 2]; 2],
        sums: [Add; 2],
    ) -> Self {
        let ([[a0, a1], [f0, f1]], [sa, sf]) = (norms, sums);
        Sequential(
            Residual(Named(a0), Then(Named(attention), Named(a1)), sa),
            Residual(Named(f0), Then(Named(feedforward), Named(f1)), sf),
        )
    }
}

/// Normalizes each residual sum after attention and after feed-forward. Each child reads under the
/// names its family gives it, as an encoder's checkpoint nests a sum's norm in its branch.
pub type PostNorm<Attention, FeedForward, AttentionNorm, FeedForwardNorm, Add> = Sequential<
    Then<Residual<(), Attention, Add>, AttentionNorm>,
    Then<Residual<(), FeedForward, Add>, FeedForwardNorm>,
>;

/// Computes attention and feed-forward from separately normalized copies of the same input.
pub type ParallelResidual<Attention, FeedForward, AttentionNorm, FeedForwardNorm, Add> = Residual<
    (),
    Sum<
        Fork<(
            Then<AttentionNorm, Attention>,
            Then<FeedForwardNorm, FeedForward>,
        )>,
        Add,
    >,
    Add,
>;

/// Applies pre-norm residual steps for self-attention, encoder cross-attention and feed-forward.
#[derive(DeviceSend)]
pub struct CrossLayer<Attention, Cross, FeedForward, Norm, Add> {
    pub attention: Attention,
    pub cross: Cross,
    pub feedforward: FeedForward,
    /// Input norms for self-attention, cross-attention and feed-forward, in that order.
    pub norms: [Norm; 3],
    pub add: Add,
}

/// The env holds the self-attention's env, then the cross-attention's; each branch has its own
/// state slot.
impl<
    's,
    X,
    AttentionEnv: Copy,
    CrossEnv: Copy,
    AttentionState,
    CrossState,
    FeedForwardState,
    Attention,
    Cross,
    FeedForward,
    Norm,
    Add,
> Module<X, (AttentionEnv, CrossEnv), &'s mut (AttentionState, CrossState, FeedForwardState)>
    for CrossLayer<Attention, Cross, FeedForward, Norm, Add>
where
    Attention: for<'x> Module<&'x X, AttentionEnv, &'s mut AttentionState, Output = X>,
    Cross: for<'x> Module<&'x X, CrossEnv, &'s mut CrossState, Output = X>,
    FeedForward: for<'x> Module<&'x X, AttentionEnv, &'s mut FeedForwardState, Output = X>,
    Norm: for<'x, 'd> Module<&'x X, AttentionEnv, &'d mut (), Output = X>
        + for<'x, 'd> Module<&'x X, CrossEnv, &'d mut (), Output = X>,
    Add: for<'a, 'b, 'd> Module<(&'a X, &'b X), AttentionEnv, &'d mut (), Output = X>
        + for<'a, 'b, 'd> Module<(&'a X, &'b X), CrossEnv, &'d mut (), Output = X>,
{
    type Output = X;

    fn forward(
        &self,
        device: &mut Device,
        (attention, cross): (AttentionEnv, CrossEnv),
        (first, second, third): &'s mut (AttentionState, CrossState, FeedForwardState),
        hidden: X,
    ) -> X {
        let hidden = Residual(&self.norms[0], &self.attention, &self.add).forward(
            &mut *device,
            attention,
            first,
            hidden,
        );
        let hidden = Residual(&self.norms[1], &self.cross, &self.add).forward(
            &mut *device,
            cross,
            second,
            hidden,
        );
        Residual(&self.norms[2], &self.feedforward, &self.add)
            .forward(device, attention, third, hidden)
    }
}

/// Self-attention and cross-attention write and read one shared storage, in that order.
impl<
    's,
    X,
    AttentionEnv: Copy,
    CrossEnv: Copy,
    S,
    FeedForwardState,
    Attention,
    Cross,
    FeedForward,
    Norm,
    Add,
> Module<X, (AttentionEnv, CrossEnv), &'s mut (S, FeedForwardState)>
    for CrossLayer<Attention, Cross, FeedForward, Norm, Add>
where
    Attention: for<'x, 't> Module<&'x X, AttentionEnv, &'t mut S, Output = X>,
    Cross: for<'x, 't> Module<&'x X, CrossEnv, &'t mut S, Output = X>,
    FeedForward: for<'x> Module<&'x X, AttentionEnv, &'s mut FeedForwardState, Output = X>,
    Norm: for<'x, 'd> Module<&'x X, AttentionEnv, &'d mut (), Output = X>
        + for<'x, 'd> Module<&'x X, CrossEnv, &'d mut (), Output = X>,
    Add: for<'a, 'b, 'd> Module<(&'a X, &'b X), AttentionEnv, &'d mut (), Output = X>
        + for<'a, 'b, 'd> Module<(&'a X, &'b X), CrossEnv, &'d mut (), Output = X>,
{
    type Output = X;

    fn forward(
        &self,
        device: &mut Device,
        (attention, cross): (AttentionEnv, CrossEnv),
        (storage, feedforward): &'s mut (S, FeedForwardState),
        hidden: X,
    ) -> X {
        let hidden = Residual(&self.norms[0], &self.attention, &self.add).forward(
            &mut *device,
            attention,
            &mut *storage,
            hidden,
        );
        let hidden = Residual(&self.norms[1], &self.cross, &self.add).forward(
            &mut *device,
            cross,
            &mut *storage,
            hidden,
        );
        Residual(&self.norms[2], &self.feedforward, &self.add).forward(
            device,
            attention,
            feedforward,
            hidden,
        )
    }
}
