//! Transformer blocks whose attention and feed-forward residuals the condition modulates.

use super::Modulated;
use crate::Module;
use crate::load::Load;
use furiosa_opt_std::prelude::{Device, DeviceSend};

/// Modulates pre-norm attention and feed-forward residuals from a shared condition; `norm` is the
/// attention's, `feedforward_norm` the feed-forward's.
#[derive(DeviceSend, Load)]
pub struct Adaptive<Modulation, Attention, FeedForward, Norm, Gate, FeedForwardNorm = Norm> {
    #[load(rename = "")]
    pub modulation: Modulation,
    #[load(rename = "")]
    pub attention: Attention,
    #[load(rename = "")]
    pub feedforward: FeedForward,
    #[load(rename = "")]
    pub norm: Norm,
    #[load(rename = "")]
    pub feedforward_norm: FeedForwardNorm,
    #[load(rename = "")]
    pub gate: Gate,
}

/// The env holds the condition the modulation reads, then the attention's context; the state
/// holds the modulation's, the attention's and the feed-forward's slots.
impl<
    's,
    'e,
    X,
    E,
    M,
    G,
    ModulationState,
    AttentionState,
    FeedForwardState,
    Modulation,
    Attention,
    FeedForward,
    Norm,
    Gate,
    FeedForwardNorm,
    Env: Copy,
> Module<X, (&'e E, Env), &'s mut (ModulationState, AttentionState, FeedForwardState)>
    for Adaptive<Modulation, Attention, FeedForward, Norm, Gate, FeedForwardNorm>
where
    Modulation: Module<&'e E, Env, &'s mut ModulationState, Output = ((M, G), (M, G))>,
    Norm: for<'a, 'b, 'd> Module<(&'a X, &'b M), Env, &'d mut (), Output = X>,
    FeedForwardNorm: for<'a, 'b, 'd> Module<(&'a X, &'b M), Env, &'d mut (), Output = X>,
    Attention: for<'a> Module<&'a X, Env, &'s mut AttentionState, Output = X>,
    FeedForward: for<'x> Module<&'x X, Env, &'s mut FeedForwardState, Output = X>,
    Gate: for<'a, 'b, 'c, 'd> Module<(&'a X, &'b X, &'c G), Env, &'d mut (), Output = X>,
{
    type Output = X;

    fn forward(
        &self,
        device: &mut Device,
        (condition, env): (&'e E, Env),
        (modulation, attention, feedforward): &'s mut (
            ModulationState,
            AttentionState,
            FeedForwardState,
        ),
        hidden: X,
    ) -> X {
        let ((attention_norm, attention_gate), (feedforward_norm, feedforward_gate)) = self
            .modulation
            .forward(&mut *device, env, modulation, condition);
        let hidden = Modulated(&self.norm, &self.attention, &self.gate).forward(
            &mut *device,
            env,
            attention,
            (&hidden, (&attention_norm, &attention_gate)),
        );
        Modulated(&self.feedforward_norm, &self.feedforward, &self.gate).forward(
            device,
            env,
            feedforward,
            (&hidden, (&feedforward_norm, &feedforward_gate)),
        )
    }
}

/// Conditions the attention input norm, then adds an unconditioned pre-norm feed-forward residual.
#[derive(DeviceSend)]
pub struct Conditioned<Norm, Attention, FeedForwardNorm, FeedForward, Add> {
    pub norm: Norm,
    pub attention: Attention,
    pub feedforward_norm: FeedForwardNorm,
    pub feedforward: FeedForward,
    pub add: Add,
}

impl<
    's,
    'e,
    X,
    E,
    N,
    A,
    F,
    Y,
    AttentionState,
    FeedForwardState,
    Norm,
    Attention,
    FeedForwardNorm,
    FeedForward,
    Add,
    Env: Copy,
> Module<X, (&'e E, Env), &'s mut (AttentionState, FeedForwardState)>
    for Conditioned<Norm, Attention, FeedForwardNorm, FeedForward, Add>
where
    Norm: for<'a, 'b, 'd> Module<(&'a X, &'b E), Env, &'d mut (), Output = N>,
    Attention: for<'a> Module<&'a N, Env, &'s mut AttentionState, Output = A>,
    Add: for<'a, 'b, 'd> Module<(&'a X, &'b A), Env, &'d mut (), Output = X>
        + for<'a, 'b, 'd> Module<(&'a X, &'b Y), Env, &'d mut (), Output = X>,
    FeedForwardNorm: for<'x, 'd> Module<&'x X, Env, &'d mut (), Output = F>,
    FeedForward: for<'x> Module<&'x F, Env, &'s mut FeedForwardState, Output = Y>,
{
    type Output = X;

    fn forward(
        &self,
        device: &mut Device,
        (condition, env): (&'e E, Env),
        (attention, feedforward): &'s mut (AttentionState, FeedForwardState),
        hidden: X,
    ) -> X {
        let normalized = self
            .norm
            .forward(&mut *device, env, &mut (), (&hidden, condition));
        let attention = self
            .attention
            .forward(&mut *device, env, attention, &normalized);
        let hidden = self
            .add
            .forward(&mut *device, env, &mut (), (&hidden, &attention));
        let normalized = self
            .feedforward_norm
            .forward(&mut *device, env, &mut (), &hidden);
        let feedforward = self
            .feedforward
            .forward(&mut *device, env, feedforward, &normalized);
        self.add
            .forward(device, env, &mut (), (&hidden, &feedforward))
    }
}

/// Inserts cross-attention between modulated residuals; the feed-forward branch gets the geometry.
#[derive(DeviceSend)]
pub struct Spatial<Modulation, Attention, CrossAttention, FeedForward, Norm, Gate, Add> {
    pub modulation: Modulation,
    pub attention: Attention,
    pub cross: CrossAttention,
    pub feedforward: FeedForward,
    pub norms: [Norm; 2],
    pub gate: Gate,
    pub add: Add,
}

/// The env holds the condition, then the cross-attention's context and the geometry the
/// feed-forward reads; the state holds the modulation's and each branch's slots.
impl<
    's,
    'e,
    X,
    E,
    S,
    M,
    G,
    ModulationState,
    AttentionState,
    CrossAttentionState,
    FeedForwardState,
    Modulation,
    Attention,
    CrossAttention,
    FeedForward,
    Norm,
    Gate,
    Add,
    Env: Copy,
>
    Module<
        X,
        (&'e E, (Env, S)),
        &'s mut (
            ModulationState,
            AttentionState,
            CrossAttentionState,
            FeedForwardState,
        ),
    > for Spatial<Modulation, Attention, CrossAttention, FeedForward, Norm, Gate, Add>
where
    Modulation: Module<&'e E, Env, &'s mut ModulationState, Output = ((M, G), (M, G))>,
    Norm: for<'a, 'b, 'd> Module<(&'a X, &'b M), Env, &'d mut (), Output = X>,
    Attention: for<'x> Module<&'x X, Env, &'s mut AttentionState, Output = X>,
    CrossAttention: for<'a> Module<&'a X, Env, &'s mut CrossAttentionState, Output = X>,
    FeedForward: for<'a, 'b> Module<(&'a X, &'b S), Env, &'s mut FeedForwardState, Output = X>,
    Gate: for<'a, 'b, 'c, 'd> Module<(&'a X, &'b X, &'c G), Env, &'d mut (), Output = X>,
    Add: for<'a, 'b, 'd> Module<(&'a X, &'b X), Env, &'d mut (), Output = X>,
{
    type Output = X;

    fn forward(
        &self,
        device: &mut Device,
        (condition, (env, shape)): (&'e E, (Env, S)),
        (modulation, attention, cross, feedforward): &'s mut (
            ModulationState,
            AttentionState,
            CrossAttentionState,
            FeedForwardState,
        ),
        hidden: X,
    ) -> X {
        let ((attention_norm, attention_gate), (feedforward_norm, feedforward_gate)) = self
            .modulation
            .forward(&mut *device, env, modulation, condition);
        let hidden = Modulated(&self.norms[0], &self.attention, &self.gate).forward(
            &mut *device,
            env,
            attention,
            (&hidden, (&attention_norm, &attention_gate)),
        );
        let attended = self.cross.forward(&mut *device, env, cross, &hidden);
        let hidden = self
            .add
            .forward(&mut *device, env, &mut (), (&hidden, &attended));
        let normalized =
            self.norms[1].forward(&mut *device, env, &mut (), (&hidden, &feedforward_norm));
        let fed = self
            .feedforward
            .forward(&mut *device, env, feedforward, (&normalized, &shape));
        self.gate
            .forward(device, env, &mut (), (&hidden, &fed, &feedforward_gate))
    }
}

/// A modulated gated residual whose branch joins attention and feed-forward outputs, projected.
#[derive(DeviceSend)]
pub struct Single<Modulation, Norm, Attention, FeedForward, Join, Projection, Gate> {
    pub modulation: Modulation,
    pub norm: Norm,
    pub attention: Attention,
    pub feedforward: FeedForward,
    pub join: Join,
    pub projection: Projection,
    pub gate: Gate,
}

/// The env holds the condition, then the attention's context; the state holds the modulation's
/// and the branches' slots.
impl<
    's,
    'e,
    X,
    E,
    M,
    G,
    A,
    F,
    J,
    ModulationState,
    AttentionState,
    FeedForwardState,
    Modulation,
    Norm,
    Attention,
    FeedForward,
    Join,
    Projection,
    Gate,
    Env: Copy,
> Module<X, (&'e E, Env), &'s mut (ModulationState, AttentionState, FeedForwardState)>
    for Single<Modulation, Norm, Attention, FeedForward, Join, Projection, Gate>
where
    Modulation: Module<&'e E, Env, &'s mut ModulationState, Output = (M, G)>,
    Norm: for<'a, 'b, 'd> Module<(&'a X, &'b M), Env, &'d mut (), Output = X>,
    Attention: for<'a> Module<&'a X, Env, &'s mut AttentionState, Output = A>,
    FeedForward: for<'x> Module<&'x X, Env, &'s mut FeedForwardState, Output = F>,
    Join: for<'a, 'b, 'd> Module<(&'a A, &'b F), Env, &'d mut (), Output = J>,
    Projection: for<'x, 'd> Module<&'x J, Env, &'d mut (), Output = X>,
    Gate: for<'a, 'b, 'c, 'd> Module<(&'a X, &'b X, &'c G), Env, &'d mut (), Output = X>,
{
    type Output = X;

    fn forward(
        &self,
        device: &mut Device,
        (condition, env): (&'e E, Env),
        (modulation, attention, feedforward): &'s mut (
            ModulationState,
            AttentionState,
            FeedForwardState,
        ),
        hidden: X,
    ) -> X {
        let (modulation, gate) = self
            .modulation
            .forward(&mut *device, env, modulation, condition);
        let normalized = self
            .norm
            .forward(&mut *device, env, &mut (), (&hidden, &modulation));
        let attended = self
            .attention
            .forward(&mut *device, env, attention, &normalized);
        let fed = self
            .feedforward
            .forward(&mut *device, env, feedforward, &normalized);
        let joined = self
            .join
            .forward(&mut *device, env, &mut (), (&attended, &fed));
        let projected = self.projection.forward(&mut *device, env, &mut (), &joined);
        self.gate
            .forward(device, env, &mut (), (&hidden, &projected, &gate))
    }
}

/// Inserts pre-norm cross-attention between modulated attention and feed-forward residuals.
#[derive(DeviceSend)]
pub struct Cross<Modulation, Attention, CrossAttention, FeedForward, Norm, CrossNorm, Gate, Add> {
    pub modulation: Modulation,
    pub attention: Attention,
    pub cross: CrossAttention,
    pub feedforward: FeedForward,
    pub norms: [Norm; 2],
    pub cross_norm: CrossNorm,
    pub gate: Gate,
    pub add: Add,
}

/// The env holds the condition, then the attention's context and the cross-attention's; the state
/// holds the modulation's and each branch's slots.
impl<
    's,
    'e,
    X,
    E,
    M,
    G,
    AttentionEnv: Copy,
    CrossAttentionEnv: Copy,
    ModulationState,
    AttentionState,
    CrossAttentionState,
    FeedForwardState,
    Modulation,
    Attention,
    CrossAttention,
    FeedForward,
    Norm,
    CrossNorm,
    Gate,
    Add,
>
    Module<
        X,
        (&'e E, (AttentionEnv, CrossAttentionEnv)),
        &'s mut (
            ModulationState,
            AttentionState,
            CrossAttentionState,
            FeedForwardState,
        ),
    > for Cross<Modulation, Attention, CrossAttention, FeedForward, Norm, CrossNorm, Gate, Add>
where
    Modulation: Module<&'e E, AttentionEnv, &'s mut ModulationState, Output = ((M, G), (M, G))>,
    Norm: for<'a, 'b, 'd> Module<(&'a X, &'b M), AttentionEnv, &'d mut (), Output = X>,
    CrossNorm: for<'x, 'd> Module<&'x X, AttentionEnv, &'d mut (), Output = X>,
    Attention: for<'a> Module<&'a X, AttentionEnv, &'s mut AttentionState, Output = X>,
    CrossAttention:
        for<'a> Module<&'a X, CrossAttentionEnv, &'s mut CrossAttentionState, Output = X>,
    FeedForward: for<'x> Module<&'x X, AttentionEnv, &'s mut FeedForwardState, Output = X>,
    Gate: for<'a, 'b, 'c, 'd> Module<(&'a X, &'b X, &'c G), AttentionEnv, &'d mut (), Output = X>,
    Add: for<'a, 'b, 'd> Module<(&'a X, &'b X), AttentionEnv, &'d mut (), Output = X>,
{
    type Output = X;

    fn forward(
        &self,
        device: &mut Device,
        (condition, (env, encoder)): (&'e E, (AttentionEnv, CrossAttentionEnv)),
        (modulation, attention, cross, feedforward): &'s mut (
            ModulationState,
            AttentionState,
            CrossAttentionState,
            FeedForwardState,
        ),
        hidden: X,
    ) -> X {
        let ((attention_norm, attention_gate), (feedforward_norm, feedforward_gate)) = self
            .modulation
            .forward(&mut *device, env, modulation, condition);
        let hidden = Modulated(&self.norms[0], &self.attention, &self.gate).forward(
            &mut *device,
            env,
            attention,
            (&hidden, (&attention_norm, &attention_gate)),
        );
        let normalized = self.cross_norm.forward(&mut *device, env, &mut (), &hidden);
        let attended = self
            .cross
            .forward(&mut *device, encoder, cross, &normalized);
        let hidden = self
            .add
            .forward(&mut *device, env, &mut (), (&hidden, &attended));
        Modulated(&self.norms[1], &self.feedforward, &self.gate).forward(
            device,
            env,
            feedforward,
            (&hidden, (&feedforward_norm, &feedforward_gate)),
        )
    }
}
