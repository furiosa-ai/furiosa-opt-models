//! MiniMax H3's 36-layer video decoder over one 7 × 16 × 16 latent tile per chip.
//!
//! Weights load relative to the physical video component supplied by the parent. This family
//! consumes `post_quant_conv.*` and the original decoder's weights. The unused
//! `decoder.mask_token` is accounted for only at the model owner, preserving the original
//! all-latents-visible decode. Unused encoders and `quant_conv` belong to the parent pipeline.
//! Rotation and key visibility belong to the call's [`Context`], separately from the weight tree.
//! The residual stream stays FP32; normalization and branch outputs round to BF16 before each
//! learned FP32 scaled add. Output rows retain the registers and padding: only the first
//! [`TOKENS`] rows are pixel patches. Tile extraction, stitching and blending are caller work.

use super::embeddings::Chips;
use crate::load::{Load, Named};
use crate::modules::{attention, residual};
use crate::ops::attention::single;
use crate::ops::normalization::RmsNorm;
use crate::ops::pointwise::SwiGlu;
use crate::ops::swiglu::GateUp;
use crate::tensor::form::Weight;
use crate::tensor::memory::residency::Residency;
use crate::tensor::memory::{Hbm, Memory};
use crate::{Layers, Module, Rolled, Sequential, Then};
use furiosa_opt_std::prelude::*;

pub use input::{Hidden, Parts};
pub use rotation::Context;

pub mod input;
pub mod parameters;
pub mod qkv;
pub mod registers;
pub mod rotation;
pub mod tiling;

/// Input embedding, rolled transformer layers, final normalization and FP32 patch readout.
/// `X` chooses retained storage; `&X` borrows those same leaves for execution.
pub type Decoder<C = Chips, X = Hbm> =
    Model<Embedding<C, X>, Layers<[Layer<C, X>; DEPTH], Rolled>, FinalNorm<C, X>, Head<C, X>>;

/// The video weight owner. Request rotation and visibility are supplied as the environment.
#[derive(DeviceSend, Load, Residency)]
#[load(ignore = "decoder.mask_token")]
#[residency(children(Embedding, Body, Norm, Head))]
pub struct Model<Embedding, Body, Norm, Head> {
    #[load(rename = "")]
    pub embedding: Embedding,
    #[load(rename = "decoder.transformer_blocks")]
    pub body: Body,
    #[load(rename = "decoder.norm_out")]
    pub norm: Norm,
    #[load(rename = "decoder.proj_out")]
    pub head: Head,
}

impl<I, A, B, N, H> !crate::Op<I> for Model<A, B, N, H> {}

impl<'x, 'e, C: M, E, B, N, H> Module<&'x Latent<C>, &'e Context<C>, &mut ()> for Model<E, B, N, H>
where
    E: for<'s> Module<&'x Latent<C>, &'e Context<C>, &'s mut ((), ()), Output = Residual<C>>,
    B: for<'s> Module<Residual<C>, &'e Context<C>, &'s mut [(); DEPTH], Output = Residual<C>>,
    N: for<'s, 'r> Module<&'r Residual<C>, &'e Context<C>, &'s mut (), Output = Normalized<C>>,
    H: for<'s, 'r> Module<&'r Normalized<C>, &'e Context<C>, &'s mut (), Output = Pixels<C>>,
{
    type Output = Pixels<C>;

    fn forward(
        &self,
        device: &mut Device,
        context: &'e Context<C>,
        _: &mut (),
        input: &'x Latent<C>,
    ) -> Self::Output {
        let hidden = self
            .embedding
            .forward(&mut *device, context, &mut ((), ()), input);
        let hidden = self
            .body
            .forward(&mut *device, context, &mut [(); DEPTH], hidden);
        let hidden = self.norm.forward(&mut *device, context, &mut (), &hidden);
        self.head.forward(device, context, &mut (), &hidden)
    }
}

/// Fold the input projection, then overwrite four learned registers and the zero class row.
pub type Embedding<C = Chips, X = Hbm> = Sequential<
    input::Input<C, X>,
    Named<
        "decoder.register_tokens",
        registers::Registers<C, <X as Memory>::Tensor<f32, C, registers::Layout>>,
    >,
>;

/// One FP32 pre-norm transformer layer, with independent attention and MLP scales.
pub type Layer<C = Chips, X = Hbm> = Block<Norm<C, X>, Attention<C, X>, Mlp<C, X>, Add<C, X>>;

/// Checkpoint names of one layer; the two shared residual compositions define its execution.
#[derive(DeviceSend, Load, Residency)]
#[residency(children(Norm, Attention, FeedForward, Add))]
pub struct Block<Norm, Attention, FeedForward, Add> {
    pub norm1: Norm,
    pub attn: Attention,
    pub scale1: Add,
    pub norm2: Norm,
    pub ff: FeedForward,
    pub scale2: Add,
}

impl<I, N, A, F, S> !crate::Op<I> for Block<N, A, F, S> {}

impl<X, Y, Z, N, A, F, S, E: Copy> Module<X, E, &mut ()> for Block<N, A, F, S>
where
    N: for<'s, 'x> Module<&'x X, E, &'s mut (), Output = Y>,
    A: for<'s, 'x> Module<&'x Y, E, &'s mut (), Output = Z>,
    F: for<'s, 'x> Module<&'x Y, E, &'s mut (), Output = Z>,
    S: for<'s, 'x, 'z> Module<(&'x X, &'z Z), E, &'s mut (), Output = X>,
{
    type Output = X;

    fn forward(&self, device: &mut Device, env: E, _: &mut (), input: X) -> X {
        let hidden = residual::Residual(&self.norm1, &self.attn, &self.scale1).forward(
            &mut *device,
            env,
            &mut (),
            input,
        );
        residual::Residual(&self.norm2, &self.ff, &self.scale2).forward(
            device,
            env,
            &mut (),
            hidden,
        )
    }
}

/// Per-head projections, unit-gain Q/K normalization, one-span attention and output projection.
pub type Attention<C = Chips, X = Hbm> = Attend<
    attention::NormalizedQkv<
        qkv::Projection<C, X>,
        parameters::HeadNorm<C, X>,
        parameters::HeadNorm<C, X>,
    >,
    Linear<AttentionWidth, Hidden, C, X>,
>;

/// The normalized projection reads `to_qkv.*`; the output reads `to_out.*`.
#[derive(DeviceSend, Load, Residency)]
#[residency(children(Qkv, Output))]
pub struct Attend<Qkv, Output> {
    #[load(rename = "to_qkv")]
    pub qkv: Qkv,
    #[load(rename = "to_out")]
    pub output: Output,
}

impl<I, Q, O> !crate::Op<I> for Attend<Q, O> {}

impl<'x, 'e, C: M, Q, O> Module<&'x Normalized<C>, &'e Context<C>, &mut ()> for Attend<Q, O>
where
    Q: for<'s> Module<
            &'x Normalized<C>,
            &'e Context<C>,
            &'s mut ((), (), ()),
            Output = (Major<C>, Major<C>, Major<C>),
        >,
    O: for<'s, 'y> Module<&'y Major<C>, &'e Context<C>, &'s mut (), Output = Normalized<C>>,
{
    type Output = Normalized<C>;

    fn forward(
        &self,
        device: &mut Device,
        env: &'e Context<C>,
        _: &mut (),
        input: &'x Normalized<C>,
    ) -> Self::Output {
        let (q, k, v) = self
            .qkv
            .forward(&mut *device, env, &mut ((), (), ()), input);
        let attend: crate::ops::attention::Attention<
            core::marker::PhantomData<Heads>,
            crate::tensor::scalar::F32<SCALE>,
        > = Default::default();
        let attended = attend.forward(
            &mut *device,
            env,
            &mut (),
            (&q, &k, &v, (&env.rotation, &env.visible)),
        );
        self.output.forward(device, env, &mut (), &attended)
    }
}

/// Packed gate/up projection, BF16 SiLU then product, and the BF16 down projection.
pub type Mlp<C = Chips, X = Hbm> =
    FeedForward<Linear<Hidden, GateUp<Intermediate>, C, X>, Linear<Intermediate, Hidden, C, X>>;

/// The checkpoint's `ff.w1` and `ff.w2` weights, composed through the public SwiGLU leaf.
#[derive(DeviceSend, Load, Residency)]
#[residency(children(Up, Down))]
pub struct FeedForward<Up, Down> {
    pub w1: Up,
    pub w2: Down,
}

impl<I, U, D> !crate::Op<I> for FeedForward<U, D> {}

impl<X, Y, Z, O, U, D, E: Copy> Module<&X, E, &mut ()> for FeedForward<U, D>
where
    U: for<'s, 'x> Module<&'x X, E, &'s mut (), Output = Y>,
    SwiGlu: for<'s, 'x> Module<&'x Y, E, &'s mut (), Output = Z>,
    D: for<'s, 'x> Module<&'x Z, E, &'s mut (), Output = O>,
{
    type Output = O;

    fn forward(&self, device: &mut Device, env: E, _: &mut (), input: &X) -> O {
        Then(&self.w1, Then(SwiGlu, &self.w2)).forward(device, env, &mut ((), ((), ())), input)
    }
}

/// Ordinary BF16 coefficients and a BF16-rounded FP32 bias, retaining generic storage.
pub type Linear<In, Out, C = Chips, X = Hbm> = parameters::Dense<
    In,
    Out,
    C,
    <X as Memory>::Tensor<bf16, C, m![Out, In]>,
    <X as Memory>::Tensor<f32, C, m![Out]>,
>;

/// RMS normalization into BF16, with the reference's `1e-5` stabilizer.
pub type Norm<C = Chips, X = Hbm> =
    RmsNorm<Hidden, EPSILON, C, Weight<m![Hidden], C, <X as Memory>::Tensor<bf16, C, m![Hidden]>>>;

/// BF16-rounded learned FP32 gate applied to the branch before the FP32 residual add.
pub type Add<C = Chips, X = Hbm> =
    parameters::Scale<C, <X as Memory>::Tensor<f32, C, parameters::Gate>>;

/// Final BF16 LayerNorm parameters and an FP32 `1e-5` epsilon on every chip.
pub type FinalNorm<C = Chips, X = Hbm> = parameters::FinalNorm<
    C,
    <X as Memory>::Tensor<bf16, C, m![Hidden]>,
    <X as Memory>::Tensor<f32, C, m![1]>,
>;

/// BF16 coefficients and rounded FP32 bias, with FP32 sums returned by public `Unrounded`.
pub type Head<C = Chips, X = Hbm> = parameters::Readout<
    Hidden,
    Patch,
    C,
    <X as Memory>::Tensor<bf16, C, m![Patch, Hidden]>,
    <X as Memory>::Tensor<f32, C, m![Patch]>,
>;

/// Split-BF16 latent parts; callers initialize padding rows to finite zero values.
pub type Latent<C = Chips> = HbmTensor<bf16, C, m![Rows, Parts]>;
/// FP32 residual stream including register, class and padding rows.
pub type Residual<C = Chips> = HbmTensor<f32, C, m![Rows, Hidden]>;
/// Normalized or projected branch rows.
pub type Normalized<C = Chips> = HbmTensor<bf16, C, m![Rows, Hidden]>;
/// Per-head Q/K/V and attended rows, 32 heads of 64 features.
pub type Major<C = Chips> = single::Major<AttentionWidth, C>;
/// Each token's channel-major 4 × 16 × 16 RGB patch, with trailing rows retained.
pub type Pixels<C = Chips> = HbmTensor<f32, C, m![Rows, Patch]>;
/// Use the single-tile numerical leaf's nominal row and head axes.
pub type Rows = single::Rows;
pub type HeadWidth = single::Head;

axes![
    AttentionWidth = 2048,
    Heads = 32,
    Intermediate = 8192,
    Patch = 3072,
    Rotated = 48
];

pub const DEPTH: usize = 36;
pub const TOKENS: usize = 7 * 16 * 16;
pub const LIVE: usize = TOKENS + 5;
pub const EPSILON: usize = 1e-5_f32.to_bits() as usize;
pub const SCALE: usize = 0.125_f32.to_bits() as usize;
