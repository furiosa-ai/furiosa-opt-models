//! H3's input projection folded into the audio decoder's first convolution.
//!
//! The preserved Furiosa loader uses a 32-to-2048 projection, then a 1024-output, seven-tap
//! convolution. Weight normalization and projection folding run in FP64, then round through
//! FP32 to BF16 in output, tap, lifted-input order. The projection bias weighs the lifted
//! ones channel, which is zero on padding rows; the convolution bias remains FP32.
//! Latent statistics are deliberately absent: the original loader ignored its statistics
//! argument. The parent decoder supplies lifted rows and the physical audio checkpoint scope.
//!
//! Arithmetic ported from the preserved Furiosa Rust implementation in
//! `furiosa-opt-engine/src/checkpoint/load/minimax_h3/audio/input.rs` (`Fold::apply`) and
//! `audio.rs` (`WeightNorm::apply`), using the public convolution's identical normalization.

use core::borrow::Borrow;
use core::marker::PhantomData;

use furiosa_opt_std::prelude::*;

use crate::Op;
use crate::load::{Checkpoint, Children, Context, Load, LoadError};
use crate::ops::convolution::frames::{Frames, Window, Zero};
use crate::ops::convolution::input::{Latent, Lifted};
use crate::ops::convolution::{self, Bias, Coefficients, Convolution, Produced};
use crate::tensor::form::Weight;
use crate::tensor::memory::residency::Residency;
use crate::tensor::memory::{Hbm, Memory};
use crate::tensor::parallel::Chip;

/// The folded input convolution with both leaves in memory `X`, replicated on chips `C`.
pub type Input<C = Chip, X = Hbm> =
    Folded<C, <X as Memory>::Tensor<bf16, C, Layout>, <X as Memory>::Tensor<f32, C, BiasLayout>>;

/// A folded coefficient image and the unchanged first-convolution bias, owned or borrowed.
///
/// Load relative to the audio component, where `dec_in_proj.*` and `decoder.conv_pre.*` live.
/// The parent tree supplies that scope with `Named` or a load rename; it must explicitly
/// account for the other decoder weights. This leaf ignores no checkpoint prefixes.
/// Projection weight/bias shapes are `[2048, 32, 1]`/`[2048]`; convolution magnitude,
/// direction and bias shapes are `[1024, 1, 1]`, `[1024, 2048, 7]` and `[1024]`.
#[derive(DeviceSend, Load, Residency)]
#[load(tree)]
#[residency(children(T, B))]
pub struct Folded<C: M = Chip, T = HbmTensor<bf16, C, Layout>, B = Bias<C1024, C>> {
    pub weight: Coefficients<Lifted, C1024, K7, C, T>,
    pub bias: B,
}

/// Reuse the public dense convolution, including its FP32 accumulation and zero output pads.
impl<'x, D, S: Window, C: M, T, B> Op<&'x Frames<D, Lifted, S, C>> for Folded<C, T, B>
where
    D: MaterializableScalar + FetchCast<bf16>,
    T: Borrow<HbmTensor<bf16, C, Layout>>,
    B: Borrow<Bias<C1024, C>>,
{
    type Output = Frames<f32, C1024, S, C>;

    fn forward(&self, device: &mut Device, input: &'x Frames<D, Lifted, S, C>) -> Self::Output {
        let convolution: Convolution<Coefficients<Lifted, C1024, K7, C, _>, _, 1, 1, Zero> =
            Convolution {
                weight: Weight::new(self.weight.tensor()),
                bias: self.bias.borrow(),
                pads: PhantomData,
            };
        convolution.forward(device, input)
    }
}

/// Build each retained leaf from a prepared host image, allowing ordinary HBM and host backends.
impl<'a, C: M + 'static, T, B> Load<&'a Checkpoint> for Folded<C, T, B>
where
    T: Load<HostTensor<bf16, m![{ C }, { Layout }]>>,
    B: Load<HostTensor<f32, m![{ C }, { BiasLayout }]>>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let mut children = Children::default();
            let weight = children.ask(cx.field("weight", "", |cx| {
                let host = Self::coefficients(from, cx)?;
                Load::load(cx, host)
            }))?;
            let bias = children.ask(cx.field("bias", "decoder.conv_pre.bias", |cx| {
                let host = Self::bias(from, cx)?;
                B::load(cx, host)
            }))?;
            let (weight, bias) = children.node(weight.zip(bias))?;
            Ok(Self { weight, bias })
        })
    }
}

/// Recompute the folded weight as one update rule; all four source tensors must be present.
/// The independent convolution bias has its own rule and writes its existing allocation.
impl<'t, 's, C: M + 'static, T, B> Load<(&'t mut Folded<C, T, B>, &'s Checkpoint)>
    for &'t mut Folded<C, T, B>
where
    for<'u> &'u mut Coefficients<Lifted, C1024, K7, C, T>: Load<(
        &'u mut Coefficients<Lifted, C1024, K7, C, T>,
        HostTensor<bf16, m![{ C }, { Layout }]>,
    )>,
    for<'u> &'u mut B: Load<(&'u mut B, HostTensor<f32, m![{ C }, { BiasLayout }]>)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut Folded<C, T, B>, &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.field("weight", "", |cx| {
                cx.rule(
                    |cx| Folded::<C, T, B>::coefficients(from, cx),
                    |cx, host| {
                        <&mut Coefficients<Lifted, C1024, K7, C, T> as Load<_>>::load(
                            cx,
                            (&mut into.weight, host),
                        )
                        .map(drop)
                    },
                )
            })?;
            cx.field("bias", "decoder.conv_pre.bias", |cx| {
                cx.rule(
                    |cx| Folded::<C, T, B>::bias(from, cx),
                    |cx, host| <&mut B as Load<_>>::load(cx, (&mut into.bias, host)).map(drop),
                )
            })
            .map(drop)
        })?;
        Ok(into)
    }
}

impl<C: M + 'static, T, B> Folded<C, T, B> {
    /// Read exact stored shapes as FP32 values (F32, BF16 or F16, as the public loaders allow).
    /// Weight normalization stays FP64 until the entire projection has been folded.
    fn coefficients(
        source: &Checkpoint,
        context: &mut Context<'_>,
    ) -> Result<HostTensor<bf16, m![{ C }, { Layout }]>, LoadError> {
        let mut children = Children::default();
        let projection = children.ask(context.field(
            "dec_in_proj.weight",
            "dec_in_proj.weight",
            |cx| cx.floats(source, &[PROJECTED, Latent::SIZE, 1]),
        ))?;
        let bias = children.ask(context.field("dec_in_proj.bias", "dec_in_proj.bias", |cx| {
            cx.floats(source, &[PROJECTED])
        }))?;
        let pre = children.ask(context.field("decoder.conv_pre", "decoder.conv_pre", |cx| {
            convolution::normed(source, cx, &[C1024::SIZE, PROJECTED, K7::SIZE])
        }))?;
        let ((projection, bias), pre) = children.node(projection.zip(bias).zip(pre))?;
        let folded = Fold { projection, bias }.apply(pre);
        let coefficients: Vec<bf16> = folded.into_iter().map(bf16::from_f32).collect();
        Ok(HostTensor::from_vec(coefficients.repeat(C::SIZE)))
    }

    /// The first convolution's own bias is added after all taps; it is not a lifted coefficient.
    fn bias(
        source: &Checkpoint,
        context: &mut Context<'_>,
    ) -> Result<HostTensor<f32, m![{ C }, { BiasLayout }]>, LoadError> {
        let bias = context.floats(source, &[C1024::SIZE])?;
        Ok(HostTensor::from_vec(bias.repeat(C::SIZE)))
    }
}

/// The projection `W_in`, `b_in`; its input is the latent itself, without a statistics transform.
struct Fold {
    projection: Vec<f32>,
    bias: Vec<f32>,
}

impl Fold {
    /// Preserve the reference's middle-channel summation order, then narrow FP64 to FP32 once.
    fn apply(&self, pre: Vec<f64>) -> Vec<f32> {
        let mut lifted = vec![0.0_f64; PROJECTED * Lifted::SIZE];
        for row in 0..PROJECTED {
            let weights = &self.projection[row * Latent::SIZE..][..Latent::SIZE];
            for channel in 0..Latent::SIZE {
                lifted[row * Lifted::SIZE + channel] = f64::from(weights[channel]);
            }
            lifted[row * Lifted::SIZE + Latent::SIZE] = f64::from(self.bias[row]);
        }
        let mut folded = vec![0.0_f64; C1024::SIZE * K7::SIZE * Lifted::SIZE];
        for out in 0..C1024::SIZE {
            for tap in 0..K7::SIZE {
                let to = &mut folded[(out * K7::SIZE + tap) * Lifted::SIZE..][..Lifted::SIZE];
                for middle in 0..PROJECTED {
                    let weight = pre[(out * PROJECTED + middle) * K7::SIZE + tap];
                    for (sum, &value) in to[..=Latent::SIZE]
                        .iter_mut()
                        .zip(&lifted[middle * Lifted::SIZE..][..=Latent::SIZE])
                    {
                        *sum += weight * value;
                    }
                }
            }
        }
        folded.iter().map(|&value| value as f32).collect()
    }
}

/// Coefficients in the public dense convolution's output, tap, lifted-input order.
pub type Layout = m![{ Symbol<Produced<C1024>> }, K7, Lifted];

/// One FP32 bias for each dense output channel.
pub type BiasLayout = m![{ Symbol<Produced<C1024>> }];

axes![C1024 = 1024, K7 = 7];

/// The preserved audio loader and checked H3 audio configuration both fix this width at 2048.
const PROJECTED: usize = 2048;
