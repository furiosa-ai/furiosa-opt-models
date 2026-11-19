//! H3's video post-quantization convolution folded into the decoder input projection.
//!
//! Ported from the preserved Furiosa Rust `checkpoint/load/minimax_h3/video.rs`
//! (`Embedding::load` and `Fold`). Each source value rounds through BF16 before the FP64
//! affine fold. The folded coefficients narrow once to FP32, then split into BF16 parts
//! `[w_hi | w_hi | w_lo | 0]` against the caller's `[z_hi | z_lo | z_hi | 0]` rows.
//! The bias remains FP32. Latent statistics are deliberately absent: the original loader
//! ignored its statistics argument. Register and class-token rows belong to the parent tree.

use core::borrow::Borrow;
use core::marker::PhantomData;

use furiosa_opt_std::prelude::*;

use crate::Op;
use crate::load::{Checkpoint, Children, Context, Load, LoadError};
use crate::ops::linear::{Affine, Unrounded};
use crate::tensor::memory::residency::Residency;
use crate::tensor::memory::{Hbm, Memory};
use crate::tensor::parallel::Chip;

/// The folded input projection with both leaves in memory `X`, replicated on chips `C`.
pub type Input<C = Chip, X = Hbm> =
    Folded<C, <X as Memory>::Tensor<bf16, C, Layout>, <X as Memory>::Tensor<f32, C, BiasLayout>>;

/// A folded BF16 coefficient image and FP32 bias, owned or borrowed.
///
/// Load relative to the video component, where `post_quant_conv.*` and
/// `decoder.x_embedder.*` live. The parent supplies the physical checkpoint scope and
/// accounts for the rest of the decoder. This leaf ignores no checkpoint prefixes.
/// Quantization weight/bias shapes are `[24, 24, 1, 1, 1]`/`[24]`; embedding
/// weight/bias shapes are `[2048, 24]`/`[2048]`.
#[derive(DeviceSend, Load, Residency)]
#[load(tree)]
#[residency(children(T, B))]
pub struct Folded<C: M = Chip, T = HbmTensor<bf16, C, Layout>, B = HbmTensor<f32, C, BiasLayout>> {
    pub weight: T,
    pub bias: B,
    shape: PhantomData<fn() -> C>,
}

/// Delegate the split-BF16 contraction and FP32 output to the public projection operation.
impl<'x, R: AxisName, C: M, T, B> Op<&'x HbmTensor<bf16, C, m![R, Parts]>> for Folded<C, T, B>
where
    T: Borrow<HbmTensor<bf16, C, Layout>>,
    B: Borrow<HbmTensor<f32, C, BiasLayout>>,
{
    type Output = HbmTensor<f32, C, m![R, Hidden]>;

    fn forward(
        &self,
        device: &mut Device,
        input: &'x HbmTensor<bf16, C, m![R, Parts]>,
    ) -> Self::Output {
        let weight: &HbmTensor<bf16, C, Layout> = self.weight.borrow();
        let bias: &HbmTensor<f32, C, BiasLayout> = self.bias.borrow();
        let projection: Unrounded<Parts, Hidden, C, _, _> =
            Unrounded(Affine::new(weight, bias), PhantomData);
        projection.forward(device, input)
    }
}

/// A composition can pass owned rows through the same borrowed execution path.
impl<R: AxisName, C: M, T, B> Op<HbmTensor<bf16, C, m![R, Parts]>> for Folded<C, T, B>
where
    T: Borrow<HbmTensor<bf16, C, Layout>>,
    B: Borrow<HbmTensor<f32, C, BiasLayout>>,
{
    type Output = HbmTensor<f32, C, m![R, Hidden]>;

    fn forward(
        &self,
        device: &mut Device,
        input: HbmTensor<bf16, C, m![R, Parts]>,
    ) -> Self::Output {
        self.forward(device, &input)
    }
}

/// Prepare both host images before loading through the selected storage backend.
impl<'a, C: M + 'static, T, B> Load<&'a Checkpoint> for Folded<C, T, B>
where
    T: Load<HostTensor<bf16, m![{ C }, { Layout }]>>,
    B: Load<HostTensor<f32, m![{ C }, { BiasLayout }]>>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let (weight, bias) = Self::folded(from, cx)?;
            let weight = cx.field("weight", "", |cx| T::load(cx, weight))?;
            let bias = cx.field("bias", "", |cx| B::load(cx, bias))?;
            Ok(Self {
                weight,
                bias,
                shape: PhantomData,
            })
        })
    }
}

/// Both retained leaves form one update rule requiring all four source tensors.
/// Missing sources defer both writes; complete images write the existing allocations.
impl<'t, 's, C: M + 'static, T, B> Load<(&'t mut Folded<C, T, B>, &'s Checkpoint)>
    for &'t mut Folded<C, T, B>
where
    for<'u> &'u mut T: Load<(&'u mut T, HostTensor<bf16, m![{ C }, { Layout }]>)>,
    for<'u> &'u mut B: Load<(&'u mut B, HostTensor<f32, m![{ C }, { BiasLayout }]>)>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (&'t mut Folded<C, T, B>, &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.rule(
                |cx| Folded::<C, T, B>::folded(from, cx),
                |cx, (weight, bias)| {
                    cx.field("weight", "", |cx| {
                        <&mut T as Load<_>>::load(cx, (&mut into.weight, weight)).map(drop)
                    })?;
                    cx.field("bias", "", |cx| {
                        <&mut B as Load<_>>::load(cx, (&mut into.bias, bias)).map(drop)
                    })
                },
            )
            .map(drop)
        })?;
        Ok(into)
    }
}

impl<C: M + 'static, T, B> Folded<C, T, B> {
    /// Read every dependency even when another is missing, preserving strict update grouping.
    fn folded(
        source: &Checkpoint,
        context: &mut Context<'_>,
    ) -> Result<
        (
            HostTensor<bf16, m![{ C }, { Layout }]>,
            HostTensor<f32, m![{ C }, { BiasLayout }]>,
        ),
        LoadError,
    > {
        let mut children = Children::default();
        let quant = children.ask(rounded(
            source,
            context,
            "post_quant_conv.weight",
            &[CHANNELS, CHANNELS, 1, 1, 1],
        ))?;
        let quant_bias = children.ask(rounded(
            source,
            context,
            "post_quant_conv.bias",
            &[CHANNELS],
        ))?;
        let embed = children.ask(rounded(
            source,
            context,
            "decoder.x_embedder.weight",
            &[Hidden::SIZE, CHANNELS],
        ))?;
        let bias = children.ask(rounded(
            source,
            context,
            "decoder.x_embedder.bias",
            &[Hidden::SIZE],
        ))?;
        let (((quant, quant_bias), embed), bias) =
            children.node(quant.zip(quant_bias).zip(embed).zip(bias))?;
        let fold = Fold { quant, quant_bias };
        let weight = fold.parts(&embed);
        let bias = fold.bias(&embed, bias);
        Ok((
            HostTensor::from_vec(weight.repeat(C::SIZE)),
            HostTensor::from_vec(bias.repeat(C::SIZE)),
        ))
    }
}

/// The post-quantization affine transform; statistics do not enter this fold.
struct Fold {
    quant: Vec<f32>,
    quant_bias: Vec<f32>,
}

impl Fold {
    /// Preserve the middle-channel summation order, then split the FP32 result into BF16 parts.
    fn parts(&self, embed: &[f32]) -> Vec<bf16> {
        let mut parts = vec![bf16::from_f32(0.0); Hidden::SIZE * Parts::SIZE];
        for (row, embedded) in embed.chunks_exact(CHANNELS).enumerate() {
            for input in 0..CHANNELS {
                let product: f64 = (0..CHANNELS)
                    .map(|middle| {
                        f64::from(embedded[middle])
                            * f64::from(self.quant[middle * CHANNELS + input])
                    })
                    .sum();
                let value = product as f32;
                let high = bf16::from_f32(value);
                let low = bf16::from_f32(value - high.to_f32());
                let at = row * Parts::SIZE + input;
                (parts[at], parts[at + CHANNELS], parts[at + 2 * CHANNELS]) = (high, high, low);
            }
        }
        parts
    }

    /// Add the embedding offset after the FP64 product, then narrow once to FP32.
    fn bias(&self, embed: &[f32], offsets: Vec<f32>) -> Vec<f32> {
        embed
            .chunks_exact(CHANNELS)
            .zip(offsets)
            .map(|(embedded, offset)| {
                let sum: f64 = (0..CHANNELS)
                    .map(|middle| f64::from(embedded[middle]) * f64::from(self.quant_bias[middle]))
                    .sum();
                (sum + f64::from(offset)) as f32
            })
            .collect()
    }
}

/// Folded coefficients in the public projection's output, input-part order.
pub type Layout = m![Hidden, Parts];

/// One FP32 bias for each decoder hidden feature.
pub type BiasLayout = m![Hidden];

// The video model re-exports these axes so its rows have the same nominal types.
axes![Parts = 128, Hidden = 2048];

/// The preserved video latent has 24 channels.
const CHANNELS: usize = 24;

/// Exact stored shapes, converted to FP32 and rounded as the preserved `Entry::rounded` did.
fn rounded(
    source: &Checkpoint,
    context: &mut Context<'_>,
    name: &str,
    shape: &[usize],
) -> Result<Vec<f32>, LoadError> {
    context.field(name, name, |cx| {
        cx.floats(source, shape).map(|values| {
            values
                .into_iter()
                .map(|value| bf16::from_f32(value).to_f32())
                .collect()
        })
    })
}
