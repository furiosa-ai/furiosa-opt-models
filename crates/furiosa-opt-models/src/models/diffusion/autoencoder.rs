//! Spatial KL autoencoders compose encoding, a diagonal posterior, and decoding; the KL
//! autoencoder by published size is `KL<Size<"256">, f32, Parallel<1>, X>`.
//!
//! - A size is the sample size it compiles; its [`Geometry`] holds the sample's extent. Every size
//!   has the one encoder and decoder of RGB pixels and four latent channels.

use crate::models::Family;
pub use crate::models::Scalar;
use crate::modules::gaussian::{self, Sample};
use crate::modules::{self, convolution, residual};
use crate::modules::{attention::Full, language};
use crate::ops::attention;
use crate::ops::gaussian::Mode;
pub use crate::ops::gaussian::Moments;
use crate::ops::{
    convolution::{Bias, Convolution, Flatten, Restore, Spatial, Upsample},
    linear::Coefficients,
    normalization::GroupNorm,
    pointwise::{activation::Silu, add::Add},
};
use crate::{Identity, Sequential, Then};
use furiosa_opt_std::prelude::*;
use serde::Deserialize;
use std::{io, marker::PhantomData};

/// Spatial KL autoencoder; defaults use RGB pixels and four-channel latents.
pub enum Model<
    I: AxisName = Channels<PIXELS>,
    O: AxisName = Channels<PIXELS>,
    L: AxisName = Channels<LATENTS>,
    Channels0: AxisName = Channels<{ LEVELS[0] }>,
    Channels1: AxisName = Channels<{ LEVELS[1] }>,
    Channels2: AxisName = Channels<{ LEVELS[2] }>,
    G: AxisName = Channels<GROUPS>,
    const ENCODER: usize = { LAYERS - 1 },
    const DECODER: usize = LAYERS,
> {
    /// Both 1×1 latent projections are present.
    Projected(Projected<I, O, L, Channels0, Channels1, Channels2, G, ENCODER, DECODER>),
    /// Neither 1×1 latent projection is present.
    Unprojected(Unprojected<I, O, L, Channels0, Channels1, Channels2, G, ENCODER, DECODER>),
}

/// Autoencoder with `quant_conv` and `post_quant_conv` around the latent posterior.
pub type Projected<I, O, L, Channels0, Channels1, Channels2, G, const E: usize, const D: usize> =
    AutoencoderKL<
        Encoder<I, L, Channels0, Channels1, Channels2, G, E>,
        Posterior,
        Decoder<O, L, Channels0, Channels1, Channels2, G, D>,
    >;

/// Encoder body followed by the checkpoint's 1×1 moment projection.
pub type Encoder<I, L, Channels0, Channels1, Channels2, G, const N: usize> = Then<
    EncoderBody<I, L, Channels0, Channels1, Channels2, G, N>,
    Convolution<Spatial<Moments<L>, Moments<L>, Point>, Bias<Moments<L>, m![1]>>,
>;

/// Convolutional encoder producing paired mean and log-variance channels.
pub type EncoderBody<I, L, Channels0, Channels1, Channels2, G, const N: usize> = VaeEncoder<
    Convolution<Spatial<I, Channels0, Kernel>, Bias<Channels0, m![1]>>,
    Down<Channels0, G, N>,
    DownTransition<Channels0, Channels1, G, N>,
    DownTransition<Channels1, Channels2, G, N>,
    Stack<Channels2, G, N>,
    Residual<Channels2, Channels2, G>,
    Attention<Channels2, G>,
    Head<Channels2, Moments<L>, G>,
>;

type Down<C, G, const N: usize> = VaeStage<
    Residual<C, C, G>,
    Residual<C, C, G>,
    Convolution<Spatial<C, C, Kernel>, Bias<C, m![1]>, 2>,
    N,
>;

/// The input's norm and activation are composite, so they lead the branch: a residual's norm place
/// holds one norm.
pub type Residual<I, O, G, A = Add> = residual::Residual<
    (),
    Then<
        Then<Norm<I, G>, Silu>,
        convolution::Branch<
            Convolution<Spatial<I, O, Kernel>, Bias<O, m![1]>>,
            Norm<O, G>,
            Silu,
            Convolution<Spatial<O, O, Kernel>, Bias<O, m![1]>>,
        >,
    >,
    A,
>;

pub type Norm<C, G> = GroupNorm<C, G>;

type DownTransition<I, O, G, const N: usize> = VaeStage<
    Projection<I, O, G>,
    Residual<O, O, G>,
    Convolution<Spatial<O, O, Kernel>, Bias<O, m![1]>, 2>,
    N,
>;

type Projection<I, O, G> =
    Residual<I, O, G, residual::Shortcut<Convolution<Spatial<I, O, Point>, Bias<O, m![1]>>, Add>>;

pub type Stack<C, G, const N: usize> = convolution::Stack<Residual<C, C, G>, Residual<C, C, G>, N>;

pub type Attention<C, G> = VaeAttention<
    Norm<C, G>,
    modules::attention::Projection<Linear<C>>,
    attention::Attention<PhantomData<Point>, Scalar>,
    Linear<C>,
    Add,
    Flatten,
    Restore,
>;

pub type Linear<C> = Coefficients<f32, C, C, 1>;

pub type Head<I, O, G> =
    VaeHead<Norm<I, G>, Silu, Convolution<Spatial<I, O, Kernel>, Bias<O, m![1]>>>;

/// Diagonal Gaussian posterior supporting its mode or caller-supplied sampling noise.
pub type Posterior = gaussian::Diagonal<Mode, Sample>;

/// Decoder body preceded by the checkpoint's 1×1 latent projection.
pub type Decoder<O, L, Channels0, Channels1, Channels2, G, const N: usize> = Then<
    Convolution<Spatial<L, L, Point>, Bias<L, m![1]>>,
    DecoderBody<O, L, Channels0, Channels1, Channels2, G, N>,
>;

/// Convolutional decoder expanding spatial latents to output pixels.
pub type DecoderBody<O, L, Channels0, Channels1, Channels2, G, const N: usize> = VaeDecoder<
    Convolution<Spatial<L, Channels2, Kernel>, Bias<Channels2, m![1]>>,
    Residual<Channels2, Channels2, G>,
    Attention<Channels2, G>,
    Then<Stack<Channels2, G, N>, Up<Channels2>>,
    Then<Stack<Channels2, G, N>, Up<Channels2>>,
    Then<Transition<Channels2, Channels1, G, N>, Up<Channels1>>,
    Transition<Channels1, Channels0, G, N>,
    Head<Channels0, O, G>,
>;

type Up<C> = Then<Upsample, Convolution<Spatial<C, C, Kernel>, Bias<C, m![1]>>>;

pub type Transition<I, O, G, const N: usize> =
    convolution::Stack<Projection<I, O, G>, Residual<O, O, G>, N>;

/// Autoencoder using moments and latents directly, with no 1×1 latent projections.
pub type Unprojected<I, O, L, Channels0, Channels1, Channels2, G, const E: usize, const D: usize> =
    AutoencoderKL<
        Sequential<EncoderBody<I, L, Channels0, Channels1, Channels2, G, E>, Identity>,
        Posterior,
        Sequential<Identity, DecoderBody<O, L, Channels0, Channels1, Channels2, G, D>>,
    >;

/// Channel count of a spatial image or latent tensor.
#[derive(Clone, Debug)]
pub struct Channels<const N: usize>;

impl<const N: usize> AxisName for Channels<N> {
    const NAME: Ident = Ident::new("Channels");
    const SIZE: usize = N;
}

/// Spatial height in pixels or latent rows.
#[derive(Clone, Debug)]
pub struct Height<const N: usize>;

impl<const N: usize> AxisName for Height<N> {
    const NAME: Ident = Ident::new("Height");
    const SIZE: usize = N;
}

/// Spatial width in pixels or latent columns.
#[derive(Clone, Debug)]
pub struct Width<const N: usize>;

impl<const N: usize> AxisName for Width<N> {
    const NAME: Ident = Ident::new("Width");
    const SIZE: usize = N;
}

pub type Kernel = Channels<3>;
type Point = Channels<1>;

/// Channels of a pixel: `in_channels` and `out_channels`.
const PIXELS: usize = 3;

/// Channels of a latent: `latent_channels`.
const LATENTS: usize = 4;

/// Feature channels of each resolution level, the last two alike: `block_out_channels`.
const LEVELS: [usize; 4] = [128, 256, 512, 512];

/// Residual blocks of an encoder level, one fewer than a decoder level's: `layers_per_block`.
const LAYERS: usize = 2;

/// Groups of every group norm: `norm_num_groups`.
const GROUPS: usize = 32;

/// A published KL autoencoder size, by the sample size it compiles.
pub struct Size<const N: &'static str>;

/// What a KL autoencoder size fixes beyond the encoder and decoder every size shares.
pub trait Geometry {
    /// Rows of a sample: `sample_size`.
    type Height: AxisName;
    /// Columns of a sample: `sample_size`.
    type Width: AxisName;

    /// The size's spelling, as `"256"`.
    const SIZE: &'static str;
}

impl Geometry for Size<"256"> {
    type Height = Height<256>;
    type Width = Width<256>;

    const SIZE: &'static str = "256";
}

impl Geometry for Size<"512"> {
    type Height = Height<512>;
    type Width = Width<512>;

    const SIZE: &'static str = "512";
}

impl Geometry for Size<"1024"> {
    type Height = Height<1024>;
    type Width = Width<1024>;

    const SIZE: &'static str = "1024";
}

/// What a KL autoencoder's config declares that its size fixes, and the features the encoder and
/// decoder do not serve.
#[derive(Deserialize)]
pub struct Config {
    sample_size: usize,
    in_channels: usize,
    out_channels: usize,
    latent_channels: usize,
    block_out_channels: Vec<usize>,
    layers_per_block: usize,
    norm_num_groups: usize,
    act_fn: String,
    down_block_types: Vec<String>,
    up_block_types: Vec<String>,
    #[serde(default)]
    mid_block_add_attention: Option<bool>,
    #[serde(default)]
    use_quant_conv: Option<bool>,
    #[serde(default)]
    use_post_quant_conv: Option<bool>,
}

/// A config is of a size when it declares the size's values, Diffusers' defaults standing for the
/// keys it omits, and none of the features the encoder and decoder do not serve.
impl<const N: &'static str> TryFrom<Config> for Size<N>
where
    Self: Geometry,
{
    type Error = io::Error;

    fn try_from(config: Config) -> io::Result<Self> {
        let invalid = |reason: String| io::Error::new(io::ErrorKind::InvalidData, reason);
        let unsupported = |reason: &str| io::Error::new(io::ErrorKind::Unsupported, reason);
        let fixed = [
            (
                "sample_size",
                config.sample_size,
                <Self as Geometry>::Height::SIZE,
            ),
            ("in_channels", config.in_channels, PIXELS),
            ("out_channels", config.out_channels, PIXELS),
            ("latent_channels", config.latent_channels, LATENTS),
            ("layers_per_block", config.layers_per_block, LAYERS),
            ("norm_num_groups", config.norm_num_groups, GROUPS),
        ];
        if let Some((key, declared, held)) = fixed
            .into_iter()
            .find(|(_, declared, held)| declared != held)
        {
            return Err(invalid(format!(
                "{key}: {declared}, the size {N} holds {held}"
            )));
        }
        if config.block_out_channels != LEVELS {
            return Err(invalid(format!(
                "block_out_channels: {:?}, the size {N} holds {LEVELS:?}",
                config.block_out_channels
            )));
        }
        let projections = [
            ("use_quant_conv", config.use_quant_conv, "moment"),
            ("use_post_quant_conv", config.use_post_quant_conv, "latent"),
        ];
        if let Some((key, _, projection)) = projections
            .into_iter()
            .find(|(_, declared, _)| *declared == Some(false))
        {
            return Err(invalid(format!(
                "{key}: false, the size {N} holds the {projection} projection"
            )));
        }
        let plain = |types: &[String], block: &str| {
            types.len() == LEVELS.len() && types.iter().all(|declared| declared == block)
        };
        let served = [
            (
                matches!(config.act_fn.as_str(), "silu" | "swish"),
                "act_fn: the encoder and decoder activate by SiLU",
            ),
            (
                plain(&config.down_block_types, "DownEncoderBlock2D"),
                "down_block_types: every encoder level is a DownEncoderBlock2D",
            ),
            (
                plain(&config.up_block_types, "UpDecoderBlock2D"),
                "up_block_types: every decoder level is an UpDecoderBlock2D",
            ),
            (
                config.mid_block_add_attention != Some(false),
                "mid_block_add_attention: both middle blocks attend",
            ),
        ];
        match served.into_iter().find(|(served, _)| !served) {
            Some((_, reason)) => Err(unsupported(reason)),
            None => Ok(Size),
        }
    }
}

impl<const N: &'static str> Family for Size<N>
where
    Self: Geometry,
{
    type Hf = Config;
}

/// Normalized spatial attention with a residual sum.
pub(crate) type VaeAttention<Norm, Qkv, Attend, Project, Add, Flatten = (), Restore = ()> =
    convolution::Attention<Norm, Flatten, Then<Identity, Full<Qkv, Attend, Project>>, Restore, Add>;

/// Two residual blocks surrounding one attention block.
pub type VaeMiddle<Residual, Attention> = Sequential<Residual, Sequential<Attention, Residual>>;

/// Normalization and activation followed by the output convolution.
pub(crate) type VaeHead<Norm, Activation, Convolution> = Then<Norm, Then<Activation, Convolution>>;

/// The first residual may change channels; each resampler owns its spatial or temporal history.
pub(crate) type VaeStage<First, Residual, Resample, const REPEATS: usize> =
    Then<convolution::Stack<First, Residual, REPEATS>, Resample>;

/// Four resolution levels retain distinct channel, extent and history types; each stage takes the
/// output of the stage before it, and the head reads the middle's.
pub(crate) type VaeEncoder<Stem, Down0, Down1, Down2, Last, Residual, Attention, Head> = Sequential<
    Stem,
    Sequential<
        Sequential<Down0, Sequential<Down1, Sequential<Down2, Last>>>,
        Then<VaeMiddle<Residual, Attention>, Head>,
    >,
>;

/// Latent stem and middle blocks followed by four decoding stages and an output head.
pub(crate) type VaeDecoder<Stem, Residual, Attention, Up3, Up2, Up1, Last, Head> = Sequential<
    Stem,
    Sequential<
        VaeMiddle<Residual, Attention>,
        Then<Sequential<Up3, Sequential<Up2, Sequential<Up1, Last>>>, Head>,
    >,
>;

/// Source: Diffusers `src/diffusers/models/autoencoders/autoencoder_kl.py`.
/// Encoder features parameterize a posterior whose latent is passed to the decoder.
pub type AutoencoderKL<Encoder, Posterior, Decoder> =
    language::Readout<language::Pooler<Encoder, Posterior>, Decoder>;
