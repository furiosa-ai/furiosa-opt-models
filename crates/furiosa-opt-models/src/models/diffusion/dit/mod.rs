//! DiT, by its published size: class-conditioned latent prediction assembled into one device entry,
//! over a [`Size`], as `Transformer2DModel<Size<"XL-2-256">, f32, Parallel<1>, X>`.
//!
//! - The size fixes every value its compiled transformer holds.
use crate::models::{Family, Scalar};
use crate::modules::diffusion::{AdaptiveNorm, SplitGates, SplitModulation};
use crate::modules::vision::{Patch, SinCos};
use crate::modules::{attention, diffusion, feedforward, language};
use crate::ops::{
    self,
    embedding::Lookup,
    linear::Coefficients,
    pointwise::{
        activation::{NewGelu, Silu},
        add::Add,
        gated,
    },
};
use crate::tensor::image::Grid;
use crate::tensor::image::{Image, Unpatch, Vector};
use crate::{Module, Then};
use furiosa_opt_std::prelude::*;
use serde::Deserialize;
use std::borrow::Borrow;
use std::{io, marker::PhantomData};

pub use crate::tensor::axis::Single;

pub mod pipeline;

/// Class-conditioned latent predictor with separate noise and variance channels.
#[derive(DeviceSend)]
pub struct Model {
    /// Patch projection plus spatial position embedding.
    pub embedding: Embedding,
    /// Adaptive-normalization transformer layers.
    pub layers: [Layer; LAYERS],
    /// Conditioning-modulated projection to patch predictions.
    pub head: Head,
    /// Restores channel, row, column order from patch predictions.
    pub unpatch: Unpatch<
        Channels<OUT_CHANNELS>,
        Height<SAMPLE_SIZE>,
        Columns<SAMPLE_SIZE>,
        Height<PATCH_SIZE>,
        Columns<PATCH_SIZE>,
    >,
}

impl<T: Borrow<Scalar>, Env: Copy> Module<(&Sample, (&Label, T)), Env, &mut ()> for Model {
    type Output = Prediction;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        (sample, (label, time)): (&Sample, (&Label, T)),
    ) -> Prediction {
        let condition = ConditioningInput {
            time: time.borrow(),
            label,
        };
        let hidden =
            self.embedding
                .forward(&mut *device, env, &mut Default::default(), (sample, sample));
        let hidden = self.layers.forward(
            &mut *device,
            (&condition, ()),
            &mut Default::default(),
            hidden,
        );
        let hidden = self.head.forward(
            &mut *device,
            env,
            &mut Default::default(),
            (&hidden, &condition),
        );
        self.unpatch.forward(device, env, &mut (), &hidden)
    }
}

type Embedding = language::ConditionedEmbedding<
    Patch<Channels<IN_CHANNELS>, Hidden, Height<PATCH_SIZE>, Columns<PATCH_SIZE>>,
    SinCos<Hidden, Height<PATCH_SIZE>, Columns<PATCH_SIZE>>,
    Add,
>;

pub type Layer = diffusion::Adaptive<
    Modulation<Width<6912>, SplitGates<Hidden>>,
    Attention,
    Mlp<Hidden, Width<4608>, Hidden, NewGelu>,
    AdaptiveNorm,
    gated::Add,
>;

type Modulation<Packed, Split> =
    Then<Conditioning, Then<Silu, Then<Linear<Hidden, Packed>, Split>>>;

pub type Conditioning = language::ConditionedEmbedding<
    Then<diffusion::Time<Width<256>>, Mlp<Width<256>, Hidden, Hidden, Silu>>,
    Lookup<HbmTensor<f32, m![1], m![Classes, {Symbol<Hidden>}]>>,
    Add,
>;

/// The state holds the time embedding's slots, then the label lookup's.
impl<'s, Env: Copy> Module<&ConditioningInput<'_>, Env, &'s mut (((), ((), ((), ()))), ())>
    for Conditioning
{
    type Output = Vector<Hidden>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        state: &'s mut (((), ((), ((), ()))), ()),
        input: &ConditioningInput<'_>,
    ) -> Self::Output {
        self.forward(device, env, state, (input.time, input.label))
    }
}

type Mlp<In, Intermediate, Out, A> =
    feedforward::Mlp<Linear<In, Intermediate>, A, Linear<Intermediate, Out>>;

pub type Linear<In, Out> = Coefficients<f32, In, Out, 1>;

type Attention = attention::Full<
    attention::Projection<Linear<Hidden, Hidden>>,
    ops::attention::Attention<PhantomData<Heads>, Scalar>,
    Linear<Hidden, Hidden>,
>;

type Head = diffusion::Head<
    Modulation<Width<2304>, SplitModulation<Hidden>>,
    AdaptiveNorm,
    Linear<Hidden, Width<32>>,
>;

/// Four-channel latent image in channel, row, column order.
pub type Sample = Image<Channels<IN_CHANNELS>, Height<SAMPLE_SIZE>, Columns<SAMPLE_SIZE>>;

/// Noise and variance predictions, each occupying four channels.
pub type Prediction = Image<Channels<OUT_CHANNELS>, Height<SAMPLE_SIZE>, Columns<SAMPLE_SIZE>>;

/// Grid of 2×2 patches in a 32×32 latent image.
pub type Patches =
    Grid<Height<SAMPLE_SIZE>, Columns<SAMPLE_SIZE>, Height<PATCH_SIZE>, Columns<PATCH_SIZE>>;

/// Class index, including the unconditional embedding-table entry.
pub type Label = HbmTensor<i32, m![1], m![Single]>;

type Condition = (Scalar, Label);

struct ConditioningInput<'a> {
    time: &'a Scalar,
    label: &'a Label,
}

/// Latent image and its timestep/class conditioning.
#[derive(DeviceSend)]
pub struct Context {
    /// Current noisy latent image.
    pub sample: Sample,
    /// Training timestep followed by class index.
    pub condition: Condition,
}

/// The sample and the condition a launch's context holds, the class label before the time.
impl<Env: Copy> Module<&mut Context, Env, &mut ()> for Model {
    type Output = Prediction;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        context: &mut Context,
    ) -> Prediction {
        let (time, label) = &context.condition;
        self.forward(device, env, &mut (), (&context.sample, (label, time)))
    }
}

impl From<(Sample, Scalar, Label)> for Context {
    fn from((sample, time, label): (Sample, Scalar, Label)) -> Self {
        Self {
            sample,
            condition: (time, label),
        }
    }
}

/// Feature width of a token or conditioning vector.
#[derive(Clone, Debug)]
pub struct Width<const N: usize>;

impl<const N: usize> AxisName for Width<N> {
    const NAME: Ident = Ident::new("Width");
    const SIZE: usize = N;
}

/// Spatial row count of a latent image.
#[derive(Clone, Debug)]
pub struct Height<const N: usize>;

impl<const N: usize> AxisName for Height<N> {
    const NAME: Ident = Ident::new("Height");
    const SIZE: usize = N;
}

/// Spatial column count of a latent image.
#[derive(Clone, Debug)]
pub struct Columns<const N: usize>;

impl<const N: usize> AxisName for Columns<N> {
    const NAME: Ident = Ident::new("Columns");
    const SIZE: usize = N;
}

/// Channel count of a latent image or predicted noise and variance.
#[derive(Clone, Debug)]
pub struct Channels<const N: usize>;

impl<const N: usize> AxisName for Channels<N> {
    const NAME: Ident = Ident::new("Channels");
    const SIZE: usize = N;
}

axes![Classes = 1001, Heads = 16];

/// Features of a row, its heads together: `num_attention_heads * attention_head_dim`.
type Hidden = Width<1152>;

/// Transformer layers: `num_layers`.
const LAYERS: usize = 28;

/// Channels of a latent sample: `in_channels`.
const IN_CHANNELS: usize = 4;

/// Channels of a prediction, its noise then its variance: `out_channels`.
const OUT_CHANNELS: usize = 8;

/// Rows and columns of a latent sample: `sample_size`.
const SAMPLE_SIZE: usize = 32;

/// Rows and columns of a patch: `patch_size`.
const PATCH_SIZE: usize = 2;

/// The feed-forward norm's epsilon, `norm_eps`, as the bits of the FP32 nearest to it.
const EPSILON: usize = 1e-5_f32.to_bits() as usize;

/// A published DiT size, by the hub's model name without its family.
pub struct Size<const N: &'static str>;

/// What a class-conditioned image transformer's config declares that the size fixes, and the
/// features the size does not serve.
#[derive(Deserialize)]
pub struct Config {
    num_attention_heads: usize,
    attention_head_dim: usize,
    num_layers: usize,
    in_channels: usize,
    #[serde(default)]
    out_channels: Option<usize>,
    sample_size: usize,
    patch_size: usize,
    num_embeds_ada_norm: usize,
    #[serde(default)]
    interpolation_scale: Option<f64>,
    #[serde(default)]
    norm_eps: Option<f64>,
    norm_type: String,
    activation_fn: String,
    norm_elementwise_affine: bool,
    attention_bias: bool,
    #[serde(default)]
    only_cross_attention: bool,
    #[serde(default)]
    double_self_attention: bool,
    #[serde(default)]
    use_linear_projection: bool,
    #[serde(default)]
    cross_attention_dim: Option<usize>,
    #[serde(default)]
    num_vector_embeds: Option<usize>,
}

/// A config is of the XL-2-256 size when it declares the size's values, Diffusers' defaults
/// standing for the keys it omits, and none of the features the size does not serve.
impl TryFrom<Config> for Size<"XL-2-256"> {
    type Error = io::Error;

    fn try_from(config: Config) -> io::Result<Self> {
        let invalid = |reason: String| io::Error::new(io::ErrorKind::InvalidData, reason);
        let unsupported = |reason: &str| io::Error::new(io::ErrorKind::Unsupported, reason);
        let fixed = [
            (
                "num_attention_heads",
                config.num_attention_heads,
                Heads::SIZE,
            ),
            (
                "attention_head_dim",
                config.attention_head_dim,
                Hidden::SIZE / Heads::SIZE,
            ),
            ("num_layers", config.num_layers, LAYERS),
            ("in_channels", config.in_channels, IN_CHANNELS),
            (
                "out_channels",
                config.out_channels.unwrap_or(config.in_channels),
                OUT_CHANNELS,
            ),
            ("sample_size", config.sample_size, SAMPLE_SIZE),
            ("patch_size", config.patch_size, PATCH_SIZE),
            (
                "num_embeds_ada_norm",
                config.num_embeds_ada_norm,
                Classes::SIZE - 1,
            ),
        ];
        if let Some((key, declared, held)) = fixed
            .into_iter()
            .find(|(_, declared, held)| declared != held)
        {
            return Err(invalid(format!(
                "{key}: {declared}, the size XL-2-256 holds {held}"
            )));
        }
        // Diffusers scales the positions of a sample of 64 rows or fewer by one.
        let scale = config.interpolation_scale.unwrap_or(1.0);
        if scale != 1.0 {
            return Err(invalid(format!(
                "interpolation_scale: {scale}, the size XL-2-256 holds 1"
            )));
        }
        let epsilon = config.norm_eps.unwrap_or(1e-5);
        if (epsilon as f32).to_bits() as usize != EPSILON {
            return Err(invalid(format!(
                "norm_eps: {epsilon}, another than the size's"
            )));
        }
        let served = [
            (
                config.norm_type == "ada_norm_zero",
                "norm_type: DiT modulates its norms by the time and class embedding",
            ),
            (
                config.activation_fn == "gelu-approximate",
                "activation_fn: DiT's feed-forward is tanh-approximated GELU",
            ),
            (
                !config.norm_elementwise_affine,
                "norm_elementwise_affine: DiT's norms hold no weights",
            ),
            (
                config.attention_bias,
                "attention_bias: DiT projects with biases",
            ),
            (
                !config.only_cross_attention,
                "only_cross_attention: DiT attends to its own rows",
            ),
            (
                !config.double_self_attention,
                "double_self_attention: DiT attends once a layer",
            ),
            (
                !config.use_linear_projection,
                "use_linear_projection: DiT embeds patches",
            ),
            (
                config.cross_attention_dim.is_none(),
                "cross_attention_dim: DiT attends to no condition rows",
            ),
            (
                config.num_vector_embeds.is_none(),
                "num_vector_embeds: DiT predicts continuous latents",
            ),
        ];
        match served.into_iter().find(|(served, _)| !served) {
            Some((_, reason)) => Err(unsupported(reason)),
            None => Ok(Size),
        }
    }
}

impl Family for Size<"XL-2-256"> {
    type Hf = Config;
}
