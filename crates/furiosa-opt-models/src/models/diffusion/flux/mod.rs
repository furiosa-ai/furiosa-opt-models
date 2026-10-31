//! FLUX.1, by published size: paired text and image streams, then one joint stream, over a
//! [`Size`], as `Transformer2DModel<Size<"schnell">, bf16, Parallel<1>, X>`.
//!
//! - A size's [`Geometry`] is what its checkpoints declare beyond the transformer all sizes share:
//!   `"dev"` embeds learned guidance, `"schnell"` does not.
use crate::models::Family;
use crate::modules::diffusion::{AdaptiveNorm, ScaleFirst, SplitGate, SplitGates, SplitModulation};
use crate::modules::{attention, diffusion, feedforward};
use crate::ops::attention::Attention;
use crate::ops::{
    attention::full,
    index::sequence,
    linear::Coefficients,
    pointwise::{
        activation::{NewGelu, Silu},
        add::Add,
        gated,
    },
    rotary,
};
use crate::tensor::image;
use crate::tensor::image::Rows;
use crate::{First, Identity, Module, Sequential, Then};
use furiosa_opt_std::prelude::*;
use pipeline::encode;
use serde::Deserialize;
use std::{io, marker::PhantomData};

pub use super::dit::Width;
use crate::{Each, Layers, Unrolled};

pub mod pipeline;

/// Latent velocity predictor with learned-guidance embedding `G`, or `()` without one.
pub type Model<I, T, G> = diffusion::Transformer<
    Embedding,
    Conditioning<G>,
    Body<I, T>,
    diffusion::Unpatch<
        diffusion::Head<
            Modulation<Width<6144>, SplitModulation<Width<3072>, ScaleFirst>>,
            AdaptiveNorm,
            Linear<Width<3072>, Latent>,
        >,
        First,
    >,
>;

/// The context holds every operand: both stages' attention position by its phases, and every
/// part's slots are units.
impl<I: AxisName, T: AxisName, G: Guidance, Env: Copy> Module<&Context<I, T, G>, Env, &mut ()>
    for Model<I, T, G>
{
    type Output = Sample<I>;

    fn forward(
        &self,
        device: &mut Device,
        _: Env,
        _: &mut (),
        context: &Context<I, T, G>,
    ) -> Self::Output {
        let attention = attention::Context {
            positions: &context.phases,
            write: (),
            read: (),
        };
        self.forward(
            device,
            (((&attention, ()), &attention), ()),
            &mut (
                Default::default(),
                Default::default(),
                (Default::default(), [Default::default(); _]),
                Default::default(),
            ),
            (
                &context.sample,
                &context.encoded,
                (&context.time, &context.guidance),
            ),
        )
    }
}

/// Projects latent patches and text tokens into their shared hidden width.
#[derive(DeviceSend)]
pub struct Embedding {
    /// Latent-patch feature projection.
    pub image: Linear<Latent, Width<3072>>,
    /// Token-level text projection.
    pub text: Linear<Joint, Width<3072>>,
}

impl<I: AxisName, T: AxisName, Env: Copy> Module<(&Sample<I>, &Encoded<T>), Env, &mut ()>
    for Embedding
{
    type Output = (Hidden<T>, Hidden<I>);

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        (image, text): (&Sample<I>, &Encoded<T>),
    ) -> Self::Output {
        let text = self.text.forward(&mut *device, env, &mut (), &text.tokens);
        let image = self.image.forward(device, env, &mut (), image);
        (text, image)
    }
}

type Linear<I, O> = Coefficients<bf16, I, O, 1>;

/// Adds time, optional learned guidance, and pooled text embeddings.
#[derive(DeviceSend)]
pub struct Conditioning<G> {
    /// Sinusoidal time embedding and learned projection.
    pub time: TimeEmbedding,
    /// The learned-guidance embedding, with the time embedding's scalar input convention.
    pub guidance: G,
    /// Pooled text projection into the conditioning width.
    pub text: Mlp<Pooled, Width<3072>, Silu>,
}

impl<T: AxisName, G: Guidance, Env: Copy> Module<((&Time, &G::Input), &Encoded<T>), Env, &mut ()>
    for Conditioning<G>
{
    type Output = Vector;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        ((time, guidance), text): ((&Time, &G::Input), &Encoded<T>),
    ) -> Vector {
        let time = self
            .time
            .forward(&mut *device, env, &mut Default::default(), time);
        let time = self.guidance.add(&mut *device, (time, guidance));
        let text = self
            .text
            .forward(&mut *device, env, &mut Default::default(), &text.pooled);
        Add.forward(device, env, &mut (), (&time, &text))
    }
}

pub type TimeEmbedding = Then<diffusion::Time<Width<256>>, Mlp<Width<256>, Width<3072>, Silu>>;

type Mlp<I, O, A> = feedforward::Mlp<Linear<I, Width<3072>>, A, Linear<Width<3072>, O>>;

/// A learned-guidance embedding added to the time embedding, or `()` for a checkpoint without one.
pub trait Guidance: DeviceSend {
    /// The guidance value the embedding reads, `()` when absent.
    type Input: DeviceSend;

    /// Whether a checkpoint with this embedding declares `guidance_embeds`.
    const PRESENT: bool;

    /// Adds the embedded guidance value to `time`.
    fn add(&self, device: &mut Device, input: (Vector, &Self::Input)) -> Vector;

    /// Applies `op` to a guidance value; an absent value stays absent.
    fn map<O: for<'x, 's> crate::Module<&'x Time, (), &'s mut (), Output = Time>>(
        op: &O,
        device: &mut Device,
        input: &Self::Input,
    ) -> Self::Input;
}

impl Guidance for () {
    type Input = ();

    const PRESENT: bool = false;

    fn add(&self, _: &mut Device, (time, ()): (Vector, &())) -> Vector {
        time
    }

    fn map<O: for<'x, 's> crate::Module<&'x Time, (), &'s mut (), Output = Time>>(
        _: &O,
        _: &mut Device,
        (): &(),
    ) {
    }
}

impl Guidance for TimeEmbedding {
    type Input = Time;

    const PRESENT: bool = true;

    fn add(&self, device: &mut Device, (time, input): (Vector, &Time)) -> Vector {
        let guidance = self.forward(&mut *device, (), &mut Default::default(), input);
        Add.forward(device, (), &mut (), (&time, &guidance))
    }

    fn map<O: for<'x, 's> crate::Module<&'x Time, (), &'s mut (), Output = Time>>(
        op: &O,
        device: &mut Device,
        input: &Time,
    ) -> Time {
        op.forward(device, (), &mut (), input)
    }
}

/// Dual-stream layers followed by joint-stream layers, retaining image rows only.
pub type Body<I, T> = diffusion::DualStream<
    [Dual<I, T>; LAYERS],
    Join,
    [Single; SINGLE_LAYERS],
    sequence::Suffix<T, I>,
>;

type Dual<I, T> = diffusion::Dual<
    diffusion::Stream<
        Modulation<Width<18432>, SplitGates<Width<3072>>>,
        AdaptiveNorm,
        FeedForward,
        gated::Add,
    >,
    diffusion::Stream<
        Modulation<Width<18432>, SplitGates<Width<3072>>>,
        AdaptiveNorm,
        FeedForward,
        gated::Add,
    >,
    attention::Joint<
        Projection,
        Projection,
        JoinQkv,
        rotary::Adjacent<Head>,
        Attend,
        sequence::Split<T, I>,
        Linear<Width<3072>, Width<3072>>,
        Linear<Width<3072>, Width<3072>>,
    >,
>;

type Modulation<O, Split> = Then<Silu, Then<Linear<Width<3072>, O>, Split>>;

pub type FeedForward =
    feedforward::Mlp<Linear<Width<3072>, Width<12288>>, NewGelu, Linear<Width<12288>, Width<3072>>>;

pub type Projection = attention::NormalizedQkv<
    attention::Projection<Linear<Width<3072>, Width<3072>>>,
    full::HeadNorm<Head>,
    full::HeadNorm<Head>,
>;

/// Concatenates matching query, key, and value streams along their row axes.
#[derive(Default, DeviceSend)]
pub struct JoinQkv;

impl<A: AxisName, B: AxisName, Env: Copy> Module<(&Qkv<A>, &Qkv<B>), Env, &mut ()> for JoinQkv {
    type Output = Qkv<sequence::Sum<A, B>>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        (first, second): (&Qkv<A>, &Qkv<B>),
    ) -> Self::Output {
        let query = sequence::Concat.forward(&mut *device, env, &mut (), (&first.0, &second.0));
        let key = sequence::Concat.forward(&mut *device, env, &mut (), (&first.1, &second.1));
        let value = sequence::Concat.forward(device, env, &mut (), (&first.2, &second.2));
        (query, key, value)
    }
}

type Attend = Attention<PhantomData<Heads>, Time>;

type Single = diffusion::Single<
    Modulation<Width<9216>, SplitGate<Width<3072>>>,
    AdaptiveNorm,
    attention::Full<Sequential<(Projection, Identity), rotary::Adjacent<Head>>, Attend, ()>,
    Then<Linear<Width<3072>, Width<12288>>, NewGelu>,
    sequence::Columns<Width<15360>>,
    Linear<Width<15360>, Width<3072>>,
    gated::Add,
>;

/// Joins text and image hidden rows, in that order, for joint-stream layers.
#[derive(Default, DeviceSend)]
pub struct Join;

impl<A: AxisName, B: AxisName, Env: Copy> Module<&(Hidden<A>, Hidden<B>), Env, &mut ()> for Join {
    type Output = (Hidden<sequence::Sum<A, B>>, ());

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        (first, second): &(Hidden<A>, Hidden<B>),
    ) -> Self::Output {
        (
            sequence::Concat.forward(device, env, &mut (), (first, second)),
            (),
        )
    }
}

/// Training timestep or learned-guidance value supplied to an embedding.
pub type Time = HbmTensor<f32, m![1], m![1]>;

/// Packed 2×2 latent patches with 64 features per row.
pub type Sample<R> = Rows<Latent, R>;

/// Rotary phases for text rows followed by image rows.
pub type Phases<I, T> = rotary::Phases<sequence::Sum<T, I>, Head>;

type Hidden<R> = Rows<Width<3072>, R>;

/// One conditioning vector in the transformer's hidden width.
pub type Vector = image::Vector<Width<3072>>;
type Qkv<R> = (Hidden<R>, Hidden<R>, Hidden<R>);

/// Token-level and pooled text conditioning from separate encoders.
#[derive(DeviceSend)]
pub struct Encoded<T: AxisName> {
    /// T5 token features, including padded rows required by the compiled shape.
    pub tokens: Rows<Joint, T>,
    /// CLIP pooled text vector.
    pub pooled: image::Vector<Pooled>,
}

/// Latent patches, text features, conditioning scalars, and joint rotary phases.
#[derive(DeviceSend)]
pub struct Context<I: AxisName, T: AxisName, G: Guidance> {
    /// Current packed latent patches.
    pub sample: Sample<I>,
    /// Token-level and pooled text features.
    pub encoded: Encoded<T>,
    /// Training timestep rather than an unscaled scheduler sigma.
    pub time: Time,
    /// The learned-guidance input, independent of classifier-free guidance.
    pub guidance: G::Input,
    /// Position phases in text-then-image row order.
    pub phases: Phases<I, T>,
}

/// Features of a packed latent patch: `in_channels`, and the prediction's `out_channels`.
type Latent = Width<64>;

/// Features of a T5 token: `joint_attention_dim`.
type Joint = Width<4096>;

/// Features of the pooled CLIP vector: `pooled_projection_dim`.
type Pooled = Width<768>;

/// Attention heads: `num_attention_heads`.
type Heads = Width<24>;

/// Features of a head: `attention_head_dim`.
type Head = Width<128>;

/// Paired text and image layers: `num_layers`.
const LAYERS: usize = 19;

/// Joint layers: `num_single_layers`.
const SINGLE_LAYERS: usize = 38;

/// Features of a head the stream, row and column coordinates rotate: `axes_dims_rope`.
const AXES: [usize; 3] = [16, 56, 56];

/// A published FLUX.1 size, by the hub's variant name.
pub struct Size<const N: &'static str>;

/// What a FLUX.1 size fixes beyond the transformer every size shares.
pub trait Geometry {
    /// The learned-guidance embedding of `guidance_embeds`.
    type Guidance: Guidance;

    /// The hub spelling of the size, as `"schnell"`.
    const SIZE: &'static str;
}

impl Geometry for Size<"dev"> {
    type Guidance = TimeEmbedding;

    const SIZE: &'static str = "dev";
}

impl Geometry for Size<"schnell"> {
    type Guidance = ();

    const SIZE: &'static str = "schnell";
}

/// What a FLUX.1 transformer's config declares that its size fixes, and the features FLUX.1 does
/// not serve.
#[derive(Deserialize)]
pub struct Config {
    patch_size: usize,
    in_channels: usize,
    #[serde(default)]
    out_channels: Option<usize>,
    num_layers: usize,
    num_single_layers: usize,
    attention_head_dim: usize,
    num_attention_heads: usize,
    joint_attention_dim: usize,
    pooled_projection_dim: usize,
    guidance_embeds: bool,
    #[serde(default)]
    axes_dims_rope: Option<Vec<usize>>,
    #[serde(default)]
    quantization_config: Option<serde_json::Value>,
}

/// A config is of a size when it declares the size's values, Diffusers' defaults standing for the
/// keys it omits, and none of the features FLUX.1 does not serve.
impl<const N: &'static str> TryFrom<Config> for Size<N>
where
    Self: Geometry,
{
    type Error = io::Error;

    fn try_from(config: Config) -> io::Result<Self> {
        let unsupported = |reason: String| io::Error::new(io::ErrorKind::Unsupported, reason);
        let fixed = [
            ("patch_size", config.patch_size, 1),
            ("in_channels", config.in_channels, Latent::SIZE),
            (
                "out_channels",
                config.out_channels.unwrap_or(config.in_channels),
                Latent::SIZE,
            ),
            ("num_layers", config.num_layers, LAYERS),
            ("num_single_layers", config.num_single_layers, SINGLE_LAYERS),
            ("attention_head_dim", config.attention_head_dim, Head::SIZE),
            (
                "num_attention_heads",
                config.num_attention_heads,
                Heads::SIZE,
            ),
            (
                "joint_attention_dim",
                config.joint_attention_dim,
                Joint::SIZE,
            ),
            (
                "pooled_projection_dim",
                config.pooled_projection_dim,
                Pooled::SIZE,
            ),
        ];
        if let Some((key, declared, held)) = fixed
            .into_iter()
            .find(|(_, declared, held)| declared != held)
        {
            return Err(unsupported(format!(
                "{key}: {declared}, the size {N} holds {held}"
            )));
        }
        let axes = config.axes_dims_rope.unwrap_or_else(|| AXES.into());
        if axes != AXES {
            return Err(unsupported(format!(
                "axes_dims_rope: {axes:?}, the size {N} rotates {AXES:?}"
            )));
        }
        let guidance = <<Self as Geometry>::Guidance as Guidance>::PRESENT;
        if config.guidance_embeds != guidance {
            return Err(unsupported(format!(
                "guidance_embeds: {}, the size {N} embeds guidance: {guidance}",
                config.guidance_embeds
            )));
        }
        match config.quantization_config {
            Some(_) => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "quantization_config: FLUX.1 computes on BF16 weights",
            )),
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

/// Source: SGLang `python/sglang/multimodal_gen/runtime/models/dits/flux_2.py`.
pub type Flux2Transformer2DModel<
    Embedding,
    Conditioning,
    ImageModulation,
    TextModulation,
    ImageQkv,
    ImageQueryNorm,
    ImageKeyNorm,
    TextQkv,
    TextQueryNorm,
    TextKeyNorm,
    JoinQkv,
    Position,
    Attend,
    Split,
    ImageProjection,
    TextProjection,
    ImageMlp,
    TextMlp,
    ImageNorm,
    TextNorm,
    ImageGate,
    TextGate,
    JoinTokens,
    SingleModulation,
    SingleNorm,
    SingleQkv,
    SingleQueryNorm,
    SingleKeyNorm,
    SingleAttend,
    SingleUp,
    SingleActivation,
    SingleJoin,
    SingleProjection,
    SingleGate,
    SplitTokens,
    OutputModulation,
    OutputNorm,
    OutputProjection,
    Restore,
    const DUAL_DEPTH: usize,
    const SINGLE_DEPTH: usize,
> = diffusion::Transformer<
    Embedding,
    Conditioning,
    diffusion::DualStream<
        Layers<
            [diffusion::Dual<
                diffusion::Stream<ImageModulation, ImageNorm, ImageMlp, ImageGate>,
                diffusion::Stream<TextModulation, TextNorm, TextMlp, TextGate>,
                attention::Joint<
                    attention::NormalizedQkv<ImageQkv, ImageQueryNorm, ImageKeyNorm>,
                    attention::NormalizedQkv<TextQkv, TextQueryNorm, TextKeyNorm>,
                    JoinQkv,
                    Position,
                    Attend,
                    Split,
                    ImageProjection,
                    TextProjection,
                >,
            >; DUAL_DEPTH],
            Unrolled,
        >,
        JoinTokens,
        Layers<
            [diffusion::Single<
                SingleModulation,
                SingleNorm,
                attention::Full<
                    Sequential<
                        (
                            attention::NormalizedQkv<SingleQkv, SingleQueryNorm, SingleKeyNorm>,
                            crate::Identity,
                        ),
                        Position,
                    >,
                    SingleAttend,
                    (),
                >,
                crate::Then<SingleUp, SingleActivation>,
                SingleJoin,
                SingleProjection,
                SingleGate,
            >; SINGLE_DEPTH],
            Unrolled,
        >,
        SplitTokens,
    >,
    diffusion::Unpatch<diffusion::Head<OutputModulation, OutputNorm, OutputProjection>, Restore>,
>;

/// Source: vLLM-Omni `vllm_omni/diffusion/models/flux/pipeline_flux.py`.
/// CLIP pooled features, T5 token features and rotary phases condition the denoiser.
pub type FluxPipeline<Clip, T5, Rotary, Prepare, Predict, Update, Decode> = diffusion::Pipeline<
    Each<encode::Model<Clip, T5, Rotary>>,
    Prepare,
    diffusion::Sampler<Predict, Update>,
    Decode,
>;
