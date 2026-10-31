//! Whisper, by published size: a [`Size`] fixes what its checkpoints declare, and the catalog
//! composition below the sizes encodes a spectrogram and decodes tokens against it.
//!
//! - Checkpoint names follow Transformers; framework audio preprocessing stays outside this model.
//! - The vocabulary projection `proj_out` is the decoder's token table: checkpoints do not store
//!   it.
//! - Every size normalizes with a 1e-5 epsilon, runs a GELU feed-forward and embeds tokens
//!   unscaled.

use super::CrossDecoder;
use crate::models::{Family, vocabulary};
pub use crate::models::{Ids, Scalar};
use crate::modules::{attention, feedforward, language, multimodal, transformer};
use crate::ops::attention::Attention;
use crate::ops::attention::geometry;
pub use crate::ops::attention::geometry::Vocabulary;
use crate::ops::swiglu;
use crate::ops::{
    convolution::{Bias, Convolution, Temporal, Transpose},
    embedding::Lookup,
    index::{gather::Gather, scatter::Scatter},
    linear::Affine,
    normalization::LayerNorm,
    pointwise::{activation::Gelu, add::Add, multiply::Scale},
};
use crate::{First, Fork, Then};
use furiosa_opt_std::prelude::*;
use serde::Deserialize;
use std::{io, marker::PhantomData};

pub mod decoder;
pub mod encoder;

/// Spectrogram encoder with conditional token generation: the env is the step's self and source
/// masks, a decoder layer's slot its self and cross attentions' state.
pub type Model<
    V,
    W,
    M,
    S,
    P,
    EncoderIntermediate,
    DecoderIntermediate,
    EncoderHeads,
    DecoderHeads,
    A,
    const E: usize,
    const D: usize,
> = multimodal::EncoderDecoder<
    Encoder<M, W, S, EncoderIntermediate, EncoderHeads, A, E>,
    Decoder<V, W, P, DecoderIntermediate, DecoderHeads, A, D>,
>;

/// Convolutional spectrogram embedding followed by unmasked encoder layers.
pub type Encoder<M, W, S, I, H, A, const N: usize> = encoder::Encoder<
    language::ConditionedEmbedding<Frontend<M, W>, Embedding<S, W>, Add>,
    [EncoderLayer<W, I, H, A>; N],
    Norm<W>,
>;

type Frontend<M, W> = Then<
    Then<
        Then<
            Then<Convolution<Temporal<M, W, Kernel>, Bias<W, m![1]>>, Gelu>,
            Convolution<Temporal<W, W, Kernel>, Bias<W, m![1]>, 2>,
        >,
        Gelu,
    >,
    Transpose,
>;

type Embedding<Count, Width> = Lookup<HbmTensor<f32, m![1], m![Count, Width]>>;

type EncoderLayer<W, I, H, A> = transformer::PreNorm<
    attention::Full<Projection<W>, Attend<H>, Linear<W, W>>,
    Mlp<W, I, A>,
    Norm<W>,
    Add,
>;

pub type Projection<W> = attention::Projection<Query<W>, Linear<W, W, 0>, Linear<W, W>>;

type Query<W> = Then<Linear<W, W>, Scale>;

/// F32 feature projection with zero or one bias vector.
pub type Linear<In, Out, const BIAS: usize = 1> =
    Affine<HbmTensor<f32, m![1], m![Out, In]>, [HbmTensor<f32, m![1], m![Out]>; BIAS]>;

type Attend<Heads> = Attention<PhantomData<Heads>, Scalar>;

pub type Mlp<W, I, A> = feedforward::Mlp<Linear<W, I>, A, Linear<I, W>>;

/// Per-row layer normalization with placed affine parameters and epsilon.
pub type Norm<Width> = LayerNorm<HbmTensor<f32, m![1], m![Width]>, Scalar>;

/// Normalized causal decoder with a bias-free vocabulary head.
pub type Decoder<V, W, P, I, H, A, const N: usize> =
    language::Readout<DecoderBody<V, W, P, I, H, A, N>, Linear<W, V, 0>>;

/// Position-embedded causal decoder reading projected spectrogram memory.
pub type DecoderBody<V, W, P, I, H, A, const N: usize> = decoder::Body<
    language::Normalized<
        language::Backbone<
            language::ConditionedEmbedding<Embedding<V, W>, Embedding<P, W>, Add>,
            [DecoderLayer<W, I, H, A>; N],
        >,
        Norm<W>,
    >,
>;

type DecoderLayer<W, I, H, A> =
    transformer::CrossLayer<SelfAttention<W, H>, CrossAttention<W, H>, Mlp<W, I, A>, Norm<W>, Add>;

type SelfAttention<W, H> =
    attention::Cached<Projection<W>, First, Scatter, Gather, Attend<H>, Linear<W, W>>;

type CrossAttention<W, H> = attention::Cross<
    Query<W>,
    Fork<(Linear<W, W, 0>, Linear<W, W>)>,
    Scatter,
    Gather,
    Attend<H>,
    Linear<W, W>,
>;

/// Frame- or token-major F32 feature rows.
pub type Rows<Width, Count> = HbmTensor<f32, m![1], m![Count, Width]>;

/// Additive query/key attention mask.
pub type Mask<Queries, Keys> = HbmTensor<f32, m![1], m![Queries, Keys]>;

axes![Kernel = 3];

/// A published Whisper checkpoint size, by its hub spelling: `"tiny"` is openai/whisper-tiny,
/// `"tiny.en"` its English-only vocabulary.
pub struct Size<const N: &'static str>;

/// A published size of Whisper: its axes and the config values its type fixes.
pub trait Geometry {
    /// Features of a row: `d_model`.
    type Hidden: AxisName;
    /// Features of the feed-forward: `encoder_ffn_dim` and `decoder_ffn_dim`.
    type Intermediate: AxisName;
    /// Attention heads of a layer: `encoder_attention_heads` and `decoder_attention_heads`.
    type Heads: AxisName;
    /// Rows of the decoder's token table and of the head: `vocab_size`.
    type Vocabulary: AxisName;
    /// Channels of a spectrogram frame: `num_mel_bins`.
    type Mels: AxisName;
    /// Rows of the encoder's position table, one a frame after the strided convolution:
    /// `max_source_positions`.
    type Frames: AxisName;
    /// Rows of the decoder's position table: `max_target_positions`.
    type Positions: AxisName;

    /// The hub spelling of the size, as `"tiny.en"`.
    const SIZE: &'static str;
    /// `encoder_layers`.
    const LAYERS: usize;
    /// `decoder_layers`.
    const DECODER_LAYERS: usize;
}

axes![Heads = 6, Mels = 80, Frames = 1500, Positions = 448];

impl Geometry for Size<"tiny"> {
    type Hidden = geometry::Hidden<384>;
    type Intermediate = swiglu::Intermediate<1536>;
    type Heads = Heads;
    type Vocabulary = Vocabulary<51865>;
    type Mels = Mels;
    type Frames = Frames;
    type Positions = Positions;

    const SIZE: &'static str = "tiny";
    const LAYERS: usize = 4;
    const DECODER_LAYERS: usize = 4;
}

impl Geometry for Size<"tiny.en"> {
    type Hidden = geometry::Hidden<384>;
    type Intermediate = swiglu::Intermediate<1536>;
    type Heads = Heads;
    type Vocabulary = Vocabulary<51864>;
    type Mels = Mels;
    type Frames = Frames;
    type Positions = Positions;

    const SIZE: &'static str = "tiny.en";
    const LAYERS: usize = 4;
    const DECODER_LAYERS: usize = 4;
}

/// What a Whisper checkpoint's config declares that its size fixes, and the features Whisper does
/// not serve; an absent key holds `WhisperConfig`'s default, whisper-tiny's.
#[derive(Deserialize)]
#[serde(default)]
pub struct Config {
    model_type: String,
    d_model: usize,
    encoder_layers: usize,
    decoder_layers: usize,
    encoder_attention_heads: usize,
    decoder_attention_heads: usize,
    encoder_ffn_dim: usize,
    decoder_ffn_dim: usize,
    num_mel_bins: usize,
    max_source_positions: usize,
    max_target_positions: usize,
    vocab_size: usize,
    tie_word_embeddings: bool,
    activation_function: String,
    scale_embedding: bool,
}

/// `WhisperConfig`'s defaults; a config without its `model_type` names none.
impl Default for Config {
    fn default() -> Self {
        Self {
            model_type: String::new(),
            d_model: 384,
            encoder_layers: 4,
            decoder_layers: 4,
            encoder_attention_heads: 6,
            decoder_attention_heads: 6,
            encoder_ffn_dim: 1536,
            decoder_ffn_dim: 1536,
            num_mel_bins: 80,
            max_source_positions: 1500,
            max_target_positions: 448,
            vocab_size: 51865,
            tie_word_embeddings: true,
            activation_function: "gelu".into(),
            scale_embedding: false,
        }
    }
}

/// A config is of a size when it declares the size's values, a head tied to the token table and
/// none of the features Whisper does not serve.
impl<const N: &'static str> TryFrom<Config> for Size<N>
where
    Self: Geometry,
{
    type Error = io::Error;

    fn try_from(config: Config) -> io::Result<Self> {
        let invalid = |reason: String| io::Error::new(io::ErrorKind::InvalidData, reason);
        let unsupported = |reason: &str| io::Error::new(io::ErrorKind::Unsupported, reason);
        if config.model_type != "whisper" {
            return Err(invalid(format!(
                "model_type: {}, the size {N} is whisper",
                config.model_type
            )));
        }
        let fixed = [
            ("d_model", config.d_model, <Self as Geometry>::Hidden::SIZE),
            ("encoder_layers", config.encoder_layers, Self::LAYERS),
            (
                "decoder_layers",
                config.decoder_layers,
                Self::DECODER_LAYERS,
            ),
            (
                "encoder_attention_heads",
                config.encoder_attention_heads,
                <Self as Geometry>::Heads::SIZE,
            ),
            (
                "decoder_attention_heads",
                config.decoder_attention_heads,
                <Self as Geometry>::Heads::SIZE,
            ),
            (
                "encoder_ffn_dim",
                config.encoder_ffn_dim,
                <Self as Geometry>::Intermediate::SIZE,
            ),
            (
                "decoder_ffn_dim",
                config.decoder_ffn_dim,
                <Self as Geometry>::Intermediate::SIZE,
            ),
            (
                "num_mel_bins",
                config.num_mel_bins,
                <Self as Geometry>::Mels::SIZE,
            ),
            (
                "max_source_positions",
                config.max_source_positions,
                <Self as Geometry>::Frames::SIZE,
            ),
            (
                "max_target_positions",
                config.max_target_positions,
                <Self as Geometry>::Positions::SIZE,
            ),
        ];
        if let Some((key, declared, held)) = fixed
            .into_iter()
            .find(|(_, declared, held)| declared != held)
        {
            return Err(invalid(format!(
                "{key}: {declared}, the size {N} holds {held}"
            )));
        }
        vocabulary::<<Self as Geometry>::Vocabulary>(config.vocab_size)?;
        if !config.tie_word_embeddings {
            return Err(invalid(
                "tie_word_embeddings: false, the size ties: true".into(),
            ));
        }
        match (config.activation_function.as_str(), config.scale_embedding) {
            ("gelu", false) => Ok(Size),
            ("gelu", true) => Err(unsupported(
                "scale_embedding: Whisper embeds tokens unscaled",
            )),
            (_, _) => Err(unsupported(
                "activation_function: Whisper's feed-forward is GELU",
            )),
        }
    }
}

impl<const N: &'static str> Family for Size<N>
where
    Self: Geometry,
{
    type Hf = Config;
}

/// Source: vLLM `vllm/model_executor/models/whisper.py`.
pub type WhisperForConditionalGeneration<
    EncoderSpectrogram,
    EncoderQkv,
    EncoderAttend,
    EncoderProject,
    EncoderMlp,
    EncoderNorm,
    EncoderAdd,
    EncoderHead,
    Embedding,
    Qkv,
    Position,
    Write,
    Read,
    Attend,
    Project,
    CrossQuery,
    CrossKeyValue,
    CrossWrite,
    CrossRead,
    CrossAttend,
    CrossProject,
    Mlp,
    Norm,
    Add,
    Head,
    const ENCODER_DEPTH: usize,
    const DECODER_DEPTH: usize,
> = multimodal::EncoderDecoder<
    language::Encoder<
        EncoderSpectrogram,
        [transformer::PreNorm<
            attention::Full<EncoderQkv, EncoderAttend, EncoderProject>,
            EncoderMlp,
            EncoderNorm,
            EncoderAdd,
        >; ENCODER_DEPTH],
        EncoderHead,
    >,
    CrossDecoder<
        Embedding,
        transformer::CrossLayer<
            attention::Cached<Qkv, Position, Write, Read, Attend, Project>,
            attention::Cross<
                CrossQuery,
                CrossKeyValue,
                CrossWrite,
                CrossRead,
                CrossAttend,
                CrossProject,
            >,
            Mlp,
            Norm,
            Add,
        >,
        Norm,
        Head,
        DECODER_DEPTH,
    >,
>;
