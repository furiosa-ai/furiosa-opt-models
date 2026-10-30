//! Qwen3-VL, by published size: its text model is an attention decoder over a [`Size`], whose
//! [`Geometry`] is what the checkpoint's `text_config` declares.
//!
//! - A text-only prompt gives every interleaved rotation section the token's index, so the text
//!   model rotates as one rotation over the whole head.

use std::io;

use serde::Deserialize;

use crate::models::Family;
use crate::models::config::{self, decoder::Interleaved};
use crate::models::decoder::attention::{Geometry, Head, KvHeads, PaddedVocabulary};
use crate::modules::{attention, language, multimodal, transformer};
use crate::ops::attention::geometry::{self, QueryGroups, QueryWidth};
use crate::ops::normalization::RmsNorm;
use crate::ops::swiglu;
use crate::tensor::form::Weight;
use crate::tensor::memory::Memory;
use furiosa_opt_std::prelude::*;

/// A published Qwen3-VL checkpoint size, by its hub spelling.
pub struct Size<const N: &'static str>;

/// The norm of every Qwen3-VL size's text model: `rms_norm_eps` 1e-6.
type Norm<W, X> =
    RmsNorm<W, { EPSILON }, m![1], Weight<m![W], m![1], <X as Memory>::Tensor<bf16, m![1], m![W]>>>;

const EPSILON: usize = 1e-6_f32.to_bits() as usize;

impl Geometry for Size<"32B"> {
    const SIZE: &'static str = "32B";
    const FAMILY: &'static str = "qwen3_vl";
    type Queries = QueryWidth<8192>;
    type Hidden = geometry::Hidden<5120>;
    type Intermediate = swiglu::Intermediate<25600>;
    type Vocabulary = PaddedVocabulary;
    type KvHeads = KvHeads;
    type Groups = QueryGroups<8>;
    type Head = Head;
    type Layers<L> = [L; Self::LAYERS];
    type Norm<W: AxisName, X: Memory> = Norm<W, X>;

    const LAYERS: usize = 64;
    const EPSILON: usize = EPSILON;
    const TIED_EMBEDDINGS: bool = false;
}

/// What a Qwen3-VL checkpoint's config declares that its size fixes, and the features its text
/// model does not serve.
#[derive(Deserialize)]
pub struct Config {
    text_config: Text,
    #[serde(default)]
    tie_word_embeddings: bool,
}

/// The text model's part of a Qwen3-VL config.
#[derive(Deserialize)]
struct Text {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    #[serde(flatten, deserialize_with = "config::flattened")]
    _rotary: Interleaved<Head>,
    #[serde(default)]
    attention_bias: bool,
}

/// A config is of a size when its text model declares the size's values and at most its compiled
/// vocabulary rows, and none of the features the text model does not serve.
impl<const N: &'static str> TryFrom<Config> for Size<N>
where
    Self: Geometry,
{
    type Error = io::Error;

    fn try_from(config: Config) -> io::Result<Self> {
        let invalid = |reason: String| io::Error::new(io::ErrorKind::InvalidData, reason);
        let unsupported = |reason: &str| io::Error::new(io::ErrorKind::Unsupported, reason);
        let text = config.text_config;
        let heads = <Self as Geometry>::KvHeads::SIZE;
        let fixed = [
            (
                "hidden_size",
                text.hidden_size,
                <Self as Geometry>::Hidden::SIZE,
            ),
            (
                "intermediate_size",
                text.intermediate_size,
                <Self as Geometry>::Intermediate::SIZE,
            ),
            ("num_hidden_layers", text.num_hidden_layers, Self::LAYERS),
            ("num_key_value_heads", text.num_key_value_heads, heads),
            (
                "num_attention_heads",
                text.num_attention_heads,
                heads * <Self as Geometry>::Groups::SIZE,
            ),
            ("head_dim", text.head_dim, <Self as Geometry>::Head::SIZE),
        ];
        if let Some((key, declared, held)) = fixed
            .into_iter()
            .find(|(_, declared, held)| declared != held)
        {
            return Err(invalid(format!(
                "text_config.{key}: {declared}, the size {N} holds {held}"
            )));
        }
        let rows = <Self as Geometry>::Vocabulary::SIZE;
        if !(1..=rows).contains(&text.vocab_size) {
            return Err(invalid(format!(
                "text_config.vocab_size: {}, the size compiles {rows} rows",
                text.vocab_size
            )));
        }
        if (text.rms_norm_eps as f32).to_bits() as usize != Self::EPSILON {
            return Err(invalid(format!(
                "text_config.rms_norm_eps: {}, another than the size's",
                text.rms_norm_eps
            )));
        }
        if config.tie_word_embeddings != Self::TIED_EMBEDDINGS {
            return Err(invalid(format!(
                "tie_word_embeddings: {}, the size ties: {}",
                config.tie_word_embeddings,
                Self::TIED_EMBEDDINGS
            )));
        }
        match text.attention_bias {
            true => Err(unsupported(
                "text_config.attention_bias: the text model projects without biases",
            )),
            false => Ok(Size),
        }
    }
}

impl<const N: &'static str> Family for Size<N>
where
    Self: Geometry,
{
    type Hf = Config;
}

/// Source: vLLM `vllm/model_executor/models/qwen3_vl.py`.
pub type Qwen3VLForConditionalGeneration<
    Embedding,
    VisionEncoder,
    VideoEncoder,
    ProjectMedia,
    MergeMedia,
    Inject,
    Qkv,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Head,
    const DEEP_LAYERS: usize,
    const TEXT_TAIL: usize,
> = language::Readout<
    language::Normalized<
        multimodal::DeepFusion<
            Embedding,
            (VisionEncoder, VideoEncoder),
            ProjectMedia,
            MergeMedia,
            transformer::PreNorm<
                attention::Cached<Qkv, Position, Write, Read, Attend, Project>,
                Mlp,
                Norm,
                Add,
            >,
            Inject,
            DEEP_LAYERS,
            TEXT_TAIL,
        >,
        Norm,
    >,
    Head,
>;
