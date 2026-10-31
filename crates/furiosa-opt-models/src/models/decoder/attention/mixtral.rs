//! Mixtral, by published size: a size's [`Geometry`] is what its checkpoints declare, and its
//! [`Config`] refuses a value the size does not hold or a feature Mixtral does not serve.
//!
//! - Every size routes each row to 2 of 8 SwiGLU experts, the row's top two router probabilities
//!   renormalized.
//! - [`Model`] is the catalog's composition, over the [`DbrxForCausalLM`] template.

use crate::models::decoder::attention::{Chunks, Geometry, Qwen2ForCausalLM};
use crate::models::{Family, config, vocabulary};
use crate::modules::feedforward::moe::Normalize;
use crate::modules::feedforward::moe::Reduce;
use crate::modules::feedforward::moe::TopK;
use crate::modules::{attention, feedforward};
use crate::ops::attention::Attention;
use crate::ops::embedding;
use crate::ops::index::scatter::Scatter;
use crate::ops::pointwise::activation::Silu;
use crate::ops::pointwise::add;
use crate::ops::pointwise::multiply::Multiply;
use crate::ops::rotary::Rotary;
use crate::ops::swiglu;
use crate::ops::{
    attention::{geometry, gqa::Cache},
    linear::{Affine, Experts},
    normalization::RmsNorm,
};
use crate::specialize::attention::Strategy;
use crate::tensor::form::Weight;
use crate::tensor::mapping::Reshape;
use crate::tensor::memory::Memory;
use crate::tensor::{axis::Single, scalar::F32};
use crate::{Identity, Sequential, Then};
use furiosa_opt_std::prelude::*;
use serde::Deserialize;
use std::{io, marker::PhantomData};

pub type Model = DbrxForCausalLM<
    embedding::Embedding<Vocabulary, Hidden>,
    Qkv,
    Rotary<Head>,
    Scatter,
    (),
    Attend,
    Linear<QueryWidth, Hidden>,
    Router,
    ExpertMlp,
    Reduce,
    Norm,
    add::Add,
    Linear<Hidden, Vocabulary>,
    LAYERS,
>;

pub type Body = Model::Body;

type Qkv = attention::Projection<
    Projection<QueryWidth, Queries>,
    Projection<KvWidth, Heads>,
    Projection<KvWidth, Heads>,
>;

type Projection<Width, Count> = Sequential<Linear<Hidden, Width>, Reshape<m![Count, Head]>>;

type Linear<I, O> = Affine<HbmTensor<bf16, m![1], m![O, I]>, ()>;

type Attend = Attention<
    PhantomData<(Heads, Groups, Head)>,
    F32<{ geometry::scale(Head::SIZE) }>,
    { Strategy::Windows },
>;

type Router = Sequential<
    Then<Then<Reshape<m![Single, Hidden]>, Gate>, TopK<ExpertCount, Selected>>,
    (Identity, Normalize),
>;

type Gate = Affine<
    HbmTensor<bf16, m![1], m![ExpertCount, Hidden]>,
    [HbmTensor<bf16, m![1], m![ExpertCount]>; 0],
>;

type ExpertMlp = feedforward::ExpertMlp<
    Experts<Hidden, Intermediate, EXPERTS>,
    Silu,
    Experts<Hidden, Intermediate, EXPERTS>,
    Multiply,
    Experts<Intermediate, Hidden, EXPERTS>,
>;

type Norm = RmsNorm<Hidden, { 1e-5_f32.to_bits() as usize }>;

pub const LAYERS: usize = 32;
pub const EXPERTS: usize = 8;

pub type Hidden = geometry::Hidden<4096>;
pub type Intermediate = swiglu::Intermediate<14336>;
pub type QueryWidth = geometry::QueryWidth<4096>;
axes![KvWidth = 1024, Head = 128, KvHeads = 8];
pub type Heads = KvHeads;
pub type Groups = geometry::QueryGroups<4>;

axes![Queries = 32, Vocabulary = 32000];

#[derive(Clone, Debug)]
pub struct ExpertCount;

impl AxisName for ExpertCount {
    const NAME: Ident = Ident::new("ExpertCount");
    const SIZE: usize = EXPERTS;
}

axes![Selected = 2];

/// A published Mixtral checkpoint size, by its hub spelling.
pub struct Size<const N: &'static str>;

/// `rms_norm_eps` 1e-5, as the bits of the FP32 nearest to it.
const EPSILON: usize = 1e-5_f32.to_bits() as usize;

impl Geometry for Size<"8x7B"> {
    const SIZE: &'static str = "8x7B";
    const FAMILY: &'static str = "mixtral";
    type Queries = QueryWidth;
    type Hidden = Hidden;
    type Intermediate = Intermediate;
    type Vocabulary = Vocabulary;
    type KvHeads = KvHeads;
    type Groups = Groups;
    type Head = Head;
    type Layers<L> = [L; LAYERS];
    type Norm<W: AxisName, X: Memory> =
        RmsNorm<W, EPSILON, m![1], Weight<m![W], m![1], <X as Memory>::Tensor<bf16, m![1], m![W]>>>;

    const LAYERS: usize = LAYERS;
    const EPSILON: usize = EPSILON;
    const TIED_EMBEDDINGS: bool = false;
}

/// What a Mixtral checkpoint's config declares that its size fixes, and the features Mixtral does
/// not serve.
#[derive(Deserialize)]
pub struct Config {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    #[serde(default)]
    head_dim: Option<usize>,
    num_local_experts: usize,
    num_experts_per_tok: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    max_position_embeddings: usize,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    hidden_act: Option<String>,
    #[serde(default)]
    sliding_window: Option<usize>,
    #[serde(flatten, deserialize_with = "config::flattened")]
    _rotary: config::decoder::Rotary,
}

/// A config is of a size when it declares the size's values, at most its compiled vocabulary rows
/// and context, and none of the features Mixtral does not serve.
impl<const N: &'static str> TryFrom<Config> for Size<N>
where
    Self: Geometry,
{
    type Error = io::Error;

    fn try_from(config: Config) -> io::Result<Self> {
        let invalid = |reason: String| io::Error::new(io::ErrorKind::InvalidData, reason);
        let unsupported = |reason: &str| io::Error::new(io::ErrorKind::Unsupported, reason);
        let heads = <Self as Geometry>::KvHeads::SIZE;
        let head = <Self as Geometry>::Head::SIZE;
        let fixed = [
            (
                "hidden_size",
                config.hidden_size,
                <Self as Geometry>::Hidden::SIZE,
            ),
            (
                "intermediate_size",
                config.intermediate_size,
                <Self as Geometry>::Intermediate::SIZE,
            ),
            ("num_hidden_layers", config.num_hidden_layers, Self::LAYERS),
            ("num_key_value_heads", config.num_key_value_heads, heads),
            (
                "num_attention_heads",
                config.num_attention_heads,
                heads * <Self as Geometry>::Groups::SIZE,
            ),
            ("head_dim", config.head_dim.unwrap_or(head), head),
            (
                "num_local_experts",
                config.num_local_experts,
                ExpertCount::SIZE,
            ),
            (
                "num_experts_per_tok",
                config.num_experts_per_tok,
                Selected::SIZE,
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
        if (config.rms_norm_eps as f32).to_bits() as usize != Self::EPSILON {
            return Err(invalid(format!(
                "rms_norm_eps: {}, another than the size's",
                config.rms_norm_eps
            )));
        }
        if config.tie_word_embeddings != Self::TIED_EMBEDDINGS {
            return Err(invalid(format!(
                "tie_word_embeddings: {}, the size ties: {}",
                config.tie_word_embeddings,
                Self::TIED_EMBEDDINGS
            )));
        }
        let context = Chunks::SIZE * Cache::SIZE;
        if config.max_position_embeddings > context {
            return Err(unsupported(&format!(
                "max_position_embeddings: past the compiled context of {context}"
            )));
        }
        match (
            config.hidden_act.as_deref().unwrap_or("silu"),
            config.sliding_window,
        ) {
            ("silu", None) => Ok(Size),
            ("silu", Some(_)) => Err(unsupported(
                "sliding_window: Mixtral attends the whole history",
            )),
            (_, _) => Err(unsupported("hidden_act: Mixtral's experts gate with SiLU")),
        }
    }
}

impl<const N: &'static str> Family for Size<N>
where
    Self: Geometry,
{
    type Hf = Config;
}

/// Source: vLLM `vllm/model_executor/models/dbrx.py`.
pub type DbrxForCausalLM<
    Embedding,
    Qkv,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Router,
    Experts,
    Weighting,
    Norm,
    Add,
    Head,
    const DEPTH: usize,
> = Qwen2ForCausalLM<
    Embedding,
    Qkv,
    Position,
    Write,
    Read,
    Attend,
    Project,
    feedforward::Routed<Router, Experts, Weighting>,
    Norm,
    Add,
    Head,
    DEPTH,
>;
