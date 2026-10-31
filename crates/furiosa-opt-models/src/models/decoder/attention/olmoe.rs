//! OLMoE, by published size: a size's [`Geometry`] is what its checkpoints declare, and its
//! [`Config`] refuses a value the size does not hold or a feature OLMoE does not serve.
//!
//! - Every size normalizes its queries and keys over their whole projection width, and routes each
//!   row to 8 of 64 SwiGLU experts, weighed by their router probabilities as they are.
//! - [`Model`] is the catalog's composition, over the [`DbrxForCausalLM`] template.

use crate::models::decoder::attention::{Chunks, Geometry, mixtral::DbrxForCausalLM};
use crate::models::{Family, config};
use crate::modules::feedforward::moe::Reduce;
use crate::modules::feedforward::moe::TopK;
use crate::modules::{attention, feedforward};
use crate::ops::attention::Attention;
use crate::ops::embedding::Embedding;
use crate::ops::index::scatter::Scatter;
use crate::ops::pointwise::activation::Silu;
use crate::ops::pointwise::add::Add;
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
use crate::{Sequential, Then};
use furiosa_opt_std::prelude::*;
use serde::Deserialize;
use std::{io, marker::PhantomData};

pub type Model = DbrxForCausalLM<
    Embedding<Vocabulary, Hidden>,
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
    Add,
    HeadProjection,
    LAYERS,
>;

pub type Body = Model::Body;

type Qkv = attention::Projection<
    NormalizedProjection<QueryWidth, Queries>,
    NormalizedProjection<KvWidth, Heads>,
    Projection<KvWidth, Heads>,
>;

type NormalizedProjection<Width, Count> =
    Sequential<Then<Linear<Hidden, Width>, RmsNorm<Width, EPSILON>>, Reshape<m![Count, Head]>>;

type Linear<I, O> = Affine<HbmTensor<bf16, m![1], m![O, I]>, ()>;

type Projection<Width, Count> = Sequential<Linear<Hidden, Width>, Reshape<m![Count, Head]>>;

type Attend = Attention<
    PhantomData<(Heads, Groups, Head)>,
    F32<{ geometry::scale(Head::SIZE) }>,
    { Strategy::Windows },
>;

type Router = Then<Then<Reshape<m![Single, Hidden]>, Gate>, TopK<ExpertCount, Selected>>;

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

type Norm = RmsNorm<Hidden, EPSILON>;

type HeadProjection =
    Sequential<Then<Reshape<m![Single, Hidden]>, HeadMatrix>, Reshape<m![Vocabulary]>>;

type HeadMatrix = Affine<
    HbmTensor<bf16, m![1], m![Vocabulary, Hidden]>,
    [HbmTensor<bf16, m![1], m![Vocabulary]>; 0],
>;

pub const LAYERS: usize = 16;
pub const EXPERTS: usize = 64;

pub type Hidden = geometry::Hidden<2048>;
pub type Intermediate = swiglu::Intermediate<1024>;
pub type QueryWidth = geometry::QueryWidth<2048>;

axes![KvWidth = 2048, Head = 128];

#[derive(Clone, Debug)]
pub struct Heads;

impl AxisName for Heads {
    const NAME: Ident = Ident::new("KvHeads");
    const SIZE: usize = 16;
}

pub type Groups = geometry::QueryGroups<1>;

axes![Queries = 16, Vocabulary = 50304];

#[derive(Clone, Debug)]
pub struct ExpertCount;

impl AxisName for ExpertCount {
    const NAME: Ident = Ident::new("ExpertCount");
    const SIZE: usize = EXPERTS;
}

axes![Selected = 8];

/// A published OLMoE checkpoint size, by its hub spelling.
pub struct Size<const N: &'static str>;

/// `rms_norm_eps` 1e-5, as the bits of the FP32 nearest to it.
const EPSILON: usize = 1e-5_f32.to_bits() as usize;

impl Geometry for Size<"1B-7B"> {
    const SIZE: &'static str = "1B-7B";
    const FAMILY: &'static str = "olmoe";
    type Queries = QueryWidth;
    type Hidden = Hidden;
    type Intermediate = Intermediate;
    type Vocabulary = Vocabulary;
    type KvHeads = Heads;
    type Groups = Groups;
    type Head = Head;
    type Layers<L> = [L; LAYERS];
    type Norm<W: AxisName, X: Memory> =
        RmsNorm<W, EPSILON, m![1], Weight<m![W], m![1], <X as Memory>::Tensor<bf16, m![1], m![W]>>>;

    const LAYERS: usize = LAYERS;
    const EPSILON: usize = EPSILON;
    const TIED_EMBEDDINGS: bool = false;
}

/// What an OLMoE checkpoint's config declares that its size fixes, and the features OLMoE does not
/// serve.
#[derive(Deserialize)]
pub struct Config {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    #[serde(default)]
    head_dim: Option<usize>,
    num_experts: usize,
    num_experts_per_tok: usize,
    vocab_size: usize,
    /// Transformers reads an absent one as 1e-5.
    #[serde(default)]
    rms_norm_eps: Option<f64>,
    max_position_embeddings: usize,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    hidden_act: Option<String>,
    #[serde(default)]
    norm_topk_prob: bool,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    clip_qkv: Option<f64>,
    #[serde(flatten, deserialize_with = "config::flattened")]
    _rotary: config::decoder::Rotary,
}

/// A config is of a size when it declares the size's values, at most its compiled vocabulary rows
/// and context, and none of the features OLMoE does not serve.
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
            ("num_experts", config.num_experts, ExpertCount::SIZE),
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
        let rows = <Self as Geometry>::Vocabulary::SIZE;
        if !(1..=rows).contains(&config.vocab_size) {
            return Err(invalid(format!(
                "vocab_size: {}, the size compiles {rows} rows",
                config.vocab_size
            )));
        }
        let epsilon = config.rms_norm_eps.unwrap_or(1e-5);
        if (epsilon as f32).to_bits() as usize != Self::EPSILON {
            return Err(invalid(format!(
                "rms_norm_eps: {epsilon}, another than the size's"
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
        let features = [
            (
                config
                    .hidden_act
                    .as_deref()
                    .is_some_and(|act| act != "silu"),
                "hidden_act: OLMoE's experts gate with SiLU",
            ),
            (
                config.norm_topk_prob,
                "norm_topk_prob: OLMoE weighs the selected experts without renormalizing",
            ),
            (
                config.attention_bias,
                "attention_bias: OLMoE projects without biases",
            ),
            (
                config.clip_qkv.is_some(),
                "clip_qkv: OLMoE projects without clipping",
            ),
        ];
        match features.into_iter().find(|(declared, _)| *declared) {
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
