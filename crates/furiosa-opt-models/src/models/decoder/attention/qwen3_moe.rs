//! Qwen3 MoE, by published size: a size's [`Geometry`] is what its checkpoints declare, and its
//! [`Config`] refuses a value the size does not hold or a feature Qwen3 MoE does not serve.
//!
//! - Every layer is sparse: a size's `Intermediate` is its experts' width, and every size routes
//!   each row to 8 of 128 SwiGLU experts, the row's top eight router probabilities renormalized.
//! - Every size normalizes its queries and keys head by head, over 128-feature heads.
//! - [`Model`] and [`Body`] are the catalog's composition, over the [`Qwen3Moe`] template.

use crate::models::decoder::attention::{Geometry, mixtral::DbrxForCausalLM};
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
use crate::ops::{
    attention::geometry,
    linear::{Affine, Experts},
    normalization::RmsNorm,
    swiglu,
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

pub type Model<
    H = Hidden,
    I = Intermediate,
    Q = Queries,
    K = Heads,
    V = Vocabulary,
    const N: usize = LAYERS,
    const E: usize = EXPERTS,
    const S: usize = 8,
> = Qwen3Moe<
    embedding::Embedding<V, H>,
    Qkv<H, Q, K>,
    RmsNorm<Head>,
    RmsNorm<Head>,
    Rotary<Head>,
    Scatter,
    (),
    Attend<Q, K>,
    Linear<QueryWidth<Q>, H>,
    Router<H, E, S>,
    ExpertMlp<H, I, E>,
    Reduce,
    Norm<H>,
    add::Add,
    HeadProjection<H, V>,
    N,
>;

pub type Body<
    H = Hidden,
    I = Intermediate,
    Q = Queries,
    K = Heads,
    V = Vocabulary,
    const N: usize = LAYERS,
    const E: usize = EXPERTS,
    const S: usize = 8,
> = Model<H, I, Q, K, V, N, E, S>::Body;

type Qkv<H, Q, K> = attention::Projection<
    Projection<H, QueryWidth<Q>, Q>,
    Projection<H, KvWidth<K>, K>,
    Projection<H, KvWidth<K>, K>,
>;

type Projection<H, Width, Count> = Sequential<Linear<H, Width>, Reshape<m![Count, Head]>>;

type Linear<I, O> = Affine<HbmTensor<bf16, m![1], m![O, I]>, ()>;

type Attend<Q, K> = Attention<
    PhantomData<(K, Groups<Q, K>, Head)>,
    F32<{ geometry::scale(Head::SIZE) }>,
    { Strategy::Windows },
>;

type Router<H, const E: usize, const S: usize> = Sequential<
    Then<Then<Reshape<m![Single, H]>, Gate<H, E>>, TopK<ExpertCount<E>, Selected<S>>>,
    (Identity, Normalize),
>;

type Gate<H, const E: usize> = Affine<
    HbmTensor<bf16, m![1], m![{Symbol<ExpertCount<E>>}, H]>,
    [HbmTensor<bf16, m![1], m![{Symbol<ExpertCount<E>>}]>; 0],
>;

type ExpertMlp<H, I, const E: usize> =
    feedforward::ExpertMlp<Experts<H, I, E>, Silu, Experts<H, I, E>, Multiply, Experts<I, H, E>>;

type Norm<H> = RmsNorm<H>;

type HeadProjection<H, V> =
    Sequential<Then<Reshape<m![Single, H]>, HeadMatrix<H, V>>, Reshape<m![V]>>;

type HeadMatrix<H, V> =
    Affine<HbmTensor<bf16, m![1], m![V, H]>, [HbmTensor<bf16, m![1], m![V]>; 0]>;

pub const LAYERS: usize = 48;
pub const EXPERTS: usize = 128;

pub type Hidden = geometry::Hidden<2048>;
pub type Intermediate = swiglu::Intermediate<768>;

#[derive(Clone, Debug)]
pub struct QueryWidth<Q = Queries>(PhantomData<Q>);

impl<Q: AxisName> AxisName for QueryWidth<Q> {
    const NAME: Ident = Ident::new("QueryWidth");
    const SIZE: usize = Q::SIZE * Head::SIZE;
}

#[derive(Clone, Debug)]
pub struct KvWidth<K = Heads>(PhantomData<K>);

impl<K: AxisName> AxisName for KvWidth<K> {
    const NAME: Ident = Ident::new("KvWidth");
    const SIZE: usize = K::SIZE * Head::SIZE;
}

axes![Head = 128];

#[derive(Clone, Debug)]
pub struct Heads<const N: usize = 4>;

impl<const N: usize> AxisName for Heads<N> {
    const NAME: Ident = Ident::new("KvHeads");
    const SIZE: usize = N;
}

#[derive(Clone, Debug)]
pub struct Groups<Q = Queries, K = Heads>(PhantomData<(Q, K)>);

impl<Q: AxisName, K: AxisName> AxisName for Groups<Q, K> {
    const NAME: Ident = Ident::new("QueryGroups");

    const SIZE: usize = {
        assert!(K::SIZE > 0 && Q::SIZE.is_multiple_of(K::SIZE));
        Q::SIZE / K::SIZE
    };
}

#[derive(Clone, Debug)]
pub struct Queries<const N: usize = 32>;

impl<const N: usize> AxisName for Queries<N> {
    const NAME: Ident = Ident::new("Queries");
    const SIZE: usize = N;
}

pub type Vocabulary = crate::ops::attention::geometry::Vocabulary<151936>;

#[derive(Clone, Debug)]
pub struct ExpertCount<const N: usize = EXPERTS>;

impl<const N: usize> AxisName for ExpertCount<N> {
    const NAME: Ident = Ident::new("ExpertCount");
    const SIZE: usize = N;
}

#[derive(Clone, Debug)]
pub struct Selected<const N: usize = 8>;

impl<const N: usize> AxisName for Selected<N> {
    const NAME: Ident = Ident::new("Selected");
    const SIZE: usize = N;
}

/// A published Qwen3 MoE checkpoint size, by its hub spelling.
pub struct Size<const N: &'static str>;

/// The norm of every Qwen3 MoE size: `rms_norm_eps` 1e-6.
type MoeNorm<W, X> =
    RmsNorm<W, EPSILON, m![1], Weight<m![W], m![1], <X as Memory>::Tensor<bf16, m![1], m![W]>>>;

const EPSILON: usize = 1e-6_f32.to_bits() as usize;

impl Geometry for Size<"30B-A3B"> {
    const SIZE: &'static str = "30B-A3B";
    const FAMILY: &'static str = "qwen3_moe";
    type Queries = QueryWidth<Queries<32>>;
    type Hidden = geometry::Hidden<2048>;
    type Intermediate = swiglu::Intermediate<768>;
    type Vocabulary = Vocabulary;
    type KvHeads = Heads;
    type Groups = Groups<Queries<32>, Heads>;
    type Head = Head;
    type Layers<L> = [L; Self::LAYERS];
    type Norm<W: AxisName, X: Memory> = MoeNorm<W, X>;

    const LAYERS: usize = LAYERS;
    const EPSILON: usize = EPSILON;
    const TIED_EMBEDDINGS: bool = false;
}

impl Geometry for Size<"235B-A22B"> {
    const SIZE: &'static str = "235B-A22B";
    const FAMILY: &'static str = "qwen3_moe";
    type Queries = QueryWidth<Queries<64>>;
    type Hidden = geometry::Hidden<4096>;
    type Intermediate = swiglu::Intermediate<1536>;
    type Vocabulary = Vocabulary;
    type KvHeads = Heads;
    type Groups = Groups<Queries<64>, Heads>;
    type Head = Head;
    type Layers<L> = [L; Self::LAYERS];
    type Norm<W: AxisName, X: Memory> = MoeNorm<W, X>;

    const LAYERS: usize = 94;
    const EPSILON: usize = EPSILON;
    const TIED_EMBEDDINGS: bool = false;
}

/// What a Qwen3 MoE checkpoint's config declares that its size fixes, and the features Qwen3 MoE
/// does not serve.
#[derive(Deserialize)]
pub struct Config {
    hidden_size: usize,
    moe_intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    num_experts: usize,
    num_experts_per_tok: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    hidden_act: Option<String>,
    #[serde(default)]
    norm_topk_prob: bool,
    #[serde(default)]
    mlp_only_layers: Vec<usize>,
    /// Transformers reads an absent one as 1.
    #[serde(default)]
    decoder_sparse_step: Option<usize>,
    #[serde(flatten, deserialize_with = "config::flattened")]
    _rotary: config::decoder::Rotary,
    #[serde(default)]
    use_sliding_window: bool,
    #[serde(default)]
    attention_bias: bool,
}

/// A config is of a size when it declares the size's values, at most its compiled vocabulary rows,
/// and none of the features Qwen3 MoE does not serve. A longer declared context is served up to the
/// compiled one.
impl<const N: &'static str> TryFrom<Config> for Size<N>
where
    Self: Geometry,
{
    type Error = io::Error;

    fn try_from(config: Config) -> io::Result<Self> {
        let unsupported = |reason: &str| io::Error::new(io::ErrorKind::Unsupported, reason);
        let heads = <Self as Geometry>::KvHeads::SIZE;
        let fixed = [
            (
                "hidden_size",
                config.hidden_size,
                <Self as Geometry>::Hidden::SIZE,
            ),
            (
                "moe_intermediate_size",
                config.moe_intermediate_size,
                <Self as Geometry>::Intermediate::SIZE,
            ),
            ("num_hidden_layers", config.num_hidden_layers, Self::LAYERS),
            ("num_key_value_heads", config.num_key_value_heads, heads),
            (
                "num_attention_heads",
                config.num_attention_heads,
                heads * <Self as Geometry>::Groups::SIZE,
            ),
            ("head_dim", config.head_dim, <Self as Geometry>::Head::SIZE),
            (
                "num_experts",
                config.num_experts,
                ExpertCount::<EXPERTS>::SIZE,
            ),
            (
                "num_experts_per_tok",
                config.num_experts_per_tok,
                Selected::<8>::SIZE,
            ),
        ];
        if let Some((key, declared, held)) = fixed
            .into_iter()
            .find(|(_, declared, held)| declared != held)
        {
            return Err(unsupported(&format!(
                "{key}: {declared}, the size {N} holds {held}"
            )));
        }
        vocabulary::<<Self as Geometry>::Vocabulary>(config.vocab_size)?;
        if (config.rms_norm_eps as f32).to_bits() as usize != Self::EPSILON {
            return Err(unsupported(&format!(
                "rms_norm_eps: {}, another than the size's",
                config.rms_norm_eps
            )));
        }
        if config.tie_word_embeddings != Self::TIED_EMBEDDINGS {
            return Err(unsupported(&format!(
                "tie_word_embeddings: {}, the size ties: {}",
                config.tie_word_embeddings,
                Self::TIED_EMBEDDINGS
            )));
        }
        let features = [
            (
                config
                    .hidden_act
                    .as_deref()
                    .is_some_and(|act| act != "silu"),
                "hidden_act: Qwen3 MoE's experts gate with SiLU",
            ),
            (
                !config.norm_topk_prob,
                "norm_topk_prob: Qwen3 MoE renormalizes the selected experts' probabilities",
            ),
            (
                !config.mlp_only_layers.is_empty(),
                "mlp_only_layers: every Qwen3 MoE layer is sparse",
            ),
            (
                config.decoder_sparse_step.is_some_and(|step| step != 1),
                "decoder_sparse_step: every Qwen3 MoE layer is sparse",
            ),
            (
                config.use_sliding_window,
                "use_sliding_window: Qwen3 MoE attends the whole history",
            ),
            (
                config.attention_bias,
                "attention_bias: Qwen3 MoE projects without biases",
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

/// Source: vLLM `vllm/model_executor/models/qwen3_moe.py`.
pub type Qwen3Moe<
    Embedding,
    Qkv,
    QueryNorm,
    KeyNorm,
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
> = DbrxForCausalLM<
    Embedding,
    attention::NormalizedQkv<Qkv, QueryNorm, KeyNorm>,
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
    DEPTH,
>;
