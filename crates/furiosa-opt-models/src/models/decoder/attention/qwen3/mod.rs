//! Qwen3, by published size: the Hugging Face classes over a [`Size`], the [`Form`] of the
//! projections, a placement, a KV cache form and a memory, e.g.
//! `ForCausalLM<Size<"0.6B">, form::Affine, Parallel<1>, cache::Bf16, X>`.
//!
//! - A size's [`Geometry`] is what its checkpoints declare; its [`Decode`] is how a call of it is
//!   compiled today, the context it serves included.
//! - [`Catalog`] is the one binding of `Qwen3ForCausalLM` both precisions instantiate: a projection
//!   [`Form`] chooses only the six projection weights, BF16 dense matrices or block-scaled E4M3.
//! - A size's [`Front`] is how its family computes a row's query, key and value heads from their
//!   projection; Llama's sizes instantiate these classes and the catalog with the plain projection.
//! - Every size has eight key and value heads of 128 features, and its vocabulary compiles as
//!   152064 rows.

use std::fmt;
use std::io;
use std::marker::PhantomData;

use serde::Deserialize;

use super::{Geometry, LlamaEmbeddingModel, Qwen2ForCausalLM};
use crate::load::Named;
use crate::models::Family;
use crate::models::config::{self, scaling::Spectrum};
use crate::models::decoder::attention::{
    Chunks, Head, KvHeads, PaddedVocabulary, Prompts, Windows,
};
use crate::models::decoder::{self, Form};
use crate::modules::attention::{self, Cached, NormalizedQkv, key_value};
use crate::modules::feedforward::glu::Glu;
use crate::modules::generic::collection::Array;
use crate::modules::generic::collection::{Stack, View};
use crate::modules::language::{Backbone, Classifier, Greedy, Normalized};
use crate::modules::lora::{self, Rank, Term};
use crate::modules::transformer::PreNorm;
use crate::ops::attention::Attention;
use crate::ops::attention::full::{self, head::Replicated, paged};
use crate::ops::attention::geometry::{
    self, HeadRows, ProjectionWidth, QueryGroups, QueryWidth, scale,
};
use crate::ops::attention::gqa::Cache;
use crate::ops::attention::{Packed, Split};
use crate::ops::embedding::Embedding;
use crate::ops::index::gather::Gather;
use crate::ops::index::scatter::{Scatter, Store};
use crate::ops::linear::Affine;
use crate::ops::linear::Linear;
use crate::ops::normalization::{RmsNorm, span::Tiled};
use crate::ops::pointwise::{activation::Silu, add::Add, multiply::Multiply};
use crate::ops::reduction::Argmax;
use crate::ops::rotary::Rotary;
use crate::ops::swiglu;
use crate::specialize::attention::{Decode, Strategy};
use crate::tensor::axis::{self, Limit};
use crate::tensor::form::Weight;
use crate::tensor::memory::Memory;
use crate::tensor::parallel::{Chip, Parallelism};
use crate::tensor::scalar::F32;
use crate::{Layers, Rolled, Sequential, Then};
use catalog::Depth;
use furiosa_opt_std::prelude::*;

pub mod fp8;
pub mod launch;

/// A published Qwen3 checkpoint size, by its hub spelling.
pub struct Size<const N: &'static str>;

/// HF `Qwen3ForCausalLM`: the base model, then the greedy vocabulary head.
pub type ForCausalLM<S, P, Q, C, X> = super::ForCausalLM<S, Model<S, P, Q, C, X>, LmHead<S, X>, P>;

/// HF `Qwen3ForSequenceClassification`: the base model, then the scores of its last row.
pub type ForSequenceClassification<S, P, Q, C, X> =
    super::ForSequenceClassification<S, Model<S, P, Q, C, X>, Classifier<Hidden<S>>, P>;

/// HF `Qwen3Model`: the embedding, the layers and the final norm, and the constants that prepare a
/// call's attention context.
pub type Model<S, P, Q, C, X> = super::Model<S, Body<S, P, Q, C, X>, RotaryEmbedding<S, C, X>, P>;

/// HF `Qwen3DecoderLayer`: one element of `model.layers`, a subtree that loads alone.
pub type DecoderLayer<S, P, Q, C, X> =
    PreNorm<SelfAttention<S, P, Q, C, X>, Mlp<S, P, Q, X>, Norm<S, Hidden<S>, X>, Add>;

/// The greedy head over the vocabulary rows: the projection whose bias lowers the padded rows, and
/// the first maximum of each row's scores.
pub type LmHead<S, X> = Greedy<
    Hidden<S>,
    Vocabulary<S>,
    Affine<
        <X as Memory>::Tensor<bf16, Chip, m![{ Symbol<Vocabulary<S>> }, { Symbol<Hidden<S>> }]>,
        <X as Memory>::Tensor<f32, Chip, m![{ Symbol<Vocabulary<S>> }]>,
    >,
    Argmax<Vocabulary<S>, Chip, <X as Memory>::Tensor<i32, Chip, m![{ Symbol<Vocabulary<S>> }]>>,
>;

type Hidden<S> = <S as Geometry>::Hidden;
type Vocabulary<S> = <S as Geometry>::Vocabulary;
type Groups<S> = <S as Geometry>::Groups;
type Norm<S, W, X> = <S as Geometry>::Norm<W, X>;
type Chips<Q> = <Q as Parallelism>::Chip;
type Rows<S> = <S as Decode>::Rows;

/// The embedding, the layers and the final norm, whose rows a call stores.
type Body<S, P, Q, C, X> = Then<
    Normalized<
        Backbone<
            Named<
                "embed_tokens",
                Embedding<
                    Vocabulary<S>,
                    Hidden<S>,
                    Chips<Q>,
                    <X as Memory>::Tensor<
                        bf16,
                        Chips<Q>,
                        m![{ Symbol<Vocabulary<S>> }, { Symbol<Hidden<S>> }],
                    >,
                >,
            >,
            Layers<<S as Geometry>::Layers<DecoderLayer<S, P, Q, C, X>>, Rolled>,
        >,
        Norm<S, Hidden<S>, X>,
    >,
    Store,
>;

/// The query, key and value heads of a row, as a family computes them from a `Projection`: Qwen3
/// normalizes each query and key head by `Norm`, Llama takes them as projected.
pub trait Front {
    type Heads<Projection, Norm>;
    /// The norm of each query or key head of the catalog's layer: `Norm`, or none, `()`.
    type HeadNorm<Norm>;
}

/// The packed query, key and value projection of a size in form `P`, unpacked into heads.
type Packing<S, P, X> = Sequential<
    <P as Form>::Of<
        Hidden<S>,
        <S as Decode>::Width,
        bf16,
        Chip,
        X,
        Packed<<S as Geometry>::KvHeads, <S as Decode>::Packed, X, <P as Form>::Stored>,
    >,
    Split<<S as Decode>::Width, <S as Geometry>::KvHeads, Groups<S>, <S as Geometry>::Head>,
>;

impl<const N: &'static str> Front for Size<N> {
    type Heads<Projection, Norm> =
        NormalizedQkv<Projection, Named<"q_norm", Norm>, Named<"k_norm", Norm>>;
    type HeadNorm<Norm> = Norm;
}

/// The size's query, key and value front, the rotation, the cache write, the attention over each
/// row's whole history and the output projection.
type SelfAttention<S, P, Q, C, X> = Cached<
    <S as Front>::Heads<Packing<S, P, X>, Norm<S, <S as Geometry>::Head, X>>,
    Rotary<<S as Geometry>::Head, <S as Geometry>::Head, C>,
    Scatter,
    (),
    Attention<
        PhantomData<(<S as Geometry>::KvHeads, Groups<S>, <S as Geometry>::Head)>,
        F32<SCALE>,
        { Strategy::Items },
    >,
    Named<"o_proj", Projection<P, <S as Geometry>::Queries, Hidden<S>, Q, X>>,
>;

/// The gated feed-forward over the size's intermediate features.
type Mlp<S, P, Q, X> = Glu<
    Projection<P, Hidden<S>, Intermediate<S>, Q, X>,
    Silu,
    Projection<P, Hidden<S>, Intermediate<S>, Q, X>,
    Multiply,
    Projection<P, Intermediate<S>, Hidden<S>, Q, X>,
>;

/// A projection in form `P` on the chips of placement `Q`, in memory `X`.
type Projection<P, In, Out, Q, X> = decoder::Projection<P, In, Out, bf16, Chips<Q>, X>;

type Intermediate<S> = <S as Geometry>::Intermediate;

/// A LoRA adapter of the size `S` as PEFT saves it, over the memory `X`: each layer's update of
/// its packed query, key and value projection, its output projection and its feed-forward's gate,
/// up and down projections.
pub type Adapter<S, X> = Named<
    "base_model",
    Named<
        "model",
        Named<
            "model",
            Named<
                "layers",
                <S as Geometry>::Layers<(
                    Named<
                        "self_attn",
                        (
                            lora::Packed<
                                <S as Geometry>::KvHeads,
                                <S as Geometry>::Queries,
                                Factors<Rank<3>, Hidden<S>, <S as Decode>::Width, X>,
                            >,
                            Named<
                                "o_proj",
                                Factors<Rank<1>, <S as Geometry>::Queries, Hidden<S>, X>,
                            >,
                        ),
                    >,
                    Named<
                        "mlp",
                        (
                            Named<"gate_proj", Factors<Rank<1>, Hidden<S>, Intermediate<S>, X>>,
                            Named<"up_proj", Factors<Rank<1>, Hidden<S>, Intermediate<S>, X>>,
                            Named<"down_proj", Factors<Rank<1>, Intermediate<S>, Hidden<S>, X>>,
                        ),
                    >,
                )>,
            >,
        >,
    >,
>;

/// An update's `[K, In]` and `[Out, K]` factors in memory `X`.
type Factors<K, In, Out, X> = Term<
    <X as Memory>::Tensor<bf16, m![1], m![K, In]>,
    <X as Memory>::Tensor<bf16, m![1], m![Out, K]>,
>;

/// The constants that prepare a call's rotations, write slots and paged history reads, for the
/// rows of a call and a cache of form `C`.
type RotaryEmbedding<S, C, X> = key_value::Preparation<
    Rows<S>,
    key_value::Paging<
        <S as Geometry>::KvHeads,
        <S as Geometry>::Head,
        Rows<S>,
        <S as Decode>::Windows,
        C,
        X,
    >,
    Chip,
    X,
>;

/// The score multiplier of Qwen3's 128-feature heads.
const SCALE: usize = scale(Head::SIZE);

/// The norm of every Qwen3 size: `rms_norm_eps` 1e-6.
type QwenNorm<W, X> =
    RmsNorm<W, { EPSILON }, m![1], Weight<m![W], m![1], <X as Memory>::Tensor<bf16, m![1], m![W]>>>;

const EPSILON: usize = 1e-6_f32.to_bits() as usize;

impl Geometry for Size<"0.6B"> {
    const SIZE: &'static str = "0.6B";
    const FAMILY: &'static str = "qwen3";
    type Queries = QueryWidth<2048>;
    type Hidden = geometry::Hidden<1024>;
    type Intermediate = swiglu::Intermediate<3072>;
    type Vocabulary = PaddedVocabulary;
    type KvHeads = KvHeads;
    type Groups = QueryGroups<2>;
    type Head = Head;
    type Layers<L> = [L; Self::LAYERS];
    type Norm<W: AxisName, X: Memory> = QwenNorm<W, X>;

    const LAYERS: usize = 28;
    const EPSILON: usize = EPSILON;
    const TIED_EMBEDDINGS: bool = true;
}

impl Geometry for Size<"1.7B"> {
    const SIZE: &'static str = "1.7B";
    const FAMILY: &'static str = "qwen3";
    type Queries = QueryWidth<2048>;
    type Hidden = geometry::Hidden<2048>;
    type Intermediate = swiglu::Intermediate<6144>;
    type Vocabulary = PaddedVocabulary;
    type KvHeads = KvHeads;
    type Groups = QueryGroups<2>;
    type Head = Head;
    type Layers<L> = [L; Self::LAYERS];
    type Norm<W: AxisName, X: Memory> = QwenNorm<W, X>;

    const LAYERS: usize = 28;
    const EPSILON: usize = EPSILON;
    const TIED_EMBEDDINGS: bool = true;
}

impl Geometry for Size<"4B"> {
    const SIZE: &'static str = "4B";
    const FAMILY: &'static str = "qwen3";
    type Queries = QueryWidth<4096>;
    type Hidden = geometry::Hidden<2560>;
    type Intermediate = swiglu::Intermediate<9728>;
    type Vocabulary = PaddedVocabulary;
    type KvHeads = KvHeads;
    type Groups = QueryGroups<4>;
    type Head = Head;
    type Layers<L> = [L; Self::LAYERS];
    type Norm<W: AxisName, X: Memory> = QwenNorm<W, X>;

    const LAYERS: usize = 36;
    const EPSILON: usize = EPSILON;
    const TIED_EMBEDDINGS: bool = true;
}

impl Geometry for Size<"8B"> {
    const SIZE: &'static str = "8B";
    const FAMILY: &'static str = "qwen3";
    type Queries = QueryWidth<4096>;
    type Hidden = geometry::Hidden<4096>;
    type Intermediate = swiglu::Intermediate<12288>;
    type Vocabulary = PaddedVocabulary;
    type KvHeads = KvHeads;
    type Groups = QueryGroups<4>;
    type Head = Head;
    type Layers<L> = [L; Self::LAYERS];
    type Norm<W: AxisName, X: Memory> = QwenNorm<W, X>;

    const LAYERS: usize = 36;
    const EPSILON: usize = EPSILON;
    const TIED_EMBEDDINGS: bool = false;
}

/// What a Qwen3 checkpoint's config declares that its size fixes, and the features Qwen3 does not
/// serve.
#[derive(Deserialize)]
pub struct Config {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    #[serde(flatten, deserialize_with = "config::flattened")]
    rope: Spectrum,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    use_sliding_window: bool,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    quantization_config: Option<serde_json::Value>,
}

/// A config is of a size when it declares the size's values, at most its compiled vocabulary rows,
/// a spectrum the rotary op holds and none of the features Qwen3 does not serve. A longer declared
/// context is served up to the compiled one, the window the paging constants hold.
impl<const N: &'static str> TryFrom<Config> for Size<N>
where
    Self: Geometry,
{
    type Error = io::Error;

    fn try_from(config: Config) -> io::Result<Self> {
        let invalid = |reason: String| io::Error::new(io::ErrorKind::InvalidData, reason);
        let unsupported = |reason: &str| io::Error::new(io::ErrorKind::Unsupported, reason);
        let heads = <Self as Geometry>::KvHeads::SIZE;
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
            ("head_dim", config.head_dim, <Self as Geometry>::Head::SIZE),
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
        if config.quantization_config.is_some() {
            return Err(unsupported(
                "quantization_config: the size's lines here read BF16 checkpoints",
            ));
        }
        // The load binds this spectrum, so one it cannot bind is refused before a weight is read.
        config
            .rope
            .bind::<<Self as Geometry>::Head, <Self as Geometry>::Head>()?;
        match (config.use_sliding_window, config.attention_bias) {
            (true, _) => Err(unsupported(
                "use_sliding_window: Qwen3 attends the whole history",
            )),
            (_, true) => Err(unsupported("attention_bias: Qwen3 projects without biases")),
            (false, false) => Ok(Size),
        }
    }
}

impl<const N: &'static str> Family for Size<N>
where
    Self: Geometry,
{
    type Hf = Config;
}

/// The work a call's host plans for a size of `G` query heads a key and value head, in `WAVES`
/// wide tiles.
type Work<G, const WAVES: usize> = paged::Work<Prompts, KvHeads, G, Head, WAVES>;

// The rotary tiles of a launch's 1024 rows.
axes![Slots = 8];

impl Decode for Size<"0.6B"> {
    type Rows = Limit<16, 64>;
    type Windows = Windows<256>;
    type Width = ProjectionWidth<4096>;
    type Packed = m![{ Symbol<ProjectionWidth<4096>> }, { Symbol<geometry::Hidden<1024>> }];
    type Work = Work<QueryGroups<2>, { tiles::<Self>() }>;
    type Capacity = axis::Rows<1024>;
    type Reach = axis::Rows<2048>;
    type Sequences = axis::Rows<8>;
    type Slots = Slots;
    type Positions = axis::Rows<32768>;
}

impl Tiled for geometry::Hidden<1024> {
    type Capacity = axis::Rows<126>;
    type Last = axis::Rows<4>;
}

impl Decode for Size<"1.7B"> {
    type Rows = Limit<16, 64>;
    type Windows = Windows<256>;
    type Width = ProjectionWidth<4096>;
    type Packed = m![{ Symbol<ProjectionWidth<4096>> }, { Symbol<geometry::Hidden<2048>> }];
    type Work = Work<QueryGroups<2>, { tiles::<Self>() }>;
    type Capacity = axis::Rows<1024>;
    type Reach = axis::Rows<2048>;
    type Sequences = axis::Rows<8>;
    type Slots = Slots;
    type Positions = axis::Rows<32768>;
}

impl Tiled for geometry::Hidden<2048> {
    type Capacity = axis::Rows<126>;
    type Last = axis::Rows<4>;
}

impl Decode for Size<"4B"> {
    type Rows = Limit<16, 64>;
    type Windows = Windows<256>;
    type Width = ProjectionWidth<6144>;
    type Packed = m![{ Symbol<ProjectionWidth<6144>> }, { Symbol<geometry::Hidden<2560>> }];
    type Work = Work<QueryGroups<4>, { tiles::<Self>() }>;
    type Capacity = axis::Rows<1024>;
    type Reach = axis::Rows<2048>;
    type Sequences = axis::Rows<8>;
    type Slots = Slots;
    type Positions = axis::Rows<32768>;
}

impl Tiled for geometry::Hidden<2560> {
    type Capacity = axis::Rows<126>;
    type Last = axis::Rows<4>;
}

impl Decode for Size<"8B"> {
    type Rows = Limit<64, 64>;
    // The shared context its checkpoint declares, by choice: its call's 192 tiles would hold 96
    // windows a row, and more windows take more tiles, a kernel change to measure.
    type Windows = Chunks;
    type Width = ProjectionWidth<6144>;
    type Packed = m![KvHeads, { Symbol<HeadRows<6>> }, Head, { Symbol<geometry::Hidden<4096>> }];
    type Work = Work<QueryGroups<4>, { tiles::<Self>() }>;
    type Capacity = axis::Rows<1024>;
    type Reach = axis::Rows<2048>;
    type Sequences = axis::Rows<8>;
    type Slots = Slots;
    type Positions = axis::Rows<32768>;
}

impl Tiled for geometry::Hidden<4096> {
    type Capacity = axis::Rows<126>;
    type Last = axis::Rows<4>;
}

/// Source: vLLM `vllm/model_executor/models/qwen3.py`.
pub type Qwen3ForCausalLM<
    Embedding,
    Qkv,
    QueryNorm,
    KeyNorm,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Head,
    const DEPTH: usize,
    Kind = Array,
> = Qwen2ForCausalLM<
    Embedding,
    attention::NormalizedQkv<Qkv, QueryNorm, KeyNorm>,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Head,
    DEPTH,
    Kind,
>;

/// Source: SGLang `python/sglang/srt/models/qwen3_rm.py`.
pub type Qwen3ForRewardModel<
    Embedding,
    Qkv,
    QueryNorm,
    KeyNorm,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Pool,
    Score,
    const DEPTH: usize,
> = super::GPT2ForSequenceClassification<
    Embedding,
    attention::NormalizedQkv<Qkv, QueryNorm, KeyNorm>,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Pool,
    Score,
    DEPTH,
>;

/// Source: SGLang `python/sglang/srt/models/qwen3_embedding.py`.
pub type Qwen3Model<
    Embedding,
    Qkv,
    QueryNorm,
    KeyNorm,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Pool,
    const DEPTH: usize,
> = LlamaEmbeddingModel<
    Embedding,
    attention::NormalizedQkv<Qkv, QueryNorm, KeyNorm>,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Pool,
    DEPTH,
>;

/// What the one catalog binding is generic over, private to Qwen3.
mod catalog {
    /// The catalog binding of a size `S` in projection form `P`, at the count of layers the
    /// size's layer array holds, and its body.
    pub trait Depth<S, P, C, X> {
        type Catalog;
        type Body;
    }
}

/// The query heads of a size `S`: `num_attention_heads`, its key and value heads' groups.
pub struct QueryHeads<S>(PhantomData<fn() -> S>);

impl<S> Clone for QueryHeads<S> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<S> fmt::Debug for QueryHeads<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("QueryHeads")
    }
}

impl<S: Geometry + 'static> AxisName for QueryHeads<S> {
    const NAME: Ident = Ident::new("QueryHeads");
    const SIZE: usize = S::KvHeads::SIZE * S::Groups::SIZE;
}

type HeadOf<S> = <S as Geometry>::Head;
type KvHeadsOf<S> = <S as Geometry>::KvHeads;

/// The norm of each query and key head, its gain replicated over the slices.
type HeadNorm<S, X> = full::HeadNorm<
    <S as Geometry>::Head,
    Chip,
    Weight<
        Replicated<<S as Geometry>::Head>,
        Chip,
        <X as Memory>::Tensor<bf16, Chip, Replicated<<S as Geometry>::Head>>,
    >,
    F32<EPSILON>,
>;

type Embed<S, X> = Named<
    "embed_tokens",
    Embedding<
        Vocabulary<S>,
        Hidden<S>,
        Chip,
        Weight<
            m![{ Symbol<Vocabulary<S>> }, { Symbol<Hidden<S>> }],
            Chip,
            <X as Memory>::Tensor<bf16, Chip, m![{ Symbol<Vocabulary<S>> }, { Symbol<Hidden<S>> }]>,
        >,
    >,
>;

/// The norms of a catalog layer's query and key heads, as the size's front takes them.
pub(crate) type QueryNorm<S, X> = <S as Front>::HeadNorm<Named<"q_norm", HeadNorm<S, X>>>;
pub(crate) type KeyNorm<S, X> = <S as Front>::HeadNorm<Named<"k_norm", HeadNorm<S, X>>>;

/// A catalog layer's query, key and value projections in form `P` and its head norms.
type Heads<S, P, X> = NormalizedQkv<Qkv<S, P, X>, QueryNorm<S, X>, KeyNorm<S, X>>;

type Qkv<S, P, X> = attention::Projection<
    Named<"q_proj", <P as Form>::Query<Hidden<S>, QueryHeads<S>, HeadOf<S>, X>>,
    Named<"k_proj", <P as Form>::Key<Hidden<S>, KvHeadsOf<S>, HeadOf<S>, X>>,
    Named<"v_proj", <P as Form>::Value<Hidden<S>, KvHeadsOf<S>, HeadOf<S>, X>>,
>;

type Attend<S> = full::Attention<
    PhantomData<(<S as Geometry>::KvHeads, Groups<S>, <S as Geometry>::Head)>,
    F32<SCALE>,
>;

type Ffn<S, P, X> = swiglu::Mlp<
    Hidden<S>,
    Intermediate<S>,
    <P as Form>::GateUp<Hidden<S>, Intermediate<S>, X>,
    <P as Form>::Down<Hidden<S>, Intermediate<S>, X>,
>;

/// One layer of the catalog binding.
pub(crate) type Layer<S, P, C, X> = super::Qwen2Layer<
    Heads<S, P, X>,
    Rotary<HeadOf<S>, HeadOf<S>, C>,
    Scatter,
    Gather,
    Attend<S>,
    Named<"o_proj", <P as Form>::Output<<S as Geometry>::Queries, Hidden<S>, X>>,
    Ffn<S, P, X>,
    Norm<S, Hidden<S>, X>,
    Add,
>;

/// The family's causal model of `N` layers over a size's axes and a form's projections:
/// `Qwen3ForCausalLM`, or `LlamaForCausalLM` as Qwen2's.
type Bound<S, P, C, X, const N: usize> = Qwen2ForCausalLM<
    Embed<S, X>,
    Heads<S, P, X>,
    Rotary<HeadOf<S>, HeadOf<S>, C>,
    Scatter,
    Gather,
    Attend<S>,
    Named<"o_proj", <P as Form>::Output<<S as Geometry>::Queries, Hidden<S>, X>>,
    Ffn<S, P, X>,
    Norm<S, Hidden<S>, X>,
    Add,
    LmHead<S, X>,
    N,
    Stack,
>;

impl<S: Geometry + Front + 'static, P: Form, C: Scalar, X: Memory, const N: usize> Depth<S, P, C, X>
    for [(); N]
where
    swiglu::Down<Intermediate<S>>: AxisName,
    Layer<S, P, C, X>: View<N>,
{
    type Catalog = Bound<S, P, C, X, N>;
    type Body = Bound<S, P, C, X, N>::Body;
}

/// The one catalog binding of a size `S` in projection form `P` over memory `X`.
pub type Catalog<S, P, C, X> = <<S as Geometry>::Layers<()> as Depth<S, P, C, X>>::Catalog;

/// The catalog's body: the embedding, the layers and the final norm.
pub type CatalogBody<S, P, C, X> = <<S as Geometry>::Layers<()> as Depth<S, P, C, X>>::Body;

/// The wide tiles of the size `S`'s work: as many as a call of its rows takes when every row
/// attends the whole context `S` is compiled for, so its call's work holds whatever it admits.
const fn tiles<S: Geometry + Decode>() -> usize {
    paged::Plan::waves::<S::Groups>(S::Rows::SIZE, S::Windows::SIZE * Cache::SIZE)
}

#[cfg(test)]
pub(in crate::models) mod tests {
    use super::*;
    use crate::load::{Checkpoint, Load};
    use crate::models::decoder::form;
    use crate::ops::index::table::FIELDS;
    use crate::tensor::memory::Hbm;
    use serde_json::{Value, json};

    /// The BF16 instance of the catalog loads from a checkpoint, as the FP8 one does.
    #[test]
    fn bf16_catalog_loads() {
        fn loads<T: for<'a> Load<&'a Checkpoint>>() {}
        loads::<Catalog<Size<"0.6B">, form::Affine, bf16, Hbm>>();
        loads::<Catalog<Size<"8B">, form::Affine, bf16, Hbm>>();
    }

    /// Qwen3-0.6B's config at the pinned revision, with `fields` over it.
    fn config(fields: Value) -> Value {
        let mut document = json!({
            "hidden_size": 1024, "intermediate_size": 3072, "num_hidden_layers": 28,
            "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
            "vocab_size": 151936, "rms_norm_eps": 1e-6, "max_position_embeddings": 40960,
            "rope_scaling": null, "rope_theta": 1000000, "tie_word_embeddings": true,
            "use_sliding_window": false, "attention_bias": false,
        });
        let object = document.as_object_mut().unwrap();
        object.extend(fields.as_object().unwrap().clone());
        document
    }

    fn refusal<S: Family>(document: &Value) -> Option<io::ErrorKind> {
        S::parse(document).err().map(|error| error.kind())
    }

    /// Qwen's YaRN recipe: four times the 32768 positions Qwen3 was trained on.
    fn yarn() -> Value {
        json!({"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768})
    }

    /// A size takes YaRN and linear spectra in either spelling, defaults included, serving a longer
    /// context up to its compiled one; a spectrum the rotary op does not hold is unsupported.
    #[test]
    fn serves_rescaled_spectra_up_to_the_compiled_context() {
        let mut nested = yarn();
        nested["rope_theta"] = json!(1000000);
        let extended = |rope: Value, positions: usize| {
            config(json!({"rope_scaling": rope, "max_position_embeddings": positions}))
        };
        let mut transformers = config(json!({"max_position_embeddings": 131072}));
        let object = transformers.as_object_mut().unwrap();
        object.remove("rope_theta");
        object.remove("rope_scaling");
        object.insert("rope_parameters".into(), nested);
        let linear = |factor: f64| json!({"rope_type": "linear", "factor": factor});
        let untrained = json!({"rope_type": "yarn", "factor": 4.0});
        let mut ratio = yarn();
        ratio["factor"] = Value::Null;
        let unsupported = Some(io::ErrorKind::Unsupported);
        for (document, kind) in [
            (config(json!({})), None),
            (extended(yarn(), 131072), None),
            (extended(untrained, 131072), None),
            (extended(ratio, 131072), None),
            (transformers, None),
            (extended(linear(3.2), 40960), None),
            (extended(yarn(), 131073), None),
            (extended(linear(3.3), 40960), None),
            (
                extended(json!({"rope_type": "dynamic", "factor": 2.0}), 40960),
                unsupported,
            ),
            (
                extended(json!({"rope_type": "longrope", "factor": 2.0}), 40960),
                unsupported,
            ),
            (config(json!({"partial_rotary_factor": 0.5})), unsupported),
        ] {
            assert_eq!(refusal::<Size<"0.6B">>(&document), kind, "{document}");
        }
        let eight = |fields: Value| {
            let mut document = config(json!({
                "hidden_size": 4096, "intermediate_size": 12288, "num_hidden_layers": 36,
                "num_attention_heads": 32, "tie_word_embeddings": false,
            }));
            document
                .as_object_mut()
                .unwrap()
                .extend(fields.as_object().unwrap().clone());
            document
        };
        assert_eq!(refusal::<Size<"8B">>(&eight(json!({}))), None);
        let past = eight(json!({"rope_scaling": yarn(), "max_position_embeddings": 40961}));
        assert_eq!(refusal::<Size<"8B">>(&past), None);
    }

    /// A call of a size's rows, each attending its whole compiled context, plans within the tiles
    /// the size's work is compiled for, under either tiling.
    #[test]
    fn plans_every_row_over_the_whole_context() {
        fn rows<S: Decode>(keys: usize) -> Vec<[i32; FIELDS]> {
            let row = |at: usize| [at, 1, keys - 1, 0, keys.div_ceil(16)].map(|f| f as i32);
            (0..S::Rows::SIZE).map(row).collect()
        }
        fn context<S: Decode>() -> usize {
            S::Windows::SIZE * Cache::SIZE
        }
        let ids = vec![1; 1 << 18];
        type Small = <Size<"0.6B"> as Decode>::Work;
        type Grouped = <Size<"4B"> as Decode>::Work;
        type Large = <Size<"8B"> as Decode>::Work;
        let small = rows::<Size<"0.6B">>(context::<Size<"0.6B">>());
        let grouped = rows::<Size<"4B">>(context::<Size<"4B">>());
        let large = rows::<Size<"8B">>(context::<Size<"8B">>());
        for tiling in [paged::Tiling::Adaptive, paged::Tiling::Fixed] {
            Small::sequential::<Rows<Size<"0.6B">>>(&small, &ids, tiling).unwrap();
            Grouped::sequential::<Rows<Size<"4B">>>(&grouped, &ids, tiling).unwrap();
            Large::sequential::<Rows<Size<"8B">>>(&large, &ids, tiling).unwrap();
        }
    }

    /// `Qwen/Qwen3-0.6B`'s config.json at revision c1899de289a04d12100db370d81485cdf75e47ca.
    pub(in crate::models) fn published() -> Value {
        json!({
            "architectures": ["Qwen3ForCausalLM"], "attention_bias": false, "attention_dropout": 0.0,
            "bos_token_id": 151643, "eos_token_id": 151645, "head_dim": 128, "hidden_act": "silu",
            "hidden_size": 1024, "initializer_range": 0.02, "intermediate_size": 3072,
            "max_position_embeddings": 40960, "max_window_layers": 28, "model_type": "qwen3",
            "num_attention_heads": 16, "num_hidden_layers": 28, "num_key_value_heads": 8,
            "rms_norm_eps": 1e-06, "rope_scaling": null, "rope_theta": 1000000,
            "sliding_window": null, "tie_word_embeddings": true, "torch_dtype": "bfloat16",
            "transformers_version": "4.51.0", "use_cache": true, "use_sliding_window": false,
            "vocab_size": 151936
        })
    }

    /// The 0.6B size reads its published config; a config declaring another value the size
    /// fixes, a vocabulary of no row or past its rows, or a feature Qwen3 does not serve is refused
    /// by its field.
    #[test]
    fn size_refuses_what_it_does_not_serve() {
        let parse = |document: &Value| <Size<"0.6B"> as Family>::parse(document).map(drop);
        parse(&published()).unwrap();
        use io::ErrorKind::{InvalidData, Unsupported};
        let rows = "the size compiles 152064 rows";
        for (key, value, kind, reason) in [
            (
                "hidden_size",
                json!(1000),
                InvalidData,
                "hidden_size: 1000, the size 0.6B holds 1024".to_owned(),
            ),
            (
                "intermediate_size",
                json!(1),
                InvalidData,
                "intermediate_size: 1, the size 0.6B holds 3072".into(),
            ),
            (
                "num_hidden_layers",
                json!(1),
                InvalidData,
                "num_hidden_layers: 1, the size 0.6B holds 28".into(),
            ),
            (
                "num_key_value_heads",
                json!(1),
                InvalidData,
                "num_key_value_heads: 1, the size 0.6B holds 8".into(),
            ),
            (
                "num_attention_heads",
                json!(1),
                InvalidData,
                "num_attention_heads: 1, the size 0.6B holds 16".into(),
            ),
            (
                "head_dim",
                json!(64),
                InvalidData,
                "head_dim: 64, the size 0.6B holds 128".into(),
            ),
            (
                "vocab_size",
                json!(0),
                InvalidData,
                format!("vocab_size: 0, {rows}"),
            ),
            (
                "vocab_size",
                json!(152065),
                InvalidData,
                format!("vocab_size: 152065, {rows}"),
            ),
            (
                "rms_norm_eps",
                json!(1e-5),
                InvalidData,
                "rms_norm_eps: 0.00001, another than the size's".into(),
            ),
            (
                "tie_word_embeddings",
                json!(false),
                InvalidData,
                "tie_word_embeddings: false, the size ties: true".into(),
            ),
            (
                "use_sliding_window",
                json!(true),
                Unsupported,
                "use_sliding_window: Qwen3 attends the whole history".into(),
            ),
            (
                "attention_bias",
                json!(true),
                Unsupported,
                "attention_bias: Qwen3 projects without biases".into(),
            ),
        ] {
            let mut document = published();
            document[key] = value.clone();
            let error = parse(&document).unwrap_err();
            assert_eq!(
                (error.kind(), error.to_string()),
                (kind, reason),
                "{key}: {value}"
            );
        }
        let mut widest = published();
        widest["vocab_size"] = json!(152064);
        parse(&widest).unwrap();
    }
}
