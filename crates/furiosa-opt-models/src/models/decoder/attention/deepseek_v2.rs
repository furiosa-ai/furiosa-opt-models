//! DeepSeek-V2, by published size: a size's [`Geometry`] is what its checkpoints declare, and its
//! [`Config`] refuses a value the size does not hold or a feature DeepSeek-V2 does not serve.
//!
//! - Every size's attention is latent: keys and values are read from one compressed row a token
//!   ([`Latent`]) and a shared rotated key ([`Rotary`]), rotated by the compiled [`Yarn`].
//! - Its first layer's feed-forward is dense; every later one routes each row to 6 of 64 SwiGLU
//!   experts, weighed by their router probabilities as they are, beside two shared experts.
//! - [`Model`] is the catalog's composition, over the [`AXK1ForCausalLM`] template.

use crate::models::Family;
use crate::models::decoder::attention::Geometry;
use crate::modules::feedforward::moe::Reduce;
use crate::modules::feedforward::moe::TopK;
use crate::modules::{attention, feedforward, feedforward::glu::Glu};
use crate::modules::{language, transformer};
use crate::ops::attention::geometry;
use crate::ops::embedding;
use crate::ops::index::table::ByteOffset;
use crate::ops::pointwise::activation::Silu;
use crate::ops::pointwise::add;
use crate::ops::pointwise::multiply::Multiply;
use crate::ops::swiglu;
use crate::ops::{
    attention::{geometry::QueryGroups, latent},
    linear::{Affine, Experts},
    normalization::RmsNorm,
    rotary,
};
use crate::tensor::axis::Single;
use crate::tensor::form::Weight;
use crate::tensor::mapping::Reshape;
use crate::tensor::mapping::Split;
use crate::tensor::memory::Memory;
use crate::{Branch, Interleave, Sequential, Then};
use furiosa_opt_std::prelude::*;
use serde::Deserialize;
use std::f32::consts::PI;
use std::io;

pub type Model = AXK1ForCausalLM<
    Embedding,
    Projection,
    latent::Scatter,
    Attend,
    Output,
    Mlp<DenseWidth>,
    Router,
    ExpertMlp,
    Reduce,
    Mlp<SharedWidth>,
    add::Add,
    Norm,
    Linear<Hidden, Vocabulary>,
    DENSE_LAYERS,
    { LAYERS - DENSE_LAYERS },
    ORDER,
>;

pub type Body = Model::Body;

pub type Head = Model::Head;

type Embedding = Then<ByteOffset<bf16, m![Hidden]>, embedding::Embedding<Vocabulary, Hidden>>;

type Projection = attention::LatentProjection<
    Query,
    KeyValue,
    RmsNorm<Latent>,
    rotary::Frequencies<Rotary>,
    rotary::Adjacent<Rotary>,
    latent::Key<Heads, Content, Value, Latent>,
>;

type Query = Then<
    Sequential<
        Linear<Hidden, QueryWidth>,
        Reshape<m![Heads,{Symbol<latent::Combined<Content,Rotary>>}]>,
    >,
    Split<Content, Rotary>,
>;

type Linear<I, O> = Affine<HbmTensor<bf16, m![1], m![O, I]>, ()>;

type KeyValue = Then<
    Then<Reshape<m![Single, Hidden]>, Matrix<Hidden, latent::Combined<Latent, Rotary>>>,
    Split<Latent, Rotary>,
>;

type Matrix<I, O> = Affine<HbmTensor<bf16, m![1], m![O, I]>, [HbmTensor<bf16, m![1], m![O]>; 0]>;

type Attend = latent::Attention<Heads, Latent, Rotary, Tokens>;

type Output = Then<latent::Value<Heads, Content, Value, Latent>, Linear<OutputWidth, Hidden>>;

pub type Attention = attention::Latent<Projection, latent::Scatter, Attend, Output>;

type Mlp<I> = Sequential<
    Then<
        Reshape<m![Single, Hidden]>,
        Glu<Matrix<Hidden, I>, Silu, Matrix<Hidden, I>, Multiply, Matrix<I, Hidden>>,
    >,
    Reshape<m![Hidden]>,
>;

type Router = Then<
    Then<Reshape<m![Single, Hidden]>, Matrix<Hidden, ExpertCount>>,
    TopK<ExpertCount, Selected>,
>;

type ExpertMlp = feedforward::ExpertMlp<
    Experts<Hidden, ExpertWidth, 64>,
    Silu,
    Experts<Hidden, ExpertWidth, 64>,
    Multiply,
    Experts<ExpertWidth, Hidden, 64>,
>;

type Norm = RmsNorm<Hidden>;

/// Compressed KV content and rotary-key storage for one latent attention layer.
pub type Cache = latent::LatentKeyValue<Latent, Rotary>;

/// Logical token history and its caller-supplied additive bias.
pub type History<B = ()> = latent::History<latent::Table<Tokens>, B>;

/// Write/read slots and logical history passed to latent attention.
pub type AttentionContext<B = ()> = attention::Context<latent::Slot, latent::Slot, History<B>>;

/// One token id, converted to an embedding byte offset by the body.
pub type TokenId = HbmTensor<i32, m![1], m![1]>;

pub const LAYERS: usize = 27;

/// The leading layers whose feed-forward is dense: `first_k_dense_replace`.
pub const DENSE_LAYERS: usize = 1;

pub const ORDER: &[Branch] = &{
    let mut order = [Branch::Second; LAYERS];
    let mut layer = 0;
    while layer < DENSE_LAYERS {
        order[layer] = Branch::First;
        layer += 1;
    }
    order
};

pub type Layers<A, B> = Interleave<([A; DENSE_LAYERS], [B; LAYERS - DENSE_LAYERS]), ORDER>;

pub type Hidden = geometry::Hidden<2048>;
pub type DenseWidth = swiglu::Intermediate<10944>;
pub type SharedWidth = swiglu::Intermediate<2816>;
pub type ExpertWidth = swiglu::Intermediate<1408>;

axes![
    Heads = 16,
    Latent = 512,
    Rotary = 64,
    Content = 128,
    Value = 128
];

#[derive(Clone, Debug)]
pub struct QueryWidth;

impl AxisName for QueryWidth {
    const NAME: Ident = Ident::new("QueryWidth");
    const SIZE: usize = Heads::SIZE * (Content::SIZE + Rotary::SIZE);
}

#[derive(Clone, Debug)]
pub struct OutputWidth;

impl AxisName for OutputWidth {
    const NAME: Ident = Ident::new("OutputWidth");
    const SIZE: usize = Heads::SIZE * Value::SIZE;
}

axes![
    Vocabulary = 102400,
    Tokens = 163840,
    ExpertCount = 64,
    Selected = 6
];

/// The compiled YaRN encoding; every latent attention layer reads its tables.
pub struct Yarn;

impl Yarn {
    const POSITIONS: usize = 4096;
    const THETA: f32 = 10000.0;
    const FACTOR: f64 = 40.0;
    const BETA_FAST: f64 = 32.0;
    const BETA_SLOW: f64 = 1.0;
    const MSCALE: f64 = 0.707;

    /// Inverse rotary frequencies, attention scale and rotation amplitude, in that order.
    pub fn tables() -> [Vec<f32>; 3] {
        let correction = |rotations: f32| {
            Rotary::SIZE as f32 * (Self::POSITIONS as f32 / (rotations * 2.0 * PI)).ln()
                / (2.0 * Self::THETA.ln())
        };
        let low = correction(Self::BETA_FAST as f32).floor().max(0.0);
        let high = correction(Self::BETA_SLOW as f32)
            .ceil()
            .min((Rotary::SIZE - 1) as f32);
        let factor = Self::FACTOR as f32;
        let inverse: Vec<f32> = (0..Rotary::SIZE / 2)
            .map(|index| {
                let frequency = Self::THETA.powf(-2.0 * index as f32 / Rotary::SIZE as f32);
                let mask = 1.0 - ((index as f32 - low) / (high - low)).clamp(0.0, 1.0);
                frequency / factor * (1.0 - mask) + frequency * mask
            })
            .collect();
        let magnitude = 0.1 * Self::MSCALE as f32 * factor.ln() + 1.0;
        [
            inverse,
            vec![magnitude * magnitude / (Content::SIZE as f32 + Rotary::SIZE as f32).sqrt()],
            vec![1.0],
        ]
    }
}

/// A published DeepSeek-V2 checkpoint size, by its hub spelling.
pub struct Size<const N: &'static str>;

/// `rms_norm_eps` 1e-6, as the bits of the FP32 nearest to it.
const EPSILON: usize = 1e-6_f32.to_bits() as usize;

/// A head's features are its value's (`v_head_dim`), so its heads' together are the output
/// projection's input; its queries and keys add the rotated features.
impl Geometry for Size<"Lite"> {
    const SIZE: &'static str = "Lite";
    const FAMILY: &'static str = "deepseek_v2";
    type Queries = OutputWidth;
    type Hidden = Hidden;
    type Intermediate = DenseWidth;
    type Vocabulary = Vocabulary;
    type KvHeads = Heads;
    type Groups = QueryGroups<1>;
    type Head = Value;
    type Layers<L> = [L; LAYERS];
    type Norm<W: AxisName, X: Memory> =
        RmsNorm<W, EPSILON, m![1], Weight<m![W], m![1], <X as Memory>::Tensor<bf16, m![1], m![W]>>>;

    const LAYERS: usize = LAYERS;
    const EPSILON: usize = EPSILON;
    const TIED_EMBEDDINGS: bool = false;
}

/// What a DeepSeek-V2 checkpoint's config declares that its size fixes, and the features
/// DeepSeek-V2 does not serve.
#[derive(Deserialize)]
pub struct Config {
    hidden_size: usize,
    intermediate_size: usize,
    moe_intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    n_routed_experts: usize,
    n_shared_experts: usize,
    num_experts_per_tok: usize,
    first_k_dense_replace: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    max_position_embeddings: usize,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    q_lora_rank: Option<usize>,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    #[serde(default)]
    hidden_act: Option<String>,
    #[serde(default)]
    scoring_func: Option<String>,
    #[serde(default)]
    topk_method: Option<String>,
    #[serde(default)]
    n_group: Option<usize>,
    #[serde(default)]
    topk_group: Option<usize>,
    #[serde(default)]
    norm_topk_prob: bool,
    #[serde(default)]
    routed_scaling_factor: Option<f64>,
    #[serde(default)]
    moe_layer_freq: Option<usize>,
    #[serde(default)]
    rope_theta: Option<f64>,
    /// Transformers 5 spells `rope_scaling` and `rope_theta` together as this.
    #[serde(default)]
    rope_parameters: Option<serde_json::Value>,
    #[serde(default)]
    rope_scaling: Option<serde_json::Value>,
}

/// A config is of a size when it declares the size's values, at most its compiled vocabulary rows
/// and context, its compiled rotation, and none of the features DeepSeek-V2 does not serve.
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
            (
                "moe_intermediate_size",
                config.moe_intermediate_size,
                ExpertWidth::SIZE,
            ),
            ("num_hidden_layers", config.num_hidden_layers, Self::LAYERS),
            ("num_key_value_heads", config.num_key_value_heads, heads),
            (
                "num_attention_heads",
                config.num_attention_heads,
                heads * <Self as Geometry>::Groups::SIZE,
            ),
            ("kv_lora_rank", config.kv_lora_rank, Latent::SIZE),
            ("qk_nope_head_dim", config.qk_nope_head_dim, Content::SIZE),
            ("qk_rope_head_dim", config.qk_rope_head_dim, Rotary::SIZE),
            (
                "v_head_dim",
                config.v_head_dim,
                <Self as Geometry>::Head::SIZE,
            ),
            (
                "n_routed_experts",
                config.n_routed_experts,
                ExpertCount::SIZE,
            ),
            (
                "n_shared_experts",
                config.n_shared_experts,
                SharedWidth::SIZE / ExpertWidth::SIZE,
            ),
            (
                "num_experts_per_tok",
                config.num_experts_per_tok,
                Selected::SIZE,
            ),
            (
                "first_k_dense_replace",
                config.first_k_dense_replace,
                DENSE_LAYERS,
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
        if config.max_position_embeddings > Tokens::SIZE {
            return Err(unsupported(&format!(
                "max_position_embeddings: past the compiled context of {}",
                Tokens::SIZE
            )));
        }
        let rope = config
            .rope_parameters
            .as_ref()
            .or(config.rope_scaling.as_ref());
        let yarn = rope.is_some_and(|rope| {
            let kind = (rope.get("rope_type").or_else(|| rope.get("type")))
                .and_then(serde_json::Value::as_str);
            let holds = |key: &str, value: f64| {
                rope.get(key).and_then(serde_json::Value::as_f64) == Some(value)
            };
            matches!(kind, Some("yarn" | "deepseek_yarn"))
                && holds("factor", Yarn::FACTOR)
                && holds("beta_fast", Yarn::BETA_FAST)
                && holds("beta_slow", Yarn::BETA_SLOW)
                && holds("mscale", Yarn::MSCALE)
                && holds("mscale_all_dim", Yarn::MSCALE)
                && holds("original_max_position_embeddings", Yarn::POSITIONS as f64)
        });
        let theta =
            (config.rope_theta).or_else(|| rope.and_then(|rope| rope.get("rope_theta")?.as_f64()));
        let not = |declared: Option<&str>, served: &str| declared.is_some_and(|it| it != served);
        let other = |declared: Option<usize>| declared.is_some_and(|it| it != 1);
        let features = [
            (
                !yarn,
                "rope_scaling: DeepSeek-V2 rotates with its compiled YaRN",
            ),
            (
                theta != Some(f64::from(Yarn::THETA)),
                "rope_theta: DeepSeek-V2 rotates its compiled YaRN at 10000",
            ),
            (
                config.q_lora_rank.is_some(),
                "q_lora_rank: DeepSeek-V2 projects its queries at full rank",
            ),
            (
                config.attention_bias,
                "attention_bias: DeepSeek-V2 projects without biases",
            ),
            (
                config.mlp_bias,
                "mlp_bias: DeepSeek-V2's experts project without biases",
            ),
            (
                not(config.hidden_act.as_deref(), "silu"),
                "hidden_act: DeepSeek-V2's experts gate with SiLU",
            ),
            (
                not(config.scoring_func.as_deref(), "softmax"),
                "scoring_func: DeepSeek-V2 routes by softmax probabilities",
            ),
            (
                not(config.topk_method.as_deref(), "greedy"),
                "topk_method: DeepSeek-V2 selects the top experts greedily",
            ),
            (
                other(config.n_group),
                "n_group: DeepSeek-V2 routes over one expert group",
            ),
            (
                other(config.topk_group),
                "topk_group: DeepSeek-V2 routes over one expert group",
            ),
            (
                config.norm_topk_prob,
                "norm_topk_prob: DeepSeek-V2 weighs the selected experts without renormalizing",
            ),
            (
                config
                    .routed_scaling_factor
                    .is_some_and(|factor| factor != 1.0),
                "routed_scaling_factor: DeepSeek-V2 weighs the routed experts unscaled",
            ),
            (
                other(config.moe_layer_freq),
                "moe_layer_freq: every DeepSeek-V2 layer past the dense ones is sparse",
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

type LatentLayer<Projection, Write, Attend, Project, FeedForward, Norm, Add> = transformer::PreNorm<
    attention::Latent<Projection, Write, Attend, Project>,
    FeedForward,
    Norm,
    Add,
>;

/// Source: vLLM `vllm/model_executor/models/AXK1.py`.
/// A layer is routed when experts are configured, it is past `first_k_dense_replace` and its index
/// divides by `moe_layer_freq`; `ORDER` records that, the dense branch first.
pub type AXK1ForCausalLM<
    Embedding,
    Projection,
    Write,
    Attend,
    Project,
    Mlp,
    Router,
    Experts,
    Weighting,
    SharedMlp,
    Add,
    Norm,
    Head,
    const DENSE_LAYERS: usize,
    const ROUTED_LAYERS: usize,
    const ORDER: &'static [crate::Branch],
> = language::CausalLm<
    Embedding,
    Interleave<
        (
            [LatentLayer<Projection, Write, Attend, Project, Mlp, Norm, Add>; DENSE_LAYERS],
            [LatentLayer<
                Projection,
                Write,
                Attend,
                Project,
                feedforward::Moe<feedforward::Routed<Router, Experts, Weighting>, SharedMlp, Add>,
                Norm,
                Add,
            >; ROUTED_LAYERS],
        ),
        ORDER,
    >,
    Norm,
    Head,
>;
