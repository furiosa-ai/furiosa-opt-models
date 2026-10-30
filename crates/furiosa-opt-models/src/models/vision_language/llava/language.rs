//! The language decoder uses the checkpoint's Llama namespace and dense projections.

use crate::models::config::{
    Positive,
    scaling::{Defaults, Spectrum},
};
use crate::models::decoder::attention::Qwen2Layer;
use crate::modules::{attention, feedforward::glu::Glu, language};
use crate::ops::attention::gqa::Read;
use crate::ops::embedding::Embedding;
use crate::ops::index::scatter::Scatter;
use crate::ops::pointwise::activation::Silu;
use crate::ops::pointwise::add::Add;
use crate::ops::pointwise::multiply::Multiply;
use crate::ops::rotary::Rotary;
use crate::ops::swiglu;
use crate::ops::{attention::geometry, linear::Affine, normalization::RmsNorm};
use crate::tensor::axis::Single;
use crate::tensor::cache::Blocks;
use crate::tensor::mapping::Reshape;
use crate::{Module, Sequential};
use furiosa_opt_std::prelude::*;
use serde::Deserialize;
use std::io;

/// Token lookup, causal decoder stack, final normalization, and vocabulary head.
pub type Model =
    language::Decoder<Embedding<Vocabulary, Hidden>, [Layer; LAYERS], Norm, HeadProjection>;

/// Pre-normalized causal attention followed by a gated feed-forward residual.
pub(super) type Layer = Qwen2Layer<
    attention::Projection<
        Projection<QueryWidth, Queries>,
        Projection<KvWidth, Heads>,
        Projection<KvWidth, Heads>,
    >,
    Rotary<Head>,
    Scatter,
    (),
    Attend,
    Linear<QueryWidth, Hidden>,
    FeedForward,
    Norm,
    Add,
>;

type Projection<Width, Count> = Sequential<Linear<Hidden, Width>, Reshape<m![Count, Head]>>;

type Linear<I, O> = Affine<HbmTensor<bf16, m![1], m![O, I]>, ()>;

// The grouped-query kernel distributes KV heads over sixteen lanes; this decoder has thirty-two.
#[derive(Default, DeviceSend)]
pub struct Attend;

impl<Env>
    Module<
        (
            &HbmTensor<bf16, m![1], m![Queries, Head]>,
            &Blocks<Heads, Head>,
            &Read<Windows>,
        ),
        Env,
        &mut (),
    > for Attend
{
    type Output = HbmTensor<bf16, m![1], m![Queries, Head]>;

    fn forward(
        &self,
        _: &mut Device,
        _: Env,
        _: &mut (),
        _: (
            &HbmTensor<bf16, m![1], m![Queries, Head]>,
            &Blocks<Heads, Head>,
            &Read<Windows>,
        ),
    ) -> Self::Output {
        todo!(
            "NPU paged multi-head attention with thirty-two KV heads and a 1/sqrt(128) score scale"
        )
    }
}

type FeedForward = Glu<
    Linear<Hidden, Intermediate>,
    Silu,
    Linear<Hidden, Intermediate>,
    Multiply,
    Linear<Intermediate, Hidden>,
>;

/// RMS normalization with the compiled checkpoint epsilon.
pub(super) type Norm = RmsNorm<Hidden, { EPSILON.to_bits() as usize }>;

/// Bias-free vocabulary projection with singleton row axes removed.
pub(super) type HeadProjection =
    Sequential<crate::Then<Reshape<m![Single, Hidden]>, HeadMatrix>, Reshape<m![Vocabulary]>>;

type HeadMatrix = Affine<
    HbmTensor<bf16, m![1], m![Vocabulary, Hidden]>,
    [HbmTensor<bf16, m![1], m![Vocabulary]>; 0],
>;

/// Number of causal decoder layers.
pub const LAYERS: usize = 32;

/// Width of each language hidden row.
pub type Hidden = geometry::Hidden<4096>;

/// Width of the gated feed-forward intermediate row.
pub type Intermediate = swiglu::Intermediate<11008>;

/// Combined feature width of all query heads.
pub type QueryWidth = geometry::QueryWidth<4096>;

/// Combined feature width of all key/value heads, equal to the query width.
pub type KvWidth = QueryWidth;

// Feature width of one attention head.
axes![Head = 128];

/// Key/value head count, equal to the query head count for this decoder.
pub type Heads = Queries;

axes![Queries = 32, Vocabulary = 32064, Windows = 8];

/// The `rms_norm_eps` the decoder is compiled with.
pub(super) const EPSILON: f32 = 1e-5;

/// `max_position_embeddings`: the positions the decoder serves.
const POSITIONS: usize = 4096;

/// What the language model's `text_config` declares that the size fixes, and the features the
/// decoder does not serve; an absent key holds `LlamaConfig`'s default.
#[derive(Deserialize)]
#[serde(default)]
pub struct Config {
    model_type: String,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    vocab_size: usize,
    rms_norm_eps: f64,
    max_position_embeddings: usize,
    #[serde(flatten, deserialize_with = "Config::spectrum")]
    rope: Spectrum,
    hidden_act: String,
    attention_bias: bool,
    mlp_bias: bool,
    tie_word_embeddings: bool,
}

/// `LlamaConfig`'s: Transformers' default base.
impl Defaults for Config {
    const THETA: Positive = Positive::new(10000.0);
}

/// `LlamaConfig`'s defaults, the model type a LLaVA config's text half names when it names none.
impl Default for Config {
    fn default() -> Self {
        Self {
            model_type: "llama".into(),
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: None,
            head_dim: None,
            vocab_size: 32000,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 2048,
            rope: Self::spectrum(serde_json::Value::Object(Default::default()))
                .expect("a configuration naming no rotary field rotates by its class's base"),
            hidden_act: "silu".into(),
            attention_bias: false,
            mlp_bias: false,
            tie_word_embeddings: false,
        }
    }
}

impl Config {
    /// Refuses a text config the decoder does not serve: a value it does not hold is invalid, a
    /// feature it does not have unsupported.
    pub fn check(&self) -> io::Result<()> {
        let invalid = |reason: String| io::Error::new(io::ErrorKind::InvalidData, reason);
        let unsupported = |reason: &str| io::Error::new(io::ErrorKind::Unsupported, reason);
        if self.model_type != "llama" {
            return Err(invalid(format!(
                "model_type: {}, the decoder is llama",
                self.model_type
            )));
        }
        let heads = self.num_attention_heads;
        let fixed = [
            ("hidden_size", self.hidden_size, Hidden::SIZE),
            (
                "intermediate_size",
                self.intermediate_size,
                Intermediate::SIZE,
            ),
            ("num_hidden_layers", self.num_hidden_layers, LAYERS),
            ("num_attention_heads", heads, Queries::SIZE),
            (
                "num_key_value_heads",
                self.num_key_value_heads.unwrap_or(heads),
                Heads::SIZE,
            ),
            (
                "head_dim",
                self.head_dim.unwrap_or(self.hidden_size / heads.max(1)),
                Head::SIZE,
            ),
        ];
        if let Some((key, declared, held)) = fixed
            .into_iter()
            .find(|(_, declared, held)| declared != held)
        {
            return Err(invalid(format!(
                "{key}: {declared}, the decoder holds {held}"
            )));
        }
        if !(1..=Vocabulary::SIZE).contains(&self.vocab_size) {
            return Err(invalid(format!(
                "vocab_size: {}, the decoder compiles {} rows",
                self.vocab_size,
                Vocabulary::SIZE
            )));
        }
        if self.rms_norm_eps as f32 != EPSILON {
            return Err(invalid(format!(
                "rms_norm_eps: {}, the decoder holds {EPSILON}",
                self.rms_norm_eps
            )));
        }
        if self.tie_word_embeddings {
            return Err(invalid(
                "tie_word_embeddings: true, the decoder's head is its own".into(),
            ));
        }
        if self.max_position_embeddings > POSITIONS {
            return Err(unsupported(
                "max_position_embeddings: past the decoder's 4096 positions",
            ));
        }
        self.rope.bind::<Head, Head>()?;
        match (self.hidden_act.as_str(), self.attention_bias, self.mlp_bias) {
            ("silu", false, false) => Ok(()),
            ("silu", true, _) => Err(unsupported(
                "attention_bias: the decoder projects without biases",
            )),
            ("silu", false, true) => {
                Err(unsupported("mlp_bias: the decoder projects without biases"))
            }
            (_, _, _) => Err(unsupported(
                "hidden_act: the decoder's feed-forward is SiLU",
            )),
        }
    }
}
