//! Llama 3, by published size: Qwen3's classes over a [`Size`] whose front takes each row's query,
//! key and value heads as projected, e.g.
//! `ForCausalLM<Size<"8B">, form::Affine, Parallel<1>, cache::Bf16, X>`.
//!
//! - A size's [`Geometry`] is what its checkpoints declare; its [`Decode`] is how a call of it is
//!   compiled, that of the Qwen3 size of the same attention geometry.
//! - The vocabulary compiles as 131072 rows, `vocab_size` 128256 padded; the head's bias lowers the
//!   padded rows.

use std::io;

use serde::Deserialize;

use super::qwen3::{self, Front};
use super::{Geometry, Head, KvHeads};
use crate::models::{Family, config};
use crate::ops::attention::geometry::{self, HeadRows, ProjectionWidth, QueryGroups, QueryWidth};
use crate::ops::normalization::RmsNorm;
use crate::ops::swiglu;
use crate::specialize::attention::Decode;
use crate::tensor::axis::Limit;
use crate::tensor::form::Weight;
use crate::tensor::memory::Memory;
use furiosa_opt_std::prelude::*;

// 128256 rows leave each of a head's slices 4008, which split only into 167 passes of 24 rows; 131072
// leave 4096, 16 passes of 256.
axes![Vocabulary = 131072, MistralVocabulary = 32768];

/// A published Llama 3 checkpoint size, by its hub spelling.
pub struct Size<const N: &'static str>;

/// HF `LlamaForCausalLM`: the base model, then the greedy vocabulary head.
pub type ForCausalLM<S, P, Q, C, X> = qwen3::ForCausalLM<S, P, Q, C, X>;

/// HF `LlamaModel`: the embedding, the layers and the final norm, and the constants that prepare a
/// call's attention context.
pub type Model<S, P, Q, C, X> = qwen3::Model<S, P, Q, C, X>;

/// HF `LlamaDecoderLayer`: one element of `model.layers`, a subtree that loads alone.
pub type DecoderLayer<S, P, Q, C, X> = qwen3::DecoderLayer<S, P, Q, C, X>;

/// The norm of every Llama 3 size: `rms_norm_eps` 1e-5.
type LlamaNorm<W, X> =
    RmsNorm<W, { EPSILON }, m![1], Weight<m![W], m![1], <X as Memory>::Tensor<bf16, m![1], m![W]>>>;

const EPSILON: usize = 1e-5_f32.to_bits() as usize;

impl Geometry for Size<"8B"> {
    const SIZE: &'static str = "8B";
    const FAMILY: &'static str = "llama";
    type Queries = QueryWidth<4096>;
    type Hidden = geometry::Hidden<4096>;
    type Intermediate = swiglu::Intermediate<14336>;
    type Vocabulary = Vocabulary;
    type KvHeads = KvHeads;
    type Groups = QueryGroups<4>;
    type Head = Head;
    type Layers<L> = [L; Self::LAYERS];
    type Norm<W: AxisName, X: Memory> = LlamaNorm<W, X>;

    const LAYERS: usize = 32;
    const EPSILON: usize = EPSILON;
    const TIED_EMBEDDINGS: bool = false;
}

impl Decode for Size<"8B"> {
    type Rows = Limit<64, 64>;
    type Windows = <qwen3::Size<"8B"> as Decode>::Windows;
    type Width = ProjectionWidth<6144>;
    type Packed = m![KvHeads, { Symbol<HeadRows<6>> }, Head, { Symbol<geometry::Hidden<4096>> }];
    type Work = <qwen3::Size<"8B"> as Decode>::Work;
    type Capacity = <qwen3::Size<"8B"> as Decode>::Capacity;
    type Reach = <qwen3::Size<"8B"> as Decode>::Reach;
    type Sequences = <qwen3::Size<"8B"> as Decode>::Sequences;
    type Slots = <qwen3::Size<"8B"> as Decode>::Slots;
}

/// Mistral 7B, whose tree is Llama's: Llama 3 8B's geometry over a 32768-token vocabulary.
impl Geometry for Size<"Mistral-7B"> {
    const SIZE: &'static str = "Mistral-7B";
    const FAMILY: &'static str = "mistral";
    type Queries = QueryWidth<4096>;
    type Hidden = geometry::Hidden<4096>;
    type Intermediate = swiglu::Intermediate<14336>;
    type Vocabulary = MistralVocabulary;
    type KvHeads = KvHeads;
    type Groups = QueryGroups<4>;
    type Head = Head;
    type Layers<L> = [L; Self::LAYERS];
    type Norm<W: AxisName, X: Memory> = LlamaNorm<W, X>;

    const LAYERS: usize = 32;
    const EPSILON: usize = EPSILON;
    const TIED_EMBEDDINGS: bool = false;
}

impl Decode for Size<"Mistral-7B"> {
    type Rows = <Size<"8B"> as Decode>::Rows;
    type Width = <Size<"8B"> as Decode>::Width;
    type Packed = <Size<"8B"> as Decode>::Packed;
    type Work = <Size<"8B"> as Decode>::Work;
    type Windows = <Size<"8B"> as Decode>::Windows;
    type Capacity = <Size<"8B"> as Decode>::Capacity;
    type Reach = <Size<"8B"> as Decode>::Reach;
    type Sequences = <Size<"8B"> as Decode>::Sequences;
    type Slots = <Size<"8B"> as Decode>::Slots;
}

/// Llama projects each row's query, key and value heads and rotates them unnormalized.
impl<const N: &'static str> Front for Size<N> {
    type Heads<Projection, Norm> = Projection;
    type HeadNorm<Norm> = ();
}

/// What a Llama checkpoint's config declares that its size fixes, and the features the sizes do
/// not serve.
#[derive(Deserialize)]
pub struct Config {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: Option<usize>,
    vocab_size: usize,
    rms_norm_eps: f64,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(flatten, deserialize_with = "config::flattened")]
    _rotary: config::decoder::Rotary,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    #[serde(default)]
    sliding_window: Option<usize>,
}

/// A config is of a size when it declares the size's values, at most its compiled vocabulary rows,
/// and none of the features the size does not serve. A longer declared context is served up to the
/// compiled one, the window the paging constants hold.
impl<const N: &'static str> TryFrom<Config> for Size<N>
where
    Self: Geometry,
{
    type Error = io::Error;

    fn try_from(config: Config) -> io::Result<Self> {
        let invalid = |reason: String| io::Error::new(io::ErrorKind::InvalidData, reason);
        let unsupported = |reason: &str| io::Error::new(io::ErrorKind::Unsupported, reason);
        let heads = <Self as Geometry>::KvHeads::SIZE;
        // Transformers derives an absent `head_dim` from the hidden width.
        let head =
            (config.head_dim).unwrap_or(config.hidden_size / config.num_attention_heads.max(1));
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
            ("head_dim", head, <Self as Geometry>::Head::SIZE),
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
        match (
            config.attention_bias || config.mlp_bias,
            config.sliding_window,
        ) {
            (true, _) => Err(unsupported(
                "attention_bias, mlp_bias: Llama projects without biases",
            )),
            (_, Some(_)) => Err(unsupported(
                "sliding_window: the size attends the whole history",
            )),
            (false, None) => Ok(Size),
        }
    }
}

impl<const N: &'static str> Family for Size<N>
where
    Self: Geometry,
{
    type Hf = Config;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::{Checkpoint, Load};
    use crate::models::decoder::form;
    use crate::tensor::cache;
    use crate::tensor::memory::Hbm;
    use crate::tensor::parallel::Parallel;

    /// Llama-3-8B's published config selects the size; a scaled spectrum is refused.
    #[test]
    fn parses_the_published_config() {
        let mut config = serde_json::json!({
            "hidden_size": 4096, "intermediate_size": 14336, "num_hidden_layers": 32,
            "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 128256,
            "rms_norm_eps": 1e-5, "max_position_embeddings": 8192, "rope_scaling": null,
            "rope_theta": 500000.0, "tie_word_embeddings": false, "attention_bias": false,
        });
        assert!(Size::<"8B">::parse(&config).is_ok());
        config["rope_scaling"] = serde_json::json!({"rope_type": "llama3", "factor": 8.0});
        let error = Size::<"8B">::parse(&config).err().expect("refused");
        assert_eq!(error.kind(), io::ErrorKind::Unsupported);
    }

    /// Mistral-7B-v0.3's published config selects the Mistral size; v0.1's sliding window is
    /// refused.
    #[test]
    fn parses_the_published_mistral_config() {
        let mut config = serde_json::json!({
            "hidden_size": 4096, "intermediate_size": 14336, "num_hidden_layers": 32,
            "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 32768,
            "rms_norm_eps": 1e-5, "max_position_embeddings": 32768, "rope_theta": 1000000.0,
            "sliding_window": null, "tie_word_embeddings": false,
        });
        assert!(Size::<"Mistral-7B">::parse(&config).is_ok());
        config["sliding_window"] = 4096.into();
        let error = Size::<"Mistral-7B">::parse(&config).err().expect("refused");
        assert_eq!(error.kind(), io::ErrorKind::Unsupported);
    }

    /// The causal model and the BF16 instance of the catalog load from a checkpoint.
    #[test]
    fn causal_model_loads() {
        fn loads<T: for<'a> Load<&'a Checkpoint>>() {}
        loads::<ForCausalLM<Size<"8B">, form::Affine, Parallel<1>, cache::Bf16, Hbm>>();
        loads::<qwen3::Catalog<Size<"8B">, form::Affine, bf16, Hbm>>();
        loads::<ForCausalLM<Size<"Mistral-7B">, form::Affine, Parallel<1>, cache::Bf16, Hbm>>();
        loads::<qwen3::Catalog<Size<"Mistral-7B">, form::Affine, bf16, Hbm>>();
    }
}
