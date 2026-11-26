//! Llama 3.1 8B with native E4M3 channel weights, dynamic token activations and BF16 KV.
//! Checkpoint codes are preserved; the catalog reuses the decoder family's layer composition.

use crate::load::{Checkpoint, Load, LoadError};
use crate::models::config::{Exclusion, Exclusions, Scheme, compiled, decoder, scaling::Spectrum};
use crate::models::decoder::attention::{
    Geometry, Head, KvHeads,
    qwen3::{self, launch},
};
use crate::models::decoder::form::TokenScaled;
use crate::models::{Class, Family, Implements};
use crate::specialize::attention::Decode;
use crate::tensor::{axis, memory::Memory};
use furiosa_opt_std::prelude::*;
use std::io;

/// Llama 3.1 with preserved E4M3 channel weights and dynamic per-token E4M3 activations.
/// Source: <https://github.com/vllm-project/vllm/blob/dc1b4a6f1300003ae27f033afbdff5e2683721ce/vllm/model_executor/models/llama.py>
#[derive(DeviceSend, Load)]
pub struct ForCausalLM {
    #[load(rename = "")]
    pub model: launch::Model<Size, TokenScaled>,
}

/// The 8B geometry with Llama 3.1's scaled rotary spectrum and position table.
pub struct Size;
type Original = super::Size<"8B">;
pub type Hidden = <Original as Geometry>::Hidden;
pub type Intermediate = <Original as Geometry>::Intermediate;
pub type Vocabulary = <Original as Geometry>::Vocabulary;
pub type Groups = <Original as Geometry>::Groups;
pub type QWidth = <Original as Geometry>::Queries;
pub const DEPTH: usize = 32;
pub const EPSILON: usize = 1e-5_f32.to_bits() as usize;

impl Geometry for Size {
    const SIZE: &'static str = "8B";
    const FAMILY: &'static str = "llama";
    type Queries = QWidth;
    type Hidden = Hidden;
    type Intermediate = Intermediate;
    type Vocabulary = Vocabulary;
    type KvHeads = KvHeads;
    type Groups = Groups;
    type Head = Head;
    type Layers<L> = [L; DEPTH];
    type Norm<W: AxisName, X: Memory> = <Original as Geometry>::Norm<W, X>;
    const LAYERS: usize = DEPTH;
    const EPSILON: usize = EPSILON;
    const TIED_EMBEDDINGS: bool = false;
}
impl Decode for Size {
    type Rows = <Original as Decode>::Rows;
    type Windows = <Original as Decode>::Windows;
    type Width = <Original as Decode>::Width;
    type Packed = <Original as Decode>::Packed;
    type Work = <Original as Decode>::Work;
    type Capacity = <Original as Decode>::Capacity;
    type Reach = <Original as Decode>::Reach;
    type Sequences = <Original as Decode>::Sequences;
    type Slots = <Original as Decode>::Slots;
    type Positions = axis::Rows<131072>;
}
impl Family for Size {
    type Hf = decoder::Config<Spectrum>;
}
impl TryFrom<decoder::Config<Spectrum>> for Size {
    type Error = io::Error;
    fn try_from(config: decoder::Config<Spectrum>) -> io::Result<Self> {
        let quantization = config.quantization_config.as_ref().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "expected channel-scaled FP8 quantization_config",
            )
        })?;
        if quantization.scheme != Scheme::ChannelToken
            || quantization.kv_cache
            || quantization.excluded != Exclusions(vec![Exclusion::Module("lm_head".into())])
        {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "expected native channel-scaled E4M3 weights, dynamic token activations, an unquantized lm_head, and BF16 KV",
            ));
        }
        let geometry = &config.geometry;
        for (field, actual, expected) in [
            ("hidden_size", geometry.hidden_size.get(), 4096),
            ("intermediate_size", geometry.intermediate_size, 14336),
            ("num_hidden_layers", geometry.num_hidden_layers.get(), DEPTH),
            (
                "num_attention_heads",
                geometry.num_attention_heads.get(),
                32,
            ),
            ("num_key_value_heads", geometry.num_key_value_heads.get(), 8),
            ("head_dim", geometry.head_dim(), 128),
            ("vocab_size", geometry.vocab_size.get(), 128256),
        ] {
            compiled(field, actual, expected)?;
        }
        compiled("rms_norm_eps", geometry.rms_norm_eps.unwrap_or(1e-5), 1e-5)?;
        compiled(
            "tie_word_embeddings",
            config.tie_word_embeddings.unwrap_or(false),
            false,
        )?;
        compiled("sliding_window", config.sliding_window, None)?;
        config.rotary.bind::<Head, Head>()?;
        Ok(Self)
    }
}
impl Implements for ForCausalLM {
    const SIZE: &'static str = "8B";
    const CLASS: Class = Class {
        family: "llama",
        name: "ForCausalLM",
    };
    fn scope(source: &Checkpoint) -> Result<&'static str, LoadError> {
        Size::parse(source.config()).map_err(|error| {
            let name = "config.json".into();
            let reason = error.to_string();
            match error.kind() {
                io::ErrorKind::Unsupported => LoadError::Unsupported { name, reason },
                _ => LoadError::InvalidData { name, reason },
            }
        })?;
        use crate::models::decoder::Form;
        <TokenScaled as Form>::REQUIREMENT.admits(source)?;
        Ok("")
    }
}

impl qwen3::Front for Size {
    type Heads<Projection, Norm> = Projection;
    type HeadNorm<Norm> = ();
}
