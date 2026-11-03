//! Llama 3.1 8B with NVFP4 weights, BF16 activations and BF16 KV cache.
//! The catalog tree uses PR #457's resident QKV, tiled MLP and input-partitioned O kernels.

use crate::load::{Checkpoint, Context, Load, LoadError, Named};
use crate::models::config::{Exclusion, Exclusions, Scheme, compiled, decoder, scaling::Spectrum};
use crate::models::decoder::{
    attention::{
        self, Geometry, Head, KvHeads,
        qwen3::{self, launch},
    },
    form,
};
use crate::models::{Class, Family, Implements};
use crate::modules::attention::Projection;
use crate::modules::generic::collection::Stack;
use crate::ops::attention::full;
use crate::ops::index::{gather::Gather, scatter::Scatter};
use crate::ops::linear::nvfp4::tiled::{mlp::Mlp, output::Output, panels::Panels};
use crate::ops::rotary::Rotary;
use crate::ops::{embedding::Embedding, normalization::RmsNorm, pointwise::add::Add};
use crate::specialize::attention::Decode;
use crate::tensor::{
    axis,
    form::{Weight, nvfp4::W4A16},
    memory::{Hbm, Memory},
    parallel::Chip,
    scalar::F32,
};
use furiosa_opt_std::prelude::*;
use std::{io, marker::PhantomData};

/// The served causal model: a native catalog tree and validated, unused cache calibration.
#[derive(DeviceSend, Load)]
pub struct ForCausalLM {
    #[load(rename = "")]
    pub model: launch::Over<Size, Body, form::Nvfp4BlockScaled<W4A16>>,
    #[load(rename = "", with = calibration)]
    pub calibration: Calibration,
}

/// Source: <https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/models/llama/modeling_llama.py>.
pub type Catalog = attention::Qwen2ForCausalLM<
    Named<
        "embed_tokens",
        Embedding<Vocabulary, Hidden, Chip, Weight<m![Vocabulary, Hidden], Chip>>,
    >,
    Qkv,
    Rotary<Head, Head, bf16>,
    Scatter,
    Gather,
    full::Attention<PhantomData<(KvHeads, Groups, Head)>, F32<SCALE>>,
    Named<"o_proj", Output<QWidth, Hidden>>,
    Mlp<Hidden, Intermediate>,
    Norm,
    Add,
    qwen3::LmHead<Size, Hbm>,
    DEPTH,
    Stack,
>;
pub type Body = Catalog::Body;
pub type Qkv = Projection<
    Named<"q_proj", Panels<Hidden, QWidth>>,
    Named<"k_proj", Panels<Hidden, KvWidth>>,
    Named<"v_proj", Panels<Hidden, KvWidth>>,
>;
pub type Norm = RmsNorm<Hidden, EPSILON, Chip>;

/// The 8B geometry with Llama 3.1's scaled rotary spectrum and position table.
pub struct Size;
type Original = super::Size<"8B">;
pub type Hidden = <Original as Geometry>::Hidden;
pub type Intermediate = <Original as Geometry>::Intermediate;
pub type Vocabulary = <Original as Geometry>::Vocabulary;
pub type Groups = <Original as Geometry>::Groups;
pub type QWidth = <Original as Geometry>::Queries;
axes![KvWidth = 1024, QueryHeads = 32];
pub const DEPTH: usize = 32;
pub const EPSILON: usize = 1e-5_f32.to_bits() as usize;
const SCALE: usize = crate::ops::attention::geometry::scale(Head::SIZE);

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
                "expected NVFP4 quantization_config",
            )
        })?;
        if quantization.scheme != Scheme::Nvfp4
            || !quantization.kv_cache
            || quantization.excluded != Exclusions(vec![Exclusion::Module("lm_head".into())])
        {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "expected NVFP4 projections, an unquantized lm_head, and checkpoint FP8 cache calibration (served with BF16 KV)",
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
        <form::Nvfp4BlockScaled<W4A16> as Form>::REQUIREMENT.admits(source)?;
        Ok("")
    }
}

/// Validate both calibrated cache scales of every layer. BF16 KV uses neither scale.
#[derive(DeviceSend, Load)]
pub struct Calibration;

fn calibration(from: &Checkpoint, cx: &mut Context<'_>) -> Result<Calibration, LoadError> {
    for layer in 0..DEPTH {
        for kind in ["k", "v"] {
            let path = format!("model.layers.{layer}.self_attn.{kind}_proj");
            cx.field("cache", &path, |cx| {
                let leaf = if kind == "k" { "k_scale" } else { "v_scale" };
                crate::tensor::form::nvfp4::scales(from, cx, [leaf]).map(drop)
            })?;
        }
    }
    Ok(Calibration)
}
