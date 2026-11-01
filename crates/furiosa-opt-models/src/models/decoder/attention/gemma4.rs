//! Gemma 4's dense text model: a scaled token embedding, sliding and full attention layers
//! interleaved in checkpoint order, a final norm, and a greedy head tied to the embedding.
//!
//! - A [`Size`] is a published size; its [`Geometry`] holds the model-wide axes, and its layers
//!   attend in [`sliding::Group`] or [`full::Group`], each a [`decoder::Attention`] of its own
//!   [`LayerType`].
//! - [`ForCausalLM`] and [`Model`] are the shared roots over [`Catalog`], the one binding of the
//!   [`Gemma4ForCausalLM`] recipe: its body is the base model's, its head the causal model's.
//! - A checkpoint keeps the text model under `model.language_model`, beside the media embedders,
//!   and its config in `text_config`; [`Quantized`] holds its projections as it stores them.

use super::{Geometry, LayerType};
use crate::load::{self, Checkpoint, Load, LoadError, Named};
use crate::models::config::decoder::Language;
use crate::models::config::{
    self, Constant, Disabled, Enabled, Literal, Positive, compiled, scaling::Spectra,
};
use crate::models::decoder::{self, Attention, Forms, Rotation, form};
use crate::models::{Family, vocabulary};
use crate::modules::attention;
use crate::modules::language::{self, Greedy};
use crate::modules::{multimodal, transformer};
use crate::ops::attention::gqa::Cache;
use crate::ops::embedding::Embedding;
use crate::ops::index::scatter::Scatter;
use crate::ops::linear::Affine;
use crate::ops::normalization::{RmsNorm, UnitRmsNorm};
use crate::ops::pointwise::activation::{NewGelu, Softcap};
use crate::ops::pointwise::{add, multiply};
use crate::ops::reduction::Argmax;
use crate::tensor::axis;
use crate::tensor::cache;
use crate::tensor::form::Weight;
use crate::tensor::memory::{Hbm, Memory};
use crate::tensor::parallel::{Chip, Parallel};
use crate::{Branch, Op, Sequential, Then};
use furiosa_opt_std::prelude::*;
use serde::Deserialize;
use std::{io, num::NonZeroUsize};

/// A published Gemma 4 checkpoint size, by its hub spelling.
pub struct Size<const N: &'static str>;

/// HF `Gemma4ForCausalLM`: the text model, then the greedy head tied to its embedding, in
/// projection forms `P`, a cache of `C` and memory `X`, for calls of `R` rows.
pub type ForCausalLM<P = Quantized, C = cache::Bf16, X = Hbm, R = Rows> =
    super::ForCausalLM<Size<"12B">, Model<P, C, X, R>, LmHead<X>, P>;

/// HF `Gemma4TextModel`: the embedding, the layers and the final norm, and the constants that
/// prepare a call's attention context in each group.
pub type Model<P = Quantized, C = cache::Bf16, X = Hbm, R = Rows> =
    super::Model<Size<"12B">, Body<P, C, X>, RotaryEmbedding<C, X, R>, P>;

/// The embedding, the layers and the final norm: the body of the catalog binding.
pub type Body<P = Quantized, C = cache::Bf16, X = Hbm> = Catalog<P, C, X>::Body;

/// The one binding of the [`Gemma4ForCausalLM`] recipe over the published size's axes, its
/// projections in forms `P`, its cache of `C` and every leaf in memory `X`.
pub type Catalog<P, C, X> = Gemma4ForCausalLM<
    Then<Named<"embed_tokens", Table<X>>, Normalizer<X>>,
    Qkv<sliding::Group, P, X, Named<"v_proj", Values<sliding::Group, P, X>>>,
    decoder::Position<sliding::Group, C>,
    decoder::Attend<sliding::Group, UNSCALED>,
    Output<sliding::Group, P, X>,
    Qkv<full::Group, P, X, Named<"k_proj", Keys<full::Group, P, X>>>,
    decoder::Position<full::Group, C>,
    decoder::Attend<full::Group, UNSCALED>,
    Output<full::Group, P, X>,
    Scatter,
    (),
    decoder::FeedForward<P, Hidden, Intermediate, NewGelu, Parallel<1>, X>,
    Norm<Hidden, X>,
    add::Add,
    Named<"layer_scalar", Scale<X>>,
    LmHead<X>,
    SLIDING,
    FULL,
    ORDER,
>;

/// The greedy head over the vocabulary rows: the projection by the tied embedding table, whose
/// bias lowers no row, and the first maximum of each row's scores, each capped at the config's
/// `final_logit_softcapping`.
pub type LmHead<X = Hbm> = Greedy<
    Hidden,
    Vocabulary,
    Affine<
        <X as Memory>::Tensor<bf16, Chip, m![Vocabulary, Hidden]>,
        <X as Memory>::Tensor<f32, Chip, m![Vocabulary]>,
    >,
    Argmax<Vocabulary, Chip, <X as Memory>::Tensor<i32, Chip, m![Vocabulary]>>,
    Softcap<<X as Memory>::Tensor<f32, Chip, m![1 # 8]>>,
>;

/// The token table.
type Table<X> = Embedding<
    Vocabulary,
    Hidden,
    Chip,
    Weight<m![Vocabulary, Hidden], Chip, <X as Memory>::Tensor<bf16, Chip, m![Vocabulary, Hidden]>>,
>;

/// Group `A`'s query, key and value heads: queries and keys normalized by learned gains, and the
/// heads of value projection `V` by unit gain.
type Qkv<A, P, X, V> = attention::NormalizedQkv<
    attention::Projection<
        Named<"q_proj", Heads<<P as Forms>::Query, <A as Attention>::QueryWidth, QueryView<A>, X>>,
        Named<"k_proj", Keys<A, P, X>>,
        Then<V, UnitNorm<<A as Attention>::Head, X>>,
    >,
    Named<"q_norm", Norm<<A as Attention>::Head, X>>,
    Named<"k_norm", Norm<<A as Attention>::Head, X>>,
>;

/// Group `A`'s query heads, each key/value head's together, as the call's rows read them.
type QueryView<A> = <Rows as decoder::Envelope>::Queries<
    Queries,
    <A as Attention>::Heads,
    <A as Attention>::Groups,
    <A as Attention>::Head,
>;

/// Group `A`'s key heads; a full group's values are the same projection's heads, as its
/// checkpoint stores keys alone.
type Keys<A, P, X> = Heads<<P as Forms>::Key, <A as Attention>::KvWidth, KvView<A>, X>;

/// Group `A`'s value heads.
type Values<A, P, X> = Heads<<P as Forms>::Value, <A as Attention>::KvWidth, KvView<A>, X>;

/// Group `A`'s key/value heads.
type KvView<A> = decoder::HeadView<<A as Attention>::Heads, <A as Attention>::Head>;

/// Heads read as `View`, projected from a row in form `F` to `Width` features.
type Heads<F, Width, View, X> = decoder::Heads<F, Hidden, Width, View, bf16, Parallel<1>, X>;

/// Group `A`'s attended heads projected back to rows.
type Output<A, P, X> = Named<"o_proj", decoder::Output<P, Hidden, A, Parallel<1>, X>>;

/// The norm of every Gemma 4 layer and head: `rms_norm_eps` 1e-6.
type Norm<W, X> = <Size<"12B"> as Geometry>::Norm<W, X>;

/// A norm of unit gain, whose checkpoint stores no weight.
type UnitNorm<W, X> = UnitRmsNorm<W, EPSILON, m![1], <X as Memory>::Tensor<bf16, m![1], m![W]>>;

/// Multiplies rows by one FP32 value held in memory `X`.
type Scale<X> = multiply::Scale<<X as Memory>::Tensor<f32, Chip, m![1 # 8]>>;

/// The constants that prepare a call's rotations, write slots and paged history reads, one set
/// a group, for calls of `R` rows and a cache of `C`.
type RotaryEmbedding<C, X, R> = (
    decoder::Paging<sliding::Group, R, C, X>,
    decoder::Paging<full::Group, R, C, X>,
);

/// The rows a call prepares its attention context for.
pub type Rows = axis::Rows<64>;

/// Normalized queries and keys make scores need no multiplier.
const UNSCALED: usize = 1.0_f32.to_bits() as usize;

/// `rms_norm_eps` of the published size, as the bits of the FP32 nearest to it.
const EPSILON: usize = 1e-6_f32.to_bits() as usize;

/// Sliding and full attention layer counts of the published size.
pub const SLIDING: usize = 40;
pub const FULL: usize = 8;

/// Every sixth layer of the published size attends over its full history.
pub const ORDER: &[Branch] = &Branch::periodic::<{ SLIDING + FULL }>(6);

/// The layer types a checkpoint names its sliding and full attention layers by.
const LAYER_TYPES: [&str; 2] = ["sliding_attention", "full_attention"];

axes![
    Hidden = 3840,
    Intermediate = 15360,
    Vocabulary = 262144,
    Queries = 16
];

/// FP8 per-channel attention projections and an NVFP4 feed-forward, as the published checkpoint
/// stores them.
pub type Quantized = decoder::Projections<
    form::ChannelScaled,
    form::ChannelScaled,
    form::ChannelScaled,
    form::ChannelScaled,
    form::Nvfp4,
>;

/// Sliding attention geometry: 256-wide heads and three cache windows.
pub mod sliding {
    use crate::models::decoder;
    use furiosa_opt_std::prelude::*;

    axes![
        Head = 256,
        Heads = 8,
        Groups = 2,
        QueryWidth = 4096,
        KvWidth = 2048,
        Windows = 3
    ];

    pub type Group = decoder::Group<Heads, Head, Groups, QueryWidth, KvWidth, Head, Windows>;
}

/// Full attention geometry: 512-wide heads, 128 rotated coordinates and 512 cache windows.
pub mod full {
    use crate::models::decoder;
    use furiosa_opt_std::prelude::*;

    axes![
        Head = 512,
        Heads = 1,
        Groups = 16,
        QueryWidth = 8192,
        KvWidth = 512,
        Rotated = 128,
        Windows = 512
    ];

    pub type Group = decoder::Group<Heads, Head, Groups, QueryWidth, KvWidth, Rotated, Windows>;
}

impl LayerType for sliding::Group {
    const NAME: Option<&'static str> = Some(LAYER_TYPES[0]);
    const WINDOW: Option<&'static str> = Some("sliding_window");
}

impl LayerType for full::Group {
    const NAME: Option<&'static str> = Some(LAYER_TYPES[1]);
}

// Each group's widths are its heads' products, and both groups hold the model's query heads.
const _: () = assert!(
    <sliding::Group as Attention>::CONSISTENT
        && <full::Group as Attention>::CONSISTENT
        && sliding::Heads::SIZE * sliding::Groups::SIZE == Queries::SIZE
        && full::Heads::SIZE * full::Groups::SIZE == Queries::SIZE
        && ORDER.len() == SLIDING + FULL
);

impl Geometry for Size<"12B"> {
    type Hidden = Hidden;
    type Intermediate = Intermediate;
    type Vocabulary = Vocabulary;
    type Layers<L> = [L; Self::LAYERS];
    type Norm<W: AxisName, X: Memory> =
        RmsNorm<W, EPSILON, m![1], Weight<m![W], m![1], <X as Memory>::Tensor<bf16, m![1], m![W]>>>;

    const SIZE: &'static str = "12B";
    const FAMILY: &'static str = "gemma4";
    const LAYERS: usize = SLIDING + FULL;
    const EPSILON: usize = EPSILON;
    const TIED_EMBEDDINGS: bool = true;
}

/// A checkpoint keeps the text model beside the media embedders, which no text class reads.
impl Family for Size<"12B"> {
    type Hf = Hf;

    const MODEL: &'static str = "model.language_model";
    const CONFIG: &'static str = "text_config";
    const IGNORE: &'static [&'static str] = &[
        "model.vision_embedder",
        "model.embed_vision",
        "model.embed_audio",
    ];
}

/// Every embedded row's multiplier: the BF16 rounding of `sqrt(hidden_size)`, which Transformers'
/// scaled embedding computes and no checkpoint stores.
#[derive(DeviceSend, Load)]
#[load(tree)]
pub struct Normalizer<X: Memory = Hbm>(Scale<X>);

/// The multiplier a load places from the config's `hidden_size`.
impl<'a, X: Memory> Load<&'a Checkpoint> for Normalizer<X>
where
    <X as Memory>::Tensor<f32, Chip, m![1 # 8]>: Load<HostTensor<f32, m![{ Chip }, { m![1 # 8] }]>>,
{
    fn load(context: &mut load::Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let (_, hidden) = cx.config_number(from, "hidden_size")?;
            let factor = bf16::from_f32((hidden as f32).sqrt()).to_f32();
            let host: HostTensor<f32, m![{ Chip }, { m![1 # 8] }]> =
                HostTensor::from_vec([factor, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
            let scale = cx.field("0", "", |cx| cx.field("0", "", |cx| Load::load(cx, host)));
            scale.map(|factor| Self(multiply::Scale(factor)))
        })
    }
}

/// The multiplier is no update target.
impl<'t, 's, X: Memory> Load<(&'t mut Normalizer<X>, &'s Checkpoint)> for &'t mut Normalizer<X> {
    fn load(
        context: &mut load::Context<'_>,
        (into, from): (&'t mut Normalizer<X>, &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |_| Ok(()))?;
        Ok(into)
    }
}

impl<I, X: Memory> Op<I> for Normalizer<X>
where
    Scale<X>: Op<I>,
{
    type Output = <Scale<X> as Op<I>>::Output;

    fn forward(&self, device: &mut Device, input: I) -> Self::Output {
        Op::forward(&self.0, device, input)
    }
}

/// A Gemma 4 checkpoint's text configuration.
#[derive(Deserialize)]
pub struct Hf {
    #[serde(flatten, deserialize_with = "config::text")]
    text: Language<Text>,
}

/// The fields only Gemma 4's text configuration has, at the sizes the published size compiles.
#[derive(Deserialize)]
struct Text {
    hidden_size: config::Size<Hidden>,
    intermediate_size: config::Size<Intermediate>,
    num_attention_heads: config::Size<Queries>,
    vocab_size: usize,
    hidden_activation: Literal<"gelu_pytorch_tanh">,
    attention_bias: Disabled,
    enable_moe_block: Disabled,
    use_double_wide_mlp: Disabled,
    num_kv_shared_layers: Constant<0>,
    hidden_size_per_layer_input: Constant<0>,
    attention_k_eq_v: Enabled,
    // Transformers' Gemma 4 configurations default to a tied head.
    tie_word_embeddings: Option<Enabled>,
    final_logit_softcapping: Positive,
    sliding_window: NonZeroUsize,
    #[serde(flatten, deserialize_with = "config::flattened")]
    spectra: Spectra,
}

/// A config is of the 12B size when it declares its geometry, layer order and head constants, a
/// rotary spectrum each group binds, at most its vocabulary rows and a sliding window within the
/// compiled cache windows. A longer declared context is served up to the compiled one.
impl TryFrom<Hf> for Size<"12B"> {
    type Error = io::Error;

    fn try_from(Hf { text }: Hf) -> io::Result<Self> {
        let geometry = &text.geometry;
        let Text {
            hidden_activation: Literal,
            attention_bias: Disabled,
            enable_moe_block: Disabled,
            use_double_wide_mlp: Disabled,
            num_kv_shared_layers: Constant,
            hidden_size_per_layer_input: Constant,
            attention_k_eq_v: Enabled,
            tie_word_embeddings: None | Some(Enabled),
            hidden_size: config::Size { .. },
            intermediate_size: config::Size { .. },
            num_attention_heads: config::Size { .. },
            final_logit_softcapping: Positive { .. },
            ..
        } = text.extra;
        vocabulary::<Vocabulary>(text.extra.vocab_size)?;
        let [sliding, full] = LAYER_TYPES;
        let (own, global) = (text.group(sliding), text.group(full));
        for (field, found, expected) in [
            (
                "num_hidden_layers",
                geometry.num_hidden_layers.get(),
                ORDER.len(),
            ),
            (
                "num_key_value_heads",
                own.num_key_value_heads,
                sliding::Heads::SIZE,
            ),
            ("head_dim", own.head_dim, sliding::Head::SIZE),
            (
                "num_global_key_value_heads",
                global.num_key_value_heads,
                full::Heads::SIZE,
            ),
            ("global_head_dim", global.head_dim, full::Head::SIZE),
        ] {
            compiled(field, found, expected)?;
        }
        compiled("rms_norm_eps", geometry.rms_norm_eps, Some(1e-6))?;
        let order: Vec<String> = ORDER
            .iter()
            .map(|branch| LAYER_TYPES[*branch as usize].to_owned())
            .collect();
        compiled("layer_types", text.layer_types(), order.as_slice())?;
        let window = text.extra.sliding_window.get();
        // An unaligned start reserves one cache window for the partial prefix.
        let maximum = (sliding::Windows::SIZE - 1) * Cache::SIZE;
        if window > maximum {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("sliding_window: found {window}, compiled maximum {maximum}"),
            ));
        }
        let within = |layer: &'static str| {
            move |error: io::Error| {
                io::Error::new(error.kind(), format!("rope_parameters.{layer}: {error}"))
            }
        };
        let spectra = &text.extra.spectra;
        <sliding::Group as Attention>::Rotated::spectrum::<sliding::Head>(spectra.of(sliding)?)
            .map_err(within(sliding))?;
        <full::Group as Attention>::Rotated::spectrum::<full::Head>(spectra.of(full)?)
            .map_err(within(full))?;
        Ok(Self)
    }
}

/// Source: Transformers `src/transformers/models/gemma4/modeling_gemma4.py`, line 1985.
pub(super) type Vision<
    Patch,
    PatchPosition,
    Qkv,
    QueryNorm,
    KeyNorm,
    ValueNorm,
    Position,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Pool,
    Select,
    Standardize,
    const DEPTH: usize,
> = language::Encoder<
    Then<Patch, PatchPosition>,
    [transformer::SandwichNorm<
        attention::Full<
            Then<Sequential<Qkv, (QueryNorm, KeyNorm, ValueNorm)>, Position>,
            Attend,
            Project,
        >,
        Mlp,
        Norm,
        Add,
    >; DEPTH],
    Then<Pool, Then<Select, Standardize>>,
>;

/// Source: vLLM `vllm/model_executor/models/gemma4_mm.py`.
pub type Gemma4ForConditionalGeneration<
    Embedding,
    Patch,
    PatchPosition,
    VisionQkv,
    VisionQueryNorm,
    VisionKeyNorm,
    VisionValueNorm,
    VisionPosition,
    VisionAttend,
    VisionProject,
    VisionMlp,
    VisionNorm,
    VisionAdd,
    VisionPool,
    VisionSelect,
    VisionStandardize,
    ProjectMedia,
    MergeMedia,
    SlidingQkv,
    SlidingPosition,
    SlidingAttend,
    SlidingProject,
    FullQkv,
    FullPosition,
    FullAttend,
    FullProject,
    Write,
    Read,
    Mlp,
    Norm,
    Add,
    Scale,
    Head,
    const SLIDING_LAYERS: usize,
    const FULL_LAYERS: usize,
    const ORDER: &'static [Branch],
    const VISION_DEPTH: usize,
> = language::Hybrid<
    multimodal::InputEmbedding<
        Embedding,
        Vision<
            Patch,
            PatchPosition,
            VisionQkv,
            VisionQueryNorm,
            VisionKeyNorm,
            VisionValueNorm,
            VisionPosition,
            VisionAttend,
            VisionProject,
            VisionMlp,
            VisionNorm,
            VisionAdd,
            VisionPool,
            VisionSelect,
            VisionStandardize,
            VISION_DEPTH,
        >,
        ProjectMedia,
        MergeMedia,
    >,
    [Then<
        transformer::SandwichNorm<
            attention::Cached<
                SlidingQkv,
                SlidingPosition,
                Write,
                Read,
                SlidingAttend,
                SlidingProject,
            >,
            Mlp,
            Norm,
            Add,
        >,
        Scale,
    >; SLIDING_LAYERS],
    [Then<
        transformer::SandwichNorm<
            attention::Cached<FullQkv, FullPosition, Write, Read, FullAttend, FullProject>,
            Mlp,
            Norm,
            Add,
        >,
        Scale,
    >; FULL_LAYERS],
    Norm,
    Head,
    ORDER,
>;

/// Source: vLLM `vllm/model_executor/models/gemma4.py`.
/// The dense decoder: sliding and full attention layers, each normalized before and after both
/// branches and scaled by its `layer_scalar`; per-layer inputs, shared KV and experts are absent.
pub type Gemma4ForCausalLM<
    Embedding,
    SlidingQkv,
    SlidingPosition,
    SlidingAttend,
    SlidingProject,
    FullQkv,
    FullPosition,
    FullAttend,
    FullProject,
    Write,
    Read,
    Mlp,
    Norm,
    Add,
    Scale,
    Head,
    const SLIDING_LAYERS: usize,
    const FULL_LAYERS: usize,
    const ORDER: &'static [crate::Branch],
> = language::Hybrid<
    Embedding,
    [crate::Then<
        transformer::SandwichNorm<
            attention::Cached<
                SlidingQkv,
                SlidingPosition,
                Write,
                Read,
                SlidingAttend,
                SlidingProject,
            >,
            Mlp,
            Norm,
            Add,
        >,
        Scale,
    >; SLIDING_LAYERS],
    [crate::Then<
        transformer::SandwichNorm<
            attention::Cached<FullQkv, FullPosition, Write, Read, FullAttend, FullProject>,
            Mlp,
            Norm,
            Add,
        >,
        Scale,
    >; FULL_LAYERS],
    Norm,
    Head,
    ORDER,
>;

#[cfg(test)]
pub(in crate::models) mod tests {
    use super::*;
    use crate::models::Implements;
    use serde_json::{Value, json};

    /// The published checkpoint's `quantization_config`, as `unsloth/gemma-4-12b-it-NVFP4` writes
    /// it: FP8 attention projections per channel, an NVFP4 feed-forward and an E4M3 cache.
    pub(in crate::models) fn quantization() -> Value {
        let unset = json!({
            "actorder": null, "block_structure": null, "group_size": null, "observer": null,
            "observer_kwargs": {}, "scale_dtype": null, "zp_dtype": null,
        });
        let with = |fields: Value| {
            let mut values = unset.clone();
            (values.as_object_mut().unwrap()).extend(fields.as_object().unwrap().clone());
            values
        };
        let fp8 = |dynamic: bool, strategy: &str| {
            with(json!({
                "num_bits": 8, "type": "float", "symmetric": true, "dynamic": dynamic,
                "strategy": strategy, "observer": "memoryless_minmax",
            }))
        };
        let nvfp4 = |dynamic: Value, actorder: Value| {
            with(json!({
                "num_bits": 4, "type": "float", "symmetric": true, "dynamic": dynamic,
                "strategy": "tensor_group", "group_size": 16, "actorder": actorder,
                "scale_dtype": "torch.float8_e4m3fn", "observer": "static_minmax",
            }))
        };
        json!({
            "config_groups": {
                "group_0": {
                    "format": "float-quantized", "output_activations": null,
                    "targets": ["re:.*self_attn\\.(q|k|v|o)_proj$"],
                    "weights": fp8(false, "channel"), "input_activations": fp8(true, "token"),
                },
                "group_1": {
                    "format": "nvfp4-pack-quantized", "output_activations": null,
                    "targets": ["re:.*language_model.*\\.mlp\\.(gate|up|down)_proj$"],
                    "weights": nvfp4(json!(false), json!("static")),
                    "input_activations": nvfp4(json!("local"), json!(null)),
                },
            },
            "format": "mixed-precision", "global_compression_ratio": null,
            "ignore": [
                "model.vision_embedder.patch_dense", "model.embed_vision.embedding_projection",
                "model.embed_audio.embedding_projection", "lm_head",
            ],
            "kv_cache_scheme": with(json!({
                "num_bits": 8, "type": "float", "symmetric": true, "dynamic": false,
                "strategy": "tensor", "observer": "static_minmax",
            })),
            "quant_method": "compressed-tensors", "quantization_status": "compressed",
            "sparsity_config": {}, "transform_config": {}, "version": "0.17.2.a20260707",
        })
    }

    /// The published size's text configuration, as Transformers before 5.15 writes it.
    fn document() -> Value {
        json!({
            "hidden_size": Hidden::SIZE, "intermediate_size": Intermediate::SIZE,
            "num_hidden_layers": ORDER.len(), "num_attention_heads": Queries::SIZE,
            "num_key_value_heads": sliding::Heads::SIZE,
            "num_global_key_value_heads": full::Heads::SIZE,
            "head_dim": sliding::Head::SIZE, "global_head_dim": full::Head::SIZE,
            "vocab_size": Vocabulary::SIZE,
            "layer_types": ORDER
                .iter()
                .map(|branch| LAYER_TYPES[*branch as usize])
                .collect::<Vec<_>>(),
            "hidden_activation": "gelu_pytorch_tanh", "attention_bias": false,
            "enable_moe_block": false, "use_double_wide_mlp": false, "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 0, "attention_k_eq_v": true,
            "tie_word_embeddings": true, "rms_norm_eps": 1e-6, "final_logit_softcapping": 30.0,
            "max_position_embeddings": 262144, "sliding_window": 1024,
            "rope_parameters": {
                "sliding_attention": {"rope_type": "default", "rope_theta": 1e4},
                "full_attention": {
                    "rope_type": "proportional", "rope_theta": 1e6, "partial_rotary_factor": 0.25,
                },
            },
        })
    }

    /// The kind and reason of the size's refusal of `document`, none if it parses.
    fn refusal(document: &Value) -> Option<(io::ErrorKind, String)> {
        let error = Size::<"12B">::parse(document).err()?;
        Some((error.kind(), error.to_string()))
    }

    /// The published configuration parses; another head count, head width, layer count or layer
    /// order is refused by its field, with what it found and what the size compiles.
    #[test]
    fn refuses_another_geometry_by_field() {
        assert_eq!(refusal(&document()), None);
        let order = ORDER.iter().map(|branch| LAYER_TYPES[*branch as usize]);
        let mut swapped: Vec<_> = order.collect();
        swapped.swap(4, 5);
        let compiled = |field: &str, found: usize, expected: usize| {
            format!("{field}: found {found}, compiled {expected}")
        };
        let layers = ORDER.len();
        for (field, value, reason) in [
            (
                "num_key_value_heads",
                json!(4),
                compiled("num_key_value_heads", 4, sliding::Heads::SIZE),
            ),
            (
                "num_global_key_value_heads",
                json!(2),
                compiled("num_global_key_value_heads", 2, full::Heads::SIZE),
            ),
            (
                "head_dim",
                json!(128),
                compiled("head_dim", 128, sliding::Head::SIZE),
            ),
            (
                "global_head_dim",
                json!(256),
                compiled("global_head_dim", 256, full::Head::SIZE),
            ),
            (
                "num_hidden_layers",
                json!(layers - 6),
                compiled("num_hidden_layers", layers - 6, layers),
            ),
        ] {
            let mut changed = document();
            changed[field] = value;
            assert_eq!(
                refusal(&changed),
                Some((io::ErrorKind::Unsupported, reason)),
                "{field}"
            );
        }
        let mut reordered = document();
        reordered["layer_types"] = json!(swapped);
        let (kind, reason) = refusal(&reordered).unwrap();
        assert_eq!(kind, io::ErrorKind::Unsupported);
        assert!(
            reason.starts_with("layer_types: found [\"sliding_attention\""),
            "{reason}"
        );
    }

    /// Each group binds its own spectrum: a group whose layers the checkpoint rotates nothing is
    /// refused, naming its layer type and the coordinates it rotates.
    #[test]
    fn refuses_groups_that_rotate_nothing() {
        for (layer, rotated) in [
            ("sliding_attention", sliding::Head::SIZE),
            ("full_attention", full::Rotated::SIZE),
        ] {
            let mut document = document();
            document["rope_parameters"][layer] = Value::Null;
            let reason = format!(
                "rope_parameters.{layer}: found layers that rotate nothing, compiled {rotated} \
                 rotated"
            );
            assert_eq!(
                refusal(&document),
                Some((io::ErrorKind::Unsupported, reason))
            );
        }
    }

    /// `document` with `fields` set apart at every layer of type `kind`, as Transformers'
    /// `per_layer_config` writes them.
    fn apart(mut document: Value, kind: &str, fields: Value) -> Value {
        let types = document["layer_types"].as_array().unwrap().clone();
        let layers = types.iter().enumerate().filter(|(_, layer)| *layer == kind);
        let overrides = layers.map(|(layer, _)| (format!("{layer:02}"), fields.clone()));
        document["per_layer_config"] = Value::Object(overrides.collect());
        document
    }

    /// The published configuration as Transformers 5.15 and later write it: its full layers' own
    /// head count and width in place of the `global_*` fields.
    fn transformers() -> Value {
        let mut document = document();
        for field in ["num_global_key_value_heads", "global_head_dim"] {
            document.as_object_mut().unwrap().remove(field);
        }
        let own = json!({"num_key_value_heads": full::Heads::SIZE, "head_dim": full::Head::SIZE});
        apart(document, "full_attention", own)
    }

    /// The full layers' head geometry reads from the checkpoint's `global_*` fields, from the
    /// full layers' per-layer values that Transformers writes instead, and from both where they
    /// agree.
    #[test]
    fn reads_full_layers_in_either_spelling() {
        let both = {
            let mut document = transformers();
            document["num_global_key_value_heads"] = json!(full::Heads::SIZE);
            document["global_head_dim"] = json!(full::Head::SIZE);
            document
        };
        for document in [document(), transformers(), both] {
            assert_eq!(refusal(&document), None);
        }
    }

    /// Per-layer geometry that the compiled groups do not hold is refused, each case by its kind
    /// and reason.
    #[test]
    fn refuses_per_layer_geometry_outside_the_groups() {
        let narrow = json!({"num_key_value_heads": full::Heads::SIZE, "head_dim": 256});
        let unheld = json!({
            "num_key_value_heads": full::Heads::SIZE, "head_dim": full::Head::SIZE,
            "intermediate_size": 1,
        });
        for (document, kind, reason) in [
            (
                apart(transformers(), "full_attention", narrow.clone()),
                io::ErrorKind::Unsupported,
                "global_head_dim: found 256, compiled 512",
            ),
            (
                apart(document(), "full_attention", narrow),
                io::ErrorKind::InvalidData,
                "global_head_dim: found 512, but the full_attention layers' head_dim is 256",
            ),
            (
                apart(document(), "sliding_attention", json!({"head_dim": 64})),
                io::ErrorKind::Unsupported,
                "head_dim: found 64, compiled 256",
            ),
            (
                apart(transformers(), "full_attention", unheld),
                io::ErrorKind::Unsupported,
                "per_layer_config: the full_attention layers set unknown field \
                 `intermediate_size`, expected `num_key_value_heads` or `head_dim`",
            ),
        ] {
            assert_eq!(refusal(&document), Some((kind, reason.into())));
        }
    }

    /// A config that omits the head's tying declares Gemma 4's tied head; an untied head is
    /// unsupported.
    #[test]
    fn reads_an_omitted_tying_as_tied() {
        let mut document = document();
        document
            .as_object_mut()
            .unwrap()
            .remove("tie_word_embeddings");
        assert_eq!(refusal(&document), None);
        document["tie_word_embeddings"] = json!(false);
        assert_eq!(
            refusal(&document),
            Some((
                io::ErrorKind::Unsupported,
                "tie_word_embeddings: found false, compiled true".into()
            ))
        );
    }

    /// A vocabulary of at most the compiled rows is served; a larger one is unsupported.
    #[test]
    fn serves_a_vocabulary_up_to_its_rows() {
        let rows = Vocabulary::SIZE;
        let mut document = document();
        document["vocab_size"] = json!(rows - 1);
        assert_eq!(refusal(&document), None);
        document["vocab_size"] = json!(rows + 1);
        let reason = format!("vocab_size: {}, the size holds 1 to {rows}", rows + 1);
        assert_eq!(
            refusal(&document),
            Some((io::ErrorKind::Unsupported, reason))
        );
    }

    /// A context past the compiled cache windows is served up to them; a sliding window past them
    /// is refused, keeping one window for an unaligned prefix.
    #[test]
    fn serves_a_context_up_to_the_compiled_windows() {
        let positions = full::Windows::SIZE * Cache::SIZE;
        let window = (sliding::Windows::SIZE - 1) * Cache::SIZE;
        let mut document = document();
        document["max_position_embeddings"] = json!(positions + 1);
        assert_eq!(refusal(&document), None);
        document["sliding_window"] = json!(window);
        assert_eq!(refusal(&document), None);
        document["sliding_window"] = json!(window + 1);
        let reason = format!(
            "sliding_window: found {}, compiled maximum {window}",
            window + 1
        );
        assert_eq!(
            refusal(&document),
            Some((io::ErrorKind::Unsupported, reason))
        );
    }

    /// The causal model admits the published configuration and its mixed-precision declaration;
    /// a declaration of another scheme, of one scheme fewer, or none is refused by it.
    #[test]
    fn admits_the_published_quantization() {
        let directory = tempfile::tempdir().unwrap();
        let scope = |quantization: Option<Value>| {
            let mut config = json!({"text_config": document(), "tie_word_embeddings": true});
            if let Some(quantization) = quantization {
                config["quantization_config"] = quantization;
            }
            let source = Checkpoint::new(directory.path(), Some(config)).unwrap();
            <ForCausalLM>::scope(&source)
        };
        assert_eq!(scope(Some(quantization())).unwrap(), "");
        let mut tensors = quantization();
        tensors["config_groups"]["group_0"]["weights"]["strategy"] = json!("tensor");
        let mut channels = quantization();
        (channels["config_groups"].as_object_mut().unwrap()).remove("group_1");
        for refused in [scope(Some(tensors)), scope(Some(channels)), scope(None)] {
            match refused {
                Err(LoadError::Unsupported { name, .. }) => {
                    assert_eq!(name, "config.json:quantization_config");
                }
                other => panic!("{other:?} is not `Unsupported`"),
            }
        }
    }
}
