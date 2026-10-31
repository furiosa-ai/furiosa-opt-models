//! Gemma 4's dense text function: a scaled token embedding, sliding and full attention layers
//! interleaved in checkpoint order, and a softcapped tied vocabulary head.
//!
//! - A size is a [`decoder::Geometry`] of the model axes with the sliding group first, and the
//!   projection forms are [`decoder::Projections`]; every default is the served 12B checkpoint,
//!   whose axes this module declares.

use crate::Sequential;
use crate::models::config::decoder::Language;
use crate::models::config::{
    self, Constant, Disabled, Enabled, Literal, Positive, Quantization, Scheme, Size, compiled,
    scaling::Spectra,
};
use crate::models::decoder::{
    self, Attention, Declaration, Forms, Requirement, Rotation, Shape, form,
};
use crate::models::{Family, vocabulary};
use crate::modules::attention;
use crate::modules::{language, multimodal, transformer};
use crate::ops::attention::gqa::Cache;
use crate::ops::embedding;
use crate::ops::index::scatter::Scatter;
use crate::ops::index::table::History;
use crate::ops::linear::Affine;
use crate::ops::normalization::{RmsNorm, UnitRmsNorm};
use crate::ops::pointwise::activation::NewGelu;
use crate::ops::pointwise::activation::Tanh;
use crate::ops::pointwise::add;
use crate::ops::pointwise::multiply;
use crate::tensor::cache::Blocks;
use crate::tensor::scalar::Value;
use crate::{Branch, Then};
use furiosa_opt_std::prelude::*;
use serde::Deserialize;
use std::{io, marker::PhantomData, num::NonZeroUsize};

/// Scaled embedding, `S` sliding and `F` full attention layers in order `O`, final norm and
/// softcapped vocabulary logits, at geometry `G` with projection forms `P`.
pub type Model<
    G = Sizes,
    P = Quantized,
    const S: usize = SLIDING,
    const F: usize = FULL,
    const O: &'static [Branch] = ORDER,
> = Gemma4ForCausalLM<
    Embedding<<G as Shape>::Vocabulary, <G as Shape>::Hidden>,
    Qkv<<G as Shape>::Hidden, <G as Shape>::Queries, <G as Shape>::First, P>,
    decoder::Position<<G as Shape>::First>,
    decoder::Attend<<G as Shape>::First, UNSCALED>,
    decoder::Output<P, <G as Shape>::Hidden, <G as Shape>::First>,
    Qkv<<G as Shape>::Hidden, <G as Shape>::Queries, <G as Shape>::Second, P>,
    decoder::Position<<G as Shape>::Second>,
    decoder::Attend<<G as Shape>::Second, UNSCALED>,
    decoder::Output<P, <G as Shape>::Hidden, <G as Shape>::Second>,
    Scatter,
    (),
    decoder::FeedForward<P, <G as Shape>::Hidden, <G as Shape>::Intermediate, NewGelu>,
    RmsNorm<<G as Shape>::Hidden>,
    add::Add,
    Scale,
    Head<<G as Shape>::Vocabulary, <G as Shape>::Hidden>,
    S,
    F,
    O,
>;

/// Scaled embedding and interleaved decoder layers followed by final RMS normalization.
pub type Body<
    G = Sizes,
    P = Quantized,
    const S: usize = SLIDING,
    const F: usize = FULL,
    const O: &'static [Branch] = ORDER,
> = Model<G, P, S, F, O>::Body;

/// Token lookup scaled by the factor its configuration fixes.
pub type Embedding<Vocabulary, Hidden> = Then<embedding::Embedding<Vocabulary, Hidden>, Scale>;

type Scale = multiply::Scale<HbmTensor<f32, Chip, m![1 # 8]>>;

type Chip = m![1];

/// Normalized queries and keys make scores need no multiplier.
const UNSCALED: usize = 1.0_f32.to_bits() as usize;

/// Group `A`'s query, key and value projections into heads: queries and keys normalized by learned
/// gains, values by unit gain.
pub type Qkv<Hidden, Queries, A, P> = attention::Projection<
    decoder::NormalizedHeads<
        <P as Forms>::Query,
        Hidden,
        <A as Attention>::QueryWidth,
        Queries,
        <A as Attention>::Head,
        RmsNorm<<A as Attention>::Head>,
    >,
    decoder::NormalizedHeads<
        <P as Forms>::Key,
        Hidden,
        <A as Attention>::KvWidth,
        <A as Attention>::Heads,
        <A as Attention>::Head,
        RmsNorm<<A as Attention>::Head>,
    >,
    decoder::NormalizedHeads<
        <P as Forms>::Value,
        Hidden,
        <A as Attention>::KvWidth,
        <A as Attention>::Heads,
        <A as Attention>::Head,
        UnitRmsNorm<<A as Attention>::Head>,
    >,
>;

/// Tied vocabulary projection whose logits `z` become `c * tanh(z / c)`.
pub type Head<Vocabulary, Hidden> =
    Then<Then<Then<Affine<HbmTensor<bf16, Chip, m![Vocabulary, Hidden]>, ()>, Scale>, Tanh>, Scale>;

/// Token id, semantic attention context and writable sliding/full layer-cache arrays.
pub type Context<
    'a,
    const ALIGNED: bool = true,
    G = Sizes,
    const S: usize = SLIDING,
    const F: usize = FULL,
> = (
    Value,
    attention::Context<&'a Value, &'a Value, History<'a, ALIGNED>>,
    (
        &'a mut [Blocks<
            <<G as Shape>::First as Attention>::Heads,
            <<G as Shape>::First as Attention>::Head,
            bf16,
            Chip,
        >; S],
        &'a mut [Blocks<
            <<G as Shape>::Second as Attention>::Heads,
            <<G as Shape>::Second as Attention>::Head,
            bf16,
            Chip,
        >; F],
    ),
);

/// Sliding and full attention layer counts of the served checkpoint.
pub const SLIDING: usize = 40;
pub const FULL: usize = 8;

/// Every sixth layer of the served checkpoint attends over its full history.
pub const ORDER: &[Branch] = &Branch::periodic::<{ SLIDING + FULL }>(6);

/// The layer types a checkpoint names its sliding and full attention layers by.
const LAYER_TYPES: [&str; 2] = ["sliding_attention", "full_attention"];

axes![
    Hidden = 3840,
    Intermediate = 15360,
    Vocabulary = 262144,
    Queries = 16
];

/// The served checkpoint's geometry.
pub type Sizes =
    decoder::Geometry<Hidden, Intermediate, Vocabulary, Queries, sliding::Group, full::Group>;

/// FP8 per-channel attention projections and an NVFP4 feed-forward, as the served checkpoint
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

/// Supported context lengths, rotary frequencies and head constants for the two compiled attention
/// groups of geometry `G`, projection forms `P` and layer order `O`.
pub struct Config<
    G: Shape = Sizes,
    P = Quantized,
    const S: usize = SLIDING,
    const F: usize = FULL,
    const O: &'static [Branch] = ORDER,
> {
    /// Maximum token positions for full attention.
    pub positions: usize,
    /// Sliding history width in tokens.
    pub window: usize,
    /// The frequencies each group's rotation binds, in sliding, full order.
    pub rotary: (Bound<Sliding<G>>, Bound<Full<G>>),
    /// Multiplier of every embedded token row: the BF16-rounded square root of the hidden width.
    pub embedding: f32,
    /// The bound `c` of the head's `c * tanh(z / c)` logits.
    pub softcap: f32,
    /// The NVFP4 projections the checkpoint declares, which NVFP4 forms require.
    pub quantization: Option<Quantization>,
    compiled: PhantomData<fn() -> (G, P)>,
}

impl<G: Shape, P, const S: usize, const F: usize, const O: &'static [Branch]> Clone
    for Config<G, P, S, F, O>
{
    fn clone(&self) -> Self {
        Self {
            rotary: self.rotary.clone(),
            quantization: self.quantization.clone(),
            ..*self
        }
    }
}

impl<G: Shape, P: Forms, const S: usize, const F: usize, const O: &'static [Branch]> Family
    for Config<G, P, S, F, O>
{
    type Hf = Hf<G>;
}

impl<G: Shape, P: Forms, const S: usize, const F: usize, const O: &'static [Branch]> decoder::Paired
    for Config<G, P, S, F, O>
{
    type Model<CacheScalar: Scalar> =
        decoder::Interleaved<G, S, F, CacheScalar, Model<G, P, S, F, O>>;

    type First = <Sliding<G> as Attention>::Rotated;
    type Second = <Full<G> as Attention>::Rotated;

    fn positions(&self) -> usize {
        self.positions
    }

    fn groups(&self) -> ((Bound<Sliding<G>>, usize), (Bound<Full<G>>, usize)) {
        let (sliding, full) = self.rotary.clone();
        ((sliding, self.window), (full, self.positions))
    }

    fn constants(&self) -> decoder::Constants {
        decoder::Constants {
            embedding: Some(self.embedding),
            softcap: Some(self.softcap),
            ..Default::default()
        }
    }
}

/// The sliding attention group of geometry `G`.
type Sliding<G> = <G as Shape>::First;

/// The full attention group of geometry `G`.
type Full<G> = <G as Shape>::Second;

/// The frequencies the rotation of group `A` binds.
type Bound<A> = <<A as Attention>::Rotated as Rotation>::Frequencies;

/// A Gemma 4 checkpoint's language configuration.
#[derive(Deserialize)]
#[serde(bound = "G: Shape")]
pub struct Hf<G: Shape> {
    #[serde(flatten, deserialize_with = "config::text")]
    text: Language<Text<G>>,
}

/// The fields only Gemma 4's language configuration has, and the sizes geometry `G` compiles.
#[derive(Deserialize)]
#[serde(bound = "G: Shape")]
struct Text<G: Shape> {
    hidden_size: Size<G::Hidden>,
    intermediate_size: Size<G::Intermediate>,
    num_attention_heads: Size<G::Queries>,
    vocab_size: usize,
    hidden_activation: Literal<"gelu_pytorch_tanh">,
    attention_bias: Disabled,
    enable_moe_block: Disabled,
    use_double_wide_mlp: Disabled,
    num_kv_shared_layers: Constant<0>,
    hidden_size_per_layer_input: Constant<0>,
    attention_k_eq_v: Enabled,
    tie_word_embeddings: Enabled,
    final_logit_softcapping: Positive,
    #[serde(flatten, deserialize_with = "config::declared")]
    max_position_embeddings: NonZeroUsize,
    sliding_window: NonZeroUsize,
    #[serde(flatten, deserialize_with = "config::flattened")]
    spectra: Spectra,
    quantization_config: Option<serde_json::Value>,
}

impl<G: Shape, P: Forms, const S: usize, const F: usize, const O: &'static [Branch]> TryFrom<Hf<G>>
    for Config<G, P, S, F, O>
{
    type Error = io::Error;

    /// Refuses checkpoint geometry, layer order or rotary options outside the compiled composition.
    fn try_from(Hf { text }: Hf<G>) -> io::Result<Self> {
        const {
            assert!(G::CONSISTENT);
            assert!(O.len() == S + F);
        }
        let geometry = &text.geometry;
        let Text {
            hidden_activation: Literal,
            attention_bias: Disabled,
            enable_moe_block: Disabled,
            use_double_wide_mlp: Disabled,
            num_kv_shared_layers: Constant,
            hidden_size_per_layer_input: Constant,
            attention_k_eq_v: Enabled,
            tie_word_embeddings: Enabled,
            hidden_size: Size { .. },
            intermediate_size: Size { .. },
            num_attention_heads: Size { .. },
            ..
        } = text.extra;
        vocabulary::<G::Vocabulary>(text.extra.vocab_size)?;
        let [sliding, full] = LAYER_TYPES;
        let (own, global) = (text.group(sliding), text.group(full));
        for (field, found, expected) in [
            (
                "num_hidden_layers",
                geometry.num_hidden_layers.get(),
                O.len(),
            ),
            (
                "num_key_value_heads",
                own.num_key_value_heads,
                <Sliding<G> as Attention>::Heads::SIZE,
            ),
            (
                "head_dim",
                own.head_dim,
                <Sliding<G> as Attention>::Head::SIZE,
            ),
            (
                "num_global_key_value_heads",
                global.num_key_value_heads,
                <Full<G> as Attention>::Heads::SIZE,
            ),
            (
                "global_head_dim",
                global.head_dim,
                <Full<G> as Attention>::Head::SIZE,
            ),
        ] {
            compiled(field, found, expected)?;
        }
        compiled("rms_norm_eps", geometry.rms_norm_eps, Some(1e-6))?;
        let order: Vec<String> = O
            .iter()
            .map(|branch| LAYER_TYPES[*branch as usize].to_owned())
            .collect();
        compiled("layer_types", text.layer_types(), order.as_slice())?;
        let positions = text.extra.max_position_embeddings.get();
        let window = text.extra.sliding_window.get();
        for (field, found, maximum) in [
            (
                "max_position_embeddings",
                positions,
                <Full<G> as Attention>::Windows::SIZE * Cache::SIZE,
            ),
            (
                "sliding_window",
                window,
                // An unaligned start reserves one cache window for the partial prefix.
                (<Sliding<G> as Attention>::Windows::SIZE - 1) * Cache::SIZE,
            ),
        ] {
            if found > maximum {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!("{field}: found {found}, compiled maximum {maximum}"),
                ));
            }
        }
        let within = |layer: &'static str| {
            move |error: io::Error| {
                io::Error::new(error.kind(), format!("rope_parameters.{layer}: {error}"))
            }
        };
        let rotary = (
            <Sliding<G> as Attention>::Rotated::spectrum::<<Sliding<G> as Attention>::Head>(
                text.extra.spectra.of(sliding)?,
            )
            .map_err(within(sliding))?,
            <Full<G> as Attention>::Rotated::spectrum::<<Full<G> as Attention>::Head>(
                text.extra.spectra.of(full)?,
            )
            .map_err(within(full))?,
        );
        let unsupported = |message: &str| io::Error::new(io::ErrorKind::Unsupported, message);
        let quantization = match (P::REQUIREMENT, text.extra.quantization_config) {
            (Requirement::Unquantized | Requirement::Undeclared, _) => None,
            (Requirement::Declared(Declaration::Nvfp4(_)), Some(declared)) => {
                let declared: Quantization = config::parse(&declared)?;
                if declared.scheme != Scheme::Nvfp4 {
                    return Err(unsupported(
                        "quantization_config: found FP8, compiled NVFP4 projections",
                    ));
                }
                Some(declared)
            }
            (Requirement::Declared(Declaration::Nvfp4(_)), None) => {
                return Err(unsupported(
                    "quantization_config: missing, compiled NVFP4 projections",
                ));
            }
            (Requirement::Declared(Declaration::Fp8(_)), _) => {
                return Err(unsupported(
                    "quantization_config: this family reads no FP8 declaration",
                ));
            }
        };
        Ok(Self {
            positions,
            window,
            rotary,
            embedding: bf16::from_f32((G::Hidden::SIZE as f32).sqrt()).to_f32(),
            softcap: text.extra.final_logit_softcapping.get(),
            quantization,
            compiled: PhantomData,
        })
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
mod tests {
    use super::*;
    use crate::models::Family;
    use crate::ops::attention::geometry::{QueryGroups, QueryWidth};
    use crate::ops::rotary::PairFrequencies;

    axes![
        Heads2 = 2,
        Heads4 = 4,
        Heads8 = 8,
        Heads16 = 16,
        Head = 128,
        Width = 1024,
        Windows = 16
    ];

    /// The key or value projection width of `H` heads.
    #[derive(Clone, Debug)]
    struct KvWidth<H>(PhantomData<H>);

    impl<H: AxisName> AxisName for KvWidth<H> {
        const NAME: Ident = Ident::new("KvWidth");
        const SIZE: usize = H::SIZE * Head::SIZE;
    }

    type Group<Heads, Groups> =
        decoder::Group<Heads, Head, Groups, QueryWidth<2048>, KvWidth<Heads>, Head, Windows>;

    type Sizes<First, Second> = decoder::Geometry<Width, Width, Width, Sixteen, First, Second>;

    axes![Sixteen = 16];

    const ORDER: &[Branch] = &Branch::periodic::<6>(3);

    /// A checkpoint document for geometry `G` with `O`'s layer types.
    fn document<G: Shape, const O: &'static [Branch]>() -> serde_json::Value {
        type First<G> = <G as Shape>::First;
        type Second<G> = <G as Shape>::Second;
        serde_json::json!({
            "hidden_size": G::Hidden::SIZE,
            "intermediate_size": G::Intermediate::SIZE,
            "num_hidden_layers": O.len(),
            "num_attention_heads": G::Queries::SIZE,
            "num_key_value_heads": <First<G> as Attention>::Heads::SIZE,
            "num_global_key_value_heads": <Second<G> as Attention>::Heads::SIZE,
            "head_dim": <First<G> as Attention>::Head::SIZE,
            "global_head_dim": <Second<G> as Attention>::Head::SIZE,
            "vocab_size": G::Vocabulary::SIZE,
            "layer_types": O
                .iter()
                .map(|branch| ["sliding_attention", "full_attention"][*branch as usize])
                .collect::<Vec<_>>(),
            "hidden_activation": "gelu_pytorch_tanh", "attention_bias": false,
            "enable_moe_block": false, "use_double_wide_mlp": false, "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 0, "attention_k_eq_v": true,
            "tie_word_embeddings": true, "rms_norm_eps": 1e-6, "final_logit_softcapping": 30.0,
            "max_position_embeddings": 4096, "sliding_window": 512,
            "rope_parameters": {
                "sliding_attention": {"rope_type": "default", "rope_theta": 1e4},
                "full_attention": {
                    "rope_type": "proportional", "rope_theta": 1e6,
                    "partial_rotary_factor": <Second<G> as Attention>::Rotated::COORDINATES as f32
                        / <Second<G> as Attention>::Head::SIZE as f32,
                },
            },
        })
    }

    /// A geometry parses the checkpoint whose head counts and widths it compiles, and refuses one
    /// whose key/value heads differ in either group, naming the field, the count found and the
    /// compiled one.
    fn parses_own_heads<First: Attention, Second: Attention>() {
        type Compiled<First, Second> = Config<Sizes<First, Second>, Quantized, 4, 2, ORDER>;
        let accepted = document::<Sizes<First, Second>, ORDER>();
        let config = Compiled::<First, Second>::parse(&accepted).unwrap();
        assert_eq!((config.window, config.positions), (512, 4096));
        for (field, heads) in [
            ("num_key_value_heads", First::Heads::SIZE),
            ("num_global_key_value_heads", Second::Heads::SIZE),
        ] {
            let mut refused = accepted.clone();
            refused[field] = serde_json::json!(2 * heads);
            let error = Compiled::<First, Second>::parse(&refused).err().unwrap();
            assert_eq!(
                (error.kind(), error.to_string()),
                (
                    io::ErrorKind::Unsupported,
                    format!("{field}: found {}, compiled {heads}", 2 * heads)
                )
            );
        }
    }

    /// The frequencies each group of a checkpoint with `rope` binds, sliding then full, whose
    /// full group has geometry `Full`.
    fn rotary<Full: Attention>(
        rope: serde_json::Value,
    ) -> io::Result<(
        PairFrequencies<Head>,
        <Full::Rotated as Rotation>::Frequencies,
    )> {
        type Compiled<Full> =
            Config<Sizes<Group<Heads8, QueryGroups<2>>, Full>, Quantized, 4, 2, ORDER>;
        let mut document = document::<Sizes<Group<Heads8, QueryGroups<2>>, Full>, ORDER>();
        document["rope_parameters"] = rope;
        Compiled::<Full>::parse(&document).map(|config| config.rotary)
    }

    /// Each group binds the frequencies its layer type's spectrum gives; a group whose checkpoint
    /// rotates nothing is refused.
    #[test]
    fn binds_each_group_spectrum() {
        use crate::models::config::scaling::Spectrum;
        use serde_json::json;
        type Rotating = Group<Heads8, QueryGroups<2>>;
        let default = |theta: f32| json!({"rope_type": "default", "rope_theta": theta});
        let llama3 = json!({
            "rope_type": "llama3", "rope_theta": 1e6, "factor": 16.0, "low_freq_factor": 1.0,
            "high_freq_factor": 4.0, "original_max_position_embeddings": 8192,
        });
        let rescaled = config::parse::<Spectrum>(&json!({"rope_parameters": llama3}))
            .and_then(|spectrum| spectrum.bind::<Head, Head>())
            .unwrap();
        let unscaled = |theta| PairFrequencies::<Head>::unscaled::<Head>(theta).unwrap();
        assert_eq!(
            rotary::<Rotating>(json!({
                "sliding_attention": default(1e4), "full_attention": default(1e6),
            }))
            .unwrap(),
            (unscaled(1e4), unscaled(1e6))
        );
        assert_eq!(
            rotary::<Rotating>(json!({
                "sliding_attention": llama3, "full_attention": default(1e6),
            }))
            .unwrap(),
            (rescaled, unscaled(1e6))
        );
        let error =
            rotary::<Rotating>(json!({"sliding_attention": llama3, "full_attention": null}))
                .map(drop)
                .unwrap_err();
        let reason = format!(
            "rope_parameters.full_attention: found layers that rotate nothing, compiled {} rotated",
            <Rotating as Attention>::Rotated::COORDINATES
        );
        assert_eq!(
            (error.kind(), error.to_string()),
            (io::ErrorKind::Unsupported, reason)
        );
        let error =
            rotary::<Rotating>(json!({"sliding_attention": null, "full_attention": default(1e6)}))
                .map(drop)
                .unwrap_err();
        let reason = format!(
            "rope_parameters.sliding_attention: found layers that rotate nothing, compiled {} \
             rotated",
            <Rotating as Attention>::Rotated::COORDINATES
        );
        assert_eq!(
            (error.kind(), error.to_string()),
            (io::ErrorKind::Unsupported, reason)
        );
    }

    /// NVFP4 projections read the ModelOpt declaration a nested checkpoint keeps at its root,
    /// with the E4M3 cache it declares, and refuse a missing or another declaration; forms that
    /// read no declaration leave it unread.
    #[test]
    fn reads_the_declared_nvfp4_projections() {
        use serde_json::json;
        type Geometry = Sizes<Group<Heads8, QueryGroups<2>>, Group<Heads4, QueryGroups<4>>>;
        type Nvfp4 = decoder::Projections<
            form::Affine,
            form::Affine,
            form::Affine,
            form::Affine,
            form::Nvfp4BlockScaled,
        >;
        type Declared = Config<Geometry, Nvfp4, 4, 2, ORDER>;
        type Unread = Config<Geometry, Quantized, 4, 2, ORDER>;
        let values = json!({"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16});
        let modelopt = json!({
            "quant_method": "modelopt", "quant_algo": "NVFP4",
            "config_groups": {
                "group_0": {"input_activations": values, "weights": values, "targets": ["Linear"]},
            },
            "ignore": ["lm_head", "model.language_model.layers.0.self_attn*"],
            "kv_cache_scheme": {"dynamic": false, "num_bits": 8, "type": "float"},
        });
        let compressed = json!({
            "quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized",
        });
        let nested = |quantization: Option<&serde_json::Value>| {
            let mut root = json!({"text_config": document::<Geometry, ORDER>()});
            if let Some(quantization) = quantization {
                root["quantization_config"] = quantization.clone();
            }
            config::language(&root).unwrap().into_owned()
        };
        let declared = Declared::parse(&nested(Some(&modelopt)))
            .unwrap()
            .quantization
            .unwrap();
        assert!(declared.excluded.excludes("lm_head"));
        assert!(
            declared
                .excluded
                .excludes("model.language_model.layers.0.self_attn.q_proj")
        );
        assert!(declared.kv_cache);
        let unread = Unread::parse(&nested(Some(&compressed))).unwrap();
        assert!(unread.quantization.is_none());
        for (refusal, reason) in [
            (
                Declared::parse(&nested(None)),
                "quantization_config: missing, compiled NVFP4 projections",
            ),
            (
                Declared::parse(&nested(Some(&compressed))),
                r#"found "nvfp4-pack-quantized", compiled "float-quantized""#,
            ),
            (
                Declared::parse(&nested(Some(
                    &json!({"quant_method": "fp8", "activation_scheme": "static"}),
                ))),
                "quantization_config: found FP8, compiled NVFP4 projections",
            ),
        ] {
            let error = refusal.err().unwrap();
            assert_eq!(
                (error.kind(), error.to_string()),
                (io::ErrorKind::Unsupported, reason.into())
            );
        }
    }

    /// Each pairing of group geometries parses its own checkpoint and refuses the others' heads.
    #[test]
    fn parses_each_group_geometry() {
        for row in [
            parses_own_heads::<Group<Heads2, QueryGroups<8>>, Group<Heads16, QueryGroups<1>>>,
            parses_own_heads::<Group<Heads4, QueryGroups<4>>, Group<Heads8, QueryGroups<2>>>,
            parses_own_heads::<Group<Heads8, QueryGroups<2>>, Group<Heads4, QueryGroups<4>>>,
            parses_own_heads::<Group<Heads16, QueryGroups<1>>, Group<Heads2, QueryGroups<8>>>,
        ] {
            row();
        }
    }

    /// `document` with `fields` set apart at every layer of type `kind`, as Transformers'
    /// `per_layer_config` writes them.
    fn apart(
        mut document: serde_json::Value,
        kind: &str,
        fields: serde_json::Value,
    ) -> serde_json::Value {
        let types = document["layer_types"].as_array().unwrap().clone();
        let layers = types.iter().enumerate().filter(|(_, layer)| *layer == kind);
        let overrides = layers.map(|(layer, _)| (format!("{layer:02}"), fields.clone()));
        document["per_layer_config"] = serde_json::Value::Object(overrides.collect());
        document
    }

    /// The served checkpoint as Transformers 5.15 and later write it: its full layers' own head
    /// count and width in place of the `global_*` fields.
    fn transformers() -> serde_json::Value {
        let mut document = document::<super::Sizes, { super::ORDER }>();
        for field in ["num_global_key_value_heads", "global_head_dim"] {
            document.as_object_mut().unwrap().remove(field);
        }
        let own = serde_json::json!({
            "num_key_value_heads": full::Heads::SIZE, "head_dim": full::Head::SIZE,
        });
        apart(document, "full_attention", own)
    }

    /// The full layers' head geometry reads from the checkpoint's `global_*` fields, from the
    /// full layers' per-layer values that Transformers writes instead, and from both where they
    /// agree.
    #[test]
    fn reads_full_layers_in_either_spelling() {
        let both = {
            let mut document = transformers();
            document["num_global_key_value_heads"] = serde_json::json!(full::Heads::SIZE);
            document["global_head_dim"] = serde_json::json!(full::Head::SIZE);
            document
        };
        let checkpoint = document::<super::Sizes, { super::ORDER }>();
        for document in [checkpoint, transformers(), both] {
            <Config>::parse(&document).unwrap();
        }
    }

    /// Per-layer geometry that the compiled groups do not hold is refused, each case by its kind
    /// and reason.
    #[test]
    fn refuses_per_layer_geometry_outside_the_groups() {
        use serde_json::json;
        let checkpoint = document::<super::Sizes, { super::ORDER }>;
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
                apart(checkpoint(), "full_attention", narrow),
                io::ErrorKind::InvalidData,
                "global_head_dim: found 512, but the full_attention layers' head_dim is 256",
            ),
            (
                apart(checkpoint(), "sliding_attention", json!({"head_dim": 64})),
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
            let error = <Config>::parse(&document).err().unwrap();
            assert_eq!((error.kind(), error.to_string()), (kind, reason.into()));
        }
    }

    /// A vocabulary of at most the compiled rows is served; a larger one is unsupported.
    #[test]
    fn serves_a_vocabulary_up_to_its_rows() {
        let rows = <super::Sizes as Shape>::Vocabulary::SIZE;
        let mut document = document::<super::Sizes, { super::ORDER }>();
        document["vocab_size"] = serde_json::json!(rows - 1);
        <Config>::parse(&document).unwrap();
        document["vocab_size"] = serde_json::json!(rows + 1);
        let error = <Config>::parse(&document).err().unwrap();
        let reason = format!("vocab_size: {}, the size holds 1 to {rows}", rows + 1);
        assert_eq!(
            (error.kind(), error.to_string()),
            (io::ErrorKind::Unsupported, reason)
        );
    }

    /// A context or a sliding window past the compiled cache windows is refused by its field;
    /// the sliding window keeps one window for an unaligned prefix.
    #[test]
    fn refuses_contexts_past_the_compiled_windows() {
        let positions = <Full<super::Sizes> as Attention>::Windows::SIZE * Cache::SIZE;
        let window = (<Sliding<super::Sizes> as Attention>::Windows::SIZE - 1) * Cache::SIZE;
        for (field, maximum) in [
            ("max_position_embeddings", positions),
            ("sliding_window", window),
        ] {
            let mut document = document::<super::Sizes, { super::ORDER }>();
            document[field] = serde_json::json!(maximum);
            <Config>::parse(&document).unwrap();
            document[field] = serde_json::json!(maximum + 1);
            let error = <Config>::parse(&document).err().unwrap();
            let reason = format!("{field}: found {}, compiled maximum {maximum}", maximum + 1);
            assert_eq!(
                (error.kind(), error.to_string()),
                (io::ErrorKind::Unsupported, reason)
            );
        }
    }
}
