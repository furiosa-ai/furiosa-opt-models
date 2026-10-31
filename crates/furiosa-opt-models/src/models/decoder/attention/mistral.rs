//! Mistral NeMo's dense decoder of separate query, key and value projections, sized by type data;
//! every default is the served checkpoint, whose projections are independently scaled FP8.

use crate::models::config::{
    self, Absent, Constant, Disabled, Scheme, Size, compiled, flattened, scaling,
};
use crate::models::decoder::attention::{Qwen2ForCausalLM, Qwen2Layer};
use crate::models::decoder::{
    self, Attention, Declaration, Envelope, Forms, Query, Requirement, Shape, form,
};
use crate::models::{Family, vocabulary};
use crate::modules::{attention, language};
use crate::ops::attention::gqa::Read;
use crate::ops::embedding::Embedding;
use crate::ops::index::scatter::Scatter;
use crate::ops::index::table::Offset;
use crate::ops::pointwise::activation::Silu;
use crate::ops::pointwise::add::Add;
use crate::ops::pointwise::multiply::{QueryScale, Temperature};
use crate::ops::rotary::PairFrequencies;
use crate::ops::rotary::Rotation;
use crate::ops::swiglu;
use crate::ops::{attention::geometry, normalization::RmsNorm};
use crate::tensor::cache::Blocks;
use crate::tensor::{
    axis::Single,
    parallel::{Parallel, Parallelism as Placement},
};
use furiosa_opt_std::prelude::*;
use serde::Deserialize;
use std::{io, marker::PhantomData};

// --8<-- [start:composition]
/// Mistral's decoder: cache scalar `D`, placement `P`, geometry `G`, projection forms `F`, `N`
/// layers, norm `EPSILON` and score `SCALE` as FP32 bits, query pass `Q` and call rows `R`.
pub type Model<
    D = bf16,
    P = Parallel<1>,
    G = Sizes,
    F = Quantized,
    const N: usize = LAYERS,
    const EPSILON: usize = NORM_EPSILON,
    const SCALE: usize = SCORE_SCALE,
    Q = (),
    R = Single,
> = Qwen2ForCausalLM<
    Embedding<<G as Shape>::Vocabulary, <G as Shape>::Hidden, <P as Placement>::Chip>,
    Qkv<<G as Shape>::Hidden, <G as Shape>::Queries, <G as Shape>::First, F, D, P, R>,
    <Q as Query>::Position<decoder::Position<<G as Shape>::First, D, <P as Placement>::Chip>>,
    Scatter,
    (),
    decoder::Attend<<G as Shape>::First, SCALE, P>,
    decoder::Output<F, <G as Shape>::Hidden, <G as Shape>::First, P>,
    decoder::FeedForward<F, <G as Shape>::Hidden, <G as Shape>::Intermediate, Silu, P>,
    Norm<<G as Shape>::Hidden, P, EPSILON>,
    Add,
    <R as Envelope>::Head<G, P>,
    N,
>;

/// Token embedding and decoder layers followed by final RMS normalization.
pub type Body<
    D = bf16,
    P = Parallel<1>,
    G = Sizes,
    F = Quantized,
    const N: usize = LAYERS,
    const EPSILON: usize = NORM_EPSILON,
    const SCALE: usize = SCORE_SCALE,
    Q = (),
    R = Single,
> = Model<D, P, G, F, N, EPSILON, SCALE, Q, R>::Body;
// --8<-- [end:composition]

type Qkv<Hidden, Queries, A, F, D, P, R> = attention::Projection<
    decoder::Heads<
        <F as Forms>::Query,
        Hidden,
        <A as Attention>::QueryWidth,
        <R as Envelope>::Queries<
            <P as Placement>::Local<Queries>,
            <P as Placement>::Local<<A as Attention>::Heads>,
            <A as Attention>::Groups,
            <A as Attention>::Head,
        >,
        bf16,
        P,
    >,
    decoder::Heads<
        <F as Forms>::Key,
        Hidden,
        <A as Attention>::KvWidth,
        decoder::HeadView<<P as Placement>::Local<<A as Attention>::Heads>, <A as Attention>::Head>,
        bf16,
        P,
    >,
    decoder::Heads<
        <F as Forms>::Value,
        Hidden,
        <A as Attention>::KvWidth,
        decoder::HeadView<<P as Placement>::Local<<A as Attention>::Heads>, <A as Attention>::Head>,
        D,
        P,
    >,
>;

type Norm<Hidden, P = Parallel<1>, const EPSILON: usize = NORM_EPSILON> =
    RmsNorm<Hidden, EPSILON, <P as Placement>::Chip>;

/// Token embedding byte offset, then mutable paged cache and per-layer attention contexts.
pub type Context<'a, D = bf16, const N: usize = LAYERS, P = Parallel<1>> = (
    Offset<<P as Placement>::Chip>,
    (
        &'a mut [Blocks<<P as Placement>::Local<Heads>, Head, D, <P as Placement>::Chip>; N],
        [attention::Context<
            Rotation<Head, <P as Placement>::Chip>,
            HbmTensor<i32, <P as Placement>::Chip, m![1]>,
            Read<Windows, <P as Placement>::Chip>,
        >; N],
    ),
);

// --8<-- [start:entry]
/// Device-call bound for a decoder producing one BF16 row on the selected placement.
pub trait Decoder<D, P: Placement = Parallel<1>, const N: usize = LAYERS, V: AxisName = Vocabulary> =
    for<'a, 'b, 's> crate::Module<
            &'a mut Context<'b, D, N, P>,
            (),
            &'s mut (),
            Output = HbmTensor<bf16, P::Chip, m![V]>,
        >;

// --8<-- [end:entry]

// Literal entries collapse when the SDK accepts a const chip count.
pub mod stages {
    use super::*;

    pub mod first {
        use super::super::*;
        use super::Layer;

        pub type Model<D = bf16> =
            language::Backbone<Embedding<Vocabulary, Hidden>, crate::Layers<[Layer<D>; LAYERS]>>;

        pub type Context<'a, D = bf16> = super::super::Context<'a, D, LAYERS>;

        pub const LAYERS: usize = super::super::LAYERS / 2;
    }

    pub mod last {
        use super::super::*;
        use super::Layer;
        use super::first;

        pub type Model<D = bf16> = language::Readout<
            language::Normalized<crate::Layers<[Layer<D>; LAYERS]>, Norm<Hidden>>,
            decoder::Logits<Hidden, Vocabulary>,
        >;

        pub type Context<'a, 'b, D = bf16> = (
            HbmTensor<bf16, m![1], m![Hidden]>,
            &'a mut (
                &'b mut [Blocks<Heads, Head, D, m![1]>; LAYERS],
                [attention::Context<Rotation<Head>, HbmTensor<i32, m![1], m![1]>, Read<Windows>>;
                    LAYERS],
            ),
        );

        pub const LAYERS: usize = super::super::LAYERS - first::LAYERS;
    }

    /// A stage holds part of `Model`'s layers, so it names the single-chip layer itself.
    type Layer<D> = Qwen2Layer<
        Qkv<Hidden, Queries, Group, Quantized, D, Parallel<1>, Single>,
        decoder::Position<Group, D>,
        Scatter,
        (),
        decoder::Attend<Group, SCORE_SCALE>,
        decoder::Output<Quantized, Hidden, Group>,
        decoder::FeedForward<Quantized, Hidden, Intermediate, Silu>,
        Norm<Hidden>,
        Add,
    >;
}

/// Decoder layer count in checkpoint order.
pub const LAYERS: usize = 40;

/// FP32 bits of the served checkpoint's norm epsilon.
pub const NORM_EPSILON: usize = 1e-5_f32.to_bits() as usize;

/// FP32 bits of the score multiplier of 128-wide heads.
pub const SCORE_SCALE: usize = geometry::scale(Head::SIZE);

/// Hidden feature width.
pub type Hidden = geometry::Hidden<5120>;

/// SwiGLU intermediate width per gate and up projection.
pub type Intermediate = swiglu::Intermediate<14336>;

/// Combined query projection width before head reshaping.
pub type QueryWidth = geometry::QueryWidth<4096>;

// The combined key or value projection width before head reshaping, the scalar coordinates in
// one attention head, and the key/value head count before tensor-parallel splitting.
axes![KvWidth = 1024, Head = 128, KvHeads = 8];

/// Key/value head count before tensor-parallel splitting.
pub type Heads = KvHeads;

/// Query heads sharing each key/value head.
pub type Groups = geometry::QueryGroups<4>;

axes![Queries = 32, Vocabulary = 131072, Windows = 32];

/// The served checkpoint's attention group.
pub type Group = decoder::Group<Heads, Head, Groups, QueryWidth, KvWidth, Head, Windows>;

/// The served checkpoint's geometry.
pub type Sizes = decoder::Geometry<Hidden, Intermediate, Vocabulary, Queries, Group>;

/// Static per-tensor FP8 projections, as the served checkpoint stores them.
pub type Quantized = decoder::Projections<form::TensorScaled>;

/// Compiled projection options and rotary limits parsed from the checkpoint, of a composition that
/// compiles the placements `T` and whose calls run the rows of `R`.
pub struct Config<
    G: Shape = Sizes,
    F = Quantized,
    const N: usize = LAYERS,
    const EPSILON: usize = NORM_EPSILON,
    const SCALE: usize = SCORE_SCALE,
    Q: Query = (),
    T = (Parallel<1>, Parallel<2>, Parallel<4>),
    R = Single,
> {
    /// Maximum rotary token positions.
    positions: usize,
    /// Rotary frequency base.
    theta: f32,
    /// The frequency of each rotated pair.
    frequencies: PairFrequencies<Rotated<G>>,
    /// What each row's rotated queries read: nothing, or the temperature a scaled `Q` reads.
    query: Q::Parameter,
    /// The quantized projections the checkpoint declares, which declared forms require.
    quantization: Option<config::Quantization>,
    compiled: PhantomData<fn() -> (G, F, Q, T, R)>,
}

impl<G: Shape, F, const N: usize, const EPSILON: usize, const SCALE: usize, Q: Query, T, R> Clone
    for Config<G, F, N, EPSILON, SCALE, Q, T, R>
{
    fn clone(&self) -> Self {
        Self {
            positions: self.positions,
            theta: self.theta,
            frequencies: self.frequencies.clone(),
            query: self.query,
            quantization: self.quantization.clone(),
            compiled: PhantomData,
        }
    }
}

/// The declaration of `Model<CacheScalar, P, G, F, N, EPSILON, SCALE, Q, R>`: each of `P`'s chips
/// holds its part of the key/value heads, whose leading `Rotated<G>` coordinates rotate.
pub type Declared<
    CacheScalar,
    P = Parallel<1>,
    G = Sizes,
    F = Quantized,
    const N: usize = LAYERS,
    const EPSILON: usize = NORM_EPSILON,
    const SCALE: usize = SCORE_SCALE,
    Q = (),
    R = Single,
> = decoder::Model<
    <P as Placement>::Local<<<G as Shape>::First as Attention>::Heads>,
    <<G as Shape>::First as Attention>::Head,
    <<G as Shape>::First as Attention>::Windows,
    N,
    CacheScalar,
    <G as Shape>::Hidden,
    <G as Shape>::Vocabulary,
    Model<CacheScalar, P, G, F, N, EPSILON, SCALE, Q, R>,
    P,
    Q,
    Rotated<G>,
    R,
>;

// --8<-- [start:specialization]
impl<
    G: Shape,
    F: Forms,
    const N: usize,
    const EPSILON: usize,
    const SCALE: usize,
    Q: Query,
    T: decoder::Placements,
    R: Envelope,
> decoder::Config for Config<G, F, N, EPSILON, SCALE, Q, T, R>
where
    Rotated<G>: AxisName,
{
    type Model<CacheScalar: Scalar> =
        Declared<CacheScalar, Parallel<1>, G, F, N, EPSILON, SCALE, Q, R>;

    const TENSORS: &'static [usize] = T::TENSORS;

    type Rotated = Rotated<G>;

    fn rotary(&self) -> (usize, f32) {
        (self.positions, self.theta)
    }

    fn frequencies(&self) -> io::Result<PairFrequencies<Rotated<G>>> {
        Ok(self.frequencies.clone())
    }
}

impl<G: Shape, F: Forms, const N: usize, const EPSILON: usize, const SCALE: usize, T, R: Envelope>
    decoder::Queries<QueryScale> for Config<G, F, N, EPSILON, SCALE, QueryScale, T, R>
where
    Rotated<G>: AxisName,
{
    fn query(&self) -> Temperature {
        self.query
    }
}

impl<
    G: Shape,
    F: Forms,
    const N: usize,
    const EPSILON: usize,
    const SCALE: usize,
    Q: Query,
    T: decoder::Placements,
    R: Envelope,
> Family for Config<G, F, N, EPSILON, SCALE, Q, T, R>
where
    Rotated<G>: AxisName,
{
    type Hf = Hf<G, N>;
}
// --8<-- [end:specialization]

/// The coordinates of each head that geometry `G`'s layers rotate: an axis, since the layers of
/// a decoder of one group rotate.
type Rotated<G> = <<G as Shape>::First as Attention>::Rotated;

/// A NeMo checkpoint's configuration: the shared decoder schema and the values the composition
/// compiles.
#[derive(Deserialize)]
#[serde(bound = "G: Shape")]
pub struct Hf<G: Shape, const N: usize> {
    #[serde(flatten, deserialize_with = "flattened")]
    decoder: config::decoder::Config<scaling::Rope>,
    #[serde(flatten, deserialize_with = "flattened")]
    compiled: Compiled<G, N>,
}

#[derive(Deserialize)]
#[serde(bound = "G: Shape")]
struct Compiled<G: Shape, const N: usize> {
    hidden_size: Size<G::Hidden>,
    intermediate_size: Size<G::Intermediate>,
    num_hidden_layers: Constant<N>,
    num_attention_heads: Size<G::Queries>,
    num_key_value_heads: Size<<G::First as Attention>::Heads>,
    // Transformers' Mistral configurations default to an untied head.
    #[serde(default)]
    tie_word_embeddings: Disabled,
    #[serde(default)]
    sliding_window: Absent,
}

impl<
    G: Shape,
    F: Forms,
    const N: usize,
    const EPSILON: usize,
    const SCALE: usize,
    Q: Query,
    T,
    R: Envelope,
> TryFrom<Hf<G, N>> for Config<G, F, N, EPSILON, SCALE, Q, T, R>
where
    Rotated<G>: AxisName,
{
    type Error = io::Error;

    fn try_from(hf: Hf<G, N>) -> io::Result<Self> {
        type First<G> = <G as Shape>::First;
        const {
            assert!(G::CONSISTENT);
            // The scores of a group of `D`-wide heads scale by 1 / sqrt(D), which the compiled
            // multiplier must be, rounded to FP32.
            let scale = f32::from_bits(SCALE as u32);
            let head = <First<G> as Attention>::Head::SIZE as f32;
            assert!((scale * scale * head - 1.0).abs() <= 4.0 * f32::EPSILON);
        };
        let Hf {
            decoder,
            compiled:
                Compiled {
                    tie_word_embeddings: Disabled,
                    sliding_window: Absent,
                    hidden_size: Size { .. },
                    intermediate_size: Size { .. },
                    num_hidden_layers: Constant,
                    num_attention_heads: Size { .. },
                    num_key_value_heads: Size { .. },
                },
        } = hf;
        vocabulary::<G::Vocabulary>(decoder.geometry.vocab_size.get())?;
        // A checkpoint may leave the head width to its default, so the width it implies compares.
        compiled(
            "head_dim",
            decoder.geometry.head_dim(),
            <First<G> as Attention>::Head::SIZE,
        )?;
        compiled(
            "rms_norm_eps",
            decoder.geometry.rms_norm_eps,
            Some(f32::from_bits(
                u32::try_from(EPSILON).expect("compiled f32 bits"),
            )),
        )?;
        let scaling::Rope {
            spectrum,
            temperature,
        } = decoder.rotary;
        let frequencies = spectrum.bind::<<First<G> as Attention>::Head, Rotated<G>>()?;
        let query = Q::bind(temperature).map_err(|error| {
            io::Error::new(error.kind(), format!("{}: {error}", scaling::TEMPERATURE))
        })?;
        let quantization = decoder.quantization_config;
        match (F::REQUIREMENT, &quantization) {
            (Requirement::Declared(Declaration::Fp8(scheme)), Some(declared))
                if declared.scheme == scheme && !declared.kv_cache => {}
            // The NVFP4 parameters refuse cache scales: a declared E4M3 cache is at unit scale, and
            // a BF16 cache holds its values unrounded.
            (Requirement::Declared(Declaration::Nvfp4(_)), Some(declared))
                if declared.scheme == Scheme::Nvfp4 => {}
            (Requirement::Unquantized | Requirement::Undeclared, None) => {}
            (Requirement::Declared(_), None) => {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "this composition expects declared quantized projections",
                ));
            }
            (_, Some(_)) => {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "checkpoint quantization does not match the projection and cache modules",
                ));
            }
        }
        Ok(Self {
            positions: decoder.max_position_embeddings.get(),
            theta: spectrum.theta().get(),
            frequencies,
            query,
            quantization,
            compiled: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::decoder::Config as _;
    use crate::ops::pointwise::multiply::QueryScale;
    use serde_json::{Value, json};

    /// The served geometry with BF16 projections, whose queries pass `Q`.
    type Unquantized<Q = ()> =
        Config<Sizes, decoder::Projections<form::Affine>, LAYERS, NORM_EPSILON, SCORE_SCALE, Q>;

    /// An untied checkpoint of the served geometry with the rotary fields of `rope`.
    fn checkpoint(rope: Value) -> Value {
        let mut document = json!({
            "hidden_size": Hidden::SIZE, "intermediate_size": Intermediate::SIZE,
            "num_hidden_layers": LAYERS, "num_attention_heads": Queries::SIZE,
            "num_key_value_heads": Heads::SIZE, "head_dim": Head::SIZE,
            "vocab_size": Vocabulary::SIZE, "hidden_act": "silu", "rms_norm_eps": 1e-5,
            "tie_word_embeddings": false, "max_position_embeddings": 262144,
        });
        let fields = document.as_object_mut().unwrap();
        fields.extend(rope.as_object().unwrap().clone());
        document
    }

    fn refusal<T>(parsed: io::Result<T>) -> Option<(io::ErrorKind, String)> {
        parsed.err().map(|error| (error.kind(), error.to_string()))
    }

    /// A configuration keeps the frequencies its checkpoint's spectrum declares, rescaled or not.
    #[test]
    fn keeps_the_declared_spectrum() {
        let mut document = checkpoint(json!({"rope_theta": 1e9}));
        let frequencies = |document: &Value| {
            let config = Unquantized::<()>::parse(document).unwrap();
            let spectrum = config::parse::<scaling::Spectrum>(document).unwrap();
            (
                config.frequencies().unwrap(),
                spectrum.bind::<Head, Head>().unwrap(),
            )
        };
        let (kept, declared) = frequencies(&document);
        assert_eq!(kept, declared);
        document["rope_scaling"] = json!({
            "rope_type": "yarn", "factor": 16.0, "beta_fast": 32.0, "beta_slow": 1.0,
            "mscale": 1.0, "mscale_all_dim": 1.0, "original_max_position_embeddings": 16384,
        });
        let (rescaled, declared) = frequencies(&document);
        assert_eq!(rescaled, declared);
        assert_ne!(rescaled, kept);
    }

    /// A composition whose queries scale binds the temperature the checkpoint's spectrum declares;
    /// a composition whose queries do not scale refuses it, and the scaled one refuses a
    /// checkpoint without it.
    #[test]
    fn binds_the_declared_query_temperature() {
        let mut document = checkpoint(json!({"rope_parameters": {
            "rope_type": "yarn", "rope_theta": 1e9, "factor": 16.0, "mscale": 1.0,
            "mscale_all_dim": 1.0, "original_max_position_embeddings": 16384,
            "llama_4_scaling_beta": 0.1,
        }}));
        let scaled = Unquantized::<QueryScale>::parse(&document).unwrap();
        let temperature = decoder::Queries::<QueryScale>::query(&scaled);
        assert_eq!((temperature.beta, temperature.original.get()), (0.1, 16384));
        let unscaled = Unquantized::<()>::parse(&document);
        let declared = "llama_4_scaling_beta: found query temperature Temperature { beta: 0.1, \
                        original: 16384 }, compiled unscaled queries";
        assert_eq!(
            refusal(unscaled),
            Some((io::ErrorKind::Unsupported, declared.into()))
        );
        let parameters = document["rope_parameters"].as_object_mut().unwrap();
        parameters.remove("llama_4_scaling_beta");
        let scaled = Unquantized::<QueryScale>::parse(&document);
        let missing = "llama_4_scaling_beta: found no query temperature, compiled queries that \
                       scale by one";
        assert_eq!(
            refusal(scaled),
            Some((io::ErrorKind::Unsupported, missing.into()))
        );
        Unquantized::<()>::parse(&document).unwrap();
    }

    /// A configuration that omits the head's tying declares an untied head; a tied head is
    /// unsupported.
    #[test]
    fn reads_an_omitted_tying_as_untied() {
        let mut document = checkpoint(json!({"rope_theta": 1e6}));
        document
            .as_object_mut()
            .unwrap()
            .remove("tie_word_embeddings");
        assert_eq!(refusal(Unquantized::<()>::parse(&document)), None);
        document["tie_word_embeddings"] = json!(true);
        let tied = Unquantized::<()>::parse(&document);
        let reason = "tie_word_embeddings: found true, compiled false";
        assert_eq!(
            refusal(tied),
            Some((io::ErrorKind::Unsupported, reason.into()))
        );
    }

    /// A vocabulary of at most the compiled rows is served; a larger one is unsupported.
    #[test]
    fn serves_a_vocabulary_up_to_its_rows() {
        let mut document = checkpoint(json!({"rope_theta": 1e6}));
        document["vocab_size"] = json!(Vocabulary::SIZE - 1);
        assert_eq!(refusal(Unquantized::<()>::parse(&document)), None);
        document["vocab_size"] = json!(Vocabulary::SIZE + 1);
        let reason = format!(
            "vocab_size: {}, the size holds 1 to {}",
            Vocabulary::SIZE + 1,
            Vocabulary::SIZE
        );
        assert_eq!(
            refusal(Unquantized::<()>::parse(&document)),
            Some((io::ErrorKind::Unsupported, reason))
        );
    }

    /// A configuration whose projections load FP8 blocks serves only a block-scaled checkpoint,
    /// one whose projections load tensor-scaled FP8 only a tensor-scaled one, and one whose
    /// projections load NVFP4 only an NVFP4 one, whatever modules the checkpoint leaves unquantized;
    /// none quantizes its cache, and projections that read no declaration take none.
    #[test]
    fn requires_the_scheme_its_forms_load() {
        use crate::models::Family;
        use crate::models::decoder::{Projections, attention::mistral};
        use form::{ChannelScaled, Nvfp4 as Channelled};
        type Blocks = mistral::Config<mistral::Sizes, Projections<form::BlockScaled>>;
        type Nvfp4 = mistral::Config<mistral::Sizes, Projections<form::Nvfp4BlockScaled>>;
        let document = |quantization: serde_json::Value| {
            json!({
                "hidden_size": mistral::Hidden::SIZE, "intermediate_size": mistral::Intermediate::SIZE,
                "num_hidden_layers": mistral::LAYERS, "num_attention_heads": mistral::Queries::SIZE,
                "num_key_value_heads": mistral::Heads::SIZE, "head_dim": mistral::Head::SIZE,
                "vocab_size": mistral::Vocabulary::SIZE, "hidden_act": "silu", "rms_norm_eps": 1e-5,
                "tie_word_embeddings": false, "rope_theta": 1e6, "max_position_embeddings": 4096,
                "quantization_config": quantization,
            })
        };
        let tensor = || {
            document(json!({
                "quant_method": "fp8", "activation_scheme": "static", "ignored_layers": ["lm_head"],
            }))
        };
        let block = || {
            document(json!({
                "quant_method": "fp8", "activation_scheme": "dynamic",
                "weight_block_size": [128, 128], "ignored_layers": ["lm_head"],
            }))
        };
        let nvfp4 = || {
            let values =
                json!({"num_bits": 4, "type": "float", "dynamic": false, "group_size": 16});
            document(json!({
                "quant_method": "modelopt", "quant_algo": "NVFP4", "ignore": ["lm_head"],
                "config_groups": {
                    "group_0": {"weights": values, "input_activations": values, "targets": ["Linear"]},
                },
                "kv_cache_scheme": {"num_bits": 8, "type": "float", "dynamic": false},
            }))
        };
        <mistral::Config>::parse(&tensor()).unwrap();
        Blocks::parse(&block()).unwrap();
        let mut excluding = block();
        excluding["quantization_config"]["ignored_layers"] =
            json!(["lm_head", "model.vision_tower", "re:.*vision.*"]);
        Blocks::parse(&excluding).unwrap();
        let declared = Nvfp4::parse(&nvfp4()).unwrap().quantization;
        assert!(
            matches!(declared, Some(nvfp4) if nvfp4.scheme == crate::models::config::Scheme::Nvfp4
            && nvfp4.kv_cache && nvfp4.excluded.excludes("lm_head"))
        );
        let mut unquantized = nvfp4();
        unquantized
            .as_object_mut()
            .unwrap()
            .remove("quantization_config");
        let mut malformed = nvfp4();
        malformed["quantization_config"]["config_groups"]["group_0"]["weights"]["num_bits"] =
            json!(8);
        let mut cached = block();
        cached["quantization_config"]["kv_cache_scheme"] =
            json!({"num_bits": 8, "type": "float", "dynamic": false});
        type Undeclared = mistral::Config<
            mistral::Sizes,
            Projections<ChannelScaled, ChannelScaled, ChannelScaled, ChannelScaled, Channelled>,
        >;
        Undeclared::parse(&unquantized).unwrap();
        let mismatch = "checkpoint quantization does not match the projection and cache modules";
        for (refusal, reason) in [
            (<mistral::Config>::parse(&block()).map(drop), mismatch),
            (Blocks::parse(&tensor()).map(drop), mismatch),
            (<mistral::Config>::parse(&nvfp4()).map(drop), mismatch),
            (Blocks::parse(&nvfp4()).map(drop), mismatch),
            (Nvfp4::parse(&block()).map(drop), mismatch),
            (Blocks::parse(&cached).map(drop), mismatch),
            (Undeclared::parse(&tensor()).map(drop), mismatch),
            (
                Nvfp4::parse(&unquantized).map(drop),
                "this composition expects declared quantized projections",
            ),
            (
                Nvfp4::parse(&malformed).map(drop),
                "quantization_config: found 8, compiled 4",
            ),
        ] {
            let error = refusal.unwrap_err();
            assert_eq!(
                (error.kind(), error.to_string()),
                (io::ErrorKind::Unsupported, reason.into())
            );
        }
    }
}
