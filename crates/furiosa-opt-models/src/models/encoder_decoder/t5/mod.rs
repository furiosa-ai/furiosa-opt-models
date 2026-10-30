//! T5, by published size: a [`Size`] fixes what its checkpoints declare, and the catalog
//! composition below the sizes encodes and decodes with relative-position attention.
//!
//! - v1.0 and v1.1 share size words but not geometry: v1.0's `"small"` has a ReLU feed-forward and
//!   reads its head from the shared token table, scaling the decoder's rows first; v1.1's
//!   `"v1_1-small"` has a gated GELU feed-forward and a head of its own.
//! - mT5's sizes implement the same [`Geometry`] over a 250112-token vocabulary.
//! - Every size has heads of 64 features, 32 relative-position buckets reaching 128 positions
//!   and a 1e-6 norm epsilon.

use crate::models::Family;
use crate::models::encoder::NormalizedEncoder;
pub use crate::models::{Ids, Scalar};
use crate::modules::attention::relative;
use crate::modules::multimodal;
use crate::modules::{attention, feedforward, feedforward::glu::Glu, transformer};
use crate::ops::attention::Attention;
pub use crate::ops::attention::geometry::Heads;
use crate::ops::attention::geometry::{self, QueryWidth};
use crate::ops::swiglu;
use crate::ops::{
    embedding::Lookup,
    index::scatter::Store,
    linear::Coefficients,
    normalization::Rms,
    pointwise::{activation::Activation, add::Add, multiply::Multiply},
};
use crate::{First, Module};
use furiosa_opt_std::prelude::*;
use serde::Deserialize;
use std::{io, marker::PhantomData, num::NonZeroUsize};

pub mod decoder;

/// A published T5 checkpoint size, by its hub spelling: `"small"` is google-t5/t5-small,
/// `"v1_1-small"` google/t5-v1_1-small.
pub struct Size<const N: &'static str>;

/// A published size of the T5 tree, T5's or mT5's: its axes and the config values its type fixes.
pub trait Geometry {
    /// Features of a row: `d_model`.
    type Hidden: AxisName;
    /// Features of the feed-forward: `d_ff`.
    type Intermediate: AxisName;
    /// Attention heads of a layer: `num_heads`.
    type Heads: AxisName;
    /// Features of every head together: `num_heads * d_kv`.
    type Queries: AxisName;
    /// Rows of the shared token table and of the head: `vocab_size`.
    type Vocabulary: AxisName;

    /// The hub spelling of the size, as `"v1_1-small"`.
    const SIZE: &'static str;
    /// The family's `model_type`.
    const MODEL_TYPE: &'static str;
    /// `num_layers`, the encoder's.
    const LAYERS: usize;
    /// `num_decoder_layers`.
    const DECODER_LAYERS: usize;
    /// `feed_forward_proj`, one of [`FORMS`].
    const FEED_FORWARD: &'static str;
    /// The head reads the shared token table, and the decoder's rows are scaled by
    /// `d_model ** -0.5` before it.
    const TIED_EMBEDDINGS: bool;
}

axes![Head = 64, Buckets = 32, Vocabulary = 32128];

/// The feed-forward forms of the T5 tree, by `feed_forward_proj`: two projections around a ReLU,
/// or a GELU-gated pair before the output projection.
pub const FORMS: [&str; 2] = ["relu", "gated-gelu"];

/// `relative_attention_max_distance`: positions this far apart or farther share a bucket.
const DISTANCE: usize = 128;

/// `layer_norm_epsilon` 1e-6, as the bits of the FP32 nearest to it.
const EPSILON: usize = 1e-6_f32.to_bits() as usize;

impl Geometry for Size<"small"> {
    type Hidden = geometry::Hidden<512>;
    type Intermediate = swiglu::Intermediate<2048>;
    type Heads = Heads<8>;
    type Queries = QueryWidth<512>;
    type Vocabulary = Vocabulary;

    const SIZE: &'static str = "small";
    const MODEL_TYPE: &'static str = "t5";
    const LAYERS: usize = 6;
    const DECODER_LAYERS: usize = 6;
    const FEED_FORWARD: &'static str = FORMS[0];
    const TIED_EMBEDDINGS: bool = true;
}

impl Geometry for Size<"base"> {
    type Hidden = geometry::Hidden<768>;
    type Intermediate = swiglu::Intermediate<3072>;
    type Heads = Heads<12>;
    type Queries = QueryWidth<768>;
    type Vocabulary = Vocabulary;

    const SIZE: &'static str = "base";
    const MODEL_TYPE: &'static str = "t5";
    const LAYERS: usize = 12;
    const DECODER_LAYERS: usize = 12;
    const FEED_FORWARD: &'static str = FORMS[0];
    const TIED_EMBEDDINGS: bool = true;
}

impl Geometry for Size<"large"> {
    type Hidden = geometry::Hidden<1024>;
    type Intermediate = swiglu::Intermediate<4096>;
    type Heads = Heads<16>;
    type Queries = QueryWidth<1024>;
    type Vocabulary = Vocabulary;

    const SIZE: &'static str = "large";
    const MODEL_TYPE: &'static str = "t5";
    const LAYERS: usize = 24;
    const DECODER_LAYERS: usize = 24;
    const FEED_FORWARD: &'static str = FORMS[0];
    const TIED_EMBEDDINGS: bool = true;
}

impl Geometry for Size<"v1_1-small"> {
    type Hidden = geometry::Hidden<512>;
    type Intermediate = swiglu::Intermediate<1024>;
    type Heads = Heads<6>;
    type Queries = QueryWidth<384>;
    type Vocabulary = Vocabulary;

    const SIZE: &'static str = "v1_1-small";
    const MODEL_TYPE: &'static str = "t5";
    const LAYERS: usize = 8;
    const DECODER_LAYERS: usize = 8;
    const FEED_FORWARD: &'static str = FORMS[1];
    const TIED_EMBEDDINGS: bool = false;
}

impl Geometry for Size<"v1_1-base"> {
    type Hidden = geometry::Hidden<768>;
    type Intermediate = swiglu::Intermediate<2048>;
    type Heads = Heads<12>;
    type Queries = QueryWidth<768>;
    type Vocabulary = Vocabulary;

    const SIZE: &'static str = "v1_1-base";
    const MODEL_TYPE: &'static str = "t5";
    const LAYERS: usize = 12;
    const DECODER_LAYERS: usize = 12;
    const FEED_FORWARD: &'static str = FORMS[1];
    const TIED_EMBEDDINGS: bool = false;
}

impl Geometry for Size<"v1_1-large"> {
    type Hidden = geometry::Hidden<1024>;
    type Intermediate = swiglu::Intermediate<2816>;
    type Heads = Heads<16>;
    type Queries = QueryWidth<1024>;
    type Vocabulary = Vocabulary;

    const SIZE: &'static str = "v1_1-large";
    const MODEL_TYPE: &'static str = "t5";
    const LAYERS: usize = 24;
    const DECODER_LAYERS: usize = 24;
    const FEED_FORWARD: &'static str = FORMS[1];
    const TIED_EMBEDDINGS: bool = false;
}

impl Geometry for Size<"v1_1-xl"> {
    type Hidden = geometry::Hidden<2048>;
    type Intermediate = swiglu::Intermediate<5120>;
    type Heads = Heads<32>;
    type Queries = QueryWidth<2048>;
    type Vocabulary = Vocabulary;

    const SIZE: &'static str = "v1_1-xl";
    const MODEL_TYPE: &'static str = "t5";
    const LAYERS: usize = 24;
    const DECODER_LAYERS: usize = 24;
    const FEED_FORWARD: &'static str = FORMS[1];
    const TIED_EMBEDDINGS: bool = false;
}

impl Geometry for Size<"v1_1-xxl"> {
    type Hidden = geometry::Hidden<4096>;
    type Intermediate = swiglu::Intermediate<10240>;
    type Heads = Heads<64>;
    type Queries = QueryWidth<4096>;
    type Vocabulary = Vocabulary;

    const SIZE: &'static str = "v1_1-xxl";
    const MODEL_TYPE: &'static str = "t5";
    const LAYERS: usize = 24;
    const DECODER_LAYERS: usize = 24;
    const FEED_FORWARD: &'static str = FORMS[1];
    const TIED_EMBEDDINGS: bool = false;
}

/// A T5 or mT5 checkpoint's config; an absent key holds its config class's default.
#[derive(Deserialize)]
pub struct Hf {
    model_type: String,
    vocab_size: Option<usize>,
    d_model: Option<usize>,
    d_kv: Option<usize>,
    d_ff: Option<usize>,
    num_layers: Option<usize>,
    num_decoder_layers: Option<usize>,
    num_heads: Option<usize>,
    relative_attention_num_buckets: Option<usize>,
    relative_attention_max_distance: Option<usize>,
    layer_norm_epsilon: Option<f64>,
    feed_forward_proj: Option<String>,
    tie_word_embeddings: Option<bool>,
    scale_decoder_outputs: Option<bool>,
}

/// A config class's defaults for the keys a checkpoint's config may omit; `num_decoder_layers`
/// defaults to `num_layers`.
pub struct Defaults {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_kv: usize,
    pub d_ff: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub relative_attention_num_buckets: usize,
    pub relative_attention_max_distance: usize,
    pub layer_norm_epsilon: f64,
    pub feed_forward_proj: &'static str,
}

/// `T5Config`'s defaults: T5-small's geometry, with a ReLU feed-forward.
pub const DEFAULTS: Defaults = Defaults {
    vocab_size: 32128,
    d_model: 512,
    d_kv: 64,
    d_ff: 2048,
    num_layers: 6,
    num_heads: 8,
    relative_attention_num_buckets: 32,
    relative_attention_max_distance: 128,
    layer_norm_epsilon: 1e-6,
    feed_forward_proj: "relu",
};

impl Hf {
    /// Refuses a config, absent keys read as `defaults`, whose values the size `G` does not hold
    /// (invalid) or whose feed-forward form [`FORMS`] lacks (unsupported).
    pub fn check<G: Geometry>(&self, defaults: &Defaults) -> io::Result<()> {
        let invalid = |reason: String| io::Error::new(io::ErrorKind::InvalidData, reason);
        if self.model_type != G::MODEL_TYPE {
            return Err(invalid(format!(
                "model_type: {}, the size {} is {}",
                self.model_type,
                G::SIZE,
                G::MODEL_TYPE
            )));
        }
        let layers = self.num_layers.unwrap_or(defaults.num_layers);
        let fixed = [
            (
                "vocab_size",
                self.vocab_size.unwrap_or(defaults.vocab_size),
                G::Vocabulary::SIZE,
            ),
            (
                "d_model",
                self.d_model.unwrap_or(defaults.d_model),
                G::Hidden::SIZE,
            ),
            ("d_kv", self.d_kv.unwrap_or(defaults.d_kv), Head::SIZE),
            (
                "d_ff",
                self.d_ff.unwrap_or(defaults.d_ff),
                G::Intermediate::SIZE,
            ),
            (
                "num_heads",
                self.num_heads.unwrap_or(defaults.num_heads),
                G::Heads::SIZE,
            ),
            ("num_layers", layers, G::LAYERS),
            (
                "num_decoder_layers",
                self.num_decoder_layers.unwrap_or(layers),
                G::DECODER_LAYERS,
            ),
            (
                "relative_attention_num_buckets",
                self.relative_attention_num_buckets
                    .unwrap_or(defaults.relative_attention_num_buckets),
                Buckets::SIZE,
            ),
            (
                "relative_attention_max_distance",
                self.relative_attention_max_distance
                    .unwrap_or(defaults.relative_attention_max_distance),
                DISTANCE,
            ),
        ];
        if let Some((key, declared, held)) = fixed
            .into_iter()
            .find(|(_, declared, held)| declared != held)
        {
            return Err(invalid(format!(
                "{key}: {declared}, the size {} holds {held}",
                G::SIZE
            )));
        }
        let epsilon = self
            .layer_norm_epsilon
            .unwrap_or(defaults.layer_norm_epsilon);
        if (epsilon as f32).to_bits() as usize != EPSILON {
            return Err(invalid(format!(
                "layer_norm_epsilon: {epsilon}, the size {} holds 1e-6",
                G::SIZE
            )));
        }
        let form = self
            .feed_forward_proj
            .as_deref()
            .unwrap_or(defaults.feed_forward_proj);
        match (form == G::FEED_FORWARD, FORMS.contains(&form)) {
            (true, _) => Ok(()),
            (false, true) => Err(invalid(format!(
                "feed_forward_proj: {form}, the size {} holds {}",
                G::SIZE,
                G::FEED_FORWARD
            ))),
            (false, false) => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "feed_forward_proj: {form}, T5 serves {}",
                    FORMS.join(" and ")
                ),
            )),
        }
    }
}

/// A config of the size's values and tying: a tied head scales the decoder's rows unless
/// `scale_decoder_outputs`, or else `tie_word_embeddings`, is false.
impl<const N: &'static str> TryFrom<Hf> for Size<N>
where
    Self: Geometry,
{
    type Error = io::Error;

    fn try_from(hf: Hf) -> io::Result<Self> {
        hf.check::<Self>(&DEFAULTS)?;
        let (key, tied) = match (hf.scale_decoder_outputs, hf.tie_word_embeddings) {
            (Some(scaled), _) => ("scale_decoder_outputs", scaled),
            (None, tied) => ("tie_word_embeddings", tied.unwrap_or(true)),
        };
        match tied == Self::TIED_EMBEDDINGS {
            true => Ok(Size),
            false => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{key}: {tied}, the size {N} ties: {}",
                    Self::TIED_EMBEDDINGS
                ),
            )),
        }
    }
}

impl<const N: &'static str> Family for Size<N>
where
    Self: Geometry,
{
    type Hf = Hf;
}

/// Encoder and decoder sharing projected source memory across decode calls; the env is the step's
/// positions, mask and source mask, a decoder layer's slot its self cache beside its cross memory.
pub type EncoderDecoder<E, D> = multimodal::EncoderDecoder<E, D>;

/// Encoder specialized to token width, attention geometry, and feed-forward form.
pub type Encoder<Vocabulary, Width, KvWidth, Heads, Buckets, F, const LAYERS: usize> = Body<
    Embedding<Vocabulary, Width>,
    [Layer<Width, KvWidth, Heads, F>; LAYERS],
    Norm<Width>,
    relative::Bias<relative::Table<Buckets, Heads>>,
>;

/// Token encoder sharing one relative-position bias across all layers.
#[derive(DeviceSend)]
pub struct Body<Embedding, Layers, Norm, Bias> {
    /// Shared token embedding table.
    pub embedding: Embedding,
    /// Pre-normalized attention and feed-forward layers.
    pub layers: Layers,
    /// Final per-row RMS normalization.
    pub norm: Norm,
    /// Relative-position table and bucket-distance configuration.
    pub bias: Bias,
}

/// The layers' attention reads the masked relative-position bias in its context; the state holds
/// the embedding's slot, a slot a layer, lent by index, and the norm's.
impl<
    's,
    'c,
    Embedding,
    Layer,
    Norm,
    B,
    EmbeddingState,
    LayerState,
    NormState,
    Width: AxisName,
    Count: AxisName,
    Heads: AxisName,
    const LAYERS: usize,
    Env: Copy,
>
    Module<
        &'c mut Context<Width, Count, LAYERS>,
        Env,
        &'s mut (EmbeddingState, [LayerState; LAYERS], NormState),
    > for Body<Embedding, [Layer; LAYERS], Norm, B>
where
    Embedding: Module<&'c Ids<Count>, Env, &'s mut EmbeddingState, Output = Rows<Width, Count>>,
    B: for<'d, 'x> Module<
            (&'x relative::Positions<&'x Ids<Count>>, &'x Mask<Count>),
            Env,
            &'d mut (),
            Output = Bias<Heads, Count>,
        >,
    Layer: for<'t, 'b> Module<
            Rows<Width, Count>,
            &'b attention::Context<&'static (), (), &'b Bias<Heads, Count>>,
            &'t mut LayerState,
            Output = Rows<Width, Count>,
        >,
    Norm:
        for<'x> Module<&'x Rows<Width, Count>, Env, &'s mut NormState, Output = Rows<Width, Count>>,
{
    type Output = Rows<Width, Count>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (embedding, layers, norm): &'s mut (EmbeddingState, [LayerState; LAYERS], NormState),
        context: &'c mut Context<Width, Count, LAYERS>,
    ) -> Self::Output {
        let Context {
            tokens,
            positions,
            mask,
            features,
        } = context;
        let relative = relative::Positions {
            query: &*positions,
            key: &*positions,
        };
        let bias = self
            .bias
            .forward(&mut *device, env, &mut (), (&relative, &*mask));
        let attention = attention::Context {
            positions: &(),
            write: (),
            read: &bias,
        };
        let mut hidden = self
            .embedding
            .forward(&mut *device, env, embedding, &*tokens);
        #[unroll]
        for index in 0..LAYERS {
            Store.forward(&mut *device, env, &mut (), (&hidden, &mut features[index]));
            hidden =
                self.layers[index].forward(&mut *device, &attention, &mut layers[index], hidden);
        }
        self.norm.forward(device, env, norm, &hidden)
    }
}

/// Token lookup into the shared F32 embedding table.
pub type Embedding<Vocabulary, Width> = Lookup<HbmTensor<f32, m![1], m![Vocabulary, Width]>>;

type Layer<Width, KvWidth, Heads, F> = transformer::PreNorm<
    attention::Full<
        attention::Positioned<attention::Projection<Linear<Width, KvWidth>>, First>,
        Attention<PhantomData<Heads>, Scalar>,
        Linear<KvWidth, Width>,
    >,
    F,
    Norm<Width>,
    Add,
>;

/// Bias-free F32 feature projection.
pub type Linear<In, Out> = Coefficients<f32, In, Out, 0>;

/// Per-row RMS normalization with a placed epsilon.
pub type Norm<Width> = Rms<HbmTensor<f32, m![1], m![Width]>, Scalar>;

/// Two-projection feed-forward residual branch.
pub type Dense<Width, Inner, A> = feedforward::Mlp<Linear<Width, Inner>, A, Linear<Inner, Width>>;

/// Feed-forward branch multiplying activated gate and up-projection features.
pub type Gated<Width, Inner, A> =
    Glu<Linear<Width, Inner>, A, Linear<Width, Inner>, Multiply, Linear<Inner, Width>>;

/// Token-major F32 feature rows.
pub type Rows<Width, Count> = HbmTensor<f32, m![1], m![Count, Width]>;

/// Additive query/key attention mask, with blocked entries set to negative infinity.
pub type Mask<Queries, Keys = Queries> = HbmTensor<f32, m![1], m![Queries, Keys]>;

/// Per-head relative-position bias after adding the attention mask.
pub type Bias<Heads, Queries, Keys = Queries> = HbmTensor<f32, m![1], m![Heads, Queries, Keys]>;

/// Query and key token positions used to select relative-position buckets.
pub type Positions<Queries, Keys> = relative::Positions<Ids<Queries>, Ids<Keys>>;

/// Encoder tokens, relative positions, additive mask, and writable layer captures.
#[derive(DeviceSend)]
pub struct Context<Width: AxisName, Count: AxisName, const LAYERS: usize> {
    /// Token indices into the shared embedding table.
    pub tokens: Ids<Count>,
    /// Token positions, which the relative-position bias reads for both queries and keys.
    pub positions: Ids<Count>,
    /// Additive attention mask broadcast across heads.
    pub mask: Mask<Count>,
    /// Captures retain layer inputs; the final normalized output is returned.
    pub features: [Rows<Width, Count>; LAYERS],
}

/// Checkpoint dimensions and feed-forward selection, each read as its key's T5 default when absent.
#[derive(Clone)]
pub struct Config {
    /// Hidden feature width.
    pub hidden: usize,
    /// Feed-forward intermediate width.
    pub intermediate: usize,
    /// Key/value width of one attention head.
    pub head: usize,
    /// Query and key/value head count.
    pub heads: usize,
    /// Encoder layer count.
    pub layers: usize,
    /// Shared token embedding-table row count.
    pub vocabulary: usize,
    /// Relative-position bucket count.
    pub buckets: usize,
    pub distance: usize,
    pub epsilon: f64,
    /// Parsed feed-forward activation; the loader's activation must agree.
    pub activation: Activation,
    /// Whether feed-forward branches include separate gate and up projections.
    pub gated: bool,
    /// Causal decoder layer count.
    pub decoder_layers: usize,
    pub tied: bool,
}

impl Family for Config {
    type Hf = Hf;
}

impl TryFrom<Hf> for Config {
    type Error = io::Error;

    /// Narrows geometry and feed-forward form.
    fn try_from(hf: Hf) -> io::Result<Self> {
        let dimension = |name: &str, value: Option<usize>, default| {
            NonZeroUsize::new(value.unwrap_or(default))
                .map(NonZeroUsize::get)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("{name}: expected positive dimension"),
                    )
                })
        };
        let projection = hf
            .feed_forward_proj
            .as_deref()
            .unwrap_or(DEFAULTS.feed_forward_proj);
        let gated = projection.starts_with("gated-");
        let activation = match projection.strip_prefix("gated-").unwrap_or(projection) {
            "relu" => Activation::Relu,
            "gelu" if gated => Activation::NewGelu,
            "gelu" => Activation::Gelu,
            "gelu_new" => Activation::NewGelu,
            "silu" | "swish" => Activation::Silu,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "unimplemented feedforward activation",
                ));
            }
        };
        let layers = dimension("num_layers", hf.num_layers, DEFAULTS.num_layers)?;
        let config = Self {
            hidden: dimension("d_model", hf.d_model, DEFAULTS.d_model)?,
            intermediate: dimension("d_ff", hf.d_ff, DEFAULTS.d_ff)?,
            head: dimension("d_kv", hf.d_kv, DEFAULTS.d_kv)?,
            heads: dimension("num_heads", hf.num_heads, DEFAULTS.num_heads)?,
            layers,
            vocabulary: dimension("vocab_size", hf.vocab_size, DEFAULTS.vocab_size)?,
            buckets: dimension(
                "relative_attention_num_buckets",
                hf.relative_attention_num_buckets,
                DEFAULTS.relative_attention_num_buckets,
            )?,
            distance: dimension(
                "relative_attention_max_distance",
                hf.relative_attention_max_distance,
                DEFAULTS.relative_attention_max_distance,
            )?,
            epsilon: hf.layer_norm_epsilon.unwrap_or(DEFAULTS.layer_norm_epsilon),
            activation,
            gated,
            decoder_layers: dimension("num_decoder_layers", hf.num_decoder_layers, layers)?,
            tied: hf.tie_word_embeddings.unwrap_or(true),
        };
        if config.head.checked_mul(config.heads).is_none()
            || !(config.epsilon as f32).is_finite()
            || config.epsilon as f32 <= 0.0
            || config.buckets < 4
            || config.buckets % 2 != 0
            || config.distance <= config.buckets / 2
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid encoder normalization or relative-position configuration",
            ));
        }
        Ok(config)
    }
}

/// Source: Transformers `src/transformers/models/t5/modeling_t5.py`.
/// Relative position bias belongs to attention; the learned table is shared across encoder layers.
pub type T5EncoderModel<Embedding, Qkv, Attend, Project, Mlp, Norm, Add, const DEPTH: usize> =
    NormalizedEncoder<Embedding, Qkv, Attend, Project, Mlp, Norm, Add, DEPTH>;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    /// An encoder of T5's defaults, positive dimensions, a known activation, a finite positive
    /// epsilon and buckets that a distance past half of them pairs parses; each other value is
    /// refused by its kind and reason.
    #[test]
    fn refuses_what_the_encoder_does_not_run() {
        let parse = |edit: &[(&str, Value)]| {
            let mut document = json!({"model_type": "t5"});
            for (key, value) in edit {
                document[*key] = value.clone();
            }
            <Config as Family>::parse(&document).map(drop)
        };
        parse(&[]).unwrap();
        use io::ErrorKind::{InvalidData, Unsupported};
        let invalid = "invalid encoder normalization or relative-position configuration";
        let mut cases = vec![
            (
                vec![("feed_forward_proj", json!("gated-tanh"))],
                Unsupported,
                "unimplemented feedforward activation".to_owned(),
            ),
            (
                vec![("feed_forward_proj", json!("tanh"))],
                Unsupported,
                "unimplemented feedforward activation".to_owned(),
            ),
            (
                vec![("d_kv", json!(1u64 << 63)), ("num_heads", json!(2))],
                InvalidData,
                invalid.to_owned(),
            ),
            (
                vec![("layer_norm_epsilon", json!(1e39))],
                InvalidData,
                invalid.to_owned(),
            ),
            (
                vec![("layer_norm_epsilon", json!(0.0))],
                InvalidData,
                invalid.to_owned(),
            ),
            (
                vec![("relative_attention_num_buckets", json!(2))],
                InvalidData,
                invalid.to_owned(),
            ),
            (
                vec![("relative_attention_num_buckets", json!(5))],
                InvalidData,
                invalid.to_owned(),
            ),
            (
                vec![("relative_attention_max_distance", json!(16))],
                InvalidData,
                invalid.to_owned(),
            ),
        ];
        for name in [
            "num_layers",
            "d_model",
            "d_ff",
            "d_kv",
            "num_heads",
            "vocab_size",
            "relative_attention_num_buckets",
            "relative_attention_max_distance",
            "num_decoder_layers",
        ] {
            let reason = format!("{name}: expected positive dimension");
            cases.push((vec![(name, json!(0))], InvalidData, reason));
        }
        for (edit, kind, reason) in cases {
            let error = parse(&edit).unwrap_err();
            assert_eq!(
                (error.kind(), error.to_string()),
                (kind, reason),
                "{edit:?}"
            );
        }
    }
}
