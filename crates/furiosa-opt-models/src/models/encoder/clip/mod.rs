//! Text encoding with caller-owned visibility, intermediate features and a pooling readout.
use super::NormalizedEncoder;
use crate::models::Family;
use crate::models::config;
pub use crate::models::{Ids, Scalar};
use crate::modules::{attention, feedforward, language, transformer};
use crate::ops::attention::Attention;
use crate::ops::{
    embedding::Lookup,
    linear::Coefficients,
    normalization::LayerNorm,
    pointwise::{
        activation::{Activation, QuickGelu},
        add::Add,
    },
};
use crate::{First, Identity, Module, Then};
use furiosa_opt_std::prelude::*;
pub use pool::Pool;
use serde::Deserialize;
use std::{io, marker::PhantomData, num::NonZeroUsize};

/// Pooled text features, with or without a learned output projection.
pub enum Model<Body, Project> {
    /// Selected hidden row without projection.
    Encoder(Encoder<Body, Identity>),
    /// Selected hidden row projected into the shared embedding space.
    Projection(Encoder<Body, Project>),
}

/// Both pooling variants at one text checkpoint geometry.
pub type Configured<
    Vocabulary,
    Positions,
    Width,
    Intermediate,
    Heads,
    Projection,
    const LAYERS: usize,
> = Model<
    Text<Vocabulary, Positions, Width, Intermediate, Heads, LAYERS>,
    Linear<Width, Projection, 0>,
>;

/// Text encoder with caller-selected row pooling and an optional projection.
pub type Encoder<Body, Project> = language::Pooler<Body, Pool<Project>>;

pub mod pool;

/// Token and absolute-position embedding followed by normalized text encoder layers.
pub type Text<
    Vocabulary,
    Positions,
    Width,
    Intermediate,
    Heads,
    const LAYERS: usize,
    A = QuickGelu,
> = Body<
    Embeddings<Vocabulary, Positions, Width>,
    [Layer<Width, Intermediate, Heads, A>; LAYERS],
    Then<Identity, Norm<Width>>,
>;

/// Embedding, encoder layers and final normalization with caller-owned feature captures.
pub type Body<Embedding, Layers, Norm> = transformer::Encoder<Embedding, Layers, Norm>;

/// The layers' attention reads the operands' bias in its context.
impl<
    's,
    'b,
    Embedding,
    Layer,
    Norm,
    EmbeddingState,
    LayerState,
    NormState,
    Width: AxisName,
    Count: AxisName,
    const LAYERS: usize,
>
    Module<
        &'b mut Operands<Width, Count, LAYERS>,
        (),
        &'s mut (EmbeddingState, [LayerState; LAYERS], NormState),
    > for Body<Embedding, [Layer; LAYERS], Norm>
where
    Embedding: for<'x, 'c> Module<
            (&'x Ids<Count>, &'x Ids<Count>),
            &'c attention::Context<&'static (), (), &'b Bias<Count>>,
            &'s mut EmbeddingState,
            Output = Rows<Width, Count>,
        >,
    Layer: for<'t, 'c> Module<
            Rows<Width, Count>,
            &'c attention::Context<&'static (), (), &'b Bias<Count>>,
            &'t mut LayerState,
            Output = Rows<Width, Count>,
        >,
    Norm: for<'c> Module<
            Rows<Width, Count>,
            &'c attention::Context<&'static (), (), &'b Bias<Count>>,
            &'s mut NormState,
            Output = Rows<Width, Count>,
        >,
{
    type Output = Rows<Width, Count>;

    fn forward(
        &self,
        device: &mut Device,
        _: (),
        state: &'s mut (EmbeddingState, [LayerState; LAYERS], NormState),
        operands: &'b mut Operands<Width, Count, LAYERS>,
    ) -> Self::Output {
        let Operands {
            tokens,
            positions,
            bias,
            features,
            last,
        } = operands;
        let context = attention::Context {
            positions: &(),
            write: (),
            read: &*bias,
        };
        self.forward(
            device,
            &context,
            state,
            ((&*tokens, &*positions), &mut *features, &mut *last),
        )
    }
}

pub type Embeddings<Vocabulary, Positions, Width> = language::ConditionedEmbedding<
    Lookup<HbmTensor<f32, m![1], m![Vocabulary, Width]>>,
    Lookup<HbmTensor<f32, m![1], m![Positions, Width]>>,
    Add,
>;

pub type Layer<Width, Intermediate, Heads, Activation> = transformer::PreNorm<
    attention::Full<
        attention::Positioned<attention::Projection<Linear<Width, Width>>, First>,
        Attention<PhantomData<Heads>, Scalar>,
        Linear<Width, Width>,
    >,
    feedforward::Mlp<Linear<Width, Intermediate>, Activation, Linear<Intermediate, Width>>,
    Norm<Width>,
    Add,
>;

pub type Linear<In, Out, const BIAS: usize = 1> = Coefficients<f32, In, Out, BIAS>;

pub type Norm<Width> = LayerNorm<Vector<Width>, Scalar>;

type Vector<Width> = HbmTensor<f32, m![1], m![Width]>;

/// Row-major F32 text features.
pub type Rows<Width, Count> = HbmTensor<f32, m![1], m![Count, Width]>;

/// Additive attention visibility indexed by query and key row.
pub type Bias<Count> = HbmTensor<f32, m![1], m![Count, Count]>;

/// Text indices, additive visibility and writable intermediate feature captures.
#[derive(DeviceSend)]
pub struct Operands<Width: AxisName, Count: AxisName, const LAYERS: usize> {
    /// Token indices per row.
    pub tokens: Ids<Count>,
    /// Absolute position indices per row.
    pub positions: Ids<Count>,
    /// Additive attention mask indexed by query and key row.
    pub bias: Bias<Count>,
    /// The input of every encoder layer, beginning with the combined embedding.
    pub features: [Rows<Width, Count>; LAYERS],
    /// The final encoder output before normalization.
    pub last: Rows<Width, Count>,
}

/// Parsed text encoder dimensions, normalization and projection choice.
#[derive(Clone)]
pub struct Config {
    /// Hidden feature width.
    pub hidden: usize,
    /// Feed-forward expansion width.
    pub intermediate: usize,
    /// Attention heads per layer.
    pub heads: usize,
    /// Encoder layer count.
    pub layers: usize,
    /// Number of token embeddings.
    pub vocabulary: usize,
    /// Number of absolute position embeddings.
    pub positions: usize,
    /// Layer-normalization epsilon.
    pub epsilon: f64,
    /// Checkpoint feed-forward activation.
    pub activation: Activation,
    /// Output width when a learned text projection is requested.
    pub projection: Option<usize>,
}

impl Family for Config {
    type Hf = Hf;
}

/// A text encoder's checkpoint configuration.
#[derive(Deserialize)]
pub struct Hf {
    architectures: Option<Vec<String>>,
    model_type: Option<String>,
    projection_dim: Option<usize>,
    #[serde(flatten, deserialize_with = "config::text")]
    text: Tower,
}

impl TryFrom<Hf> for Config {
    type Error = io::Error;

    /// Narrows text geometry with whether the declared architecture has a projection head.
    fn try_from(hf: Hf) -> io::Result<Self> {
        let projected = (hf.architectures.iter().flatten())
            .any(|name| matches!(name.as_str(), "CLIPTextModelWithProjection" | "CLIPModel"));
        let siglip = hf
            .model_type
            .as_deref()
            .is_some_and(|kind| matches!(kind, "siglip" | "siglip_text_model"));
        let text = hf.text;
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
        let default = if siglip {
            "gelu_pytorch_tanh"
        } else {
            "quick_gelu"
        };
        let activation = match text.hidden_act.as_deref().unwrap_or(default) {
            "quick_gelu" => Activation::QuickGelu,
            "gelu" => Activation::Gelu,
            "gelu_new" | "gelu_pytorch_tanh" => Activation::NewGelu,
            "relu" => Activation::Relu,
            "silu" => Activation::Silu,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "unimplemented feedforward activation",
                ));
            }
        };
        let config = Self {
            activation,
            hidden: dimension(
                "hidden_size",
                text.hidden_size,
                if siglip { 768 } else { 512 },
            )?,
            intermediate: dimension(
                "intermediate_size",
                text.intermediate_size,
                if siglip { 3072 } else { 2048 },
            )?,
            heads: dimension(
                "num_attention_heads",
                text.num_attention_heads,
                if siglip { 12 } else { 8 },
            )?,
            layers: dimension("num_hidden_layers", text.num_hidden_layers, 12)?,
            vocabulary: dimension(
                "vocab_size",
                text.vocab_size,
                if siglip { 32000 } else { 49408 },
            )?,
            positions: dimension(
                "max_position_embeddings",
                text.max_position_embeddings,
                if siglip { 64 } else { 77 },
            )?,
            epsilon: text
                .layer_norm_eps
                .unwrap_or(if siglip { 1e-6 } else { 1e-5 }),
            projection: projected
                .then(|| match hf.projection_dim {
                    Some(dimension) => NonZeroUsize::new(dimension)
                        .map(NonZeroUsize::get)
                        .ok_or_else(|| {
                            io::Error::new(
                                io::ErrorKind::InvalidData,
                                "projection_dim must be positive",
                            )
                        }),
                    None => dimension("projection_dim", text.projection_dim, 512),
                })
                .transpose()?,
        };
        if config.hidden % config.heads != 0 || !config.epsilon.is_finite() || config.epsilon <= 0.0
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid text encoder dimensions or normalization",
            ));
        }
        Ok(config)
    }
}

/// The text tower's geometry, each field defaulting by the model type.
#[derive(Deserialize)]
struct Tower {
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_attention_heads: Option<usize>,
    num_hidden_layers: Option<usize>,
    vocab_size: Option<usize>,
    max_position_embeddings: Option<usize>,
    hidden_act: Option<String>,
    layer_norm_eps: Option<f64>,
    projection_dim: Option<usize>,
}

/// Source: Transformers `src/transformers/models/clip/modeling_clip.py`.
/// Token features and EOS pooling remain separately callable through body and head.
pub type CLIPTextModel<Embedding, Qkv, Attend, Project, Mlp, Norm, Add, Pool, const DEPTH: usize> =
    language::Pooler<
        NormalizedEncoder<Embedding, Qkv, Attend, Project, Mlp, Norm, Add, DEPTH>,
        Pool,
    >;

/// Source: Transformers `src/transformers/models/clip/modeling_clip.py`.
pub type CLIPTextModelWithProjection<
    Embedding,
    Qkv,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Pool,
    Projection,
    const DEPTH: usize,
> = CLIPTextModel<
    Embedding,
    Qkv,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    language::SequenceHead<Pool, Projection>,
    DEPTH,
>;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    /// The CLIP text encoder's config with a projection, its other fields at their defaults.
    fn document() -> Value {
        json!({"architectures": ["CLIPTextModelWithProjection"], "model_type": "clip_text_model"})
    }

    /// A text encoder of positive dimensions, a known activation, a positive epsilon and, where
    /// it projects, a positive projection parses; each other value is refused by its reason.
    #[test]
    fn refuses_what_the_text_encoder_does_not_run() {
        let parse = |edit: &[(&str, Value)]| {
            let mut document = document();
            for (key, value) in edit {
                document[*key] = value.clone();
            }
            <Config as Family>::parse(&document).map(drop)
        };
        parse(&[]).unwrap();
        use io::ErrorKind::{InvalidData, Unsupported};
        let invalid = "invalid text encoder dimensions or normalization";
        let mut cases = vec![
            (
                vec![("hidden_act", json!("tanh"))],
                Unsupported,
                "unimplemented feedforward activation".to_owned(),
            ),
            (
                vec![("num_attention_heads", json!(3))],
                InvalidData,
                invalid.to_owned(),
            ),
            (
                vec![("layer_norm_eps", json!(0.0))],
                InvalidData,
                invalid.to_owned(),
            ),
            (
                vec![("projection_dim", json!(0))],
                InvalidData,
                "projection_dim must be positive".to_owned(),
            ),
            (
                vec![("text_config", json!({"projection_dim": 0}))],
                InvalidData,
                "projection_dim: expected positive dimension".to_owned(),
            ),
        ];
        for name in [
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_hidden_layers",
            "vocab_size",
            "max_position_embeddings",
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
