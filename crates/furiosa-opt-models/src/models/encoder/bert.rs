//! Bidirectional token encoding with separately callable, rowwise prediction heads.

use crate::models::Family;
pub use crate::models::{Ids, Scalar};
use crate::modules::{attention, feedforward, language, transformer};
use crate::ops::attention::Attention;
use crate::ops::index::gather::Gather;
use crate::ops::{
    embedding::Lookup,
    linear::Affine,
    normalization::LayerNorm,
    pointwise::{
        activation::{Activation, Gelu, Tanh},
        add::Add,
        cast::Convert as Cast,
    },
};
use crate::{First, Module, Then};
use furiosa_opt_std::prelude::*;
use serde::{Deserialize, de::IgnoredAny};
use std::{collections::BTreeMap, io, marker::PhantomData, num::NonZeroUsize};

/// A pooler consumes rows selected by the caller; it does not select CLS implicitly.
pub enum Model<Body, Pooler, Classifier, TokenClassifier, MaskedHead> {
    /// Token features without a prediction head.
    Encoder(Body),
    /// Caller-selected rows transformed by the pooler.
    Pooling(language::Pooler<Body, Pooler>),
    /// Sequence-label logits for caller-selected rows.
    Classification(language::Pooler<Body, Classifier>),
    /// Label logits for every token row.
    TokenClassification(language::Readout<Body, TokenClassifier>),
    /// Vocabulary logits for every token row.
    Masked(language::Readout<Body, MaskedHead>),
}

/// Encoder and supported heads at one checkpoint geometry.
pub type Configured<
    Vocabulary,
    Positions,
    Segments,
    Width,
    Intermediate,
    Heads,
    Labels,
    const LAYERS: usize,
> = Model<
    Body<Vocabulary, Positions, Segments, Width, Intermediate, Heads, LAYERS>,
    Selected<Pooler<Width>>,
    Selected<Classifier<Width, Labels>>,
    TokenClassifier<Width, Labels>,
    MaskedHead<Width, Vocabulary>,
>;

/// Encoder specialized to checkpoint vocabulary, positions, segments and layer dimensions.
pub type Body<Vocabulary, Positions, Segments, Width, Intermediate, Heads, const LAYERS: usize> =
    Encoder<
        Embeddings<Vocabulary, Positions, Segments, Width>,
        [Layer<Width, Intermediate, Heads>; LAYERS],
    >;

/// Token, position and segment embedding followed by bidirectional encoder layers.
#[derive(DeviceSend)]
pub struct Encoder<Embedding, Layers> {
    /// Combined token, position and segment embedding with normalization.
    pub embedding: Embedding,
    /// Ordered bidirectional attention and feed-forward layers.
    pub layers: Layers,
}

/// The layers' attention reads the operands' bias in its context; the state holds the embedding's
/// slot, then one a layer.
impl<
    's,
    'x,
    Embedding,
    Layer,
    EmbeddingState,
    LayerState,
    Tokens,
    Width: AxisName,
    Count: AxisName,
    const LAYERS: usize,
> Module<&'x Operands<Tokens, Count>, (), &'s mut (EmbeddingState, [LayerState; LAYERS])>
    for Encoder<Embedding, [Layer; LAYERS]>
where
    Embedding: Module<
            (&'x Tokens, &'x Ids<Count>, &'x Ids<Count>),
            (),
            &'s mut EmbeddingState,
            Output = Rows<Width, Count>,
        >,
    Layer: for<'c> Module<
            Rows<Width, Count>,
            &'c attention::Context<&'static (), (), &'x Bias<Count>>,
            &'s mut LayerState,
            Output = Rows<Width, Count>,
        >,
{
    type Output = Rows<Width, Count>;

    fn forward(
        &self,
        device: &mut Device,
        env: (),
        (embedding, layers): &'s mut (EmbeddingState, [LayerState; LAYERS]),
        input: &'x Operands<Tokens, Count>,
    ) -> Self::Output {
        let hidden = self.embedding.forward(
            &mut *device,
            env,
            embedding,
            (&input.tokens, &input.positions, &input.segments),
        );
        let context = attention::Context {
            positions: &(),
            write: (),
            read: &input.bias,
        };
        self.layers.forward(device, &context, layers, hidden)
    }
}

pub type Embeddings<Vocabulary, Positions, Segments, Width> = language::Embeddings<
    Embedding<Vocabulary, Width>,
    Embedding<Positions, Width>,
    Embedding<Segments, Width>,
    Add,
    Norm<Width>,
>;

pub type Embedding<Vocabulary, Width> = Lookup<HbmTensor<bf16, m![1], m![Vocabulary, Width]>>;

pub type Norm<Width, D = bf16> = LayerNorm<HbmTensor<D, m![1], m![Width]>, Scalar>;

pub type Layer<Width, Intermediate, Heads> = transformer::PostNorm<
    attention::Full<
        attention::Positioned<attention::Projection<Linear<Width, Width>>, First>,
        Attention<PhantomData<Heads>, Scalar>,
        Linear<Width, Width>,
    >,
    feedforward::Mlp<Linear<Width, Intermediate>, Gelu, Linear<Intermediate, Width>>,
    Norm<Width>,
    Norm<Width>,
    Add,
>;

pub type Linear<In, Out, D = bf16> =
    Affine<HbmTensor<D, m![1], m![Out, In]>, [HbmTensor<D, m![1], m![Out]>; 1]>;

/// Caller-selected feature rows passed to a sequence head.
pub type Selected<Head> = language::SequenceHead<Gather, Head>;

/// F32 dense projection and tanh applied to selected rows.
pub type Pooler<Width> = Then<Cast<f32>, Then<Linear<Width, Width, f32>, Tanh>>;

/// Pooled rows projected to sequence-label logits.
pub type Classifier<Width, Labels> = Then<Pooler<Width>, Linear<Width, Labels, f32>>;

/// Every token row projected to F32 label logits.
pub type TokenClassifier<Width, Labels> = Then<Cast<f32>, Linear<Width, Labels, f32>>;

/// Dense GELU and normalization followed by vocabulary logits for each token.
pub type MaskedHead<Width, Vocabulary> = Then<
    Cast<f32>,
    language::MaskedHead<
        Linear<Width, Width, f32>,
        Gelu,
        Norm<Width, f32>,
        Linear<Width, Vocabulary, f32>,
    >,
>;

/// Row-major token features, with BF16 storage by default.
pub type Rows<Width, Count, D = bf16> = HbmTensor<D, m![1], m![Count, Width]>;

/// Additive attention mask indexed by query row and key row.
pub type Bias<Count> = HbmTensor<f32, m![1], m![Count, Count]>;

/// Token indices and additive visibility shared by the encoder layers.
#[derive(DeviceSend)]
pub struct Operands<Tokens, Count: AxisName> {
    /// Token lookup input, or features accepted by the selected embedding.
    pub tokens: Tokens,
    /// Absolute position indices per token.
    pub positions: Ids<Count>,
    /// Token-type indices per token.
    pub segments: Ids<Count>,
    /// Additive attention mask, indexed by query and key row.
    pub bias: Bias<Count>,
}

/// Prediction contract selected by the checkpoint's caller.
#[derive(Clone, Copy)]
pub enum Head {
    /// Hidden features, with a pooler when its checkpoint weights exist.
    Hidden,
    /// Masked-token vocabulary logits.
    Masked,
    /// Sequence-label logits from caller-selected rows.
    SequenceClassification,
    /// Per-token label logits.
    TokenClassification,
}

/// Parsed encoder geometry and prediction-head settings.
#[derive(Clone)]
pub struct Config {
    /// Requested prediction contract.
    pub head: Head,
    /// Hidden feature width.
    pub hidden: usize,
    /// Feed-forward expansion width.
    pub intermediate: usize,
    /// Number of token embeddings.
    pub vocabulary: usize,
    /// Number of absolute position embeddings.
    pub positions: usize,
    /// Number of token-type embeddings.
    pub segments: usize,
    /// Attention heads per layer.
    pub heads: usize,
    /// Encoder layer count.
    pub layers: usize,
    /// Number of classifier output labels.
    pub labels: usize,
    /// Checkpoint feed-forward activation.
    pub activation: Activation,
    /// Layer-normalization epsilon.
    pub epsilon: f64,
    /// Whether the masked head shares token embedding weights.
    pub tied: bool,
    /// The checkpoint's `model_type`, which names its position convention.
    pub model_type: Option<String>,
    /// The padding token's index, by which RoBERTa offsets its positions.
    pub padding: Option<i64>,
}

impl Family for Config {
    type Hf = Hf;
}

/// A bidirectional encoder's checkpoint configuration.
#[derive(Deserialize)]
pub struct Hf {
    architectures: Option<Vec<String>>,
    model_type: Option<String>,
    pad_token_id: Option<i64>,
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    type_vocab_size: Option<usize>,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_labels: Option<usize>,
    id2label: Option<BTreeMap<String, IgnoredAny>>,
    hidden_act: Option<String>,
    layer_norm_eps: Option<f64>,
    tie_word_embeddings: Option<bool>,
    position_embedding_type: Option<String>,
    is_decoder: Option<bool>,
}

impl TryFrom<Hf> for Config {
    type Error = io::Error;

    /// Narrows a bidirectional encoder checkpoint to the prediction head its architecture declares.
    fn try_from(hf: Hf) -> io::Result<Self> {
        // The first declared name that names a head decides it, since a framework's fallback names
        // may precede the native one.
        let head = (hf.architectures.iter().flatten())
            .find_map(|name| match name.as_str() {
                "BertForMaskedLM" => Some(Head::Masked),
                "BertForSequenceClassification" => Some(Head::SequenceClassification),
                "BertForTokenClassification" => Some(Head::TokenClassification),
                _ => None,
            })
            .unwrap_or(Head::Hidden);
        if hf.position_embedding_type.as_deref().unwrap_or("absolute") != "absolute"
            || hf.is_decoder.unwrap_or(false)
        {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "this encoder uses absolute positions and bidirectional attention",
            ));
        }
        let dimension = |name: &str, value: usize| {
            NonZeroUsize::new(value)
                .map(NonZeroUsize::get)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("{name}: expected positive dimension"),
                    )
                })
        };
        let labels = hf
            .num_labels
            .unwrap_or_else(|| hf.id2label.map_or(2, |labels| labels.len()));
        let config = Self {
            head,
            hidden: dimension("hidden_size", hf.hidden_size)?,
            intermediate: dimension("intermediate_size", hf.intermediate_size)?,
            vocabulary: dimension("vocab_size", hf.vocab_size)?,
            positions: dimension("max_position_embeddings", hf.max_position_embeddings)?,
            segments: dimension("type_vocab_size", hf.type_vocab_size.unwrap_or(2))?,
            heads: dimension("num_attention_heads", hf.num_attention_heads)?,
            layers: dimension("num_hidden_layers", hf.num_hidden_layers)?,
            labels: dimension("num_labels", labels)?,
            activation: match hf.hidden_act.as_deref().unwrap_or("gelu") {
                "gelu" => Activation::Gelu,
                "gelu_new" | "gelu_pytorch_tanh" => Activation::NewGelu,
                "relu" => Activation::Relu,
                "silu" | "swish" => Activation::Silu,
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::Unsupported,
                        "unimplemented encoder activation",
                    ));
                }
            },
            epsilon: hf.layer_norm_eps.unwrap_or(1e-12),
            tied: hf.tie_word_embeddings.unwrap_or(true),
            model_type: hf.model_type,
            padding: hf.pad_token_id,
        };
        if config.hidden % config.heads != 0 || !config.epsilon.is_finite() || config.epsilon <= 0.0
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid attention width or normalization epsilon",
            ));
        }
        Ok(config)
    }
}

pub(super) type BertBody<Embedding, Attention, Mlp, Norm, Add, const DEPTH: usize> =
    Encoder<Embedding, [transformer::PostNorm<Attention, Mlp, Norm, Norm, Add>; DEPTH]>;

/// Source: vLLM `vllm/model_executor/models/bert.py`.
pub type BertModel<Embedding, Qkv, Attend, Project, Mlp, Norm, Add, Pool, const DEPTH: usize> =
    language::Pooler<
        BertBody<Embedding, attention::Full<Qkv, Attend, Project>, Mlp, Norm, Add, DEPTH>,
        Pool,
    >;

/// Source: vLLM `vllm/model_executor/models/bert.py`.
pub type BertForMaskedLM<
    Embedding,
    Qkv,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Projection,
    Activation,
    HeadNorm,
    Vocabulary,
    const DEPTH: usize,
> = language::Readout<
    BertBody<Embedding, attention::Full<Qkv, Attend, Project>, Mlp, Norm, Add, DEPTH>,
    language::MaskedHead<Projection, Activation, HeadNorm, Vocabulary>,
>;

/// Source: vLLM `vllm/model_executor/models/bert.py`.
pub type BertForSequenceClassification<
    Embedding,
    Qkv,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Pool,
    Score,
    const DEPTH: usize,
> = BertModel<
    Embedding,
    Qkv,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    language::SequenceHead<Pool, Score>,
    DEPTH,
>;

/// Source: vLLM `vllm/model_executor/models/bert.py`.
pub type BertForTokenClassification<
    Embedding,
    Qkv,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Classify,
    const DEPTH: usize,
> = language::Readout<
    BertBody<Embedding, attention::Full<Qkv, Attend, Project>, Mlp, Norm, Add, DEPTH>,
    Classify,
>;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    /// A bidirectional encoder's config of every positive dimension.
    fn document() -> Value {
        json!({
            "architectures": ["BertModel"], "hidden_size": 8, "intermediate_size": 16,
            "vocab_size": 10, "max_position_embeddings": 4, "num_attention_heads": 2,
            "num_hidden_layers": 1,
        })
    }

    /// An absolute-position bidirectional encoder of positive dimensions, a known activation and a
    /// positive epsilon parses; each other value is refused by its kind and reason.
    #[test]
    fn refuses_what_the_encoder_does_not_run() {
        let parse = |edit: &[(&str, Value)]| {
            let mut document = document();
            for (key, value) in edit {
                document[*key] = value.clone();
            }
            <Config as Family>::parse(&document).map(drop)
        };
        parse(&[]).unwrap();
        use io::ErrorKind::{InvalidData, Unsupported};
        let positions = "this encoder uses absolute positions and bidirectional attention";
        let width = "invalid attention width or normalization epsilon";
        let positive = |name: &str| format!("{name}: expected positive dimension");
        let mut cases = vec![
            (
                vec![("position_embedding_type", json!("relative_key"))],
                Unsupported,
                positions.to_owned(),
            ),
            (
                vec![("is_decoder", json!(true))],
                Unsupported,
                positions.to_owned(),
            ),
            (
                vec![("hidden_act", json!("tanh"))],
                Unsupported,
                "unimplemented encoder activation".to_owned(),
            ),
            (
                vec![("num_attention_heads", json!(3))],
                InvalidData,
                width.to_owned(),
            ),
            (
                vec![("layer_norm_eps", json!(0.0))],
                InvalidData,
                width.to_owned(),
            ),
        ];
        for name in [
            "hidden_size",
            "intermediate_size",
            "vocab_size",
            "max_position_embeddings",
            "type_vocab_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_labels",
        ] {
            cases.push((vec![(name, json!(0))], InvalidData, positive(name)));
        }
        cases.push((
            vec![("id2label", json!({}))],
            InvalidData,
            positive("num_labels"),
        ));
        for (edit, kind, reason) in cases {
            let error = parse(&edit).unwrap_err();
            assert_eq!(
                (error.kind(), error.to_string()),
                (kind, reason),
                "{edit:?}"
            );
        }
        let infinite = r#"{"architectures": ["BertModel"], "hidden_size": 8, "intermediate_size": 16,
            "vocab_size": 10, "max_position_embeddings": 4, "num_attention_heads": 2,
            "num_hidden_layers": 1, "layer_norm_eps": 1e999}"#;
        let error = <Config as Family>::parse(&serde_json::from_str(infinite).unwrap()).map(drop);
        let error = error.unwrap_err();
        assert_eq!(
            (error.kind(), error.to_string()),
            (InvalidData, width.to_owned())
        );
    }
}
