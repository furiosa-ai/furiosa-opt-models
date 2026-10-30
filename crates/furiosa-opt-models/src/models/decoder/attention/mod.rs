//! Attention decoders: a token embedding, a stack of attention and feed-forward layers and a final
//! norm, whose rows a language-model head, a sequence classifier or no head reads.
//!
//! - A family's published sizes implement [`Geometry`], and its classes are aliases of the roots
//!   here. Fields carry Hugging Face's names, so a checkpoint loads into a class by name.
//! - [`Model`] runs a call: it prepares the call's attention context from the call's history once,
//!   then runs the body on the call's tokens.
//! - A causal model has no call of its own: its body runs, then its head reads the kept rows.

use std::io;
use std::marker::PhantomData;
use std::num::NonZeroUsize;

use serde::Deserialize;

use super::hybrid::{
    Glm5NextForCausalLM, GraniteMoeHybridForCausalLM, K3DSparkModel, OlmoHybridForCausalLM,
};
use crate::Sequential;
use crate::load::{self, Checkpoint, Load, LoadError};
use crate::models::config::{
    self,
    scaling::{Spectra, Spectrum},
};
use crate::models::decoder::{Form, form};
use crate::models::{Class, Family, Implements};
use crate::modules::attention::key_value;
use crate::modules::language::{self, Greedy};
use crate::modules::{
    attention, diffusion, feedforward,
    generic::collection::{Array, Collected},
    mixing, recurrent, residual, speculative, transformer,
};
use crate::ops::rotary::{self, PairFrequencies, SplitHalf};
use crate::tensor::memory::Memory;
use crate::{Either, First, Fork, Identity, Interleave, Layers, Repeat, Then, Unrolled};
use furiosa_opt_std::prelude::*;
use qwen3::Qwen3ForCausalLM;

pub mod deepseek_v2;
pub mod gemma4;
pub mod llama;
pub mod mistral;
pub mod mixtral;
pub mod olmoe;
pub mod qwen3;
pub mod qwen3_moe;

/// A published size of an attention decoder: its axes and the config values its type fixes.
pub trait Geometry {
    /// Features of a row: `hidden_size`.
    type Hidden: AxisName;

    /// Features of the feed-forward: `intermediate_size`.
    type Intermediate: AxisName;

    /// Rows of the embedding and of the head as compiled: `vocab_size` tokens at most.
    type Vocabulary: AxisName;

    /// Key and value heads: `num_key_value_heads`.
    type KvHeads: AxisName;

    /// Query heads each key and value head serves: `num_attention_heads / num_key_value_heads`.
    type Groups: AxisName;

    /// Features of a head: `head_dim`.
    type Head: AxisName;

    /// Features of every query head together: `num_attention_heads * head_dim`.
    type Queries: AxisName;

    /// `num_hidden_layers` layers of `L`.
    type Layers<L>;

    /// A norm of `W` features of memory `X` with the size's `rms_norm_eps`.
    type Norm<W: AxisName, X: Memory>;

    /// The hub spelling of the size, as `"0.6B"`.
    const SIZE: &'static str;

    /// The family whose classes the size is of, as `"qwen3"`.
    const FAMILY: &'static str;

    /// `num_hidden_layers`.
    const LAYERS: usize;

    /// `rms_norm_eps`, as the bits of the FP32 nearest to it.
    const EPSILON: usize;

    /// `tie_word_embeddings`: the head reads the embedding's table.
    const TIED_EMBEDDINGS: bool;
}

/// A causal language model of size `G` whose projections hold form `P`: the base model, then the
/// head that reads its rows.
#[derive(DeviceSend, Load)]
pub struct ForCausalLM<G: Geometry, Model, Head, P = form::Affine> {
    pub size: PhantomData<G>,
    /// The form of the body's projections, by which a line's scope admits a checkpoint.
    pub form: PhantomData<P>,
    pub model: Model,
    #[load(tied = "model.embed_tokens", config = "tie_word_embeddings")]
    pub lm_head: Head,
}

/// The base model of size `G` whose projections hold form `P`: the body, and the constants that
/// prepare each call's attention context.
#[derive(DeviceSend, Load)]
pub struct Model<G: Geometry, Body, Rotary, P = form::Affine> {
    pub size: PhantomData<G>,
    /// The form of the body's projections, by which a line's scope admits a checkpoint.
    pub form: PhantomData<P>,
    #[load(rename = "")]
    pub body: Body,
    pub rotary_emb: Rotary,
}

/// A sequence classifier of size `G` whose projections hold form `P`: the base model, then the
/// scores of its last row.
#[derive(DeviceSend, Load)]
pub struct ForSequenceClassification<G: Geometry, Model, Score, P = form::Affine> {
    pub size: PhantomData<G>,
    /// The form of the body's projections, by which a line's scope admits a checkpoint.
    pub form: PhantomData<P>,
    pub model: Model,
    pub score: Score,
}

/// The scope a decoder's base model is read at in a checkpoint of a class with a head: Hugging
/// Face's `base_model_prefix`.
const PREFIX: &str = "model";

/// The HF layer type whose rotary spectrum and attention window a paging load binds.
pub trait LayerType {
    const NAME: Option<&'static str>;
}

impl LayerType for () {
    const NAME: Option<&'static str> = None;
}

/// The rotary constants a checkpoint's config binds: its spectrum's pair frequencies and
/// amplitude, over the positions it serves, as no checkpoint stores them.
impl<
    'a,
    Heads: AxisName,
    Head: AxisName,
    Rows: AxisName,
    Windows: AxisName,
    D,
    X: Memory,
    Chips: M + 'static,
    Rotated: AxisName,
    Layer: LayerType,
> Load<&'a Checkpoint>
    for key_value::Paging<Heads, Head, Rows, Windows, D, X, Chips, Rotated, Layer>
where
    Self: for<'f> Load<(&'f PairFrequencies<Rotated>, usize)>,
{
    fn load(context: &mut load::Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        #[derive(Deserialize)]
        struct Trained {
            max_position_embeddings: NonZeroUsize,
        }
        context.checkpoint(from, |cx| {
            let (spectrum, trained) = match Layer::NAME {
                None => {
                    let spectrum: Spectrum = config::parse(from.config()).map_err(refusal)?;
                    let trained: Trained = config::parse(from.config()).map_err(refusal)?;
                    (spectrum, trained.max_position_embeddings)
                }
                Some(layer) => {
                    let (_, spectra) = cx.config_value(from, "rope_parameters")?;
                    let spectra = serde_json::json!({ "rope_parameters": spectra });
                    let spectrum = config::parse::<Spectra>(&spectra)
                        .map_err(refusal)?
                        .of(layer)
                        .map_err(refusal)?
                        .ok_or_else(|| LoadError::Unsupported {
                            name: "config.json:rope_parameters".into(),
                            reason: format!("layer type {layer:?} has no rotary spectrum"),
                        })?;
                    let key = match layer {
                        "sliding_attention" => "sliding_window",
                        _ => "max_position_embeddings",
                    };
                    let (_, trained) = cx.config_value(from, key)?;
                    (spectrum, config::parse(trained).map_err(refusal)?)
                }
            };
            let frequencies = spectrum.bind::<Head, Rotated>().map_err(refusal)?;
            let positions = spectrum.context(trained);
            Load::load(cx, (&frequencies, positions))
        })
    }
}

/// The rotation of every position a table holds under the spectrum a checkpoint's config binds,
/// as no checkpoint stores it.
impl<'a, Head: AxisName, Position: AxisName> Load<&'a Checkpoint>
    for SplitHalf<Head, rotary::group::Table<Position, rotary::Row<Head>>>
{
    fn load(context: &mut load::Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            let spectrum: Spectrum = config::parse(from.config()).map_err(refusal)?;
            let frequencies = spectrum.bind::<Head, Head>().map_err(refusal)?;
            Load::load(cx, &frequencies)
        })
    }
}

impl<G: Geometry, Model, Projection, Select, P>
    ForCausalLM<G, Model, Greedy<G::Hidden, G::Vocabulary, Projection, Select>, P>
{
    /// The head's vocabulary projection: the engine runs the base model, then this child, for the
    /// scores of every vocabulary row.
    pub fn scores(&self) -> &Projection {
        &self.lm_head.head
    }
}

/// A causal model reads a checkpoint at its root, if its form admits it.
impl<G: Geometry + Family, Model, Head, P: Form> Implements for ForCausalLM<G, Model, Head, P> {
    const SIZE: &'static str = G::SIZE;
    const CLASS: Class = Class {
        family: G::FAMILY,
        name: "ForCausalLM",
    };

    fn scope(source: &Checkpoint) -> Result<&'static str, LoadError> {
        check::<G>(source.config())?;
        P::REQUIREMENT.admits(source).map(|()| "")
    }
}

/// A base model reads its own layout at a checkpoint's root, and a causal or classifier
/// checkpoint's under `model`, not both, if its form admits the checkpoint.
impl<G: Geometry + Family, Body, Rotary, P: Form> Implements for Model<G, Body, Rotary, P> {
    const SIZE: &'static str = G::SIZE;
    const CLASS: Class = Class {
        family: G::FAMILY,
        name: "Model",
    };

    fn scope(source: &Checkpoint) -> Result<&'static str, LoadError> {
        check::<G>(source.config())?;
        P::REQUIREMENT.admits(source)?;
        let names = source.names();
        let holds = |name: &str| names.iter().any(|stored| stored == name);
        match (
            holds("embed_tokens.weight"),
            holds("model.embed_tokens.weight"),
        ) {
            (true, true) => Err(LoadError::InvalidData {
                name: "embed_tokens.weight".into(),
                reason: format!("the checkpoint holds a base model at its root and under {PREFIX}"),
            }),
            (false, true) => Ok(PREFIX),
            _ => Ok(""),
        }
    }
}

/// A sequence classifier reads a checkpoint at its root, if its form admits it.
impl<G: Geometry + Family, Model, Score, P: Form> Implements
    for ForSequenceClassification<G, Model, Score, P>
{
    const SIZE: &'static str = G::SIZE;
    const CLASS: Class = Class {
        family: G::FAMILY,
        name: "ForSequenceClassification",
    };

    fn scope(source: &Checkpoint) -> Result<&'static str, LoadError> {
        check::<G>(source.config())?;
        P::REQUIREMENT.admits(source).map(|()| "")
    }
}

axes![KvHeads = 8, Head = 128, Chunks = 80, Prompts = 1];

/// `N` windows of 512 cached keys: the history a row attends at most, for a model compiled for
/// another context than the shared [`Chunks`].
#[derive(Clone, Debug)]
pub struct Windows<const N: usize>;

impl<const N: usize> AxisName for Windows<N> {
    const NAME: Ident = Ident::new("Windows");
    const SIZE: usize = N;
}

axes![PaddedVocabulary = 152064];

/// Qwen3-8B: hidden 4096, q 4096, kv 1024, intermediate 12288.
pub mod extra_large {
    use crate::ops::{attention::geometry, swiglu};

    pub type Hidden = geometry::Hidden<4096>;
    pub type Intermediate = swiglu::Intermediate<12288>;
    pub type Groups = geometry::QueryGroups<4>;
    pub type QWidth = geometry::QueryWidth<4096>;
    pub type QkvWidth = geometry::ProjectionWidth<6144>;
}

/// Source: vLLM-Omni `vllm_omni/model_executor/models/fish_speech/fish_speech_slow_ar.py`.
pub type FishSpeechSlowARForConditionalGeneration<
    Embedding,
    Qkv,
    QueryNorm,
    KeyNorm,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Head,
    DepthEmbedding,
    DepthQkv,
    DepthQueryNorm,
    DepthKeyNorm,
    DepthPosition,
    DepthWrite,
    DepthRead,
    DepthAttend,
    DepthProject,
    DepthMlp,
    DepthNorm,
    DepthAdd,
    DepthHead,
    const TEMPORAL_DEPTH: usize,
    const DECODER_DEPTH: usize,
> = (
    Qwen3ForCausalLM<
        Embedding,
        Qkv,
        QueryNorm,
        KeyNorm,
        Position,
        Write,
        Read,
        Attend,
        Project,
        Mlp,
        Norm,
        Add,
        Head,
        TEMPORAL_DEPTH,
    >,
    Qwen3ForCausalLM<
        DepthEmbedding,
        DepthQkv,
        DepthQueryNorm,
        DepthKeyNorm,
        DepthPosition,
        DepthWrite,
        DepthRead,
        DepthAttend,
        DepthProject,
        DepthMlp,
        DepthNorm,
        DepthAdd,
        DepthHead,
        DECODER_DEPTH,
    >,
);

/// Source: vLLM-Omni `vllm_omni/model_executor/models/moss_tts/modeling_moss_tts_talker.py`.
pub type MossTTSDelayModel<
    Embedding,
    ConditionProjection,
    Add,
    Qkv,
    QueryNorm,
    KeyNorm,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Head,
    const DEPTH: usize,
> = Qwen3ForCausalLM<
    language::ConditionedEmbedding<Embedding, ConditionProjection, Add>,
    Qkv,
    QueryNorm,
    KeyNorm,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    speculative::Heads<Head>,
    DEPTH,
>;

/// Source: vLLM-Omni `vllm_omni/model_executor/models/moss_tts/modeling_moss_tts_talker.py`.
pub type MossTTSLocalModel<
    Embedding,
    ConditionProjection,
    Add,
    Qkv,
    QueryNorm,
    KeyNorm,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Head,
    DepthEmbedding,
    DepthQkv,
    DepthQueryNorm,
    DepthKeyNorm,
    DepthPosition,
    DepthWrite,
    DepthRead,
    DepthAttend,
    DepthProject,
    DepthMlp,
    DepthNorm,
    DepthAdd,
    DepthHead,
    const TEMPORAL_DEPTH: usize,
    const DECODER_DEPTH: usize,
> = Qwen3TTSTalkerForConditionalGeneration<
    Embedding,
    ConditionProjection,
    Add,
    Qkv,
    QueryNorm,
    KeyNorm,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    speculative::Heads<Head>,
    DepthEmbedding,
    DepthQkv,
    DepthQueryNorm,
    DepthKeyNorm,
    DepthPosition,
    DepthWrite,
    DepthRead,
    DepthAttend,
    DepthProject,
    DepthMlp,
    DepthNorm,
    DepthAdd,
    DepthHead,
    TEMPORAL_DEPTH,
    DECODER_DEPTH,
>;

/// Source: vLLM-Omni `vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_talker.py`.
pub type Qwen3OmniMoeTalkerForConditionalGeneration<
    Embedding,
    ConditionProjection,
    Add,
    Qkv,
    QueryNorm,
    KeyNorm,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Router,
    Experts,
    Weighting,
    Norm,
    Head,
    DepthEmbedding,
    DepthQkv,
    DepthQueryNorm,
    DepthKeyNorm,
    DepthPosition,
    DepthWrite,
    DepthRead,
    DepthAttend,
    DepthProject,
    DepthMlp,
    DepthNorm,
    DepthAdd,
    DepthHead,
    const TEMPORAL_DEPTH: usize,
    const DECODER_DEPTH: usize,
> = Qwen3TTSTalkerForConditionalGeneration<
    Embedding,
    ConditionProjection,
    Add,
    Qkv,
    QueryNorm,
    KeyNorm,
    Position,
    Write,
    Read,
    Attend,
    Project,
    feedforward::Routed<Router, Experts, Weighting>,
    Norm,
    Head,
    DepthEmbedding,
    DepthQkv,
    DepthQueryNorm,
    DepthKeyNorm,
    DepthPosition,
    DepthWrite,
    DepthRead,
    DepthAttend,
    DepthProject,
    DepthMlp,
    DepthNorm,
    DepthAdd,
    DepthHead,
    TEMPORAL_DEPTH,
    DECODER_DEPTH,
>;

/// Source: vLLM-Omni `vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py`.
pub type Qwen3TTSTalkerForConditionalGeneration<
    Embedding,
    ConditionProjection,
    Add,
    Qkv,
    QueryNorm,
    KeyNorm,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Head,
    DepthEmbedding,
    DepthQkv,
    DepthQueryNorm,
    DepthKeyNorm,
    DepthPosition,
    DepthWrite,
    DepthRead,
    DepthAttend,
    DepthProject,
    DepthMlp,
    DepthNorm,
    DepthAdd,
    DepthHead,
    const TEMPORAL_DEPTH: usize,
    const DECODER_DEPTH: usize,
> = (
    language::FeatureHead<
        language::Normalized<
            language::Backbone<
                language::ConditionedEmbedding<Embedding, ConditionProjection, Add>,
                Layers<
                    [transformer::PreNorm<
                        attention::Cached<
                            attention::NormalizedQkv<Qkv, QueryNorm, KeyNorm>,
                            Position,
                            Write,
                            Read,
                            Attend,
                            Project,
                        >,
                        Mlp,
                        Norm,
                        Add,
                    >; TEMPORAL_DEPTH],
                >,
            >,
            Norm,
        >,
        Head,
    >,
    Qwen3ForCausalLM<
        DepthEmbedding,
        DepthQkv,
        DepthQueryNorm,
        DepthKeyNorm,
        DepthPosition,
        DepthWrite,
        DepthRead,
        DepthAttend,
        DepthProject,
        DepthMlp,
        DepthNorm,
        DepthAdd,
        DepthHead,
        DECODER_DEPTH,
    >,
);

/// Source: vLLM-Omni `vllm_omni/model_executor/models/personaplex/personaplex_talker.py`.
pub type PersonaPlexTalkerForConditionalGeneration<
    TextEmbedding,
    AudioEmbedding,
    Sum,
    Qkv,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    TextHead,
    DepthEmbedding,
    DepthInput,
    DepthQkv,
    DepthPosition,
    DepthWrite,
    DepthRead,
    DepthAttend,
    DepthProject,
    DepthMlp,
    DepthNorm,
    DepthAdd,
    DepthHead,
    const DEPTH: usize,
    const DEPTH_LAYERS: usize,
> = (
    language::FeatureHead<
        language::Normalized<
            language::Backbone<
                Sequential<(TextEmbedding, AudioEmbedding), Sum>,
                Layers<
                    [transformer::PreNorm<
                        attention::Cached<Qkv, Position, Write, Read, Attend, Project>,
                        Mlp,
                        Norm,
                        Add,
                    >; DEPTH],
                >,
            >,
            Norm,
        >,
        TextHead,
    >,
    Depformer<
        DepthEmbedding,
        DepthInput,
        DepthQkv,
        DepthPosition,
        DepthWrite,
        DepthRead,
        DepthAttend,
        DepthProject,
        DepthMlp,
        DepthNorm,
        DepthAdd,
        DepthHead,
        DEPTH_LAYERS,
    >,
);

/// Source: vLLM-Omni `vllm_omni/model_executor/models/personaplex/personaplex_depformer.py`.
type Depformer<
    Embedding,
    Input,
    Qkv,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Head,
    const DEPTH: usize,
> = language::Readout<
    language::Backbone<
        language::ConditionedEmbedding<Embedding, Input, Add>,
        Layers<
            [transformer::PreNorm<
                attention::Cached<Qkv, Position, Write, Read, Attend, Project>,
                Mlp,
                Norm,
                Add,
            >; DEPTH],
        >,
    >,
    Head,
>;

/// Source: vLLM-Omni `vllm_omni/model_executor/models/breeze_tts_2/modeling_breeze.py`.
pub type BreezeForConditionalGeneration<
    TextEncoder,
    TextProjection,
    ReferenceEncoder,
    Embedding,
    Qkv,
    QueryNorm,
    KeyNorm,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Head,
    DepthEmbedding,
    DepthInput,
    DepthQkv,
    DepthPosition,
    DepthWrite,
    DepthRead,
    DepthAttend,
    DepthProject,
    DepthMlp,
    DepthNorm,
    DepthAdd,
    DepthHead,
    const DEPTH: usize,
    const DEPTH_LAYERS: usize,
> = (
    (Then<TextEncoder, TextProjection>, ReferenceEncoder),
    (
        language::FeatureHead<
            language::Normalized<
                language::Backbone<
                    Embedding,
                    Layers<
                        [transformer::PreNorm<
                            attention::Cached<
                                attention::NormalizedQkv<Qkv, QueryNorm, KeyNorm>,
                                Position,
                                Write,
                                Read,
                                Attend,
                                Project,
                            >,
                            Mlp,
                            Norm,
                            Add,
                        >; DEPTH],
                    >,
                >,
                Norm,
            >,
            Head,
        >,
        Qwen2ForCausalLM<
            Then<DepthEmbedding, DepthInput>,
            DepthQkv,
            DepthPosition,
            DepthWrite,
            DepthRead,
            DepthAttend,
            DepthProject,
            DepthMlp,
            DepthNorm,
            DepthAdd,
            DepthHead,
            DEPTH_LAYERS,
        >,
    ),
);

/// Source: vLLM-Omni `vllm_omni/model_executor/models/higgs_audio_v2/higgs_audio_v2_talker.py`.
/// A modality mask routes both norms and FF; attention sees the restored token order.
pub type HiggsAudioV2TalkerForConditionalGeneration<
    Embedding,
    Split,
    TextNorm,
    AudioNorm,
    Attention,
    TextMlp,
    AudioMlp,
    Merge,
    Add,
    FinalNorm,
    TextHead,
    AudioHead,
    const DEPTH: usize,
> = language::Decoder<
    Embedding,
    [transformer::RoutedNorm<
        transformer::Partition<Split, TextNorm, AudioNorm, Merge>,
        Attention,
        transformer::Partition<Split, TextNorm, AudioNorm, Merge>,
        transformer::Partition<Split, TextMlp, AudioMlp, Merge>,
        Add,
    >; DEPTH],
    FinalNorm,
    Fork<(TextHead, AudioHead)>,
>;

/// Source: vLLM-Omni `vllm_omni/model_executor/models/mammoth_moda2/mammoth_moda2.py`.
/// Shared norms/attention; the generation mask selects FF weights and vocabulary embeddings.
pub type MammothModa2Qwen2ForCausalLM<
    Embedding,
    Attention,
    Norm,
    Split,
    TextMlp,
    ImageMlp,
    MergeTokens,
    Add,
    TextHead,
    ImageHead,
    MergeLogits,
    const DEPTH: usize,
> = language::Decoder<
    Embedding,
    [transformer::RoutedNorm<
        Norm,
        Attention,
        Norm,
        transformer::Partition<Split, TextMlp, ImageMlp, MergeTokens>,
        Add,
    >; DEPTH],
    Norm,
    Sequential<Fork<(TextHead, ImageHead)>, MergeLogits>,
>;

/// Source: vLLM-Omni `vllm_omni/model_executor/models/mimo_audio/mimo_audio_llm.py`.
/// Input depth encoding, temporal prediction and output depth prediction have independent KV
/// histories.
pub type MiMoAudioLLMModel<InputDepth, Temporal, OutputDepth> =
    (InputDepth, (Temporal, OutputDepth));

/// Source: vLLM-Omni `vllm_omni/model_executor/models/ming_tts/ming_tts_llm.py`.
/// Temporal hidden drives continuous acoustic flow, stop prediction and speaker prediction.
pub type MingLLMModel<Temporal, AcousticVelocity, Stop, Speaker> =
    (Temporal, (AcousticVelocity, Fork<(Stop, Speaker)>));

/// Source: vLLM-Omni `vllm_omni/model_executor/models/voxcpm2/voxcpm2_talker.py`.
/// Temporal and residual decoders keep separate histories; a local flow predicts the next audio
/// latent.
pub type VoxCPM2TalkerForConditionalGeneration<
    Temporal,
    Residual,
    LocalVelocity,
    Stop,
    AudioDecoder,
> = (Temporal, (Residual, (LocalVelocity, (Stop, AudioDecoder))));

/// Source: vLLM-Omni `vllm_omni/model_executor/models/qwen2_5_omni/qwen2_5_omni_talker.py`.
pub type Qwen2_5OmniTalkerModel<
    Tokens,
    Condition,
    Attention,
    Mlp,
    Norm,
    Add,
    Head,
    const DEPTH: usize,
> = language::Decoder<
    language::ConditionedEmbedding<Tokens, Condition, Add>,
    [transformer::PreNorm<Attention, Mlp, Norm, Add>; DEPTH],
    Norm,
    Head,
>;

/// Source: vLLM-Omni `vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py`.
/// Reference codec and speaker encoding precede the talker's temporal/depth prediction stages.
pub type Qwen3TTSForConditionalGeneration<
    ReferenceEncoder,
    SpeakerEncoder,
    Embedding,
    ConditionProjection,
    Add,
    Qkv,
    QueryNorm,
    KeyNorm,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Head,
    DepthEmbedding,
    DepthQkv,
    DepthQueryNorm,
    DepthKeyNorm,
    DepthPosition,
    DepthWrite,
    DepthRead,
    DepthAttend,
    DepthProject,
    DepthMlp,
    DepthNorm,
    DepthAdd,
    DepthHead,
    const TEMPORAL_DEPTH: usize,
    const DECODER_DEPTH: usize,
> = (
    (ReferenceEncoder, SpeakerEncoder),
    Qwen3TTSTalkerForConditionalGeneration<
        Embedding,
        ConditionProjection,
        Add,
        Qkv,
        QueryNorm,
        KeyNorm,
        Position,
        Write,
        Read,
        Attend,
        Project,
        Mlp,
        Norm,
        Head,
        DepthEmbedding,
        DepthQkv,
        DepthQueryNorm,
        DepthKeyNorm,
        DepthPosition,
        DepthWrite,
        DepthRead,
        DepthAttend,
        DepthProject,
        DepthMlp,
        DepthNorm,
        DepthAdd,
        DepthHead,
        TEMPORAL_DEPTH,
        DECODER_DEPTH,
    >,
);

/// Source: vLLM-Omni `vllm_omni/model_executor/models/higgs_audio_v3/higgs_audio_v3_talker.py`.
/// V3 uses the Qwen3 layer body; text/audio output heads do not imply V2's DualFFN layers.
pub type HiggsAudioV3TalkerForConditionalGeneration<
    Embedding,
    Attention,
    Mlp,
    Norm,
    Add,
    TextHead,
    AudioHead,
    const DEPTH: usize,
> = language::Decoder<
    Embedding,
    [transformer::PreNorm<Attention, Mlp, Norm, Add>; DEPTH],
    Norm,
    Fork<(TextHead, AudioHead)>,
>;

/// Source: vLLM-Omni `vllm_omni/model_executor/models/gepard/gepard_talker.py`.
/// Codebook logits are emitted together; a second head predicts stop, without a depth decoder.
pub type GepardTalkerForConditionalGeneration<
    Embedding,
    Attention,
    Mlp,
    Norm,
    Add,
    Codes,
    Stop,
    const DEPTH: usize,
> = language::Decoder<
    Embedding,
    [transformer::PreNorm<Attention, Mlp, Norm, Add>; DEPTH],
    Norm,
    Fork<(Codes, Stop)>,
>;

/// Source: vLLM-Omni `vllm_omni/model_executor/models/audio8_tts/audio8_tts_slow_ar.py`.
pub type Audio8TTSSlowARForConditionalGeneration<
    Embedding,
    Attention,
    Mlp,
    Norm,
    Add,
    Head,
    FastAR,
    const DEPTH: usize,
> = (
    language::Decoder<
        Embedding,
        [transformer::PreNorm<Attention, Mlp, Norm, Add>; DEPTH],
        Norm,
        Head,
    >,
    FastAR,
);

/// Source: vLLM-Omni `vllm_omni/model_executor/models/indextts2/indextts2_talker.py`.
pub type IndexTTS2TalkerForConditionalGeneration<
    SpeakerEncoder,
    EmotionEncoder,
    Embedding,
    Attention,
    Mlp,
    Norm,
    Add,
    MelHead,
    const DEPTH: usize,
> = (
    (SpeakerEncoder, EmotionEncoder),
    language::Decoder<
        Embedding,
        [transformer::PreNorm<Attention, Mlp, Norm, Add>; DEPTH],
        Norm,
        MelHead,
    >,
);

/// Source: vLLM-Omni
/// `vllm_omni/model_executor/models/nemotron_voicechat/nemotron_voicechat_talker.py`.
/// The head returns acoustic distribution parameters; sampling from them is an engine operation.
pub type NemotronVoiceChatTalkerForConditionalGeneration<
    Embedding,
    Attention,
    Mlp,
    Norm,
    Add,
    Distribution,
    const DEPTH: usize,
> = language::Decoder<
    Embedding,
    [transformer::PreNorm<Attention, Mlp, Norm, Add>; DEPTH],
    Norm,
    Distribution,
>;

/// Source: vLLM-Omni `vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_omni_tts.py`.
/// Text/semantic/code embeddings feed a Llama decoder and one codec-logit head.
pub type MiniCPMO45OmniTTSForConditionalGeneration<
    Embedding,
    Attention,
    Mlp,
    Norm,
    Add,
    Codes,
    const DEPTH: usize,
> = language::Decoder<
    Embedding,
    [transformer::PreNorm<Attention, Mlp, Norm, Add>; DEPTH],
    Norm,
    Codes,
>;

/// Source: vLLM-Omni `vllm_omni/model_executor/models/moss_tts_nano/modeling_moss_tts_nano.py`.
/// Both children are checkpoint-configured remote models; generation control stays outside this
/// product.
pub type MossTTSNanoForCausalLM<Language, AudioCodec> = (Language, AudioCodec);

type BranchNormLayer<
    Qkv,
    QueryNorm,
    KeyNorm,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
> = transformer::BranchNorm<
    attention::Cached<
        attention::NormalizedQkv<Qkv, QueryNorm, KeyNorm>,
        Position,
        Write,
        Read,
        Attend,
        Project,
    >,
    Mlp,
    Norm,
    Add,
>;

type GatedSandwichLayer<
    Qkv,
    QkNorm,
    QueryScale,
    Position,
    Write,
    Read,
    Attend,
    Concat,
    Gate,
    Activation,
    Multiply,
    Project,
    Mlp,
    Norm,
    Add,
> = transformer::SandwichNorm<
    Then<
        residual::Sum<
            Fork<(
                attention::Cached<
                    attention::NormalizedQkv<Qkv, Then<QkNorm, QueryScale>, QkNorm>,
                    Position,
                    Write,
                    Read,
                    Attend,
                    Concat,
                >,
                Then<Gate, Activation>,
            )>,
            Multiply,
        >,
        Project,
    >,
    Mlp,
    Norm,
    Add,
>;

/// Source: vLLM `vllm/model_executor/models/afmoe.py`.
pub type AfmoeForCausalLM<
    Embedding,
    Qkv,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Router,
    Experts,
    Weighting,
    SharedMlp,
    Add,
    Norm,
    Head,
    const DEPTH: usize,
> = Qwen2ForCausalLM<
    Embedding,
    Qkv,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Either<Mlp, feedforward::Moe<feedforward::Routed<Router, Experts, Weighting>, SharedMlp, Add>>,
    Norm,
    Add,
    Head,
    DEPTH,
>;

/// Source: SGLang `python/sglang/srt/models/bailing_moe_nextn.py`.
pub type BailingMoeForCausalLMNextN<
    Embedding,
    Join,
    DraftProjection,
    Projection,
    Write,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Head,
    const DEPTH: usize,
> = speculative::Indexed<
    Eagle3DeepseekV2ForCausalLM<
        Embedding,
        Join,
        DraftProjection,
        Projection,
        Write,
        Attend,
        Project,
        Mlp,
        Norm,
        Add,
        Head,
        DEPTH,
    >,
>;

/// Source: vLLM `vllm/model_executor/models/qwen3_dflash2.py`.
pub type DFlash2DraftModel<
    Embedding,
    Join,
    DraftProjection,
    Qkv,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Head,
    const DEPTH: usize,
> = language::Features<
    speculative::Embedding<Embedding, Join, DraftProjection>,
    Layers<
        [transformer::PreNorm<
            attention::Attention<
                attention::Positioned<Qkv, Position>,
                attention::KeyValue<Write, Read>,
                Attend,
                Project,
            >,
            Mlp,
            Norm,
            Add,
        >; DEPTH],
    >,
    Head,
>;

/// Source: vLLM `vllm/models/deepseek_v4/__init__.py`.
pub type DSparkDraftModel<
    Embedding,
    Join,
    DraftProjection,
    Expand,
    Projection,
    Write,
    Select,
    Attend,
    Project,
    Mlp,
    Router,
    Experts,
    Weighting,
    SharedMlp,
    Add,
    Norm,
    Merge,
    Head,
    const DEPTH: usize,
> = language::Features<
    speculative::Embedding<Embedding, Join, DraftProjection>,
    mixing::HyperBody<
        Expand,
        Layers<
            [mixing::HyperConnection<
                transformer::PreNorm<
                    attention::Sparse<Projection, Write, Select, Attend, Project>,
                    Either<
                        Mlp,
                        feedforward::Moe<
                            feedforward::Routed<Router, Experts, Weighting>,
                            SharedMlp,
                            Add,
                        >,
                    >,
                    Norm,
                    Add,
                >,
                Merge,
                Expand,
            >; DEPTH],
        >,
    >,
    Head,
>;

/// Source: vLLM `vllm/model_executor/models/deepseek_mtp.py`.
pub type DeepSeekMTPModel<
    Embedding,
    Join,
    DraftProjection,
    Qkv,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Router,
    Experts,
    Weighting,
    SharedMlp,
    Add,
    Norm,
    Head,
    const DEPTH: usize,
> = EagleDeepSeekMTPModel<
    Embedding,
    Join,
    DraftProjection,
    Qkv,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Either<Mlp, feedforward::Moe<feedforward::Routed<Router, Experts, Weighting>, SharedMlp, Add>>,
    Norm,
    Add,
    Head,
    DEPTH,
>;

/// Source: vLLM `vllm/models/deepseek_v4/__init__.py`.
pub type DeepSeekV4MTPModel<
    Embedding,
    Join,
    DraftProjection,
    Expand,
    Projection,
    Write,
    Select,
    Attend,
    Project,
    Mlp,
    Router,
    Experts,
    Weighting,
    SharedMlp,
    Add,
    Norm,
    Merge,
    Head,
    const DEPTH: usize,
> = speculative::Indexed<
    DSparkDraftModel<
        Embedding,
        Join,
        DraftProjection,
        Expand,
        Projection,
        Write,
        Select,
        Attend,
        Project,
        Mlp,
        Router,
        Experts,
        Weighting,
        SharedMlp,
        Add,
        Norm,
        Merge,
        Head,
        DEPTH,
    >,
>;

/// Source: vLLM `vllm/model_executor/models/deepseek_v2.py`.
pub type DeepseekForCausalLM<
    Embedding,
    Projection,
    Write,
    Select,
    Attend,
    Project,
    Mlp,
    Router,
    Experts,
    Weighting,
    SharedMlp,
    Add,
    Norm,
    Head,
    const DEPTH: usize,
> = language::Decoder<
    Embedding,
    [transformer::PreNorm<
        attention::Sparse<Projection, Write, Select, Attend, Project>,
        Either<
            Mlp,
            feedforward::Moe<feedforward::Routed<Router, Experts, Weighting>, SharedMlp, Add>,
        >,
        Norm,
        Add,
    >; DEPTH],
    Norm,
    Head,
>;

/// Source: vLLM `vllm/models/deepseek_v32/__init__.py`.
pub type DeepseekV32MTPModel<
    Embedding,
    Join,
    DraftProjection,
    Projection,
    Write,
    Select,
    Attend,
    Project,
    Mlp,
    Router,
    Experts,
    Weighting,
    SharedMlp,
    Add,
    Norm,
    Head,
    const DEPTH: usize,
> = Dots3NoteForCausalLMNextN<
    Embedding,
    Join,
    DraftProjection,
    Projection,
    Write,
    Select,
    Attend,
    Project,
    Either<Mlp, feedforward::Moe<feedforward::Routed<Router, Experts, Weighting>, SharedMlp, Add>>,
    Norm,
    Add,
    Head,
    DEPTH,
>;

/// Source: vLLM `vllm/model_executor/models/deepseek_v2.py`.
pub type DeepseekV3BidirectionalModel<
    Embedding,
    Projection,
    Write,
    Select,
    Attend,
    Project,
    Mlp,
    Router,
    Experts,
    Weighting,
    SharedMlp,
    Add,
    Norm,
    Pool,
    const DEPTH: usize,
> = language::Pooler<
    language::Backbone<
        Embedding,
        Layers<
            [transformer::PreNorm<
                attention::Sparse<Projection, Write, Select, Attend, Project>,
                Either<
                    Mlp,
                    feedforward::Moe<
                        feedforward::Routed<Router, Experts, Weighting>,
                        SharedMlp,
                        Add,
                    >,
                >,
                Norm,
                Add,
            >; DEPTH],
        >,
    >,
    Pool,
>;

/// Source: vLLM `vllm/models/deepseek_v4/__init__.py`.
pub type DeepseekV4ForCausalLM<
    Embedding,
    Ngram,
    Add,
    Expand,
    Projection,
    Write,
    Select,
    Attend,
    Project,
    Mlp,
    Router,
    Experts,
    Weighting,
    SharedMlp,
    Norm,
    Merge,
    Head,
    const DEPTH: usize,
> = language::CausalLm<
    language::NgramEmbedding<Embedding, Ngram, Add>,
    mixing::HyperBody<
        Expand,
        Layers<
            [mixing::HyperConnection<
                transformer::PreNorm<
                    attention::Sparse<Projection, Write, Select, Attend, Project>,
                    Either<
                        Mlp,
                        feedforward::Moe<
                            feedforward::Routed<Router, Experts, Weighting>,
                            SharedMlp,
                            Add,
                        >,
                    >,
                    Norm,
                    Add,
                >,
                Merge,
                Expand,
            >; DEPTH],
        >,
    >,
    Norm,
    Head,
>;

/// Source: vLLM `vllm/model_executor/models/diffusion_gemma.py`.
pub type DiffusionGemmaForBlockDiffusion<
    Embedding,
    Norm,
    Mlp,
    Add,
    Qkv,
    Position,
    Write,
    Read,
    Join,
    Attend,
    Project,
    Head,
    const DEPTH: usize,
> = language::CausalLm<
    diffusion::ConditionedEmbedding<Embedding, diffusion::SelfConditioning<Norm, Mlp, Add, Norm>>,
    Layers<
        [transformer::PreNorm<
            attention::Attention<
                attention::Positioned<Qkv, Position>,
                attention::Appended<Write, Read, Join>,
                Attend,
                Project,
            >,
            Mlp,
            Norm,
            Add,
        >; DEPTH],
    >,
    Norm,
    Head,
>;

/// Source: SGLang `python/sglang/srt/models/dots3_nextn.py`.
pub type Dots3NoteForCausalLMNextN<
    Embedding,
    Join,
    DraftProjection,
    Projection,
    Write,
    Select,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Head,
    const DEPTH: usize,
> = speculative::Indexed<
    language::Features<
        speculative::Embedding<Embedding, Join, DraftProjection>,
        Layers<
            [transformer::PreNorm<
                attention::Sparse<Projection, Write, Select, Attend, Project>,
                Mlp,
                Norm,
                Add,
            >; DEPTH],
        >,
        Head,
    >,
>;

/// Source: vLLM `vllm/model_executor/models/deepseek_eagle3.py`.
pub type Eagle3DeepseekV2ForCausalLM<
    Embedding,
    Join,
    DraftProjection,
    Projection,
    Write,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Head,
    const DEPTH: usize,
> = language::Features<
    speculative::Embedding<Embedding, Join, DraftProjection>,
    Layers<
        [transformer::PreNorm<
            attention::Latent<Projection, Write, Attend, Project>,
            Mlp,
            Norm,
            Add,
        >; DEPTH],
    >,
    Head,
>;

/// Source: vLLM `vllm/model_executor/models/deepseek_eagle.py`.
pub type EagleDeepSeekMTPModel<
    Embedding,
    Join,
    DraftProjection,
    Qkv,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Head,
    const DEPTH: usize,
> = speculative::Indexed<
    DFlash2DraftModel<
        Embedding,
        Join,
        DraftProjection,
        Qkv,
        Position,
        Write,
        Read,
        Attend,
        Project,
        Mlp,
        Norm,
        Add,
        Head,
        DEPTH,
    >,
>;

/// Source: SGLang `python/sglang/srt/models/gemma3_causal.py`.
pub type EmbeddingGemmaModel<
    Embedding,
    Qkv,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Pool,
    const DEPTH: usize,
> = language::Pooler<
    language::Normalized<
        language::Backbone<
            Embedding,
            Layers<
                [transformer::SandwichNorm<
                    attention::Cached<Qkv, Position, Write, Read, Attend, Project>,
                    Mlp,
                    Norm,
                    Add,
                >; DEPTH],
            >,
        >,
        Norm,
    >,
    Pool,
>;

/// Source: vLLM `vllm/model_executor/models/exaone4.py`.
/// A windowed checkpoint rotates its sliding layers and leaves its full layers unrotated (NoPE);
/// one without a window rotates every layer.
pub type Exaone4ForCausalLM<
    Embedding,
    Qkv,
    QueryNorm,
    KeyNorm,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Head,
    const ROTARY_LAYERS: usize,
    const NOPE_LAYERS: usize,
    const ORDER: &'static [crate::Branch],
> = language::CausalLm<
    Embedding,
    Interleave<
        (
            [BranchNormLayer<
                Qkv,
                QueryNorm,
                KeyNorm,
                Position,
                Write,
                Read,
                Attend,
                Project,
                Mlp,
                Norm,
                Add,
            >; ROTARY_LAYERS],
            [BranchNormLayer<
                Qkv,
                QueryNorm,
                KeyNorm,
                First,
                Write,
                Read,
                Attend,
                Project,
                Mlp,
                Norm,
                Add,
            >; NOPE_LAYERS],
        ),
        ORDER,
    >,
    Norm,
    Head,
>;

/// Source: vLLM `vllm/model_executor/models/falcon.py`.
pub type FalconForCausalLM<
    Embedding,
    Qkv,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Head,
    const DEPTH: usize,
> = language::Decoder<
    Embedding,
    [transformer::ParallelResidual<
        attention::Cached<Qkv, Position, Write, Read, Attend, Project>,
        Mlp,
        Norm,
        Norm,
        Add,
    >; DEPTH],
    Norm,
    Head,
>;

/// Source: Transformers `src/transformers/models/flex_olmo/modeling_flex_olmo.py`.
pub type FlexOlmoForCausalLM<
    Embedding,
    Qkv,
    QueryNorm,
    KeyNorm,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Router,
    Experts,
    Weighting,
    Norm,
    Add,
    Head,
    const DEPTH: usize,
> = Olmo2ForCausalLM<
    Embedding,
    Qkv,
    QueryNorm,
    KeyNorm,
    Position,
    Write,
    Read,
    Attend,
    Project,
    feedforward::Routed<Router, Experts, Weighting>,
    Norm,
    Add,
    Head,
    DEPTH,
>;

/// Source: vLLM `vllm/model_executor/models/gpt2.py`.
pub type GPT2ForSequenceClassification<
    Embedding,
    Qkv,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Pool,
    Score,
    const DEPTH: usize,
> = LlamaEmbeddingModel<
    Embedding,
    Qkv,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    language::SequenceHead<Pool, Score>,
    DEPTH,
>;

/// Source: vLLM `vllm/model_executor/models/gemma2.py`.
pub type Gemma2ForCausalLM<
    Embedding,
    Qkv,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Head,
    const DEPTH: usize,
> = language::Decoder<
    Embedding,
    [transformer::SandwichNorm<
        attention::Cached<Qkv, Position, Write, Read, Attend, Project>,
        Mlp,
        Norm,
        Add,
    >; DEPTH],
    Norm,
    Head,
>;

/// Source: SGLang `python/sglang/srt/models/gemma2_reward.py`.
pub type Gemma2ForSequenceClassification<
    Embedding,
    Qkv,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Pool,
    Score,
    const DEPTH: usize,
> = EmbeddingGemmaModel<
    Embedding,
    Qkv,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    language::SequenceHead<Pool, Score>,
    DEPTH,
>;

/// Source: vLLM `vllm/model_executor/models/gemma3n.py`.
pub type Gemma3nForCausalLM<
    Embedding,
    Expand,
    Reduce,
    LayerEmbedding,
    Qkv,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Mix,
    Query,
    Head,
    const PRODUCER_LAYERS: usize,
    const READER_LAYERS: usize,
    const ORDER: &'static [crate::Branch],
> = language::Readout<
    language::Normalized<
        Sequential<
            Fork<(Embedding, Identity)>,
            mixing::StreamBody<
                Expand,
                Interleave<
                    (
                        [mixing::StreamLayer<
                            Reduce,
                            LayerEmbedding,
                            Qwen2Layer<Qkv, Position, Write, Read, Attend, Project, Mlp, Norm, Add>,
                            Mix,
                        >; PRODUCER_LAYERS],
                        [mixing::StreamLayer<
                            Reduce,
                            LayerEmbedding,
                            transformer::PreNorm<
                                attention::Shared<Query, Position, Read, Attend, Project>,
                                Mlp,
                                Norm,
                                Add,
                            >,
                            Mix,
                        >; READER_LAYERS],
                    ),
                    ORDER,
                >,
                Reduce,
            >,
        >,
        Norm,
    >,
    Head,
>;

/// Source: SGLang `python/sglang/srt/models/gemma4_mtp.py`.
pub type Gemma4AssistantForCausalLM<
    Embedding,
    Join,
    DraftProjection,
    Query,
    Position,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    PostProjection,
    Head,
    const DEPTH: usize,
> = language::Features<
    speculative::Embedding<Embedding, Join, DraftProjection>,
    Layers<
        [transformer::SandwichNorm<
            attention::Shared<Query, Position, Read, Attend, Project>,
            Mlp,
            Norm,
            Add,
        >; DEPTH],
    >,
    Fork<(PostProjection, Head)>,
>;

/// Source: vLLM `vllm/model_executor/models/hrm_text.py`.
pub type HrmTextForCausalLM<
    Embedding,
    Initial,
    Qkv,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Head,
    const LOW_DEPTH: usize,
    const HIGH_DEPTH: usize,
    const LOW_CYCLES: usize,
    const HIGH_CYCLES: usize,
> = language::CausalLm<
    Embedding,
    recurrent::Hierarchy<
        Initial,
        Layers<
            [transformer::PreNorm<attention::Full<Qkv, Attend, Project>, Mlp, Norm, Add>;
                LOW_DEPTH],
        >,
        Layers<
            [transformer::PreNorm<attention::Full<Qkv, Attend, Project>, Mlp, Norm, Add>;
                HIGH_DEPTH],
        >,
        Add,
        LOW_CYCLES,
        HIGH_CYCLES,
    >,
    Norm,
    Head,
>;

/// Source: vLLM `vllm/model_executor/models/iquest_loopcoder.py`.
pub type IQuestLoopCoderForCausalLM<
    Embedding,
    Qkv,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Head,
    const DEPTH: usize,
    const LOOPS: usize,
> = language::CausalLm<
    Embedding,
    Layers<
        Repeat<
            Layers<
                [Qwen2Layer<Qkv, Position, Write, Read, Attend, Project, Mlp, Norm, Add>; DEPTH],
            >,
            LOOPS,
        >,
        Unrolled,
    >,
    Norm,
    Head,
>;

/// Source: SGLang `python/sglang/srt/models/jet_nemotron.py`.
pub type JetNemotronForCausalLM<
    Embedding,
    Qkv,
    QueryNorm,
    KeyNorm,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Router,
    Experts,
    Weighting,
    SharedMlp,
    Add,
    Norm,
    InputProjection,
    Convolution,
    QkNorm,
    Recurrence,
    OutputGate,
    OutputProjection,
    Head,
    const ATTENTION_LAYERS: usize,
    const RECURRENT_LAYERS: usize,
    const ORDER: &'static [crate::Branch],
> = OlmoHybridForCausalLM<
    Embedding,
    attention::NormalizedQkv<Qkv, QueryNorm, KeyNorm>,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Either<Mlp, feedforward::Moe<feedforward::Routed<Router, Experts, Weighting>, SharedMlp, Add>>,
    Norm,
    Add,
    InputProjection,
    Convolution,
    QkNorm,
    Recurrence,
    OutputGate,
    OutputProjection,
    Head,
    ATTENTION_LAYERS,
    RECURRENT_LAYERS,
    ORDER,
>;

/// Source: vLLM `vllm/models/kimi_k3/__init__.py`.
pub type KimiK3MTPModel<
    Embedding,
    Join,
    DraftProjection,
    Expand,
    Projection,
    Write,
    Attend,
    Project,
    Mlp,
    Router,
    Experts,
    Weighting,
    SharedMlp,
    Add,
    Norm,
    Merge,
    InputProjection,
    Convolution,
    QkNorm,
    Recurrence,
    OutputGate,
    OutputProjection,
    Head,
    const FIRST_LAYERS: usize,
    const SECOND_LAYERS: usize,
    const ORDER: &'static [crate::Branch],
> = speculative::Indexed<
    K3DSparkModel<
        Embedding,
        Join,
        DraftProjection,
        Expand,
        Projection,
        Write,
        Attend,
        Project,
        Mlp,
        Router,
        Experts,
        Weighting,
        SharedMlp,
        Add,
        Norm,
        Merge,
        InputProjection,
        Convolution,
        QkNorm,
        Recurrence,
        OutputGate,
        OutputProjection,
        Head,
        FIRST_LAYERS,
        SECOND_LAYERS,
        ORDER,
    >,
>;

/// Source: SGLang `python/sglang/srt/models/llama_embedding.py`.
pub type LlamaEmbeddingModel<
    Embedding,
    Qkv,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Pool,
    const DEPTH: usize,
> = language::Pooler<
    language::Normalized<
        language::Backbone<
            Embedding,
            Layers<
                [Qwen2Layer<Qkv, Position, Write, Read, Attend, Project, Mlp, Norm, Add>; DEPTH],
            >,
        >,
        Norm,
    >,
    Pool,
>;

/// Source: vLLM `vllm/model_executor/models/longcat_flash_ngram.py`.
pub type LongcatFlashNgramForCausalLM<
    Embedding,
    Ngram,
    Add,
    Qkv,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Head,
    const DEPTH: usize,
> = Qwen2ForCausalLM<
    language::NgramEmbedding<Embedding, Ngram, Add>,
    Qkv,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Head,
    DEPTH,
>;

/// Source: vLLM `vllm/model_executor/models/medusa.py`.
pub type MedusaModel<Projection, Activation, Add, Head> =
    speculative::Heads<speculative::Prediction<Projection, Activation, Add, Head>>;

/// Rotary sliding layers interleaved with unrotated (NoPE) full layers.
pub(crate) type MuseGlimmerBody<
    Qkv,
    QkNorm,
    QueryScale,
    Position,
    Write,
    Read,
    Attend,
    Concat,
    Gate,
    Activation,
    Multiply,
    Project,
    Mlp,
    Norm,
    Add,
    const ROTARY_LAYERS: usize,
    const NOPE_LAYERS: usize,
    const ORDER: &'static [crate::Branch],
> = Interleave<
    (
        [GatedSandwichLayer<
            Qkv,
            QkNorm,
            QueryScale,
            Position,
            Write,
            Read,
            Attend,
            Concat,
            Gate,
            Activation,
            Multiply,
            Project,
            Mlp,
            Norm,
            Add,
        >; ROTARY_LAYERS],
        [GatedSandwichLayer<
            Qkv,
            QkNorm,
            QueryScale,
            First,
            Write,
            Read,
            Attend,
            Concat,
            Gate,
            Activation,
            Multiply,
            Project,
            Mlp,
            Norm,
            Add,
        >; NOPE_LAYERS],
    ),
    ORDER,
>;

/// Source: vLLM `vllm/model_executor/models/muse_glimmer.py`.
/// Token embeddings pass their own norm, and the final norm's gain form differs from the layers'.
pub type MuseGlimmerForCausalLM<
    Embedding,
    EmbeddingNorm,
    Qkv,
    QkNorm,
    QueryScale,
    Position,
    Write,
    Read,
    Attend,
    Concat,
    Gate,
    Activation,
    Multiply,
    Project,
    Mlp,
    Norm,
    Add,
    FinalNorm,
    Head,
    const ROTARY_LAYERS: usize,
    const NOPE_LAYERS: usize,
    const ORDER: &'static [crate::Branch],
> = language::CausalLm<
    Then<Embedding, EmbeddingNorm>,
    MuseGlimmerBody<
        Qkv,
        QkNorm,
        QueryScale,
        Position,
        Write,
        Read,
        Attend,
        Concat,
        Gate,
        Activation,
        Multiply,
        Project,
        Mlp,
        Norm,
        Add,
        ROTARY_LAYERS,
        NOPE_LAYERS,
        ORDER,
    >,
    FinalNorm,
    Head,
>;

/// Source: SGLang `python/sglang/srt/models/nanbeige.py`.
pub type NanbeigeForCausalLM<
    Embedding,
    Qkv,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    LoopNorm,
    Head,
    const DEPTH: usize,
    const ITERATIONS: usize,
> = language::CausalLm<
    Embedding,
    Layers<
        language::Loop<
            Layers<
                [Qwen2Layer<Qkv, Position, Write, Read, Attend, Project, Mlp, Norm, Add>; DEPTH],
            >,
            [LoopNorm; ITERATIONS],
        >,
        Unrolled,
    >,
    Norm,
    Head,
>;

/// Source: SGLang `python/sglang/srt/models/nanbeige.py`.
pub type NanbeigeForSequenceClassification<
    Embedding,
    Qkv,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    LoopNorm,
    Pool,
    Score,
    const DEPTH: usize,
    const ITERATIONS: usize,
> = language::Pooler<
    language::Normalized<
        language::Backbone<
            Embedding,
            Layers<
                language::Loop<
                    Layers<
                        [Qwen2Layer<Qkv, Position, Write, Read, Attend, Project, Mlp, Norm, Add>;
                            DEPTH],
                    >,
                    [LoopNorm; ITERATIONS],
                >,
                Unrolled,
            >,
        >,
        Norm,
    >,
    language::SequenceHead<Pool, Score>,
>;

/// Source: vLLM `vllm/model_executor/models/nemotron_h.py`.
pub type NemotronHForCausalLM<
    Embedding,
    Norm,
    Qkv,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Add,
    InputProjection,
    Convolution,
    Selection,
    Time,
    Recurrence,
    OutputGate,
    OutputProjection,
    Mlp,
    Router,
    Experts,
    Weighting,
    SharedMlp,
    Head,
    const ATTENTION_LAYERS: usize,
    const RECURRENT_LAYERS: usize,
    const FEEDFORWARD_LAYERS: usize,
    const ORDER: &'static [crate::Branch],
> = GraniteMoeHybridForCausalLM<
    Embedding,
    Norm,
    Qkv,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Add,
    InputProjection,
    Convolution,
    Selection,
    Time,
    Recurrence,
    OutputGate,
    OutputProjection,
    Either<Mlp, feedforward::Moe<feedforward::Routed<Router, Experts, Weighting>, SharedMlp, Add>>,
    Head,
    ATTENTION_LAYERS,
    RECURRENT_LAYERS,
    FEEDFORWARD_LAYERS,
    ORDER,
>;

/// Source: SGLang `python/sglang/srt/models/olmo2.py`.
pub type Olmo2ForCausalLM<
    Embedding,
    Qkv,
    QueryNorm,
    KeyNorm,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Head,
    const DEPTH: usize,
> = language::Decoder<
    Embedding,
    [transformer::BranchNorm<
        attention::Cached<
            attention::NormalizedQkv<Qkv, QueryNorm, KeyNorm>,
            Position,
            Write,
            Read,
            Attend,
            Project,
        >,
        Mlp,
        Norm,
        Add,
    >; DEPTH],
    Norm,
    Head,
>;

/// Source: vLLM `vllm/model_executor/models/openai_privacy_filter.py`.
pub type OpenAIPrivacyFilterForTokenClassification<
    Embedding,
    Qkv,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Classify,
    const DEPTH: usize,
> = language::Encoder<
    Embedding,
    [Qwen2Layer<Qkv, Position, Write, Read, Attend, Project, Mlp, Norm, Add>; DEPTH],
    Classify,
>;

/// One Qwen2 decoder layer.
pub type Qwen2Layer<Qkv, Position, Write, Read, Attend, Project, Mlp, Norm, Add> =
    transformer::PreNorm<
        attention::Cached<Qkv, Position, Write, Read, Attend, Project>,
        Mlp,
        Norm,
        Add,
    >;

/// Source: vLLM `vllm/model_executor/models/qwen2.py`.
pub type Qwen2ForCausalLM<
    Embedding,
    Qkv,
    Position,
    Write,
    Read,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Head,
    const DEPTH: usize,
    Kind = Array,
> = language::Decoder<
    Embedding,
    Collected<Kind, Qwen2Layer<Qkv, Position, Write, Read, Attend, Project, Mlp, Norm, Add>, DEPTH>,
    Norm,
    Head,
>;

/// Source: vLLM `vllm/models/qwen4_exp/__init__.py`.
pub type Qwen4ExpForCausalLM<
    Embedding,
    Ngram,
    Add,
    Expand,
    Projection,
    Write,
    Select,
    Attend,
    Project,
    Mlp,
    Router,
    Experts,
    Weighting,
    SharedMlp,
    Norm,
    Merge,
    InputProjection,
    Convolution,
    QkNorm,
    Recurrence,
    OutputGate,
    OutputProjection,
    Head,
    const FIRST_LAYERS: usize,
    const SECOND_LAYERS: usize,
    const ORDER: &'static [crate::Branch],
> = Glm5NextForCausalLM<
    language::NgramEmbedding<Embedding, Ngram, Add>,
    Expand,
    Projection,
    Write,
    Select,
    Attend,
    Project,
    Mlp,
    Router,
    Experts,
    Weighting,
    SharedMlp,
    Add,
    Norm,
    Merge,
    InputProjection,
    Convolution,
    QkNorm,
    Recurrence,
    OutputGate,
    OutputProjection,
    Head,
    FIRST_LAYERS,
    SECOND_LAYERS,
    ORDER,
>;

/// Refuses a checkpoint config that the size `G` does not serve, as its family parses the config.
fn check<G: Family>(config: &serde_json::Value) -> Result<(), LoadError> {
    G::parse(config).map(drop).map_err(refusal)
}

/// A config parse's refusal as a load's: unsupported, or invalid data.
fn refusal(error: io::Error) -> LoadError {
    let (name, reason) = ("config.json".into(), error.to_string());
    match error.kind() {
        io::ErrorKind::Unsupported => LoadError::Unsupported { name, reason },
        _ => LoadError::InvalidData { name, reason },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Branch;
    use crate::Module;
    use crate::models::vision_language::MuseGlimmerForConditionalGeneration;
    use crate::modules::attention::{Attention, Context, KeyValue, NormalizedQkv, Positioned};
    use crate::modules::language::{Backbone, Normalized};
    use crate::modules::multimodal::InputEmbedding;
    use crate::modules::residual::Sum;
    use crate::tests::DEVICE;
    use furiosa_opt_std::prelude::{Device, Topology};

    /// Modulus of the toy arithmetic, so products stay exact.
    const P: i64 = 1_000_003;

    /// One rotary layer, then one NoPE layer.
    const ORDER: &[Branch] = &[Branch::First, Branch::Second];

    const TOKEN: i64 = 2;

    const POSITION: i64 = 3;

    /// `x -> scale * x + shift`, the form of every unary leaf.
    struct Affine(i64, i64);

    impl<Env> Module<&i64, Env, &mut ()> for Affine {
        type Output = i64;

        fn forward(&self, _: &mut Device, _: Env, _: &mut (), x: &i64) -> i64 {
            (self.0 * x + self.1).rem_euclid(P)
        }
    }

    struct Qkv;

    impl<Env> Module<&i64, Env, &mut ()> for Qkv {
        type Output = (i64, i64, i64);

        fn forward(&self, _: &mut Device, _: Env, _: &mut (), x: &i64) -> (i64, i64, i64) {
            (*x, 2 * x % P, 3 * x % P)
        }
    }

    struct Rotary;

    impl<Env> Module<((i64, i64), &i64), Env, &mut ()> for Rotary {
        type Output = (i64, i64);

        fn forward(
            &self,
            _: &mut Device,
            _: Env,
            _: &mut (),
            ((q, k), p): ((i64, i64), &i64),
        ) -> (i64, i64) {
            ((q + 10 * p) % P, (k + 20 * p) % P)
        }
    }

    /// Keeps this call's key and value as the whole history.
    struct Write;

    impl<Env> Module<(&mut (i64, i64), &(), (&i64, &i64)), Env, &mut ()> for Write {
        type Output = ();

        fn forward(
            &self,
            _: &mut Device,
            _: Env,
            _: &mut (),
            (cache, _, (key, value)): (&mut (i64, i64), &(), (&i64, &i64)),
        ) {
            *cache = (*key, *value);
        }
    }

    struct Attend;

    impl<Env> Module<(&i64, &(i64, i64), &()), Env, &mut ()> for Attend {
        type Output = i64;

        fn forward(
            &self,
            _: &mut Device,
            _: Env,
            _: &mut (),
            (q, (k, v), _): (&i64, &(i64, i64), &()),
        ) -> i64 {
            (q * k + v) % P
        }
    }

    struct Add;

    impl<Env> Module<(&i64, &i64), Env, &mut ()> for Add {
        type Output = i64;

        fn forward(&self, _: &mut Device, _: Env, _: &mut (), (x, y): (&i64, &i64)) -> i64 {
            (x + y) % P
        }
    }

    struct Multiply;

    impl<Env> Module<(&i64, &i64), Env, &mut ()> for Multiply {
        type Output = i64;

        fn forward(&self, _: &mut Device, _: Env, _: &mut (), (x, y): (&i64, &i64)) -> i64 {
            x * y % P
        }
    }

    /// Adds the selected multiple of a media feature to a token row.
    struct Merge;

    impl<Env> Module<(&i64, &i64, &i64), Env, &mut ()> for Merge {
        type Output = i64;

        fn forward(
            &self,
            _: &mut Device,
            _: Env,
            _: &mut (),
            (x, feature, selection): (&i64, &i64, &i64),
        ) -> i64 {
            (x + selection * feature) % P
        }
    }

    /// The attention context every layer reads: the call's position, no write or read operand.
    const CONTEXT: Context<i64, (), ()> = Context {
        positions: POSITION,
        write: (),
        read: (),
    };

    /// Both layer collections read the same attention context.
    const ENV: (&Context<i64, (), ()>, &Context<i64, (), ()>) = (&CONTEXT, &CONTEXT);

    /// Where a layer applies its norm around each branch.
    #[derive(Clone, Copy)]
    enum Placement {
        /// `x + norm(branch(x))`.
        Output,
        /// `norm(x + branch(x))`.
        Sum,
        /// `x + branch(norm(x))`.
        Input,
    }

    /// One way the toy gated sandwich stack can be computed.
    #[derive(Clone, Copy)]
    struct Gating {
        /// The hidden row that enters the layers.
        embedded: i64,
        rotary: [bool; 2],
        /// Whether the gate multiplies the heads before the output projection, not after it.
        before_projection: bool,
        /// Whether each branch output passes its output norm.
        output_norms: bool,
    }

    const UPSTREAM: Gating = Gating {
        embedded: 2 * (TOKEN + 10),
        rotary: [true, false],
        before_projection: true,
        output_norms: true,
    };

    /// The state of a toy readout over one rotary and one NoPE layer: units beside each layer's
    /// view of its cache in `caches`, `layer` building a layer's slot around the view. An
    /// interleave's state does not name its slot counts, so they are written here.
    fn state<'c, EmbeddingState: Default, LayerState, F: Fn(&'c mut (i64, i64)) -> LayerState>(
        caches: &'c mut [(i64, i64); 2],
        layer: F,
    ) -> (
        ((EmbeddingState, ([LayerState; 1], [LayerState; 1])), ()),
        (),
    ) {
        let [first, second] = caches.each_mut();
        (
            ((Default::default(), ([layer(first)], [layer(second)])), ()),
            (),
        )
    }

    fn rotate(rotary: bool, (q, k): (i64, i64)) -> (i64, i64) {
        match rotary {
            true => ((q + 10 * POSITION) % P, (k + 20 * POSITION) % P),
            false => (q, k),
        }
    }

    /// The toy branch-normalized stack computed in upstream order.
    fn branch_reference(placement: Placement, rotary: [bool; 2]) -> i64 {
        let m = |x: i64| x.rem_euclid(P);
        let attention = |x: i64, rotary| {
            let (q, k) = rotate(rotary, (m(x + 100), m(2 * x + 200)));
            m(q * k + 3 * x + 7)
        };
        let (attention_norm, mlp, mlp_norm) =
            (|x| m(2 * x + 1), |x| m(3 * x + 1), |x| m(5 * x + 2));
        let layer = |x, rotary| match placement {
            Placement::Output => {
                let x = m(x + attention_norm(attention(x, rotary)));
                m(x + mlp_norm(mlp(x)))
            }
            Placement::Sum => {
                let x = attention_norm(m(x + attention(x, rotary)));
                mlp_norm(m(x + mlp(x)))
            }
            Placement::Input => {
                let x = m(x + attention(attention_norm(x), rotary));
                m(x + mlp(mlp_norm(x)))
            }
        };
        m(7 * rotary.into_iter().fold(m(TOKEN + 10), layer) + 5)
    }

    fn branch_layer<Position>(
        positions: Position,
    ) -> BranchNormLayer<
        Qkv,
        Affine,
        Affine,
        Position,
        Write,
        (),
        Attend,
        Affine,
        Affine,
        Affine,
        Add,
    > {
        transformer::BranchNorm::new(
            Attention {
                qkv: Positioned {
                    projection: NormalizedQkv {
                        projection: Qkv,
                        query_norm: Affine(1, 100),
                        key_norm: Affine(1, 200),
                    },
                    positions,
                },
                cache: KeyValue {
                    write: Write,
                    read: (),
                },
                attend: Attend,
                output: Affine(1, 7),
            },
            Affine(3, 1),
            [Affine(2, 1), Affine(5, 2)],
            [Add, Add],
        )
    }

    /// Layers add each normalized branch output to the residual, and NoPE layers keep queries and
    /// keys unrotated.
    #[tokio::test(flavor = "current_thread")]
    async fn adds_normalized_branch_outputs() {
        let model: Exaone4ForCausalLM<
            Affine,
            Qkv,
            Affine,
            Affine,
            Rotary,
            Write,
            (),
            Attend,
            Affine,
            Affine,
            Affine,
            Add,
            Affine,
            1,
            1,
            ORDER,
        > = Then(
            Normalized::new(
                Backbone {
                    embedding: Affine(1, 10),
                    body: Interleave {
                        layers: ([branch_layer(Rotary)], [branch_layer(First)]),
                    },
                },
                Affine(7, 5),
            ),
            Affine(1, 0),
        );
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let mut caches = Default::default();
        let layer = |cache| (((Default::default(), cache, ()), ()), Default::default());
        let logit = model.forward(&mut device, ENV, &mut state(&mut caches, layer), &TOKEN);
        assert_eq!(logit, branch_reference(Placement::Output, [true, false]));
        for other in [
            branch_reference(Placement::Output, [true, true]),
            branch_reference(Placement::Sum, [true, false]),
            branch_reference(Placement::Input, [true, false]),
        ] {
            assert_ne!(logit, other);
        }
    }

    /// The toy gated sandwich stack computed as `gating` says.
    fn gated_reference(gating: Gating) -> i64 {
        let m = |x: i64| x.rem_euclid(P);
        let output = |norm: fn(i64) -> i64, x| match gating.output_norms {
            true => norm(x),
            false => x,
        };
        let attention = |y: i64, rotary| {
            let (q, k) = rotate(rotary, (m(4 * (y + 100)), m(2 * y + 100)));
            let (heads, gate) = (m(q * k + 3 * y), m(2 * (y + 5)));
            match gating.before_projection {
                true => m(heads * gate + 7),
                false => m(m(heads + 7) * gate),
            }
        };
        let layer = |x, rotary| {
            let branch = attention(m(3 * x + 1), rotary);
            let x = m(x + output(|x| (5 * x + 2).rem_euclid(P), branch));
            let branch = m(7 * m(11 * x + 4) + 3);
            m(x + output(|x| (13 * x + 6).rem_euclid(P), branch))
        };
        let x = gating.rotary.into_iter().fold(gating.embedded, layer);
        m(x - 3)
    }

    fn gated_layer<Position>(
        positions: Position,
    ) -> GatedSandwichLayer<
        Qkv,
        Affine,
        Affine,
        Position,
        Write,
        (),
        Attend,
        Affine,
        Affine,
        Affine,
        Multiply,
        Affine,
        Affine,
        Affine,
        Add,
    > {
        transformer::SandwichNorm::new(
            Then(
                Sum(
                    Fork((
                        Attention {
                            qkv: Positioned {
                                projection: NormalizedQkv {
                                    projection: Qkv,
                                    query_norm: Then(Affine(1, 100), Affine(4, 0)),
                                    key_norm: Affine(1, 100),
                                },
                                positions,
                            },
                            cache: KeyValue {
                                write: Write,
                                read: (),
                            },
                            attend: Attend,
                            output: Affine(1, 0),
                        },
                        Then(Affine(1, 5), Affine(2, 0)),
                    )),
                    Multiply,
                ),
                Affine(1, 7),
            ),
            Affine(7, 3),
            [[Affine(3, 1), Affine(5, 2)], [Affine(11, 4), Affine(13, 6)]],
            [Add, Add],
        )
    }

    /// Layers normalize each branch's input and output, the attention input gates the attention
    /// heads before the output projection, and NoPE layers keep queries and keys unrotated.
    #[tokio::test(flavor = "current_thread")]
    async fn gates_heads_before_projection() {
        let model: MuseGlimmerForCausalLM<
            Affine,
            Affine,
            Qkv,
            Affine,
            Affine,
            Rotary,
            Write,
            (),
            Attend,
            Affine,
            Affine,
            Affine,
            Multiply,
            Affine,
            Affine,
            Affine,
            Add,
            Affine,
            Affine,
            1,
            1,
            ORDER,
        > = Then(
            Normalized::new(
                Backbone {
                    embedding: Then(Affine(1, 10), Affine(2, 0)),
                    body: Interleave {
                        layers: ([gated_layer(Rotary)], [gated_layer(First)]),
                    },
                },
                Affine(1, -3),
            ),
            Affine(1, 0),
        );
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let mut caches = Default::default();
        let layer = |cache| {
            let attention = (Default::default(), cache, ());
            (
                ((((attention, Default::default()), ()), ()), ()),
                Default::default(),
            )
        };
        let logit = model.forward(&mut device, ENV, &mut state(&mut caches, layer), &TOKEN);
        assert_eq!(logit, gated_reference(UPSTREAM));
        for other in [
            Gating {
                rotary: [true, true],
                ..UPSTREAM
            },
            Gating {
                before_projection: false,
                ..UPSTREAM
            },
            Gating {
                output_norms: false,
                ..UPSTREAM
            },
        ] {
            assert_ne!(logit, gated_reference(other));
        }
    }

    /// Media features merge into the normalized token embeddings, and the merged rows pass the
    /// same layers as text.
    #[tokio::test(flavor = "current_thread")]
    async fn merges_media_into_normalized_tokens() {
        let model: MuseGlimmerForConditionalGeneration<
            Affine,
            Affine,
            (),
            (),
            (),
            Merge,
            Qkv,
            Affine,
            Affine,
            Rotary,
            Write,
            (),
            Attend,
            Affine,
            Affine,
            Affine,
            Multiply,
            Affine,
            Affine,
            Affine,
            Add,
            Affine,
            Affine,
            1,
            1,
            ORDER,
        > = Then(
            Normalized::new(
                Backbone {
                    embedding: InputEmbedding {
                        tokens: Then(Affine(1, 10), Affine(2, 0)),
                        encoder: ((), ()),
                        projector: (),
                        merge: Merge,
                    },
                    body: Interleave {
                        layers: ([gated_layer(Rotary)], [gated_layer(First)]),
                    },
                },
                Affine(1, -3),
            ),
            Affine(1, 0),
        );
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let media = [(9, 1)];
        let mut caches = Default::default();
        let layer = |cache| {
            let attention = (Default::default(), cache, ());
            (
                ((((attention, Default::default()), ()), ()), ()),
                Default::default(),
            )
        };
        // Encoded media skip the image encoder and projector, so their slots are units.
        let mut state = state::<(((), ()), (), (), ()), _, _>(&mut caches, layer);
        let logit = model.forward(&mut device, ENV, &mut state, (&TOKEN, &media));
        let merged = |embedded| {
            gated_reference(Gating {
                embedded,
                ..UPSTREAM
            })
        };
        assert_eq!(logit, merged(2 * (TOKEN + 10) + 9));
        assert_ne!(logit, merged(2 * (TOKEN + 10 + 9)));
    }

    /// A base model is read at the root of a checkpoint of its own layout and under `model` of a
    /// causal one; a checkpoint holding both, or a config its size does not serve, is refused,
    /// the config by the kind of its refusal.
    #[test]
    fn scopes_a_base_model_at_its_layout() {
        type Base = Model<qwen3::Size<"0.6B">, (), ()>;
        let directory = tempfile::tempdir().unwrap();
        let scope = |config: serde_json::Value, names: &[&str]| {
            let mut source = Checkpoint::new(directory.path(), Some(config)).unwrap();
            for name in names {
                let bytes = vec![0; 2];
                source
                    .insert(name, safetensors::Dtype::BF16, &[1], bytes)
                    .unwrap();
            }
            Base::scope(&source)
        };
        let published = qwen3::tests::published;
        let root = "embed_tokens.weight";
        let nested = "model.embed_tokens.weight";
        assert_eq!(scope(published(), &[root]).unwrap(), "");
        assert_eq!(scope(published(), &[nested]).unwrap(), "model");
        let both = "the checkpoint holds a base model at its root and under model";
        let mut rescaled = published();
        rescaled["rope_scaling"] = serde_json::json!({"rope_type": "linear", "factor": 2.0});
        let mut wide = published();
        wide["hidden_size"] = serde_json::json!(1000);
        for (refused, expected) in [
            (
                scope(published(), &[root, nested]),
                format!("invalid {root}: {both}"),
            ),
            (
                scope(rescaled, &[]),
                "unsupported config.json: rope_scaling: Qwen3 rotates without scaling".into(),
            ),
            (
                scope(wide, &[]),
                "invalid config.json: hidden_size: 1000, the size 0.6B holds 1024".into(),
            ),
        ] {
            let refusal = match refused {
                Err(LoadError::InvalidData { name, reason }) => format!("invalid {name}: {reason}"),
                Err(LoadError::Unsupported { name, reason }) => {
                    format!("unsupported {name}: {reason}")
                }
                other => panic!("{other:?}"),
            };
            assert_eq!(refusal, expected);
        }
    }
}
