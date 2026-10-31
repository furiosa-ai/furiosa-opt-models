//! Encoder-decoder families and the catalog of their upstream compositions.

use crate::models::Ids;
use crate::modules::{attention, language, multimodal, residual, transformer};
use crate::ops::index::{gather::RowStorage, table::KeyValue};
use crate::{Identity, Sequential, Then};
use furiosa_opt_std::prelude::{Broadcast, HbmTensor, m};

pub mod mt5;
pub mod t5;
pub mod whisper;

/// Opaque F32 key/value row storage.
pub type Cache<Width> = KeyValue<RowStorage<Width>>;

/// Key/value byte offsets for each selected row.
pub type Offsets<Count> = KeyValue<Ids<Count>>;

/// Key/value write destinations and number of rows to write.
pub type Write<Count> = (Offsets<Count>, HbmTensor<i32, m![1], m![1]>);

/// Self-history write destinations and read offsets, with no rotary input.
pub type History<Queries, Keys> = attention::Context<(), Write<Queries>, Offsets<Keys>>;

/// Writable self-attention cache and its per-call history operands.
pub type State<Width, Queries, Keys> = (Cache<Width>, History<Queries, Keys>);

/// Projected source cache with the offsets used when creating or reading it.
pub type Cross<Width, Source> = (Cache<Width>, Write<Source>);

type ScaledFeedForward<Mlp, Norm, Scale, Add> = residual::Residual<Norm, Then<Mlp, Scale>, Add>;

type Conformer<Attention, Convolution, Mlp, Norm, Scale, Add> = Then<
    Sequential<
        ScaledFeedForward<Mlp, Norm, Scale, Add>,
        Sequential<
            residual::Residual<Norm, Attention, Add>,
            Sequential<
                residual::Residual<Norm, Convolution, Add>,
                ScaledFeedForward<Mlp, Norm, Scale, Add>,
            >,
        >,
    >,
    Norm,
>;

type CrossDecoder<Embedding, Layer, Norm, Head, const DEPTH: usize> = language::Readout<
    language::Normalized<language::Backbone<Embedding, [Layer; DEPTH]>, Norm>,
    Head,
>;

/// Source: vLLM `vllm/model_executor/models/cohere_asr.py`.
pub type CohereAsrForConditionalGeneration<
    EncoderSpectrogram,
    EncoderQkv,
    EncoderAttend,
    EncoderProject,
    EncoderMlp,
    EncoderConvolution,
    EncoderNorm,
    EncoderScale,
    EncoderAdd,
    EncoderHead,
    Embedding,
    Qkv,
    Position,
    Write,
    Read,
    Attend,
    Project,
    CrossQuery,
    CrossKeyValue,
    CrossWrite,
    CrossRead,
    CrossAttend,
    CrossProject,
    Mlp,
    Norm,
    Add,
    Head,
    const ENCODER_DEPTH: usize,
    const DECODER_DEPTH: usize,
> = multimodal::EncoderDecoder<
    language::Encoder<
        EncoderSpectrogram,
        [Conformer<
            attention::Full<EncoderQkv, EncoderAttend, EncoderProject>,
            EncoderConvolution,
            EncoderMlp,
            EncoderNorm,
            EncoderScale,
            EncoderAdd,
        >; ENCODER_DEPTH],
        EncoderHead,
    >,
    CrossDecoder<
        Embedding,
        transformer::CrossLayer<
            attention::Cached<Qkv, Position, Write, Read, Attend, Project>,
            attention::Cross<
                CrossQuery,
                CrossKeyValue,
                CrossWrite,
                CrossRead,
                CrossAttend,
                CrossProject,
            >,
            Mlp,
            Norm,
            Add,
        >,
        Norm,
        Head,
        DECODER_DEPTH,
    >,
>;

type RadioNeck<FeatureProjection, SummaryProjection, Spatial, Downsample, Flatten, Norm, Join> =
    Then<
        (
            Then<Identity, Then<SummaryProjection, Norm>>,
            Then<
                Identity,
                Then<
                    FeatureProjection,
                    Then<Norm, Then<Spatial, Then<Downsample, Then<Flatten, Norm>>>>,
                >,
            >,
        ),
        Join,
    >;

/// Source: vLLM `vllm/model_executor/models/nemotron_parse.py`.
pub type NemotronParseForConditionalGeneration<
    EncoderPatch,
    EncoderQkv,
    EncoderAttend,
    EncoderProject,
    EncoderMlp,
    EncoderNorm,
    EncoderScale,
    EncoderAdd,
    EncoderHead,
    FeatureProjection,
    SummaryProjection,
    NeckNorm,
    Spatial,
    Downsample,
    Flatten,
    Join,
    Embedding,
    Qkv,
    Position,
    Write,
    Read,
    Attend,
    Project,
    CrossQuery,
    CrossKeyValue,
    CrossWrite,
    CrossRead,
    CrossAttend,
    CrossProject,
    Mlp,
    Norm,
    Add,
    Head,
    const ENCODER_DEPTH: usize,
    const DECODER_DEPTH: usize,
> = multimodal::EncoderDecoder<
    Sequential<
        language::Encoder<
            EncoderPatch,
            [transformer::PreNorm<
                Then<attention::Full<EncoderQkv, EncoderAttend, EncoderProject>, EncoderScale>,
                Then<EncoderMlp, EncoderScale>,
                EncoderNorm,
                EncoderAdd,
            >; ENCODER_DEPTH],
            EncoderHead,
        >,
        RadioNeck<
            FeatureProjection,
            SummaryProjection,
            Spatial,
            Downsample,
            Flatten,
            NeckNorm,
            Join,
        >,
    >,
    CrossDecoder<
        Embedding,
        transformer::CrossLayer<
            attention::Cached<Qkv, Position, Write, Read, Attend, Project>,
            attention::Cross<
                CrossQuery,
                CrossKeyValue,
                CrossWrite,
                CrossRead,
                CrossAttend,
                CrossProject,
            >,
            Mlp,
            Norm,
            Add,
        >,
        Norm,
        Head,
        DECODER_DEPTH,
    >,
>;
