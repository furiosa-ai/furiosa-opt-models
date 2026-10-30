//! Encoder-decoder families and the catalog of their upstream compositions.

use crate::modules::{attention, language, multimodal, residual, transformer};
use crate::{Identity, Sequential, Then};

pub mod mt5;
pub mod t5;
pub mod whisper;

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
