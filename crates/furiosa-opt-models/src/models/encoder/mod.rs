//! Encoder families and the catalog of unsized encoder compositions.

use crate::modules::{attention, language, residual, transformer};
use crate::{Either, Identity, Layers, Sequential};
use bert::BertBody;

pub mod bert;
pub mod bge_m3;
pub mod clip;
pub mod contrastive;
pub mod vision;

pub(super) type NormalizedEncoder<
    Embedding,
    Qkv,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    const DEPTH: usize,
> = language::Normalized<
    language::Backbone<
        Embedding,
        Layers<
            [transformer::PreNorm<attention::Full<Qkv, Attend, Project>, Mlp, Norm, Add>; DEPTH],
        >,
    >,
    Norm,
>;

/// Source: vLLM `vllm/model_executor/models/bert.py`.
pub type BertSpladeSparseEmbeddingModel<
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
    Pool,
    const DEPTH: usize,
> = language::Pooler<
    BertBody<Embedding, attention::Full<Qkv, Attend, Project>, Mlp, Norm, Add, DEPTH>,
    language::SparseHead<language::MaskedHead<Projection, Activation, HeadNorm, Vocabulary>, Pool>,
>;

type ModernBertBody<Embedding, Qkv, Position, Attend, Project, Mlp, Norm, Add, const DEPTH: usize> =
    language::Normalized<
        language::Backbone<
            Embedding,
            Layers<
                [Sequential<
                    residual::Residual<
                        Either<Identity, Norm>,
                        attention::Full<Sequential<(Qkv, Identity), Position>, Attend, Project>,
                        Add,
                    >,
                    residual::Residual<Norm, Mlp, Add>,
                >; DEPTH],
            >,
        >,
        Norm,
    >;

/// Source: vLLM `vllm/model_executor/models/modernbert.py`.
pub type ModernBertModel<
    Embedding,
    Qkv,
    Position,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Pool,
    const DEPTH: usize,
> = language::Pooler<
    ModernBertBody<Embedding, Qkv, Position, Attend, Project, Mlp, Norm, Add, DEPTH>,
    Pool,
>;

/// Source: vLLM `vllm/model_executor/models/modernbert.py`.
pub type ModernBertForSequenceClassification<
    Embedding,
    Qkv,
    Position,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Pool,
    Projection,
    Activation,
    HeadNorm,
    Classify,
    const DEPTH: usize,
> = ModernBertModel<
    Embedding,
    Qkv,
    Position,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    language::SequenceHead<Pool, language::MaskedHead<Projection, Activation, HeadNorm, Classify>>,
    DEPTH,
>;

/// Source: vLLM `vllm/model_executor/models/modernbert.py`.
pub type ModernBertForTokenClassification<
    Embedding,
    Qkv,
    Position,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Projection,
    Activation,
    HeadNorm,
    Classify,
    const DEPTH: usize,
> = language::Readout<
    ModernBertBody<Embedding, Qkv, Position, Attend, Project, Mlp, Norm, Add, DEPTH>,
    language::MaskedHead<Projection, Activation, HeadNorm, Classify>,
>;

/// Source: vLLM `vllm/model_executor/models/bert_with_rope.py`.
pub type NomicBertModel<
    Embedding,
    Qkv,
    Position,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Pool,
    const DEPTH: usize,
> = language::Pooler<
    language::Backbone<
        Embedding,
        Layers<
            [transformer::PostNorm<
                attention::Full<Sequential<(Qkv, Identity), Position>, Attend, Project>,
                Mlp,
                Norm,
                Norm,
                Add,
            >; DEPTH],
        >,
    >,
    Pool,
>;

/// Source: vLLM `vllm/model_executor/models/bert_with_rope.py`.
pub type GteNewForSequenceClassification<
    Embedding,
    Qkv,
    Position,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Pool,
    Score,
    const DEPTH: usize,
> = NomicBertModel<
    Embedding,
    Qkv,
    Position,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    language::SequenceHead<Pool, Score>,
    DEPTH,
>;

/// Source: SGLang `python/sglang/srt/models/pixtral.py`.
pub type PixtralVisionModel<
    PatchEmbedding,
    Qkv,
    Attend,
    Project,
    Mlp,
    Norm,
    Add,
    Head,
    const DEPTH: usize,
> = language::Encoder<
    PatchEmbedding,
    [transformer::PreNorm<attention::Full<Qkv, Attend, Project>, Mlp, Norm, Add>; DEPTH],
    Head,
>;
