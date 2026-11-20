//! H3's text-only Qwen3-VL encoder, read after layer 50 and before the final norm.
//!
//! The language subtree loads relative to `text_encoder.model.language_model`.
//! The pipeline accounts for the unused vision tower, later layers, final norm and head.
//! Every chip holds the embedding, norms and residual rows; projections partition attention
//! heads and MLP features, and row-parallel outputs sum those parts. Independent layer owners
//! support the pipeline's shared weight residency without changing the numerical composition.
//! Callers supply causal positions and each layer's KV cache separately from the weight tree.

use super::config::TextEncoder;
use crate::load::Named;
use crate::models::decoder::{
    self,
    attention::{Head, KvHeads},
};
use crate::modules::{
    attention, feedforward::glu::Glu, language::Backbone, parallel::RowParallel, transformer,
};
use crate::ops::{
    embedding,
    normalization::{PerRow, RmsNorm},
    pointwise::{activation::Silu, add::Add, multiply::Multiply},
};
use crate::tensor::{
    form::Weight,
    mapping::Reshape,
    memory::{Hbm, Memory},
    parallel::{Parallel, Parallelism},
};
use crate::{Layers, Rolled, Sequential};
use furiosa_opt_std::prelude::*;

pub mod launch;
pub mod parameters;
pub mod position;
pub mod request;

/// The encoder's language body, without final normalization or a vocabulary readout.
/// Arrays retain independently loadable layers; `N` also permits a one-layer launch.
pub type Body<P = Parallel<4>, X = Hbm, const N: usize = LAYERS> =
    Backbone<Named<"embed_tokens", Embedding<P, X>>, Layers<[Layer<P, X>; N], Rolled>>;

/// The complete token table, replicated on the four chips.
pub type Embedding<P = Parallel<4>, X = Hbm> = embedding::Embedding<
    Vocabulary,
    Hidden,
    Chips<P>,
    Weight<
        m![Vocabulary, Hidden],
        Chips<P>,
        <X as Memory>::Tensor<bf16, Chips<P>, m![Vocabulary, Hidden]>,
    >,
>;

/// One ordinary pre-norm Qwen3 layer; its state carries its own causal KV cache.
pub type Layer<P = Parallel<4>, X = Hbm> =
    transformer::PreNorm<Attention<P, X>, Mlp<P, X>, Norm<Hidden, P, X>, Add>;

/// Separate Q/K/V projections with the checkpoint's independent per-head Q/K gains.
pub type Attention<P = Parallel<4>, X = Hbm> = attention::Cached<
    attention::NormalizedQkv<
        attention::Projection<
            Named<"q_proj", Heads<QueryWidth, Queries, P, X>>,
            Named<"k_proj", Heads<KvWidth, KvHeads, P, X>>,
            Named<"v_proj", Heads<KvWidth, KvHeads, P, X>>,
        >,
        Named<"q_norm", PerRow<Norm<Head, P, X>>>,
        Named<"k_norm", PerRow<Norm<Head, P, X>>>,
    >,
    decoder::Position<Group, bf16, Chips<P>>,
    crate::ops::index::scatter::Scatter,
    (),
    decoder::Attend<Group, SCALE, P>,
    Named<"o_proj", RowParallel<Rows<QueryWidth, Hidden, P, X>, P>>,
>;

/// The checkpoint's separate gate/up matrices, BF16 SiLU/product, then a summed down projection.
pub type Mlp<P = Parallel<4>, X = Hbm> = Glu<
    Columns<Hidden, Intermediate, P, X>,
    Silu,
    Columns<Hidden, Intermediate, P, X>,
    Multiply,
    RowParallel<Rows<Intermediate, Hidden, P, X>, P>,
>;

type Chips<P> = <P as Parallelism>::Chip;
type Local<P, W> = <P as Parallelism>::Local<W>;
type Norm<W, P, X> = RmsNorm<
    W,
    EPSILON,
    Chips<P>,
    Weight<m![W], Chips<P>, <X as Memory>::Tensor<bf16, Chips<P>, m![W]>>,
>;
type Heads<Width, Count, P, X> =
    Sequential<Columns<Hidden, Width, P, X>, Reshape<m![{ Symbol<Local<P, Count>> }, Head]>>;
type Columns<In, Out, P, X> = parameters::Projection<
    In,
    Local<P, Out>,
    In,
    Out,
    Chips<P>,
    <X as Memory>::Tensor<bf16, Chips<P>, m![{ Symbol<Local<P, Out>> }, In]>,
>;
type Rows<In, Out, P, X> = parameters::Projection<
    Local<P, In>,
    Out,
    In,
    Out,
    Chips<P>,
    <X as Memory>::Tensor<bf16, Chips<P>, m![Out, { Symbol<Local<P, In>> }]>,
>;

/// The same one-row, full-head rotation and paged causal attention geometry as the repro.
pub type Group =
    decoder::Group<KvHeads, Head, Groups, QueryWidth, KvWidth, Head, decoder::attention::Chunks>;
pub type Hidden = crate::ops::attention::geometry::Hidden<5120>;
pub type Intermediate = crate::ops::swiglu::Intermediate<25600>;
pub type Groups = crate::ops::attention::geometry::QueryGroups<8>;
pub type QueryWidth = crate::ops::attention::geometry::QueryWidth<8192>;
axes![Queries = 64, KvWidth = 1024, Vocabulary = 151936];
pub const LAYERS: usize = TextEncoder::READ_LAYERS;
const EPSILON: usize = 1e-6_f32.to_bits() as usize;
const SCALE: usize = crate::ops::attention::geometry::scale(Head::SIZE);
