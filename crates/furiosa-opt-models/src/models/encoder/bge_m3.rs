//! XLM-RoBERTa encoding with independently callable dense, ColBERT and sparse heads.
use super::bert;
use crate::models::Family;
use crate::models::config::{Constant, Literal, flattened};
use crate::{
    Then,
    modules::language,
    ops::{
        index::gather::Gather,
        linear::Affine,
        normalization::Normalize,
        pointwise::{activation::Relu, cast::Convert as Cast},
        reduction::ScatterMax,
    },
    tensor::axis::Single,
};
use furiosa_opt_std::prelude::*;
use serde::Deserialize;
use std::io;

/// Shared encoder with the requested retrieval heads.
pub type Model<B, W, V> = language::Pooler<
    B,
    language::RetrievalHead<Option<Dense>, Option<Tokens<W>>, Option<Sparse<W, V>>>,
>;

/// Encoder with all three retrieval heads.
pub type Retrieval<V, P, S, W, I, H, const L: usize> =
    language::Pooler<bert::Body<V, P, S, W, I, H, L>, Head<W, V>>;

/// Caller-selected row normalized to a dense retrieval vector.
pub type Dense = language::Pooler<Cast<f32>, language::SequenceHead<Gather, Normalize>>;

/// Maximum token weight scattered to each vocabulary index.
pub type Sparse<W, V> = language::Pooler<Weights<W>, ScatterMax<V>>;

/// Nonnegative sparse weight for every encoded token.
pub type Weights<W> = language::Readout<Then<Cast<f32>, Linear<W, Single>>, Relu>;

pub type Linear<W, O> = Affine<HbmTensor<f32, m![1], m![O, W]>, [HbmTensor<f32, m![1], m![O]>; 1]>;

/// Dense, token and sparse retrieval heads over the same hidden rows.
pub type Head<W, V> = language::RetrievalHead<Dense, Tokens<W>, Sparse<W, V>>;

/// ColBERT token projections normalized independently per row.
pub type Tokens<W> = language::Pooler<Then<Cast<f32>, Linear<W, W>>, Normalize>;

/// XLM-RoBERTa encoder configuration and retrieval output choice.
#[derive(Clone)]
pub struct Config {
    /// Shared bidirectional encoder geometry.
    pub encoder: bert::Config,
    /// Requested retrieval heads.
    pub head: language::RetrievalHead<bool, bool, bool>,
}

impl Family for Config {
    type Hf = Hf;
}

/// An XLM-RoBERTa encoder's checkpoint configuration.
#[derive(Deserialize)]
pub struct Hf {
    #[serde(flatten, deserialize_with = "flattened")]
    encoder: bert::Hf,
    #[serde(flatten, deserialize_with = "flattened")]
    positions: Positions,
}

/// XLM-RoBERTa's position convention, which offsets each position past the padding index.
#[derive(Deserialize)]
struct Positions {
    model_type: Literal<"xlm-roberta">,
    pad_token_id: Constant<1>,
}

impl TryFrom<Hf> for Config {
    type Error = io::Error;

    /// Narrows the encoder, requesting every retrieval head.
    fn try_from(hf: Hf) -> io::Result<Self> {
        let Hf {
            encoder,
            positions:
                Positions {
                    model_type: Literal,
                    pad_token_id: Constant,
                },
        } = hf;
        Ok(Self {
            encoder: encoder.try_into()?,
            head: language::RetrievalHead {
                dense: true,
                tokens: true,
                sparse: true,
            },
        })
    }
}
