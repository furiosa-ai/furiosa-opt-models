//! mT5, by published size: the T5 tree over a 250112-token vocabulary, each [`Size`] a
//! [`Geometry`] of it.
//!
//! - mT5 reads no tying from its config: its published checkpoints hold a head of their own, and
//!   its decoder's rows are never scaled before it.

use std::io;

use super::t5::{Defaults, FORMS, Geometry, Heads, Hf};
use crate::models::Family;
use crate::ops::attention::geometry::{self, QueryWidth};
use crate::ops::swiglu;
use furiosa_opt_std::prelude::*;

/// A published mT5 checkpoint size, by its hub spelling: `"small"` is google/mt5-small.
pub struct Size<const N: &'static str>;

axes![Vocabulary = 250112];

impl Geometry for Size<"small"> {
    type Hidden = geometry::Hidden<512>;
    type Intermediate = swiglu::Intermediate<1024>;
    type Heads = Heads<6>;
    type Queries = QueryWidth<384>;
    type Vocabulary = Vocabulary;

    const SIZE: &'static str = "small";
    const MODEL_TYPE: &'static str = "mt5";
    const LAYERS: usize = 8;
    const DECODER_LAYERS: usize = 8;
    const FEED_FORWARD: &'static str = FORMS[1];
    const TIED_EMBEDDINGS: bool = false;
}

impl Geometry for Size<"base"> {
    type Hidden = geometry::Hidden<768>;
    type Intermediate = swiglu::Intermediate<2048>;
    type Heads = Heads<12>;
    type Queries = QueryWidth<768>;
    type Vocabulary = Vocabulary;

    const SIZE: &'static str = "base";
    const MODEL_TYPE: &'static str = "mt5";
    const LAYERS: usize = 12;
    const DECODER_LAYERS: usize = 12;
    const FEED_FORWARD: &'static str = FORMS[1];
    const TIED_EMBEDDINGS: bool = false;
}

impl Geometry for Size<"large"> {
    type Hidden = geometry::Hidden<1024>;
    type Intermediate = swiglu::Intermediate<2816>;
    type Heads = Heads<16>;
    type Queries = QueryWidth<1024>;
    type Vocabulary = Vocabulary;

    const SIZE: &'static str = "large";
    const MODEL_TYPE: &'static str = "mt5";
    const LAYERS: usize = 24;
    const DECODER_LAYERS: usize = 24;
    const FEED_FORWARD: &'static str = FORMS[1];
    const TIED_EMBEDDINGS: bool = false;
}

/// `MT5Config`'s defaults: mT5-small's geometry, with a gated GELU feed-forward.
const DEFAULTS: Defaults = Defaults {
    vocab_size: 250112,
    d_model: 512,
    d_kv: 64,
    d_ff: 1024,
    num_layers: 8,
    num_heads: 6,
    relative_attention_num_buckets: 32,
    relative_attention_max_distance: 128,
    layer_norm_epsilon: 1e-6,
    feed_forward_proj: FORMS[1],
};

/// A config is of a size when it declares the size's values.
impl<const N: &'static str> TryFrom<Hf> for Size<N>
where
    Self: Geometry,
{
    type Error = io::Error;

    fn try_from(hf: Hf) -> io::Result<Self> {
        hf.check::<Self>(&DEFAULTS).map(|()| Size)
    }
}

impl<const N: &'static str> Family for Size<N>
where
    Self: Geometry,
{
    type Hf = Hf;
}
