//! Recurrent decoders: a token embedding, a stack of state-space mixer layers and a final norm,
//! whose rows a language-model head reads.
//!
//! - A family's published sizes implement [`Geometry`], and its classes are aliases of the roots
//!   here. Fields carry Hugging Face's names, so a checkpoint loads into a class by name.
//! - A layer's recurrent state, its convolution's and its scan's, is a call input, not a weight.

use std::io;
use std::marker::PhantomData;

use crate::load::{Checkpoint, Load, LoadError};
use crate::models::{Class, Family, Implements};
use crate::tensor::memory::Memory;
use furiosa_opt_std::prelude::*;

pub mod mamba;
pub mod mamba2;

/// A published size of a recurrent decoder: its axes and the config values its type fixes.
pub trait Geometry {
    /// Features of a row: `hidden_size`.
    type Hidden: AxisName;
    /// Rows of the embedding and of the head as compiled: `vocab_size` tokens at most.
    type Vocabulary: AxisName;
    /// `num_hidden_layers` layers of `L`.
    type Layers<L>;
    /// A norm of `W` features of memory `X` with the size's `layer_norm_epsilon`.
    type Norm<W: AxisName, X: Memory>;

    /// The hub spelling of the size, as `"130m"`.
    const SIZE: &'static str;
    /// The family whose classes the size is of, as `"mamba"`.
    const FAMILY: &'static str;
    /// `num_hidden_layers`.
    const LAYERS: usize;
    /// `layer_norm_epsilon`, as the bits of the FP32 nearest to it.
    const EPSILON: usize;
    /// `tie_word_embeddings`: the head reads the embedding's table.
    const TIED_EMBEDDINGS: bool;
}

/// A causal language model of size `G`: the backbone, then the head that reads its rows.
#[derive(DeviceSend, Load)]
pub struct ForCausalLM<G: Geometry, Backbone, Head> {
    pub size: PhantomData<G>,
    pub backbone: Backbone,
    #[load(tied = "backbone.embeddings", config = "tie_word_embeddings")]
    pub lm_head: Head,
}

/// A causal model reads a checkpoint at its root.
impl<G: Geometry + Family, Backbone, Head> Implements for ForCausalLM<G, Backbone, Head> {
    const SIZE: &'static str = G::SIZE;
    const CLASS: Class = Class {
        family: G::FAMILY,
        name: "ForCausalLM",
    };

    fn scope(source: &Checkpoint) -> Result<&'static str, LoadError> {
        G::parse(source.config()).map(|_| "").map_err(|error| {
            let (name, reason) = ("config.json".into(), error.to_string());
            match error.kind() {
                io::ErrorKind::Unsupported => LoadError::Unsupported { name, reason },
                _ => LoadError::InvalidData { name, reason },
            }
        })
    }
}
