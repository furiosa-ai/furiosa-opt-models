//! Vocabulary projection and first-maximum token selection.
//! The dense head has 152064 padded token slots and `Padded` heads any count; a mask bounds a
//! selection to the checkpoint's tokens, of which there may be fewer.

pub mod argmax;
pub mod readout;

pub use argmax::vector::Argmax;

use furiosa_opt_std::prelude::*;

axes![Vocab = 151936, VocabPad = 152064];
