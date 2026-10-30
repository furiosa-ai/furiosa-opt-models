//! The order a model's leaves run in, kept apart from the tree that holds them and the leaves
//! themselves: a leaf computes, the schedule says when.
//!
//! - [`decoder`]: a decoder's family loops over its layers and its row tiles.

pub mod decoder;

pub use decoder::{Decoder, Tokenwise};
