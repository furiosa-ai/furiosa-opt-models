//! Numerical operations and their device layouts.

#![deny(clippy::arbitrary_source_item_ordering)]

// Elementwise functions.
pub mod pointwise;
pub(crate) mod vector;

// Projection and normalization.
pub mod linear;
pub mod normalization;
pub mod swiglu;

// Attention and sequence mixing.
pub mod attention;
pub mod recurrent;
pub mod rotary;

// Indexing and communication.
pub mod collective;
/// Rows selected or stored by index, and words copied between device buffers.
pub mod index;
pub mod reduction;

// Image and embedding functions.
pub mod convolution;
pub mod embedding;
pub mod gaussian;
