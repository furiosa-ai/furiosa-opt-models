//! Numerical operations and their device layouts.

// Elementwise functions.
pub mod add;
pub mod cast;
mod pointwise;
pub(crate) mod vector;

// Projection and normalization.
pub mod linear;
pub mod normalization;
pub mod swiglu;

// Attention and sequence mixing.
pub mod attention;

// Layout, indexing and communication.
pub mod collective;
pub mod indexing;
pub mod layout;

// Image and embedding functions.
pub mod diffusion;
pub mod embedding;
pub mod vocabulary;

// Host storage boundary.
pub mod weight;
