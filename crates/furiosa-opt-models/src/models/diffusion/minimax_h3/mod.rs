//! MiniMax H3 FL2VA configuration, component weight trees and request metadata.
//!
//! Checked inputs describe the reference geometry. Input projections and final heads compose
//! public numerical leaves; packing and positions describe runtime work. The engine sequences
//! the component calls on one four-chip pipeline with separately owned request state. Latent
//! statistics are retained without changing the normalization boundary or folding them into
//! decoder parameters.

pub use config::{Config, Shifts, Statistics};
pub use inputs::{Component, Inputs, Layout};

pub mod audio;
pub mod config;
pub mod denoiser;
pub mod embeddings;
mod inputs;
pub mod joint;
pub mod packing;
pub mod positions;
pub mod request;
pub mod text;
pub mod transformer;
pub mod video;

#[cfg(test)]
mod test_memory;
