//! Multimodal structure: media merged into token embeddings, fused encoder features, encoder
//! memory read by cross layers, and modality-routed generation.

pub use cross::EncoderDecoder;
pub use embedding::{InputEmbedding, Replace, TokenEmbedding};
pub use fusion::DeepFusion;
pub use generative::Generative;

/// Fuses token and media embeddings before running a context-bearing language body.
pub use super::language::Backbone as Multimodal;

mod cross;
mod embedding;
mod fusion;
mod generative;
