#![deny(clippy::all)]
//! Reusable model structures composed from ops: attention, feed-forward, layers and heads.

#![deny(clippy::arbitrary_source_item_ordering)]

pub mod attention;
pub mod audio;
pub mod contrastive;
pub mod convolution;
pub mod diffusion;
pub mod feedforward;
pub mod gaussian;
pub mod generic;
pub mod language;
pub mod lora;
pub mod mixing;
pub mod multimodal;
pub mod parallel;
pub mod recurrent;
pub mod residual;
pub mod sharing;
pub mod speculative;
pub mod transformer;
pub mod vision;
