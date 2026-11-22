//! Shared four-chip entry bindings for the model's text embedding and layer stages.

pub(super) use crate::forward as embedding;
pub(super) use crate::forward as layer;
pub(super) use furiosa_opt_models::models::diffusion::minimax_h3::text::launch::Layer;
