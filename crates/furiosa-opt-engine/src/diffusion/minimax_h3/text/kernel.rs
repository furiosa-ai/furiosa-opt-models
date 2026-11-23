//! Shared four-chip entry bindings for the model's text embedding and layer stages.

pub(super) use crate::forward_four as embedding;
pub(super) use crate::forward_four as layer;
pub(super) use furiosa_opt_models::models::diffusion::minimax_h3::text::launch::Layer;
