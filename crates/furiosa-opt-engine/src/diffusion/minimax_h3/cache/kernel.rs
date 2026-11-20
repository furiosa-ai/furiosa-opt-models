//! Shared four-chip entry bindings for the separate TeaCache probe and statistics stages.

pub(super) use crate::forward_four as l1;
pub(super) use crate::forward_four as modulate;
pub(super) use crate::forward_four as norm;
pub(super) use furiosa_opt_models::models::diffusion::minimax_h3::denoiser::cache::{
    L1, Modulation, Modulations, Norm, NormTables, Totals,
};
