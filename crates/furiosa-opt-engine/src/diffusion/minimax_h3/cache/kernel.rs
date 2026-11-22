//! Shared four-chip entry bindings for the separate TeaCache probe and statistics stages.

pub(super) use crate::forward as l1;
pub(super) use crate::forward as modulate;
pub(super) use crate::forward as norm;
pub(super) use furiosa_opt_models::models::diffusion::minimax_h3::denoiser::cache::{
    L1, Modulation, Modulations, Norm, NormTables, Totals,
};
