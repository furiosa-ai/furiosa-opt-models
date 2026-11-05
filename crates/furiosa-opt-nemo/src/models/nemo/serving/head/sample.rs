//! Sampling selected rows on the device: units of 16 rows, a row a PE (`unit`), chained over a
//! launch's drawn rows.
//!
//! A row's draw, as the host reference defines it:
//! - a banned row's stop tokens drop below every other entry, with no weight;
//! - top-k keeps every entry at or above the k-th largest key (ties kept);
//! - weights are `exp((x - max) * inverse)` in f32;
//! - top-p keeps the top-k entries by (key desc, id asc) while the weight before an entry is below
//!   `top_p` times the top-k weight: the whole levels above the boundary level, and the boundary
//!   level's first ties in id order;
//! - the token is the smallest id whose cumulative kept weight in id order exceeds `uniform` times
//!   the kept weight.
//!
//! A key is the BF16 logit's signed monotone order, both zeros sharing one key.

use furiosa_opt_std::prelude::*;

axes![Full = 131072];

/// The sampler's per-entry table of this chip's shard, loaded with the weights.
pub type Table = crate::models::nemo::parameters::Sampling;

pub mod unit;
