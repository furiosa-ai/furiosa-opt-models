//! Mistral NeMo's dense decoder served as one step a launch: the device code in `serving`, the
//! weights it reads in `parameters`.

use crate::ops::attention::geometry;
use crate::ops::layout::axes::{Parallel, Parallelism as Placement};
use crate::ops::normalization::RmsNorm;

pub mod parameters;
/// The host tables of one step launch.
pub mod plan;
/// The serving step: the decode and prefill kernels the vLLM plugin runs, one image a step.
pub mod serving;

/// Decoder layer count in checkpoint order.
pub const LAYERS: usize = 40;

/// FP32 bits of the served checkpoint's norm epsilon.
pub const NORM_EPSILON: usize = 1e-5_f32.to_bits() as usize;

/// FP32 bits of the score multiplier of 128-wide heads.
pub const SCORE_SCALE: usize = (std::f32::consts::FRAC_1_SQRT_2 / 8.0).to_bits() as usize;

/// Hidden feature width.
pub type Hidden = geometry::Hidden<5120>;

type Norm<Hidden, P = Parallel<1>, const EPSILON: usize = NORM_EPSILON> =
    RmsNorm<Hidden, EPSILON, <P as Placement>::Chip>;
