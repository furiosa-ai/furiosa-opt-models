//! E4M3 span projections with row activation scales and folded weight channel scales.

use super::*;

pub(crate) const LARGEST: f32 = 448.0;
pub(crate) const SMALLEST: f32 = 1e-12;
axes![Copies = 32, Side = 2, Panels = 3];
