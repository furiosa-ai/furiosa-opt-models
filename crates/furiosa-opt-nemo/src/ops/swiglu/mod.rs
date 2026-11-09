//! Packed gate/up projections compute down(silu(gate(x)) * up(x)) on BF16 rows.

use furiosa_opt_std::prelude::*;

pub mod spans;

// Gate then up in the packed projection.
axes![Term = 2];
