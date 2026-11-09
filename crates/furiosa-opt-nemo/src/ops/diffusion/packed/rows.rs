//! The modulated norm and the gated residual of packed rows in HBM, a row a group of slices.
//!
//! - A strip of rows puts each cluster's half of it one row a group of slices, each slice one
//!   contiguous run of the row's features in whole 256-byte bursts, so every transfer moves
//!   aligned runs and a row's statistic reduces over the slices of its group.
//! - Strips are 64 rows (`wide`), or 16 (`narrow`) for a span whose edge falls inside a 64-row
//!   strip, as a sequence split over chips has.
//! - A span's per-feature operands sit in the vector registers of every slice that holds their
//!   features; the next strip lands while the current one computes, and a strip's result is
//!   stored while the next one computes.

/// 64-row strips: 32 rows a cluster, 8 slices a row, each `Run` features of the row.
pub(super) mod wide {

    use furiosa_opt_std::prelude::*;

    axes![Lines = 64];
}

/// 16-row strips: 8 rows a cluster, 32 slices a row, each `Spread` features of the row.
pub(super) mod narrow {

    use furiosa_opt_std::prelude::*;

    axes![Lines = 16];
}
