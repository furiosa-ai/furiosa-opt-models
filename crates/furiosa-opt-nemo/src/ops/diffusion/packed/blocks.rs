//! The modulated norm and the gated residual of packed rows in HBM in blocks of 512 rows (256
//! when the rows do not come in whole 512-row blocks), eight (four) rows a slice.
//!
//! - A block puts each cluster's half of its rows on 32 groups of slices, each slice one contiguous
//!   run of each of its rows' features in whole 256-byte bursts; per-feature operands repeat over
//!   a slice's rows, so a pass reads each operand vector for all of them in a row.
//! - A span's strips before its first whole block and after its last run as strips.

use furiosa_opt_std::prelude::*;

axes![Term = 2];
