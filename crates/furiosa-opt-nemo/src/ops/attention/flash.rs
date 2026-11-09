//! Bidirectional attention of a 4,096-row query tile over a long key sequence, one chip's heads,
//! each span of keys resting in the tensor register files while the queries stream past it.
//!
//! - A cluster attends one head at a time; the chip's two clusters take a pair of heads.
//! - A span of 2,048 keys spreads over a cluster's 256 slices as eight query groups of 512 rows by
//!   32 key slices of 64 keys. A slice keeps its keys and its values in its tensor register file;
//!   its group's queries stream past the keys through the switch, and the weights it computed
//!   stream past the values.
//! - The values carry 16 channels of ones after their 128, so one product holds a row's weighted
//!   values and its mass; the products sum over the key slices onto the slice that holds the row.
//! - Each span's peaks combine over its key slices into every row's running maximum.
//! - Scores<C> round to BF16 before they are scaled; the maximum and the products are FP32.

use furiosa_opt_std::prelude::*;

axes![
    Rows = 4096,
    TailRows = 1024,
    Span = 2048,
    Duo = 2,
    Head = 128,
    Channel = 144,
    Term = 2
];
