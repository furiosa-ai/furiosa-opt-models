//! Tiles of a projection onto a vocabulary: the hidden features each slice contracts and the
//! vocabulary rows one weight pass holds.
//!
//! - Each pass issues the next pass's weights before its own contraction, so the weights stream
//!   under the compute; two passes' weights sit beside the input rows in data memory.
//! - Sixteen slices share a row's contraction where two passes of their chunks fit the weights'
//!   share of a slice's data memory, else thirty-two; a row whose parts are no whole contraction
//!   steps splits into fewer, wider ones, whose passes hold fewer rows.

#![deny(clippy::arbitrary_source_item_ordering)]

use crate::ops::linear::kernel::Step;
use furiosa_opt_std::prelude::{AxisName, bf16};

/// A slice's data memory is sixteen pages of 32 KiB.
const PAGE: usize = 32 * 1024;

/// Data memory two passes of BF16 weights may take: the rest holds a pass's FP32 logits, the
/// resident per-vocabulary vectors and the staged rows.
const WEIGHTS: usize = 9 * PAGE;

/// The most vocabulary rows a pass holds.
pub(crate) const ROWS: usize = 264;

/// A pass's rows come eight at a time, since a per-vocabulary vector holds eight rows a word.
pub(crate) const STEP: usize = 8;

/// Hidden features each slice contracts: whole contraction steps of a power-of-two part of the
/// row, the sixteenth or thirty-second where those are, else the widest part that is.
pub(crate) const fn chunk(hidden: usize) -> usize {
    let mut parts = match 2 * ROWS * (hidden / 16) * 2 <= WEIGHTS {
        true => 16,
        false => 32,
    };
    while parts > 1 && !(hidden / parts).is_multiple_of(<Step<bf16> as AxisName>::SIZE) {
        parts /= 2;
    }
    hidden / parts
}

/// The most rows a pass of a slice's `chunk` of features holds: two passes' weights fit their
/// share of data memory.
pub(crate) const fn rows(chunk: usize) -> usize {
    let fit = WEIGHTS / (2 * chunk * 2);
    if fit < ROWS { fit } else { ROWS }
}

/// The rows a pass holds: the most rows, at most `most` and a multiple of `step`, that split a
/// slice's `block` of vocabulary rows into whole passes.
pub(crate) const fn pass(block: usize, most: usize, step: usize) -> usize {
    let mut rows = most - most % step;
    while rows > 0 && !block.is_multiple_of(rows) {
        rows -= step;
    }
    assert!(
        rows > 0,
        "no pass of whole steps splits the vocabulary block"
    );
    rows
}
