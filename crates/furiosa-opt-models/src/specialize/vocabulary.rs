//! Tiles of a projection onto a vocabulary: the hidden features each slice contracts and the
//! vocabulary rows one weight pass holds.
//!
//! - Each pass issues the next pass's weights before its own contraction, so the weights stream
//!   under the compute; two passes' weights sit beside the input rows in data memory.
//! - Sixteen slices share a row's contraction where two passes of their chunks fit the weights'
//!   share of a slice's data memory, else thirty-two.

#![deny(clippy::arbitrary_source_item_ordering)]

/// A slice's data memory is sixteen pages of 32 KiB.
const PAGE: usize = 32 * 1024;

/// Data memory two passes of BF16 weights may take: the rest holds a pass's FP32 logits, the
/// resident per-vocabulary vectors and the staged rows.
const WEIGHTS: usize = 9 * PAGE;

/// The most vocabulary rows a pass holds.
pub(crate) const ROWS: usize = 264;

/// A pass's rows come eight at a time, since a per-vocabulary vector holds eight rows a word.
pub(crate) const STEP: usize = 8;

/// Hidden features each slice contracts.
pub(crate) const fn chunk(hidden: usize) -> usize {
    match 2 * ROWS * (hidden / 16) * 2 <= WEIGHTS {
        true => hidden / 16,
        false => hidden / 32,
    }
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
