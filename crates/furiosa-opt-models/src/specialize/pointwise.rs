//! The windows an elementwise function moves its flattened input in: a part of at most 512
//! elements in each of at most a cluster's slices, both dividing the input.

#![deny(clippy::arbitrary_source_item_ordering)]

use super::SLICES;

/// Elements a part holds at most.
const PART: usize = 512;

/// Elements per slice for `elements` inputs: the smallest multiple of eight that spreads them
/// over the slices, at most [`PART`], lowered by eights until it divides them.
pub(crate) const fn part(elements: usize) -> usize {
    assert!(
        elements > 0 && elements.is_multiple_of(8),
        "pointwise tensors require complete eight-element vectors"
    );
    let minimum = elements.div_ceil(SLICES).next_multiple_of(8);
    let mut part = if minimum < PART { minimum } else { PART };
    while elements % part != 0 {
        part -= 8;
    }
    part
}

/// Elements per window for `elements` inputs: as many parts as divide them, at most one a slice.
pub(crate) const fn window(elements: usize) -> usize {
    let parts = elements / part(elements);
    let mut lanes = if parts < SLICES { parts } else { SLICES };
    while parts % lanes != 0 {
        lanes -= 1;
    }
    lanes * part(elements)
}
