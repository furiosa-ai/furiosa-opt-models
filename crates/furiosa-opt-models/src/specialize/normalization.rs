//! The tiles an RMS norm's rows run in, budgeted against a slice's vector registers: the inverse
//! of each row of a tile, the gain, and what the preceding residual add still holds.

#![deny(clippy::arbitrary_source_item_ordering)]

use super::VRF;
use furiosa_opt_std::prelude::M;

/// VRF bytes of one row's inverse: eight FP32 lanes.
const ROW: usize = 8 * 4;

/// VRF bytes the preceding residual add can still hold when a norm starts: one flit.
const OPERAND: usize = 32;

/// VRF bytes a pass with register operands keeps for its register cache.
const CACHE: usize = 1024;

/// Rows per tile when tiles alternate, so two inverses fit beside the gain of `E` features and
/// the operand.
pub(crate) const fn capacity<E: M>() -> usize {
    (VRF - OPERAND - E::SIZE * 4) / (2 * ROW)
}

/// Full tiles of `T` rows before the last one; none when a single inverse holds every row.
pub(crate) const fn whole<T: M, E: M>() -> usize {
    if OPERAND + E::SIZE * 4 + T::SIZE * ROW <= VRF {
        0
    } else {
        (T::SIZE - 1) / capacity::<E>()
    }
}

/// Rows of the last tile, the only one that may be partial.
pub(crate) const fn last<T: M, E: M>() -> usize {
    T::SIZE - whole::<T, E>() * capacity::<E>()
}

/// Slices a row of `width` features spans in a norm over HBM rows that keeps `vectors` feature
/// vectors and two row scalars in registers: the fewest, four or more, of whole 16-feature packets.
pub(crate) const fn parts(width: usize, vectors: usize) -> usize {
    let mut parts = 4;
    while !width.is_multiple_of(16 * parts) || vectors * width / parts * 4 + 2 * ROW + CACHE > VRF {
        parts *= 2;
        assert!(parts <= 64, "no part of the row fits the vector registers");
    }
    parts
}
