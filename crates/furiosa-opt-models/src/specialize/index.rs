//! Tiles of the index ops: the features of one row gather, the pieces of a copy, the rows of each
//! embedding gather a launch issues, and the rows of each store into the KV cache.

#![deny(clippy::arbitrary_source_item_ordering)]

/// Features one gather moves of a row: 16 divides it and it divides 4096 = 256 * 16, so a tile
/// fills 32 slices and `Hidden / FEATURES` gathers cover any such width.
pub(crate) const FEATURES: usize = 512;

/// The pieces of a copy between HBM buffers.
pub(crate) mod copy {
    /// Words of a piece, longest first, since a launch's operand shapes are static.
    pub(crate) const LENGTHS: [usize; 5] = [1024, 64, 16, 4, 1];
}

/// A launch's embedding gathers: a whole span of rows each, then a block each, and the last at most
/// `REST` rows as one, since a gather whose idle rows repeat one address serializes on it.
pub(crate) mod embedding {
    /// Rows of a gather of whole rows, one DMA element a row.
    pub(crate) const SPAN: usize = 512;

    /// Rows of a gather of feature tiles.
    pub(crate) const BLOCK: usize = 128;

    /// Rows of the last gather of feature tiles.
    pub(crate) const REST: usize = 8;
}

/// The stores of a launch's KV rows into the cache.
pub(crate) mod store {
    /// Rows of a store, longest first: each store is the longest that the remaining live rows
    /// fill, and the rows past the last full store write a packet, then a row, at a time.
    pub(crate) const LENGTHS: [usize; 3] = [512, 128, 32];

    /// Rows of a packet: the eight lanes of an index packet.
    pub(crate) const PACKET: usize = 8;
}
