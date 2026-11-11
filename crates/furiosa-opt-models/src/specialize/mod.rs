//! Values that specialize an op to the hardware, starting with the hardware's own limits: the
//! slices of a cluster and the register and data memory bytes of a slice, which an op's tiles
//! budget against.

pub mod attention;
pub(crate) mod contraction;
pub(crate) mod convolution;
pub(crate) mod diffusion;
pub(crate) mod index;
pub(crate) mod normalization;
pub(crate) mod pointwise;
pub(crate) mod vocabulary;

/// Slices of a cluster.
pub(crate) const SLICES: usize = 256;

/// Bytes of the vector register file a slice holds.
pub(crate) const VRF: usize = 8192;

/// Bytes of the tensor register file a slice holds.
pub(crate) const TRF: usize = 65_536;

/// Bytes of the data memory a slice holds.
pub(crate) const DM: usize = 524_288;

/// Bytes of a data memory page: a place larger than one takes whole pages, smaller ones share them.
pub(crate) const PAGE: usize = DM / 16;

/// FP32 cells of a slice's contraction accumulators: a reduction in time keeps every output it has
/// not finished in them.
pub(crate) const ACCUMULATOR: usize = 1024;

/// Bytes of the HBM access a transfer's packets stream at full rate in: a run that is a whole
/// number of them lands in them, any other run in packets that each cost an access more.
pub(crate) const PACKET: usize = 256;

/// The plan of a pairwise all-reduce: one round for each bit of a chip's index, each adding a
/// chip's partial sum to the partial of the chip whose index differs in that bit.
pub(crate) mod collective {
    /// Rounds of an all-reduce over `chips` chips, a power of two.
    pub(crate) const fn rounds(chips: usize) -> usize {
        assert!(
            chips.is_power_of_two(),
            "a pairwise all-reduce pairs the chips bit by bit"
        );
        chips.ilog2() as usize
    }
}
