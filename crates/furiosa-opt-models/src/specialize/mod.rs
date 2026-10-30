//! Values that specialize an op to the hardware, starting with the hardware's own limits: the
//! slices of a cluster and the register bytes of a slice, which an op's tiles budget against.

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
