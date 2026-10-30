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

    /// Each of `N` chips' partner in each round: round `r` pairs the chips whose indices differ in
    /// bit `r`. The first `rounds(N)` of the `N` rows are the rounds.
    pub(crate) const fn partners<const N: usize>() -> [[usize; N]; N] {
        let mut partners = [[0; N]; N];
        let (mut round, mut mask) = (0, 1);
        while mask < N {
            let mut chip = 0;
            while chip < N {
                partners[round][chip] = chip ^ mask;
                chip += 1;
            }
            (round, mask) = (round + 1, mask * 2);
        }
        partners
    }

    /// Where each of `N` chips finds, in each round, the partial of the pair's chip whose round bit
    /// is `high`: its own (0) or its partner's (1).
    pub(crate) const fn selections<const N: usize>(high: bool) -> [[usize; N]; N] {
        let mut selections = [[0; N]; N];
        let (mut round, mut mask) = (0, 1);
        while mask < N {
            let mut chip = 0;
            while chip < N {
                selections[round][chip] = if (chip & mask != 0) == high { 0 } else { 1 };
                chip += 1;
            }
            (round, mask) = (round + 1, mask * 2);
        }
        selections
    }
}
