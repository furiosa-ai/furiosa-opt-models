//! Compile-time execution choices for linear projections.

use core::marker::ConstParamTy;

/// Where the FP32 split projection keeps its BF16 parts between contractions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ConstParamTy)]
pub enum Split {
    /// Materialize split rows and partial sums in HBM before combining them.
    Staged,
    /// Keep coefficients in DM and complete each input strip before storing it.
    Resident,
}
