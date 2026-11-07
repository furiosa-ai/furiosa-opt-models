//! Retained BF16 weights and parameters a checkpoint may omit.

use furiosa_opt_std::prelude::*;

/// Retained BF16 parameter in HBM; shape and chip mapping come from the wrapped tensor.
#[derive(DeviceSend)]
pub struct Weight<S: M, Chips: M = m![1]> {
    tensor: HbmTensor<bf16, Chips, S>,
}

impl<S: M, Chips: M> Weight<S, Chips> {
    /// Retain the tensor without copying or adding layout constraints.
    pub const fn new(tensor: HbmTensor<bf16, Chips, S>) -> Self {
        Self { tensor }
    }

    /// Borrow the retained BF16 tensor view.
    pub fn tensor(&self) -> &HbmTensor<bf16, Chips, S> {
        &self.tensor
    }

    /// The retained tensor itself, for a parameter form that holds it directly.
    pub fn into_tensor(self) -> HbmTensor<bf16, Chips, S> {
        self.tensor
    }
}

/// A parameter a checkpoint may omit: `()` when absent, the tensor itself when present.
pub trait Optional<T> {}

impl<D: Scalar, Chips: M, S: M> Optional<HbmTensor<D, Chips, S>> for () {}

impl<D: Scalar, Chips: M, S: M> Optional<HbmTensor<D, Chips, S>> for HbmTensor<D, Chips, S> {}
