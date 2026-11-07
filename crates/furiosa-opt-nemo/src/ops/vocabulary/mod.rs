//! Vocabulary projection and first-maximum token selection.
//! The dense head has 152064 padded token slots and `Padded` heads any count; a mask bounds a
//! selection to the checkpoint's tokens, of which there may be fewer.

pub mod argmax;
pub(crate) mod grouped;
pub(crate) mod head;
mod padded;
pub mod projection;
pub mod readout;
pub(crate) mod select;

pub use argmax::vector::Argmax;
pub use padded::Padded;

use crate::ops::weight::Weight;
use furiosa_opt_std::prelude::*;

axes![Vocab = 151936, VocabPad = 152064];

/// Full vocabulary projection, independently composable with a token-selection module.
#[derive(DeviceSend)]
pub struct LmHead<H: AxisName, Parameter = Weight<m![VocabPad, H]>>(
    pub Parameter,
    pub core::marker::PhantomData<H>,
);

/// Vocabulary projection followed by greedy selection. Its fused lowering retains per-pass
/// candidates instead of materializing the full logits tensor.
#[derive(DeviceSend)]
pub struct Greedy<
    H: AxisName,
    Parameter = Weight<m![VocabPad, H]>,
    Mask = HbmTensor<f32, m![1], m![VocabPad]>,
    Iota = HbmTensor<i32, m![1], m![VocabPad]>,
> {
    /// BF16 [padded vocabulary, hidden width] projection weight.
    pub head: LmHead<H, Parameter>,
    /// Zero for valid tokens and -PENALTY for padded tokens.
    pub mask: Mask,
    /// Reversed token IDs, VocabPad - 1 - token; larger ranks choose the first tied token.
    pub iota: Iota,
}

impl<H: AxisName> Greedy<H> {
    pub const fn new(
        weight: Weight<m![VocabPad, H]>,
        mask: HbmTensor<f32, m![1], m![VocabPad]>,
        iota: HbmTensor<i32, m![1], m![VocabPad]>,
    ) -> Self {
        Self {
            head: LmHead(weight, core::marker::PhantomData),
            mask,
            iota,
        }
    }
}

/// One I32 token per row in four-element padded storage for the selection transfer layout.
pub type Tokens<Rows> = HbmTensor<i32, m![1], m![Rows, 1 # 4]>;
