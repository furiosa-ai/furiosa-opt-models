//! An FP32 value as two BF16 parts: its rounding (the high part) and the rounding of what remains
//! (the low part).
//!
//! - A contraction over both parts recovers an FP32 product to within 2^-17 of each term: the sum
//!   over the high parts plus the sum over the low parts, then the FP32 offset, in that order.
//! - Coefficients stack their parts as output features (`Stacked`); rows put theirs side by side
//!   as input features (`Sides`), against coefficients `[high | high]` and `[low | 0]`.

use core::marker::PhantomData;

use furiosa_opt_std::prelude::*;

/// Two features per output feature: its BF16 high part, then its BF16 low part.
#[derive(Clone, Debug)]
pub struct Stacked<Out>(PhantomData<Out>);

impl<Out: AxisName> AxisName for Stacked<Out> {
    const NAME: Ident = Ident::new("Stacked");
    const SIZE: usize = 2 * Out::SIZE;
}

/// Two features per input feature: its BF16 high part's, then its low part's.
#[derive(Clone, Debug)]
pub struct Sides<In>(PhantomData<In>);

impl<In> Sides<In> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<In: AxisName> AxisName for Sides<In> {
    const NAME: Ident = Ident::new("Sides");
    const SIZE: usize = 2 * In::SIZE;
}
