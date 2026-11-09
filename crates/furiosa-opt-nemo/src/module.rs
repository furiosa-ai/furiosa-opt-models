//! The `Module` trait every model piece implements, and its device-call alias.

use furiosa_opt_std::prelude::Device;

// --8<-- [start:module]
/// A parameterized computation whose input carries any writable invocation state.
pub trait Module<Input> {
    /// Result of the computation for this input type.
    type Output;

    /// Computes the output from caller-supplied operands and state.
    fn forward(&self, input: Input) -> Self::Output;
}
// --8<-- [end:module]

/// A module callable with any device borrow and the declared operands.
pub trait DeviceModule<Input> = for<'d> Module<(&'d mut Device, Input)>;

impl<I, M: Module<I> + ?Sized> Module<I> for &M {
    type Output = M::Output;

    fn forward(&self, input: I) -> Self::Output {
        (**self).forward(input)
    }
}

impl<I, M: Module<I>> Module<I> for Option<M> {
    type Output = Option<M::Output>;

    fn forward(&self, input: I) -> Self::Output {
        self.as_ref().map(|module| module.forward(input))
    }
}
