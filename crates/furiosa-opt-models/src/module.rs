//! The `Module` trait every model piece implements, `Op`, the module of device and input alone, and
//! `Begin`, the input of a module's first phase.

use furiosa_opt_std::prelude::Device;

// --8<-- [start:module]
/// A parameterized computation on the device, over the call's shared reader, the state it borrows
/// and its input operands.
pub trait Module<I, E, S> {
    /// Result of the computation for this input type.
    type Output;

    /// Computes the output from the reader, the state and the input.
    fn forward(&self, device: &mut Device, env: E, state: S, input: I) -> Self::Output;
}

/// An op: a leaf operation of `ops/`, made of VISA code and other ops; it reads no env and holds
/// no state. A generic wrapper module declares `!Op`; the blanket impl below otherwise overlaps it.
pub trait Op<I> {
    /// Result of the computation for this input type.
    type Output;

    /// Computes the output from the input.
    fn forward(&self, device: &mut Device, input: I) -> Self::Output;
}

/// A module's input taken up to its first boundary, where it returns its state; given back, the
/// state runs the next step, and the state types refuse any other order at compile time.
pub struct Begin<T>(pub T);

/// Every op is a module: it ignores the env and holds the empty state.
impl<T: Op<I>, I, E> Module<I, E, &mut ()> for T {
    type Output = T::Output;

    fn forward(&self, device: &mut Device, _: E, _: &mut (), input: I) -> T::Output {
        Op::forward(self, device, input)
    }
}
// --8<-- [end:module]

impl<I, E, S, M: Module<I, E, S> + ?Sized> Module<I, E, S> for &M {
    type Output = M::Output;

    fn forward(&self, device: &mut Device, env: E, state: S, input: I) -> Self::Output {
        (**self).forward(device, env, state, input)
    }
}

/// A borrowed op is a module, not an op.
impl<I, T: ?Sized> !Op<I> for &T {}

/// The unit module never reads a borrowed input, so a composition can take `()` for an absent part.
impl<X: ?Sized, E, S> !Module<&X, E, S> for () {}

/// Nor is it an op on one, which the blanket impl would make such a module.
impl<X: ?Sized> !Op<&X> for () {}

/// No work: a host's boundary where no other module issues.
impl<E> Module<&mut (), E, &mut ()> for () {
    type Output = ();

    fn forward(&self, _: &mut Device, _: E, _: &mut (), _: &mut ()) {}
}
