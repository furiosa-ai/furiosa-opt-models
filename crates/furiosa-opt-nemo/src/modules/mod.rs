#![deny(clippy::all)]
//! Reusable model structures composed from ops.
//!
//! Only [`generic`] remains: combinators that route operands between child modules and apply
//! layer arrays in order.

pub mod generic;
