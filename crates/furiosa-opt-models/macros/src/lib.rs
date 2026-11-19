//! Loading and structural residency passes generated from weight-tree declarations.
//!
//! `Load` owns checkpoint naming and loading directions; `Residency` maps and visits the leaves
//! of a tree over `Memory` (see `furiosa_opt_models::tensor::memory::residency`).

#![deny(clippy::arbitrary_source_item_ordering)]

use proc_macro::TokenStream;

mod attribute;
mod expand;
mod residency;

/// Derives structural map/visit passes; `memory(X)` and `children(A, B)` name the generic
/// positions that change. Every field must support traversal; no fields are skipped.
#[proc_macro_derive(Residency, attributes(residency))]
pub fn residency(input: TokenStream) -> TokenStream {
    residency::derive(&syn::parse_macro_input!(input))
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Derives the `Load` impls of a struct; `#[load(..)]` attributes are described in
/// `furiosa_opt_models::load`.
#[proc_macro_derive(Load, attributes(load))]
pub fn load(input: TokenStream) -> TokenStream {
    expand::Shape::derive(&syn::parse_macro_input!(input))
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
