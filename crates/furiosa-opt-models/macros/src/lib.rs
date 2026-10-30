//! `#[derive(Load)]`: the field-wise `Load` impls of a weight tree, one per direction a tree is
//! loaded in (see `furiosa_opt_models::load`).

#![deny(clippy::arbitrary_source_item_ordering)]

use proc_macro::TokenStream;

mod attribute;
mod expand;

/// Derives the `Load` impls of a struct; `#[load(..)]` attributes are described in
/// `furiosa_opt_models::load`.
#[proc_macro_derive(Load, attributes(load))]
pub fn load(input: TokenStream) -> TokenStream {
    expand::Shape::derive(&syn::parse_macro_input!(input))
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
