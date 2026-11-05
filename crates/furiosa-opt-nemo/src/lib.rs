//! Models as trees of `Module`s: numerical operations, reusable module structures and sized model
//! definitions.

#![feature(register_tool)]
#![feature(proc_macro_hygiene, stmt_expr_attributes)]
#![feature(inherent_associated_types)]
#![feature(trait_alias)]
#![feature(adt_const_params, unsized_const_params)]
#![feature(const_trait_impl)]
#![register_tool(furiosa_opt)]
#![deny(rustdoc::broken_intra_doc_links)]

pub mod compose;
mod module;
pub use module::{Binary, DeviceModule, Module, Unary};
pub use modules::generic::{
    Borrow, Branch, Chain, Contexts, Each, Either, First, Fork, Identity, Interleave, Layers,
    Product, ProductWithContexts, Repeat, Sequential, Stacked, Then, View,
};
pub(crate) use modules::generic::{Entry, Schedule};

pub mod models;
pub mod modules;
pub mod ops;

#[cfg(test)]
mod tests {
    // Device tests share one chip, so they run one at a time.
    pub(crate) static DEVICE: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
}
