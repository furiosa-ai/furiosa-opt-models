//! Models as trees of `Module`s: numerical operations, reusable module structures and sized model
//! definitions.

#![feature(register_tool)]
#![feature(proc_macro_hygiene, stmt_expr_attributes)]
#![feature(trait_alias)]
#![feature(adt_const_params)]
#![register_tool(furiosa_opt)]
#![deny(rustdoc::broken_intra_doc_links)]

pub mod compose;
mod module;
pub use module::{DeviceModule, Module};
pub use modules::generic::{First, Identity};

pub mod models;
pub mod modules;
pub mod ops;

#[cfg(test)]
mod tests {
    // Device tests share one chip, so they run one at a time.
    pub(crate) static DEVICE: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
}
