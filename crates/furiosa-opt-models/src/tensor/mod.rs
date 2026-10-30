//! The tensor vocabulary the ops share: row axes, chip parallelism, mappings between layouts,
//! broadcast scalars, device storage, the KV cache layout and the live rows of a launch.

#![deny(clippy::arbitrary_source_item_ordering)]

pub mod axis;
pub mod cache;
pub mod form;
pub mod image;
pub mod live;
pub mod mapping;
pub mod memory;
pub mod parallel;
pub mod scalar;
