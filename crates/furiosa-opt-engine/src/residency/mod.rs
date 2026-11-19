//! Completion tracking and reusable residency for typed model weight trees.
//!
//! Host sources, pinned images, optional device copies, and scoped readers have separate owners.

pub use context::{DeviceContext, SharedDevice};
pub use work::{Completion, block_on};

pub(crate) use cell::Cells;
pub(crate) use resident::{Launches, Resident};

mod cell;
mod context;
mod gate;
mod parameter;
mod resident;
mod source;
mod upload;
mod work;
