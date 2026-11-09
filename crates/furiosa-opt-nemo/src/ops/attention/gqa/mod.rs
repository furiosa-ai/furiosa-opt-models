use furiosa_opt_std::prelude::*;

mod carried;
mod combination;
pub(crate) mod completion;
mod mass;
mod paged;
mod partition;
pub(crate) mod reduction;
pub mod score;
pub(crate) mod shard;
pub(crate) mod softmax;
pub(crate) mod value;
pub(crate) mod visibility;
mod weighting;

pub use paged::PagedAttention;

axes![
    Cache = 512, // tokens one attention window holds; histories and block tables tile by it
];
