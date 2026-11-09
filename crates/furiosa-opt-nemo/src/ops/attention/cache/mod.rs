//! Device cache storage and the layout a paged decode step reads.

mod rows;
pub mod write;

pub use rows::Scatter;

use furiosa_opt_std::prelude::*;

use super::{geometry::Kv, gqa::Cache};
use crate::ops::layout::axes::Chip;

axes![Block = 16];

#[derive(Clone, Debug)]
pub struct BlockIndex;

impl AxisName for BlockIndex {
    const NAME: Ident = Ident::new("BlockIndex");
    const SIZE: usize = Cache::SIZE / Block::SIZE;
}

/// One layer's runtime-sized cache, with each token's key and value stored together.
#[derive(DeviceSend)]
pub struct Blocks<Heads: AxisName, Dimension: AxisName, D: Scalar = bf16, Chips: M = Chip> {
    /// Interleaved key/value blocks in physical allocation order.
    pub tokens: Slice<HbmTensor<D, Chips, m![Block, Kv, Heads, Dimension]>>,
}

/// Writable layer range in a cache allocation.
pub struct Layer<'a, H: AxisName, W: AxisName, D: Scalar = bf16, Chips: M = Chip> {
    /// Allocation containing this layer's blocks.
    pub blocks: &'a mut Blocks<H, W, D, Chips>,
    /// First block of this layer's range in the allocation.
    pub entry: usize,
}
