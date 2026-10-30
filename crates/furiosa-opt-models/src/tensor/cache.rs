//! The KV cache layout: blocks of token rows, each token's key and value stored together, and the
//! bytes they span.

use super::parallel::Chip;
use crate::tensor::memory::blocks;
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::HbmBuffer as Buffer;
use std::io;
#[cfg(test)]
use std::{error::Error, ops::Range};

axes![
    Block = 16,   // token rows one cache block stores
    KeyValue = 2, // the key and the value each token stores
];

/// A cache that stores each key and value in BF16.
pub type Bf16 = bf16;

/// One layer's runtime-sized cache, with each token's key and value stored together.
#[derive(DeviceSend)]
pub struct Blocks<Heads: AxisName, Dimension: AxisName, D: Scalar = bf16, Chips: M = Chip> {
    /// Interleaved key/value blocks in physical allocation order.
    pub tokens: Slice<HbmTensor<D, Chips, m![Block, KeyValue, Heads, Dimension]>>,
}

impl<Heads: AxisName, Dimension: AxisName, D: Scalar, Chips: M> Blocks<Heads, Dimension, D, Chips> {
    /// Byte stride of one interleaved key/value token row.
    pub const ROW_STRIDE: i32 =
        (KeyValue::SIZE * Heads::SIZE * Dimension::SIZE * D::BITS / 8) as i32;

    /// Byte stride of one block of rows, what a table entry is scaled by.
    pub const BLOCK_STRIDE: i32 = Block::SIZE as i32 * Self::ROW_STRIDE;

    /// Maximum block count whose byte offsets fit the device's signed index range.
    pub const MAX_BLOCKS: usize = (i32::MAX as usize + 1) / Self::BLOCK_STRIDE as usize;

    /// One layer's storage split into blocks; refuse a partial or unaligned block, or more blocks
    /// than signed byte offsets reach.
    pub fn from_buffer(buffer: Buffer) -> io::Result<Self> {
        let tokens = blocks(buffer, Self::BLOCK_STRIDE as usize)?;
        if tokens.len() > Self::MAX_BLOCKS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cache blocks exceed signed byte offsets",
            ));
        }
        Ok(Self {
            tokens: tokens.into(),
        })
    }
}

#[cfg(test)]
impl<Heads: AxisName, Dimension: AxisName> Blocks<Heads, Dimension> {
    pub(crate) async fn write(
        &mut self,
        device: &mut Device,
        blocks: Range<usize>,
        values: impl IntoIterator<Item = bf16>,
    ) -> Result<(), Box<dyn Error>> {
        let values: Vec<_> = values.into_iter().collect();
        let width = Self::BLOCK_STRIDE as usize / size_of::<bf16>();
        assert_eq!(values.len(), blocks.len() * width);
        for (block, values) in blocks.zip(values.chunks_exact(width)) {
            HostTensor::<bf16, m![Block, KeyValue, Heads, Dimension]>::from_vec(
                values.iter().copied(),
            )
            .to_hbm(&mut device.pdma)
            .output(&mut self.tokens[block])
            .await?;
        }
        Ok(())
    }

    pub(crate) async fn read(
        &self,
        device: &mut Device,
        blocks: Range<usize>,
    ) -> Result<Vec<bf16>, Box<dyn Error>> {
        let mut values = Vec::new();
        for block in blocks {
            values.extend(
                self.tokens[block]
                    .to_host::<m![Block, KeyValue, Heads, Dimension]>(&mut device.pdma)
                    .await?
                    .into_vec(),
            );
        }
        Ok(values)
    }

    pub(crate) async fn from_rows(device: &mut Device, values: Vec<bf16>) -> Self {
        let width = Self::BLOCK_STRIDE as usize / size_of::<bf16>();
        assert!(values.len().is_multiple_of(width));
        let mut blocks =
            Self::from_buffer(Buffer::alloc(device, values.len() * size_of::<bf16>()).unwrap())
                .unwrap();
        blocks
            .write(device, 0..values.len() / width, values)
            .await
            .unwrap();
        blocks
    }
}
