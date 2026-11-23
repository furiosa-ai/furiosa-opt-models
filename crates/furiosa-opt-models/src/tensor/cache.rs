//! The KV cache layout: blocks of token rows, each token's key and value stored together, the
//! bytes they span, and one layer's range of a cache that holds several.

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

/// A runtime-sized cache of one layer or of several, each layer's blocks an equal range after the
/// previous layer's, with each token's key and value stored together.
#[derive(DeviceSend)]
pub struct Blocks<Heads: AxisName, Dimension: AxisName, D: Scalar = bf16, Chips: M = Chip> {
    /// Interleaved key/value blocks in physical allocation order, layer by layer.
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

    /// The layers' storages split into blocks, one layer's after another; refuse a partial or
    /// unaligned block, layers of unequal blocks, or more blocks a layer than signed byte offsets
    /// reach.
    pub fn from_buffers(layers: impl IntoIterator<Item = Buffer>) -> io::Result<Self> {
        let layers = (layers.into_iter())
            .map(|layer| blocks(layer, Self::BLOCK_STRIDE as usize))
            .collect::<io::Result<Vec<Vec<_>>>>()?;
        let invalid = |message| io::Error::new(io::ErrorKind::InvalidInput, message);
        let entries = layers.first().map_or(0, Vec::len);
        if layers.iter().any(|layer| layer.len() != entries) {
            return Err(invalid("cache layers hold unequal blocks"));
        }
        if entries > Self::MAX_BLOCKS {
            return Err(invalid("cache blocks exceed signed byte offsets"));
        }
        Ok(Self {
            tokens: layers.into_iter().flatten().collect(),
        })
    }
}

/// One layer's range of a cache's blocks: the blocks, borrowed as `B`, and the range's first
/// entry, which a write or read through it addresses its byte offsets from.
#[derive(Clone, Copy, Debug)]
pub struct Layer<B> {
    pub blocks: B,
    pub entry: usize,
}

impl<Heads: AxisName, Dimension: AxisName, D: Scalar, Chips: M>
    Layer<&mut Blocks<Heads, Dimension, D, Chips>>
{
    /// The same range, to read.
    pub fn read(&self) -> Layer<&Blocks<Heads, Dimension, D, Chips>> {
        Layer {
            blocks: &*self.blocks,
            entry: self.entry,
        }
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
            Self::from_buffers([Buffer::alloc(device, values.len() * size_of::<bf16>()).unwrap()])
                .unwrap();
        blocks
            .write(device, 0..values.len() / width, values)
            .await
            .unwrap();
        blocks
    }
}
