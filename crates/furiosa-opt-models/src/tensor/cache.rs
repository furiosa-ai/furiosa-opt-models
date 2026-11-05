//! The KV cache layout: blocks of token rows, each token's key and value stored together, and the
//! bytes they span; and the static scales of a cache that stores E4M3 codes.

use super::parallel::Chip;
use super::scalar::Copies;
use crate::Op;
use crate::load::Load;
use crate::tensor::memory::{Hbm, Memory, blocks};
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::HbmBuffer as Buffer;
use std::borrow::Borrow;
use std::io;
#[cfg(test)]
use std::{error::Error, ops::Range};

axes![
    Block = 16,   // token rows one cache block stores
    KeyValue = 2, // the key and the value each token stores
    Slices = 512, // the slices of a chip's two clusters
];

/// A cache that stores each key and value in BF16.
pub type Bf16 = bf16;

/// A cache that stores each key and value as an E4M3 code under its layer's [`Scales`].
pub type E4m3 = f8e4m3;

/// A layer's static key and value scales, as a checkpoint whose cache holds E4M3 codes stores
/// them: a cached key is `e4m3(k / k_scale)`, saturated, read back as `code * k_scale`, a value
/// alike. HBM holds each a copy a slice, so a landing into every slice reads a copy of its own.
#[derive(DeviceSend, Load)]
pub struct Scales<X: Memory = Hbm> {
    pub k_scale: X::Tensor<f32, Chip, Copies>,
    pub v_scale: X::Tensor<f32, Chip, Copies>,
}

/// A layer's [`Scales`] landed once in every slice of both clusters, where the write that codes
/// its rows and the attention that reads its codes each take them in their own layout.
pub struct Landed {
    key: Spread,
    value: Spread,
}

/// An FP32 scale's packet in every slice.
type Spread = DmTensor<f32, Chip, m![Slices / 256], m![Slices % 256], m![1 # 8]>;

impl Default for Landed {
    fn default() -> Self {
        Self {
            key: DmTensor::new(),
            value: DmTensor::new(),
        }
    }
}

impl Landed {
    /// The key and value scales in the registers of a stream over `Cluster` and `Slice`, which
    /// span both clusters and their 256 slices.
    pub fn registers<Cluster: M, Slice: M>(
        &self,
        device: &mut Device,
    ) -> (
        VrfTensor<f32, Chip, Cluster, Slice, m![1 # 8]>,
        VrfTensor<f32, Chip, Cluster, Slice, m![1 # 8]>,
    ) {
        let mut register = |scale: &Spread| {
            // SAFETY: every slice of both clusters holds the same packet, so a mapping of the
            // slices in any order reads it in each one.
            let packet = unsafe { scale.view().reshape::<Chip, Cluster, Slice, m![1 # 8]>() };
            (device.sub.begin(packet))
                .fetch::<m![1], m![1 # 8]>()
                .collect::<m![1], m![1 # 8]>()
                .to_vrf()
        };
        (register(&self.key), register(&self.value))
    }
}

/// The scales landed into every slice of `landed`, one transfer each.
impl<X: Memory> Op<&mut Landed> for Scales<X>
where
    X::Tensor<f32, Chip, Copies>: Borrow<HbmTensor<f32, Chip, Copies>>,
{
    type Output = ();

    fn forward(&self, device: &mut Device, landed: &mut Landed) {
        let tdma = &mut device.tdma;
        (self.k_scale.borrow().view()).to_dm_view(tdma, landed.key.view_mut());
        (self.v_scale.borrow().view()).to_dm_view(tdma, landed.value.view_mut());
    }
}

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
