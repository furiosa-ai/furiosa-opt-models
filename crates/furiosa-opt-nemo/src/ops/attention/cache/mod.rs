//! Device cache storage and the layout a paged decode step reads.

use furiosa_opt_std::runtime::HbmBuffer as Buffer;

mod pool;
mod rows;
pub mod write;

pub use pool::Pool;
pub(crate) use pool::fuse;
pub use rows::{Gather, Inspect, KeyValue, Read, Scatter, Selection, Slots, Store};

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

#[cfg(test)]
pub(crate) const TEST_TOKENS: usize = 8192;

/// Token rows stored by one cache block.
pub const BLOCK: usize = Block::SIZE;

/// Number of blocks each query reads in each statically sized attention window.
pub type BlockCounts<Rows, Windows> = HbmTensor<i32, Chip, m![Windows, Rows]>;

#[cfg(test)]
pub(crate) fn block_counts<Rows: AxisName, Windows: AxisName>(
    lengths: impl IntoIterator<Item = i32>,
) -> HostTensor<i32, m![Windows, Rows]> {
    let lengths: Vec<_> = lengths.into_iter().collect();
    assert_eq!(lengths.len(), Rows::SIZE);
    HostTensor::from_vec((0..Windows::SIZE).flat_map(|window| {
        lengths.iter().map(move |&length| {
            (length.max(0) as usize)
                .saturating_sub(window * Cache::SIZE)
                .min(Cache::SIZE)
                .div_ceil(BLOCK) as i32
        })
    }))
}

/// One layer's runtime-sized cache, with each token's key and value stored together.
#[derive(DeviceSend)]
pub struct Blocks<Heads: AxisName, Dimension: AxisName, D: Scalar = bf16, Chips: M = Chip> {
    /// Interleaved key/value blocks in physical allocation order.
    pub tokens: Slice<HbmTensor<D, Chips, m![Block, Kv, Heads, Dimension]>>,
}

impl<H: AxisName, W: AxisName, D: Scalar, Chips: M> Blocks<H, W, D, Chips> {
    /// Byte stride of one interleaved key/value token row.
    pub const SLOT_BYTES: i32 = (Kv::SIZE * H::SIZE * W::SIZE * D::BITS / 8) as i32;

    /// Bytes one block of rows spans, what a table entry is scaled by.
    pub const BLOCK_BYTES: i32 = BLOCK as i32 * Self::SLOT_BYTES;

    /// Maximum block count whose byte offsets fit the device's signed index range.
    pub const MAX_BLOCKS: usize = (i32::MAX as usize + 1) / Self::BLOCK_BYTES as usize;

    /// One layer's storage split into blocks; refuse a partial or unaligned block, or more blocks
    /// than signed byte offsets reach.
    pub fn from_buffer(buffer: Buffer) -> std::io::Result<Self> {
        let tokens = crate::ops::tensor::blocks(buffer, Self::BLOCK_BYTES as usize)?;
        if tokens.len() > Self::MAX_BLOCKS {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "cache blocks exceed signed byte offsets",
            ));
        }
        Ok(Self {
            tokens: tokens.into(),
        })
    }
}

/// Writable layer range in a cache allocation.
pub struct Layer<'a, H: AxisName, W: AxisName, D: Scalar = bf16, Chips: M = Chip> {
    /// Allocation containing this layer's blocks.
    pub blocks: &'a mut Blocks<H, W, D, Chips>,
    /// First block of this layer's range in the allocation.
    pub entry: usize,
}

#[cfg(test)]
impl<H: AxisName, W: AxisName> Blocks<H, W> {
    pub(crate) async fn write(
        &mut self,
        device: &mut Device,
        blocks: std::ops::Range<usize>,
        values: impl IntoIterator<Item = bf16>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let values: Vec<_> = values.into_iter().collect();
        let width = Self::BLOCK_BYTES as usize / size_of::<bf16>();
        assert_eq!(values.len(), blocks.len() * width);
        for (block, values) in blocks.zip(values.chunks_exact(width)) {
            HostTensor::<bf16, m![Block, Kv, H, W]>::from_vec(values.iter().copied())
                .to_hbm(&mut device.pdma)
                .output(&mut self.tokens[block])
                .await?;
        }
        Ok(())
    }

    pub(crate) async fn read(
        &self,
        device: &mut Device,
        blocks: std::ops::Range<usize>,
    ) -> Result<Vec<bf16>, Box<dyn std::error::Error>> {
        let mut values = Vec::new();
        for block in blocks {
            values.extend(
                self.tokens[block]
                    .to_host::<m![Block, Kv, H, W]>(&mut device.pdma)
                    .await?
                    .into_vec(),
            );
        }
        Ok(values)
    }

    pub(crate) async fn from_rows(device: &mut Device, values: Vec<bf16>) -> Self {
        let width = Self::BLOCK_BYTES as usize / size_of::<bf16>();
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

/// Marks a token outside the visible history; masked attention assigns it zero mass.
/// Kernels compare a staged bias with zero, so it masks as the device's own `-inf` masks do.
pub const PADDING_BIAS: f32 = -1e30;

#[cfg(test)]
mod tests {
    use crate::compose::{Count, tests::count};
    use crate::ops::attention::cache::PADDING_BIAS;
    use crate::{
        Module,
        ops::{
            attention::{
                cache::{BLOCK, Block, BlockIndex, Layer, Scatter},
                geometry::Kv,
                gqa::{self, Cache, GroupedQuery, Paged},
                tests::{Blocks, Chunks, Head, KvHeads},
            },
            layout::{
                Live,
                axes::{Chip, Eight, Single, Sixteen, SixtyFour},
            },
            normalization::tests::random,
        },
    };
    use furiosa_opt_std::{prelude::*, runtime::launch};

    type PrefixGroups = crate::ops::attention::geometry::QueryGroups<4>;

    type PrefixQuery<Count> = HbmTensor<bf16, Chip, m![Count, KvHeads, PrefixGroups, Head]>;
    type PrefixOutput<Count> = HbmTensor<bf16, Chip, m![KvHeads, Count, PrefixGroups, Head]>;
    type Bias<Count> = HbmTensor<f32, Chip, m![Count, Chunks, Cache]>;

    #[device(chip = 1)]
    fn checked(
        device: &mut Device,
        query: &PrefixQuery<Eight>,
        blocks: &Blocks,
        table: &HbmTensor<i32, Chip, m![Eight, Chunks, BlockIndex]>,
        lengths: &crate::ops::attention::cache::BlockCounts<Eight, Chunks>,
        bias: &Bias<Eight>,
        rest: &Count,
        output: &mut PrefixOutput<Eight>,
    ) {
        let query: gqa::Heads<Eight, KvHeads, PrefixGroups, Head> = query.to_dm(&mut device.tdma);
        let history = Paged {
            blocks,
            entry: 0,
            table,
            lengths,
        };
        GroupedQuery::<
            Eight,
            KvHeads,
            PrefixGroups,
            Head,
            Chunks,
            { (std::f32::consts::FRAC_1_SQRT_2 / 8.0).to_bits() as usize },
        >::new()
        .forward((
            &mut *device,
            (&query, &history, bias, rest, Live::over(Eight::SIZE)),
        ))
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn attends_context_limit() {
        const CONTEXT: usize = Chunks::SIZE * Cache::SIZE;
        const LENGTHS: [usize; 8] = [40_960, 40_959, 40_449, 40_448, 513, 512, 1, 0];
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(checked.topology()).unwrap();
        let bytes = 2048 * <Blocks>::BLOCK_BYTES as usize;
        let mut layer = Blocks::from_buffer(HbmBuffer::alloc(&device, bytes).unwrap()).unwrap();
        let blocks = layer.tokens.len();
        assert!(blocks > Eight::SIZE * BlockIndex::SIZE + Eight::SIZE);
        let base = |row: usize, head: usize, channel: usize| {
            bf16::from_f32(
                ((if row == 2 { 0.0 } else { row as f32 / 8.0 }) + (head % 4) as f32 / 32.0)
                    / (1 << (channel % 4)) as f32,
            )
            .to_f32()
        };
        let end = |row: usize, head: usize, channel: usize| {
            bf16::from_f32(
                base(row, head, channel) + Chunks::SIZE as f32 / (1 << (channel % 4)) as f32,
            )
            .to_f32()
        };
        let tip = |row: usize, head: usize, channel: usize| {
            bf16::from_f32(end(row, head, channel) + CONTEXT as f32 / (1 << (channel % 4)) as f32)
                .to_f32()
        };
        for row in 0..Eight::SIZE {
            let normal: Vec<_> = (0..<m![Block, Kv, KvHeads, Head]>::SIZE)
                .map(|at| {
                    let channel = at % Head::SIZE;
                    let head = at / Head::SIZE % KvHeads::SIZE;
                    let kind = at / (Head::SIZE * KvHeads::SIZE) % Kv::SIZE;
                    let token = at / (Head::SIZE * KvHeads::SIZE * Kv::SIZE);
                    bf16::from_f32(if token >= LENGTHS[row] {
                        if channel % 2 == 0 {
                            f32::NAN
                        } else {
                            f32::INFINITY
                        }
                    } else if kind == 0 {
                        0.0
                    } else {
                        base(row, head, channel)
                    })
                })
                .collect();
            layer
                .write(&mut device, row..row + 1, normal)
                .await
                .unwrap();
            let begin = blocks - (row + 1) * BlockIndex::SIZE;
            let final_window: Vec<_> = (0..<m![Cache, Kv, KvHeads, Head]>::SIZE)
                .map(|at| {
                    let channel = at % Head::SIZE;
                    let head = at / Head::SIZE % KvHeads::SIZE;
                    let kind = at / (Head::SIZE * KvHeads::SIZE) % Kv::SIZE;
                    let token = (Chunks::SIZE - 1) * Cache::SIZE
                        + at / (Head::SIZE * KvHeads::SIZE * Kv::SIZE);
                    bf16::from_f32(if token >= LENGTHS[row] {
                        if channel % 2 == 0 {
                            f32::NAN
                        } else {
                            f32::NEG_INFINITY
                        }
                    } else if kind == 0 {
                        0.0
                    } else if token == CONTEXT - 1 {
                        tip(row, head, channel)
                    } else {
                        end(row, head, channel)
                    })
                })
                .collect();
            layer
                .write(&mut device, begin..begin + BlockIndex::SIZE, final_window)
                .await
                .unwrap();
        }
        let last = layer.read(&mut device, blocks - 1..blocks).await.unwrap();
        assert_eq!(
            last.last().unwrap().to_f32(),
            tip(0, KvHeads::SIZE - 1, Head::SIZE - 1)
        );
        let query = HostTensor::<bf16, m![Eight, KvHeads, PrefixGroups, Head]>::zero()
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let table = HostTensor::<i32, m![Eight, Chunks, BlockIndex]>::from_vec(
            (0..Eight::SIZE).flat_map(|row| {
                (0..Chunks::SIZE * BlockIndex::SIZE).map(move |entry| {
                    let block = if entry < (Chunks::SIZE - 1) * BlockIndex::SIZE {
                        row
                    } else {
                        blocks - (row + 1) * BlockIndex::SIZE + entry % BlockIndex::SIZE
                    };
                    i32::try_from(block * <Blocks>::BLOCK_BYTES as usize).unwrap()
                })
            }),
        )
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        let bias = HostTensor::<f32, m![Eight, Chunks, Cache]>::from_vec(LENGTHS.iter().flat_map(
            |&length| {
                (0..CONTEXT).map(move |token| if token < length { 0.0 } else { PADDING_BIAS })
            },
        ))
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        let rest = count(Chunks::SIZE - 1)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let mut output = HostTensor::<bf16, m![KvHeads, Eight, PrefixGroups, Head]>::zero()
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let mut previous: Option<Vec<bf16>> = None;
        for phase in 0..2 {
            let lengths = crate::ops::attention::cache::block_counts::<Eight, Chunks>(
                LENGTHS.iter().map(|&length| {
                    if phase == 0 {
                        CONTEXT as i32
                    } else {
                        length as i32
                    }
                }),
            )
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
            launch(
                checked,
                (
                    &mut device,
                    &query,
                    &layer,
                    &table,
                    &lengths,
                    &bias,
                    &rest,
                    &mut output,
                ),
            )
            .await
            .unwrap();
            let got = output
                .to_host::<m![KvHeads, Eight, PrefixGroups, Head]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            for (at, actual) in got.iter().enumerate() {
                let channel = at % Head::SIZE;
                let row = at / (Head::SIZE * PrefixGroups::SIZE) % Eight::SIZE;
                let head = at / (Head::SIZE * PrefixGroups::SIZE * Eight::SIZE);
                let length = LENGTHS[row];
                let tail = length.saturating_sub((Chunks::SIZE - 1) * Cache::SIZE);
                let expected = if length == 0 {
                    0.0
                } else {
                    ((length - tail) as f32 * base(row, head, channel)
                        + tail as f32 * end(row, head, channel)
                        + if length == CONTEXT {
                            tip(row, head, channel) - end(row, head, channel)
                        } else {
                            0.0
                        })
                        / length as f32
                };
                let error = (actual.to_f32() - expected).abs();
                assert!(
                    actual.to_f32().is_finite() && error <= 0.01 * expected.abs().max(1.0 / 128.0),
                    "phase {phase} row {row} head {head} channel {channel}: {} != {expected}",
                    actual.to_f32()
                );
            }
            if let Some(previous) = &previous {
                assert!(
                    got.iter()
                        .zip(previous)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "untransferred poisoned tails must not change the valid result"
                );
            }
            previous = Some(got);
        }
    }

    #[device(chip = 1)]
    fn write_rows<R: AxisName>(
        device: &mut Device,
        blocks: &mut Blocks,
        slot: &HbmTensor<i32, Chip, m![R]>,
        rows: &Count,
        token: &HbmTensor<bf16, Chip, m![KvHeads, R, Kv, Head]>,
    ) {
        let token: DmTensor<bf16, Chip, m![1 # 2], m![KvHeads, 1 # 32], m![R, Kv, Head]> =
            token.to_dm(&mut device.tdma);
        Scatter.forward((
            device,
            (
                &mut Layer { blocks, entry: 0 },
                (slot, Live::read(rows)),
                &token,
            ),
        ));
    }

    #[device(chip = 1)]
    fn write_lanes<R: AxisName>(
        device: &mut Device,
        blocks: &mut Blocks,
        slot: &HbmTensor<i32, Chip, m![R]>,
        rows: &Count,
        key: &HbmTensor<bf16, Chip, m![R, KvHeads, Head]>,
        value: &HbmTensor<bf16, Chip, m![R, KvHeads, Head]>,
    ) {
        let key: crate::ops::attention::Heads<R, KvHeads, Single, Head> =
            key.to_dm(&mut device.tdma);
        let value: crate::ops::attention::Heads<R, KvHeads, Single, Head> =
            value.to_dm(&mut device.tdma);
        Scatter.forward((
            device,
            (
                &mut Layer { blocks, entry: 0 },
                (slot, Live::read(rows)),
                (&key, &value),
            ),
        ));
    }

    #[device(chip = 1)]
    fn write_row(
        device: &mut Device,
        blocks: &mut Blocks,
        slot: &HbmTensor<i32, Chip, m![1]>,
        key: &HbmTensor<bf16, Chip, m![KvHeads, Head]>,
        value: &HbmTensor<bf16, Chip, m![KvHeads, Head]>,
    ) {
        Scatter.forward((
            device,
            (&mut Layer { blocks, entry: 0 }, slot, (key, value)),
        ));
    }

    /// Token slots of the pools the write tests fill.
    const TOKENS: usize = 4096;

    /// One call over an envelope of `R` rows, `live` of them live, leaves the pool one-row calls
    /// leave for its live rows, with its rows held whole by each head or spread over lanes. Each
    /// padding row's entry names a writable slot of its own, and that slot keeps its value.
    async fn writes_only_live_rows<R: AxisName>(device: &mut Device, live: usize) {
        let width = Kv::SIZE * KvHeads::SIZE * Head::SIZE;
        let values = random(R::SIZE * width, 0x3c6e_f372_fe94_f82b);
        let token = |row: usize, kv: usize, head: usize, channel: usize| {
            values[((row * Kv::SIZE + kv) * KvHeads::SIZE + head) * Head::SIZE + channel]
        };
        let slots: Vec<usize> = (0..R::SIZE)
            .map(|row| {
                if row < live {
                    (3 * row + 1) * BLOCK + row % BLOCK
                } else {
                    TOKENS - R::SIZE + row
                }
            })
            .collect();
        let offset = |slot: usize| slot as i32 * <Blocks>::SLOT_BYTES;
        let canary = || vec![bf16::from_f32(3.0); TOKENS * width];

        let mut sequential = Blocks::from_rows(device, canary()).await;
        for row in 0..live {
            let half = |kv: usize| {
                HostTensor::<bf16, m![KvHeads, Head]>::from_vec(
                    (0..KvHeads::SIZE * Head::SIZE)
                        .map(|at| token(row, kv, at / Head::SIZE, at % Head::SIZE)),
                )
            };
            let key = half(0).to_hbm(&mut device.pdma).await.unwrap();
            let value = half(1).to_hbm(&mut device.pdma).await.unwrap();
            let slot = HostTensor::<i32, m![1]>::from_vec([offset(slots[row])])
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            launch(
                write_row,
                (&mut *device, &mut sequential, &slot, &key, &value),
            )
            .await
            .unwrap();
        }

        let slot = HostTensor::<i32, m![R]>::from_vec(slots.iter().map(|&slot| offset(slot)))
            .to_hbm::<Chip, m![R]>(&mut device.pdma)
            .await
            .unwrap();
        let rows = count(live).to_hbm(&mut device.pdma).await.unwrap();
        let mut whole = Blocks::from_rows(device, canary()).await;
        let tokens = HostTensor::<bf16, m![KvHeads, R, Kv, Head]>::from_vec(
            (0..width * R::SIZE).map(|at| {
                let channel = at % Head::SIZE;
                let kv = at / Head::SIZE % Kv::SIZE;
                let row = at / (Head::SIZE * Kv::SIZE) % R::SIZE;
                token(row, kv, at / (Head::SIZE * Kv::SIZE * R::SIZE), channel)
            }),
        )
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        launch(
            write_rows,
            (&mut *device, &mut whole, &slot, &rows, &tokens),
        )
        .await
        .unwrap();
        let mut lanes = Blocks::from_rows(device, canary()).await;
        let half = |kv: usize| {
            HostTensor::<bf16, m![R, KvHeads, Head]>::from_vec(
                (0..R::SIZE * KvHeads::SIZE * Head::SIZE).map(|at| {
                    let head = at / Head::SIZE % KvHeads::SIZE;
                    token(at / (Head::SIZE * KvHeads::SIZE), kv, head, at % Head::SIZE)
                }),
            )
        };
        let key = half(0).to_hbm(&mut device.pdma).await.unwrap();
        let value = half(1).to_hbm(&mut device.pdma).await.unwrap();
        launch(
            write_lanes,
            (&mut *device, &mut lanes, &slot, &rows, &key, &value),
        )
        .await
        .unwrap();

        let blocks = 0..TOKENS / BLOCK;
        let sequential = sequential.read(device, blocks.clone()).await.unwrap();
        for (form, pool) in [("whole", whole), ("lanes", lanes)] {
            let pool = pool.read(device, blocks.clone()).await.unwrap();
            for slot in 0..TOKENS {
                let range = slot * width..(slot + 1) * width;
                assert!(
                    pool[range.clone()]
                        .iter()
                        .zip(&sequential[range])
                        .all(|(got, want)| got.to_bits() == want.to_bits()),
                    "{form}, {live} of {} rows live: slot {slot} differs",
                    R::SIZE
                );
            }
        }
    }

    /// Only live rows write, at no row, one, a partial and a whole first packet, a partial second
    /// packet, and one short of and at a whole 64-row envelope.
    #[tokio::test(flavor = "current_thread")]
    async fn writes_only_live_rows_at_every_count() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        for live in [0, 1, 7, 8] {
            writes_only_live_rows::<Eight>(&mut device, live).await;
        }
        writes_only_live_rows::<Sixteen>(&mut device, 9).await;
        for live in [63, 64] {
            writes_only_live_rows::<SixtyFour>(&mut device, live).await;
        }
    }
}
