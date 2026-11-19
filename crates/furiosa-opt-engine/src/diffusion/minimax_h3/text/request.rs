//! One prompt's positions, cache addresses, visibility and IDs in a single replicated transfer.

use super::{Cache, Chips, DEFAULT_MAX_TEXT, Head, Hidden, Windows, text};
use furiosa_opt_models::ops::{attention::gqa, index::table::BlockIndex};
use furiosa_opt_models::tensor::{axis::Single, cache::Block, memory::alias};
use furiosa_opt_std::prelude::*;
use std::{io, ops::Range};

/// A row's metadata aliases the prompt's one allocation. Every table entry, including rounded
/// overfetch, addresses allocated cache storage; the causal bias hides unfilled token rows.
#[derive(DeviceSend)]
pub(super) struct Row {
    pub position: HbmTensor<f32, Chips, m![Single, Head]>,
    pub slot: HbmTensor<i32, Chips, m![1]>,
    pub history: gqa::Read<Windows, Chips>,
    pub token: HbmTensor<i32, Chips, m![1]>,
}

pub(super) struct Request {
    pub rows: Vec<Row>,
    image: HbmBuffer,
    cache: HbmBuffer,
}

impl Request {
    /// Each layer's normal SDK launch prepares this cache's indirect addresses. The Flight
    /// retains its address table and backing allocations until every reader has completed.
    pub(super) fn cache(&self) -> io::Result<Cache> {
        Cache::from_buffer(self.cache.alias(..).map_err(io::Error::other)?)
    }
}

/// Checked host image. Alignment is shared by sizing and alias construction; no independent
/// list of byte offsets can drift from the array dimensions.
pub(super) struct Prepared {
    image: Vec<u8>,
    layout: Layout,
}

pub(super) struct Layout {
    position: Range<usize>,
    slot: Range<usize>,
    blocks: Range<usize>,
    lengths: Range<usize>,
    bias: Range<usize>,
    rest: Range<usize>,
    token: Range<usize>,
    stride: usize,
    rows: usize,
    cache_bytes: usize,
}

impl Layout {
    pub(super) fn new(rows: usize, capacity: usize) -> io::Result<Self> {
        if !(1..=DEFAULT_MAX_TEXT).contains(&capacity) || rows == 0 || rows > capacity {
            return Err(invalid_tokens());
        }
        let mut end = 0_usize;
        let mut take = |words: usize| {
            let start = end.next_multiple_of(8);
            end = start + words * size_of::<i32>();
            start..end
        };
        let position = take(Head::SIZE);
        let slot = take(1);
        let blocks = take(Windows::SIZE * BlockIndex::SIZE);
        let lengths = take(Windows::SIZE);
        let bias = take(Windows::SIZE * gqa::Cache::SIZE);
        let rest = take(1);
        let token = take(1);
        Ok(Self {
            position,
            slot,
            blocks,
            lengths,
            bias,
            rest,
            token,
            stride: end.next_multiple_of(8),
            rows,
            cache_bytes: rows.div_ceil(Block::SIZE) * Cache::BLOCK_STRIDE as usize,
        })
    }

    /// Allocate the same metadata and cache storage for real upload and capacity admission.
    /// Contents become readable only after Prepared::upload completes initialization.
    fn allocate(&self, device: &mut Device) -> io::Result<Request> {
        let image = HbmBuffer::alloc(device, self.rows * self.stride).map_err(io::Error::other)?;
        let cache = HbmBuffer::alloc(device, self.cache_bytes).map_err(io::Error::other)?;
        let rows = (0..self.rows)
            .map(|at| {
                let shifted = |range: &Range<usize>| {
                    at * self.stride + range.start..at * self.stride + range.end
                };
                Ok(Row {
                    position: alias(&image, shifted(&self.position))?,
                    slot: alias(&image, shifted(&self.slot))?,
                    history: gqa::Read {
                        blocks: alias(&image, shifted(&self.blocks))?,
                        lengths: alias(&image, shifted(&self.lengths))?,
                        bias: alias(&image, shifted(&self.bias))?,
                        rest: alias(&image, shifted(&self.rest))?,
                    },
                    token: alias(&image, shifted(&self.token))?,
                })
            })
            .collect::<io::Result<_>>()?;
        Ok(Request { rows, image, cache })
    }

    /// Reserve the serving request, its indirect address table and both coexisting row banks.
    /// No token payload transfer or numerical execution is needed for capacity admission.
    pub(super) fn workspace_bytes(&self, device: &mut Device) -> io::Result<usize> {
        super::super::memory::measure(device, |device| {
            let request = self.allocate(device)?;
            // Blocks::from_buffer splits one allocation into equal contiguous blocks. The
            // pinned SDK's tensor/memory/slice/table.rs uses a 64-byte strided header for that
            // representation. Reserve it with the same local-DRAM allocator as Slice::prepare;
            // actual calls initialize the table through their normal SDK launch, not a warm-up.
            let table = HbmBuffer::alloc(device, CACHE_TABLE_BYTES).map_err(io::Error::other)?;
            let rows = (0..2 * self.rows)
                .map(|_| {
                    <Hidden as furiosa_opt_std::runtime::DeviceOutput>::alloc(device)
                        .map(Hidden::into_buffer)
                        .map_err(io::Error::other)
                })
                .collect::<io::Result<Vec<_>>>()?;
            Ok((request, table, rows))
        })
    }
}

// The strided Slice address-table header of the SDK revision pinned by Cargo.toml.
const CACHE_TABLE_BYTES: usize = 64;

impl Prepared {
    /// Count formatted tokens before padding; refuse over-capacity and invalid IDs before any
    /// allocation on the device. Execution uses the actual count with the one maximum envelope.
    pub(super) fn new(tokens: &[i32], capacity: usize) -> io::Result<Self> {
        let layout = Layout::new(tokens.len(), capacity)?;
        if tokens
            .iter()
            .any(|&id| id < 0 || id as usize >= text::Vocabulary::SIZE)
        {
            return Err(invalid_tokens());
        }
        let rows = tokens.len();
        let blocks = rows.div_ceil(Block::SIZE);
        let mut image = vec![0; rows * layout.stride];
        for (position, (&id, row)) in tokens
            .iter()
            .zip(image.chunks_exact_mut(layout.stride))
            .enumerate()
        {
            let write =
                |row: &mut [u8], range: Range<usize>, values: &mut dyn Iterator<Item = [u8; 4]>| {
                    for (word, value) in row[range].chunks_exact_mut(4).zip(values) {
                        word.copy_from_slice(&value);
                    }
                };
            write(
                row,
                layout.position.clone(),
                &mut std::iter::repeat((position as f32).to_le_bytes()),
            );
            write(
                row,
                layout.slot.clone(),
                &mut std::iter::once((position as i32 * Cache::ROW_STRIDE).to_le_bytes()),
            );
            write(
                row,
                layout.blocks.clone(),
                &mut (0..Windows::SIZE * BlockIndex::SIZE).map(|at| {
                    let offset = if at < blocks {
                        at as i32 * Cache::BLOCK_STRIDE
                    } else {
                        0
                    };
                    offset.to_le_bytes()
                }),
            );
            write(
                row,
                layout.lengths.clone(),
                &mut (0..Windows::SIZE).map(|window| {
                    let count = (position + 1)
                        .saturating_sub(window * gqa::Cache::SIZE)
                        .min(gqa::Cache::SIZE);
                    (count.div_ceil(Block::SIZE) as i32).to_le_bytes()
                }),
            );
            write(
                row,
                layout.bias.clone(),
                &mut (0..Windows::SIZE * gqa::Cache::SIZE).map(|at| {
                    (if at <= position {
                        0.0_f32
                    } else {
                        f32::NEG_INFINITY
                    })
                    .to_le_bytes()
                }),
            );
            write(
                row,
                layout.rest.clone(),
                &mut std::iter::once(((position / gqa::Cache::SIZE) as i32).to_le_bytes()),
            );
            write(
                row,
                layout.token.clone(),
                &mut std::iter::once(id.to_le_bytes()),
            );
        }
        Ok(Self { image, layout })
    }

    /// Initial request setup lands one metadata image before submitting readers. Each row and
    /// the returned request retain aliases of the image, independent of this host staging owner.
    pub(super) async fn upload(&self, device: &mut Device) -> io::Result<Request> {
        let mut request = self.layout.allocate(device)?;
        let copies = self.image.repeat(Chips::SIZE);
        // The first layer reads whole cache blocks even for a partial causal prefix. Initialize
        // future rows too: masking cannot make arbitrary NaNs from unwritten keys safe.
        let zeros = vec![0; self.layout.cache_bytes * Chips::SIZE];
        let target = request.image.slice_mut(.., ..).map_err(io::Error::other)?;
        let cache_target = request.cache.slice_mut(.., ..).map_err(io::Error::other)?;
        device
            .pdma
            .to_hbm([
                (HostBufferSlice::from(&copies[..]), target),
                (HostBufferSlice::from(&zeros[..]), cache_target),
            ])
            .await
            .map_err(io::Error::other)?;
        Ok(request)
    }
}

fn invalid_tokens() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "H3 text requires nonempty in-vocabulary tokens within its configured capacity (at most 2048)",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_formatted_tokens_before_device_allocation() {
        for tokens in [
            vec![0],
            vec![(text::Vocabulary::SIZE - 1) as i32; DEFAULT_MAX_TEXT],
        ] {
            assert!(Prepared::new(&tokens, DEFAULT_MAX_TEXT).is_ok());
        }
        for (tokens, capacity) in [
            (vec![], DEFAULT_MAX_TEXT),
            (vec![-1], DEFAULT_MAX_TEXT),
            (vec![text::Vocabulary::SIZE as i32], DEFAULT_MAX_TEXT),
            (vec![0; DEFAULT_MAX_TEXT + 1], DEFAULT_MAX_TEXT),
            (vec![0, 1], 1),
            (vec![0], 0),
            (vec![0], DEFAULT_MAX_TEXT + 1),
        ] {
            let error = Prepared::new(&tokens, capacity)
                .err()
                .expect("invalid text accepted");
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        }
    }
}
