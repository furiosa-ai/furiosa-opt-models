//! Checked prompt metadata and KV storage for the four-chip H3 text encoder.

use super::super::packing::DEFAULT_MAX_TEXT;
use super::{
    self as text,
    launch::{Cache, Row, Windows},
};
use crate::models::decoder::attention::Head;
use crate::ops::{attention::gqa, index::table::BlockIndex};
use crate::tensor::{
    cache::Block,
    memory::alias,
    parallel::{Parallel, Parallelism},
};
use furiosa_opt_std::prelude::*;
use std::{io, ops::Range};

type Chips = <Parallel<4> as Parallelism>::Chip;

/// Uploaded prompt metadata and its allocated KV backing storage.
pub struct Request {
    rows: Vec<Row>,
    image: HbmBuffer,
    cache: HbmBuffer,
}

impl Request {
    /// Borrow the rows only after the complete prompt metadata upload has finished.
    pub fn rows(&self) -> &[Row] {
        &self.rows
    }

    /// Each layer's normal SDK launch prepares this cache's indirect addresses. The Flight
    /// retains its address table and backing allocations until every reader has completed.
    pub fn cache(&self) -> io::Result<Cache> {
        Cache::from_buffer(self.cache.alias(..).map_err(io::Error::other)?)
    }
}

/// Checked host image. Alignment is shared by sizing and alias construction; no independent
/// list of byte offsets can drift from the array dimensions.
pub struct Prepared {
    image: Vec<u8>,
    layout: Layout,
}

struct Layout {
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
    fn new(rows: usize, capacity: usize) -> io::Result<Self> {
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
}

impl Prepared {
    /// Count formatted tokens before padding; refuse over-capacity and invalid IDs before any
    /// allocation on the device. Execution uses the actual count with the one maximum envelope.
    pub fn new(tokens: &[i32], capacity: usize) -> io::Result<Self> {
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
    pub async fn upload(&self, device: &mut Device) -> io::Result<Request> {
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

/// Uninitialized prompt storage held only for capacity admission; no rows are exposed.
pub struct Reservation {
    _request: Request,
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

fn invalid_tokens() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "H3 text requires nonempty in-vocabulary tokens within its configured capacity (at most 2048)",
    )
}

/// Reserve the ordinary metadata and cache allocations without exposing numerical rows.
pub fn reserve(device: &mut Device, rows: usize, capacity: usize) -> io::Result<Reservation> {
    Ok(Reservation {
        _request: Layout::new(rows, capacity)?.allocate(device)?,
    })
}
