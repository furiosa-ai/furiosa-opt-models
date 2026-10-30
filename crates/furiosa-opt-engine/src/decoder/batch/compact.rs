//! A call's compact history: each request's call rows, cached positions and block table, in the
//! I32 words its device preparation reads.

use furiosa_opt_models::ops::index::table::BlockIndex;
use furiosa_opt_models::ops::index::table::FIELDS;
use furiosa_opt_models::tensor::cache::Block;
use furiosa_opt_std::prelude::AxisName;
use std::{io, ops::Range};

/// One request in a call: its call rows, the positions cached before them and the blocks they
/// use.
pub(super) struct Segment<'a> {
    pub(super) rows: Range<usize>,
    pub(super) cached: usize,
    pub(super) blocks: &'a [u32],
}

/// How far into a call's block table its device preparation reads.
#[derive(Clone, Copy, Debug)]
pub enum Table {
    /// Up to the table's capacity, so the words past the call's blocks are zeros.
    Capacity,
    /// Only the blocks the call names.
    Blocks,
}

#[derive(Debug)]
pub(super) struct Descriptor {
    /// Each segment's first call row, rows, cached positions, table start and table length.
    segments: Vec<[i32; FIELDS]>,
    /// Each segment's block IDs from a whole table row, the last one repeated to its end.
    blocks: Vec<i32>,
    /// The call rows the call returns, ascending.
    selected: Vec<i32>,
}

impl Descriptor {
    /// The segments in call order, and the rows the call returns; refuses segments that do not
    /// cover the call's rows from row 0 one after another, or values past I32.
    pub(super) fn encode(segments: &[Segment<'_>], selected: &[usize]) -> io::Result<Self> {
        let invalid = |message| io::Error::new(io::ErrorKind::InvalidInput, message);
        let int = |value| i32::try_from(value).map_err(|_| invalid("descriptor exceeds I32"));
        let mut descriptor = Self {
            segments: Vec::with_capacity(segments.len()),
            blocks: Vec::new(),
            selected: selected
                .iter()
                .copied()
                .map(int)
                .collect::<io::Result<_>>()?,
        };
        let mut end = 0;
        for segment in segments {
            if segment.rows.start != end || segment.rows.end <= end || segment.blocks.is_empty() {
                return Err(invalid(
                    "segments are not a contiguous nonempty prefix with tables",
                ));
            }
            let len = segment.rows.len();
            int(segment.rows.end)?;
            int((segment.cached.checked_add(len)).ok_or_else(|| invalid("history overflows"))?)?;
            let start = descriptor.blocks.len();
            let storage = (segment.blocks.len())
                .checked_next_multiple_of(BlockIndex::SIZE)
                .ok_or_else(|| invalid("table padding overflows"))?;
            int((start.checked_add(storage)).ok_or_else(|| invalid("table overflows"))?)?;
            descriptor.segments.push([
                int(end)?,
                int(len)?,
                int(segment.cached)?,
                int(start)?,
                int(segment.blocks.len())?,
            ]);
            let blocks = (segment.blocks.iter().copied())
                .map(|id| i32::try_from(id).map_err(|_| invalid("block ID exceeds I32")));
            descriptor
                .blocks
                .extend(blocks.collect::<io::Result<Vec<_>>>()?);
            let last = descriptor.blocks[descriptor.blocks.len() - 1];
            descriptor.blocks.resize(start + storage, last);
            end = segment.rows.end;
        }
        Ok(descriptor)
    }

    /// The call's rows.
    pub(super) fn rows(&self) -> usize {
        self.segments.last().map_or(0, |&[start, len, ..]| {
            usize::try_from(start + len).unwrap_or_else(|_| unreachable!("rows are positive"))
        })
    }

    /// Each call row as a segment of its own, whose history is its request's cached positions
    /// and the call's rows before it, and the block IDs the segments name.
    pub(super) fn row_segments(&self) -> (Vec<[i32; FIELDS]>, &[i32]) {
        let block = i32::try_from(Block::SIZE).unwrap_or_else(|_| unreachable!());
        let rows = (self.segments.iter()).flat_map(|&[start, len, cached, table, _]| {
            (0..len).map(move |row| {
                let keys = cached + row + 1;
                [start + row, 1, keys - 1, table, (keys + block - 1) / block]
            })
        });
        (rows.collect(), &self.blocks)
    }

    /// The segments, the block IDs and the selected rows at the capacities of `S` segments, `B`
    /// blocks (the words `table` reads) and `R` rows, then the three extents.
    pub(super) fn arrays<S: AxisName, B: AxisName, R: AxisName>(
        &self,
        table: Table,
    ) -> io::Result<[Vec<i32>; 4]> {
        let invalid = |message| io::Error::new(io::ErrorKind::InvalidInput, message);
        let pad = |mut values: Vec<i32>, capacity| {
            if values.len() > capacity {
                return Err(invalid("descriptor exceeds the compiled envelope"));
            }
            values.resize(capacity, 0);
            Ok(values)
        };
        let blocks = match table {
            Table::Capacity => B::SIZE,
            Table::Blocks => self.blocks.len().min(B::SIZE),
        };
        let extents = [self.segments.len(), self.blocks.len(), self.selected.len()];
        Ok([
            pad(
                self.segments.iter().flatten().copied().collect(),
                S::SIZE * FIELDS,
            )?,
            pad(self.blocks.clone(), blocks)?,
            pad(self.selected.clone(), R::SIZE)?,
            (extents.into_iter())
                .map(|len| i32::try_from(len).map_err(|_| invalid("descriptor extent exceeds I32")))
                .collect::<io::Result<_>>()?,
        ])
    }
}
