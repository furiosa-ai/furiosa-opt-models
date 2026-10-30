//! The operands a launch's attention front reads beside its token IDs and live count, planned on
//! the host in the launch's units: each row's position and cache slot, its attention work and the
//! rows it returns.
//!
//! - A one-row request is a work item over its history in the pool; any other request is a prompt
//!   chunk over its cached history and its own rows.
//! - Slots and chunk tables are pool byte offsets, selected rows blocked hidden-row byte offsets.

use super::Batch;
use furiosa_opt_models::ops::attention::full::paged::{Plan, Tiling, chunks};
use furiosa_opt_models::ops::index::table::FIELDS;
use furiosa_opt_models::tensor::cache::{Block, Blocks};
use furiosa_opt_std::prelude::*;
use std::{io, ops::Range};

/// Bytes between two rows of the blocked hidden state, which interleaves 16 features of each row.
// ponytail: the readout's selection could take row indices and convert them itself.
const ROW_BYTES: usize = 16 * size_of::<bf16>();

/// Block IDs a work item reads in one run: a one-row request's table is padded to whole runs.
const RUN: usize = 16;

/// One request's appended rows in a call: its call rows, the positions cached before them and the
/// blocks its positions use.
pub(super) struct Appended {
    pub(super) rows: Range<usize>,
    pub(super) cached: usize,
    pub(super) blocks: Vec<u32>,
}

/// A launch's rows as the host plans them, at the capacities of `Capacity` rows, `Reach` written
/// rows and the chunk plan's chunks, each over at most `W` windows of history.
#[derive(Debug)]
struct Plans<W> {
    /// Each row's position; a padding row keeps its index.
    positions: Vec<i32>,
    /// The pool byte offset each live row's keys and values are written to.
    slots: Vec<i32>,
    /// Each one-row request's segment: its row, one row, its cached positions, the index of its
    /// first block ID in `history` and its block count.
    decode: Vec<[i32; FIELDS]>,
    /// The one-row requests' block IDs, each padded to whole runs with its last.
    history: Vec<i32>,
    chunks: chunks::Plan<W>,
    /// The blocked hidden-row byte offset of each row the launch returns, in order.
    selected: Vec<i32>,
}

impl<W: AxisName> Plans<W> {
    /// The launch of `appended`, in call row order, returning the call rows `kept`, for the cache
    /// blocks of `KvHeads` heads of `Head` features; refuses what the launch cannot hold.
    fn new<
        Capacity: AxisName,
        Reach: AxisName,
        Sequences: AxisName,
        KvHeads: AxisName,
        Head: AxisName,
    >(
        appended: &[Appended],
        kept: &[usize],
    ) -> io::Result<Self> {
        let invalid = |message| io::Error::new(io::ErrorKind::InvalidInput, message);
        let int =
            |value: usize| i32::try_from(value).map_err(|_| invalid("an operand exceeds I32"));
        let rows = appended.last().map_or(0, |request| request.rows.end);
        if rows > Capacity::SIZE {
            return Err(invalid("live rows exceed the compiled envelope"));
        }
        let (slot, table) = (
            Blocks::<KvHeads, Head>::ROW_STRIDE as usize,
            Blocks::<KvHeads, Head>::BLOCK_STRIDE as usize,
        );
        let mut plans = Self {
            positions: (0..Capacity::SIZE).map(int).collect::<io::Result<_>>()?,
            slots: vec![0; Reach::SIZE],
            decode: Vec::new(),
            history: Vec::new(),
            chunks: chunks::Plan::new::<Sequences>(),
            selected: (kept.iter())
                .map(|&row| int(row * ROW_BYTES))
                .collect::<io::Result<_>>()?,
        };
        for request in appended {
            for (row, position) in request.rows.clone().zip(request.cached..) {
                let block = request.blocks[position / Block::SIZE] as usize;
                plans.positions[row] = int(position)?;
                plans.slots[row] = int((block * Block::SIZE + position % Block::SIZE) * slot)?;
            }
            if request.rows.len() == 1 {
                let ids = (request.blocks.iter()).map(|&id| int(id as usize));
                let start = plans.history.len();
                plans.decode.push([
                    int(request.rows.start)?,
                    1,
                    int(request.cached)?,
                    int(start)?,
                    int(request.blocks.len())?,
                ]);
                plans.history.extend(ids.collect::<io::Result<Vec<_>>>()?);
                let last = plans.history[plans.history.len() - 1];
                let end = start + request.blocks.len().next_multiple_of(RUN);
                plans.history.resize(end, last);
                continue;
            }
            let chunks = &mut plans.chunks;
            let at = chunks.count;
            if at == Sequences::SIZE {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "call has more prompt requests than compiled",
                ));
            }
            let (tile, entries) = (
                chunks.lanes.len() / Sequences::SIZE,
                chunks.tables.len() / Sequences::SIZE,
            );
            if request.blocks.len() > entries {
                return Err(invalid("a prompt request's blocks exceed a chunk's table"));
            }
            chunks.q[at] = int(request.rows.len())?;
            chunks.h[at] = int(request.cached)?;
            chunks.start[at] = int(request.rows.start)?;
            chunks.lanes[at * tile..(at + 1) * tile].fill(int(request.cached)?);
            for (entry, &id) in chunks.tables[at * entries..]
                .iter_mut()
                .zip(&request.blocks)
            {
                *entry = int(id as usize * table)?;
            }
            chunks.count += 1;
        }
        Ok(plans)
    }
}

impl<A> Batch<'_, A> {
    /// The arrays a launch's attention front reads, in the order its operands bind them: positions,
    /// write slots, the work items of `KvHeads` heads grouped `G` of `Head` features in `Waves`,
    /// the prompt chunks over at most `W` windows, the count of selected rows and the selected rows
    /// of cache group `group`; each as the values the launch uses and all its words. Refuses a call
    /// the launch cannot hold before anything moves.
    pub fn operands<
        Capacity: AxisName,
        Reach: AxisName,
        Sequences: AxisName,
        KvHeads: AxisName,
        G: AxisName,
        Head: AxisName,
        Waves: AxisName,
        W: AxisName,
    >(
        &self,
        group: usize,
    ) -> io::Result<Vec<(Vec<i32>, usize)>> {
        let plans = Plans::<W>::new::<Capacity, Reach, Sequences, KvHeads, Head>(
            &self.appended[group],
            &self.kept,
        )?;
        // A step attends its prompt chunks apart from their history, so no tiling makes it batch
        // invariant.
        let items = Plan::new::<KvHeads, G, Head, Waves, Reach>(
            &plans.decode,
            &plans.history,
            Tiling::Adaptive,
        );
        let items = (items.reads()).map(|(values, words)| (values.to_vec(), words));
        let mut selected = plans.selected.clone();
        selected.resize(Capacity::SIZE, 0);
        let count = i32::try_from(plans.selected.len()).map_err(io::Error::other)?;
        let arrays: Vec<_> = [
            (plans.positions, Capacity::SIZE),
            (plans.slots, Reach::SIZE),
        ]
        .into_iter()
        .chain(items)
        .chain(plans.chunks.reads()?)
        .chain([(vec![count], 1), (selected, Capacity::SIZE)])
        .collect();
        if arrays.iter().any(|(values, words)| values.len() > *words) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "a launch operand exceeds the compiled envelope",
            ));
        }
        Ok(arrays)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    axes![
        Capacity = 64,
        Reach = 128,
        Sequences = 2,
        KvHeads = 8,
        Head = 128,
        Windows = 80
    ];

    const SLOT: i32 = Blocks::<KvHeads, Head>::ROW_STRIDE;
    const BYTES: i32 = Blocks::<KvHeads, Head>::BLOCK_STRIDE;
    const ROW: i32 = ROW_BYTES as i32;

    /// The plans of requests `(rows, cached, blocks)` in call row order, keeping `kept`.
    fn plan(
        requests: &[(Range<usize>, usize, &[u32])],
        kept: &[usize],
    ) -> io::Result<Plans<Windows>> {
        let appended: Vec<_> = (requests.iter())
            .map(|(rows, cached, blocks)| Appended {
                rows: rows.clone(),
                cached: *cached,
                blocks: blocks.to_vec(),
            })
            .collect();
        Plans::new::<Capacity, Reach, Sequences, KvHeads, Head>(&appended, kept)
    }

    /// One-row requests attend as items from row 0, each writing its next slot; a prompt request
    /// is a chunk after them, with its history, first row and a table entry per block.
    #[test]
    fn plans_items_then_chunks() -> io::Result<()> {
        let plans = plan(
            &[
                (0..1, 30, &[4, 5]),
                (1..2, 5, &[9]),
                (2..22, 40, &[1, 2, 3, 4]),
            ],
            &[0, 1, 21],
        )?;
        assert_eq!(plans.positions[..4], [30, 5, 40, 41]);
        assert_eq!(plans.positions[22], 22);
        assert_eq!(
            plans.slots[..2],
            [(5 * 16 + 14) * SLOT, (9 * 16 + 5) * SLOT]
        );
        assert_eq!(plans.slots[2], (3 * 16 + 8) * SLOT);
        assert_eq!(plans.decode, [[0, 1, 30, 0, 2], [1, 1, 5, 16, 1]]);
        let padded = |ids: &[i32]| {
            let mut table = ids.to_vec();
            table.resize(RUN, ids[ids.len() - 1]);
            table
        };
        assert_eq!(plans.history, [padded(&[4, 5]), padded(&[9])].concat());
        let chunks = &plans.chunks;
        assert_eq!(chunks.count, 1);
        assert_eq!((chunks.q[0], chunks.h[0], chunks.start[0]), (20, 40, 2));
        assert_eq!(
            chunks.tables[..5],
            [BYTES, 2 * BYTES, 3 * BYTES, 4 * BYTES, 0]
        );
        assert!(
            chunks.lanes[..chunks.lanes.len() / 2]
                .iter()
                .all(|&h| h == 40)
        );
        assert_eq!(plans.selected, [0, ROW, 21 * ROW]);
        Ok(())
    }

    /// The selected rows follow the kept order, not the call rows' order.
    #[test]
    fn selects_rows_in_kept_order() -> io::Result<()> {
        let plans = plan(&[(0..1, 3, &[7]), (1..4, 0, &[2])], &[3, 0])?;
        assert_eq!(plans.selected, [3 * ROW, 0]);
        Ok(())
    }

    /// A call that keeps no row selects none and still plans its rows.
    #[test]
    fn plans_an_empty_selection() -> io::Result<()> {
        let plans = plan(&[(0..3, 0, &[2])], &[])?;
        assert!(plans.selected.is_empty());
        assert_eq!(
            (plans.chunks.count, plans.positions[..3].to_vec()),
            (1, vec![0, 1, 2])
        );
        Ok(())
    }

    /// A call past the launch's rows or prompt chunks is refused with the bound's own message.
    #[test]
    fn refuses_what_a_launch_cannot_hold() {
        let rows = plan(&[(0..65, 0, &[1, 2, 3, 4, 5])], &[]).unwrap_err();
        assert_eq!(rows.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(rows.to_string(), "live rows exceed the compiled envelope");
        let three = [
            (0..2, 0, &[1][..]),
            (2..4, 0, &[2][..]),
            (4..6, 0, &[3][..]),
        ];
        let chunks = plan(&three, &[]).unwrap_err();
        assert_eq!(chunks.kind(), io::ErrorKind::Unsupported);
        assert_eq!(
            chunks.to_string(),
            "call has more prompt requests than compiled"
        );
    }
}
