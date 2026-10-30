//! A launch's wide rows as work items, planned on the host from the rows' compact history.
//!
//! - A row of `keys` keys holds `ceil(keys / 16)` blocks and takes `ceil(blocks / 64 C)`
//!   consecutive items; `C` columns serve an item, `16 / C` items a wave.
//! - Adaptively, `C` is the least that holds the longest row in four blocks a slice, widened while
//!   the rows leave slices idle, at most half the launch's columns over the rows: the columns of
//!   all items then fill at most the launch's waves. A fixed tiling takes [`FIXED`] columns.
//! - A column's tables name the pool byte offset of every block it fills, how many keys it sees,
//!   and where its row's query and output rows and its item's partials lie.
//! - A launch uses its tables' rows of its waves rounded up to a pair, its jobs' rows of its merge
//!   tiles, the key positions only with waves and the merge inputs only with merges, and no other
//!   word; the columns of a wave past the last are columns of no item.

use super::super::plan::Tiling;
use super::*;
use crate::ops::index::table::FIELDS;
use crate::ops::index::table::{CACHED, LEN, START, TABLE};
use crate::tensor::cache::Blocks;
use std::mem;

/// `log2` of a fixed tiling's columns an item: four, 4,096 keys. Of the constant depths, its
/// slowest call is the least slower than the adaptive tiling's.
const FIXED: i32 = 2;

/// The arrays of a launch's wide rows, which [`Wide::bind`] takes in turn.
pub(in super::super) const ARRAYS: usize = 16;

/// `log2` of the blocks an item holds a column, 16 quads of four.
const ITEM_SHIFT: i32 = (Quads::SIZE * Quad::SIZE).ilog2() as i32;
const BLOCK_SHIFT: i32 = Block::SIZE.ilog2() as i32;
/// Byte stride of a quad of block IDs.
const QUAD_STRIDE: i32 = (Quad::SIZE * size_of::<i32>()) as i32;
/// The columns of a flit's eight cells, one value each.
const FLIT: usize = 8;
/// The waves whose queries and rows the item body moves in one transfer.
const PAIR: usize = 2;

/// A launch's wide rows as the arrays the wide body reads, each the leading words the launch uses
/// of its compiled capacity.
pub(in super::super) struct Plan {
    images: [Vec<i32>; ARRAYS],
    words: [usize; ARRAYS],
}

/// How rows of `keys` keys are laid out: each row's blocks and items, the columns an item takes
/// and the waves they fill.
struct Layout {
    blocks: Vec<i32>,
    longest: i32,
    depth: i32,
    width: i32,
    groups: i32,
    counts: Vec<i32>,
    waves: i32,
}

impl Layout {
    /// The layout of rows of `keys` keys over `Waves` waves of the wide domain, tiled as `tiling`
    /// says.
    fn new<Waves: AxisName>(keys: &[i32], tiling: Tiling) -> Self {
        let blocks: Vec<i32> = keys.iter().map(|&keys| Self::blocks(keys)).collect();
        let ceil_log2 = |x: i32| (0..16).filter(|bit| (1 << bit) < x).count() as i32;
        let longest = blocks.iter().copied().max().unwrap_or(0);
        let (reach, spread) = (ceil_log2(longest), ceil_log2(keys.len() as i32));
        let half = Waves::SIZE * Column::SIZE / 2;
        let widened = ((reach - 6).clamp(0, 4) + spread)
            .max(reach.min(4))
            .min(half.ilog2() as i32);
        let depth = match tiling {
            Tiling::Adaptive => (widened - spread).min(4),
            Tiling::Fixed => FIXED,
        };
        let (width, groups) = (1 << depth, Column::SIZE as i32 >> depth);
        let counts: Vec<i32> = blocks
            .iter()
            .map(|&blocks| Self::items(blocks, depth))
            .collect();
        let items: i32 = counts.iter().sum();
        let waves = Self::waves(items, depth);
        Self {
            blocks,
            longest,
            depth,
            width,
            groups,
            counts,
            waves,
        }
    }

    /// The blocks of a row of `keys` keys.
    const fn blocks(keys: i32) -> i32 {
        (keys + (1 << BLOCK_SHIFT) - 1) >> BLOCK_SHIFT
    }

    /// The items a row of `blocks` blocks takes at `depth`: `64 << depth` blocks an item.
    const fn items(blocks: i32, depth: i32) -> i32 {
        let groups = Column::SIZE as i32 >> depth;
        (blocks * groups + (1 << (ITEM_SHIFT + 4)) - 1) >> (ITEM_SHIFT + 4)
    }

    /// The waves `items` items of `1 << depth` columns fill.
    const fn waves(items: i32, depth: i32) -> i32 {
        ((items << depth) + Column::SIZE as i32 - 1) / Column::SIZE as i32
    }

    /// Whether `capacity` waves, used in pairs, and the item slots hold `waves` waves of `items`
    /// items.
    const fn fits(waves: i32, items: i32, capacity: usize) -> bool {
        (waves as usize).next_multiple_of(PAIR) <= capacity && items as usize <= Columns::SIZE
    }
}

impl Plan {
    /// Byte stride of an item's slot of f32 outputs, and of log-sum-exps.
    pub(crate) const fn partial_stride<KvHeads: AxisName, G: AxisName, Head: AxisName>() -> i32 {
        (<m![KvHeads, G, Head]>::SIZE * size_of::<f32>()) as i32
    }

    pub(crate) const fn scale_stride<KvHeads: AxisName, G: AxisName>() -> i32 {
        (<m![KvHeads, G, Lane]>::SIZE * size_of::<f32>()) as i32
    }

    /// The most waves a launch of at most `rows` rows of at most `keys` keys uses: a wave's 16
    /// columns hold 64 blocks of a row each at any column count, and a launch uses whole pairs.
    pub(in super::super) const fn most(rows: usize, keys: usize) -> usize {
        (rows * keys.div_ceil(Column::SIZE * Quads::SIZE * Quad::SIZE * Block::SIZE))
            .next_multiple_of(PAIR)
    }

    /// Whether `rows` wide rows of at most `keys` keys each fit `waves` waves and the item slots,
    /// tiled fixed.
    pub(in super::super) const fn fixes(rows: usize, keys: usize, waves: usize) -> bool {
        let items = rows as i32 * Layout::items(Layout::blocks(keys as i32), FIXED);
        rows <= Decodes::SIZE && Layout::fits(Layout::waves(items, FIXED), items, waves)
    }

    /// The waves of `Waves` wide rows `segments` fill, tiled adaptively.
    pub(in super::super) fn waves<Waves: AxisName>(segments: &[&[i32; FIELDS]]) -> usize {
        let keys: Vec<i32> = segments.iter().map(|row| row[LEN] + row[CACHED]).collect();
        Layout::new::<Waves>(&keys, Tiling::Adaptive).waves as usize
    }

    /// The items of wide rows `segments` with block IDs `ids`, each row padded to whole runs, tiled
    /// as `tiling` says over `Waves` waves; `Reach` holds the attended rows, the last 16 the sinks.
    pub(in super::super) fn new<
        KvHeads: AxisName,
        G: AxisName,
        Head: AxisName,
        Waves: AxisName,
        Reach: AxisName,
    >(
        segments: &[&[i32; FIELDS]],
        ids: &[i32],
        tiling: Tiling,
    ) -> Self {
        let rows = segments.len();
        assert!(rows <= Decodes::SIZE, "decode rows exceed the envelope");
        let keys: Vec<i32> = segments.iter().map(|row| row[LEN] + row[CACHED]).collect();
        let Layout {
            blocks,
            longest,
            depth,
            width,
            groups,
            counts,
            waves,
        } = Layout::new::<Waves>(&keys, tiling);
        let firsts: Vec<i32> = counts
            .iter()
            .scan(0, |sum, count| Some(mem::replace(sum, *sum + count)))
            .collect();
        let items: i32 = counts.iter().sum();
        let peak = counts.iter().copied().max().unwrap_or(0);
        let paired = (waves as usize).next_multiple_of(PAIR);
        let units = (((longest * groups + 63) >> 6) + 1) >> 1;
        let merges = if peak > 1 { (rows as i32 + 7) >> 3 } else { 0 };
        assert!(
            Layout::fits(waves, items, Waves::SIZE),
            "items exceed the envelope"
        );
        let wave = |cells: usize| paired * cells;
        let (waving, merging) = (usize::from(waves > 0), usize::from(merges > 0));
        let jobs_read = merges as usize * Eight::SIZE * FLIT;
        let mut entries = vec![0; wave(Column::SIZE * Time::SIZE)];
        let [
            mut lengths,
            mut queries,
            mut outputs,
            mut partials,
            mut scales,
        ] = [(); 5].map(|()| vec![0; wave(Column::SIZE)]);
        let mut limits = vec![0; waving * (Column::SIZE - 1) * Waves::SIZE * FLIT + wave(FLIT)];
        let row_stride = (<m![KvHeads, G, Head]>::SIZE * size_of::<bf16>()) as i32;
        let last = (ids.len() as i32 - Quad::SIZE as i32) * size_of::<i32>() as i32;
        for physical in 0..paired * Column::SIZE {
            // Items take consecutive slices; the slices of column `ordinal` are those of `physical`
            // with the two low bits swapped ([`ColumnSlices`]).
            let ordinal = (physical & !3) | ((physical & 1) << 1) | ((physical >> 1) & 1);
            let k = physical as i32;
            let item = (k * groups) >> Column::SIZE.ilog2();
            let present = i32::from(item < items);
            let row = firsts.partition_point(|first| first * width <= k) - 1;
            let before = ((item - firsts[row]) * width) << ITEM_SHIFT;
            let used = blocks[row] - before;
            let first = segments[row][TABLE] + before;
            let column = k - item * width;
            let rounds = (used * groups) >> (Column::SIZE * Quad::SIZE).ilog2();
            let partial = (used - (((rounds * width) + column) << 2)).clamp(0, 4);
            let length = (((rounds << 2) + partial) * present).min(Time::SIZE as i32);
            let skipped = ((column * 4) + before) << BLOCK_SHIFT;
            let query = segments[row][START] * row_stride;
            let sink =
                ((k & (Column::SIZE as i32 - 1)) + (Reach::SIZE - SINKS) as i32) * row_stride;
            let writes = ((present << 1) - counts[row]).clamp(0, 1);
            lengths[ordinal] = length;
            let (wave, at) = (ordinal / Column::SIZE, ordinal % Column::SIZE);
            limits[(at * Waves::SIZE + wave) * FLIT] = (keys[row] - skipped) * present;
            queries[ordinal] = query;
            outputs[ordinal] = (query - sink) * writes + sink;
            partials[ordinal] = item * Self::partial_stride::<KvHeads, G, Head>();
            scales[ordinal] = item * Self::scale_stride::<KvHeads, G>();
            // A quad of the item's blocks a slot of four, clamped into the row's padded table.
            let origin = (column << QUAD_STRIDE.ilog2()) + (first << 2);
            let bound = ((((used + Block::SIZE as i32 - 1) >> BLOCK_SHIFT)
                << (Block::SIZE * size_of::<i32>()).ilog2())
                + (first << 2)
                - QUAD_STRIDE)
                .min(last);
            for slot in 0..length as usize {
                let quad = (((slot as i32 / 4) * width) << QUAD_STRIDE.ilog2()) + origin;
                let id = (quad.min(bound).max(0) >> 2) as usize + slot % 4;
                entries[(wave * Time::SIZE + slot) * Column::SIZE + at] =
                    ids[id] * Blocks::<KvHeads, Head>::BLOCK_STRIDE;
            }
        }
        // A merge tile's eight jobs take eight rows, so the launch's tiles take every row.
        let jobs = |value: &dyn Fn(usize) -> i32| -> Vec<i32> {
            let mut flits = vec![0; jobs_read];
            for (flit, row) in flits.iter_mut().step_by(FLIT).zip(0..rows) {
                *flit = value(row);
            }
            flits
        };
        // Input `q` of each of a tile's eight jobs.
        let mut inputs = vec![0; merging * Eight::SIZE * Input::SIZE * FLIT];
        for (flit, at) in inputs.iter_mut().step_by(FLIT).zip(0..) {
            *flit = (at % Input::SIZE) as i32;
        }
        let key = |at: usize| {
            let (token, slot) = (at / Time::SIZE, at % Time::SIZE);
            (token + 16 * (slot % 4) + 64 * (1 << depth) * (slot / 4)) as i32
        };
        Self {
            images: [
                vec![waves],
                vec![depth],
                vec![units.min((Time::SIZE / 8) as i32)],
                vec![merges],
                entries,
                lengths,
                queries,
                outputs,
                partials,
                scales,
                limits,
                (0..waving * Block::SIZE * Time::SIZE).map(key).collect(),
                jobs(&|row| firsts[row]),
                jobs(&|row| counts[row]),
                jobs(&|row| segments[row][START]),
                inputs,
            ],
            // The words of each array of `Waves` waves, in [`Wide::bind`] order.
            words: {
                let capacity = Waves::SIZE * Column::SIZE;
                let decodes = Decodes::SIZE * FLIT;
                [
                    1,
                    1,
                    1,
                    1,
                    capacity * Time::SIZE,
                    capacity,
                    capacity,
                    capacity,
                    capacity,
                    capacity,
                    capacity * FLIT,
                    Block::SIZE * Time::SIZE,
                    decodes,
                    decodes,
                    decodes,
                    Eight::SIZE * Input::SIZE * FLIT,
                ]
            },
        }
    }

    /// Each array in [`Wide::bind`] order: the leading words the launch uses, and all its words.
    pub(in super::super) fn reads(&self) -> impl Iterator<Item = (&[i32], usize)> {
        (self.images.iter().zip(self.words)).map(|(image, words)| (&image[..], words))
    }
}
