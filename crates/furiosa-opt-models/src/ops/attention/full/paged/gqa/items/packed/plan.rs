//! A launch's packed rows as work items, planned on the host from the rows' compact history.
//!
//! - A row of `keys` keys holds `ceil(keys / 16)` blocks, at most 16384 keys: a cluster's
//!   columns' every block round.
//! - A group of `C` columns holds one row a lane, in row order: column `c` of a group takes block
//!   `b` of each of its lanes' rows when `b % C = c`, in block round `b / C`, lane one's rounds
//!   from a launch-wide `half` on.
//! - An adaptive launch takes the group span and lane count of fewest waves, then fewest live lane
//!   slots, then the narrowest span, then one lane; a fixed one a column and one lane a group.
//! - A column's tables name the pool byte offset of every block round it fills, the rounds it
//!   fills, each lane's limit, and where each lane's query and output rows lie: a group's first
//!   column writes its lanes' rows, every other column lane its own sinks.
//! - A launch uploads its tables' rows of its waves rounded up to a pair, the key positions only
//!   with waves, and no other word.

use super::super::plan::Tiling;
use super::*;
use crate::ops::index::table::FIELDS;
use crate::ops::index::table::{CACHED, LEN, START, TABLE};
use crate::tensor::cache::Blocks;

/// The arrays of a launch's packed rows, which [`Packed::bind`] takes in turn.
pub(in super::super) const ARRAYS: usize = 9;
/// The columns of a flit's eight cells, one value each.
const FLIT: usize = 8;
/// Tokens of a block on each token slice of its column.
const QUAD: usize = 4;
/// Keys one column holds: its every block round.
pub(in super::super) const COLUMN: usize = Round::SIZE * Block::SIZE;
/// Keys a packed row holds at most: every block round of every column of a cluster.
const SEGMENT: usize = Column::SIZE * COLUMN;
/// The key position of a slot its lane never reads: above every limit.
const UNSEEN: i32 = i32::MAX / 2;

/// A launch's packed rows as the arrays the packed body reads, each the leading words the launch
/// uses of its compiled capacity.
pub(in super::super) struct Plan {
    images: [Vec<i32>; ARRAYS],
    words: [usize; ARRAYS],
}

/// A launch-wide placement: groups of `columns` columns holding `lanes` live lanes, lane one's
/// rounds from `half` on, at most `rounds` rounds a column, and the waves the groups take.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Placement {
    columns: usize,
    lanes: usize,
    half: usize,
    rounds: usize,
    waves: usize,
}

impl Placement {
    /// The placement of items of `blocks` blocks each tiled as `tiling` says: adaptively the span
    /// and lanes of fewest waves, then fewest live lane slots, then the narrowest span, then one
    /// lane; fixed, one column and one lane a group.
    fn of(blocks: &[usize], tiling: Tiling) -> Self {
        // Items in row order, one a lane of a group of `columns` columns, if every column holds
        // its lanes' rounds.
        let place = |columns: usize, lanes: usize| {
            let longest = |lane: usize| {
                let items = blocks.iter().skip(lane).step_by(lanes);
                items
                    .map(|blocks| blocks.div_ceil(columns))
                    .max()
                    .unwrap_or(0)
            };
            let half = longest(0);
            let rounds = half + if lanes > 1 { longest(1) } else { 0 };
            (rounds <= Round::SIZE).then_some(Self {
                columns,
                lanes,
                half,
                rounds,
                waves: Self::waves(blocks.len(), columns, lanes),
            })
        };
        if tiling == Tiling::Fixed {
            return place(1, 1).expect("a fixed tiling packs only rows that fit one column");
        }
        let placements = (0..=Depth::SIZE)
            .flat_map(|depth| [1, 2].map(|lanes| place(1 << depth, lanes)))
            .flatten();
        let cost = |placement: &Self| {
            let Self {
                columns,
                lanes,
                waves,
                ..
            } = *placement;
            (waves, waves * lanes * placement.units(), columns, lanes)
        };
        placements
            .min_by_key(cost)
            .expect("a row of at most 16384 keys fits every column")
    }

    /// The waves `rows` rows take in groups of `columns` columns holding `lanes` rows each.
    const fn waves(rows: usize, columns: usize, lanes: usize) -> usize {
        rows.div_ceil(lanes).div_ceil(Column::SIZE / columns)
    }

    /// The eight-slot units a slice scores, four slots a round.
    fn units(&self) -> usize {
        (self.rounds * QUAD).div_ceil(8)
    }
}

impl Plan {
    /// The items of packed rows `segments`, whose block IDs are `ids`, tiled as `tiling` says;
    /// `Reach` holds the attended rows, its last [`SINKS`] the column lanes' sinks.
    pub(in super::super) fn new<KvHeads: AxisName, G: AxisName, Head: AxisName, Reach: AxisName>(
        segments: &[&[i32; FIELDS]],
        ids: &[i32],
        tiling: Tiling,
    ) -> Self {
        let int = |value: usize| i32::try_from(value).expect("a table value fits i32");
        let keys = Self::keys(segments);
        assert!(
            keys.iter().all(|&keys| keys <= SEGMENT),
            "a packed row exceeds its columns"
        );
        let blocks: Vec<usize> = keys.iter().map(|keys| keys.div_ceil(Block::SIZE)).collect();
        let placement = Placement::of(&blocks, tiling);
        let Placement {
            columns,
            lanes,
            half,
            waves,
            ..
        } = placement;
        assert!(waves <= Waves::SIZE, "packed rows exceed the envelope");
        let row_stride = (<m![KvHeads, G, Head]>::SIZE * size_of::<bf16>()) as i32;
        // Every column lane of the waves rounded up to a pair: the packed body moves two waves'
        // queries, limits and rows a transfer.
        let cells = ((waves + 1) & !1) * Column::SIZE * Pack::SIZE;
        let mut entries = vec![0; cells / Pack::SIZE * Round::SIZE];
        let mut lengths = vec![0; cells / Pack::SIZE];
        let [mut queries, mut outputs] = [(); 2].map(|()| vec![0; cells]);
        let mut limits = vec![0; cells * FLIT];
        for cell in 0..cells {
            let (column, lane) = (cell / Pack::SIZE, cell % Pack::SIZE);
            let group = column / columns;
            let row = group * lanes + lane;
            let live = lane < lanes && row < segments.len();
            // An empty lane reads its group's first lane's query, or the first row's; it sees no
            // key.
            let read = match (live, group * lanes < segments.len()) {
                (true, _) => row,
                (false, true) => group * lanes,
                (false, false) => 0,
            };
            queries[cell] = segments[read][START] * row_stride;
            outputs[cell] = int(Reach::SIZE - SINKS + cell % SINKS) * row_stride;
            if !live {
                continue;
            }
            let place = column % columns;
            let (base, rounds) = (
                lane * half,
                blocks[row].saturating_sub(place).div_ceil(columns),
            );
            let table = usize::try_from(segments[row][TABLE]).expect("a table index");
            for round in 0..rounds {
                entries[column * Round::SIZE + base + round] =
                    ids[table + round * columns + place] * Blocks::<KvHeads, Head>::BLOCK_STRIDE;
            }
            if rounds > 0 {
                lengths[column] = lengths[column].max(int(base + rounds));
            }
            limits[cell * FLIT] = int(keys[row]) - int(Block::SIZE * place);
            if place == 0 {
                outputs[cell] = segments[row][START] * row_stride;
            }
        }
        // A lane's key position of a slot: from its first round, the row's keys before the slot's
        // in the column's first block of the round.
        let slices = Block::SIZE / QUAD;
        let key = |at: usize| {
            let (lane, token, slot) = (
                at / (slices * Slot::SIZE),
                at / Slot::SIZE % slices,
                at % Slot::SIZE,
            );
            let (round, base) = (slot / QUAD, lane * half);
            match lane < lanes && round >= base {
                true => int(Block::SIZE * columns * (round - base) + QUAD * token + slot % QUAD),
                false => UNSEEN,
            }
        };
        // The words of each array, in [`Packed::bind`] order.
        let words = {
            let capacity = Waves::SIZE * Column::SIZE * Pack::SIZE;
            [
                1,
                1,
                1,
                Waves::SIZE * Column::SIZE * Round::SIZE,
                Waves::SIZE * Column::SIZE,
                capacity,
                capacity,
                capacity * FLIT,
                <m![Pack, Block / 4, Slot]>::SIZE,
            ]
        };
        Self {
            images: [
                vec![int(waves)],
                vec![int(columns.ilog2() as usize)],
                vec![int(placement.units())],
                entries,
                lengths,
                queries,
                outputs,
                limits,
                (0..usize::from(waves > 0) * words[8]).map(key).collect(),
            ],
            words,
        }
    }

    /// The waves packed rows `segments` fill.
    pub(in super::super) fn waves(segments: &[&[i32; FIELDS]]) -> usize {
        let blocks: Vec<usize> = Self::keys(segments)
            .iter()
            .map(|keys| keys.div_ceil(Block::SIZE))
            .collect();
        Placement::of(&blocks, Tiling::Adaptive).waves
    }

    /// Whether `rows` packed rows fit the compiled waves, tiled fixed: one column and one lane a
    /// row.
    pub(in super::super) const fn fixes(rows: usize) -> bool {
        Placement::waves(rows, 1, 1) <= Waves::SIZE
    }

    /// Each array in [`Packed::bind`] order: the leading words the launch uses, and all its words.
    pub(in super::super) fn reads(&self) -> impl Iterator<Item = (&[i32], usize)> {
        self.images
            .iter()
            .zip(self.words)
            .map(|(image, words)| (&image[..], words))
    }

    fn keys(segments: &[&[i32; FIELDS]]) -> Vec<usize> {
        let keys = |row: &&[i32; FIELDS]| usize::try_from(row[LEN] + row[CACHED]);
        segments
            .iter()
            .map(|row| keys(row).expect("a row sees its keys"))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::tests::{Heads as KvHeads, QueryGroups as Groups, Width as Head};
    use super::*;
    use crate::tensor::axis::Rows;
    use std::collections::HashSet;

    type Reach = Rows<1280>;
    const ROW_STRIDE: usize = <m![KvHeads, Groups, Head]>::SIZE * size_of::<bf16>();
    const BLOCK_STRIDE: i32 = Blocks::<KvHeads, Head>::BLOCK_STRIDE;
    /// Block IDs a launch's decode rows hold at most.
    const TABLES: usize = 65536;
    /// Pool blocks the ID of a row's block cycles through, from 1, so ID 0 is no row's.
    const POOL: usize = Blocks::<KvHeads, Head>::MAX_BLOCKS - 1;

    /// Rows of `lengths` keys: every visible key of every row is admitted exactly once, by the
    /// lane that reads the row's query, from a block of the row (the block's ID names its row when
    /// the rows hold fewer blocks than the pool), inside the scored slots and the column's
    /// gathered rounds; no slot admits two lanes; every block the gather reads up to its wave's
    /// longest prefix is a pool block; each pair's store writes distinct rows, each row once and
    /// from its own lane.
    fn check(lengths: &[usize]) {
        // Block `b` of row `r` takes the ID after every earlier block's, cycling through the pool.
        let (mut segments, mut ids, mut firsts, mut next) = (Vec::new(), Vec::new(), Vec::new(), 0);
        for (row, &length) in lengths.iter().enumerate() {
            let used = length.div_ceil(Block::SIZE);
            segments.push([row, 1, length - 1, ids.len(), used].map(|field| field as i32));
            let id = |b: usize| (1 + (next + b.min(used - 1)) % POOL) as i32;
            ids.extend((0..used.next_multiple_of(16)).map(id));
            firsts.push(next);
            next += used;
        }
        let plan = Plan::new::<KvHeads, Groups, Head, Reach>(
            &segments.iter().collect::<Vec<_>>(),
            &ids,
            Tiling::Adaptive,
        );
        let image = |at: usize| &plan.images[at];
        let (waves, units) = (image(0)[0] as usize, image(2)[0] as usize);
        let cell = |wave: usize, column: usize, lane: usize| {
            (wave * Column::SIZE + column) * Pack::SIZE + lane
        };
        let blocks: usize = lengths.iter().map(|keys| keys.div_ceil(Block::SIZE)).sum();
        // Row `row`'s block of ID `id`, which names its row when the rows' blocks fit the pool.
        let owner = |row: usize, id: usize| {
            if blocks < POOL {
                let named = firsts.partition_point(|&first| first < id) - 1;
                assert_eq!(named, row, "a lane reads another row");
            }
            (id - 1 + POOL - firsts[row] % POOL) % POOL
        };
        let mut seen: Vec<Vec<bool>> = lengths.iter().map(|&keys| vec![false; keys]).collect();
        for wave in 0..waves {
            let gathered = |column: usize| image(4)[wave * Column::SIZE + column] as usize;
            let longest = (0..Column::SIZE).map(gathered).max().unwrap_or(0);
            for column in 0..Column::SIZE {
                let entry =
                    |round: usize| image(3)[(wave * Column::SIZE + column) * Round::SIZE + round];
                for round in 0..longest {
                    let id = entry(round) / BLOCK_STRIDE;
                    assert!(
                        (0..=POOL as i32).contains(&id),
                        "column {column} reads block {id}"
                    );
                }
                let slots = (0..Block::SIZE / QUAD)
                    .flat_map(|token| (0..Slot::SIZE).map(move |slot| (token, slot)));
                for (token, slot) in slots {
                    let admits = |lane: usize| {
                        image(8)[(lane * Block::SIZE / QUAD + token) * Slot::SIZE + slot]
                            < image(7)[cell(wave, column, lane) * FLIT]
                    };
                    let lanes: Vec<usize> = (0..Pack::SIZE).filter(|&lane| admits(lane)).collect();
                    assert!(
                        lanes.len() <= 1,
                        "column {column} slot {slot} admits two lanes"
                    );
                    for lane in lanes {
                        assert!(
                            slot < 8 * units && slot / QUAD < gathered(column),
                            "slot {slot} unscored or ungathered"
                        );
                        let row = image(5)[cell(wave, column, lane)] as usize / ROW_STRIDE;
                        let block = owner(row, (entry(slot / QUAD) / BLOCK_STRIDE) as usize);
                        let key = Block::SIZE * block + QUAD * token + slot % QUAD;
                        assert!(
                            key < lengths[row] && !seen[row][key],
                            "row {row} key {key} past its keys or twice"
                        );
                        seen[row][key] = true;
                    }
                }
            }
        }
        assert!(
            seen.iter().all(|seen| seen.iter().all(|&seen| seen)),
            "a row misses a key"
        );
        let mut written = vec![0; lengths.len()];
        for pair in 0..waves.div_ceil(2) {
            let mut rows = HashSet::new();
            for at in cell(2 * pair, 0, 0)..cell(2 * pair + 2, 0, 0) {
                let row = image(6)[at] as usize / ROW_STRIDE;
                assert!(rows.insert(row), "pair {pair} writes row {row} twice");
                if row < Reach::SIZE - SINKS {
                    assert_eq!(
                        image(5)[at] as usize,
                        row * ROW_STRIDE,
                        "a lane writes another row"
                    );
                    written[row] += 1;
                }
            }
        }
        assert!(
            written.iter().all(|&written| written == 1),
            "a row written other than once"
        );
    }

    /// The gate's cells take the spans, lanes, scored slots and waves of the placement rule.
    #[test]
    fn places_the_gate_cells_by_the_rule() {
        let cells: [(usize, usize, _); 14] = [
            (1, 25, (4, 1, 8, 1)),
            (8, 25, (4, 1, 8, 1)),
            (64, 25, (4, 1, 8, 1)),
            (128, 25, (4, 2, 16, 1)),
            (512, 25, (4, 2, 16, 4)),
            (1024, 25, (4, 2, 16, 8)),
            (1, 1024, (128, 1, 8, 1)),
            (8, 1024, (32, 1, 32, 1)),
            (32, 1024, (16, 1, 64, 2)),
            (128, 1024, (16, 1, 64, 8)),
            (1, 4096, (256, 1, 16, 1)),
            (8, 4096, (64, 1, 64, 2)),
            (32, 4096, (64, 1, 64, 8)),
            (1, 16384, (256, 1, 64, 1)),
        ];
        for (rows, keys, expected) in cells {
            let placement =
                Placement::of(&vec![keys.div_ceil(Block::SIZE); rows], Tiling::Adaptive);
            let Placement {
                columns,
                lanes,
                waves,
                ..
            } = placement;
            let found = (QUAD * columns, lanes, 8 * placement.units(), waves);
            assert_eq!(found, expected, "{rows} rows of {keys} keys");
        }
        assert_eq!(Placement::of(&[], Tiling::Adaptive).waves, 0);
    }

    /// Row counts at and around every wave boundary of the gate's lengths, within the launch's
    /// block IDs, and rows of many lengths together, attend every key once and store distinct rows.
    #[test]
    fn attends_every_key_once_at_the_wave_boundaries() {
        let counts = [
            0, 1, 2, 3, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257,
        ];
        let counts = counts.into_iter().chain([511, 512, 513, 1023, 1024]);
        for rows in counts {
            for keys in [25usize, 1024, 4096, 16384] {
                if rows * keys.div_ceil(Block::SIZE) <= TABLES {
                    check(&vec![keys; rows]);
                }
            }
        }
        check(
            &(0..300)
                .map(|row| [25, 1024, 1, 600, 16, 17][row % 6])
                .collect::<Vec<_>>(),
        );
        check(&[16384, 1, 4096, 25, 16384, 17, 2, 1]);
    }
}
