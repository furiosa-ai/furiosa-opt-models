//! A launch's decode rows as the items of its two domains, planned on the host.
//!
//! - A row of more than four query groups a KV head is its group tiles, each an item row over the
//!   row's history: tile `t` of row `r` is item row `r T + t` of `T` tiles a row.
//! - An item row of at most one packed column's keys may be packed: an adaptive launch packs them
//!   all when that takes fewer waves in all than the wide domain alone would, and none otherwise; a
//!   fixed one packs them all. Each domain keeps its rows in row order and plans them alone.
//! - A row that attends its last `window` keys reads its blocks from the one holding the window's
//!   first key, as if its history began there; a slot is seen when its key lies below its column's
//!   limit by at most `window`, so the keys of that block below the window are not.
//! - The arrays are the launch's row count and window floor, then the packed domain's, then the
//!   wide domain's.

use super::*;
use crate::ops::index::table::FIELDS;
use crate::ops::index::table::{CACHED, LEN, START, TABLE};
use crate::specialize::attention::DECODES;
use std::iter;
use std::slice;

/// How a call's host tiles each row's history into items.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Tiling {
    /// The domains and item widths that take the fewest waves for the call's rows: a row's sums
    /// then depend on the rows beside it.
    #[default]
    Adaptive,
    /// Each row by its own history alone, every call the same way: a row's bits then do not
    /// change with the rows that share its calls.
    Fixed,
}

/// How a call's host plans each row's items: their tiling, and the keys a row attends, its last
/// `window` with itself among them, or with no window its whole history.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Options {
    pub tiling: Tiling,
    pub window: Option<usize>,
}

/// A tiling alone attends each row's whole history.
impl From<Tiling> for Options {
    fn from(tiling: Tiling) -> Self {
        Self {
            tiling,
            window: None,
        }
    }
}

/// The window of a row that attends its whole history: past every position a row holds.
const WHOLE: usize = 1 << 24;

/// A launch's items as the arrays the two domains' bodies read, at the compiled capacities.
pub struct Plan {
    rows: i32,
    floor: [i32; FLOOR],
    packed: packed::plan::Plan,
    wide: wide::plan::Plan,
}

impl Plan {
    /// The items of decode rows `segments`, block IDs `ids` padded to whole runs, planned as
    /// `options` says in `Waves` wide waves; `Reach` holds the rows as group tiles, the last
    /// [`SINKS`] sinks.
    pub fn new<KvHeads: AxisName, G: AxisName, Head: AxisName, Waves: AxisName, Reach: AxisName>(
        segments: &[[i32; FIELDS]],
        ids: &[i32],
        options: impl Into<Options>,
    ) -> Self {
        let Options { tiling, window } = options.into();
        let window = window.unwrap_or(WHOLE);
        assert!(
            (1..=WHOLE).contains(&window),
            "a row's window holds the row and at most 2^24 positions"
        );
        let window = window as i32;
        let block = Block::SIZE.ilog2();
        let tiles = GroupTile::<G>::TILES;
        let tiled: Vec<[i32; FIELDS]> = segments
            .iter()
            .flat_map(|row| {
                let skipped = (row[LEN] + row[CACHED] - window).max(0) >> block;
                let mut row = *row;
                row[CACHED] -= skipped << block;
                row[TABLE] += skipped;
                (0..tiles).map(move |tile| {
                    let mut row = row;
                    row[START] = row[START] * tiles as i32 + tile as i32;
                    row
                })
            })
            .collect();
        assert!(tiled.len() <= DECODES, "decode rows exceed the envelope");
        // The longest history a packed row has: a column's, so a fixed tiling packs it in one.
        let short = packed::plan::Plan::column::<KvHeads, Head>();
        let keys = |row: &[i32; FIELDS]| usize::try_from(row[LEN] + row[CACHED]);
        let (packed, wide): (Vec<_>, Vec<_>) = tiled
            .iter()
            .partition(|row| keys(row).is_ok_and(|keys| keys <= short));
        let wide_waves = wide::plan::Plan::waves::<KvHeads, Head, Waves>;
        let packs = tiling == Tiling::Fixed || {
            let alone = wide_waves(&tiled.iter().collect::<Vec<_>>());
            packed::plan::Plan::waves::<KvHeads, Head>(&packed) + wide_waves(&wide) < alone
        };
        let (packed, wide) = match packs {
            true => (packed, wide),
            false => (Vec::new(), tiled.iter().collect()),
        };
        let mut floor = [0; FLOOR];
        floor[0] = -window - 1;
        Self {
            rows: i32::try_from(segments.len()).expect("the decode rows fit i32"),
            floor,
            packed: packed::plan::Plan::new::<KvHeads, GroupTile<G>, Head, Tiled<Reach, G>>(
                &packed, ids, tiling,
            ),
            wide: wide::plan::Plan::new::<KvHeads, GroupTile<G>, Head, Waves, Tiled<Reach, G>>(
                &wide, ids, tiling,
            ),
        }
    }

    /// The most wide waves a launch of at most `rows` rows of `G` query groups to each of `KvHeads`
    /// KV heads of `Head` channels uses when each row attends at most `keys` keys, either tiled:
    /// the capacity that holds any such launch.
    pub const fn waves<KvHeads: AxisName, G: AxisName, Head: AxisName>(
        rows: usize,
        keys: usize,
    ) -> usize {
        let rows = rows * GroupTile::<G>::TILES;
        let waves = wide::plan::Plan::most::<KvHeads, Head>(rows, keys);
        // A batch-invariant load serves every call its line admits.
        assert!(
            rows <= DECODES
                && packed::plan::Plan::fixes(rows)
                && wide::plan::Plan::fixes::<KvHeads, Head>(rows, keys, waves),
            "a fixed tiling of every row over the whole context exceeds the compiled items"
        );
        waves
    }

    /// Each array in [`Items::bind`] order: the leading words the launch uses, and all its words.
    pub fn reads(&self) -> impl Iterator<Item = (&[i32], usize)> {
        iter::once((slice::from_ref(&self.rows), 1))
            .chain(iter::once((&self.floor[..], FLOOR)))
            .chain(self.packed.reads())
            .chain(self.wide.reads())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::specialize::attention::{HISTORY_WAVES, WAVES};
    use crate::tensor::axis::Rows;

    use super::super::tests::{Heads, QueryGroups as Groups, QueryPairs as Pairs, Width as Head};

    /// The waves a launch of rows of `lengths` keys, `G` query groups a KV head, tiled as `tiling`,
    /// gives each domain, `WAVES` wide waves at most, packed then wide.
    fn waves<G: AxisName, const WAVES: usize>(lengths: &[usize], tiling: Tiling) -> [i32; 2] {
        let segments: Vec<[i32; FIELDS]> = lengths
            .iter()
            .enumerate()
            .map(|(row, &keys)| [row, 1, keys - 1, 0, keys.div_ceil(16)].map(|field| field as i32))
            .collect();
        let ids = vec![1; 1 << 18];
        let plan = Plan::new::<Heads, G, Head, Waves<WAVES>, Rows<1280>>(&segments, &ids, tiling);
        let leading = |at: usize| plan.reads().nth(at).map_or(0, |(read, _)| read[0]);
        [leading(2), leading(2 + packed::plan::ARRAYS)]
    }

    /// A short row is packed only when that takes fewer waves in all than every row wide would;
    /// a row past the packed bound never is.
    #[test]
    fn packs_short_rows_only_when_it_saves_waves() {
        let packed_and_wide = |packed: usize, wide: usize| (packed > 0, wide > 0);
        let domains = |lengths: &[usize]| {
            let [packed, wide] = waves::<Groups, WAVES>(lengths, Tiling::Adaptive);
            packed_and_wide(packed as usize, wide as usize)
        };
        assert_eq!(domains(&[25]), (false, true));
        assert_eq!(domains(&[25; 8]), (false, true));
        assert_eq!(domains(&[25; 16]), (false, true));
        assert_eq!(domains(&[25; 64]), (true, false));
        assert_eq!(domains(&[25; 1024]), (true, false));
        assert_eq!(domains(&[256; 64]), (true, false));
        assert_eq!(domains(&[512; 64]), (false, true));
        assert_eq!(domains(&[16384]), (false, true));
        let mixed = [vec![25; 512], vec![4096; 32]].concat();
        assert_eq!(domains(&mixed), (true, true));
    }

    /// Within [`HISTORY_WAVES`], 64 rows of the whole 40,960-key history take every wide wave, as
    /// fewer or shorter rows take fewer.
    #[test]
    fn holds_64_rows_of_the_whole_history_in_192_waves() {
        const LONG: usize = 40960;
        let adaptive =
            |lengths: &[usize]| waves::<Groups, HISTORY_WAVES>(lengths, Tiling::Adaptive);
        assert_eq!(adaptive(&[LONG; 64]), [0, 192]);
        assert_eq!(adaptive(&[LONG; 63]), [0, 189]);
        for rows in 1..=64 {
            let [_, wide] = adaptive(&vec![LONG; rows]);
            assert!(wide as usize <= 3 * rows, "{rows} rows take {wide} waves");
        }
        let mixed: Vec<usize> = (0..64).map(|row| [LONG, 1, 16385, 25][row % 4]).collect();
        assert!(adaptive(&mixed)[1] < 192);
    }

    /// 77 rows of the whole history take items of eight columns, 193 waves past
    /// [`HISTORY_WAVES`], so their plan is refused; 76 take 190.
    #[test]
    #[should_panic(expected = "items exceed the envelope")]
    fn refuses_77_rows_of_the_whole_history() {
        assert_eq!(
            waves::<Groups, HISTORY_WAVES>(&[40960; 76], Tiling::Adaptive),
            [0, 190]
        );
        waves::<Groups, HISTORY_WAVES>(&[40960; 77], Tiling::Adaptive);
    }

    /// The capacity [`Plan::waves`] names holds a launch of odd waves: one row of a wave's 16,384
    /// keys fills one, and a launch uses its waves rounded up to a pair.
    #[test]
    fn holds_a_launch_of_odd_waves() {
        const KEYS: usize = 16384;
        const CAPACITY: usize = Plan::waves::<Heads, Pairs, Head>(1, KEYS);
        assert_eq!(waves::<Pairs, CAPACITY>(&[KEYS], Tiling::Adaptive), [0, 1]);
    }
}
