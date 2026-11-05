//! What a kernel's modules leave to their composition: the live rows of a launch.
//!
//! - A launch is compiled for an envelope of rows and serves any number up to it. Its step names
//!   the live rows `0..n` once, as a [`Count`]; [`Tiles::read`] reads that count at the launch's
//!   start and hands the value on.
//! - A leaf takes the [`Tiles`] of its own iteration [`Space`] and derives every loop count from
//!   them, so a row past `n` costs no trip and takes no effect. For example, 9 live rows of a
//!   64-row envelope are two tiles of 8 rows: one whole, one holding a single row.
//! - A launch value in HBM moves on chip where it enters a loop ([`Fill`]) and back where it
//!   leaves it ([`Spill`]); the binding implements each move for its layouts.
//! - Beyond the moves the composer only orders: it fuses modules into one loop over a space
//!   ([`fuse`]), rotates a loop ([`window`]), issues other modules' transfers at a running module's
//!   points ([`Points`], [`Deferred`]), and reads each part of the step once ([`Once`]).

use crate::ops::layout::axes::Chip;
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

pub mod step;

/// A count a launch reads at its start: its live rows, or the history windows it spans; each chip
/// of `Chips` holds a copy.
pub type Count<Chips = Chip> = HbmTensor<i32, Chips, m![1]>;

/// An iteration space: a loop over the live rows of an envelope, `SIZE` rows a trip.
pub trait Space: AxisName {
    /// The rows a launch is compiled for; a loop over them runs `Envelope::SIZE / SIZE` trips.
    type Envelope: AxisName;
}

/// Tiles of `Unit` rows within an `Envelope` of rows.
#[derive(Clone, Debug)]
pub struct Tile<Envelope, Unit>(PhantomData<(Envelope, Unit)>);

impl<Envelope: AxisName, Unit: AxisName> AxisName for Tile<Envelope, Unit> {
    const NAME: Ident = Unit::NAME;
    const SIZE: usize = Unit::SIZE;
}

impl<Envelope: AxisName, Unit: AxisName> Space for Tile<Envelope, Unit> {
    type Envelope = Envelope;
}

/// The live rows `0..rows()` of a launch, as the tiles of space `S` that cover them: every tile
/// whole but the last.
pub struct Tiles<S> {
    rows: usize,
    start: usize,
    space: PhantomData<S>,
}

impl<S> Clone for Tiles<S> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S> Copy for Tiles<S> {}

impl<S: Space> Tiles<S> {
    /// The trips over a whole envelope.
    pub const TRIPS: usize = {
        assert!(S::SIZE > 0 && S::Envelope::SIZE.is_multiple_of(S::SIZE));
        S::Envelope::SIZE / S::SIZE
    };

    /// The live rows a step's count names, at most the envelope; leaves take these tiles, never
    /// the count, so only the code that binds the step reads it.
    pub fn read<Chips: M>(count: &Count<Chips>) -> Self {
        Self::over(count.extent::<m![{ Symbol<S::Envelope> }]>())
    }

    /// `rows` live rows, at most `S::Envelope::SIZE`.
    pub(crate) fn over(rows: usize) -> Self {
        let _ = Self::TRIPS;
        Self {
            rows,
            start: 0,
            space: PhantomData,
        }
    }

    pub fn rows(self) -> usize {
        self.rows
    }

    /// The first row of this domain within the launch envelope.
    pub fn start(self) -> usize {
        self.start
    }

    /// The same live domain expressed in another space of the same envelope.
    pub fn retile<T: Space<Envelope = S::Envelope>>(self) -> Tiles<T> {
        let _ = Tiles::<T>::TRIPS;
        Tiles {
            rows: self.rows,
            start: self.start,
            space: PhantomData,
        }
    }

    /// Restrict this domain to a relative row interval; rows outside it remain padding.
    pub fn subrange(self, start: usize, rows: usize) -> Self {
        let start = start.min(self.rows);
        Self {
            rows: rows.min(self.rows - start),
            start: self.start + start,
            space: PhantomData,
        }
    }

    /// One live tile, with its origin relative to this domain's origin.
    pub fn tile(self, index: usize) -> Self {
        let whole = (self.count() + Self::TRIPS - 1 - index) / (Self::TRIPS + 1);
        let tail = (self.rows + S::SIZE - 1) % S::SIZE + 1;
        Self {
            rows: tail + whole * (S::SIZE - tail),
            start: self.start + index * S::SIZE,
            space: PhantomData,
        }
    }

    /// The tiles holding live rows, the last possibly partial. Rounded up by addition, so a loop
    /// over them is allocated for exactly the envelope's trips.
    pub fn count(self) -> usize {
        (self.rows + S::SIZE - 1) / S::SIZE
    }

    /// Runs `every` when every row of the envelope is live; otherwise `whole` on each tile whose
    /// rows are all live, then `row` on each live row of the partial last tile. Each live row's
    /// effect issues once and no other row's does.
    pub fn cover<D>(
        self,
        context: &mut D,
        (every, mut whole, mut row): (
            impl FnOnce(&mut D),
            impl FnMut(&mut D, usize),
            impl FnMut(&mut D, usize),
        ),
    ) {
        if self.rows == S::Envelope::SIZE {
            every(context);
        } else {
            let first = (S::SIZE - self.start % S::SIZE) % S::SIZE;
            let first = first.min(self.rows);
            for lane in 0..first {
                row(context, self.start + lane);
            }
            let tiles = (self.rows - first) / S::SIZE;
            for tile in 0..tiles {
                whole(context, (self.start + first) / S::SIZE + tile);
            }
            for lane in 0..(self.rows - first) % S::SIZE {
                row(context, self.start + first + tiles * S::SIZE + lane);
            }
        }
    }
}

mod points;
mod window;
pub use points::{Deferred, Issued, Points, Prefetch, Prefetched, Prefetching};
pub use window::{Bounded, Fixed, Periods, window};

/// A value known up to a finite, positive factor `c` per row and head: an RMS norm cancels it, its
/// epsilon then acting on the true value `q` as `q / sqrt(mean(q²) + epsilon c²)`. Only the head
/// norms take it. A head with `c = 0` (all-zero weights) sums and normalizes to 0, which the
/// formula leaves out.
pub struct Scaled<X>(X);

impl<X> Scaled<X> {
    pub fn new(value: X) -> Self {
        Self(value)
    }

    /// The value, for an approved consumer.
    pub fn cancel(self) -> X {
        self.0
    }
}

/// A launch value in HBM that a loop reads on chip: [`Fill::fill`] lands the rows of one tile `T`
/// where the value enters the loop.
pub trait Fill<D, T> {
    /// The tile's rows, on chip.
    type Rows;
    fn fill(&self, context: &mut D, tile: T) -> Self::Rows;
}

/// A launch value in HBM that a loop writes from chip: [`Spill::spill`] stores the rows `R` of
/// one tile `T` where they leave the loop.
pub trait Spill<D, R, T> {
    fn spill(&mut self, context: &mut D, rows: R, tile: T);
}

/// The tiles of an iteration space that cover a launch's live rows: a loop of [`Tiling::count`]
/// trips, trip `index` on [`Tiling::get`]`(index)`.
pub trait Tiling: Copy {
    type Tile;
    fn count(self) -> usize;
    fn get(self, index: usize) -> Self::Tile;
}

impl<S: Space> Tiling for Tiles<S> {
    type Tile = (usize, Self);

    fn count(self) -> usize {
        self.count()
    }

    fn get(self, index: usize) -> Self::Tile {
        (index, self.tile(index))
    }
}

/// One loop over `tiles`: each trip runs the modules fused into it on one tile, so the values
/// between them stay on chip.
pub fn fuse<D, I: Tiling, T>(
    context: &mut D,
    (mut state, tiles): (T, I),
    mut trip: impl FnMut(&mut D, T, I::Tile) -> T,
) -> T {
    for index in 0..tiles.count() {
        state = trip(context, state, tiles.get(index));
    }
    state
}

/// A part of the step that a launch reads once: a read of launch metadata holds the command issue
/// until the DMA queue drains, so each part is read at the launch's start, or at the stage boundary
/// where its binding wants that drain, and its value handed on.
/// It is not `Clone`: a part read twice does not compile.
pub struct Once<T>(T);

impl<T> Once<T> {
    pub fn new(part: T) -> Self {
        Self(part)
    }

    pub fn read<V>(self, read: impl FnOnce(T) -> V) -> V {
        read(self.0)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A step's count, to upload.
    pub(crate) fn count(value: usize) -> HostTensor<i32, m![1]> {
        HostTensor::from_vec([i32::try_from(value).expect("counts fit I32")])
    }

    axes![Eight = 8, SixtyFour = 64];

    impl Space for Eight {
        type Envelope = SixtyFour;
    }

    /// Restricted domains cover only their intersection with the parent's live rows.
    #[test]
    fn restricts_live_rows() {
        for live in [0, 1, 7, 8, 9, 63, 64] {
            for start in 0..=SixtyFour::SIZE {
                for rows in [0, 1, 7, 8, 9, 64] {
                    let tiles = Tiles::<Eight>::over(live).subrange(start, rows);
                    let mut covered = Vec::new();
                    tiles.cover(
                        &mut covered,
                        (
                            |out| out.extend(0..SixtyFour::SIZE),
                            |out, at| out.extend(at * Eight::SIZE..(at + 1) * Eight::SIZE),
                            |out, at| out.push(at),
                        ),
                    );
                    let begin = start.min(live);
                    let end = begin + rows.min(live - begin);
                    assert_eq!(covered, (begin..end).collect::<Vec<_>>());
                    let nested = tiles.subrange(1, 64);
                    assert_eq!(nested.start(), begin + 1.min(end - begin));
                    assert_eq!(nested.rows(), (end - begin).saturating_sub(1));
                    for at in 0..tiles.count() {
                        let tile = tiles.tile(at);
                        assert_eq!(tile.start(), begin + at * Eight::SIZE);
                        assert_eq!(tile.rows(), (end - tile.start()).min(Eight::SIZE));
                    }
                }
            }
        }
    }

    /// A transfer of the whole envelope, of a whole tile or of one row, by the first row it moves.
    #[derive(Debug, PartialEq)]
    enum Transfer {
        Every,
        Tile(usize),
        Row(usize),
    }

    /// Each live row is covered once: by one transfer when every row is live, else by a whole
    /// tile while one fits and singly in the partial last tile. No row at or past the live count
    /// is covered, and no rows cover nothing.
    #[test]
    fn covers_each_live_row_once() {
        for rows in [0, 1, 7, 8, 9, 63, 64] {
            let tiles = Tiles::<Eight>::over(rows);
            let mut transfers = Vec::new();
            tiles.cover(
                &mut transfers,
                (
                    |transfers: &mut Vec<_>| transfers.push(Transfer::Every),
                    |transfers: &mut Vec<_>, tile| transfers.push(Transfer::Tile(tile * 8)),
                    |transfers: &mut Vec<_>, row| transfers.push(Transfer::Row(row)),
                ),
            );
            let covered: Vec<usize> = transfers
                .iter()
                .flat_map(|transfer| match *transfer {
                    Transfer::Every => 0..SixtyFour::SIZE,
                    Transfer::Tile(start) => start..start + 8,
                    Transfer::Row(row) => row..row + 1,
                })
                .collect();
            assert_eq!(covered, (0..rows).collect::<Vec<_>>(), "{rows} rows");
            let singles = transfers
                .iter()
                .filter(|transfer| matches!(transfer, Transfer::Row(_)))
                .count();
            assert_eq!(singles, rows % 8, "{rows} rows");
            assert_eq!(
                transfers.contains(&Transfer::Every),
                rows == SixtyFour::SIZE,
                "{rows} rows"
            );
            assert_eq!(tiles.count(), rows.div_ceil(8), "{rows} rows");
        }
    }

    /// Trips of `SIZE` rows over `rows` live rows, each whole but the last.
    #[derive(Clone, Copy)]
    struct Trips(usize);

    impl Tiling for Trips {
        type Tile = (usize, usize);

        fn count(self) -> usize {
            self.0.div_ceil(4)
        }

        fn get(self, index: usize) -> (usize, usize) {
            (4 * index, (self.0 - 4 * index).min(4))
        }
    }

    /// A fused loop runs every module of a trip on its tile before the next trip starts.
    #[test]
    fn fuse_runs_each_trip_on_its_tile() {
        let trace = fuse(
            &mut (),
            (Vec::new(), Trips(6)),
            |_, mut trace: Vec<_>, (start, rows)| {
                trace.push(("norm", start, rows));
                trace.push(("add", start, rows));
                trace
            },
        );
        assert_eq!(
            trace,
            [("norm", 0, 4), ("add", 0, 4), ("norm", 4, 2), ("add", 4, 2)]
        );
    }
}
