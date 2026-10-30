//! The live rows of a launch: an envelope of rows compiled once, and the rows each call fills.
//!
//! - The step reads the live rows once from a count in HBM ([`Live::read`]); leaves take the rows,
//!   never the count.
//! - A leaf derives every loop count from the live rows, so a row past them costs no iteration and
//!   takes no effect: 9 live rows of a 64-row envelope are two tiles of 8 rows, one whole.
//! - Live rows may start past the envelope's first row; a leaf finds the groups of rows they touch
//!   with [`Live::groups`].
//! - A module that reads its live rows in tiles of its own sizes runs its work on each
//!   ([`Live::tiles`]).

use super::scalar::Value;
use core::marker::PhantomData;
use furiosa_opt_std::prelude::*;

/// `value` and its live rows: `rows` rows from row `start` of an envelope of `R` rows, never past
/// it. A leaf computes those rows of `value`; a row outside them is padding nothing reads.
pub struct Live<X, R> {
    value: X,
    rows: usize,
    start: usize,
    envelope: PhantomData<R>,
}

/// Written out, as a derive would ask the envelope to be `Clone` too.
impl<X: Clone, R> Clone for Live<X, R> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            rows: self.rows,
            start: self.start,
            envelope: PhantomData,
        }
    }
}

impl<X: Copy, R> Copy for Live<X, R> {}

impl<R: AxisName> Live<(), R> {
    /// The live rows a step's count names, at most `R`; leaves take these rows, never the count,
    /// so only the code that binds the step reads it.
    pub fn read(count: &Value<impl M>) -> Self {
        Self::first(count.extent::<m![{ Symbol<R> }]>())
    }

    /// Every row of the envelope.
    pub fn all() -> Self {
        Self::first(R::SIZE)
    }

    /// The first `rows` rows of a count the host holds, which must be at most `R`.
    pub fn over(rows: usize) -> Self {
        assert!(
            rows <= R::SIZE,
            "{rows} live rows exceed the {} of the envelope",
            R::SIZE
        );
        Self::first(rows)
    }

    /// The first `rows` rows, `rows` at most `R`.
    fn first(rows: usize) -> Self {
        Self {
            value: (),
            rows,
            start: 0,
            envelope: PhantomData,
        }
    }
}

impl<X, R: AxisName> Live<X, R> {
    /// The value these rows are live in.
    pub fn value(&self) -> &X {
        &self.value
    }

    /// The value, without its live rows.
    pub fn into_value(self) -> X {
        self.value
    }

    /// How many rows are live.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The envelope row the live rows start at.
    pub fn start(&self) -> usize {
        self.start
    }

    /// These live rows, holding `value`.
    pub fn with<Y>(&self, value: Y) -> Live<Y, R> {
        Live {
            value,
            rows: self.rows,
            start: self.start,
            envelope: PhantomData,
        }
    }

    /// These rows restricted to a relative row interval; rows outside it remain padding.
    pub fn subrange(&self, start: usize, rows: usize) -> Live<(), R> {
        let start = start.min(self.rows);
        Live {
            value: (),
            rows: rows.min(self.rows - start),
            start: self.start + start,
            envelope: PhantomData,
        }
    }

    /// Live tile `index` of `T` rows, for `index` below [`Self::count`]: the tiles start with
    /// these rows and are whole but the last.
    pub fn tile<T: AxisName>(&self, index: usize) -> Live<(), R> {
        let tiles = const {
            assert!(T::SIZE > 0 && R::SIZE.is_multiple_of(T::SIZE));
            R::SIZE / T::SIZE
        };
        let whole = (self.count::<T>() + tiles - 1 - index) / (tiles + 1);
        let tail = (self.rows + T::SIZE - 1) % T::SIZE + 1;
        Live {
            value: (),
            rows: tail + whole * (T::SIZE - tail),
            start: self.start + index * T::SIZE,
            envelope: PhantomData,
        }
    }

    /// The tiles of `T` rows holding live rows, the last possibly partial. Rounded up by addition,
    /// so a loop over them is allocated for exactly the envelope's tiles.
    pub fn count<T: AxisName>(&self) -> usize {
        (self.rows + T::SIZE - 1) / T::SIZE
    }

    /// The groups of `L` envelope rows the live rows touch, as `(first, count)`; none for no rows.
    /// The count subtracts nothing, so a loop over it stays bounded for a runtime start.
    pub fn groups<L: AxisName>(&self) -> (usize, usize) {
        const { assert!(L::SIZE > 0 && R::SIZE.is_multiple_of(L::SIZE)) };
        let count = if self.rows == 0 {
            0
        } else {
            (self.start % L::SIZE + self.rows + L::SIZE - 1) / L::SIZE
        };
        (self.start / L::SIZE, count)
    }

    /// Runs `every` when all `R` rows are live; otherwise `whole` on each tile of `T` rows whose
    /// rows are all live, then `row` on each other live row, each row's effect issuing once.
    pub fn cover<T: AxisName, D>(
        &self,
        context: &mut D,
        (every, mut whole, mut row): (
            impl FnOnce(&mut D),
            impl FnMut(&mut D, usize),
            impl FnMut(&mut D, usize),
        ),
    ) {
        if self.rows == R::SIZE {
            every(context);
        } else {
            let first = (T::SIZE - self.start % T::SIZE) % T::SIZE;
            let first = first.min(self.rows);
            for lane in 0..first {
                row(context, self.start + lane);
            }
            let tiles = (self.rows - first) / T::SIZE;
            for tile in 0..tiles {
                whole(context, (self.start + first) / T::SIZE + tile);
            }
            for lane in 0..(self.rows - first) % T::SIZE {
                row(context, self.start + first + tiles * T::SIZE + lane);
            }
        }
    }

    /// Covers these rows with tiles of `Large` rows while more than `Medium` remain, then one of
    /// `Medium` rows if more than `Small` remain, else one of `Small` rows if any: a loop each.
    pub(crate) fn tiles<D, Large: AxisName, Medium: AxisName, Small: AxisName>(
        &self,
        context: &mut D,
        work: &mut (impl OnTile<D, Large> + OnTile<D, Medium> + OnTile<D, Small>),
    ) {
        const {
            assert!(Large::SIZE == 2 * Medium::SIZE && Small::SIZE < Medium::SIZE);
            assert!(Large::SIZE.is_multiple_of(Small::SIZE));
        };
        // Spelled without comparisons, as the rows are a runtime count; no step goes negative.
        let count = (self.rows + const { Medium::SIZE - 1 }) / const { Large::SIZE };
        let rest = (self.rows % const { Large::SIZE } + const { Large::SIZE - Small::SIZE - 1 })
            % const { Large::SIZE };
        for index in 0..count {
            OnTile::<D, Large>::run(work, context, index);
        }
        for _ in 0..(rest + const { Large::SIZE - Medium::SIZE + Small::SIZE })
            % const { Large::SIZE }
            / const { Large::SIZE - Medium::SIZE + Small::SIZE }
        {
            OnTile::<D, Medium>::run(work, context, const { Large::SIZE / Medium::SIZE } * count);
        }
        for _ in 0..rest / const { Large::SIZE - Small::SIZE } {
            OnTile::<D, Small>::run(work, context, const { Large::SIZE / Small::SIZE } * count);
        }
    }
}

/// What a module that reads its live rows tile by tile runs on tile `index` of `W` rows, counted
/// in tiles of `W` from the first live row.
pub(crate) trait OnTile<D, W> {
    fn run(&mut self, context: &mut D, index: usize);
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::tensor::axis::{Eight, Sixteen, SixtyFour};

    /// A step's count, to upload.
    pub(crate) fn count(value: usize) -> HostTensor<i32, m![1]> {
        HostTensor::from_vec([i32::try_from(value).expect("counts fit I32")])
    }

    /// The live rows `(start, rows)` of a 16-row envelope that every row op is tested on, besides
    /// a prefix: one starting at a group boundary, one crossing it.
    pub(crate) fn origins() -> [Live<(), Sixteen>; 2] {
        [(8, 2), (7, 2)].map(|(start, rows)| Live::over(16).subrange(start, rows))
    }

    /// Restricted rows cover only their intersection with the parent's live rows; a value keeps
    /// them, and their tiles start and end with them.
    #[test]
    fn restricts_live_rows() {
        for live in [0, 1, 7, 8, 9, 63, 64] {
            for start in 0..=SixtyFour::SIZE {
                for rows in [0, 1, 7, 8, 9, 64] {
                    let restricted = Live::<(), SixtyFour>::over(live)
                        .with('x')
                        .subrange(start, rows);
                    let mut covered = Vec::new();
                    restricted.cover::<Eight, _>(
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
                    assert_eq!(
                        (restricted.start(), restricted.rows()),
                        (begin, end - begin)
                    );
                    let nested = restricted.subrange(1, 64);
                    let skipped = 1.min(end - begin);
                    assert_eq!(
                        (nested.start(), nested.rows()),
                        (begin + skipped, end - begin - skipped)
                    );
                    assert_eq!(
                        restricted.count::<Eight>(),
                        (end - begin).div_ceil(Eight::SIZE)
                    );
                    for at in 0..restricted.count::<Eight>() {
                        let tile = restricted.tile::<Eight>(at);
                        assert_eq!(tile.start(), begin + at * Eight::SIZE);
                        assert_eq!(tile.rows(), (end - tile.start()).min(Eight::SIZE));
                    }
                }
            }
        }
    }

    /// The groups live rows touch are those holding one of their rows, counted from the first
    /// such group; rows that are not live touch none.
    #[test]
    fn touches_the_groups_holding_live_rows() {
        for start in 0..=SixtyFour::SIZE {
            for rows in 0..=SixtyFour::SIZE - start {
                let live = Live::<(), SixtyFour>::over(64).subrange(start, rows);
                let touched: Vec<usize> = (start..start + rows).map(|row| row / 8).fold(
                    Vec::new(),
                    |mut groups, group| {
                        if groups.last() != Some(&group) {
                            groups.push(group);
                        }
                        groups
                    },
                );
                let (first, count) = live.groups::<Eight>();
                assert_eq!(
                    (first..first + count).collect::<Vec<_>>(),
                    touched,
                    "{start}, {rows}"
                );
            }
        }
        let [aligned, crossing] = origins();
        assert_eq!(
            (aligned.groups::<Eight>(), crossing.groups::<Eight>()),
            ((1, 1), (0, 2))
        );
    }
}
