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
use std::io;

/// `value` and its live rows: `rows` rows from row `start` of an envelope of `R` rows, never past
/// it. A leaf computes those rows of `value`; a row outside them is padding nothing reads.
pub struct Live<X, R> {
    value: X,
    rows: usize,
    start: usize,
    envelope: PhantomData<R>,
}

/// A zero-based prefix made of whole `B`-row blocks within capacity `R`.
/// Its private count is bounded and aligned by construction. It describes computed work,
/// including initialized padding; it does not identify semantically valid model tokens.
pub struct BlockPrefix<X, R, B> {
    value: X,
    blocks: usize,
    shape: PhantomData<(R, B)>,
}

impl<R: AxisName, B: AxisName> BlockPrefix<(), R, B> {
    /// A statically known work extent inside the same allocation envelope. Invalid bounds or
    /// alignment fail at compilation; runtime requests use the checked host count or HBM reader.
    pub const fn fixed<Count: AxisName>() -> Self {
        const {
            assert!(B::SIZE > 0 && R::SIZE.is_multiple_of(B::SIZE));
            assert!(Count::SIZE <= R::SIZE && Count::SIZE.is_multiple_of(B::SIZE));
        }
        Self {
            value: (),
            blocks: Count::SIZE / B::SIZE,
            shape: PhantomData,
        }
    }

    /// Check a host row count without rounding it or discarding an invalid suffix.
    pub fn try_over(rows: usize) -> Option<Self> {
        const {
            assert!(B::SIZE > 0 && R::SIZE.is_multiple_of(B::SIZE));
        }
        (rows <= R::SIZE && rows.is_multiple_of(B::SIZE)).then(|| Self {
            value: (),
            blocks: rows / B::SIZE,
            shape: PhantomData,
        })
    }

    /// Read a bounded count of **blocks**, not rows, from HBM. The host validates the request
    /// before upload; bounding units here preserves the type's invariant without a device panic.
    pub fn read(count: &Value<impl M>) -> Self {
        const {
            assert!(B::SIZE > 0 && R::SIZE.is_multiple_of(B::SIZE));
        }
        Self {
            value: (),
            blocks: count.extent::<m![R / B]>(),
            shape: PhantomData,
        }
    }
}

impl<X, R: AxisName, B: AxisName> BlockPrefix<X, R, B> {
    pub fn value(&self) -> &X {
        &self.value
    }

    pub fn blocks(&self) -> usize {
        self.blocks
    }

    pub fn rows(&self) -> usize {
        self.blocks * B::SIZE
    }

    /// The same initialized work as an ordinary live interval, always starting at zero.
    pub fn interval(&self) -> Live<(), R> {
        Live::first(self.rows())
    }

    pub fn with<Y>(&self, value: Y) -> BlockPrefix<Y, R, B> {
        BlockPrefix {
            value,
            blocks: self.blocks,
            shape: PhantomData,
        }
    }

    pub fn into_value(self) -> X {
        self.value
    }
}

/// A nonempty, evenly spaced range of fixed row extents, including both endpoints.
/// A caller supplies this type once for its envelope.
pub trait Extents {
    /// The smallest extent.
    type First: AxisName;
    /// The largest extent.
    type Last: AxisName;
    /// The positive row increment between extents.
    type Step: AxisName;
}

/// Fixed row extents from `First` through `Last` in increments of `Step`.
pub struct ExtentRange<First, Last, Step>(PhantomData<(First, Last, Step)>);

impl<First: AxisName, Last: AxisName, Step: AxisName> Extents for ExtentRange<First, Last, Step> {
    type First = First;
    type Last = Last;
    type Step = Step;
}

/// The next fixed extent in a checked range.
#[derive(Clone, Debug)]
pub(crate) struct Next<Current, Step>(PhantomData<(Current, Step)>);

impl<Current: AxisName, Step: AxisName> AxisName for Next<Current, Step> {
    const NAME: Ident = Ident::new("PrefixExtent");
    const SIZE: usize = Current::SIZE + Step::SIZE;
}

/// Zero-based work whose row count belongs to `G`, in whole `B`-row blocks.
/// Construct on the host before upload; device readers retain the same checked extent set.
pub struct PrefixPlan<X, R, G, B> {
    prefix: BlockPrefix<X, R, B>,
    extents: PhantomData<G>,
}

impl<R: AxisName, G: Extents, B: AxisName> PrefixPlan<(), R, G, B> {
    /// Refuse work outside the fixed extent set before allocating device metadata.
    pub fn new(prefix: BlockPrefix<(), R, B>) -> io::Result<Self> {
        const { assert!(valid::<R, G, B>()) };
        if !contains::<G>(prefix.rows()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "prefix work is not one of the fixed extents",
            ));
        }
        Ok(Self {
            prefix,
            extents: PhantomData,
        })
    }
}

impl<X, R: AxisName, G: Extents, B: AxisName> PrefixPlan<X, R, G, B> {
    /// Borrow the same bounded, aligned block prefix.
    pub fn prefix(&self) -> &BlockPrefix<X, R, B> {
        &self.prefix
    }

    /// Bind another value without changing the checked work or its extent set.
    pub fn with<Y>(&self, value: Y) -> PrefixPlan<Y, R, G, B> {
        PrefixPlan {
            prefix: self.prefix.with(value),
            extents: PhantomData,
        }
    }
}

/// Immutable uploaded metadata; only a checked [`PrefixPlan`] can construct it.
#[derive(DeviceSend)]
pub struct PrefixInputs<R, G, B, Chips: M> {
    count: Value<Chips>,
    shape: PhantomData<(R, G, B)>,
}

impl<R: AxisName, G: Extents, B: AxisName, Chips: M> PrefixInputs<R, G, B, Chips> {
    /// Upload the checked block count once; its immutable owner must outlive every reader.
    pub async fn upload(device: &mut Device, plan: &PrefixPlan<(), R, G, B>) -> io::Result<Self> {
        let count = HostTensor::<i32, m![1]>::from_vec([plan.prefix.blocks() as i32])
            .to_hbm::<Chips, m![1]>(&mut device.pdma)
            .await
            .map_err(io::Error::other)?;
        Ok(PrefixInputs {
            count,
            shape: PhantomData,
        })
    }

    /// Decode the privately owned count once, retaining the host-checked extent set.
    pub fn read(&self) -> PrefixPlan<(), R, G, B> {
        PrefixPlan {
            prefix: BlockPrefix::read(&self.count),
            extents: PhantomData,
        }
    }
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

/// Equal parts of `value`, each with the same live interval inside its `R`-row slot.
/// The parts keep their capacity stride: their live rows need not form one contiguous prefix.
/// `N` is the number of logical parts, independent of where their storage is placed.
pub struct Partitioned<X, R, const N: usize> {
    value: X,
    interval: Live<(), R>,
}

impl<X, R: AxisName, const N: usize> Partitioned<X, R, N> {
    /// Retain `value` and the live interval shared by its `N` parts.
    pub fn new(value: X, interval: Live<(), R>) -> Self {
        const { assert!(N > 0, "a partitioned value has at least one part") };
        Self { value, interval }
    }

    /// The value in capacity-strided part order.
    pub fn value(&self) -> &X {
        &self.value
    }

    /// The live interval within every part's slot, not a count over the joined value.
    pub fn interval(&self) -> Live<(), R> {
        self.interval
    }

    /// These same parts and intervals, holding another value.
    pub fn with<Y>(&self, value: Y) -> Partitioned<Y, R, N> {
        Partitioned::new(value, self.interval)
    }

    /// The value without its per-part live intervals.
    pub fn into_value(self) -> X {
        self.value
    }
}

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

const fn valid<R: AxisName, G: Extents, B: AxisName>() -> bool {
    B::SIZE > 0
        && R::SIZE.is_multiple_of(B::SIZE)
        && R::SIZE / B::SIZE <= i32::MAX as usize
        && G::First::SIZE > 0
        && G::First::SIZE <= G::Last::SIZE
        && G::Last::SIZE <= R::SIZE
        && G::First::SIZE.is_multiple_of(B::SIZE)
        && G::Step::SIZE > 0
        && G::Step::SIZE.is_multiple_of(B::SIZE)
        && (G::Last::SIZE - G::First::SIZE).is_multiple_of(G::Step::SIZE)
}

const fn contains<G: Extents>(rows: usize) -> bool {
    rows >= G::First::SIZE
        && rows <= G::Last::SIZE
        && (rows - G::First::SIZE).is_multiple_of(G::Step::SIZE)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::tensor::axis::{Eight, Rows, Sixteen, SixtyFour};

    #[test]
    fn refuses_prefix_work_outside_the_fixed_extents_before_upload() {
        type C = Rows<1280>;
        type G = ExtentRange<Rows<256>, Rows<1280>, Rows<512>>;
        for rows in [0, 512, 1024] {
            let prefix = BlockPrefix::<(), C, Rows<256>>::try_over(rows).unwrap();
            let error = PrefixPlan::<(), C, G, Rows<256>>::new(prefix)
                .err()
                .unwrap();
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
            assert_eq!(
                error.to_string(),
                "prefix work is not one of the fixed extents"
            );
        }
    }

    /// Block prefixes refuse nonaligned and oversized host counts instead of rounding them.
    #[test]
    fn checked_block_prefixes_preserve_bounds_and_units() {
        for rows in [0, 8, 16, 56, 64] {
            let prefix = BlockPrefix::<(), SixtyFour, Eight>::try_over(rows).unwrap();
            assert_eq!(prefix.rows(), rows);
            assert_eq!(prefix.blocks(), rows / 8);
            assert_eq!(
                (prefix.interval().start(), prefix.interval().rows()),
                (0, rows)
            );
            let rows = prefix.with("rows");
            assert_eq!(rows.value(), &"rows");
            assert_eq!(rows.into_value(), "rows");
        }
        for rows in [1, 7, 9, 63, 65, usize::MAX] {
            assert!(BlockPrefix::<(), SixtyFour, Eight>::try_over(rows).is_none());
        }
    }

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
