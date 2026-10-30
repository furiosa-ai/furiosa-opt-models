//! The wide domain: rows of longer histories, an item a group of 16-slice columns.
//!
//! - A launch picks one group span for every wide item from its longest row and its row count, from
//!   16 to 256 slices; each slice holds the item's KV heads of a cluster and a [`Time`] of keys.
//! - A group of `s` slices is `s / 16` columns of 16 token slices; an item deals its blocks to its
//!   columns four at a time in turn, each block's 16 tokens on the column's 16 slices.
//! - Every wave runs one body: 16 columns gathered, scored, normalized over each item's group and
//!   stored from the group's first column; the wave count and the span are launch values.
//! - Two consecutive waves share one query transfer and one row store, so a single small transfer
//!   sits between two histories.
//! - A row longer than an item takes consecutive items, and a merge after the waves combines
//!   their outputs into the row.

use super::*;
use crate::specialize::attention::{DECODES, INPUTS, SLOTS, landing};
use crate::tensor::axis::Rows;
use std::io;

use plan::Plan;

mod attend;
mod merge;
pub(super) mod plan;

axes![Column = 16, Depth = 5, Quad = 4, Eight = 8, Lane = 8];

/// A slice's slots of a wave's history, a token of each of its column's blocks a slot: as many as
/// [`landing`] leaves a cluster's KV heads.
#[derive(Clone, Debug)]
pub(super) struct Time<KvHeads, Head>(PhantomData<(KvHeads, Head)>);

impl<KvHeads: AxisName, Head: AxisName> AxisName for Time<KvHeads, Head> {
    const NAME: Ident = Ident::new("Time");
    const SIZE: usize = landing(Head::SIZE, LocalHeads::<KvHeads>::SIZE);
}

/// Every column's `Inner` slices, a PE's four columns bit-reversed: a history gather then visits a
/// slot's columns in table order, so each PE copies its index list in runs.
pub(super) type ColumnSlices<Inner> = m![Column / 4, Column % 2, Column % 4 / 2, { Inner }];

/// Wide item slots: every launch's.
type Columns = Rows<SLOTS>;
/// Wide rows: every launch's.
type Decodes = Rows<DECODES>;

/// Items a merge combines into one row.
#[derive(Clone, Debug)]
pub(super) struct Input;

impl AxisName for Input {
    const NAME: Ident = Ident::new("Input");
    const SIZE: usize = INPUTS;
}

/// The smallest group: one column.
const LEAST: usize = 4;
/// Rows past the attended rows' last 16 take no writes; each column of a wave owns one of them.
pub(super) const SINKS: usize = Column::SIZE;

/// A launch's wide rows as items, planned on the host: the launch counts as loop extents and the
/// tables the wide body reads, its waves `Waves`. [`plan::Plan`] makes the arrays.
#[derive(DeviceSend)]
pub(super) struct Wide<KvHeads: AxisName, Head: AxisName, Waves: AxisName> {
    waves: Value,
    depth: Value,
    units: Value,
    merges: Value,
    /// Each wave's columns' pool byte offset of every block they fill.
    entries: Entries<KvHeads, Head, Waves>,
    tables: Tables<Waves>,
    jobs: Jobs<Decodes, Input>,
    /// Each token slice's key positions, which each layer writes for its waves.
    keys: Keys<KvHeads, Head>,
    /// Every wave's column limits, a flit a column and each column's waves contiguous, which each
    /// layer writes at once; a launch writes and its chains read only its pairs of waves.
    limits: Limits<Waves>,
}

type Entries<KvHeads, Head, Waves> =
    HbmTensor<i32, Chip, m![Waves, { Symbol<Time<KvHeads, Head>> }, Column]>;
type Keys<KvHeads, Head> = HbmTensor<i32, Chip, m![Block, { Symbol<Time<KvHeads, Head>> }]>;
type Limits<Waves> = HbmTensor<i32, Chip, m![Column, Waves, 1 # 8]>;

/// The counts of a [`Wide`], which a launch reads once, at its start.
#[derive(Clone, Copy)]
pub(crate) struct Shape {
    pub waves: usize,
    pub depth: usize,
    /// The eight-slot units a slice scores: its item's longest column, rounded up.
    pub units: usize,
    /// Merge tiles of eight rows; none when every row fits one item.
    pub merges: usize,
}

/// Each wave's columns: their filled slots, keys past their own offset, query and output row
/// offsets, and their item's f32 output and log-sum-exp offsets for a merge.
#[derive(DeviceSend)]
pub(crate) struct Tables<Waves: AxisName> {
    pub lengths: HbmTensor<i32, Chip, m![Waves, Column]>,
    pub queries: HbmTensor<i32, Chip, m![Waves, Column]>,
    pub outputs: HbmTensor<i32, Chip, m![Waves, Column]>,
    pub partials: HbmTensor<i32, Chip, m![Waves, Column]>,
    pub scales: HbmTensor<i32, Chip, m![Waves, Column]>,
}

/// What a merge reads of each row, a row a flit: its first item, its items, and its index; and
/// each input's ordinal `q` within its job, for the eight jobs of a tile.
#[derive(DeviceSend)]
pub(crate) struct Jobs<Decodes: AxisName, Input: AxisName> {
    pub firsts: HbmTensor<i32, Chip, m![Decodes, 1 # 8]>,
    pub counts: HbmTensor<i32, Chip, m![Decodes, 1 # 8]>,
    pub rows: HbmTensor<i32, Chip, m![Decodes, 1 # 8]>,
    pub inputs: HbmTensor<i32, Chip, m![{ m![Eight, Input] }, 1 # 8]>,
}

impl<KvHeads: AxisName, Head: AxisName, Waves: AxisName> Wide<KvHeads, Head, Waves> {
    /// The items whose arrays [`plan::Plan::reads`] lists, as `next` names each in turn.
    pub(super) fn bind(next: &mut impl FnMut() -> io::Result<HbmBuffer>) -> io::Result<Self> {
        Ok(Self {
            waves: alias(&next()?, ..)?,
            depth: alias(&next()?, ..)?,
            units: alias(&next()?, ..)?,
            merges: alias(&next()?, ..)?,
            entries: alias(&next()?, ..)?,
            tables: Tables {
                lengths: alias(&next()?, ..)?,
                queries: alias(&next()?, ..)?,
                outputs: alias(&next()?, ..)?,
                partials: alias(&next()?, ..)?,
                scales: alias(&next()?, ..)?,
            },
            limits: alias(&next()?, ..)?,
            keys: alias(&next()?, ..)?,
            jobs: Jobs {
                firsts: alias(&next()?, ..)?,
                counts: alias(&next()?, ..)?,
                rows: alias(&next()?, ..)?,
                inputs: alias(&next()?, ..)?,
            },
        })
    }

    /// Bytes of each HBM allocation a layer's items of `G` groups a KV head make: every item's
    /// f32 output, and its log-sum-exp.
    pub(crate) const fn scratch<G: AxisName>() -> [usize; 2] {
        [
            <m![Columns, KvHeads, G, Head]>::SIZE * size_of::<f32>(),
            <m![Columns, KvHeads, G, Lane]>::SIZE * size_of::<f32>(),
        ]
    }

    /// The launch's counts.
    pub(crate) fn shape(&self) -> Shape {
        Shape {
            waves: self.waves.extent::<m![Waves]>(),
            depth: self.depth.extent::<m![Depth]>() + LEAST,
            units: self
                .units
                .extent::<m![{ Symbol<Time<KvHeads, Head>> } / 8]>(),
            merges: self.merges.extent::<m![Decodes / 8]>(),
        }
    }
}

impl<Waves: AxisName> Tables<Waves> {
    /// Wave `wave`'s rows of `table`.
    fn wave<Cells: M>(
        table: &HbmTensor<i32, Chip, m![Waves, { Cells }]>,
        wave: usize,
    ) -> HbmTensor<i32, Chip, Cells> {
        // SAFETY: one wave's row of a table nothing writes once derived.
        unsafe {
            table
                .view()
                .tile::<m![Waves], 1, m![Waves = 1 # Waves, { Cells }]>(wave)
                .alias()
        }
    }

    /// Waves `2 pair` and `2 pair + 1`'s rows of `table`.
    fn pair<Cells: M>(
        table: &HbmTensor<i32, Chip, m![Waves, { Cells }]>,
        pair: usize,
    ) -> HbmTensor<i32, Chip, m![Waves % 2, { Cells }]> {
        // SAFETY: two waves' rows of a table nothing writes once derived.
        unsafe {
            table
                .view()
                .tile::<m![Waves / 2], 1, m![Waves / 2 = 1 # (Waves / 2), Waves % 2, { Cells }]>(
                    pair,
                )
                .alias()
        }
    }
}

impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName> Attention<KvHeads, Groups, Head> {
    /// An item tensor on the token slices of a cluster's 16 columns, `E` its elements.
    type WideDm<D: Scalar, E: M> = Self::ItemDm<ColumnSlices<m![Block]>, D, E>;
    type WideVrf<D: MaterializableScalar, E: M> = Self::ItemVrf<ColumnSlices<m![Block]>, D, E>;
    type WideTime = Symbol<Time<KvHeads, Head>>;
    /// The eight-slot units a slice scores.
    type WideUnits = m![{ Self::WideTime } / 8];
    type WideInDm =
        Self::WideDm<bf16, m![{ Self::WideTime }, KeyValue, KvHeads % ({ Local<KvHeads> }), Head]>;
    type WideKeys = Self::WideDm<i32, Self::WideTime>;
    /// A column's limit on each of its token slices.
    type WideLimit = Self::WideDm<i32, m![1 # 8]>;
    /// Every wave's column limits on each column's first slice.
    type WideLimits<Waves: AxisName> =
        DmTensor<i32, Chip, Self::ItemCluster, ColumnSlices<m![1 # 16]>, m![Waves, 1 # 8]>;
    /// Four consecutive waves' queries, wave `w` at `w % 4`, a (head, group) row a column and token
    /// slice, so a pair of waves arrives while the pair before is read.
    type WideQueries<Waves: AxisName> = DmTensor<
        bf16,
        Chip,
        Self::ItemCluster,
        ColumnSlices<m![KvHeads % ({ Local<KvHeads> }) # 4, Groups # 4]>,
        m![Waves % 4, Head],
    >;
    /// A slice's query rows: its item's groups of every KV head the slice holds.
    type WideRows = m![KvHeads % ({ Local<KvHeads> }), Groups];
    /// Two consecutive waves' rounded rows, wave `w`'s at `w % 2`, stored together.
    type WideRounded<Waves: AxisName> = Self::WideDm<bf16, m![Waves % 2, { Self::WideRows }, Head]>;
    /// What a merge keeps of a wave: each item's f32 output, mass and largest score.
    type WideKept = Self::Kept<ColumnSlices<m![Block]>, Self::WideRows>;
    type WideLocal = Self::WideDm<f32, m![KvHeads % ({ Local<KvHeads> }), Groups, 1 # 8]>;
    /// Each item's f32 output and log-sum-exp, a slot an item, as a merge reads them; a
    /// log-sum-exp's register is stored whole, its lanes named, as a transfer moves whole flits.
    type WidePartials<Columns: AxisName> = HbmTensor<f32, Chip, m![Columns, KvHeads, Groups, Head]>;
    type WideScales<Columns: AxisName> = HbmTensor<f32, Chip, m![Columns, KvHeads, Groups, Lane]>;
    /// A merge tile: eight jobs by 32 slices, each slice two consecutive inputs of its job.
    type MergeDm<Input: AxisName, E: M> =
        DmTensor<f32, Chip, Self::ItemCluster, m![Eight, Input / 2], E>;
}
