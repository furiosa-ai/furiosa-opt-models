//! The packed domain: rows of short histories, two rows a group of four-slice columns.
//!
//! - A cluster's 256 slices are 64 columns of four token slices; a block's 16 tokens go four to
//!   each token slice of one column, so a slice holds 64 slots of the cluster's four KV heads.
//! - A launch picks one group span for every packed item, from 4 to 256 slices, and every group
//!   holds two query lanes, two rows' queries scored against the group's one key stream; the host
//!   may leave a lane empty. A lane's item deals its blocks to its group's columns a block round
//!   each, the second lane's from a launch-wide round on.
//! - Every wave runs one body: 64 columns gathered, both lanes scored and normalized over each
//!   group, each lane's row stored from the group's first column; the wave count, the span, the
//!   scored slots and each lane's limits are launch values.
//! - Two consecutive waves share one query transfer, one limit transfer and one row store, so a
//!   single small transfer sits between two histories.
//! - A packed row is one item: a row longer than a cluster's columns hold is wide.

use super::*;
use crate::specialize::attention::WAVES;
use std::io;

mod attend;
pub(super) mod plan;

axes![Column = 64, Round = 16, Slot = 64, Pack = 2, Depth = 6];

/// `log2` of the smallest group, one column.
const LEAST: usize = 2;
/// Rows past the attended rows' last 256 take no writes; each column lane of each wave of a pair
/// owns one of them.
pub(super) const SINKS: usize = 2 * Column::SIZE * Pack::SIZE;

/// A launch's packed rows as items, planned on the host: the launch counts as loop extents and
/// the tables the packed body reads. [`plan::Plan`] makes the arrays.
#[derive(DeviceSend)]
pub(super) struct Packed<KvHeads: AxisName, Head: AxisName> {
    waves: Value,
    depth: Value,
    units: Value,
    tables: Tables<Waves>,
    /// Each lane's key position of every slot of each token slice, which each layer writes for its
    /// waves: a lane sees a slot whose position is below its column's limit.
    keys: HbmTensor<i32, Chip, m![Pack, Block / 4, Slot]>,
    /// Every wave's limit of each column lane, a flit each, which a pair of waves writes together.
    limits: Limits<Waves>,
    geometry: PhantomData<(KvHeads, Head)>,
}

/// The waves of every launch's packed domain.
type Waves = super::Waves<WAVES>;

type Limits<Waves> = HbmTensor<i32, Chip, m![Waves, Column, Pack, 1 # 8]>;

/// The counts of a [`Packed`], which a launch reads once, at its start.
#[derive(Clone, Copy)]
pub(crate) struct Shape {
    pub waves: usize,
    /// `log2` of a group's slices.
    pub depth: usize,
    /// The eight-slot units a slice scores: its longest column's rounds, rounded up.
    pub units: usize,
}

/// Each wave's columns: the pool byte offset of every block round they fill and the rounds they
/// fill, and for each lane the byte offsets of its query and output rows.
#[derive(DeviceSend)]
pub(crate) struct Tables<Waves: AxisName> {
    pub entries: HbmTensor<i32, Chip, m![Waves, Column, Round]>,
    pub lengths: HbmTensor<i32, Chip, m![Waves, Column]>,
    pub queries: HbmTensor<i32, Chip, m![Waves, Column, Pack]>,
    pub outputs: HbmTensor<i32, Chip, m![Waves, Column, Pack]>,
}

impl<KvHeads: AxisName, Head: AxisName> Packed<KvHeads, Head> {
    /// The items whose arrays [`plan::Plan::reads`] lists, as `next` names each in turn.
    pub(super) fn bind(next: &mut impl FnMut() -> io::Result<HbmBuffer>) -> io::Result<Self> {
        use crate::tensor::memory::alias;
        Ok(Self {
            waves: alias(&next()?, ..)?,
            depth: alias(&next()?, ..)?,
            units: alias(&next()?, ..)?,
            tables: Tables {
                entries: alias(&next()?, ..)?,
                lengths: alias(&next()?, ..)?,
                queries: alias(&next()?, ..)?,
                outputs: alias(&next()?, ..)?,
            },
            limits: alias(&next()?, ..)?,
            keys: alias(&next()?, ..)?,
            geometry: PhantomData,
        })
    }

    /// The launch's counts.
    pub(super) fn shape(&self) -> Shape {
        Shape {
            waves: self.waves.extent::<m![Waves]>(),
            depth: self.depth.extent::<m![Depth]>() + LEAST,
            units: self.units.extent::<m![Slot / 8]>(),
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
    /// An item tensor on the token slices of a cluster's 64 columns, `E` its elements.
    type PackedDm<D: Scalar, E: M> = Self::ItemDm<m![Column, Block / 4], D, E>;
    type PackedVrf<D: MaterializableScalar, E: M> = Self::ItemVrf<m![Column, Block / 4], D, E>;
    type PackedInDm =
        Self::PackedDm<bf16, m![Slot, KeyValue, KvHeads % ({ Local<KvHeads> }), Head]>;
    type PackedKeys = Self::PackedDm<i32, m![Pack, Slot]>;
    /// Four consecutive waves' lane limits of each column, wave `w`'s at `w % 4`, on each of its
    /// token slices: a pair of waves arrives while the pair before it is still being read.
    type PackedLimits<Waves: AxisName> = Self::PackedDm<i32, m![Waves % 4, Pack, 1 # 8]>;
    /// Four consecutive waves' queries, wave `w`'s at `w % 4`, a column's two rows of one head a
    /// token slice: a pair of waves arrives while the pair before it is still being read.
    type PackedQueries<Waves: AxisName> = DmTensor<
        bf16,
        Chip,
        Self::ItemCluster,
        m![Column, KvHeads % ({ Local<KvHeads> }) # 4],
        m![Waves % 4, Pack, Groups, Head],
    >;
    /// The queries and lane limits of four consecutive waves.
    type PackedLanes<Waves: AxisName> = (Self::PackedQueries<Waves>, Self::PackedLimits<Waves>);
    /// A slice's query rows: each lane's groups of every KV head the slice holds.
    type PackedRows = m![Pack, KvHeads % ({ Local<KvHeads> }), Groups];
    /// Two consecutive waves' rounded rows, wave `w`'s at `w % 2`, stored together.
    type PackedRounded<Waves: AxisName> =
        Self::PackedDm<bf16, m![Waves % 2, { Self::PackedRows }, Head]>;
}
