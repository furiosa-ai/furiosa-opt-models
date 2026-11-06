//! Index operands: byte offsets of indexed rows and the block tables, extents and visibility of
//! a paged history, which a gather or scatter and the attention over the history read.

use crate::Op;
use crate::ops::attention::gqa::{Cache, Wrapped};
use crate::ops::index::gather::RowStorage;
use crate::ops::linear::Groups;
use crate::tensor::cache::Block;
use crate::tensor::scalar::{self, Packet, Value};
use crate::tensor::{
    axis::{Lanes, Single},
    live::Live,
    mapping::{BroadcastClusters as Cluster, BroadcastSlices as Slices, Packets, Rows as Layout},
    parallel::Chip,
};
pub use derive::Derive;
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

mod derive;

const _: () = assert!(Cache::SIZE.is_power_of_two());

/// Byte offset from the source allocation or tensor base.
pub type Offset<Chips = m![1]> = HbmTensor<i32, Chips, m![Single]>;

/// Convert a row index to the exact, nonnegative I32 byte offset of a row shaped `S` with elements
/// `D`, laid out as `O`. The caller supplies indices whose product with the row size fits I32.
#[derive(DeviceSend)]
pub struct ByteOffset<D, S, O = m![Single]>(PhantomData<(D, S, O)>);

impl<D, S, O> Default for ByteOffset<D, S, O> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// One index.
impl<D: Scalar, S: M, O: M, Chips: M> Op<&HbmTensor<i32, Chips, m![1]>> for ByteOffset<D, S, O> {
    type Output = HbmTensor<i32, Chips, O>;

    fn forward(&self, device: &mut Device, input: &HbmTensor<i32, Chips, m![1]>) -> Self::Output {
        let input = scalar::packet(device, input);
        scalar::map(device, &input, |index| {
            index.vector_fxp(
                FxpBinaryOp::MulInt,
                const { (S::SIZE * D::BITS / 8) as i32 },
            )
        })
        .to_hbm(&mut device.tdma)
    }
}

/// The ids of `Rows` tokens, a packet of eight per row group.
impl<D: Scalar, S: M, O: M, Rows: AxisName, Chips: M> Op<&HbmTensor<i32, Chips, Packets<Rows>>>
    for ByteOffset<D, S, O>
{
    type Output = HbmTensor<i32, Chips, O>;

    fn forward(
        &self,
        device: &mut Device,
        ids: &HbmTensor<i32, Chips, Packets<Rows>>,
    ) -> Self::Output {
        use crate::tensor::mapping::{BroadcastClusters as Cluster, BroadcastSlices};

        let ids: DmTensor<i32, Chips, Cluster, BroadcastSlices, Packets<Rows>> =
            ids.to_dm(&mut device.tdma);
        let offsets: DmTensor<i32, Chips, Cluster, BroadcastSlices, Packets<Rows>> = device
            .main
            .begin(ids.view())
            .fetch::<m![{ Groups<Rows> }], m![{ Layout<Rows>::Lanes } # 8]>()
            .fetch_cast::<i32>()
            .collect::<m![{ Groups<Rows> }], m![{ Layout<Rows>::Lanes } # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(
                FxpBinaryOp::MulInt,
                const { (S::SIZE * D::BITS / 8) as i32 },
            )
            .vector_final()
            .commit_trim::<m![{ Layout<Rows>::Lanes } # 8]>()
            .commit();
        offsets.to_hbm(&mut device.tdma)
    }
}

pub type KeyValue<T> = (T, T);

pub(crate) type Offsets<Count, Chips = Chip> = KeyValue<HbmTensor<i32, Chips, m![Count]>>;

pub(crate) type Length<Chips = Chip> = HbmTensor<i32, Chips, m![1]>;

pub(crate) type Access<Width, Rows, D = f32, Chips = Chip> = (
    KeyValue<RowStorage<Width, D, Chips>>,
    (Offsets<Rows, Chips>, Length<Chips>),
);

/// Physical key/value byte offsets and additive visibility for an indexed attention read,
/// borrowed from wherever the call keeps them, so layers can read one count or bias.
pub struct Read<'a, Chips: M = Chip> {
    /// Number of logical rows to read from each offset list.
    pub count: &'a Length<Chips>,
    /// Key and value byte offsets in logical row order.
    pub offsets: &'a Selection<Chips>,
    /// Query-major additive mask over the selected rows.
    pub bias: &'a Slice<HbmTensor<f32, Chips, m![1]>>,
}

/// Key and value byte offsets of selected rows, one scalar per row.
pub type Selection<Chips = Chip> = KeyValue<Slice<HbmTensor<i32, Chips, m![1]>>>;

type Lane<W> = Symbol<Lanes<W>>;

#[derive(Clone, Debug)]
pub struct BlockIndex;

impl AxisName for BlockIndex {
    const NAME: Ident = Ident::new("BlockIndex");
    const SIZE: usize = Cache::SIZE / Block::SIZE;
}

#[cfg(test)]
pub(crate) const TEST_TOKENS: usize = 8192;

/// Number of blocks each query reads in each statically sized attention window.
pub type BlockCounts<Rows, Windows, Chips = Chip> = HbmTensor<i32, Chips, m![Windows, Rows]>;

/// Marks a token outside the visible history; masked attention assigns it zero mass.
/// Kernels compare a staged bias with zero, so it masks as the device's own `-inf` masks do.
pub const PADDING_BIAS: f32 = -1e30;

/// Caller-owned block IDs and visibility for one logical history.
/// `ALIGNED` means each full block-ID row starts at an eight-byte-aligned address.
#[derive(DeviceSend)]
pub struct History<'a, const ALIGNED: bool = true, Chips: M = Chip> {
    /// Full windows of block IDs in logical token order.
    pub blocks: &'a Slice<HbmTensor<i32, Chips, m![BlockIndex]>>,
    /// Individually addressed block IDs of the partial final window.
    pub tail: &'a Slice<Value<Chips>>,
    /// Number of history tokens, including the current token.
    pub length: &'a Value<Chips>,
    /// Additive mask per history token; an empty slice supplies no extra mask.
    pub bias: &'a Slice<HbmTensor<f32, Chips, m![1]>>,
}

/// Logical meanings shared by every attention geometry of one invocation.
pub struct Logical<Rows: AxisName, Chips: M = Chip> {
    pub positions: HbmTensor<i32, Chips, m![Rows]>,
    pub slots: HbmTensor<i32, Chips, m![Rows]>,
    pub lengths: HbmTensor<i32, Chips, m![Rows]>,
    pub tables: HbmTensor<i32, Chips, m![Rows]>,
}

pub(crate) struct Row<Chips: M = Chip> {
    pub(crate) position: Value<Chips>,
    pub(crate) write: Value<Chips>,
    pub(crate) length: Value<Chips>,
    pub(crate) table: Value<Chips>,
}

impl<Rows: AxisName, Chips: M> Logical<Rows, Chips> {
    pub(crate) fn row(&self, device: &mut Device, row: usize) -> Row<Chips> {
        Row {
            position: scalar::read(
                device,
                self.positions
                    .view()
                    .tile::<m![Rows], 1, m![Rows = 1 # Rows]>(row),
            ),
            write: scalar::read(
                device,
                self.slots
                    .view()
                    .tile::<m![Rows], 1, m![Rows = 1 # Rows]>(row),
            ),
            length: scalar::read(
                device,
                self.lengths
                    .view()
                    .tile::<m![Rows], 1, m![Rows = 1 # Rows]>(row),
            ),
            table: scalar::read(
                device,
                self.tables
                    .view()
                    .tile::<m![Rows], 1, m![Rows = 1 # Rows]>(row),
            ),
        }
    }
}

/// The segments of an `R`-row step, an axis apart from its rows.
#[derive(Clone, Debug)]
pub struct Segment<R>(PhantomData<R>);

impl<R: AxisName> AxisName for Segment<R> {
    const NAME: Ident = Ident::new("Segment");
    const SIZE: usize = R::SIZE;
}

axes![Quad = 4];

/// The lane groups of eight rows that an `R`-row step fills, an axis apart from its rows.
#[derive(Clone, Debug)]
pub struct RowGroup<R>(PhantomData<R>);

impl<R: AxisName> AxisName for RowGroup<R> {
    const NAME: Ident = Ident::new("RowGroup");
    const SIZE: usize = {
        assert!(R::SIZE.is_multiple_of(RowLane::SIZE));
        R::SIZE / RowLane::SIZE
    };
}

axes![RowLane = 8, PacketLane = 8];

/// One value per row and segment, a value a flit.
pub type Grid<R> = m![R, { Symbol<Segment<R>> }, 1 # 8];
/// Four lanes a row, a row a flit.
pub type Pairs<R> = m![R, Quad # 8];
/// Each row's lane of its group's packet, a row a flit: one in that lane, zero in the others.
pub type Places<R> = m![{ Symbol<RowGroup<R>> }, RowLane, PacketLane];

/// The segment words: first row, rows, cached tokens, first block ID and block IDs.
pub(crate) const START: usize = 0;
pub(crate) const LEN: usize = 1;
pub(crate) const CACHED: usize = 2;
pub(crate) const TABLE: usize = 3;
/// The words of a segment.
pub const FIELDS: usize = 5;

axes![Fields = 5];
const _: () = assert!(Fields::SIZE == FIELDS);

/// Window rows of block IDs that a flat table of `B` entries holds.
#[derive(Clone, Debug)]
pub struct TableRows<B>(PhantomData<B>);

impl<B: AxisName> AxisName for TableRows<B> {
    const NAME: Ident = Ident::new("TableRows");
    const SIZE: usize = {
        assert!(B::SIZE.is_multiple_of(BlockIndex::SIZE));
        B::SIZE / BlockIndex::SIZE
    };
}

/// One value per row of an `R`-row envelope, padded to whole packets, on every slice.
pub(crate) type Vector<R, D = i32, Chips = Chip> = DmTensor<D, Chips, Cluster, Slices, Packets<R>>;
/// One value per window and row, on every slice.
type Table<Chips, W, R, D = i32> = DmTensor<D, Chips, Cluster, Slices, m![W, { Packets<R> }]>;
/// A vector as a vector-engine operand.
type VectorOperand<Chips, R> = VrfTensor<i32, Chips, Cluster, Slices, Packets<R>>;

/// Tokens of a block pair, which a row's first window starts on a multiple of: its first table
/// entry then lies 8-byte aligned, and the slots before its first visible token fill a pair.
const PAIR: usize = 2 * Block::SIZE;

/// What the rows' visible histories span: per-window block counts, the windows after the first,
/// the windows' ring, and each row's first window token, span from it, lowest visible token past it
/// and tokens past the ring.
///
/// The windows a row's history needs are a ring of whole windows, its window rounded up to them: a
/// span longer than the ring wraps its tokens past it into the first window's slots before the
/// lowest visible token, which are at least as many.
pub(crate) struct Extents<R: AxisName, W: AxisName, Chips: M = Chip> {
    pub(crate) counts: BlockCounts<R, W, Chips>,
    pub(crate) rest: Value<Chips>,
    spanned: Value<Chips>,
    ring: Packet<i32, Chips>,
    first: Vector<R, i32, Chips>,
    span: Vector<R, i32, Chips>,
    lower: Vector<R, i32, Chips>,
    wrapped: Vector<R, i32, Chips>,
    /// Whether any row wraps.
    pub(crate) wraps: Value<Chips>,
}

impl<R: AxisName, W: AxisName, Chips: M> Extents<R, W, Chips> {
    /// What the rows of visible lengths `length` span over windows that start at `starts`, at most
    /// `window` tokens a row.
    pub(crate) fn new(
        device: &mut Device,
        (window, starts): (&Value<Chips>, &HbmTensor<i32, Chips, m![W, { Packets<R> }]>),
        length: &Vector<R, i32, Chips>,
    ) -> Self {
        let window = scalar::packet(device, window);
        let ring = scalar::map(device, &window, |window| {
            window
                .vector_fxp(FxpBinaryOp::AddFxp, const { Cache::SIZE as i32 - 1 })
                .vector_fxp(
                    FxpBinaryOp::LogicRightShift,
                    const { Cache::SIZE.ilog2() as i32 },
                )
                .vector_fxp(FxpBinaryOp::LeftShift, const { Cache::SIZE.ilog2() as i32 })
        });
        let window = scalar::register(device, &window);
        // The lowest visible token, then the first token of the block pair that holds it.
        let low: Vector<R, i32, Chips> = device
            .main
            .begin(length.view())
            .fetch::<m![R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
            .collect::<m![R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::SubFxp, &window)
            .vector_clip(ClipBinaryOpI32::Max, 0)
            .vector_final()
            .commit_trim::<m![R % ({ Lane<R> }) # 8]>()
            .commit();
        let first: Vector<R, i32, Chips> = device
            .main
            .begin(low.view())
            .fetch::<m![R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
            .collect::<m![R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::LogicRightShift, const { PAIR.ilog2() as i32 })
            .vector_fxp(FxpBinaryOp::LeftShift, const { PAIR.ilog2() as i32 })
            .vector_final()
            .commit_trim::<m![R % ({ Lane<R> }) # 8]>()
            .commit();
        let first_operand = operand(device, &first);
        let span: Vector<R, i32, Chips> = device
            .main
            .begin(length.view())
            .fetch::<m![R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
            .collect::<m![R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::SubFxp, &first_operand)
            .vector_final()
            .commit_trim::<m![R % ({ Lane<R> }) # 8]>()
            .commit();
        let lower: Vector<R, i32, Chips> = device
            .main
            .begin(low.view())
            .fetch::<m![R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
            .collect::<m![R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::SubFxp, &first_operand)
            .vector_final()
            .commit_trim::<m![R % ({ Lane<R> }) # 8]>()
            .commit();
        let ring_operand = scalar::register(device, &ring);
        let wrapped: Vector<R, i32, Chips> = device
            .main
            .begin(span.view())
            .fetch::<m![R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
            .collect::<m![R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::SubFxp, &ring_operand)
            .vector_clip(ClipBinaryOpI32::Max, 0)
            .vector_final()
            .commit_trim::<m![R % ({ Lane<R> }) # 8]>()
            .commit();
        // The slots the ring holds of each row's span.
        let kept: Vector<R, i32, Chips> = device
            .main
            .begin(span.view())
            .fetch::<m![R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
            .collect::<m![R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpI32::Min, &ring_operand)
            .vector_final()
            .commit_trim::<m![R % ({ Lane<R> }) # 8]>()
            .commit();
        let wraps = rows(device, &wrapped);
        let wraps: HbmTensor<i32, Chips, m![R, 1 # 8]> = wraps.to_hbm(&mut device.tdma);
        let wraps: DmTensor<i32, Chips, Cluster, Slices, m![R, 1 # 8]> =
            wraps.to_dm(&mut device.tdma);
        let wraps: DmTensor<i32, Chips, Cluster, Slices, m![1 # 8]> = device
            .main
            .begin(wraps.view())
            .fetch::<m![R], m![1 # 8]>()
            .collect::<m![R], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(IntraSliceReduceOpI32::Max)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_clip(ClipBinaryOpI32::Min, 1)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        // The windows after the first that the longest span covers, reduced over one row a flit;
        // a row with no span covers none.
        let spans = rows(device, &kept);
        let spans: HbmTensor<i32, Chips, m![R, 1 # 8]> = spans.to_hbm(&mut device.tdma);
        let spans: DmTensor<i32, Chips, Cluster, Slices, m![R, 1 # 8]> =
            spans.to_dm(&mut device.tdma);
        let rest: DmTensor<i32, Chips, Cluster, Slices, m![1 # 8]> = device
            .main
            .begin(spans.view())
            .fetch::<m![R], m![1 # 8]>()
            .collect::<m![R], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::SubFxp, 1)
            .vector_fxp(
                FxpBinaryOp::ArithRightShift,
                const { Cache::SIZE.ilog2() as i32 },
            )
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(IntraSliceReduceOpI32::Max)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        let spanned = scalar::map(device, &rest, |rest| {
            rest.vector_fxp(FxpBinaryOp::AddFxp, 1)
        });
        // count = clip(kept - 512 w, 0, 512) rounded up to blocks.
        let starts: Table<Chips, W, R> = starts.to_dm(&mut device.tdma);
        let span_operand = operand(device, &kept);
        let counts: Table<Chips, W, R> = device
            .main
            .begin(starts.view())
            .fetch::<m![W, R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
            .collect::<m![W, R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::AddFxp, &span_operand)
            .vector_clip(ClipBinaryOpI32::Max, 0)
            .vector_clip(ClipBinaryOpI32::Min, const { Cache::SIZE as i32 })
            .vector_final()
            .commit_trim::<m![R % ({ Lane<R> }) # 8]>()
            .commit();
        let counts: Table<Chips, W, R> = device
            .main
            .begin(counts.view())
            .fetch::<m![W, R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
            .collect::<m![W, R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::AddFxp, const { Block::SIZE as i32 - 1 })
            .vector_fxp(
                FxpBinaryOp::LogicRightShift,
                const { Block::SIZE.ilog2() as i32 },
            )
            .vector_final()
            .commit_trim::<m![R % ({ Lane<R> }) # 8]>()
            .commit();
        let counts: HbmTensor<i32, Chips, m![W, { Packets<R> }]> = counts.to_hbm(&mut device.tdma);
        Extents {
            counts: counts.to_hbm(&mut device.tdma),
            rest: rest.to_hbm(&mut device.tdma),
            spanned: spanned.to_hbm(&mut device.tdma),
            ring,
            first,
            span,
            lower,
            wrapped,
            wraps: wraps.to_hbm(&mut device.tdma),
        }
    }
}

/// One window's table row of every row, a row a slice.
type Entries<Chips, R> = DmTensor<i32, Chips, Cluster, m![R # 256], m![BlockIndex]>;

impl<R: AxisName, W: AxisName, Chips: M> Extents<R, W, Chips> {
    /// Each row's pool-block byte offsets for the windows the rows span, a window's rows together:
    /// its table's window rows from its first window, `bytes` a block; a window past the table
    /// reads its last row.
    pub(crate) fn table<B: AxisName>(
        &self,
        device: &mut Device,
        (rows, blocks): (
            &HbmTensor<i32, Chips, m![W, { Packets<R> }]>,
            &HbmTensor<i32, Chips, m![B]>,
        ),
        starts: &HbmTensor<i32, Chips, m![R]>,
        bytes: i32,
    ) -> HbmTensor<i32, Chips, m![W, R, BlockIndex]> {
        let (first, spanned) = (&self.first, &self.spanned);
        let starts = vector(device, starts);
        let head: Vector<R, i32, Chips> = device
            .main
            .begin(first.view())
            .fetch::<m![R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
            .collect::<m![R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(
                FxpBinaryOp::LogicRightShift,
                const { Block::SIZE.ilog2() as i32 },
            )
            .vector_final()
            .commit_trim::<m![R % ({ Lane<R> }) # 8]>()
            .commit();
        let head = operand(device, &head);
        let base: Vector<R, i32, Chips> = device
            .main
            .begin(starts.view())
            .fetch::<m![R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
            .collect::<m![R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::AddFxp, &head)
            .vector_fxp(FxpBinaryOp::MulInt, const { size_of::<i32>() as i32 })
            .vector_final()
            .commit_trim::<m![R % ({ Lane<R> }) # 8]>()
            .commit();
        let base = operand(device, &base);
        let windows: Table<Chips, W, R> = rows.to_dm(&mut device.tdma);
        let offsets: Table<Chips, W, R> = device
            .main
            .begin(windows.view())
            .fetch::<m![W, R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
            .collect::<m![W, R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::AddFxp, &base)
            .vector_clip(ClipBinaryOpI32::Max, 0)
            .vector_clip(
                ClipBinaryOpI32::Min,
                const { ((B::SIZE - BlockIndex::SIZE) * size_of::<i32>()) as i32 },
            )
            .vector_final()
            .commit_trim::<m![R % ({ Lane<R> }) # 8]>()
            .commit();
        let mut output: HbmTensor<i32, Chips, m![W, R, BlockIndex]> = HbmTensor::new();
        for w in 0..spanned.extent::<m![W]>() {
            let mut index: HbmTensor<i32, Chips, m![R]> = HbmTensor::new();
            offsets
                .view()
                .tile::<m![W], 1, m![W = 1 # W, { Packets<R> }]>(w)
                .to_hbm_view(&mut device.tdma, index.view_mut());
            // SAFETY: the flat table is whole window rows in order.
            let rows = unsafe {
                blocks
                    .view()
                    .reshape::<Chips, m![{ Symbol<TableRows<B>> }, BlockIndex]>()
            };
            let ids: Entries<Chips, R> = rows
                .gather::<m![{ Symbol<TableRows<B>> }], m![BlockIndex]>()
                .by_byte_offsets(&index)
                .to_dm(&mut device.tdma);
            let scaled: Entries<Chips, R> = device
                .main
                .begin(ids.view())
                .fetch::<m![BlockIndex / 8], m![BlockIndex % 8]>()
                .collect::<m![BlockIndex / 8], m![BlockIndex % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_fxp(FxpBinaryOp::MulInt, bytes)
                .vector_final()
                .commit_trim::<m![BlockIndex % 8]>()
                .commit();
            scaled.view().to_hbm_view(
                &mut device.tdma,
                output
                    .view_mut()
                    .tile::<m![W], 1, m![W = 1 #{!} W, R, BlockIndex]>(w),
            );
        }
        output
    }
}

impl<R: AxisName, W: AxisName, Chips: M> Extents<R, W, Chips> {
    /// How many tokens each row holds past its ring.
    pub(crate) fn wrapped(&self, device: &mut Device) -> HbmTensor<i32, Chips, m![R]> {
        rows(device, &self.wrapped)
    }

    /// The pool byte offset of each token row past each row's ring into `offsets`, from the two
    /// entries past the ring in its table at `starts`, `bytes` a block and `row` a token row; a
    /// window slot's middle `t + 1/2` in `slots` places token `t` of a block.
    pub(crate) fn wrap<B: AxisName>(
        &self,
        device: &mut Device,
        (blocks, slots): (
            &HbmTensor<i32, Chips, m![B]>,
            &HbmTensor<f32, Chips, m![R, Cache]>,
        ),
        (starts, (bytes, row)): (&HbmTensor<i32, Chips, m![R]>, (i32, i32)),
        offsets: &mut Wrapped<R, Chips>,
    ) {
        let ring = scalar::map(device, &self.ring, |ring| {
            ring.vector_fxp(
                FxpBinaryOp::LogicRightShift,
                const { Block::SIZE.ilog2() as i32 },
            )
        });
        let ring = scalar::register(device, &ring);
        // The first block past the ring.
        let head: Vector<R, i32, Chips> = device
            .main
            .begin(self.first.view())
            .fetch::<m![R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
            .collect::<m![R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(
                FxpBinaryOp::LogicRightShift,
                const { Block::SIZE.ilog2() as i32 },
            )
            .vector_fxp(FxpBinaryOp::AddFxp, &ring)
            .vector_final()
            .commit_trim::<m![R % ({ Lane<R> }) # 8]>()
            .commit();
        let head = operand(device, &head);
        let starts = vector(device, starts);
        let index: Vector<R, i32, Chips> = device
            .main
            .begin(starts.view())
            .fetch::<m![R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
            .collect::<m![R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::AddFxp, &head)
            .vector_fxp(FxpBinaryOp::MulInt, const { size_of::<i32>() as i32 })
            .vector_clip(ClipBinaryOpI32::Max, 0)
            .vector_clip(
                ClipBinaryOpI32::Min,
                const { ((B::SIZE - BlockIndex::SIZE) * size_of::<i32>()) as i32 },
            )
            .vector_final()
            .commit_trim::<m![R % ({ Lane<R> }) # 8]>()
            .commit();
        let index = rows(device, &index);
        // SAFETY: the flat table is whole window rows in order.
        let table = unsafe {
            blocks
                .view()
                .reshape::<Chips, m![{ Symbol<TableRows<B>> }, BlockIndex]>()
        };
        let ids: Entries<Chips, R> = table
            .gather::<m![{ Symbol<TableRows<B>> }], m![BlockIndex]>()
            .by_byte_offsets(&index)
            .to_dm(&mut device.tdma);
        let scaled: Entries<Chips, R> = device
            .main
            .begin(ids.view())
            .fetch::<m![BlockIndex / 8], m![BlockIndex % 8]>()
            .collect::<m![BlockIndex / 8], m![BlockIndex % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::MulInt, bytes)
            .vector_final()
            .commit_trim::<m![BlockIndex % 8]>()
            .commit();
        let scaled: HbmTensor<i32, Chips, m![R, BlockIndex]> = scaled.to_hbm(&mut device.tdma);
        let slots: DmTensor<f32, Chips, Cluster, m![R # 256], m![Cache % ({ Symbol<Block> })]> =
            slots
                .view()
                .tile::<m![Cache / ({ Symbol<Block> })], 1, m![
                    R,
                    Cache / ({ Symbol<Block> }) = 1 # (Cache / ({ Symbol<Block> })),
                    Cache % ({ Symbol<Block> })
                ]>(0)
                .to_dm(&mut device.tdma);
        let tokens: DmTensor<i32, Chips, Cluster, m![R # 256], m![Cache % ({ Symbol<Block> })]> =
            device
                .main
                .begin(slots.view())
                .fetch::<m![Cache % ({ Symbol<Block> }) / 8], m![Cache % 8]>()
                .collect::<m![Cache % ({ Symbol<Block> }) / 8], m![Cache % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<
                    m![Cache % ({ Symbol<Block> }) / 8, Cache % 8 / 4],
                    m![Cache % 4],
                >()
                .vector_fp_binary(FpBinaryOp::SubF, 0.5)
                .vector_widen_concat::<m![Cache % ({ Symbol<Block> }) / 8], m![Cache % 8]>()
                .vector_fp_to_fxp(31)
                .vector_final()
                .commit_trim::<m![Cache % 8]>()
                .commit();
        // SAFETY: a block's tokens take the first `Block` slots of a window, in order.
        let tokens: DmTensor<i32, Chips, Cluster, m![R # 256], m![Block]> =
            unsafe { tokens.reshape() };
        for pair in 0..2 {
            // A row's block in a cell of its own: writes into one HBM cell serialize.
            let mut block: HbmTensor<i32, Chips, m![R, 1 # 8]> = HbmTensor::new();
            scaled
                .view()
                .tile::<m![BlockIndex], 1, m![R, BlockIndex = 1 # BlockIndex]>(pair)
                .to_hbm_view(&mut device.tdma, block.view_mut());
            let block: RowValues<Chips, R> = block.to_dm(&mut device.tdma);
            let block = scalar::register(device, &block);
            let run: DmTensor<i32, Chips, Cluster, m![R # 256], m![Block]> = device
                .main
                .begin(tokens.view())
                .fetch::<m![Block / 8], m![Block % 8]>()
                .collect::<m![Block / 8], m![Block % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_fxp(FxpBinaryOp::MulInt, row)
                .vector_fxp(FxpBinaryOp::AddFxp, &block)
                .vector_final()
                .commit_trim::<m![Block % 8]>()
                .commit();
            run.view().to_hbm_view(
                &mut device.tdma,
                offsets
                    .view_mut()
                    .tile::<m![BlockIndex % 2], 1, m![R, BlockIndex % 2 = 1 #{!} 2, Block]>(pair),
            );
        }
    }
}

/// One window's token slots of a row on the row's slice, so a window's mask stores a row a run.
type RowSlots<Chips, R> = DmTensor<f32, Chips, Cluster, m![R # 256], m![Cache]>;
/// One value of a row on the row's slice.
type RowValues<Chips, R, D = i32> = DmTensor<D, Chips, Cluster, m![R # 256], m![1 # 8]>;
/// One value of a row on the row's slice, the per-row operand of a pass over [`RowSlots`].
type PerRow<Chips, R, D = f32> = VrfTensor<D, Chips, Cluster, m![R # 256], m![1 # 8]>;

impl<R: AxisName, W: AxisName, Chips: M> Extents<R, W, Chips> {
    /// The mask of every window the rows span: 0 where a row's token `512 w + t`, taken cyclically
    /// over the ring, lies in `[lower, span)` from its first window, padding elsewhere; later
    /// windows stay unwritten.
    pub(crate) fn bias(
        &self,
        device: &mut Device,
        (starts, slots): (
            &HbmTensor<i32, Chips, m![W, { Packets<R> }]>,
            &HbmTensor<f32, Chips, m![R, Cache]>,
        ),
    ) -> HbmTensor<f32, Chips, m![R, W, Cache]> {
        let extents = self;
        let lower = operand(device, &extents.lower);
        let width: Vector<R, f32, Chips> = device
            .main
            .begin(extents.span.view())
            .fetch::<m![R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
            .collect::<m![R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::SubFxp, &lower)
            .vector_fxp_to_fp(31)
            .vector_final()
            .commit_trim::<m![R % ({ Lane<R> }) # 8]>()
            .commit();
        let width = rows(device, &width);
        let width = per_row(device, &width);
        let ring = scalar::register(device, &extents.ring);
        let rings: Vector<R, f32, Chips> = device
            .main
            .begin(extents.span.view())
            .fetch::<m![R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
            .collect::<m![R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::MulInt, 0)
            .vector_fxp(FxpBinaryOp::AddFxp, &ring)
            .vector_fxp_to_fp(31)
            .vector_final()
            .commit_trim::<m![R % ({ Lane<R> }) # 8]>()
            .commit();
        let rings = rows(device, &rings);
        let rings = per_row(device, &rings);
        // Window w's first visible token of each row, relative to the row's first window.
        let starts: Table<Chips, W, R> = starts.to_dm(&mut device.tdma);
        let firsts: Table<Chips, W, R, f32> = device
            .main
            .begin(starts.view())
            .fetch::<m![W, R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
            .collect::<m![W, R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::AddFxp, &lower)
            .vector_fxp_to_fp(31)
            .vector_final()
            .commit_trim::<m![R % ({ Lane<R> }) # 8]>()
            .commit();
        let firsts: HbmTensor<f32, Chips, m![W, { Packets<R> }]> = firsts.to_hbm(&mut device.tdma);
        let firsts: HbmTensor<f32, Chips, m![W, R]> = firsts.to_hbm(&mut device.tdma);
        let slots: RowSlots<Chips, R> = slots.to_dm(&mut device.tdma);
        let mut output: HbmTensor<f32, Chips, m![R, W, Cache]> = HbmTensor::new();
        for w in 0..extents.spanned.extent::<m![W]>() {
            let mut start: HbmTensor<f32, Chips, m![R]> = HbmTensor::new();
            firsts
                .view()
                .tile::<m![W], 1, m![W = 1 # W, R]>(w)
                .to_hbm_view(&mut device.tdma, start.view_mut());
            let start = per_row(device, &start);
            // u (u - width) < 0 for u = t + 1/2 - start exactly where token t is visible.
            let shifted: RowSlots<Chips, R> = device
                .main
                .begin(slots.view())
                .fetch::<m![Cache / 8], m![Cache % 8]>()
                .collect::<m![Cache / 8], m![Cache % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Cache / 8, Cache % 8 / 4], m![Cache % 4]>()
                .vector_fp_binary(FpBinaryOp::SubF, &start)
                .vector_widen_concat::<m![Cache / 8], m![Cache % 8]>()
                .vector_final()
                .commit_trim::<m![Cache % 8]>()
                .commit();
            // A slot before the lowest visible token holds one a ring later.
            let shifted: RowSlots<Chips, R> = device
                .main
                .begin(shifted.view())
                .fetch::<m![Cache / 8], m![Cache % 8]>()
                .collect::<m![Cache / 8], m![Cache % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Comparison([Cmp::Less(0.0f32); 4]))
                .vector_narrow_split::<m![Cache / 8, Cache % 8 / 4], m![Cache % 4]>()
                .vector_fp_binary(
                    FpBinaryOp::AddF,
                    Branched::rf(
                        TagGuard::matches([
                            BitReq::One,
                            BitReq::Ignore,
                            BitReq::Ignore,
                            BitReq::Ignore,
                        ]),
                        &rings,
                    ),
                )
                .vector_widen_concat::<m![Cache / 8], m![Cache % 8]>()
                .vector_final()
                .commit_trim::<m![Cache % 8]>()
                .commit();
            let product: RowSlots<Chips, R> = device
                .main
                .begin(shifted.view())
                .fetch::<m![Cache / 8], m![Cache % 8]>()
                .collect::<m![Cache / 8], m![Cache % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![Cache / 8, Cache % 8 / 4], m![Cache % 4]>()
                .vector_stash()
                .vector_fp_binary(FpBinaryOp::SubF, &width)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
                .vector_widen_concat::<m![Cache / 8], m![Cache % 8]>()
                .vector_final()
                .commit_trim::<m![Cache % 8]>()
                .commit();
            let masked: RowSlots<Chips, R> = device
                .main
                .begin(product.view())
                .fetch::<m![Cache / 8], m![Cache % 8]>()
                .collect::<m![Cache / 8], m![Cache % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Comparison([Cmp::Less(0.0f32); 4]))
                .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)
                .vector_logic(
                    LogicBinaryOpF32::BitOr,
                    Branched::imm(
                        TagGuard::matches([
                            BitReq::Zero,
                            BitReq::Ignore,
                            BitReq::Ignore,
                            BitReq::Ignore,
                        ]),
                        PADDING_BIAS,
                    ),
                )
                .vector_final()
                .commit_trim::<m![Cache % 8]>()
                .commit();
            masked.view().to_hbm_view(
                &mut device.tdma,
                output
                    .view_mut()
                    .tile::<m![W], 1, m![R, W = 1 #{!} W, Cache]>(w),
            );
        }
        output
    }
}

#[cfg(test)]
pub(crate) fn block_counts<Rows: AxisName, Windows: AxisName>(
    lengths: impl IntoIterator<Item = i32>,
) -> HostTensor<i32, m![Windows, Rows]> {
    let lengths: Vec<_> = lengths.into_iter().collect();
    assert_eq!(lengths.len(), Rows::SIZE);
    HostTensor::from_vec((0..Windows::SIZE).flat_map(|window| {
        lengths.iter().map(move |&length| {
            (length.max(0) as usize)
                .saturating_sub(window * Cache::SIZE)
                .min(Cache::SIZE)
                .div_ceil(Block::SIZE) as i32
        })
    }))
}

/// An I32 per row, padded to whole packets on every slice.
pub(crate) fn vector<R: AxisName, Chips: M>(
    device: &mut Device,
    values: &HbmTensor<i32, Chips, m![R]>,
) -> Vector<R, i32, Chips> {
    let padded: HbmTensor<i32, Chips, Packets<R>> = values.to_hbm(&mut device.tdma);
    padded.to_dm(&mut device.tdma)
}

/// The rows of a padded vector.
fn rows<R: AxisName, D: MaterializableScalar, Chips: M>(
    device: &mut Device,
    vector: &Vector<R, D, Chips>,
) -> HbmTensor<D, Chips, m![R]> {
    let padded: HbmTensor<D, Chips, Packets<R>> = vector.to_hbm(&mut device.tdma);
    padded.to_hbm(&mut device.tdma)
}

fn operand<R: AxisName, Chips: M>(
    device: &mut Device,
    vector: &Vector<R, i32, Chips>,
) -> VectorOperand<Chips, R> {
    device
        .sub
        .begin(vector.view())
        .fetch::<m![R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
        .collect::<m![R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
        .to_vrf()
}

/// Each live row's value, zero past the live rows, whose values `zeros` holds.
pub(crate) fn live<R: AxisName, Chips: M>(
    device: &mut Device,
    zeros: &HbmTensor<i32, Chips, m![R]>,
    tiles: Live<(), R>,
    values: &HbmTensor<i32, Chips, m![R]>,
) -> Vector<R, i32, Chips> {
    let mut output: HbmTensor<i32, Chips, m![R]> = zeros.to_hbm(&mut device.tdma);
    tiles.cover::<Lanes<R>, _>(
        &mut (&mut *device, &mut output),
        (
            |(device, output)| {
                values
                    .view()
                    .to_hbm_view(&mut device.tdma, output.view_mut())
            },
            |(device, output), tile| {
                values
                    .view()
                    .tile::<
                        Layout<R>::Groups,
                        1,
                        m![{ Layout<R>::Selected }, { Layout<R>::Lanes }],
                    >(tile)
                    .to_hbm_view(
                        &mut device.tdma,
                        output.view_mut().tile::<Layout<R>::Groups, 1, m![
                            { Layout<R>::SelectedMut }, { Layout<R>::Lanes }
                        ]>(tile),
                    )
            },
            |(device, output), row| {
                values
                    .view()
                    .tile::<m![R], 1, m![R = 1 # R]>(row)
                    .to_hbm_view(
                        &mut device.tdma,
                        output.view_mut().tile::<m![R], 1, m![R = 1 #{!} R]>(row),
                    )
            },
        ),
    );
    vector(device, &output)
}

/// A per-row value on each row's slice.
fn per_row<R: AxisName, D: VeScalar + MaterializableScalar, Chips: M>(
    device: &mut Device,
    values: &HbmTensor<D, Chips, m![R]>,
) -> PerRow<Chips, R, D> {
    let padded: HbmTensor<D, Chips, m![R, 1 # 8]> = values.to_hbm(&mut device.tdma);
    let rows: RowValues<Chips, R, D> = padded.to_dm(&mut device.tdma);
    scalar::register(device, &rows)
}

/// Each row's token position as F32, repeated over its rotated coordinates: one pass over a row a
/// slice, whose stores are whole rows.
pub(crate) fn positions<R: AxisName, E: AxisName, Chips: M>(
    device: &mut Device,
    frequencies: &HbmTensor<f32, Chips, m![R, E]>,
    positions: &HbmTensor<i32, Chips, m![R]>,
) -> HbmTensor<f32, Chips, m![R, E]> {
    let positions: HbmTensor<i32, Chips, m![R, 1 # 8]> = positions.to_hbm(&mut device.tdma);
    let positions: DmTensor<i32, Chips, Cluster, m![R # 256], m![1 # 8]> =
        positions.to_dm(&mut device.tdma);
    let positions = scalar::register(device, &positions);
    let rows: DmTensor<f32, Chips, Cluster, m![R # 256], m![E]> =
        frequencies.to_dm(&mut device.tdma);
    // Zero bits plus the position, then fixed-point exponent 31 keeps it in integer units.
    let output: DmTensor<f32, Chips, Cluster, m![R # 256], m![E]> = device
        .main
        .begin(rows.view())
        .fetch::<m![E / 8], m![E % 8]>()
        .collect::<m![E / 8], m![E % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, 0.0f32)
        .vector_reinterpret::<i32>()
        .vector_fxp(FxpBinaryOp::AddFxp, &positions)
        .vector_fxp_to_fp(31)
        .vector_final()
        .commit_trim::<m![E % 8]>()
        .commit();
    output.to_hbm(&mut device.tdma)
}

/// Each row's write slot as a byte offset into its layer's blocks.
pub(crate) fn writes<R: AxisName, Chips: M>(
    device: &mut Device,
    slots: &HbmTensor<i32, Chips, m![R]>,
    stride: i32,
) -> HbmTensor<i32, Chips, m![R]> {
    let slots = vector(device, slots);
    let offsets: Vector<R, i32, Chips> = device
        .main
        .begin(slots.view())
        .fetch::<m![R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
        .collect::<m![R / ({ Lane<R> })], m![R % ({ Lane<R> }) # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::MulInt, stride)
        .vector_final()
        .commit_trim::<m![R % ({ Lane<R> }) # 8]>()
        .commit();
    rows(device, &offsets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::attention::Attention;
    use crate::ops::attention::geometry::QueryGroups;
    use crate::ops::attention::gqa::{self, Paged};
    use crate::ops::attention::tests::{Blocks, Chunks, Head, KvHeads};
    use crate::specialize::attention::Strategy;
    use crate::tensor::axis::Eight;
    use crate::tensor::cache::KeyValue;
    use crate::tensor::live::tests::count;
    use crate::tensor::scalar::F32;
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;

    type PrefixGroups = QueryGroups<4>;

    type PrefixQuery<Count> = HbmTensor<bf16, Chip, m![Count, KvHeads, PrefixGroups, Head]>;
    type PrefixOutput<Count> = HbmTensor<bf16, Chip, m![KvHeads, Count, PrefixGroups, Head]>;
    type Bias<Count> = HbmTensor<f32, Chip, m![Count, Chunks, Cache]>;

    #[device(chip = 1)]
    fn checked(
        device: &mut Device,
        query: &PrefixQuery<Eight>,
        blocks: &Blocks,
        table: &HbmTensor<i32, Chip, m![Chunks, Eight, BlockIndex]>,
        lengths: &BlockCounts<Eight, Chunks>,
        bias: &Bias<Eight>,
        rest: &Value,
        output: &mut PrefixOutput<Eight>,
    ) {
        let query: gqa::Heads<Eight, KvHeads, PrefixGroups, Head> = query.to_dm(&mut device.tdma);
        let history = Paged {
            blocks,
            table,
            lengths,
            scales: &(),
            wrap: (),
        };
        Attention::<
            PhantomData<(KvHeads, PrefixGroups, Head)>,
            F32<{ (std::f32::consts::FRAC_1_SQRT_2 / 8.0).to_bits() as usize }>,
            { Strategy::Windows },
        >::default()
        .forward(&mut *device, (&query, &history, bias, rest, Live::all()))
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn attends_context_limit() {
        const CONTEXT: usize = Chunks::SIZE * Cache::SIZE;
        const LENGTHS: [usize; 8] = [40_960, 40_959, 40_449, 40_448, 513, 512, 1, 0];
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(checked.topology()).unwrap();
        let bytes = 2048 * <Blocks>::BLOCK_STRIDE as usize;
        let mut layer = Blocks::from_buffer(HbmBuffer::alloc(&device, bytes).unwrap()).unwrap();
        let blocks = layer.tokens.len();
        assert!(blocks > Eight::SIZE * BlockIndex::SIZE + Eight::SIZE);
        let base = |row: usize, head: usize, channel: usize| {
            bf16::from_f32(
                ((if row == 2 { 0.0 } else { row as f32 / 8.0 }) + (head % 4) as f32 / 32.0)
                    / (1 << (channel % 4)) as f32,
            )
            .to_f32()
        };
        let end = |row: usize, head: usize, channel: usize| {
            bf16::from_f32(
                base(row, head, channel) + Chunks::SIZE as f32 / (1 << (channel % 4)) as f32,
            )
            .to_f32()
        };
        let tip = |row: usize, head: usize, channel: usize| {
            bf16::from_f32(end(row, head, channel) + CONTEXT as f32 / (1 << (channel % 4)) as f32)
                .to_f32()
        };
        for row in 0..Eight::SIZE {
            let normal: Vec<_> = (0..<m![Block, KeyValue, KvHeads, Head]>::SIZE)
                .map(|at| {
                    let channel = at % Head::SIZE;
                    let head = at / Head::SIZE % KvHeads::SIZE;
                    let kind = at / (Head::SIZE * KvHeads::SIZE) % KeyValue::SIZE;
                    let token = at / (Head::SIZE * KvHeads::SIZE * KeyValue::SIZE);
                    bf16::from_f32(if token >= LENGTHS[row] {
                        if channel % 2 == 0 {
                            f32::NAN
                        } else {
                            f32::INFINITY
                        }
                    } else if kind == 0 {
                        0.0
                    } else {
                        base(row, head, channel)
                    })
                })
                .collect();
            layer
                .write(&mut device, row..row + 1, normal)
                .await
                .unwrap();
            let begin = blocks - (row + 1) * BlockIndex::SIZE;
            let final_window: Vec<_> = (0..<m![Cache, KeyValue, KvHeads, Head]>::SIZE)
                .map(|at| {
                    let channel = at % Head::SIZE;
                    let head = at / Head::SIZE % KvHeads::SIZE;
                    let kind = at / (Head::SIZE * KvHeads::SIZE) % KeyValue::SIZE;
                    let token = (Chunks::SIZE - 1) * Cache::SIZE
                        + at / (Head::SIZE * KvHeads::SIZE * KeyValue::SIZE);
                    bf16::from_f32(if token >= LENGTHS[row] {
                        if channel % 2 == 0 {
                            f32::NAN
                        } else {
                            f32::NEG_INFINITY
                        }
                    } else if kind == 0 {
                        0.0
                    } else if token == CONTEXT - 1 {
                        tip(row, head, channel)
                    } else {
                        end(row, head, channel)
                    })
                })
                .collect();
            layer
                .write(&mut device, begin..begin + BlockIndex::SIZE, final_window)
                .await
                .unwrap();
        }
        let last = layer.read(&mut device, blocks - 1..blocks).await.unwrap();
        assert_eq!(
            last.last().unwrap().to_f32(),
            tip(0, KvHeads::SIZE - 1, Head::SIZE - 1)
        );
        let query = HostTensor::<bf16, m![Eight, KvHeads, PrefixGroups, Head]>::zero()
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let table = HostTensor::<i32, m![Chunks, Eight, BlockIndex]>::from_vec(
            (0..Chunks::SIZE).flat_map(|window| {
                (0..Eight::SIZE).flat_map(move |row| {
                    (0..BlockIndex::SIZE).map(move |entry| {
                        let block = if window < Chunks::SIZE - 1 {
                            row
                        } else {
                            blocks - (row + 1) * BlockIndex::SIZE + entry
                        };
                        i32::try_from(block * <Blocks>::BLOCK_STRIDE as usize).unwrap()
                    })
                })
            }),
        )
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        let bias = HostTensor::<f32, m![Eight, Chunks, Cache]>::from_vec(LENGTHS.iter().flat_map(
            |&length| {
                (0..CONTEXT).map(move |token| if token < length { 0.0 } else { PADDING_BIAS })
            },
        ))
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        let rest = count(Chunks::SIZE - 1)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let mut output = HostTensor::<bf16, m![KvHeads, Eight, PrefixGroups, Head]>::zero()
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let mut previous: Option<Vec<bf16>> = None;
        for phase in 0..2 {
            let lengths = block_counts::<Eight, Chunks>(LENGTHS.iter().map(|&length| {
                if phase == 0 {
                    CONTEXT as i32
                } else {
                    length as i32
                }
            }))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
            launch(
                checked,
                (
                    &mut device,
                    &query,
                    &layer,
                    &table,
                    &lengths,
                    &bias,
                    &rest,
                    &mut output,
                ),
            )
            .await
            .unwrap();
            let got = output
                .to_host::<m![KvHeads, Eight, PrefixGroups, Head]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            for (at, actual) in got.iter().enumerate() {
                let channel = at % Head::SIZE;
                let row = at / (Head::SIZE * PrefixGroups::SIZE) % Eight::SIZE;
                let head = at / (Head::SIZE * PrefixGroups::SIZE * Eight::SIZE);
                let length = LENGTHS[row];
                let tail = length.saturating_sub((Chunks::SIZE - 1) * Cache::SIZE);
                let expected = if length == 0 {
                    0.0
                } else {
                    ((length - tail) as f32 * base(row, head, channel)
                        + tail as f32 * end(row, head, channel)
                        + if length == CONTEXT {
                            tip(row, head, channel) - end(row, head, channel)
                        } else {
                            0.0
                        })
                        / length as f32
                };
                let error = (actual.to_f32() - expected).abs();
                assert!(
                    actual.to_f32().is_finite() && error <= 0.01 * expected.abs().max(1.0 / 128.0),
                    "phase {phase} row {row} head {head} channel {channel}: {} != {expected}",
                    actual.to_f32()
                );
            }
            if let Some(previous) = &previous {
                assert!(
                    got.iter()
                        .zip(previous)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "untransferred poisoned tails must not change the valid result"
                );
            }
            previous = Some(got);
        }
    }
}
