//! The logical rows of a step: each packed row's position, write slot, visible length and table
//! entry, derived on the device from its history's segment table and load-time coordinates.

use super::{
    CACHED, Fields, Grid, LEN, Logical, PacketLane, Pairs, Places, Quad, RowGroup, RowLane, START,
    Segment, TABLE,
};
use crate::Op;
use crate::load::{self, Checkpoint, Load, LoadError};
use crate::tensor::cache::Block;
use crate::tensor::mapping::{BroadcastClusters as Cluster, BroadcastSlices as Slices};
use crate::tensor::memory::{Hbm, Memory};
use crate::tensor::parallel::Chip;
use crate::tensor::scalar;
use furiosa_opt_std::prelude::*;
use std::borrow::Borrow;

/// Packed-row coordinates bound at load time, independent of request packing, a copy on each chip
/// of `Chips`, in memory `X`.
#[derive(DeviceSend, Load)]
#[load(tree)]
pub struct Derive<Rows: AxisName, Chips: M + 'static = Chip, X: Memory = Hbm> {
    /// Each row's index, a row a flit.
    pub iota: X::Tensor<i32, Chips, m![Rows, 1 # 8]>,
    /// Each row's index for every segment.
    pub grid: X::Tensor<i32, Chips, Grid<Rows>>,
    /// The four lanes of an entry quad, 0 to 3, on every row.
    pub pairs: X::Tensor<i32, Chips, Pairs<Rows>>,
    /// Each row's lane of its group's packet.
    pub places: X::Tensor<i32, Chips, Places<Rows>>,
}

/// The coordinates a load computes, as no checkpoint stores them.
/// By hand, with the update below: a derived `with` field would rebuild them in every update.
impl<'a, Rows: AxisName, Chips: M + 'static, X: Memory> Load<&'a Checkpoint>
    for Derive<Rows, Chips, X>
where
    X::Tensor<i32, Chips, m![Rows, 1 # 8]>:
        Load<HostTensor<i32, m![{ Chips }, { m![Rows, 1 # 8] }]>>,
    X::Tensor<i32, Chips, Grid<Rows>>: Load<HostTensor<i32, m![{ Chips }, { Grid<Rows> }]>>,
    X::Tensor<i32, Chips, Pairs<Rows>>: Load<HostTensor<i32, m![{ Chips }, { Pairs<Rows> }]>>,
    X::Tensor<i32, Chips, Places<Rows>>: Load<HostTensor<i32, m![{ Chips }, { Places<Rows> }]>>,
{
    fn load(context: &mut load::Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        let lanes = |count: usize, value: &dyn Fn(usize) -> Option<usize>| -> Vec<i32> {
            let values = (0..count).map(|at| value(at).map_or(0, |index| index as i32));
            values.collect::<Vec<_>>().repeat(Chips::SIZE)
        };
        let row = Rows::SIZE;
        context.checkpoint(from, |cx| {
            Ok(Self {
                iota: cx.field("iota", "iota", |cx| {
                    let iota = lanes(row * 8, &|at| (at % 8 == 0).then_some(at / 8));
                    Load::load(cx, HostTensor::from_vec(iota))
                })?,
                grid: cx.field("grid", "grid", |cx| {
                    let grid = lanes(row * row * 8, &|at| (at % 8 == 0).then_some(at / 8 / row));
                    Load::load(cx, HostTensor::from_vec(grid))
                })?,
                pairs: cx.field("pairs", "pairs", |cx| {
                    let pairs = lanes(row * 8, &|at| (at % 8 < 4).then_some(at % 8));
                    Load::load(cx, HostTensor::from_vec(pairs))
                })?,
                places: cx.field("places", "places", |cx| {
                    let places = lanes(row * 8, &|at| (at % 8 == at / 8 % 8).then_some(1));
                    Load::load(cx, HostTensor::from_vec(places))
                })?,
            })
        })
    }
}

/// The coordinates are no update target.
impl<'t, 's, Rows: AxisName, Chips: M + 'static, X: Memory>
    Load<(&'t mut Derive<Rows, Chips, X>, &'s Checkpoint)> for &'t mut Derive<Rows, Chips, X>
{
    fn load(
        context: &mut load::Context<'_>,
        (into, from): (&'t mut Derive<Rows, Chips, X>, &'s Checkpoint),
    ) -> Result<Self, LoadError> {
        context.checkpoint(from, |_| Ok(()))?;
        Ok(into)
    }
}

/// Row `r` on slice `r`, so a pass over every row and segment moves a row's segments a slice.
type RowSlices<R> = m![R # 256];

/// Each row's value for every segment, a value a flit.
type Cells<R, Chips> =
    DmTensor<i32, Chips, Cluster, RowSlices<R>, m![{ Symbol<Segment<R>> }, 1 # 8]>;

/// Each row's value, on the row's slice.
type Sums<R, Chips> = DmTensor<i32, Chips, Cluster, RowSlices<R>, m![1 # 8]>;

/// One value per row, a row a flit.
type Rows<R, Chips> = DmTensor<i32, Chips, Cluster, Slices, m![R, 1 # 8]>;
type RowOperand<R, Chips> = VrfTensor<i32, Chips, Cluster, Slices, m![R, 1 # 8]>;

/// One value per segment, a segment a flit, on every row's slice.
type SegmentOperand<R, Chips> =
    VrfTensor<i32, Chips, Cluster, RowSlices<R>, m![{ Symbol<Segment<R>> }, 1 # 8]>;

/// The segment table, field by field, on every row's slice.
type Columns<R, Chips> =
    DmTensor<i32, Chips, Cluster, RowSlices<R>, m![Fields, { Symbol<Segment<R>> }, 1 # 8]>;

/// The rows in their lane groups of eight.
type Members<R> = m![{ Symbol<RowGroup<R>> }, RowLane];
/// One value per row, a row a flit, in lane groups, as a vector-engine operand.
type MemberOperand<R, Chips> = VrfTensor<i32, Chips, Cluster, Slices, m![{ Members<R> }, 1 # 8]>;
/// One value per row, each lane group's rows in a packet of their own.
type Grouped<R, Chips> =
    DmTensor<i32, Chips, Cluster, Slices, m![{ Symbol<RowGroup<R>> }, PacketLane]>;

/// Each row's logical position, write slot, visible length and table entry, from the segment
/// table of a history and its flat table of block IDs.
impl<R: AxisName, B: AxisName, Chips: M + 'static, X: Memory>
    Op<(
        &HbmTensor<i32, Chips, m![R, Fields]>,
        &HbmTensor<i32, Chips, m![B]>,
    )> for Derive<R, Chips, X>
where
    X::Tensor<i32, Chips, m![R, 1 # 8]>: Borrow<HbmTensor<i32, Chips, m![R, 1 # 8]>>,
    X::Tensor<i32, Chips, Grid<R>>: Borrow<HbmTensor<i32, Chips, Grid<R>>>,
    X::Tensor<i32, Chips, Pairs<R>>: Borrow<HbmTensor<i32, Chips, Pairs<R>>>,
    X::Tensor<i32, Chips, Places<R>>: Borrow<HbmTensor<i32, Chips, Places<R>>>,
{
    type Output = Logical<R, Chips>;

    fn forward(
        &self,
        device: &mut Device,
        (segments, blocks): (
            &HbmTensor<i32, Chips, m![R, Fields]>,
            &HbmTensor<i32, Chips, m![B]>,
        ),
    ) -> Self::Output {
        let column = |device: &mut Device,
                      columns: &Columns<R, Chips>,
                      field: usize|
         -> SegmentOperand<R, Chips> {
            device
                .sub
                .begin(
                    columns
                        .view()
                        .tile::<
                            m![Fields],
                            1,
                            m![Fields = 1 # Fields, { Symbol<Segment<R>> }, 1 # 8],
                        >(field),
                )
                .fetch::<m![{ Symbol<Segment<R>> }], m![1 # 8]>()
                .collect::<m![{ Symbol<Segment<R>> }], m![1 # 8]>()
                .to_vrf()
        };
        let row_operand = |device: &mut Device, rows: &Rows<R, Chips>| -> RowOperand<R, Chips> {
            device
                .sub
                .begin(rows.view())
                .fetch::<m![R], m![1 # 8]>()
                .collect::<m![R], m![1 # 8]>()
                .to_vrf()
        };
        // `cells - value` for every row and segment.
        let less = |device: &mut Device,
                    cells: &Cells<R, Chips>,
                    value: &SegmentOperand<R, Chips>|
         -> Cells<R, Chips> {
            device
                .main
                .begin(cells.view())
                .fetch::<m![{ Symbol<Segment<R>> }], m![1 # 8]>()
                .collect::<m![{ Symbol<Segment<R>> }], m![1 # 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_fxp(FxpBinaryOp::SubFxp, value)
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit()
        };
        // Each row's `-sum over s of [d < 0] v_s`.
        let masked = |device: &mut Device,
                      d: &Cells<R, Chips>,
                      value: &SegmentOperand<R, Chips>|
         -> Sums<R, Chips> {
            device
                .main
                .begin(d.view())
                .fetch::<m![{ Symbol<Segment<R>> }], m![1 # 8]>()
                .collect::<m![{ Symbol<Segment<R>> }], m![1 # 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_fxp(FxpBinaryOp::ArithRightShift, 31)
                .vector_fxp(FxpBinaryOp::MulInt, value)
                .vector_narrow_trim::<m![1 # 4]>()
                .vector_intra_slice_reduce::<Segment<R>, m![1], m![1 # 4]>(
                    IntraSliceReduceOpI32::AddSat,
                )
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit()
        };
        // Each row's value of the segment that holds it, from the two masked sums, a row a flit
        // of one slice as the steps after the grid take their rows.
        let select = |device: &mut Device,
                      (starts, ends): (&Sums<R, Chips>, &Sums<R, Chips>)|
         -> Rows<R, Chips> {
            let ends = scalar::register(device, ends);
            let selected: Sums<R, Chips> = device
                .main
                .begin(starts.view())
                .fetch::<m![1], m![1 # 8]>()
                .collect::<m![1], m![1 # 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_fxp(FxpBinaryOp::SubFxp, &ends)
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit();
            let mut rows = Rows::<R, Chips>::new();
            selected
                .view()
                .to_dm_view(&mut device.tdma, rows.view_mut());
            rows
        };
        // Each row's value moved into its lane of its group's packet, so a store writes whole HBM
        // cells: stored a row a cell, the writes into one cell serialize.
        let packed = |device: &mut Device,
                      places: &DmTensor<i32, Chips, Cluster, Slices, Places<R>>,
                      rows: &Rows<R, Chips>|
         -> HbmTensor<i32, Chips, m![R]> {
            // SAFETY: a regroup of the rows into groups of eight, in order.
            let rows = unsafe {
                rows.view()
                    .reshape::<Chips, Cluster, Slices, m![{ Members<R> }, 1 # 8]>()
            };
            let rows: MemberOperand<R, Chips> = device
                .sub
                .begin(rows)
                .fetch::<Members<R>, m![1 # 8]>()
                .collect::<Members<R>, m![1 # 8]>()
                .to_vrf();
            let packets: Grouped<R, Chips> = device
                .main
                .begin(places.view())
                .fetch::<Members<R>, m![PacketLane]>()
                .collect::<Members<R>, m![PacketLane]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_fxp(FxpBinaryOp::MulInt, &rows)
                .vector_narrow_split::<m![{ Members<R> }, PacketLane / 4], m![PacketLane % 4]>()
                .vector_intra_slice_reduce::<
                    RowLane,
                    m![{ Symbol<RowGroup<R>> }, PacketLane / 4],
                    m![PacketLane % 4],
                >(IntraSliceReduceOpI32::AddSat)
                .vector_widen_concat::<m![{ Symbol<RowGroup<R>> }], m![PacketLane]>()
                .vector_final()
                .commit_trim::<m![PacketLane]>()
                .commit();
            // SAFETY: lane `l` of group `g`'s packet holds row `8 g + l`, the rows in order.
            let packets: DmTensor<i32, Chips, Cluster, Slices, m![R]> =
                unsafe { packets.reshape() };
            packets.to_hbm(&mut device.tdma)
        };
        // A field a flit in HBM first: a DMA into data memory moves whole eight-byte cells.
        let mut padded: HbmTensor<i32, Chips, m![Fields, { Symbol<Segment<R>> }, 1 # 8]> =
            HbmTensor::new();
        // SAFETY: a relabel of the segment axis, which keeps its size and order.
        unsafe {
            segments
                .view()
                .reshape::<Chips, m![{ Symbol<Segment<R>> }, Fields]>()
        }
        .to_hbm_view(&mut device.tdma, padded.view_mut());
        let columns: Columns<R, Chips> = padded.to_dm(&mut device.tdma);
        let start = column(device, &columns, START);
        let len = column(device, &columns, LEN);
        let table = column(device, &columns, TABLE);
        let grid: Cells<R, Chips> = self.grid.borrow().to_dm(&mut device.tdma);
        // d = r - start and e = r - end: negative exactly where the segment begins or ends past r.
        let d = less(device, &grid, &start);
        let e = less(device, &d, &len);
        // The segment's position offset, cached - start.
        let offset: DmTensor<
            i32,
            Chips,
            Cluster,
            RowSlices<R>,
            m![{ Symbol<Segment<R>> }, 1 # 8],
        > = device
            .main
            .begin(
                columns
                    .view()
                    .tile::<
                        m![Fields],
                        1,
                        m![Fields = 1 # Fields, { Symbol<Segment<R>> }, 1 # 8],
                    >(CACHED),
            )
            .fetch::<m![{ Symbol<Segment<R>> }], m![1 # 8]>()
            .collect::<m![{ Symbol<Segment<R>> }], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::SubFxp, &start)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        let offset = device
            .sub
            .begin(offset.view())
            .fetch::<m![{ Symbol<Segment<R>> }], m![1 # 8]>()
            .collect::<m![{ Symbol<Segment<R>> }], m![1 # 8]>()
            .to_vrf();
        let shift = {
            let starts = masked(device, &d, &offset);
            let ends = masked(device, &e, &offset);
            let shift = select(device, (&starts, &ends));
            row_operand(device, &shift)
        };
        let tables = {
            let starts = masked(device, &d, &table);
            let ends = masked(device, &e, &table);
            select(device, (&starts, &ends))
        };
        let iota: Rows<R, Chips> = self.iota.borrow().to_dm(&mut device.tdma);
        let places: DmTensor<i32, Chips, Cluster, Slices, Places<R>> =
            self.places.borrow().to_dm(&mut device.tdma);
        let positions: Rows<R, Chips> = device
            .main
            .begin(iota.view())
            .fetch::<m![R], m![1 # 8]>()
            .collect::<m![R], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::AddFxp, &shift)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        let lengths: Rows<R, Chips> = device
            .main
            .begin(iota.view())
            .fetch::<m![R], m![1 # 8]>()
            .collect::<m![R], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::AddFxp, &shift)
            .vector_clip(ClipBinaryOpI32::AddFxp, 1)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        // Each row's token block, flat-table entry `e = table + position / 16`: a gather moves
        // eight-byte cells, four entries, and a weight of 16 on `e`'s lane picks it.
        let tables_operand = row_operand(device, &tables);
        let entries: Rows<R, Chips> = device
            .main
            .begin(positions.view())
            .fetch::<m![R], m![1 # 8]>()
            .collect::<m![R], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_logic(
                LogicBinaryOpI32::LogicRightShift,
                const { Block::SIZE.ilog2() as i32 },
            )
            .vector_fxp(FxpBinaryOp::AddFxp, &tables_operand)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        let pairs: Rows<R, Chips> = device
            .main
            .begin(entries.view())
            .fetch::<m![R], m![1 # 8]>()
            .collect::<m![R], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_logic(LogicBinaryOpI32::BitAnd, -(Quad::SIZE as i32))
            .vector_fxp(
                FxpBinaryOp::LeftShift,
                const { size_of::<i32>().ilog2() as i32 },
            )
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        let pairs = packed(device, &places, &pairs);
        // SAFETY: the flat table holds whole window rows, a multiple of four entries.
        let blocks = unsafe { blocks.view().reshape::<Chips, m![B / 4, Quad]>() };
        let ids: DmTensor<i32, Chips, Cluster, Slices, m![R, Quad # 8]> = blocks
            .gather::<m![B / 4], m![Quad]>()
            .by_byte_offsets(&pairs)
            .to_dm(&mut device.tdma);
        let entries = row_operand(device, &entries);
        let lanes: DmTensor<i32, Chips, Cluster, Slices, m![R, Quad # 8]> =
            self.pairs.borrow().to_dm(&mut device.tdma);
        let weights: DmTensor<i32, Chips, Cluster, Slices, m![R, Quad # 8]> = device
            .main
            .begin(lanes.view())
            .fetch::<m![R], m![Quad # 8]>()
            .collect::<m![R], m![Quad # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_logic(LogicBinaryOpI32::BitXor, &entries)
            .vector_logic(LogicBinaryOpI32::BitAnd, Quad::SIZE as i32 - 1)
            .vector_fxp(FxpBinaryOp::SubFxp, 1)
            .vector_fxp(FxpBinaryOp::ArithRightShift, 31)
            .vector_fxp(FxpBinaryOp::MulInt, -(Block::SIZE as i32))
            .vector_final()
            .commit_trim::<m![Quad # 8]>()
            .commit();
        let weights: VrfTensor<i32, Chips, Cluster, Slices, m![R, Quad # 8]> = device
            .sub
            .begin(weights.view())
            .fetch::<m![R], m![Quad # 8]>()
            .collect::<m![R], m![Quad # 8]>()
            .to_vrf();
        let first: Rows<R, Chips> = device
            .main
            .begin(ids.view())
            .fetch::<m![R], m![Quad # 8]>()
            .collect::<m![R], m![Quad # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::MulInt, &weights)
            .vector_narrow_trim::<m![Quad]>()
            .vector_intra_slice_reduce::<Quad, m![R], m![1 # 4]>(IntraSliceReduceOpI32::AddSat)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        let lanes: Rows<R, Chips> = device
            .main
            .begin(positions.view())
            .fetch::<m![R], m![1 # 8]>()
            .collect::<m![R], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_logic(LogicBinaryOpI32::BitAnd, const { Block::SIZE as i32 - 1 })
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        let lanes = row_operand(device, &lanes);
        let slots: Rows<R, Chips> = device
            .main
            .begin(first.view())
            .fetch::<m![R], m![1 # 8]>()
            .collect::<m![R], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::AddFxp, &lanes)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        Logical {
            positions: packed(device, &places, &positions),
            slots: packed(device, &places, &slots),
            lengths: packed(device, &places, &lengths),
            tables: packed(device, &places, &tables),
        }
    }
}
