//! The sampler over units of 16 selected rows, a row a PE: chip `c` draws the unit's rows
//! `8 c..8 c + 8`, four a cluster, each over its PE's 64 slices and both vocabulary halves, so every
//! reduce stays in a PE.
//!
//! Layout, which needs the serving head order (chip `c`'s entry `1024 g + 8 m + l` holds token
//! `1024 m + 512 c + 8 g + l`, as the served weights bind it):
//! - slice `s` of a row's PE holds the tokens `16384 j + 256 s + f` of the eight runs `j` as landed:
//!   entry `256 j + f` (`Entry / 256` is `j`, `Entry % 256` is `f`), so time then lane is token order
//!   within a run;
//! - a chip receives the peer's half in the slices of slot `Full / 65536` 0, where it banned its
//!   own half, and copies its own half to slot 1, so a slice's place `s` is read off its landed token
//!   ids, not off its index; the peer half's ids are this chip's with the chip bit (9) flipped;
//! - each chip bans its own half of the unit's rows before the all-to-all, so no stop table crosses
//!   the chip link;
//! - a prefix in token order is a run's running sum plus the run's offset (the same run of the
//!   places before it, and the whole runs before it): the runs go two a flit by one transpose, and a
//!   log of each run's first and last flit, on every lane, segments one running sum over them;
//! - units chain: a unit's searches run while the next unit's parameters and logits land in its
//!   `Ahead` and its peer half crosses the link into a fresh owner, so no transfer waits on the
//!   searches; the stop table and the token order are a launch's, landed once.
//!
//! A row's draw: top-k by a 16-bit count search over the order keys, top-p by the 16-bit level
//! search over keys and weights, the boundary level's first ties in token order by their running
//! count, and the token by the running kept weight at the uniform.

use furiosa_opt_std::prelude::*;

use super::{Full, Table};
use crate::models::nemo::serving::Chips;
use crate::models::nemo::serving::head::Vocabulary;
use crate::models::nemo::serving::residual::Term;

axes![
    Batch = 16,
    Entry = 2048,
    Pos = 256,
    Run = 8,
    Wide = 64,
    Spread = 64,
    Gathered = 64,
    Copy = 8,
    Half = 1024,
    Quad = 4,
    Two = 2,
    Param = 3
];

/// Below every key.
const LOWEST: i32 = -32768;
/// Entries of a row over both chips.
const ENTRIES: i32 = 131072;
/// The peer half's token of a reversed id `r`: `r ^ PEER_TOKENS` (bit 9 is the chip).
const PEER_TOKENS: i32 = (ENTRIES - 1) ^ 512;

/// A row's cluster.
pub type Cluster = m![Batch % 8 / 4];
/// The chip that draws a row.
pub type Block = m![Batch / 8];
/// A row's PE; slot `Full / 65536` 0 holds this chip's half.
pub type Slices = m![
    Batch % 4,
    Full / 65536,
    Full % 128 / 8,
    Full % 65536 / 32768
];
/// One half's slices of every PE.
type HalfSlices = m![Batch % 4, 1 # 2, Full % 128 / 8, Full % 65536 / 32768];
/// A slice's entries as landed: run `Full % 1024 / 128`, then its 256 positions.
type Natural = m![Full % 1024 / 128, Full % 32768 / 1024, Full % 8];
/// A per-entry tensor of a row's PE, run then position.
pub type Values<D> = DmTensor<D, Chips, Cluster, Slices, m![Entry]>;
/// A unit's logits of this chip's rows over both halves, as landed.
pub type Logits = DmTensor<bf16, Chips, Cluster, Slices, Natural>;
/// A unit's logits of this chip's rows in the peer's half, received in slot 0's slices.
pub type Peer = DmTensor<bf16, Chips, Cluster, HalfSlices, Natural>;
/// A unit's logits of this chip's half for both chips' rows.
pub type Halves = DmTensor<
    bf16,
    Chips,
    Cluster,
    HalfSlices,
    m![Batch / 8, Full % 1024 / 128, Full % 32768 / 1024, Full % 8],
>;
/// One value a row, in every slice of its PE.
pub type Row<D> = DmTensor<D, Chips, Cluster, Slices, m![1 # 8]>;
type Operand<D> = VrfTensor<D, Chips, Cluster, Slices, m![1 # 8]>;
/// One value a run, in a slice.
type Lanes<D> = DmTensor<D, Chips, Cluster, Slices, m![Run]>;
type LaneOperand<D> = VrfTensor<D, Chips, Cluster, Slices, m![Run]>;
/// One value a run of every place of a PE.
type Gather<D> = DmTensor<D, Chips, Cluster, Slices, m![Gathered, Run]>;
type GatherOperand<D> = VrfTensor<D, Chips, Cluster, Slices, m![Gathered, Run]>;
/// A PE reduce's result, in every slice of the PE.
type Reduced<D, E> = DmTensor<D, Chips, Cluster, m![Batch % 4, Wide], E>;
/// A unit's per-row parameter of both chips' rows, on the first slice of each PE.
pub type Both<D> = DmTensor<D, Chips, Cluster, m![Batch % 4, 1 # 64], m![Batch / 8, 1 # 8]>;
/// This chip's rows of a parameter on each PE's first slice, zero on the rest.
type Sums<D> = DmTensor<D, Chips, Cluster, m![Batch % 4, Spread], m![1 # 8]>;

const FIRST: [BitReq; 4] = [BitReq::One, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore];
const LAST: [BitReq; 4] = [BitReq::Ignore, BitReq::One, BitReq::Ignore, BitReq::Ignore];

/// A unit's draw parameters, one a row.
pub struct Draw {
    /// The f32 parameters, `INVERSE`, `TOP_P` and `UNIFORM`.
    params: Params,
    /// The top-k (1..=131072) as the count search's offset: 2048, minus the top-k on a PE's first
    /// slice, so a PE reduce of the per-slice counts of 2048 entries gives the count minus it.
    pub top_k: Operand<i32>,
}

/// A row's f32 draw parameters in every slice of its PE, one a flit.
type Params = DmTensor<f32, Chips, Cluster, Slices, m![Param, 1 # 8]>;
/// `1 / max(temperature, 1e-5)`.
const INVERSE: usize = 0;
/// The top-p; +inf keeps the whole top-k.
const TOP_P: usize = 1;
/// The draw, in [0, 1).
const UNIFORM: usize = 2;

/// Draw parameter `J` of every row as an operand.
fn param<const J: usize>(device: &mut Device, params: &Params) -> Operand<f32> {
    device
        .sub
        .begin(
            params
                .view()
                .tile::<m![Param], 1, m![Param = 1 # 3, 1 # 8]>(J),
        )
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf::<m![1 # 8]>()
}

/// The token order of a row's PE: each entry's token id and the masks of each slice's place.
pub struct Order {
    /// Each entry's token id.
    tokens: Values<i32>,
    /// The tags of each run's first and last paired flit, on every lane, built once a launch.
    log: Log,
    /// The tags of each run's first and last position, one run a lane, built once a launch.
    lanes: LaneLog,
    /// -1 at this slice's place, 0 elsewhere.
    onehot: Gather<i32>,
    /// -1 at the places below this slice's.
    before: Gather<i32>,
    /// -1 in the runs below this slice's place (places 0..8 gather the run offsets).
    below: Lanes<i32>,
    /// -1 in the run of this slice's place, for places 0..8.
    own: Lanes<i32>,
}

/// The tables of a row's token order: their masks set while the first unit's peer half crosses the
/// link, the token ids landed under the first unit's searches.
pub struct Landed {
    /// Each entry's reversed token id, this chip's in both slots.
    reversed: DmTensor<i32, Chips, Cluster, Slices, Natural>,
    /// The bits that turn a reversed id into its token: `ENTRIES - 1`, and bit 9 on slot 0.
    flip: Row<i32>,
    /// 1 on every lane of each run's first paired flit, 2 on its last, 0 elsewhere.
    marks: DmTensor<i8, Chips, Cluster, Slices, m![Quad, Pos, Copy]>,
    /// 1 on every lane of the first position, 2 on the last, 0 elsewhere.
    ends: DmTensor<i8, Chips, Cluster, Slices, m![Pos, Copy]>,
}

// The serving head order puts the peer chip's token of an entry at this chip's token with bit 9
// flipped, so both slots take this chip's reversed `ids` and slot 0 flips the bit.
impl Landed {
    /// The masks; the ids land later, by `land`.
    pub fn new(device: &mut Device) -> Self {
        // `ENTRIES - 1 - id` is `id ^ (ENTRIES - 1)` for ids below `ENTRIES`; slot 0 also flips bit 9.
        let mut flip = Row::<i32>::new();
        flip.view_mut().memset(PEER_TOKENS, &mut device.sub);
        let mut own = Row::<i32>::new();
        own.view_mut().memset(ENTRIES - 1, &mut device.sub);
        own.view()
            .slice_tile::<m![Full / 65536], 1, PlaceSlot>(1)
            .to_dm_view(
                &mut device.tdma,
                flip.view_mut().slice_tile::<m![Full / 65536], 1, Slot>(1),
            );
        // 8-bit marks: a quarter of the bytes to set while the first unit's peer half crosses.
        let mut marks = DmTensor::<i8, Chips, Cluster, Slices, m![Quad, Pos, Copy]>::new();
        marks.view_mut().memset(0i8, &mut device.sub);
        marks
            .view_mut()
            .tile::<m![Pos], 1, m![Quad, Pos = 1 #{!} 256, Copy]>(0)
            .memset(1i8, &mut device.sub);
        marks
            .view_mut()
            .tile::<m![Pos], 1, m![Quad, Pos = 1 #{!} 256, Copy]>(255)
            .memset(2i8, &mut device.sub);
        let mut ends = DmTensor::<i8, Chips, Cluster, Slices, m![Pos, Copy]>::new();
        ends.view_mut().memset(0i8, &mut device.sub);
        ends.view_mut()
            .tile::<m![Pos], 1, m![Pos = 1 #{!} 256, Copy]>(0)
            .memset(1i8, &mut device.sub);
        ends.view_mut()
            .tile::<m![Pos], 1, m![Pos = 1 #{!} 256, Copy]>(255)
            .memset(2i8, &mut device.sub);
        Self {
            reversed: DmTensor::new(),
            flip,
            marks,
            ends,
        }
    }

    /// This chip's reversed `ids` into both slots in one transfer (a second transfer into the same
    /// owner would hold the in-order command stream until the first lands). Nothing reads them
    /// before the order passes, so issued behind the first unit's loads the transfer holds no command.
    pub fn land(&mut self, device: &mut Device, ids: &Ids) {
        // SAFETY: the slot digit is renamed to an axis the ids lack, so the transfer copies them to
        // both slots; the mapping keeps the slices' order.
        let both = unsafe {
            self.reversed
                .view_mut()
                .reshape::<Chips, Cluster, m![Batch % 4, Two, Full % 128 / 8, Full % 65536 / 32768], Natural>()
        };
        local(ids).to_dm_view(&mut device.tdma, both);
    }

    /// The token order's passes over the landed tables.
    pub fn order(&self, device: &mut Device) -> Order {
        let Self {
            reversed,
            flip,
            marks,
            ends,
        } = self;
        // SAFETY: as landed, `Entry` is run then position.
        let reversed = unsafe {
            reversed
                .view()
                .reshape::<Chips, Cluster, Slices, m![Entry]>()
        };
        let flip = operand(device, flip);
        // Token ids, then each slice's place and each entry's position.
        let tokens: Values<i32> = device
            .main
            .begin(reversed)
            .fetch::<m![Entry / 8], m![Entry % 8]>()
            .collect::<m![Entry / 8], m![Entry % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_logic(LogicBinaryOpI32::BitXor, &flip)
            .vector_final()
            .commit_trim::<m![Entry % 8]>()
            .commit();
        let positions: Values<i32> = device
            .main
            .begin(tokens.view())
            .fetch::<m![Entry / 8], m![Entry % 8]>()
            .collect::<m![Entry / 8], m![Entry % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_logic(LogicBinaryOpI32::BitAnd, 255i32)
            .vector_final()
            .commit_trim::<m![Entry % 8]>()
            .commit();
        let place: Row<i32> = device
            .main
            .begin(tokens.view())
            .fetch::<m![Entry / 8], m![Entry % 8]>()
            .collect::<m![Entry / 8], m![Entry % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_logic(LogicBinaryOpI32::ArithRightShift, 8i32)
            .vector_logic(LogicBinaryOpI32::BitAnd, 63i32)
            .vector_narrow_split::<m![Entry / 4], m![Entry % 4]>()
            .vector_intra_slice_reduce::<Entry, m![1], m![1 # 4]>(IntraSliceReduceOpI32::Max)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        let next = add::<1>(device, &place);
        let place = operand(device, &place);
        let next = operand(device, &next);
        // Each entry's flit in its slice, `32 run + position / 8`: the first 64 flits count the
        // places, a lane each run.
        let flits: Values<i32> = device
            .main
            .begin_interleaved::<Term, _, _, _, _, _>(positions.view(), tokens.view())
            .fetch::<m![Entry / 128, Term, Entry % 128 / 8], m![Entry % 8]>()
            .collect::<m![Entry / 128, Term, Entry % 128 / 8], m![Entry % 8]>()
            .vector_init()
            .vector_intra_slice_unzip::<Term, m![Entry / 8]>()
            .vector_logic(LogicBinaryOpI32::ArithRightShift, 3i32, 9i32)
            .vector_logic(LogicBinaryOpI32::BitAnd, 31i32, 224i32)
            .vector_logic_zip(LogicBinaryOpI32::BitOr)
            .vector_final()
            .commit_trim::<m![Entry % 8]>()
            .commit();
        let before = compare(device, counted(&flits), &place);
        let upto = compare(device, counted(&flits), &next);
        let onehot = difference(device, (&upto, &before));
        // SAFETY: the first flit's eight entries, renamed: their lane is `token % 8`.
        let first = unsafe {
            tokens
                .view()
                .tile::<m![Entry], 8, m![Entry = 8 # 2048]>(0)
                .reshape::<Chips, Cluster, Slices, m![Run # 2048]>()
        };
        let runs: Lanes<i32> = device
            .main
            .begin(first)
            .fetch::<m![1], m![Run]>()
            .collect::<m![1], m![Run]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_logic(LogicBinaryOpI32::BitAnd, 7i32)
            .vector_final()
            .commit_trim::<m![Run]>()
            .commit();
        let below = compare_lanes(device, &runs, &place);
        let upto = compare_lanes(device, &runs, &next);
        let own = difference_lanes(device, (&upto, &below));
        // SAFETY: eight lanes a flit either way; the pad lanes keep the live lanes' marks.
        let marks = unsafe {
            marks
                .view()
                .reshape::<Chips, Cluster, Slices, m![Quad, Pos, Two # 8]>()
        };
        let log = device
            .main
            .begin(marks)
            .fetch::<m![Quad, Pos], m![Two # 8]>()
            .fetch_cast::<i32>()
            .collect::<m![Quad, Pos], m![Two # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Comparison([
                Cmp::Equal(1),
                Cmp::Equal(2),
                Cmp::False,
                Cmp::False,
            ]))
            .vector_log(&mut device.sub);
        // SAFETY: eight lanes a flit either way; every lane holds the same mark.
        let ends = unsafe {
            ends.view()
                .reshape::<Chips, Cluster, Slices, m![Pos, Run]>()
        };
        let lanes = device
            .main
            .begin(ends)
            .fetch::<m![Pos], m![Run]>()
            .fetch_cast::<i32>()
            .collect::<m![Pos], m![Run]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Comparison([
                Cmp::Equal(1),
                Cmp::Equal(2),
                Cmp::False,
                Cmp::False,
            ]))
            .vector_log(&mut device.sub);
        Order {
            tokens,
            log,
            lanes,
            onehot,
            before,
            below,
            own,
        }
    }
}

/// The first 64 flits, a flit a place of the PE.
fn counted(
    flits: &Values<i32>,
) -> DmTensorView<'_, i32, Chips, Cluster, Slices, m![Gathered # 256, Run]> {
    // SAFETY: the first 64 flits are the first 512 entries, renamed.
    unsafe {
        flits
            .view()
            .tile::<m![Entry], 512, m![Entry = 512 # 2048]>(0)
            .reshape::<Chips, Cluster, Slices, m![Gathered # 256, Run]>()
    }
}

/// -1 where `values` is below `place`, 0 elsewhere.
fn compare(
    device: &mut Device,
    values: DmTensorView<'_, i32, Chips, Cluster, Slices, m![Gathered # 256, Run]>,
    place: &Operand<i32>,
) -> Gather<i32> {
    device
        .main
        .begin(values)
        .fetch::<m![Gathered], m![Run]>()
        .collect::<m![Gathered], m![Run]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::SubFxpSat, place)
        .vector_fxp(FxpBinaryOp::ArithRightShift, 31i32)
        .vector_final()
        .commit_trim::<m![Run]>()
        .commit()
}

/// `left - right` of two masks.
fn difference(device: &mut Device, (left, right): (&Gather<i32>, &Gather<i32>)) -> Gather<i32> {
    device
        .main
        .begin_interleaved::<Term, _, _, _, _, _>(left.view(), right.view())
        .fetch::<m![Gathered, Term], m![Run]>()
        .collect::<m![Gathered, Term], m![Run]>()
        .vector_init()
        .vector_intra_slice_unzip::<Term, m![Gathered]>()
        .vector_fxp_zip(FxpBinaryOp::SubFxp)
        .vector_final()
        .commit_trim::<m![Run]>()
        .commit()
}

/// -1 in the runs below `place`.
fn compare_lanes(device: &mut Device, runs: &Lanes<i32>, place: &Operand<i32>) -> Lanes<i32> {
    device
        .main
        .begin(runs.view())
        .fetch::<m![1], m![Run]>()
        .collect::<m![1], m![Run]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::SubFxpSat, place)
        .vector_fxp(FxpBinaryOp::ArithRightShift, 31i32)
        .vector_final()
        .commit_trim::<m![Run]>()
        .commit()
}

fn difference_lanes(device: &mut Device, (left, right): (&Lanes<i32>, &Lanes<i32>)) -> Lanes<i32> {
    device
        .main
        .begin_interleaved::<Term, _, _, _, _, _>(left.view(), right.view())
        .fetch::<m![Term], m![Run]>()
        .collect::<m![Term], m![Run]>()
        .vector_init()
        .vector_intra_slice_unzip::<Term, m![1]>()
        .vector_fxp_zip(FxpBinaryOp::SubFxp)
        .vector_final()
        .commit_trim::<m![Run]>()
        .commit()
}

/// One PE slot of a PE's slices: `Slot` written, `PlaceSlot` read.
type Slot = m![Batch % 4, Full / 65536 = 1 #{!} 2, Full % 128 / 8, Full % 65536 / 32768];
type PlaceSlot = m![Batch % 4, Full / 65536 = 1 # 2, Full % 128 / 8, Full % 65536 / 32768];
/// This chip's token ids, reversed (`ENTRIES - 1 - id`), in its head order.
pub type Ids = HbmTensor<i32, Chips, m![Vocabulary]>;

/// This chip's half of a per-entry `table`.
fn local<D: VeScalar, E: M>(
    table: &HbmTensor<D, Chips, E>,
) -> HbmTensorView<'_, D, Chips, m![Full % 65536]> {
    // SAFETY: a chip's table entries are its half's entries; only the axis is renamed.
    unsafe { table.view().reshape::<Chips, m![Full % 65536]>() }
}

/// A unit's halves' entries: a row block, then the landed order.
type HalfElement = m![Batch / 8, Full % 1024 / 128, Full % 32768 / 1024, Full % 8];
/// One row block of a unit's halves, read and written.
type HalfBlock = m![Batch / 8 = 1 # 2, Full % 1024 / 128, Full % 32768 / 1024, Full % 8];
type HalfBlockMut = m![Batch / 8 = 1 #{!} 2, Full % 1024 / 128, Full % 32768 / 1024, Full % 8];
/// The stop table of this chip's half in row block 0 of each slice of a unit's halves, `f32::MIN`
/// at each stop token's entry and 0.0 elsewhere; both row blocks read it.
pub type Stops = DmTensor<f32, Chips, Cluster, HalfSlices, HalfElement>;

/// The stop table of this chip's half.
pub fn stops(device: &mut Device, table: &Table) -> Stops {
    let mut stops = Stops::new();
    local(&table.stops).to_dm_view(
        &mut device.tdma,
        stops.view_mut().tile::<m![Batch / 8], 1, HalfBlockMut>(0),
    );
    stops
}

/// Row block `AT` of the halves' `values` with a banned row's stop entries at -inf (a stop entry's
/// `f32::MIN` times 2 overflows), rounded back to BF16 into `banned`.
fn ban<const AT: usize>(
    device: &mut Device,
    (values, stops): (
        &DmTensor<f32, Chips, Cluster, HalfSlices, HalfElement>,
        &Stops,
    ),
    bans: &DmTensor<f32, Chips, Cluster, Slices, m![Batch / 8, 1 # 8]>,
    banned: &mut Halves,
) {
    // SAFETY: slot 0's slices of every PE are the halves' slices; only the mapping is renamed.
    let half = unsafe {
        bans.view()
            .slice_tile::<m![Full / 65536], 1, PlaceSlot>(0)
            .reshape::<Chips, Cluster, HalfSlices, m![Batch / 8, 1 # 8]>()
    };
    let ban: VrfTensor<f32, Chips, Cluster, HalfSlices, m![1 # 8]> = device
        .sub
        .begin(half.tile::<m![Batch / 8], 1, m![Batch / 8 = 1 # 2, 1 # 8]>(AT))
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf::<m![1 # 8]>();
    device
        .main
        .begin_interleaved::<Term, _, _, _, _, _>(
            values.view().tile::<m![Batch / 8], 1, HalfBlock>(AT),
            stops.view().tile::<m![Batch / 8], 1, HalfBlock>(0),
        )
        .fetch::<m![Full % 1024 / 128, Term, Full % 32768 / 1024], m![Full % 8]>()
        .collect::<m![Full % 1024 / 128, Term, Full % 32768 / 1024], m![Full % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Term, m![Full % 1024 / 128, Full % 32768 / 1024]>()
        .vector_narrow_split::<m![Full % 1024 / 128, Full % 32768 / 1024, Full % 8 / 4], m![Full % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), (), &ban)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), (), 2.0f32)
        .vector_fp_zip(FpBinaryOp::AddF)
        .vector_widen_concat::<m![Full % 1024 / 128, Full % 32768 / 1024], m![Full % 8]>()
        .vector_final()
        .cast::<bf16, m![Full % 8 # 16]>()
        .commit_trim::<m![Full % 8]>()
        .commit_view(banned.view_mut().tile::<m![Batch / 8], 1, HalfBlockMut>(AT));
}

/// A unit's HBM rows: its logits of this chip's half, then its per-row parameters.
pub struct Rows<'a, E: M, F: M> {
    pub logits: HbmTensorView<'a, bf16, Chips, E>,
    pub inverse: HbmTensorView<'a, f32, Chips, F>,
    pub top_k: HbmTensorView<'a, i32, Chips, F>,
    pub top_p: HbmTensorView<'a, f32, Chips, F>,
    pub ban: HbmTensorView<'a, f32, Chips, F>,
    pub uniform: HbmTensorView<'a, f32, Chips, F>,
}

/// One f32 parameter's flit of the landed parameters, written.
type ParamSlot = m![Batch / 8, Param = 1 #{!} 3, 1 # 8];

/// A unit's per-row draw parameters: the top-k, and the f32 ones a flit each.
struct Parameters<I, F> {
    top_k: I,
    params: F,
}

/// A unit's inputs landed ahead of its chain: its logits, banned and exchanged, and its per-row
/// parameters, both chips' rows as landed and then this chip's on each PE's first slice of zeros,
/// so one PE reduce spreads each.
pub struct Ahead {
    logits: Logits,
    halves: Halves,
    /// The halves banned: owned here so that no pass reuses their memory while the exchange reads it.
    banned: Halves,
    ban: DmTensor<f32, Chips, Cluster, m![Batch % 4, Spread], m![Batch / 8, 1 # 8]>,
    landed: Parameters<
        Both<i32>,
        DmTensor<f32, Chips, Cluster, m![Batch % 4, 1 # 64], m![Batch / 8, Param, 1 # 8]>,
    >,
    own: Parameters<
        Sums<i32>,
        DmTensor<f32, Chips, Cluster, m![Batch % 4, Spread], m![Param, 1 # 8]>,
    >,
}

impl Ahead {
    pub fn new(device: &mut Device) -> Self {
        let mut ahead = Self {
            logits: Logits::new(),
            halves: Halves::new(),
            banned: Halves::new(),
            ban: DmTensor::new(),
            landed: Parameters {
                top_k: Both::new(),
                params: DmTensor::new(),
            },
            own: Parameters {
                top_k: Sums::new(),
                params: DmTensor::new(),
            },
        };
        ahead.ban.view_mut().memset(0.0f32, &mut device.sub);
        ahead.own.top_k.view_mut().memset(0i32, &mut device.sub);
        ahead.own.params.view_mut().memset(0.0f32, &mut device.sub);
        ahead
    }

    /// A unit's loads from its HBM `rows`, none of them waiting for another transfer.
    pub fn land<E: M, F: M>(&mut self, device: &mut Device, rows: Rows<'_, E, F>) {
        let Rows {
            logits,
            inverse,
            top_k,
            top_p,
            ban,
            uniform,
        } = rows;
        ban.to_dm_view(
            &mut device.tdma,
            self.ban
                .view_mut()
                .slice_tile::<m![Spread], 1, m![Batch % 4, Spread = 1 #{!} 64]>(0),
        );
        let params = &mut self.landed.params;
        inverse.to_dm_view(
            &mut device.tdma,
            params.view_mut().tile::<m![Param], 1, ParamSlot>(INVERSE),
        );
        top_k.to_dm_view(&mut device.tdma, self.landed.top_k.view_mut());
        top_p.to_dm_view(
            &mut device.tdma,
            params.view_mut().tile::<m![Param], 1, ParamSlot>(TOP_P),
        );
        uniform.to_dm_view(
            &mut device.tdma,
            params.view_mut().tile::<m![Param], 1, ParamSlot>(UNIFORM),
        );
        logits.to_dm_view(&mut device.tdma, self.halves.view_mut());
    }

    /// This chip's rows of the landed parameters, then the landed halves with a banned row's stop
    /// entries at -inf (a stop entry's `f32::MIN` times 2 overflows), rounded back to BF16, this
    /// chip's rows of them into `logits`; `send` then sends the peer's half of them.
    pub fn exchange(&mut self, device: &mut Device, stops: &Stops) {
        land_own(device, &self.landed.top_k, &mut self.own.top_k);
        self.landed
            .params
            .view()
            .chip_slice::<Block, m![Param, 1 # 8]>([0, 1])
            .to_dm_view(
                &mut device.tdma,
                self.own
                    .params
                    .view_mut()
                    .slice_tile::<m![Spread], 1, m![Batch % 4, Spread = 1 #{!} 64]>(0),
            );
        let reduced: Reduced<f32, m![Batch / 8, 1 # 8]> = device
            .main
            .begin(self.ban.view())
            .fetch::<m![Batch / 8], m![1 # 8]>()
            .collect::<m![Batch / 8], m![1 # 8]>()
            .vector_init()
            .vector_inter_slice_reduce::<m![Batch % 4, Wide], m![Batch / 8]>(
                InterSliceReduceOpF32::Add,
            )
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        let bans = spread(reduced);
        let values: DmTensor<f32, Chips, Cluster, HalfSlices, HalfElement> = device
            .main
            .begin(self.halves.view())
            .fetch::<m![Batch / 8, Full % 1024 / 128, Full % 32768 / 1024], m![Full % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Batch / 8, Full % 1024 / 128, Full % 32768 / 1024], m![Full % 8]>()
            .commit_trim::<m![Full % 8]>()
            .commit();
        ban::<0>(device, (&values, stops), &bans, &mut self.banned);
        ban::<1>(device, (&values, stops), &bans, &mut self.banned);
        self.banned
            .view()
            .chip_slice::<Block, m![Full % 1024 / 128, Full % 32768 / 1024, Full % 8]>([0, 1])
            .to_dm_view(
                &mut device.tdma,
                self.logits
                    .view_mut()
                    .slice_tile::<m![Full / 65536], 1, Slot>(1),
            );
    }

    /// The peer's half of the banned halves over the chip link, sent right after `exchange`; its
    /// chip sync waits for every transfer issued before it.
    pub fn send(&mut self, device: &mut Device) -> Peer {
        self.banned
            .view()
            .chip_shuffle([1, 0])
            .chip_slice::<Block, m![Full % 1024 / 128, Full % 32768 / 1024, Full % 8]>([0, 1])
            .to_dm(&mut device.tdma)
    }

    /// The peer half from `exchange` into the logits' slot 0, once its transfer is done.
    pub fn receive(&mut self, device: &mut Device, peer: &Peer) {
        // SAFETY: slot 0's slices of every PE are the halves' slices; only the mapping is renamed.
        let slot = unsafe {
            self.logits
                .view_mut()
                .slice_tile::<m![Full / 65536], 1, Slot>(0)
                .reshape::<Chips, Cluster, HalfSlices, Natural>()
        };
        peer.view().to_dm_view(&mut device.tdma, slot);
    }

    /// The landed unit's parameters in every slice of a row's PE.
    fn draw(&self, device: &mut Device) -> Draw {
        let reduced: Reduced<f32, m![Param, 1 # 8]> = device
            .main
            .begin(self.own.params.view())
            .fetch::<m![Param], m![1 # 8]>()
            .collect::<m![Param], m![1 # 8]>()
            .vector_init()
            .vector_inter_slice_reduce::<m![Batch % 4, Wide], m![Param]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        Draw {
            params: spread(reduced),
            top_k: {
                // SAFETY: a PE's slices in wire order, the first slice first; only renamed.
                let top_k = unsafe {
                    self.own
                        .top_k
                        .view()
                        .reshape::<Chips, Cluster, Slices, m![1 # 8]>()
                };
                device
                    .main
                    .begin(top_k)
                    .fetch::<m![1], m![1 # 8]>()
                    .collect::<m![1], m![1 # 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_fxp_with_mode(FxpBinaryOp::SubFxpSat, BinaryArgMode::Mode10, 2048i32)
                    .vector_final()
                    .to_vrf::<m![1 # 8]>(&mut device.sub)
            },
        }
    }
}

fn operand<D: VeScalar>(device: &mut Device, value: &Row<D>) -> Operand<D> {
    device
        .sub
        .begin(value.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf::<m![1 # 8]>()
}

fn lane_operand<D: VeScalar>(device: &mut Device, value: &Lanes<D>) -> LaneOperand<D> {
    device
        .sub
        .begin(value.view())
        .fetch::<m![1], m![Run]>()
        .collect::<m![1], m![Run]>()
        .to_vrf::<m![Run]>()
}

fn gather_operand<D: VeScalar>(device: &mut Device, value: &Gather<D>) -> GatherOperand<D> {
    device
        .sub
        .begin(value.view())
        .fetch::<m![Gathered], m![Run]>()
        .collect::<m![Gathered], m![Run]>()
        .to_vrf::<m![Gathered, Run]>()
}

fn spread<D: VeScalar, E: M>(reduced: Reduced<D, E>) -> DmTensor<D, Chips, Cluster, Slices, E> {
    // SAFETY: the PE reduce gives every slice of a PE the same values.
    unsafe { reduced.reshape::<Chips, Cluster, Slices, E>() }
}

/// The unit's logits as landed, run then position: `Entry`'s order.
fn landed(logits: &Logits) -> DmTensorView<'_, bf16, Chips, Cluster, Slices, m![Entry]> {
    // SAFETY: as landed, `Entry` is run then position.
    unsafe { logits.view().reshape::<Chips, Cluster, Slices, m![Entry]>() }
}

/// Each entry's order key: `(k + 1) >> 16` of the f32 bits' signed monotone order `k`.
pub fn keys(device: &mut Device, logits: &Logits) -> Values<i32> {
    device
        .main
        .begin(landed(logits))
        .fetch::<m![Entry / 8], m![Entry % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Entry / 8], m![Entry % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_stash()
        .vector_logic(LogicBinaryOpI32::ArithRightShift, 31i32)
        .vector_logic(LogicBinaryOpI32::BitAnd, i32::MAX)
        .vector_logic(LogicBinaryOpI32::BitXor, Stash)
        .vector_fxp(FxpBinaryOp::AddFxp, 1i32)
        .vector_fxp(FxpBinaryOp::ArithRightShift, 16i32)
        .vector_final()
        .commit_trim::<m![Entry % 8]>()
        .commit()
}

/// Each entry's weight `exp((x - peak) * inverse)`, as its f32 bits.
fn weights(
    device: &mut Device,
    logits: &Logits,
    (peak, inverse): (&Operand<f32>, &Operand<f32>),
) -> Values<i32> {
    device
        .main
        .begin(landed(logits))
        .fetch::<m![Entry / 8], m![Entry % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Entry / 8], m![Entry % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Entry / 4], m![Entry % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, peak)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), inverse)
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_widen_concat::<m![Entry / 8], m![Entry % 8]>()
        .vector_reinterpret::<i32>()
        .vector_final()
        .commit_trim::<m![Entry % 8]>()
        .commit()
}

/// `weights` (f32 bits) where `(keys - threshold) >> SHIFT` is at least 1, i.e. keys at or above
/// `threshold + (1 << SHIFT)`, else 0.
fn select<const SHIFT: i32>(
    device: &mut Device,
    (keys, weights): (&Values<i32>, &Values<i32>),
    threshold: &Operand<i32>,
) -> Values<i32> {
    device
        .main
        .begin_interleaved::<Term, _, _, _, _, _>(keys.view(), weights.view())
        .fetch::<m![Entry / 128, Term, Entry % 128 / 8], m![Entry % 8]>()
        .collect::<m![Entry / 128, Term, Entry % 128 / 8], m![Entry % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Term, m![Entry / 8]>()
        .vector_fxp(FxpBinaryOp::SubFxpSat, threshold, ())
        .vector_fxp(FxpBinaryOp::ArithRightShift, SHIFT, ())
        .vector_fxp(FxpBinaryOp::LeftShiftSat, 31i32, ())
        .vector_clip_zip(ClipBinaryOpI32::Min)
        .vector_clip(ClipBinaryOpI32::Max, 0i32)
        .vector_final()
        .commit_trim::<m![Entry % 8]>()
        .commit()
}

/// Each row's sum of the f32 `bits`.
fn mass(device: &mut Device, bits: &Values<i32>) -> Row<f32> {
    let reduced: Reduced<f32, m![1 # 8]> = device
        .main
        .begin(bits.view())
        .fetch::<m![Entry / 8], m![Entry % 8]>()
        .collect::<m![Entry / 8], m![Entry % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![Entry / 4], m![Entry % 4]>()
        .vector_intra_slice_reduce::<Entry, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<m![Batch % 4, Wide], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    spread(reduced)
}

/// Each row's maximum of `values`.
fn maximum(device: &mut Device, logits: &Logits) -> Row<f32> {
    let reduced: Reduced<f32, m![1 # 8]> = device
        .main
        .begin(landed(logits))
        .fetch::<m![Entry / 8], m![Entry % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Entry / 8], m![Entry % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Entry / 4], m![Entry % 4]>()
        .vector_intra_slice_reduce::<Entry, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<m![Batch % 4, Wide], m![1]>(InterSliceReduceOpF32::Max)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    spread(reduced)
}

/// Each row's count of `keys` below `pivot`.
fn below(device: &mut Device, keys: &Values<i32>, pivot: &Operand<i32>) -> Row<i32> {
    let reduced: Reduced<i32, m![1 # 8]> = device
        .main
        .begin(keys.view())
        .fetch::<m![Entry / 8], m![Entry % 8]>()
        .collect::<m![Entry / 8], m![Entry % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::SubFxpSat, pivot)
        .vector_fxp(FxpBinaryOp::LogicRightShift, 31i32)
        .vector_narrow_split::<m![Entry / 4], m![Entry % 4]>()
        .vector_intra_slice_reduce::<Entry, m![1], m![1 # 4]>(IntraSliceReduceOpI32::AddSat)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<m![Batch % 4, Wide], m![1]>(InterSliceReduceOpI32::AddSat)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    spread(reduced)
}

/// This chip's rows of `rows` on each PE's first slice of `sums`.
fn land_own<D: VeScalar>(device: &mut Device, rows: &Both<D>, sums: &mut Sums<D>) {
    rows.view()
        .chip_slice::<Block, m![1 # 8]>([0, 1])
        .to_dm_view(
            &mut device.tdma,
            sums.view_mut()
                .slice_tile::<m![Spread], 1, m![Batch % 4, Spread = 1 #{!} 64]>(0),
        );
}

/// `value + STEP` in every row.
fn add<const STEP: i32>(device: &mut Device, value: &Row<i32>) -> Row<i32> {
    device
        .main
        .begin(value.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::AddFxp, STEP)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

/// `value * factor`, then the f32 bits of the product minus `OFFSET`.
fn target<const OFFSET: i32>(
    device: &mut Device,
    value: &Row<f32>,
    factor: &Operand<f32>,
) -> Row<i32> {
    device
        .main
        .begin(value.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), factor)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_reinterpret::<i32>()
        .vector_clip(ClipBinaryOpI32::AddFxpSat, -OFFSET)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

/// `threshold + STEP` where `total` (non-negative f32) is above `bound` (the f32 bits of the target,
/// minus one for "at least"), else `threshold`.
fn advance<const STEP: i32>(
    device: &mut Device,
    total: &Row<f32>,
    (bound, threshold): (&Operand<i32>, &Operand<i32>),
) -> Row<i32> {
    device
        .main
        .begin(total.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_fxp_with_mode(FxpBinaryOp::SubFxpSat, BinaryArgMode::Mode10, bound)
        .vector_fxp(FxpBinaryOp::ArithRightShift, 31i32)
        .vector_fxp(FxpBinaryOp::MulInt, -STEP)
        .vector_clip(ClipBinaryOpI32::AddFxp, threshold)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

/// Each row's count of `keys` at or above `candidate` minus its top-k (`top_k` from `Ahead::draw`).
fn excess(
    device: &mut Device,
    keys: &Values<i32>,
    (candidate, top_k): (&Operand<i32>, &Operand<i32>),
) -> Row<i32> {
    let reduced: Reduced<i32, m![1 # 8]> = device
        .main
        .begin(keys.view())
        .fetch::<m![Entry / 8], m![Entry % 8]>()
        .collect::<m![Entry / 8], m![Entry % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::SubFxpSat, candidate)
        .vector_fxp(FxpBinaryOp::ArithRightShift, 31i32)
        .vector_narrow_split::<m![Entry / 4], m![Entry % 4]>()
        .vector_intra_slice_reduce::<Entry, m![1], m![1 # 4]>(IntraSliceReduceOpI32::AddSat)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpI32::AddFxpSat, top_k)
        .vector_inter_slice_reduce::<m![Batch % 4, Wide], m![1]>(InterSliceReduceOpI32::AddSat)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    spread(reduced)
}

/// The next candidate of a count search: `candidate + NEXT` when at least the top-k entries of
/// `keys` reach `candidate`, else `candidate - STEP + NEXT`.
fn count_bit<const STEP: i32, const NEXT: i32>(
    device: &mut Device,
    (keys, top_k): (&Values<i32>, &Operand<i32>),
    candidate: &Operand<i32>,
) -> Operand<i32> {
    let excess = excess(device, keys, (candidate, top_k));
    device
        .main
        .begin(excess.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::ArithRightShift, 31i32)
        .vector_fxp(FxpBinaryOp::MulInt, STEP)
        .vector_fxp(FxpBinaryOp::AddFxp, NEXT)
        .vector_clip(ClipBinaryOpI32::AddFxp, candidate)
        .vector_final()
        .to_vrf::<m![1 # 8]>(&mut device.sub)
}

/// The largest value from `start` on (16 bits) that at least the top-k entries of `keys` reach.
/// The candidates stay in the vector registers from bit to bit.
fn count16(
    device: &mut Device,
    keys: &Values<i32>,
    (top_k, start): (&Operand<i32>, i32),
) -> Row<i32> {
    let mut first = Row::<i32>::new();
    first.view_mut().memset(start + 0x8000, &mut device.sub);
    let c = operand(device, &first);
    let c = count_bit::<0x8000, 0x4000>(device, (keys, top_k), &c);
    let c = count_bit::<0x4000, 0x2000>(device, (keys, top_k), &c);
    let c = count_bit::<0x2000, 0x1000>(device, (keys, top_k), &c);
    let c = count_bit::<0x1000, 0x0800>(device, (keys, top_k), &c);
    let c = count_bit::<0x0800, 0x0400>(device, (keys, top_k), &c);
    let c = count_bit::<0x0400, 0x0200>(device, (keys, top_k), &c);
    let c = count_bit::<0x0200, 0x0100>(device, (keys, top_k), &c);
    let c = count_bit::<0x0100, 0x0080>(device, (keys, top_k), &c);
    let c = count_bit::<0x0080, 0x0040>(device, (keys, top_k), &c);
    let c = count_bit::<0x0040, 0x0020>(device, (keys, top_k), &c);
    let c = count_bit::<0x0020, 0x0010>(device, (keys, top_k), &c);
    let c = count_bit::<0x0010, 0x0008>(device, (keys, top_k), &c);
    let c = count_bit::<0x0008, 0x0004>(device, (keys, top_k), &c);
    let c = count_bit::<0x0004, 0x0002>(device, (keys, top_k), &c);
    let c = count_bit::<0x0002, 0x0001>(device, (keys, top_k), &c);
    let excess = excess(device, keys, (&c, top_k));
    device
        .main
        .begin(excess.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::ArithRightShift, 31i32)
        .vector_clip(ClipBinaryOpI32::AddFxp, &c)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

/// The next threshold of the top-p level: `threshold + 2^SHIFT` when the weight of the keys at or
/// above it reaches the target (`bound` = its f32 bits minus one).
fn level_bit<const SHIFT: i32, const STEP: i32>(
    device: &mut Device,
    (keys, weights, bound): (&Values<i32>, &Values<i32>, &Operand<i32>),
    threshold: &Operand<i32>,
) -> Operand<i32> {
    let kept = select::<SHIFT>(device, (keys, weights), threshold);
    let reached = mass(device, &kept);
    device
        .main
        .begin(reached.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_fxp_with_mode(FxpBinaryOp::SubFxpSat, BinaryArgMode::Mode10, bound)
        .vector_fxp(FxpBinaryOp::ArithRightShift, 31i32)
        .vector_fxp(FxpBinaryOp::MulInt, -STEP)
        .vector_clip(ClipBinaryOpI32::AddFxp, threshold)
        .vector_final()
        .to_vrf::<m![1 # 8]>(&mut device.sub)
}

/// The top-p level's threshold after its four high bits, from `cutoff`.
fn level_high(
    device: &mut Device,
    (keys, weights): (&Values<i32>, &Values<i32>),
    (cutoff, bound): (&Row<i32>, &Operand<i32>),
) -> Operand<i32> {
    let t = operand(device, cutoff);
    let t = level_bit::<15, 0x8000>(device, (keys, weights, bound), &t);
    let t = level_bit::<14, 0x4000>(device, (keys, weights, bound), &t);
    let t = level_bit::<13, 0x2000>(device, (keys, weights, bound), &t);
    level_bit::<12, 0x1000>(device, (keys, weights, bound), &t)
}

/// The top-p level's threshold after its next two bits, from the threshold after its high bits.
fn level_mid(
    device: &mut Device,
    (keys, weights): (&Values<i32>, &Values<i32>),
    (high, bound): (&Operand<i32>, &Operand<i32>),
) -> Operand<i32> {
    let t = level_bit::<11, 0x800>(device, (keys, weights, bound), high);
    level_bit::<10, 0x400>(device, (keys, weights, bound), &t)
}

/// The top-p boundary level from the threshold after its six high bits: the largest key from the
/// top-k cutoff on whose top-k weight at or above it reaches the target; the cutoff when none does
/// above it (a +inf target).
fn level(
    device: &mut Device,
    (keys, weights): (&Values<i32>, &Values<i32>),
    (mid, bound): (&Operand<i32>, &Operand<i32>),
) -> Row<i32> {
    let t = level_bit::<9, 0x200>(device, (keys, weights, bound), mid);
    let t = level_bit::<8, 0x100>(device, (keys, weights, bound), &t);
    let t = level_bit::<7, 0x80>(device, (keys, weights, bound), &t);
    let t = level_bit::<6, 0x40>(device, (keys, weights, bound), &t);
    let t = level_bit::<5, 0x20>(device, (keys, weights, bound), &t);
    let t = level_bit::<4, 0x10>(device, (keys, weights, bound), &t);
    let t = level_bit::<3, 0x8>(device, (keys, weights, bound), &t);
    let t = level_bit::<2, 0x4>(device, (keys, weights, bound), &t);
    let t = level_bit::<1, 0x2>(device, (keys, weights, bound), &t);
    let kept = select::<0>(device, (keys, weights), &t);
    let reached = mass(device, &kept);
    advance::<1>(device, &reached, (bound, &t))
}

/// The boundary level's weight `exp((value(level) - peak) * inverse)`: the level's key decoded
/// back to its BF16 value (a key at or below -inf's gives weight 0).
fn level_weight(
    device: &mut Device,
    level: &Row<i32>,
    (peak, inverse): (&Operand<f32>, &Operand<f32>),
) -> Row<f32> {
    // u = key, or key + 65535 for a negative key (the BF16 bits xor 0x7fff).
    let low: Row<i32> = device
        .main
        .begin(level.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_stash()
        .vector_fxp(FxpBinaryOp::ArithRightShift, 31i32)
        .vector_fxp(FxpBinaryOp::MulInt, -65535i32)
        .vector_clip(ClipBinaryOpI32::AddFxp, Stash)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    device
        .main
        .begin(low.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_stash()
        .vector_logic(LogicBinaryOpI32::LeftShift, 16i32)
        .vector_logic(LogicBinaryOpI32::ArithRightShift, 31i32)
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7fffi32)
        .vector_logic(LogicBinaryOpI32::BitXor, Stash)
        .vector_fxp(FxpBinaryOp::LeftShift, 16i32)
        .vector_reinterpret::<f32>()
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, peak)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), inverse)
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

/// The number of the boundary level's ties kept: `ceil((target - above) / weight)`, as an i32
/// (saturating; a +inf target keeps them all).
fn kept_ties(
    device: &mut Device,
    above: &Row<f32>,
    (target, weight): (&Operand<f32>, &Operand<f32>),
) -> Row<i32> {
    let quotient: Row<f32> = device
        .main
        .begin(above.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary_with_mode(FpBinaryOp::SubF, BinaryArgMode::Mode10, target)
        .vector_fp_binary(FpBinaryOp::DivF, weight)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rounded: Row<i32> = device
        .main
        .begin(quotient.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fp_to_fxp(31)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let quotient = operand(device, &quotient);
    // `rounded + 1` where the quotient is above it: the ceiling of a positive quotient.
    device
        .main
        .begin(rounded.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_stash()
        .vector_fxp_to_fp(31)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary_with_mode(FpBinaryOp::SubF, BinaryArgMode::Mode10, &quotient)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_reinterpret::<i32>()
        .vector_clip(ClipBinaryOpI32::Min, 1i32)
        .vector_clip(ClipBinaryOpI32::Max, 0i32)
        .vector_clip(ClipBinaryOpI32::AddFxpSat, Stash)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

/// `value * factor` in every row.
fn product(device: &mut Device, value: &Row<f32>, factor: &Operand<f32>) -> Row<f32> {
    device
        .main
        .begin(value.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), factor)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

/// Each row's weight of the entries whose key is at least `floor + 1`.
fn weight_above(
    device: &mut Device,
    (keys, weights): (&Values<i32>, &Values<i32>),
    floor: &Row<i32>,
) -> Row<f32> {
    let floor = operand(device, floor);
    let selected = select::<0>(device, (keys, weights), &floor);
    mass(device, &selected)
}

/// The f32 bits of `value * factor`, minus `OFFSET`, as an operand.
fn bound<const OFFSET: i32>(
    device: &mut Device,
    value: &Row<f32>,
    factor: &Operand<f32>,
) -> Operand<i32> {
    let bound = target::<OFFSET>(device, value, factor);
    operand(device, &bound)
}

/// Each entry's tie with `level`: 1 where its key is `level`, 0 elsewhere.
fn ties(
    device: &mut Device,
    keys: &Values<i32>,
    level: &Operand<i32>,
) -> DmTensor<i8, Chips, Cluster, Slices, m![Entry]> {
    // `x = key ^ level` saturated to 1, 0 or -2 by its sign; `1 - x * x` is positive at 0 alone.
    device
        .main
        .begin(keys.view())
        .fetch::<m![Entry / 8], m![Entry % 8]>()
        .collect::<m![Entry / 8], m![Entry % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpI32::BitXor, level)
        .vector_fxp(FxpBinaryOp::LeftShiftSat, 31i32)
        .vector_fxp(FxpBinaryOp::ArithRightShift, 30i32)
        .vector_stash()
        .vector_fxp(FxpBinaryOp::MulInt, Stash)
        .vector_fxp_with_mode(FxpBinaryOp::SubFxpSat, BinaryArgMode::Mode10, 1i32)
        .vector_clip(ClipBinaryOpI32::Max, 0i32)
        .vector_final()
        .cast::<i8, m![Entry % 8 # 32]>()
        .commit_trim::<m![Entry % 8]>()
        .commit()
}

/// Each entry's keep code: at least 0 where its key is above `level`, or is `level` with its token
/// at most `last`, negative elsewhere.
fn codes(
    device: &mut Device,
    (keys, tokens): (&Values<i32>, &Values<i32>),
    (level, last): (&Operand<i32>, &Operand<i32>),
) -> Values<i32> {
    // Tokens: -1 above `last`, else 0; keys: `(key - level) << 20`, at least 2^20 above the level.
    let after: Values<i32> = device
        .main
        .begin(tokens.view())
        .fetch::<m![Entry / 8], m![Entry % 8]>()
        .collect::<m![Entry / 8], m![Entry % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp_with_mode(FxpBinaryOp::SubFxpSat, BinaryArgMode::Mode10, last)
        .vector_fxp(FxpBinaryOp::ArithRightShift, 31i32)
        .vector_final()
        .commit_trim::<m![Entry % 8]>()
        .commit();
    device
        .main
        .begin_interleaved::<Term, _, _, _, _, _>(keys.view(), after.view())
        .fetch::<m![Entry / 128, Term, Entry % 128 / 8], m![Entry % 8]>()
        .collect::<m![Entry / 128, Term, Entry % 128 / 8], m![Entry % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Term, m![Entry / 8]>()
        .vector_fxp(FxpBinaryOp::SubFxpSat, level, ())
        .vector_fxp(FxpBinaryOp::LeftShiftSat, 20i32, ())
        .vector_clip_zip(ClipBinaryOpI32::AddFxpSat)
        .vector_final()
        .commit_trim::<m![Entry % 8]>()
        .commit()
}

/// `weights` (f32 bits) where `codes` is at least 0, else 0.
fn kept(device: &mut Device, (codes, weights): (&Values<i32>, &Values<i32>)) -> Values<i32> {
    device
        .main
        .begin_interleaved::<Term, _, _, _, _, _>(codes.view(), weights.view())
        .fetch::<m![Entry / 128, Term, Entry % 128 / 8], m![Entry % 8]>()
        .collect::<m![Entry / 128, Term, Entry % 128 / 8], m![Entry % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Term, m![Entry / 8]>()
        .vector_fxp(FxpBinaryOp::ArithRightShift, 31i32, ())
        .vector_fxp(FxpBinaryOp::AddFxp, 1i32, ())
        .vector_fxp_zip(FxpBinaryOp::MulInt)
        .vector_final()
        .commit_trim::<m![Entry % 8]>()
        .commit()
}

/// The position of each row's `need`-th tie in token order (131072 with fewer ties): its count of
/// entries whose running tie count is below `need`.
fn rank(
    device: &mut Device,
    counts: &DmTensor<i8, Chips, Cluster, Slices, m![Entry]>,
    (order, need): (&Order, &Operand<i32>),
) -> Row<i32> {
    let runs = lanewise(device, counts);
    let totals = run_totals(device, &runs);
    let offsets = offsets_i32(device, &totals, order);
    // `need` minus each run's offset, so the runs' own running counts compare against it.
    let pivots: Lanes<i32> = device
        .main
        .begin(offsets.view())
        .fetch::<m![1], m![Run]>()
        .collect::<m![1], m![Run]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp_with_mode(FxpBinaryOp::SubFxpSat, BinaryArgMode::Mode10, need)
        .vector_final()
        .commit_trim::<m![Run]>()
        .commit();
    let pivots = lane_operand(device, &pivots);
    let running = running_lanes(device, &runs, order);
    // Each run's count of entries whose running count is below its pivot, then the row's.
    let counts: Lanes<i32> = device
        .main
        .begin(running.view())
        .fetch::<m![Pos], m![Run]>()
        .collect::<m![Pos], m![Run]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::SubFxpSat, &pivots)
        .vector_fxp(FxpBinaryOp::LogicRightShift, 31i32)
        .vector_narrow_split::<m![Pos, Run / 4], m![Run % 4]>()
        .vector_intra_slice_reduce::<Pos, m![Run / 4], m![Run % 4]>(IntraSliceReduceOpI32::AddSat)
        .vector_widen_concat::<m![1], m![Run]>()
        .vector_final()
        .commit_trim::<m![Run]>()
        .commit();
    let reduced: Reduced<i32, m![1 # 8]> = device
        .main
        .begin(counts.view())
        .fetch::<m![1], m![Run]>()
        .collect::<m![1], m![Run]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Run / 4], m![Run % 4]>()
        .vector_intra_slice_reduce::<Run, m![1], m![1 # 4]>(IntraSliceReduceOpI32::AddSat)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<m![Batch % 4, Wide], m![1]>(InterSliceReduceOpI32::AddSat)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    spread(reduced)
}

/// A per-entry tensor with one run a lane: position, then run.
type Lanewise<D> = DmTensor<D, Chips, Cluster, Slices, m![Pos, Run]>;

/// The tags of each run's first and last position, one run a lane.
type LaneLog = VrfTensor<i4, Chips, Cluster, Slices, m![Pos, Run]>;

/// `values` with run `j` in lane `j`: eight 8-bit runs a transpose.
fn lanewise(
    device: &mut Device,
    values: &DmTensor<i8, Chips, Cluster, Slices, m![Entry]>,
) -> Lanewise<i8> {
    // SAFETY: `Entry` is run then position; only the axes are renamed.
    let runs = unsafe {
        values
            .view()
            .reshape::<Chips, Cluster, Slices, m![Run, Pos]>()
    };
    device
        .main
        .begin(runs)
        .fetch::<m![Pos / 8, Run], m![Pos % 8]>()
        .collect::<m![Pos / 8, Run], m![Pos % 8 # 32]>()
        .transpose::<m![Pos / 8, Pos % 8], m![Run # 32]>()
        .commit_trim::<m![Run]>()
        .commit()
}

/// Each run's sum of the 8-bit `runs` in this slice.
fn run_totals(device: &mut Device, runs: &Lanewise<i8>) -> Lanes<i32> {
    device
        .main
        .begin(runs.view())
        .fetch::<m![Pos], m![Run]>()
        .fetch_cast::<i32>()
        .collect::<m![Pos], m![Run]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Pos, Run / 4], m![Run % 4]>()
        .vector_intra_slice_reduce::<Pos, m![Run / 4], m![Run % 4]>(IntraSliceReduceOpI32::AddSat)
        .vector_widen_concat::<m![1], m![Run]>()
        .vector_final()
        .commit_trim::<m![Run]>()
        .commit()
}

/// Each run's running sum of the 8-bit `runs` in position order, one run a lane.
fn running_lanes(device: &mut Device, runs: &Lanewise<i8>, order: &Order) -> Lanewise<i32> {
    device
        .main
        .begin(runs.view())
        .fetch::<m![Pos], m![Run]>()
        .fetch_cast::<i32>()
        .collect::<m![Pos], m![Run]>()
        .vector_init()
        .vector_tagged(&order.lanes)
        .vector_cumsum(
            ClipBinaryOpI32::AddFxpSat,
            TagGuard::not_matches(FIRST),
            TagGuard::not_matches(LAST),
        )
        .vector_final()
        .commit_trim::<m![Run]>()
        .commit()
}

/// Each row's token: the count of entries whose running kept weight in token order is at most the
/// uniform's share of the row's kept weight (f32 bits of non-negative sums order as integers).
fn token(
    device: &mut Device,
    kept: &Values<i32>,
    (order, uniform): (&Order, &Operand<f32>),
) -> Row<i32> {
    let mut pairs = paired(device, kept);
    let totals = totals_f32(device, &pairs);
    let offsets = offsets_f32(device, &totals, order);
    let offsets = bits(device, &offsets);
    inject_f32(device, &mut pairs, &offsets);
    let running = running_f32(device, &pairs, order);
    // SAFETY: as in `rank`.
    let running = unsafe { running.reshape::<Chips, Cluster, Slices, m![Entry]>() };
    let total = maximum_view(device, running.view());
    let cap = {
        let last = operand(device, &total);
        // One past the target's bits, at most the total's.
        let capped: Row<i32> = device
            .main
            .begin(total.view())
            .fetch::<m![1], m![1 # 8]>()
            .collect::<m![1], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_reinterpret::<f32>()
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), uniform)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_reinterpret::<i32>()
            .vector_clip(ClipBinaryOpI32::AddFxpSat, 1i32)
            .vector_clip(ClipBinaryOpI32::Min, &last)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        operand(device, &capped)
    };
    below(device, &running, &cap)
}

/// Each row's maximum of the i32 `values`, from a per-entry view.
fn maximum_view<E: M>(
    device: &mut Device,
    values: DmTensorView<'_, i32, Chips, Cluster, Slices, E>,
) -> Row<i32> {
    let reduced: Reduced<i32, m![1 # 8]> = device
        .main
        .begin(values)
        .fetch::<m![Entry / 8], m![Entry % 8]>()
        .collect::<m![Entry / 8], m![Entry % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Entry / 4], m![Entry % 4]>()
        .vector_intra_slice_reduce::<Entry, m![1], m![1 # 4]>(IntraSliceReduceOpI32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<m![Batch % 4, Wide], m![1]>(InterSliceReduceOpI32::Max)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    spread(reduced)
}

/// Each run's count before it in token order: the same run of the places below, then the whole
/// runs below.
fn offsets_i32(device: &mut Device, totals: &Lanes<i32>, order: &Order) -> Lanes<i32> {
    let own = lane_operand(device, totals);
    // Every place's run totals, in every slice of the PE.
    let gathered: Reduced<i32, m![Gathered, Run]> = device
        .main
        .begin(order.onehot.view())
        .fetch::<m![Gathered], m![Run]>()
        .collect::<m![Gathered], m![Run]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpI32::BitAnd, &own)
        .vector_inter_slice_reduce::<m![Batch % 4, Wide], m![Gathered]>(
            InterSliceReduceOpI32::AddSat,
        )
        .vector_final()
        .commit_trim::<m![Run]>()
        .commit();
    let gathered = spread(gathered);
    let before = gather_operand(device, &order.before);
    let within: Lanes<i32> = device
        .main
        .begin(gathered.view())
        .fetch::<m![Gathered], m![Run]>()
        .collect::<m![Gathered], m![Run]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpI32::BitAnd, &before)
        .vector_narrow_split::<m![Gathered, Run / 4], m![Run % 4]>()
        .vector_intra_slice_reduce::<Gathered, m![Run / 4], m![Run % 4]>(
            IntraSliceReduceOpI32::AddSat,
        )
        .vector_widen_concat::<m![1], m![Run]>()
        .vector_final()
        .commit_trim::<m![Run]>()
        .commit();
    // The row's run totals, then each place 0..8's count of the runs below it.
    let whole: Reduced<i32, m![Run]> = device
        .main
        .begin(totals.view())
        .fetch::<m![1], m![Run]>()
        .collect::<m![1], m![Run]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![Batch % 4, Wide], m![1]>(InterSliceReduceOpI32::AddSat)
        .vector_final()
        .commit_trim::<m![Run]>()
        .commit();
    let whole = spread(whole);
    let below = lane_operand(device, &order.below);
    let lower: Row<i32> = device
        .main
        .begin(whole.view())
        .fetch::<m![1], m![Run]>()
        .collect::<m![1], m![Run]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpI32::BitAnd, &below)
        .vector_narrow_split::<m![Run / 4], m![Run % 4]>()
        .vector_intra_slice_reduce::<Run, m![1], m![1 # 4]>(IntraSliceReduceOpI32::AddSat)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let lower = operand(device, &lower);
    let runs: Reduced<i32, m![Run]> = device
        .main
        .begin(order.own.view())
        .fetch::<m![1], m![Run]>()
        .collect::<m![1], m![Run]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpI32::BitAnd, &lower)
        .vector_inter_slice_reduce::<m![Batch % 4, Wide], m![1]>(InterSliceReduceOpI32::AddSat)
        .vector_final()
        .commit_trim::<m![Run]>()
        .commit();
    let runs = spread(runs);
    let runs = lane_operand(device, &runs);
    device
        .main
        .begin(within.view())
        .fetch::<m![1], m![Run]>()
        .collect::<m![1], m![Run]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpI32::AddFxpSat, &runs)
        .vector_final()
        .commit_trim::<m![Run]>()
        .commit()
}

/// Each run's weight before it in token order, from each slice's run totals (f32 bits).
fn offsets_f32(device: &mut Device, totals: &Lanes<i32>, order: &Order) -> Lanes<f32> {
    let own = lane_operand(device, totals);
    let gathered: Reduced<f32, m![Gathered, Run]> = device
        .main
        .begin(order.onehot.view())
        .fetch::<m![Gathered], m![Run]>()
        .collect::<m![Gathered], m![Run]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpI32::BitAnd, &own)
        .vector_reinterpret::<f32>()
        .vector_inter_slice_reduce::<m![Batch % 4, Wide], m![Gathered]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![Run]>()
        .commit();
    let gathered = spread(gathered);
    let before = gather_operand(device, &order.before);
    let within: Lanes<f32> = device
        .main
        .begin(gathered.view())
        .fetch::<m![Gathered], m![Run]>()
        .collect::<m![Gathered], m![Run]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, &before)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![Gathered, Run / 4], m![Run % 4]>()
        .vector_intra_slice_reduce::<Gathered, m![Run / 4], m![Run % 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_concat::<m![1], m![Run]>()
        .vector_final()
        .commit_trim::<m![Run]>()
        .commit();
    let whole: Reduced<f32, m![Run]> = device
        .main
        .begin(totals.view())
        .fetch::<m![1], m![Run]>()
        .collect::<m![1], m![Run]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpI32::BitOr, 0i32)
        .vector_reinterpret::<f32>()
        .vector_inter_slice_reduce::<m![Batch % 4, Wide], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![Run]>()
        .commit();
    let whole = spread(whole);
    let below = lane_operand(device, &order.below);
    let lower: Row<i32> = device
        .main
        .begin(whole.view())
        .fetch::<m![1], m![Run]>()
        .collect::<m![1], m![Run]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, &below)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![Run / 4], m![Run % 4]>()
        .vector_intra_slice_reduce::<Run, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_reinterpret::<i32>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let lower = operand(device, &lower);
    let runs: Reduced<f32, m![Run]> = device
        .main
        .begin(order.own.view())
        .fetch::<m![1], m![Run]>()
        .collect::<m![1], m![Run]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpI32::BitAnd, &lower)
        .vector_reinterpret::<f32>()
        .vector_inter_slice_reduce::<m![Batch % 4, Wide], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![Run]>()
        .commit();
    let runs = spread(runs);
    let runs = lane_operand(device, &runs);
    device
        .main
        .begin(within.view())
        .fetch::<m![1], m![Run]>()
        .collect::<m![1], m![Run]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &runs)
        .vector_final()
        .commit_trim::<m![Run]>()
        .commit()
}

/// Each run's first entry of a paired tensor, `[Quad, Two]`.
fn starts_of<D: VeScalar>(
    pairs: &Paired<D>,
) -> DmTensorView<'_, D, Chips, Cluster, Slices, m![Quad, Pos = 1 # 256, Two]> {
    pairs
        .view()
        .tile::<m![Pos], 1, m![Quad, Pos = 1 # 256, Two]>(0)
}

fn starts_of_mut<D: VeScalar>(
    pairs: &mut Paired<D>,
) -> DmTensorViewMut<'_, D, Chips, Cluster, Slices, m![Quad, Pos = 1 #{!} 256, Two]> {
    pairs
        .view_mut()
        .tile::<m![Pos], 1, m![Quad, Pos = 1 #{!} 256, Two]>(0)
}

/// One value a run, as `[Quad, Two]`.
type Quads<D> = DmTensor<D, Chips, Cluster, Slices, m![Quad, Two]>;

fn quads<D: VeScalar>(
    lanes: &Lanes<D>,
) -> DmTensorView<'_, D, Chips, Cluster, Slices, m![Quad, Two]> {
    // SAFETY: run `2 Quad + Two`, renamed.
    unsafe {
        lanes
            .view()
            .reshape::<Chips, Cluster, Slices, m![Quad, Two]>()
    }
}

/// The f32 `values` as their bits.
fn bits(device: &mut Device, values: &Lanes<f32>) -> Lanes<i32> {
    device
        .main
        .begin(values.view())
        .fetch::<m![1], m![Run]>()
        .collect::<m![1], m![Run]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_final()
        .commit_trim::<m![Run]>()
        .commit()
}

/// The tags of each run's first and last paired flit, on every lane.
type Log = VrfTensor<i4, Chips, Cluster, Slices, m![Quad, Pos, Two # 8]>;

fn inject_f32(device: &mut Device, pairs: &mut Paired<i32>, starts: &Lanes<i32>) {
    let mut firsts = Quads::<i32>::new();
    starts_of(pairs).to_dm_view(&mut device.tdma, firsts.view_mut());
    let sum: Quads<i32> = device
        .main
        .begin_interleaved::<Term, _, _, _, _, _>(firsts.view(), quads(starts))
        .fetch::<m![Quad, Term], m![Two # 8]>()
        .collect::<m![Quad, Term], m![Two # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Term, m![Quad]>()
        .vector_reinterpret::<f32>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_reinterpret::<i32>()
        .vector_final()
        .commit_trim::<m![Two]>()
        .commit();
    sum.view()
        .to_dm_view(&mut device.tdma, starts_of_mut(pairs));
}

/// Each run's sum of `pairs` (f32 bits) in this slice, as f32 bits.
fn totals_f32(device: &mut Device, pairs: &Paired<i32>) -> Lanes<i32> {
    let sums: Quads<i32> = device
        .main
        .begin(pairs.view())
        .fetch::<m![Quad, Pos], m![Two # 8]>()
        .collect::<m![Quad, Pos], m![Two # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<f32>()
        .vector_narrow_trim::<m![Two # 4]>()
        .vector_intra_slice_reduce::<Pos, m![Quad], m![Two # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![Two # 8]>()
        .vector_reinterpret::<i32>()
        .vector_final()
        .commit_trim::<m![Two]>()
        .commit();
    // SAFETY: as `quads`.
    unsafe { sums.reshape::<Chips, Cluster, Slices, m![Run]>() }
}

/// Each run's running sum of `values` (f32 bits) in position order, from its first entry.
pub fn running_f32(device: &mut Device, values: &Paired<i32>, order: &Order) -> Paired<i32> {
    let running = device
        .main
        .begin(values.view())
        .fetch::<m![Quad, Pos], m![Two # 8]>()
        .collect::<m![Quad, Pos], m![Two # 8]>()
        .vector_init()
        .vector_tagged(&order.log)
        .vector_reinterpret::<f32>()
        .vector_cumsum(
            ClipBinaryOpF32::Add,
            TagGuard::not_matches(FIRST),
            TagGuard::not_matches(LAST),
        )
        .vector_reinterpret::<i32>()
        .vector_final()
        .commit_trim::<m![Two]>()
        .commit();
    running
}

/// A unit's searches done: what the rest of its draw reads.
pub struct Searched {
    keys: Values<i32>,
    weights: Values<i32>,
    level: Row<i32>,
    need: Row<i32>,
    params: Params,
}

/// The searches of the unit landed in `ahead`, with the loads of the `next` unit's rows (when there is
/// one) issued as they start, then the token `order`'s ids (the first unit's), and the next unit's
/// exchange after four bits of the top-p search, so its link transfer ends before the prefixes queue
/// theirs: the next unit's peer half, for `Ahead::receive`.
pub fn search<E: M, F: M>(
    device: &mut Device,
    (ahead, stops): (&mut Ahead, &Stops),
    (next, order): (Option<Rows<'_, E, F>>, Option<(&mut Landed, &Ids)>),
) -> (Searched, Option<Peer>) {
    let draw = ahead.draw(device);
    let landing = next.is_some();
    // The order's ids queue behind the next unit's loads, so the first unit exchanges the next one
    // after its whole level search: the transfer queue is in order, and the exchange's copies would
    // otherwise wait for the ids under the level bits.
    let first = order.is_some();
    let early = if first { false } else { landing };
    let late = if first { landing } else { false };
    if let Some(next) = next {
        ahead.land(device, next);
    }
    if let Some((landed, ids)) = order {
        landed.land(device, ids);
    }
    let keys = keys(device, &ahead.logits);
    // Top-k.
    let cutoff = count16(device, &keys, (&draw.top_k, LOWEST));
    // Weights from the row's maximum.
    let peak = maximum(device, &ahead.logits);
    let weights = {
        let peak = operand(device, &peak);
        let inverse = param::<INVERSE>(device, &draw.params);
        weights(device, &ahead.logits, (&peak, &inverse))
    };
    // The top-k weight: entries at or above the cutoff.
    let floor = add::<-1>(device, &cutoff);
    let mass_k = weight_above(device, (&keys, &weights), &floor);
    let bound = {
        let top_p = param::<TOP_P>(device, &draw.params);
        bound::<1>(device, &mass_k, &top_p)
    };
    // Top-p: the boundary level, its first ties in token order kept.
    let high = level_high(device, (&keys, &weights), (&cutoff, &bound));
    if early {
        ahead.exchange(device, stops);
    }
    let peer = if early {
        Some(ahead.send(device))
    } else {
        None
    };
    let mid = level_mid(device, (&keys, &weights), (&high, &bound));
    let level = level(device, (&keys, &weights), (&mid, &bound));
    if late {
        ahead.exchange(device, stops);
    }
    let peer = if late { Some(ahead.send(device)) } else { peer };
    let above = weight_above(device, (&keys, &weights), &level);
    let weight = {
        let peak = operand(device, &peak);
        let inverse = param::<INVERSE>(device, &draw.params);
        level_weight(device, &level, (&peak, &inverse))
    };
    let goal = {
        let top_p = param::<TOP_P>(device, &draw.params);
        product(device, &mass_k, &top_p)
    };
    let need = {
        let goal = operand(device, &goal);
        let weight = operand(device, &weight);
        kept_ties(device, &above, (&goal, &weight))
    };
    (
        Searched {
            keys,
            weights,
            level,
            need,
            params: draw.params,
        },
        peer,
    )
}

/// Each row's sampled token id of a searched unit.
pub fn finish(device: &mut Device, searched: Searched, order: &Order) -> Row<i32> {
    let Searched {
        keys,
        weights,
        level,
        need,
        params,
    } = searched;
    let level = operand(device, &level);
    let last = {
        let ties = ties(device, &keys, &level);
        let need = operand(device, &need);
        rank(device, &ties, (order, &need))
    };
    let kept = {
        let last = operand(device, &last);
        let codes = codes(device, (&keys, &order.tokens), (&level, &last));
        kept(device, (&codes, &weights))
    };
    let uniform = param::<UNIFORM>(device, &params);
    token(device, &kept, (order, &uniform))
}

/// This chip's rows of a value on each PE's first slice.
type Own = DmTensor<i32, Chips, Cluster, m![Batch % 4, 1 # 64], m![1 # 4]>;
/// Both chips' rows of a value on each PE's first slice, a block of rows a chip.
type Blocks = DmTensor<i32, Chips, Cluster, m![Batch % 4, 1 # 64], m![Batch / 8, 1 # 4]>;
type BlockSlot = m![Batch / 8 = 1 #{!} 2, 1 # 4];

/// `value` of this chip's rows into the unit's HBM `rows` on both chips.
pub fn store<E: M>(
    device: &mut Device,
    value: &Row<i32>,
    rows: HbmTensorViewMut<'_, i32, Chips, E>,
) {
    let own: Own = device
        .main
        .begin(value.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![Batch % 4, 1 # 64], m![1]>(InterSliceReduceOpI32::Max)
        .vector_final()
        .commit_trim::<m![1 # 4]>()
        .commit();
    // This chip's rows in block 0, the peer chip's in block 1: chip c's block b holds row block
    // b xor c.
    let mut gathered = Blocks::new();
    own.view().to_dm_view(
        &mut device.tdma,
        gathered.view_mut().tile::<Block, 1, BlockSlot>(0),
    );
    own.view().chip_shuffle([1, 0]).to_dm_view(
        &mut device.tdma,
        gathered.view_mut().tile::<Block, 1, BlockSlot>(1),
    );
    let mut blocks = Blocks::new();
    gathered
        .view()
        .chip_slice::<Block, m![1 # 4]>([0, 1])
        .to_dm_view(
            &mut device.tdma,
            blocks.view_mut().tile::<Block, 1, BlockSlot>(0),
        );
    gathered
        .view()
        .chip_slice::<Block, m![1 # 4]>([1, 0])
        .to_dm_view(
            &mut device.tdma,
            blocks.view_mut().tile::<Block, 1, BlockSlot>(1),
        );
    blocks.view().to_hbm_view(&mut device.tdma, rows);
}

/// A per-entry tensor with two runs a flit, `Entry / 512` then the position: run `2 Quad + Two`.
pub type Paired<D> = DmTensor<D, Chips, Cluster, Slices, m![Quad, Pos, Two]>;

/// `values` with runs `2 q` and `2 q + 1` side by side in a flit.
pub fn paired<D: VeScalar + MaterializableScalar>(
    device: &mut Device,
    values: &Values<D>,
) -> Paired<D> {
    // SAFETY: `Entry / 256 = 2 Quad + Two`; only the axes are renamed.
    let quads = unsafe {
        values
            .view()
            .reshape::<Chips, Cluster, Slices, m![Quad, Two, Pos]>()
    };
    device
        .main
        .begin(quads)
        .fetch::<m![Quad, Pos / 8, Two], m![Pos % 8]>()
        .collect::<m![Quad, Pos / 8, Two], m![Pos % 8]>()
        .transpose::<m![Quad, Pos / 8, Pos % 8], m![Two # 8]>()
        .commit_trim::<m![Two]>()
        .commit()
}
