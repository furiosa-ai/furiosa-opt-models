//! A prompt chunk's layer after attention with the layer D-sharded: chip `c` owns hidden half
//! `c` (`Hidden / 2560`) of every row between the O projection and the down's reduce-scatter.
//! Each wave's own heads are rounded and quantized as TP2's O input, the peer's heads arrive over
//! the link into the second half of the every-head rows, the O projects the chip's own half
//! (`output::dshard`), and the wave's output is moved into its 256-row block in the D-sharded
//! layout the feed-forward's panels write (`mlp::dpanel::Own`), where the residual adds.
use super::*;
use crate::compose::{Tile, Tiles};
use crate::models::nemo::serving::mlp::{
    block::{Block, Blocks},
    dpanel,
};
use crate::models::nemo::serving::output::{dshard, oblock};
use crate::ops::{layout::Dm, normalization::RmsNorm};

/// Wave `wave`'s own heads (of a chunk starting at wave `start`), rounded to BF16 and quantized
/// by the O input scale; the peer's arrive in exchange. Returns (own, peer).
pub(super) fn heads(
    device: &mut Device,
    (prompt, (start, wave)): (&attention::prompt::Attended, (usize, usize)),
    divisor: &scale::InputScale<Chips, dshard::Cluster, dshard::Landing>,
) -> (dshard::Rows8, dshard::Rows8) {
    let attended = prompt.view().tile::<m![Prompt], 64, m![
        Prompt = 64 # Prompt,
        AttentionHeads,
        AttentionGroups,
        AttentionHead
    ]>(64 * (wave - start));
    // SAFETY: the 64-row tile keeps the chunk's row stride, and its row-major heads flatten
    // Heads/Groups/Head in wire order.
    let attended = unsafe { attended.reshape::<Chips, m![Rows # Prompt, Width]>() };
    let landed: dshard::Rows32 = attended.to_dm(&mut device.tdma);
    let own = dshard::quantize(device, &landed, divisor);
    let peer = own
        .chip_shuffle([1, 0])
        .to_dm::<dshard::Element>(&mut device.tdma);
    (own, peer)
}

/// A wave's own-half O output into rows `64 wave ..` of its block in the D-sharded layout.
pub(super) fn place(
    device: &mut Device,
    output: &dshard::Output,
    (block, wave): (&mut dpanel::Own, usize),
) {
    // SAFETY: a wave is rows `64 wave ..` of its block; only the row label changes.
    let rows = unsafe {
        output.view().reshape::<Chips, dpanel::Cluster, m![Hidden % 2560 / 1280, Hidden % 640 / 80, Block % 64 / 4], m![
            Block % 4,
            Hidden % 80 / 8,
            Hidden % 8
        ]>()
    };
    rows.to_dm_view(
        &mut device.tdma,
        block
            .view_mut()
            .slice_tile::<m![Block / 64], 1, m![Hidden % 640 / 320, Block / 64 = 1 #{!} 4, Block % 64 / 8, Hidden % 320 / 80]>(wave),
    );
}

/// A block's own-half O output plus its own half of the layer input `hidden`, rounded once to
/// BF16 as the step's residual is, into `output`.
pub(super) fn residual(
    device: &mut Device,
    (sum, hidden): (&dpanel::Own, &dpanel::Own),
    output: &mut dpanel::Own,
) {
    #[unroll]
    for m in 0..2 {
        dpanel::add(device, (sum, hidden), (output, m));
    }
}

type Cluster = dpanel::Cluster;
type Reduced = dpanel::Reduced;
type Time = dpanel::Time;
type Packet = dpanel::Packet;
/// A block's per-row f32 partial, replicated over a row slice's eight feature slices.
pub(super) type RowSums = DmTensor<f32, Chips, Cluster, Reduced, m![Block % 8, 1 # 8]>;
/// `RowSums` as the inter-slice reduction leaves it: the eight feature slices one partition.
type SumSlices = m![Hidden % 640 / 320, Block / 8, 4];
/// The post-attention norm's gain of this chip's half, in every row slice of a block.
pub(super) type Gain =
    DmTensor<bf16, Chips, Cluster, Reduced, m![Hidden % 2560 / 1280, Hidden % 80]>;
type GainBoth =
    DmTensor<bf16, Chips, Cluster, Reduced, m![Hidden / 2560, Hidden % 2560 / 1280, Hidden % 80]>;
pub(super) type PostNorm = RmsNorm<Hidden, { 1e-5_f32.to_bits() as usize }, Chips, Gain>;
type NormSpace = Tile<Block, Block>;
type Groups = m![Hidden % 2560 / 1280];
type Features = m![Hidden % 80];

pub(super) fn post_norm(device: &mut Device, layers: &stack::Layers, index: usize) -> PostNorm {
    land_norm(device, layers.post_norm.tensor.clone(), index)
}

pub(super) fn input_norm(device: &mut Device, layers: &stack::Layers, index: usize) -> PostNorm {
    land_norm(device, layers.norm.tensor.clone(), index)
}

fn land_norm(
    device: &mut Device,
    tensor: HbmTensorView<'_, bf16, Chips, m![Depth, Hidden]>,
    index: usize,
) -> PostNorm {
    let norm = RmsNorm::<Hidden, { 1e-5_f32.to_bits() as usize }, Chips, _>::new(
        tensor.tile::<m![Depth], 1, m![Depth = 1 # Depth, Hidden]>(index),
    );
    let both: GainBoth = norm.weight().clone().to_dm(&mut device.tdma);
    let norm = RmsNorm::<Hidden, { 1e-5_f32.to_bits() as usize }, Chips, _>::new(both);
    RmsNorm::new(
        norm.weight()
            .view()
            .chip_slice::<m![Hidden / 2560], m![Hidden % 2560 / 1280, Hidden % 80]>([0, 1])
            .to_dm(&mut device.tdma),
    )
}
/// A block's FP8 activation of this chip's half in the D-sharded layout: slot 0 this chip's,
/// slot 1 the peer's.
pub(crate) type Slots =
    DmTensor<f8e4m3, Chips, Cluster, Reduced, m![Slot, Hidden % 2560 / 1280, { Time }, { Packet }]>;
axes![Slot = 2, Copy640 = 2, Copy320 = 4];

/// The post-attention norm's gain (replicated) as this chip's half in every row slice.
pub(super) fn gain(
    device: &mut Device,
    weight: &HbmTensorView<'_, bf16, Chips, m![Depth = 1 # Depth, Hidden]>,
) -> Gain {
    let both: GainBoth = weight.clone().to_dm(&mut device.tdma);
    both.view()
        .chip_slice::<m![Hidden / 2560], m![Hidden % 2560 / 1280, Hidden % 80]>([0, 1])
        .to_dm(&mut device.tdma)
}

/// Each row's sum of squares over this cluster's 1,280 features, over 5,120, plus a 32nd of the
/// norm's epsilon (32 partitions of 160 features a row), replicated over the row slice.
pub(super) fn squares(device: &mut Device, residual: &dpanel::Own) -> RowSums {
    squares_with::<false>(device, residual)
}

/// `squares` with the inter-slice reduction (`INTER`) or each slice's own partial (debug).
pub(super) fn squares_with<const INTER: bool>(
    device: &mut Device,
    residual: &dpanel::Own,
) -> RowSums {
    let value = device
        .main
        .begin(residual.view())
        .fetch::<m![Block % 8, Hidden % 2560 / 1280, Hidden % 80 / 8], m![Hidden % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Block % 8, Hidden % 2560 / 1280, Hidden % 80 / 8], m![Hidden % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Block % 8, Hidden % 2560 / 1280, Hidden % 80 / 4], m![Hidden % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Hidden, m![Block % 8], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(5120.0_f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, 1e-5_f32 / 32.0);
    if INTER {
        let reduced: DmTensor<f32, Chips, Cluster, SumSlices, m![Block % 8, 1 # 8]> = value
            .vector_inter_slice_reduce::<SumSlices, m![Block % 8]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        // SAFETY: every feature slice of a row slice carries the same reduced row sums in wire order.
        unsafe { reduced.reshape() }
    } else {
        value.vector_final().commit_trim::<m![1 # 8]>().commit()
    }
}

/// `a + b` of two row sums.
fn sum(device: &mut Device, (a, b): (&RowSums, &RowSums)) -> RowSums {
    axes![Term = 2];
    device
        .main
        .begin_interleaved::<Term, _, _, _, _, _>(a.view(), b.view())
        .fetch::<m![Block % 8, Term], m![1 # 8]>()
        .collect::<m![Block % 8, Term], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Term, m![Block % 8]>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

/// `a + b` of two row sums into `out`.
fn sum_into(device: &mut Device, (a, b): (&RowSums, &RowSums), out: &mut RowSums) {
    axes![Term = 2];
    device
        .main
        .begin_interleaved::<Term, _, _, _, _, _>(a.view(), b.view())
        .fetch::<m![Block % 8, Term], m![1 # 8]>()
        .collect::<m![Block % 8, Term], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Term, m![Block % 8]>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit_view(out.view_mut())
}

/// A block's rows normalized over this chip's half (their mean squares all-reduced over the
/// clusters and the chips), times `gain`, quantized by the gate input scale into slot 0 of
/// `slots`; the peer's normalized half arrives into slot 1.
pub(super) fn normalize<const STAGE: usize>(
    device: &mut Device,
    residual: &dpanel::Own,
    (gain, divisor): (&Gain, &scale::InputScale<Chips, Cluster, Reduced>),
    slots: &mut Slots,
) {
    // The VRU's inter-slice reduction over the four inner hidden slices hangs the device
    // (`compiler/lowerfix/mre/vru-reduce-four`); the eight hidden slices of a row are summed by
    // three swap rounds instead.
    let partial = squares(device, residual);
    let local = if STAGE >= 2 {
        local_sums(device, &partial)
    } else {
        partial
    };
    let chip = if STAGE >= 3 {
        let other = local
            .cluster_swap()
            .to_dm::<m![Block % 8, 1 # 8]>(&mut device.tdma);
        sum(device, (&local, &other))
    } else {
        local
    };
    let total = if STAGE >= 4 {
        let peer = chip
            .chip_shuffle([1, 0])
            .to_dm::<m![Block % 8, 1 # 8]>(&mut device.tdma);
        sum(device, (&chip, &peer))
    } else {
        chip
    };
    let inverse: VrfTensor<f32, Chips, Cluster, Reduced, m![Block % 8, 1 # 8]> = device
        .main
        .begin(total.view())
        .fetch::<m![Block % 8], m![1 # 2]>()
        .collect::<m![Block % 8], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_fp_div_with_mode(BinaryArgMode::Mode10, 1.0_f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .to_vrf::<m![Block % 8, 1 # 8]>(&mut device.sub);
    let weight: VrfTensor<f32, Chips, Cluster, Reduced, m![Hidden % 2560 / 1280, Hidden % 80]> =
        device
            .sub
            .begin(gain.view())
            .fetch::<m![Hidden % 2560 / 1280, Hidden % 80 / 8], m![Hidden % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Hidden % 2560 / 1280, Hidden % 80 / 8], m![Hidden % 8]>()
            .to_vrf::<m![Hidden % 2560 / 1280, Hidden % 80]>();
    device
        .main
        .begin(residual.view())
        .fetch::<m![Hidden % 2560 / 1280, Block % 8, Hidden % 80 / 8], m![Hidden % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Hidden % 2560 / 1280, Block % 8, Hidden % 80 / 8], m![Hidden % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Hidden % 2560 / 1280, Block % 8, Hidden % 80 / 8, Hidden % 8 / 4], m![Hidden % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &inverse)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight)
        .vector_fp_div(divisor.input_scale())
        .vector_widen_concat::<m![Hidden % 2560 / 1280, Block % 8, Hidden % 80 / 8], m![Hidden % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Hidden % 8 # 32]>()
        .commit_trim::<m![Hidden % 8]>()
        .commit_view(
            slots
                .view_mut()
                .tile::<m![Slot], 1, m![Slot = 1 #{!} 2, Hidden % 2560 / 1280, { Time }, { Packet }]>(0),
        );
}

/// A block's FP8 activation of both halves, absolute: this chip's half and the peer's.
pub(crate) type Absolute = DmTensor<
    f8e4m3,
    Chips,
    Cluster,
    Reduced,
    m![Hidden / 2560, Hidden % 2560 / 1280, { Time }, { Packet }],
>;

/// A block's rows normalized by the all-reduced row sums `total`, times `gain`, quantized by
/// `divisor` into slot 0 of the returned slots.
fn normalize_with(
    device: &mut Device,
    (gain, divisor): (&Gain, &scale::InputScale<Chips, Cluster, Reduced>),
    (residual, total): (&dpanel::Own, &RowSums),
) -> Slots {
    let mut slots = Slots::new();
    let inverse: VrfTensor<f32, Chips, Cluster, Reduced, m![Block % 8, 1 # 8]> = device
        .main
        .begin(total.view())
        .fetch::<m![Block % 8], m![1 # 2]>()
        .collect::<m![Block % 8], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_fp_div_with_mode(BinaryArgMode::Mode10, 1.0_f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .to_vrf::<m![Block % 8, 1 # 8]>(&mut device.sub);
    let weight: VrfTensor<f32, Chips, Cluster, Reduced, m![Hidden % 2560 / 1280, Hidden % 80]> =
        device
            .sub
            .begin(gain.view())
            .fetch::<m![Hidden % 2560 / 1280, Hidden % 80 / 8], m![Hidden % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Hidden % 2560 / 1280, Hidden % 80 / 8], m![Hidden % 8]>()
            .to_vrf::<m![Hidden % 2560 / 1280, Hidden % 80]>();
    device
        .main
        .begin(residual.view())
        .fetch::<m![Hidden % 2560 / 1280, Block % 8, Hidden % 80 / 8], m![Hidden % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Hidden % 2560 / 1280, Block % 8, Hidden % 80 / 8], m![Hidden % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Hidden % 2560 / 1280, Block % 8, Hidden % 80 / 8, Hidden % 8 / 4], m![Hidden % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &inverse)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight)
        .vector_fp_div(divisor.input_scale())
        .vector_widen_concat::<m![Hidden % 2560 / 1280, Block % 8, Hidden % 80 / 8], m![Hidden % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Hidden % 8 # 32]>()
        .commit_trim::<m![Hidden % 8]>()
        .commit_view(
            slots
                .view_mut()
                .tile::<m![Slot], 1, m![Slot = 1 #{!} 2, Hidden % 2560 / 1280, { Time }, { Packet }]>(0),
        );
    slots
}

/// The all-gather of a normalized block (slot 0 of `slots`): the peer's half over the link into
/// slot 1, then both halves in absolute order (`chip_slice` on the slot: [0, 1] for half 0,
/// [1, 0] for half 1) into block `block` of the K80-blocked activation the gate/up lands.
pub(super) fn gather(
    device: &mut Device,
    slots: &mut Slots,
    (blocked, block): (
        &mut crate::models::nemo::serving::mlp::block::Blocked,
        usize,
    ),
) {
    type Own8 = m![Hidden % 2560 / 1280, { Time }, { Packet }];
    let peer = slots
        .view()
        .tile::<m![Slot], 1, m![Slot = 1 # 2, Hidden % 2560 / 1280, { Time }, { Packet }]>(0)
        .chip_shuffle([1, 0])
        .to_dm::<m![Slot = 1 # 2, Hidden % 2560 / 1280, { Time }, { Packet }]>(&mut device.tdma);
    peer.view().to_dm_view(
        &mut device.tdma,
        slots
            .view_mut()
            .tile::<m![Slot], 1, m![Slot = 1 #{!} 2, Hidden % 2560 / 1280, { Time }, { Packet }]>(
                1,
            ),
    );
    let mut absolute = Absolute::new();
    slots.view().chip_slice::<m![Slot], Own8>([0, 1]).to_dm_view(
        &mut device.tdma,
        absolute
            .view_mut()
            .tile::<m![Hidden / 2560], 1, m![Hidden / 2560 = 1 #{!} 2, Hidden % 2560 / 1280, { Time }, { Packet }]>(0),
    );
    slots.view().chip_slice::<m![Slot], Own8>([1, 0]).to_dm_view(
        &mut device.tdma,
        absolute
            .view_mut()
            .tile::<m![Hidden / 2560], 1, m![Hidden / 2560 = 1 #{!} 2, Hidden % 2560 / 1280, { Time }, { Packet }]>(1),
    );
    absolute.view().to_hbm_view(
        &mut device.tdma,
        blocked.view_mut().tile::<m![Blocks], 1, m![
            Blocks = 1 #{!} Blocks,
            Hidden / 80,
            Block,
            Hidden % 80
        ]>(block),
    );
}

/// The peer's half of a normalized block (slot 0 of `slots`) on its way over the link; `place`
/// moves it into absolute order once it lands.
pub(super) fn gather_send(device: &mut Device, slots: &Slots) -> Peer {
    Peer(
        slots
            .view()
            .tile::<m![Slot], 1, m![Slot = 1 # 2, Hidden % 2560 / 1280, { Time }, { Packet }]>(0)
            .chip_shuffle([1, 0])
            .to_dm::<m![Slot = 1 # 2, Hidden % 2560 / 1280, { Time }, { Packet }]>(
                &mut device.tdma,
            ),
    )
}

/// Every block's FP8 rows of both halves for the next layer's Q/K/V, in the absolute halves'
/// own order (each block one contiguous run); the Q/K/V reload regroups them.
pub(crate) type NextRows = HbmTensor<
    f8e4m3,
    Chips,
    m![
        Blocks,
        Hidden % 1280 / 640,
        Hidden % 640 / 320,
        Block / 8,
        Hidden % 320 / 80,
        Hidden / 2560,
        Hidden % 2560 / 1280,
        Block % 8,
        Hidden % 80 / 8,
        Hidden % 8
    ],
>;

/// `NextRows` under `Held`'s names: the same bytes, `Slot` where `NextRows` has `Hidden / 2560`.
type Halved = m![
    Blocks,
    Hidden % 1280 / 640,
    Hidden % 640 / 320,
    Block / 8,
    Hidden % 320 / 80,
    Slot,
    Hidden % 2560 / 1280,
    Block % 8,
    Hidden % 80 / 8,
    Hidden % 8
];
type HalfTile = m![
    Blocks = 1 #{!} Blocks,
    Hidden % 1280 / 640,
    Hidden % 640 / 320,
    Block / 8,
    Hidden % 320 / 80,
    Slot = 1 #{!} 2,
    Hidden % 2560 / 1280,
    Block % 8,
    Hidden % 80 / 8,
    Hidden % 8
];
type HalfRead = m![
    Blocks = 1 # Blocks,
    Hidden % 1280 / 640,
    Hidden % 640 / 320,
    Block / 8,
    Hidden % 320 / 80,
    Slot = 1 # 2,
    Hidden % 2560 / 1280,
    Block % 8,
    Hidden % 80 / 8,
    Hidden % 8
];
/// One half of a block's normalized FP8 rows in data memory, slot-shaped.
pub(crate) type Half = DmTensor<
    f8e4m3,
    Chips,
    Cluster,
    Reduced,
    m![Slot = 1 # 2, Hidden % 2560 / 1280, { Time }, { Packet }],
>;

/// Half `half` (absolute, `Hidden / 2560`) of block `block` of `rows`.
fn half_mut(
    rows: &mut NextRows,
    (block, half): (usize, usize),
) -> HbmTensorViewMut<'_, f8e4m3, Chips, HalfTile> {
    // SAFETY: a relabel: `Slot` names `Hidden / 2560`'s two positions in the same order.
    let halved = unsafe { rows.view_mut().reshape::<Chips, Halved>() };
    halved
        .tile::<m![Blocks], 1, m![Blocks = 1 #{!} Blocks, Hidden % 1280 / 640, Hidden % 640 / 320, Block / 8, Hidden % 320 / 80, Slot, Hidden % 2560 / 1280, Block % 8, Hidden % 80 / 8, Hidden % 8]>(block)
        .tile::<m![Slot], 1, HalfTile>(half)
}

/// This chip's normalized half of a block (slot 0 of `slots`) into its own position `chip` of
/// block `block` of the next layer's rows.
pub(crate) fn hold_own(
    device: &mut Device,
    slots: &Slots,
    (rows, block, chip): (&mut NextRows, usize, usize),
) {
    slots
        .view()
        .tile::<m![Slot], 1, m![Slot = 1 # 2, Hidden % 2560 / 1280, { Time }, { Packet }]>(0)
        .to_hbm_view(&mut device.tdma, half_mut(rows, (block, chip)));
}

/// This chip's half of block `block` of `rows` (placed by `hold_own`) landed and sent to the peer.
pub(crate) fn send_own(
    device: &mut Device,
    rows: &NextRows,
    (block, chip): (usize, usize),
) -> Peer {
    let mut peer = Peer::new();
    send_into(device, rows, (block, chip), &mut peer);
    peer
}

/// `send_own` into an existing `peer`.
fn send_into(device: &mut Device, rows: &NextRows, (block, chip): (usize, usize), peer: &mut Peer) {
    // SAFETY: as in `half_mut`.
    let halved = unsafe { rows.view().reshape::<Chips, Halved>() };
    let own: Half = halved
        .tile::<m![Blocks], 1, m![Blocks = 1 # Blocks, Hidden % 1280 / 640, Hidden % 640 / 320, Block / 8, Hidden % 320 / 80, Slot, Hidden % 2560 / 1280, Block % 8, Hidden % 80 / 8, Hidden % 8]>(block)
        .tile::<m![Slot], 1, HalfRead>(chip)
        .to_dm(&mut device.tdma);
    own.chip_shuffle([1, 0])
        .to_dm_view(&mut device.tdma, peer.0.view_mut());
}

/// The peer's normalized half as it arrives.
pub(crate) struct Peer(
    DmTensor<
        f8e4m3,
        Chips,
        Cluster,
        Reduced,
        m![Slot = 1 # 2, Hidden % 2560 / 1280, { Time }, { Packet }],
    >,
);

impl Peer {
    /// A peer half not yet sent, for an owner a later `gather_send` replaces.
    pub(crate) fn new() -> Self {
        Self(DmTensor::new())
    }

    /// The arrived half into the peer's position `1 - chip` of block `block` of `rows`.
    pub(crate) fn place_half(
        &self,
        device: &mut Device,
        (rows, block, chip): (&mut NextRows, usize, usize),
    ) {
        self.0
            .view()
            .to_hbm_view(&mut device.tdma, half_mut(rows, (block, 1 - chip)));
    }

    /// `place` into block `block` of the next layer's rows.
    pub(crate) fn place_next(
        &self,
        device: &mut Device,
        slots: &mut Slots,
        (rows, block): (&mut NextRows, usize),
    ) {
        type Own8 = m![Hidden % 2560 / 1280, { Time }, { Packet }];
        self.0.view().to_dm_view(
            &mut device.tdma,
            slots
                .view_mut()
                .tile::<m![Slot], 1, m![Slot = 1 #{!} 2, Hidden % 2560 / 1280, { Time }, { Packet }]>(1),
        );
        let mut absolute = Absolute::new();
        slots.view().chip_slice::<m![Slot], Own8>([0, 1]).to_dm_view(
            &mut device.tdma,
            absolute
                .view_mut()
                .tile::<m![Hidden / 2560], 1, m![Hidden / 2560 = 1 #{!} 2, Hidden % 2560 / 1280, { Time }, { Packet }]>(0),
        );
        slots.view().chip_slice::<m![Slot], Own8>([1, 0]).to_dm_view(
            &mut device.tdma,
            absolute
                .view_mut()
                .tile::<m![Hidden / 2560], 1, m![Hidden / 2560 = 1 #{!} 2, Hidden % 2560 / 1280, { Time }, { Packet }]>(1),
        );
        absolute.view().to_hbm_view(
            &mut device.tdma,
            rows.view_mut().tile::<m![Blocks], 1, m![
                Blocks = 1 #{!} Blocks,
                Hidden % 1280 / 640,
                Hidden % 640 / 320,
                Block / 8,
                Hidden % 320 / 80,
                Hidden / 2560,
                Hidden % 2560 / 1280,
                Block % 8,
                Hidden % 80 / 8,
                Hidden % 8
            ]>(block),
        );
    }

    /// The arrived peer half into slot 1 of `slots`, both halves in absolute order into block
    /// `block` of the K80-blocked activation the gate/up lands.
    pub(super) fn place(
        self,
        device: &mut Device,
        slots: &mut Slots,
        (blocked, block): (
            &mut crate::models::nemo::serving::mlp::block::Blocked,
            usize,
        ),
    ) {
        type Own8 = m![Hidden % 2560 / 1280, { Time }, { Packet }];
        self.0.view().to_dm_view(
            &mut device.tdma,
            slots
                .view_mut()
                .tile::<m![Slot], 1, m![Slot = 1 #{!} 2, Hidden % 2560 / 1280, { Time }, { Packet }]>(1),
        );
        let mut absolute = Absolute::new();
        slots.view().chip_slice::<m![Slot], Own8>([0, 1]).to_dm_view(
            &mut device.tdma,
            absolute
                .view_mut()
                .tile::<m![Hidden / 2560], 1, m![Hidden / 2560 = 1 #{!} 2, Hidden % 2560 / 1280, { Time }, { Packet }]>(0),
        );
        slots.view().chip_slice::<m![Slot], Own8>([1, 0]).to_dm_view(
            &mut device.tdma,
            absolute
                .view_mut()
                .tile::<m![Hidden / 2560], 1, m![Hidden / 2560 = 1 #{!} 2, Hidden % 2560 / 1280, { Time }, { Packet }]>(1),
        );
        absolute.view().to_hbm_view(
            &mut device.tdma,
            blocked.view_mut().tile::<m![Blocks], 1, m![
                Blocks = 1 #{!} Blocks,
                Hidden / 80,
                Block,
                Hidden % 80
            ]>(block),
        );
    }
}

/// `gather` straight into the gate/up's landed activation, a copy in every output group, without
/// the HBM round trip.
pub(super) fn gather_into(
    device: &mut Device,
    slots: &mut Slots,
    (absolute, activation): (
        &mut Absolute,
        &mut crate::models::nemo::serving::mlp::block::Activation,
    ),
) {
    type Own8 = m![Hidden % 2560 / 1280, { Time }, { Packet }];
    let peer = slots
        .view()
        .tile::<m![Slot], 1, m![Slot = 1 # 2, Hidden % 2560 / 1280, { Time }, { Packet }]>(0)
        .chip_shuffle([1, 0])
        .to_dm::<m![Slot = 1 # 2, Hidden % 2560 / 1280, { Time }, { Packet }]>(&mut device.tdma);
    peer.view().to_dm_view(
        &mut device.tdma,
        slots
            .view_mut()
            .tile::<m![Slot], 1, m![Slot = 1 #{!} 2, Hidden % 2560 / 1280, { Time }, { Packet }]>(
                1,
            ),
    );
    slots.view().chip_slice::<m![Slot], Own8>([0, 1]).to_dm_view(
        &mut device.tdma,
        absolute
            .view_mut()
            .tile::<m![Hidden / 2560], 1, m![Hidden / 2560 = 1 #{!} 2, Hidden % 2560 / 1280, { Time }, { Packet }]>(0),
    );
    slots.view().chip_slice::<m![Slot], Own8>([1, 0]).to_dm_view(
        &mut device.tdma,
        absolute
            .view_mut()
            .tile::<m![Hidden / 2560], 1, m![Hidden / 2560 = 1 #{!} 2, Hidden % 2560 / 1280, { Time }, { Packet }]>(1),
    );
    absolute
        .view()
        .to_dm_view(&mut device.tdma, activation.view_mut());
}

/// Stage 2 of a finished block of the layer output: its rows normalized by the next layer's
/// input norm (row sums all-reduced as phase A's), quantized by the QKV input scale, and the
/// peer's half on its way over the link; `Peer::place` lands both halves in `next_rows`.
fn next_input(device: &mut Device, tail: &Tail<'_>, own: &dpanel::Own) -> (Slots, Peer) {
    let partial = squares(device, own);
    let local = local_sums(device, &partial);
    let other = local
        .cluster_swap()
        .to_dm::<m![Block % 8, 1 # 8]>(&mut device.tdma);
    let chip = sum(device, (&local, &other));
    let peer = chip
        .chip_shuffle([1, 0])
        .to_dm::<m![Block % 8, 1 # 8]>(&mut device.tdma);
    let total = sum(device, (&chip, &peer));
    let slots = normalize_with(device, (tail.next.0.weight(), tail.next.1), (own, &total));
    let sent = gather_send(device, &slots);
    (slots, sent)
}

/// What a layer's tail reads besides the rows: the O weight and input scale, the norm's gain and
/// the gate input scale it quantizes by.
pub(super) struct Tail<'a> {
    pub(super) output: &'a dshard::Weight,
    pub(super) divisor: &'a scale::InputScale<Chips, dshard::Cluster, dshard::Landing>,
    pub(super) gain: &'a PostNorm,
    pub(super) quantize: &'a scale::InputScale<Chips, Cluster, Reduced>,
    /// The next layer's input norm gain and QKV input scale (stage 2: its FP8 QKV input).
    pub(super) next: (&'a PostNorm, &'a scale::InputScale<Chips, Cluster, Reduced>),
}

/// Phase A of block `block`: each wave's own heads exchanged and projected, the block's residual
/// (into `residuals`), its norm, and its FP8 activation all-gathered into `blocked`.
pub(super) fn tail<const NORM: usize, const GATHER: bool>(
    device: &mut Device,
    tail: &Tail<'_>,
    (prompt, hidden): (&attention::prompt::Attended, &dpanel::Owned),
    (residuals, blocked): (
        &mut dpanel::Owned,
        &mut crate::models::nemo::serving::mlp::block::Blocked,
    ),
    block: usize,
) {
    let filled = tail.output.fill(device);
    let mut sums = dpanel::Own::new();
    #[unroll]
    for wave in 0..4 {
        let rows = heads(device, (prompt, (0, 4 * block + wave)), tail.divisor);
        let mut projected = dshard::Output::new();
        filled.project_into(device, (&rows.0, &rows.1), projected.view_mut());
        place(device, &projected, (&mut sums, wave));
    }
    let input = dpanel::load(device, hidden, block);
    let mut residual = dpanel::Own::new();
    self::residual(device, (&sums, &input), &mut residual);
    dpanel::store(device, &residual, (residuals, block));
    let mut slots = Slots::new();
    if NORM > 0 {
        normalize::<NORM>(
            device,
            &residual,
            (tail.gain.weight(), tail.quantize),
            &mut slots,
        );
    } else {
        slots
            .view_mut()
            .memset(const { f8e4m3::from_f32(0.0) }, &mut device.sub);
    }
    if GATHER {
        gather(device, &mut slots, (blocked, block));
    } else {
        // SAFETY: debug store of the slots' bytes only.
        let bytes = unsafe {
            slots.view().reshape::<Chips, Cluster, Reduced, m![Hidden / 2560, Hidden % 2560 / 1280, { Time }, { Packet }]>()
        };
        bytes.to_hbm_view(
            &mut device.tdma,
            blocked
                .view_mut()
                .tile::<m![Blocks], 1, m![Blocks = 1 #{!} Blocks, Hidden / 80, Block, Hidden % 80]>(
                    block,
                ),
        );
    }
}

/// A prompt chunk's whole layer after attention over blocks `0..blocks`, D-sharded: phase A of
/// block `k + 1` (O, residual, norm, activation all-gather) after the gate/up of block `k` and the
/// down of block `k - 1` (`mlp::dpanel`), their registers in the other TRF half.
#[allow(clippy::too_many_arguments)]
pub(super) fn layer(
    device: &mut Device,
    (tail_parts, gates, panels, scales): (
        &Tail<'_>,
        &crate::models::nemo::serving::mlp::block::Staged,
        &dpanel::Staged,
        &scale::Scales<Chips>,
    ),
    (prompt, hidden): (&attention::prompt::Attended, &dpanel::Owned),
    (residuals, blocked, output): (
        &mut dpanel::Owned,
        &mut crate::models::nemo::serving::mlp::block::Blocked,
        &mut dpanel::Owned,
    ),
    blocks: usize,
) {
    use crate::models::nemo::serving::mlp::block;
    let mut product = block::Product::new();
    let mut activation = block::Activation::new();
    let mut landed = dpanel::Activation::new();
    // The padded K groups past the 7,168 features stay zero for every block.
    landed
        .view_mut()
        .memset(const { f8e4m3::from_f32(0.0) }, &mut device.sub);
    tail::<4, true>(
        device,
        tail_parts,
        (prompt, hidden),
        (&mut *residuals, &mut *blocked),
        0,
    );
    block::land_blocked(device, blocked, 0, &mut activation);
    let filled = block::fill(device, &activation);
    #[unroll]
    for half in 0..2 {
        let up = block::up(device, (gates, scales), (&filled, half));
        let gate = block::gate(device, (gates, scales), (&filled, half));
        block::finish(device, scales, (gate, up), (&mut product, 0, half));
    }
    for _ in 0..(blocks - 1).min(1) {
        tail::<4, true>(
            device,
            tail_parts,
            (prompt, hidden),
            (&mut *residuals, &mut *blocked),
            1,
        );
        block::land_blocked(device, blocked, 1, &mut activation);
    }
    for k in 1..blocks {
        let filled = block::fill(device, &activation);
        land(device, &product, k - 1, &mut landed);
        let residual = dpanel::load(device, residuals, k - 1);
        let mut own = dpanel::Own::new();
        let up = block::up(device, (gates, scales), (&filled, 0));
        let mut first = dpanel::Both::new();
        let register = dpanel::fill(device, panels, 0);
        dpanel::contract_into(device, (&landed, &register), (&mut first, 0));
        let gate = block::gate(device, (gates, scales), (&filled, 0));
        block::finish(device, scales, (gate, up), (&mut product, k, 0));
        let register = dpanel::fill(device, panels, 2);
        dpanel::contract_into(device, (&landed, &register), (&mut first, 1));
        let sent = dpanel::send(device, &first);
        let up = block::up(device, (gates, scales), (&filled, 1));
        let mut second = dpanel::Both::new();
        let register = dpanel::fill(device, panels, 1);
        dpanel::contract_into(device, (&landed, &register), (&mut second, 0));
        dpanel::finish(device, (&sent, scales), (&residual, &mut own, 0));
        let register = dpanel::fill(device, panels, 3);
        dpanel::contract_into(device, (&landed, &register), (&mut second, 1));
        let sent = dpanel::send(device, &second);
        let gate = block::gate(device, (gates, scales), (&filled, 1));
        block::finish(device, scales, (gate, up), (&mut product, k, 1));
        // Phase A of block k + 1 under the second pair's reduce-scatter; it lands for the next trip.
        for _ in 0..(blocks - k - 1).min(1) {
            tail::<4, true>(
                device,
                tail_parts,
                (prompt, hidden),
                (&mut *residuals, &mut *blocked),
                k + 1,
            );
        }
        dpanel::finish(device, (&sent, scales), (&residual, &mut own, 1));
        dpanel::store(device, &own, (&mut *output, k - 1));
        for _ in 0..(blocks - k - 1).min(1) {
            block::land_blocked(device, blocked, k + 1, &mut activation);
        }
    }
    land(device, &product, blocks - 1, &mut landed);
    let residual = dpanel::load(device, residuals, blocks - 1);
    let mut own = dpanel::Own::new();
    dpanel::down(device, (panels, scales), &landed, (&residual, &mut own));
    dpanel::store(device, &own, (output, blocks - 1));
}

use dpanel::land;

/// Debug: inverse RMS of `sums` into a VRF, then `residual` times it, committed as BF16.
pub(super) fn scaled(
    device: &mut Device,
    (sums, residual): (&RowSums, &dpanel::Own),
    output: &mut dpanel::Own,
) {
    let inverse: VrfTensor<f32, Chips, Cluster, Reduced, m![Block % 8, 1 # 8]> = device
        .main
        .begin(sums.view())
        .fetch::<m![Block % 8], m![1 # 2]>()
        .collect::<m![Block % 8], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_fp_div_with_mode(BinaryArgMode::Mode10, 1.0_f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .to_vrf::<m![Block % 8, 1 # 8]>(&mut device.sub);
    device
        .main
        .begin(residual.view())
        .fetch::<m![Hidden % 2560 / 1280, Block % 8, Hidden % 80 / 8], m![Hidden % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Hidden % 2560 / 1280, Block % 8, Hidden % 80 / 8], m![Hidden % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Hidden % 2560 / 1280, Block % 8, Hidden % 80 / 8, Hidden % 8 / 4], m![Hidden % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &inverse)
        .vector_widen_concat::<m![Hidden % 2560 / 1280, Block % 8, Hidden % 80 / 8], m![Hidden % 8]>()
        .vector_final()
        .cast::<bf16, m![Hidden % 8 # 16]>()
        .commit_trim::<m![Hidden % 8]>()
        .commit_view(output.view_mut())
}

/// Debug: `residual` times the norm gain, committed as BF16.
pub(super) fn gained(
    device: &mut Device,
    (gain, residual): (&Gain, &dpanel::Own),
    output: &mut dpanel::Own,
) {
    let weight: VrfTensor<f32, Chips, Cluster, Reduced, m![Hidden % 2560 / 1280, Hidden % 80]> =
        device
            .sub
            .begin(gain.view())
            .fetch::<m![Hidden % 2560 / 1280, Hidden % 80 / 8], m![Hidden % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Hidden % 2560 / 1280, Hidden % 80 / 8], m![Hidden % 8]>()
            .to_vrf::<m![Hidden % 2560 / 1280, Hidden % 80]>();
    device
        .main
        .begin(residual.view())
        .fetch::<m![Hidden % 2560 / 1280, Block % 8, Hidden % 80 / 8], m![Hidden % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Hidden % 2560 / 1280, Block % 8, Hidden % 80 / 8], m![Hidden % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Hidden % 2560 / 1280, Block % 8, Hidden % 80 / 8, Hidden % 8 / 4], m![Hidden % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight)
        .vector_widen_concat::<m![Hidden % 2560 / 1280, Block % 8, Hidden % 80 / 8], m![Hidden % 8]>()
        .vector_final()
        .cast::<bf16, m![Hidden % 8 # 16]>()
        .commit_trim::<m![Hidden % 8]>()
        .commit_view(output.view_mut())
}

type Split = m![
    Hidden % 640 / 320,
    Block / 8,
    Hidden % 320 / 160,
    Hidden % 160 / 80
];
type SplitSums<'a> = DmTensorView<'a, f32, Chips, Cluster, Split, m![Block % 8, 1 # 8]>;
type SplitSumsMut<'a> = DmTensorViewMut<'a, f32, Chips, Cluster, Split, m![Block % 8, 1 # 8]>;

/// Slice digit `DIGIT` (0: `Hidden % 160 / 80`, 1: `Hidden % 320 / 160`, 2: `Hidden % 640 / 320`)
/// of `from`'s half `half` into `to`'s other half.
fn cross<const DIGIT: usize, const HALF: usize, const OTHER: usize>(
    device: &mut Device,
    (from, to): (SplitSums<'_>, SplitSumsMut<'_>),
) {
    if DIGIT == 0 {
        from.slice_tile::<m![Hidden % 160 / 80], 1, m![Hidden % 640 / 320, Block / 8, Hidden % 320 / 160, Hidden % 160 / 80 = 1 # 2]>(HALF)
            .to_dm_view(&mut device.tdma, to.slice_tile::<m![Hidden % 160 / 80], 1, m![Hidden % 640 / 320, Block / 8, Hidden % 320 / 160, Hidden % 160 / 80 = 1 #{!} 2]>(OTHER));
    } else if DIGIT == 1 {
        from.slice_tile::<m![Hidden % 320 / 160], 1, m![Hidden % 640 / 320, Block / 8, Hidden % 320 / 160 = 1 # 2, Hidden % 160 / 80]>(HALF)
            .to_dm_view(&mut device.tdma, to.slice_tile::<m![Hidden % 320 / 160], 1, m![Hidden % 640 / 320, Block / 8, Hidden % 320 / 160 = 1 #{!} 2, Hidden % 160 / 80]>(OTHER));
    } else {
        from.slice_tile::<m![Hidden % 640 / 320], 1, m![Hidden % 640 / 320 = 1 # 2, Block / 8, Hidden % 320 / 160, Hidden % 160 / 80]>(HALF)
            .to_dm_view(&mut device.tdma, to.slice_tile::<m![Hidden % 640 / 320], 1, m![Hidden % 640 / 320 = 1 #{!} 2, Block / 8, Hidden % 320 / 160, Hidden % 160 / 80]>(OTHER));
    }
}

/// Every slice's eight partials of its row group (the group's eight hidden slices), one transfer.
type Gathered = DmTensor<
    f32,
    Chips,
    Cluster,
    m![Copy640, Block / 8, Copy320],
    m![Hidden % 640 / 320, Hidden % 320 / 80, Block % 8, 1 # 8],
>;

/// The eight gathered partials of every slice's row group summed (`local_sums`' second half).
fn reduce_gathered(device: &mut Device, gathered: &Gathered) -> RowSums {
    let sums: DmTensor<f32, Chips, Cluster, m![Copy640, Block / 8, Copy320], m![Block % 8, 1 # 8]> =
        device
            .main
            .begin(gathered.view())
            .fetch::<m![Hidden % 640 / 320, Hidden % 320 / 80, Block % 8], m![1 # 8]>()
            .collect::<m![Hidden % 640 / 320, Hidden % 320 / 80, Block % 8], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_intra_slice_reduce::<Hidden, m![Block % 8], m![1 # 4]>(
                IntraSliceReduceOpF32::Add,
            )
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
    // SAFETY: as in `local_sums`.
    unsafe { sums.reshape() }
}

/// `partial` summed over each row group's eight hidden slices: one gathering transfer that writes
/// every slice's eight partials into each slice of its group, then one pass over them.
fn local_sums(device: &mut Device, partial: &RowSums) -> RowSums {
    let mut gathered = Gathered::new();
    partial
        .view()
        .to_dm_view(&mut device.tdma, gathered.view_mut());
    let sums: DmTensor<f32, Chips, Cluster, m![Copy640, Block / 8, Copy320], m![Block % 8, 1 # 8]> =
        device
            .main
            .begin(gathered.view())
            .fetch::<m![Hidden % 640 / 320, Hidden % 320 / 80, Block % 8], m![1 # 8]>()
            .collect::<m![Hidden % 640 / 320, Hidden % 320 / 80, Block % 8], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_intra_slice_reduce::<Hidden, m![Block % 8], m![1 # 4]>(
                IntraSliceReduceOpF32::Add,
            )
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
    // SAFETY: every slice of a row group holds the group's sums; the copy digits name the hidden
    // slices they stand in.
    unsafe { sums.reshape() }
}

/// `sums` plus its copy with slice digit `DIGIT` swapped.
fn swapped<const DIGIT: usize>(device: &mut Device, sums: &RowSums) -> RowSums {
    let mut other = RowSums::new();
    // SAFETY: the row sums' slices split into the three hidden digits and the row digit.
    let split = || unsafe {
        sums.view()
            .reshape::<Chips, Cluster, Split, m![Block % 8, 1 # 8]>()
    };
    cross::<DIGIT, 0, 1>(device, (split(), unsafe { other.view_mut().reshape() }));
    cross::<DIGIT, 1, 0>(device, (split(), unsafe { other.view_mut().reshape() }));
    sum(device, (sums, &other))
}

/// A block's four waves of exchanged heads: (own, peer) rows a wave.
pub(super) struct Heads([(dshard::Rows8, dshard::Rows8); 4]);

/// Block `block`'s heads: each wave landed, quantized and sent to the peer chip.
pub(super) fn heads_of(
    device: &mut Device,
    (prompt, divisor): (
        &attention::prompt::Attended,
        &scale::InputScale<Chips, dshard::Cluster, dshard::Landing>,
    ),
    block: usize,
) -> Heads {
    Heads([
        heads(device, (prompt, (0, 4 * block)), divisor),
        heads(device, (prompt, (0, 4 * block + 1)), divisor),
        heads(device, (prompt, (0, 4 * block + 2)), divisor),
        heads(device, (prompt, (0, 4 * block + 3)), divisor),
    ])
}

/// Wave `wave`'s attended rows landed, ahead of their quantization.
fn land_wave(
    device: &mut Device,
    prompt: &attention::prompt::Attended,
    wave: usize,
) -> dshard::Rows32 {
    let attended = prompt.view().tile::<m![Prompt], 64, m![
        Prompt = 64 # Prompt,
        AttentionHeads,
        AttentionGroups,
        AttentionHead
    ]>(64 * wave);
    // SAFETY: as in `heads`.
    let attended = unsafe { attended.reshape::<Chips, m![Rows # Prompt, Width]>() };
    attended.to_dm(&mut device.tdma)
}

/// Block `block`'s four waves of attended rows landed.
pub(super) fn land_heads(
    device: &mut Device,
    prompt: &attention::prompt::Attended,
    block: usize,
) -> [dshard::Rows32; 4] {
    [
        land_wave(device, prompt, 4 * block),
        land_wave(device, prompt, 4 * block + 1),
        land_wave(device, prompt, 4 * block + 2),
        land_wave(device, prompt, 4 * block + 3),
    ]
}

/// One wave's quantized own heads sent to the peer chip.
fn peer_of(device: &mut Device, own: &dshard::Rows8) -> dshard::Rows8 {
    own.chip_shuffle([1, 0])
        .to_dm::<dshard::Element>(&mut device.tdma)
}

/// Landed heads quantized wave by wave, then each wave's sent to the peer chip.
pub(super) fn exchange_heads(
    device: &mut Device,
    landed: &[dshard::Rows32; 4],
    divisor: &scale::InputScale<Chips, dshard::Cluster, dshard::Landing>,
) -> Heads {
    let a = dshard::quantize(device, &landed[0], divisor);
    let b = dshard::quantize(device, &landed[1], divisor);
    let c = dshard::quantize(device, &landed[2], divisor);
    let d = dshard::quantize(device, &landed[3], divisor);
    let pa = peer_of(device, &a);
    let pb = peer_of(device, &b);
    let pc = peer_of(device, &c);
    let pd = peer_of(device, &d);
    Heads([(a, pa), (b, pb), (c, pc), (d, pd)])
}

/// Waves `FIRST`, `FIRST + 1` of a block projected into `sums`.
pub(super) fn project_pair<const FIRST: usize>(
    device: &mut Device,
    (filled, heads): (&dshard::Filled, &Heads),
    sums: &mut dpanel::Own,
) {
    #[unroll]
    for at in 0..2 {
        let (own, peer) = &heads.0[FIRST + at];
        let mut projected = dshard::Output::new();
        filled.project_into(device, (own, peer), projected.view_mut());
        place(device, &projected, (sums, FIRST + at));
    }
}

/// A block's residual (its O sums plus its own half of `hidden`, stored into `residuals`) and its
/// norm into slot 0 of the returned slots.
pub(super) fn settle(
    device: &mut Device,
    tail: &Tail<'_>,
    (sums, input): (&dpanel::Own, &dpanel::Own),
    residuals: &mut dpanel::Owned,
    block: usize,
) -> Slots {
    let mut residual = dpanel::Own::new();
    self::residual(device, (sums, input), &mut residual);
    dpanel::store(device, &residual, (residuals, block));
    let mut slots = Slots::new();
    normalize::<4>(
        device,
        &residual,
        (tail.gain.weight(), tail.quantize),
        &mut slots,
    );
    slots
}

/// The buffers a layer's trips share: the gate/up's block activation and product, the panels'
/// landed product and pairs, the down of the trip before's block (its residual, output and first
/// pair in flight, finished in the next trip), the next block's own heads, quantized in the trip
/// before, and the last product half of the trip before, stored in the next trip.
pub(super) struct Buffers {
    product: crate::models::nemo::serving::mlp::block::Product,
    activation: crate::models::nemo::serving::mlp::block::Activation,
    landed: dpanel::Activation,
    first: dpanel::Both,
    second: dpanel::Both,
    residual: dpanel::Own,
    own: dpanel::Own,
    sent: [dpanel::Sent; 2],
    quantized: [oblock::Rows8; 4],
    half: crate::models::nemo::serving::mlp::block::Stored,
    staged: oblock::Staged,
    /// A finished block's stage-2 rows, sent in the next trip.
    next_slots: Slots,
}

/// Both pairs of the block before last: their finishes and the store.
fn finish_deferred(
    device: &mut Device,
    scales: &scale::Scales<Chips>,
    output: &mut dpanel::Owned,
    buffers: &mut Buffers,
    block: usize,
) {
    dpanel::finish(
        device,
        (&buffers.sent[0], scales),
        (&buffers.residual, &mut buffers.own, 0),
    );
    dpanel::finish(
        device,
        (&buffers.sent[1], scales),
        (&buffers.residual, &mut buffers.own, 1),
    );
    dpanel::store(device, &buffers.own, (output, block));
}

/// A block's landed heads quantized as TP2's O input.
fn quantized(
    device: &mut Device,
    landed: &[dshard::Rows32; 4],
    divisor: &scale::InputScale<Chips, dshard::Cluster, dshard::Landing>,
) -> [dshard::Rows8; 4] {
    [
        dshard::quantize(device, &landed[0], divisor),
        dshard::quantize(device, &landed[1], divisor),
        dshard::quantize(device, &landed[2], divisor),
        dshard::quantize(device, &landed[3], divisor),
    ]
}

/// Block `block`'s four waves of attended rows landed in `oblock`'s layout.
fn land_block(
    device: &mut Device,
    prompt: &attention::prompt::Attended,
    block: usize,
) -> [oblock::Rows32; 4] {
    [
        oblock::land(device, prompt, 4 * block),
        oblock::land(device, prompt, 4 * block + 1),
        oblock::land(device, prompt, 4 * block + 2),
        oblock::land(device, prompt, 4 * block + 3),
    ]
}

/// A block's landed heads quantized as the O input, wave by wave.
fn quantize_block(
    device: &mut Device,
    landed: &[oblock::Rows32; 4],
    factors: &oblock::Factors,
) -> [oblock::Rows8; 4] {
    [
        oblock::quantize(device, &landed[0], factors),
        oblock::quantize(device, &landed[1], factors),
        oblock::quantize(device, &landed[2], factors),
        oblock::quantize(device, &landed[3], factors),
    ]
}

/// A block's four waves of quantized heads into part `part` of block `block` of `staged`.
fn store_block(
    device: &mut Device,
    rows: &[oblock::Rows8; 4],
    (staged, part, block): (&mut oblock::Staged, usize, usize),
) {
    #[unroll]
    for wave in 0..4 {
        oblock::store(device, &rows[wave], (staged, part, (block, wave)));
    }
}

/// Block `block`'s staged heads (both parts) projected once: its own-half O output.
fn project_block(
    device: &mut Device,
    (projection, factors): (&oblock::Projection, &oblock::Factors),
    (staged, block): (&oblock::Staged, usize),
) -> dpanel::Own {
    let mut heads = oblock::Heads::new();
    oblock::stage(device, (staged, block), &mut heads);
    let register = oblock::fill(device, &heads);
    let projected = projection.project(device, (&register, factors));
    oblock::transpose(device, &projected)
}

/// The O weight and scales of the block geometry (`output::oblock`).
pub(super) type Heads0<'a> = (&'a oblock::Projection, &'a oblock::Factors);

/// The layer whose MLP weights the D-sharded layer lands from the stack's one stored copy.
pub(super) type Packed<'a> = (&'a stack::Layers<'a>, usize);

/// Every block's FP8 half of this chip normalized for the next layer, slot 0 of each block; the
/// peer's half of a block lands in its slot 1 when the next layer's Q/K/V reaches it.
pub(crate) type Halves = DmTensor<
    f8e4m3,
    Chips,
    Cluster,
    Reduced,
    m![Blocks, Slot, Hidden % 2560 / 1280, { Time }, { Packet }],
>;

/// Blocks `0..blocks` of `owned` normalized by `gain` and quantized by `divisor` into `halves`.
pub(crate) fn normalize_halves(
    device: &mut Device,
    (gain, divisor): (&Gain, &scale::InputScale<Chips, Cluster, Reduced>),
    owned: &dpanel::Owned,
    (halves, blocks): (&mut Halves, usize),
) {
    for block in 0..blocks {
        let own = dpanel::load(device, owned, block);
        let mut slots = Slots::new();
        normalize::<4>(device, &own, (gain, divisor), &mut slots);
        slots.view().to_dm_view(
            &mut device.tdma,
            halves.view_mut().tile::<m![Blocks], 1, m![
                Blocks = 1 #{!} Blocks,
                Slot,
                Hidden % 2560 / 1280,
                { Time },
                { Packet }
            ]>(block),
        );
    }
}

/// Block `block` of `halves` exchanged with the peer chip, both halves in absolute order.
pub(crate) fn exchanged(device: &mut Device, halves: &mut Halves, block: usize) -> Absolute {
    type Own8 = m![Hidden % 2560 / 1280, { Time }, { Packet }];
    let peer = halves
        .view()
        .tile::<m![Blocks], 1, m![Blocks = 1 # Blocks, Slot, Hidden % 2560 / 1280, { Time }, { Packet }]>(block)
        .tile::<m![Slot], 1, m![Blocks = 1 # Blocks, Slot = 1 # 2, Hidden % 2560 / 1280, { Time }, { Packet }]>(0)
        .chip_shuffle([1, 0])
        .to_dm::<m![Blocks = 1 # Blocks, Slot = 1 # 2, Hidden % 2560 / 1280, { Time }, { Packet }]>(&mut device.tdma);
    peer.view().to_dm_view(
        &mut device.tdma,
        halves
            .view_mut()
            .tile::<m![Blocks], 1, m![Blocks = 1 #{!} Blocks, Slot, Hidden % 2560 / 1280, { Time }, { Packet }]>(block)
            .tile::<m![Slot], 1, m![Blocks = 1 #{!} Blocks, Slot = 1 #{!} 2, Hidden % 2560 / 1280, { Time }, { Packet }]>(1),
    );
    let both = || {
        halves
            .view()
            .tile::<m![Blocks], 1, m![Blocks = 1 # Blocks, Slot, Hidden % 2560 / 1280, { Time }, { Packet }]>(block)
    };
    let mut absolute = Absolute::new();
    both().chip_slice::<m![Slot], m![Blocks = 1 # Blocks, { Own8 }]>([0, 1]).to_dm_view(
        &mut device.tdma,
        absolute
            .view_mut()
            .tile::<m![Hidden / 2560], 1, m![Hidden / 2560 = 1 #{!} 2, Hidden % 2560 / 1280, { Time }, { Packet }]>(0),
    );
    both().chip_slice::<m![Slot], m![Blocks = 1 # Blocks, { Own8 }]>([1, 0]).to_dm_view(
        &mut device.tdma,
        absolute
            .view_mut()
            .tile::<m![Hidden / 2560], 1, m![Hidden / 2560 = 1 #{!} 2, Hidden % 2560 / 1280, { Time }, { Packet }]>(1),
    );
    absolute
}

/// Block `block`'s four waves of quantized heads sent to the peer chip.
fn send_block(device: &mut Device, own: &[oblock::Rows8; 4]) -> [oblock::Rows8; 4] {
    [
        oblock::send(device, &own[0]),
        oblock::send(device, &own[1]),
        oblock::send(device, &own[2]),
        oblock::send(device, &own[3]),
    ]
}

/// Every block's normalized slots in HBM between phase O and the gate/up's all-gather (each
/// slice's slots one run).
pub(crate) type Held = HbmTensor<
    f8e4m3,
    Chips,
    m![
        Blocks,
        Hidden % 1280 / 640,
        Hidden % 640 / 320,
        Block / 8,
        Hidden % 320 / 80,
        Slot,
        Hidden % 2560 / 1280,
        Block % 8,
        Hidden % 80 / 8,
        Hidden % 8
    ],
>;

/// What the next layer's Q/K/V lands under this layer's down.
pub(crate) trait Prefetch {
    fn land(&mut self, device: &mut Device);
    /// Work issued once the last phase O block's heads stage, ahead of its projection.
    fn staged(&mut self, _device: &mut Device, _rows: &NextRows) {}
    /// `land`, for a prefetch that places into the MLP input `rows`.
    fn land_into(&mut self, device: &mut Device, _rows: &mut NextRows) {
        self.land(device);
    }
}

impl Prefetch for () {
    fn land(&mut self, _device: &mut Device) {}
}

/// Phase G's first input: block 0's peer half, already across the link, placed and landed; then
/// the first up half, so phase G's first contraction waits for neither.
struct FirstInput<'a> {
    peer: &'a mut Peer,
    chip: usize,
    activation: &'a mut crate::models::nemo::serving::mlp::block::Activation,
    gates: (
        &'a mut crate::models::nemo::serving::mlp::block::Staged,
        Packed<'a>,
    ),
}

impl Prefetch for FirstInput<'_> {
    fn land(&mut self, _device: &mut Device) {
        unreachable!("the first input places into the MLP input rows");
    }

    /// Block 0's MLP input crosses the link under the last block's projection, so no earlier
    /// block's heads stage waits behind it.
    fn staged(&mut self, device: &mut Device, rows: &NextRows) {
        send_into(device, rows, (0, self.chip), &mut *self.peer);
    }

    fn land_into(&mut self, device: &mut Device, rows: &mut NextRows) {
        first_input(
            device,
            (&*self.peer, self.chip),
            (rows, &mut *self.activation),
        );
        self.gates.0.land_staged(device, self.gates.1, (1, 0));
    }
}

/// Block 0's peer half placed and both halves landed as phase G's first input.
fn first_input(
    device: &mut Device,
    (peer, chip): (&Peer, usize),
    (blocked, activation): (
        &mut NextRows,
        &mut crate::models::nemo::serving::mlp::block::Activation,
    ),
) {
    peer.place_half(device, (&mut *blocked, 0, chip));
    land_rows(device, blocked, 0, activation);
}

/// The next layer's K/V and Q weights (`qkv::tile::land_spans`, `Weights::land_query`).
pub(crate) struct NextWeights<'a> {
    pub(crate) weights: &'a mut crate::models::nemo::serving::qkv::Weights,
    pub(crate) layer: (&'a crate::models::nemo::serving::stack::Layers<'a>, usize),
    pub(crate) scale: &'a mut crate::models::nemo::serving::qkv::tile::PairScale,
}

impl Prefetch for NextWeights<'_> {
    fn land(&mut self, device: &mut Device) {
        *self.scale =
            crate::models::nemo::serving::qkv::tile::land_pairs(device, self.weights, self.layer);
        self.weights.land_query(device, self.layer);
        self.weights.land_norm(device, self.layer);
    }
}

/// A normalized block's slots into block `block` of `held`.
fn hold(device: &mut Device, slots: &Slots, (held, block): (&mut Held, usize)) {
    slots.view().to_hbm_view(
        &mut device.tdma,
        held.view_mut().tile::<m![Blocks], 1, m![
            Blocks = 1 #{!} Blocks,
            Hidden % 1280 / 640,
            Hidden % 640 / 320,
            Block / 8,
            Hidden % 320 / 80,
            Slot,
            Hidden % 2560 / 1280,
            Block % 8,
            Hidden % 80 / 8,
            Hidden % 8
        ]>(block),
    );
}

/// Block `block`'s held half exchanged with the peer, both halves placed into block `block` of
/// `rows`.
pub(crate) fn exchange_next(
    device: &mut Device,
    held: &Held,
    (rows, block): (&mut NextRows, usize),
) {
    let (mut slots, peer) = send_held(device, held, block);
    peer.place_next(device, &mut slots, (rows, block));
}

/// Block `block`'s held slots landed and this chip's half sent to the peer; `Peer::place` lands
/// both halves in the MLP input once the peer's half has arrived.
pub(crate) fn send_held(device: &mut Device, held: &Held, block: usize) -> (Slots, Peer) {
    let mut slots = Slots::new();
    held.view()
        .tile::<m![Blocks], 1, m![
            Blocks = 1 # Blocks,
            Hidden % 1280 / 640,
            Hidden % 640 / 320,
            Block / 8,
            Hidden % 320 / 80,
            Slot,
            Hidden % 2560 / 1280,
            Block % 8,
            Hidden % 80 / 8,
            Hidden % 8
        ]>(block)
        .to_dm_view(&mut device.tdma, slots.view_mut());
    let peer = gather_send(device, &slots);
    (slots, peer)
}

/// Block `block` of rows in the absolute halves' own order into the gate/up's activation, a copy
/// in every output group: the transfer regroups them.
fn land_rows(
    device: &mut Device,
    rows: &NextRows,
    block: usize,
    activation: &mut crate::models::nemo::serving::mlp::block::Activation,
) {
    rows.view()
        .tile::<m![Blocks], 1, m![
            Blocks = 1 # Blocks,
            Hidden % 1280 / 640,
            Hidden % 640 / 320,
            Block / 8,
            Hidden % 320 / 80,
            Hidden / 2560,
            Hidden % 2560 / 1280,
            Block % 8,
            Hidden % 80 / 8,
            Hidden % 8
        ]>(block)
        .to_dm_view(&mut device.tdma, activation.view_mut());
}

/// What `layer_seq`'s phase D carries from one block to the next: the landed product of the
/// current block (padded K groups zero) and its first pair with panel 0 contracted, the pairs'
/// reduce-scatters, the block's residual and output, and the block before's next-layer slots.
struct DownCarried {
    landed: dpanel::Activation,
    first: dpanel::Both,
    second: dpanel::Both,
    sent: [dpanel::Received; 2],
    residual: dpanel::Own,
    own: dpanel::Own,
    slots: Slots,
}

/// Block `k`'s heads (landed into `landed`) quantized as the O input, this chip's sent to the
/// peer; the returned pair is staged by `stage_heads`.
fn send_heads(
    device: &mut Device,
    heads0: Heads0<'_>,
    landed: &[oblock::Rows32; 4],
) -> ([oblock::Rows8; 4], [oblock::Rows8; 4]) {
    let own = quantize_block(device, landed, heads0.1);
    let peers = send_block(device, &own);
    (own, peers)
}

/// Phase O of block `k` (its heads staged by the iteration before): projected once, its
/// residual into `residual` (and `residuals`); block `k + 1`'s heads landed, quantized, sent and
/// staged under the projection (when `NEXT`); the block before's norm one step behind each
/// command, its slots into `held` (when `PREVIOUS`).
#[allow(clippy::too_many_arguments)]
fn phase_o<const NEXT: bool, const PREVIOUS: bool, const HOOK: bool, P: Prefetch, H: HeadsIn>(
    device: &mut Device,
    (tail, heads0): (&Tail<'_>, Heads0<'_>),
    (prompt, hidden): (&attention::prompt::Attended, &dpanel::Owned),
    (staged, residuals, (blocked, rank)): (
        &mut oblock::Staged,
        &mut dpanel::Owned,
        (&mut NextRows, usize),
    ),
    (residual, hook, mut heads): (&mut dpanel::Own, &mut P, H),
    k: usize,
) {
    heads.stage(device, (&*staged, k));
    let mut ahead = None;
    if NEXT {
        ahead = Some(oblock::land_rows(device, prompt, k + 1));
    }
    let input = Dm::<NormSpace, _> {
        value: residual.view(),
        tiles: Tiles::over(Block::SIZE),
    };
    let mut gathered = None;
    if PREVIOUS {
        gathered = Some(tail.gain.statistics::<NormSpace, m![Block % 8], Groups, Features, _>(device, &input)
            .gather::<m![Copy640, Block / 8, Copy320], m![Hidden % 640 / 320, Hidden % 320 / 80]>(device));
    }
    if HOOK {
        hook.staged(device, &*blocked);
    }
    let register = heads.fill(device);
    let mut quantized = None;
    if let Some(ahead) = &ahead {
        quantized = Some(oblock::quantize_rows(device, ahead, heads0.1));
    }
    let projected = heads0.0.project(device, (&register, heads0.1));
    if HOOK {
        // Phase G's first input lands under the last block's projection.
        hook.land_into(device, &mut *blocked);
    }
    let mut sent = None;
    if let Some(own) = &quantized {
        let peers = oblock::send_rows(device, own);
        oblock::store_rows(device, own, (&mut *staged, 1 - rank, k + 1));
        sent = Some(peers);
    }
    let mut local = None;
    if let Some(gathered) = gathered {
        local = Some(gathered.reduce(device).clusters(device));
    }
    let sums = oblock::transpose(device, &projected);
    let mut chip = None;
    if let Some(local) = local {
        chip = Some(local.combine(device).chips(device));
    }
    let input = dpanel::load(device, hidden, k);
    if let Some(chip) = chip {
        let total = chip.combine(device).finish();
        let mut slots = Slots::new();
        total.complete::<Groups, Features>(device).value
            .vector_fp_div(tail.quantize.input_scale())
            .vector_widen_concat::<m![Hidden % 2560 / 1280, Block % 8, Hidden % 80 / 8], m![Hidden % 8]>()
            .vector_final()
            .cast::<f8e4m3, m![Hidden % 8 # 32]>()
            .commit_trim::<m![Hidden % 8]>()
            .commit_view(slots.view_mut().tile::<m![Slot], 1, m![Slot = 1 #{!} 2, Hidden % 2560 / 1280, { Time }, { Packet }]>(0));
        hold_own(device, &slots, (&mut *blocked, k - 1, rank));
    }
    self::residual(device, (&sums, &input), residual);
    dpanel::store(device, &*residual, (&mut *residuals, k));
    if let Some(peers) = &sent {
        oblock::store_rows(device, peers, (&mut *staged, rank, k + 1));
    }
}

/// A block's heads for its O projection: staged from HBM in its phase (`Stage`), or staged
/// already (`Given`, block 0 by `first_heads`).
trait HeadsIn {
    fn stage(&mut self, device: &mut Device, staged: (&oblock::Staged, usize));
    fn fill(&self, device: &mut Device) -> oblock::Register;
}

struct Stage(oblock::Heads);

impl HeadsIn for Stage {
    fn stage(&mut self, device: &mut Device, staged: (&oblock::Staged, usize)) {
        oblock::stage(device, staged, &mut self.0);
    }

    fn fill(&self, device: &mut Device) -> oblock::Register {
        oblock::fill(device, &self.0)
    }
}

struct Given<'a>(&'a oblock::Heads);

impl HeadsIn for Given<'_> {
    fn stage(&mut self, _device: &mut Device, _staged: (&oblock::Staged, usize)) {}

    fn fill(&self, device: &mut Device) -> oblock::Register {
        oblock::fill(device, self.0)
    }
}

/// Phase G of block `k`: its gate and up into `product`; block `k + 1`'s MLP input sent under
/// the first up, placed and landed under the second half (when `NEXT`), the down's panels landed
/// behind it (when `PANELS`).
#[allow(clippy::too_many_arguments)]
fn phase_g<
    const NEXT: bool,
    const FIRST: bool,
    const PANELS: bool,
    const PREFETCH: bool,
    const NORM: bool,
    P: Prefetch,
>(
    device: &mut Device,
    (gates, scales, packed): (
        &mut crate::models::nemo::serving::mlp::block::Staged,
        &scale::Scales<Chips>,
        Packed<'_>,
    ),
    (rank, blocked, product, panels): (
        usize,
        &mut NextRows,
        &mut crate::models::nemo::serving::mlp::block::Product,
        &mut dpanel::Staged,
    ),
    (activation, prefetch, resident): (
        &mut crate::models::nemo::serving::mlp::block::Activation,
        &mut P,
        &mut dpanel::Resident,
    ),
    (residual, gains, last): (
        &dpanel::Own,
        (&PostNorm, &scale::InputScale<Chips, Cluster, Reduced>),
        usize,
    ),
    (k, lands): (usize, usize),
) {
    use crate::models::nemo::serving::mlp::block;
    let filled = block::fill(device, &*activation);
    let input = Dm::<NormSpace, _> {
        value: residual.view(),
        tiles: Tiles::over(Block::SIZE),
    };
    if FIRST {
        gates.land_part(device, packed, (0, 0), 1);
    }
    // Allocate the norm gather before the send so it cannot reuse the send's in-flight source.
    let mut gathered = None;
    if NORM {
        gathered = Some(gains.0.statistics::<NormSpace, m![Block % 8], Groups, Features, _>(device, &input)
            .gather::<m![Copy640, Block / 8, Copy320], m![Hidden % 640 / 320, Hidden % 320 / 80]>(device));
    }
    let up = block::up(device, (&*gates, scales), (&filled, 0));
    if FIRST && !NEXT {
        gates.land_staged(device, packed, (1, 1));
    }
    if PANELS {
        // The down's panels land under the last block, whose queue carries no next block's rows.
        #[unroll]
        for panel in 0..4 {
            panels.land_packed(device, packed, (panel, rank));
        }
    }
    let mut sent = None;
    if NEXT {
        sent = Some(send_own(device, &*blocked, (k + 1, rank)));
    }
    if FIRST && NEXT {
        // The first finish waits for the send, so this half follows it but precedes the store.
        gates.land_staged(device, packed, (1, 1));
    }
    let gate = block::gate(device, (&*gates, scales), (&filled, 0));
    if FIRST && !NEXT {
        gates.land_staged(device, packed, (0, 1));
    }
    block::finish(device, scales, (gate, up), (&mut *product, k, 0));
    let mut local = None;
    if let Some(gathered) = gathered {
        local = Some(gathered.reduce(device).clusters(device));
    }
    // The next block's half is placed while the second up runs.
    let up = block::up(device, (&*gates, scales), (&filled, 1));
    if let Some(peer) = sent {
        peer.place_half(device, (&mut *blocked, k + 1, rank));
    }
    if FIRST && NEXT {
        gates.land_staged(device, packed, (0, 1));
    }
    if NEXT {
        // A fresh owner: the current activation's register still feeds the second half.
        let mut next = block::Activation::new();
        land_rows(device, blocked, k + 1, &mut next);
        *activation = next;
    }
    // The block before's product, stored by its iteration, lands for the down here (`lands`, the
    // blocks the down's ring holds), where the queue has room for it.
    for _ in 0..lands {
        dpanel::land_resident(device, product, (k - 1, k - 1), resident);
    }
    let mut chip = None;
    if let Some(local) = local {
        chip = Some(local.combine(device).chips(device));
    }
    if PREFETCH {
        // The next layer's Q/K/V weights land under the last block's second up.
        prefetch.land(device);
    }
    let gate = block::gate(device, (&*gates, scales), (&filled, 1));
    block::finish(device, scales, (gate, up), (product, k, 1));
    if let Some(chip) = chip {
        let total = chip.combine(device).finish();
        let mut slots = Slots::new();
        total.complete::<Groups, Features>(device).value
            .vector_fp_div(gains.1.input_scale())
            .vector_widen_concat::<m![Hidden % 2560 / 1280, Block % 8, Hidden % 80 / 8], m![Hidden % 8]>()
            .vector_final()
            .cast::<f8e4m3, m![Hidden % 8 # 32]>()
            .commit_trim::<m![Hidden % 8]>()
            .commit_view(slots.view_mut().tile::<m![Slot], 1, m![Slot = 1 #{!} 2, Hidden % 2560 / 1280, { Time }, { Packet }]>(0));
        hold_own(device, &slots, (&mut *blocked, last, rank));
    }
}

/// Phase D of block `k` (its panels 0 and 2 contracted by the iteration before): the block
/// before's output normalized for the next layer under panel 2's tail and held (when
/// `PREVIOUS`); the first pair reduce-scattered under panels 1 and 3, the second under block
/// `k + 1`'s landing and panel 0 (when `NEXT`); the output stored; block `k + 1`'s panel 2.
#[allow(clippy::too_many_arguments)]
fn phase_d<
    const PREVIOUS: bool,
    const NEXT: bool,
    const ENTRY: bool,
    const PREFETCH: bool,
    P: Prefetch,
>(
    device: &mut Device,
    (panels, scales, tail): (&dpanel::Staged, &scale::Scales<Chips>, &Tail<'_>),
    (product, residuals, output, (next_rows, rank)): (
        &crate::models::nemo::serving::mlp::block::Product,
        &dpanel::Owned,
        &mut dpanel::Owned,
        (&mut NextRows, usize),
    ),
    (carried, entry, prefetch): (&mut DownCarried, &dpanel::Activation, &mut P),
    k: usize,
) {
    dpanel::send_slot(device, &carried.first, &mut carried.sent[0]);
    residuals
        .view()
        .tile::<m![Blocks], 1, m![Blocks = 1 # Blocks, Hidden % 2560 / 80, Block, Hidden % 80]>(k)
        .to_dm_view(&mut device.tdma, carried.residual.view_mut());
    let register = dpanel::fill(device, panels, 1);
    if ENTRY {
        dpanel::contract_into(device, (entry, &register), (&mut carried.second, 0));
    } else {
        dpanel::contract_into(
            device,
            (&carried.landed, &register),
            (&mut carried.second, 0),
        );
    }
    let register3 = dpanel::fill(device, panels, 3);
    // The block before's next-layer norm, one step behind each contraction: its squares and
    // the gathering transfer here, before this block's output overwrites it.
    let mut gathered = None;
    if PREVIOUS {
        let input = Dm::<NormSpace, _> {
            value: carried.own.view(),
            tiles: Tiles::over(Block::SIZE),
        };
        gathered = Some(tail.next.0.statistics::<NormSpace, m![Block % 8], Groups, Features, _>(device, &input)
            .gather::<m![Copy640, Block / 8, Copy320], m![Hidden % 640 / 320, Hidden % 320 / 80]>(device).into_value());
    }
    // Block k + 1's product lands into a fresh owner under panels 1 and 3, so its panels 0 and 2
    // run under the second reduce-scatter.
    let mut next = dpanel::Activation::new();
    if NEXT && ENTRY {
        // Nothing reads the carried owner yet: block k + 1 lands into it in place, so no copy
        // carries it into the loop.
        carried
            .landed
            .view_mut()
            .memset(const { f8e4m3::from_f32(0.0) }, &mut device.sub);
        land(device, product, k + 1, &mut carried.landed);
    }
    if NEXT && !ENTRY {
        next.view_mut()
            .memset(const { f8e4m3::from_f32(0.0) }, &mut device.sub);
        land(device, product, k + 1, &mut next);
    }
    if ENTRY {
        dpanel::contract_into(device, (entry, &register3), (&mut carried.second, 1));
    } else {
        dpanel::contract_into(
            device,
            (&carried.landed, &register3),
            (&mut carried.second, 1),
        );
    }
    // Panel 0's fill of block k + 1 ahead of the receipt the finish waits for.
    let mut fill0 = None;
    if NEXT {
        fill0 = Some(dpanel::fill(device, panels, 0));
    }
    let mut local = None;
    if let Some(gathered) = gathered {
        let input = Dm::<NormSpace, _> {
            value: carried.own.view(),
            tiles: Tiles::over(Block::SIZE),
        };
        local = Some(
            gathered
                .resume(tail.next.0, &input)
                .reduce(device)
                .clusters(device)
                .into_value(),
        );
    }
    dpanel::finish_slot(
        device,
        ((&carried.first, &carried.sent[0]), scales),
        (&carried.residual, &mut carried.own, 0),
    );
    dpanel::send_slot(device, &carried.second, &mut carried.sent[1]);
    if PREFETCH {
        // The next layer's Q/K/V weights land once, behind the first block's second
        // reduce-scatter.
        prefetch.land(device);
    }
    let mut fill2 = None;
    if let Some(register) = &fill0 {
        if ENTRY {
            dpanel::contract_into(device, (&carried.landed, register), (&mut carried.first, 0));
        } else {
            dpanel::contract_into(device, (&next, register), (&mut carried.first, 0));
        }
        fill2 = Some(dpanel::fill(device, panels, 2));
    }
    let mut chip = None;
    if let Some(local) = local {
        let input = Dm::<NormSpace, _> {
            value: carried.own.view(),
            tiles: Tiles::over(Block::SIZE),
        };
        chip = Some(
            local
                .resume(tail.next.0, &input)
                .combine(device)
                .chips(device)
                .into_value(),
        );
    }
    if let Some(register) = &fill2 {
        if ENTRY {
            dpanel::contract_into(device, (&carried.landed, register), (&mut carried.first, 1));
        } else {
            dpanel::contract_into(device, (&next, register), (&mut carried.first, 1));
        }
    }
    if NEXT && !ENTRY {
        carried.landed = next;
    }
    if let Some(chip) = chip {
        let input = Dm::<NormSpace, _> {
            value: carried.own.view(),
            tiles: Tiles::over(Block::SIZE),
        };
        let total = chip
            .resume(tail.next.0, &input)
            .combine(device)
            .finish()
            .into_value();
        let own = dpanel::load(device, output, k - 1);
        let input = Dm::<NormSpace, _> {
            value: own.view(),
            tiles: Tiles::over(Block::SIZE),
        };
        let total = total.resume(tail.next.0, &input);
        let mut slots = Slots::new();
        total.complete::<Groups, Features>(device).value
            .vector_fp_div(tail.next.1.input_scale())
            .vector_widen_concat::<m![Hidden % 2560 / 1280, Block % 8, Hidden % 80 / 8], m![Hidden % 8]>()
            .vector_final()
            .cast::<f8e4m3, m![Hidden % 8 # 32]>()
            .commit_trim::<m![Hidden % 8]>()
            .commit_view(slots.view_mut().tile::<m![Slot], 1, m![Slot = 1 #{!} 2, Hidden % 2560 / 1280, { Time }, { Packet }]>(0));
        hold_own(device, &slots, (&mut *next_rows, k - 1, rank));
    }
    dpanel::finish_slot(
        device,
        ((&carried.second, &carried.sent[1]), scales),
        (&carried.residual, &mut carried.own, 1),
    );
    dpanel::store(device, &carried.own, (&mut *output, k));
}

/// Block 0's heads landed, quantized, exchanged with the peer and staged: the layer's first
/// transfers, ahead of every weight and scale its later phases need.
pub(super) fn first_heads(
    device: &mut Device,
    prompt: &attention::prompt::Attended,
    (factors, heads): (&oblock::Factors, &mut oblock::Heads),
    (projection, weight): (&mut oblock::Projection, (&stack::Layers, usize)),
    rank: usize,
) -> oblock::Staged {
    let mut staged = oblock::Staged::new();
    let landed = oblock::land_rows(device, prompt, 0);
    let own = oblock::quantize_rows(device, &landed, factors);
    let peers = oblock::send_rows(device, &own);
    // Part `p` holds chip `1 - p`'s heads.
    oblock::store_rows(device, &own, (&mut staged, 1 - rank, 0));
    // The O weight lands while the queue waits for the peer's heads.
    projection.land_weight(device, (weight.0, weight.1, rank));
    oblock::store_rows(device, &peers, (&mut staged, rank, 0));
    oblock::stage(device, (&staged, 0), heads);
    staged
}

/// A prompt chunk's whole layer after attention over blocks `0..blocks` (at least 1), D-sharded,
/// one phase at a time over every block as the oracle's tokenwise layer runs it: the O, residual
/// and norm of every block (O); each block's gate and up with the next block's MLP input
/// all-gathered under them (G); each block's down with its reduce-scatter (D); every block's
/// next-layer FP8 input into `next_rows` (S).
#[allow(clippy::too_many_arguments)]
pub(super) fn layer_seq<P: Prefetch>(
    device: &mut Device,
    prefetch: &mut P,
    (tail_parts, packed, scales): (&Tail<'_>, Packed<'_>, &scale::Scales<Chips>),
    (heads0, mut staged, first): (Heads0<'_>, oblock::Staged, &oblock::Heads),
    (prompt, hidden): (&attention::prompt::Attended, &dpanel::Owned),
    (residuals, blocked, output): (&mut dpanel::Owned, &mut NextRows, &mut dpanel::Owned),
    (blocks, next_rows, chip): (usize, &mut NextRows, usize),
) {
    use crate::models::nemo::serving::mlp::block;
    let mut product = block::Product::new();
    let o = (tail_parts, heads0);
    let last = blocks - 1;
    let mut activation = block::Activation::new();
    let mut panels = dpanel::Staged::new();
    // Phase O: each block projected with the next block's heads exchanged under it and its norm
    // one block behind; block 0's heads were exchanged and staged by `first_heads`.
    let mut residual = dpanel::Own::new();
    let rows = (prompt, hidden);
    let three = (blocks - blocks.min(2)).min(1);
    let mut gates = block::Staged::from([
        block::Weight::new(),
        block::Weight::new(),
        block::Weight::new(),
        block::Weight::new(),
    ]);
    let mut parts = (&mut staged, &mut *residuals, &mut *blocked);
    // Block 0's MLP input crosses the link after the last block's phase O (one or two blocks), or
    // under the last block's projection (three or more, `FirstInput::staged`).
    let mut peer = Peer::new();
    for _ in 0..last.min(1) {
        phase_o::<true, false, false, _, _>(
            device,
            o,
            rows,
            (&mut *parts.0, &mut *parts.1, (&mut *parts.2, chip)),
            (&mut residual, &mut (), Given(first)),
            0,
        );
    }
    for _ in 0..1 - last.min(1) {
        phase_o::<false, false, false, _, _>(
            device,
            o,
            rows,
            (&mut *parts.0, &mut *parts.1, (&mut *parts.2, chip)),
            (&mut residual, &mut (), Given(first)),
            0,
        );
        let mut slots = Slots::new();
        normalize::<4>(
            device,
            &residual,
            (tail_parts.gain.weight(), tail_parts.quantize),
            &mut slots,
        );
        hold_own(device, &slots, (&mut *parts.2, 0, chip));
        send_into(device, &*parts.2, (0, chip), &mut peer);
    }
    // The down's product ring, zeroed while phase O waits for block 1's heads (the landings leave
    // its padded K groups untouched).
    let mut resident = dpanel::Resident::new();
    resident
        .view_mut()
        .memset(const { f8e4m3::from_f32(0.0) }, &mut device.sub);
    // A part fits the queue's idle interval while the ring memset occupies Main.
    gates.land_part(device, packed, (0, 0), 0);
    for _ in 0..three {
        phase_o::<true, true, false, _, _>(
            device,
            o,
            rows,
            (&mut *parts.0, &mut *parts.1, (&mut *parts.2, chip)),
            (&mut residual, &mut (), Stage(oblock::Heads::new())),
            1,
        );
    }
    for _ in 0..last.min(1) - three {
        phase_o::<false, true, false, _, _>(
            device,
            o,
            rows,
            (&mut *parts.0, &mut *parts.1, (&mut *parts.2, chip)),
            (&mut residual, &mut (), Stage(oblock::Heads::new())),
            1,
        );
    }
    for k in 2..blocks - 1 {
        phase_o::<true, true, false, _, _>(
            device,
            o,
            rows,
            (&mut *parts.0, &mut *parts.1, (&mut *parts.2, chip)),
            (&mut residual, &mut (), Stage(oblock::Heads::new())),
            k,
        );
    }
    // Two blocks: the last norm runs before the loop that reads block 0's slots, whose entry
    // waits for their send.
    for _ in 0..last.min(1) - three {
        send_into(device, &*parts.2, (0, chip), &mut peer);
        let mut slots = Slots::new();
        normalize::<4>(
            device,
            &residual,
            (tail_parts.gain.weight(), tail_parts.quantize),
            &mut slots,
        );
        hold_own(device, &slots, (&mut *parts.2, last, chip));
    }
    for _ in 0..three {
        let mut input = FirstInput {
            peer: &mut peer,
            chip,
            activation: &mut activation,
            gates: (&mut gates, packed),
        };
        phase_o::<false, true, true, _, _>(
            device,
            o,
            rows,
            (&mut *parts.0, &mut *parts.1, (&mut *parts.2, chip)),
            (&mut residual, &mut input, Stage(oblock::Heads::new())),
            last,
        );
    }
    for _ in 0..1 - three {
        first_input(device, (&peer, chip), (&mut *parts.2, &mut activation));
        gates.land_staged(device, packed, (1, 0));
    }
    // The first gate part landed in phase O and the first up half with phase G's first input.
    let norm = (&residual, (tail_parts.gain, tail_parts.quantize), last);
    // Three blocks or more: the last phase O block's norm runs inside phase G's first block.
    for _ in 0..three {
        phase_g::<true, true, false, false, true, _>(
            device,
            (&mut gates, scales, packed),
            (chip, &mut *blocked, &mut product, &mut panels),
            (&mut activation, &mut *prefetch, &mut resident),
            norm,
            (0, 0),
        );
    }
    for _ in 0..(blocks - 1).min(1) - three {
        phase_g::<true, true, false, false, false, _>(
            device,
            (&mut gates, scales, packed),
            (chip, &mut *blocked, &mut product, &mut panels),
            (&mut activation, &mut *prefetch, &mut resident),
            norm,
            (0, 0),
        );
    }
    // Phase G lands the blocks the down's ring holds (`dpanel::RING`), each one iteration behind.
    let ring = dpanel::RING;
    for k in 1..(blocks - 1).min(ring + 1) {
        phase_g::<true, false, false, false, false, _>(
            device,
            (&mut gates, scales, packed),
            (chip, &mut *blocked, &mut product, &mut panels),
            (&mut activation, &mut *prefetch, &mut resident),
            norm,
            (k, 1),
        );
    }
    for t in 0..blocks - 1 - (blocks - 1).min(ring + 1) {
        phase_g::<true, false, false, false, false, _>(
            device,
            (&mut gates, scales, packed),
            (chip, &mut *blocked, &mut product, &mut panels),
            (&mut activation, &mut *prefetch, &mut resident),
            norm,
            (ring + 1 + t, 0),
        );
    }
    for _ in 0..(blocks - 1).min(1) {
        let lands = (ring + 2 - blocks.min(ring + 2)).min(1);
        phase_g::<false, false, true, true, false, _>(
            device,
            (&mut gates, scales, packed),
            (chip, &mut *blocked, &mut product, &mut panels),
            (&mut activation, &mut *prefetch, &mut resident),
            norm,
            (blocks - 1, lands),
        );
    }
    for _ in 0..1 - (blocks - 1).min(1) {
        phase_g::<false, true, true, true, false, _>(
            device,
            (&mut gates, scales, packed),
            (chip, &mut *blocked, &mut product, &mut panels),
            (&mut activation, &mut *prefetch, &mut resident),
            norm,
            (0, 0),
        );
    }
    // Even block counts contract two blocks' rows a panel (`down_spans`), so a layer issues half
    // as many down contractions; an odd count keeps one block a contraction.
    let down = (&panels, scales, tail_parts);
    down_spans(
        device,
        down,
        (&product, &mut resident, &*residuals, &mut *output),
        (blocks, &mut *next_rows, chip),
    );
}

/// Phase D one block a contraction, then the last block's next-layer norm.
fn down_blocks<P: Prefetch>(
    device: &mut Device,
    prefetch: &mut P,
    (panels, scales, tail_parts): (&dpanel::Staged, &scale::Scales<Chips>, &Tail<'_>),
    (product, residuals, output): (
        &crate::models::nemo::serving::mlp::block::Product,
        &dpanel::Owned,
        &mut dpanel::Owned,
    ),
    (blocks, next_rows, chip): (usize, &mut NextRows, usize),
) {
    let last = blocks - 1;
    // Block 0's product lands outside the carried owners, which the first block's iteration
    // fills in place.
    let mut entry = dpanel::Activation::new();
    entry
        .view_mut()
        .memset(const { f8e4m3::from_f32(0.0) }, &mut device.sub);
    land(device, product, 0, &mut entry);
    // The down's owners are declared here, so their pages are free during the gate and up.
    let mut carried = DownCarried {
        landed: dpanel::Activation::new(),
        first: dpanel::Both::new(),
        second: dpanel::Both::new(),
        sent: [dpanel::Received::new(), dpanel::Received::new()],
        residual: dpanel::Own::new(),
        own: dpanel::Own::new(),
        slots: Slots::new(),
    };
    let register = dpanel::fill(device, panels, 0);
    dpanel::contract_into(device, (&entry, &register), (&mut carried.first, 0));
    let register = dpanel::fill(device, panels, 2);
    dpanel::contract_into(device, (&entry, &register), (&mut carried.first, 1));
    let d = (panels, scales, tail_parts);
    for _ in 0..(blocks - 1).min(1) {
        phase_d::<false, true, true, false, _>(
            device,
            d,
            (product, residuals, &mut *output, (&mut *next_rows, chip)),
            (&mut carried, &entry, &mut *prefetch),
            0,
        );
    }
    for k in 1..blocks - 1 {
        phase_d::<true, true, false, false, _>(
            device,
            d,
            (product, residuals, &mut *output, (&mut *next_rows, chip)),
            (&mut carried, &entry, &mut *prefetch),
            k,
        );
    }
    for _ in 0..(blocks - 1).min(1) {
        phase_d::<true, false, false, false, _>(
            device,
            d,
            (product, residuals, &mut *output, (&mut *next_rows, chip)),
            (&mut carried, &entry, &mut *prefetch),
            blocks - 1,
        );
    }
    for _ in 0..1 - (blocks - 1).min(1) {
        phase_d::<false, false, true, false, _>(
            device,
            d,
            (product, residuals, &mut *output, (&mut *next_rows, chip)),
            (&mut carried, &entry, &mut *prefetch),
            0,
        );
    }
    // Block 0's next-layer half crosses the link behind the last reduce-scatter and the last
    // block's norm runs while it does; every later block's half crosses in the next layer's Q/K/V
    // chunk (`qkv::tile::chunk_exchanged`).
    for _ in 0..last.min(1) {
        let peer = send_own(device, &*next_rows, (0, chip));
        normalize::<4>(
            device,
            &carried.own,
            (tail_parts.next.0.weight(), tail_parts.next.1),
            &mut carried.slots,
        );
        peer.place_half(device, (&mut *next_rows, 0, chip));
        hold_own(device, &carried.slots, (&mut *next_rows, last, chip));
    }
    for _ in 0..1 - last.min(1) {
        normalize::<4>(
            device,
            &carried.own,
            (tail_parts.next.0.weight(), tail_parts.next.1),
            &mut carried.slots,
        );
        hold_own(device, &carried.slots, (&mut *next_rows, 0, chip));
        let peer = send_own(device, &*next_rows, (0, chip));
        peer.place_half(device, (&mut *next_rows, 0, chip));
    }
}

/// A block of rows with both hidden halves in the element: half `Hidden / 2560` of each row
/// beside the other, in the own-half layout's slices.
type BothHalves = DmTensor<
    bf16,
    Chips,
    Cluster,
    Reduced,
    m![Hidden / 2560, Hidden % 2560 / 1280, { Time }, { Packet }],
>;
/// A block's own half (slot 0) and the peer's (slot 1).
type Parts =
    DmTensor<bf16, Chips, Cluster, Reduced, m![Slot, Hidden % 2560 / 1280, { Time }, { Packet }]>;
type Own16 = m![Hidden % 2560 / 1280, { Time }, { Packet }];

/// The `blocks` 256-row blocks of replicated hidden rows (both halves on both chips) from wave
/// `wave` as each chip's own half.
pub(crate) fn own(
    device: &mut Device,
    hidden: &HiddenRows,
    (owned, wave, blocks): (&mut dpanel::Owned, usize, usize),
) {
    for block in 0..blocks {
        let rows = hidden
            .view()
            .tile::<m![Waves], 4, m![Waves = 4 # Waves, { crate::models::nemo::serving::Blocked }]>(
                wave + 4 * block,
            );
        // SAFETY: four consecutive waves are one 256-row block; only the wave and row labels change.
        let rows = unsafe {
            rows.reshape::<Chips, m![Block / 64 # Waves, Hidden / 80, Block % 64, Hidden % 80]>()
        };
        let both: BothHalves = rows.to_dm(&mut device.tdma);
        let own: dpanel::Own = both
            .view()
            .chip_slice::<m![Hidden / 2560], Own16>([0, 1])
            .to_dm(&mut device.tdma);
        dpanel::store(device, &own, (&mut *owned, block));
    }
}

/// The `blocks` blocks of each chip's own half joined back into replicated hidden rows from wave
/// `wave`.
pub(crate) fn join(
    device: &mut Device,
    owned: &dpanel::Owned,
    (hidden, wave, blocks): (&mut HiddenRows, usize, usize),
) {
    for block in 0..blocks {
        let mut pair = Parts::new();
        owned
            .view()
            .tile::<m![Blocks], 1, m![Blocks = 1 # Blocks, Hidden % 2560 / 80, Block, Hidden % 80]>(
                block,
            )
            .to_dm_view(
                &mut device.tdma,
                pair.view_mut()
                    .tile::<m![Slot], 1, m![Slot = 1 #{!} 2, { Own16 }]>(0),
            );
        let peer: DmTensor<bf16, Chips, Cluster, Reduced, m![Slot = 1 # 2, { Own16 }]> = pair
            .view()
            .tile::<m![Slot], 1, m![Slot = 1 # 2, { Own16 }]>(0)
            .chip_shuffle([1, 0])
            .to_dm(&mut device.tdma);
        peer.view().to_dm_view(
            &mut device.tdma,
            pair.view_mut()
                .tile::<m![Slot], 1, m![Slot = 1 #{!} 2, { Own16 }]>(1),
        );
        let mut both = BothHalves::new();
        pair.view()
            .chip_slice::<m![Slot], Own16>([0, 1])
            .to_dm_view(
                &mut device.tdma,
                both.view_mut()
                    .tile::<m![Hidden / 2560], 1, m![Hidden / 2560 = 1 #{!} 2, { Own16 }]>(0),
            );
        pair.view()
            .chip_slice::<m![Slot], Own16>([1, 0])
            .to_dm_view(
                &mut device.tdma,
                both.view_mut()
                    .tile::<m![Hidden / 2560], 1, m![Hidden / 2560 = 1 #{!} 2, { Own16 }]>(1),
            );
        let rows = hidden
            .view_mut()
            .tile::<m![Waves], 4, m![Waves = 4 #{!} Waves, { crate::models::nemo::serving::Blocked }]>(wave + 4 * block);
        // SAFETY: as in `own`.
        let rows = unsafe {
            rows.reshape::<Chips, m![Block / 64 #{!} Waves, Hidden / 80, Block % 64, Hidden % 80]>()
        };
        both.view().to_hbm_view(&mut device.tdma, rows);
    }
}

/// What phase D carries from one span (two blocks) to the next: each panel's partials (own place
/// `m`, peer place `m`, the peer's received), the blocks' residuals and outputs, and the norm sums.
struct SpanCarried {
    own: [dpanel::Spanned; 2],
    peer: [dpanel::Spanned; 2],
    received: [dpanel::Spanned; 2],
    residual: [dpanel::Own; 2],
    output: [dpanel::Own; 2],
    /// The span before's chip row sums and the peer's, crossing into the next span.
    total: [RowSums; 2],
    crossed: [RowSums; 2],
}

/// The squares of `own` gathered for the cross-cluster reduction.
fn gathered(
    device: &mut Device,
    (norm, own): (&PostNorm, &dpanel::Own),
) -> crate::ops::normalization::staging::Gathered<
    DmTensor<
        f32,
        Chips,
        Cluster,
        m![Copy640, Block / 8, Copy320],
        m![{ GatherFeatures }, Block % 8, 1 # 8],
    >,
> {
    let input = Dm::<NormSpace, _> {
        value: own.view(),
        tiles: Tiles::over(Block::SIZE),
    };
    norm.statistics::<NormSpace, m![Block % 8], Groups, Features, _>(device, &input)
        .gather::<m![Copy640, Block / 8, Copy320], GatherFeatures>(device)
        .into_value()
}

type GatherFeatures = m![Hidden % 640 / 320, Hidden % 320 / 80];
type GatherStage = crate::ops::normalization::staging::Gathered<
    DmTensor<
        f32,
        Chips,
        Cluster,
        m![Copy640, Block / 8, Copy320],
        m![{ GatherFeatures }, Block % 8, 1 # 8],
    >,
>;
type ClusterStage = crate::ops::normalization::staging::Exchange<
    RowSums,
    crate::ops::normalization::staging::Cluster,
>;

/// A cluster's row sums of `gathered` and the other cluster's, crossing.
fn cluster_sums(
    device: &mut Device,
    (norm, own, gathered): (&PostNorm, &dpanel::Own, GatherStage),
) -> ClusterStage {
    let input = Dm::<NormSpace, _> {
        value: own.view(),
        tiles: Tiles::over(Block::SIZE),
    };
    gathered
        .resume(norm, &input)
        .reduce(device)
        .clusters(device)
        .into_value()
}

/// A chip's row sums of both clusters and the peer chip's, crossing.
fn chip_sums(device: &mut Device, (local, other): &(RowSums, RowSums)) -> (RowSums, RowSums) {
    let total = sum(device, (local, other));
    let peer = total
        .chip_shuffle([1, 0])
        .to_dm::<m![Block % 8, 1 # 8]>(&mut device.tdma);
    (total, peer)
}

/// Phase D of span `j` (blocks `2 j`, `2 j + 1`, in slots `2 slot`, `2 slot + 1` of `resident`),
/// each panel one contraction over both blocks' rows: the two peer-half panels first, each sent
/// once contracted, so the link runs under the own-half panels and the next span's first. The
/// span before (when `PREVIOUS`) is finished and stored under the first contraction, once both
/// its sends have landed, and its next-layer norm runs one exchange per contraction; the norm
/// ends in the next span (`final_norms`), so each exchange waits behind one send at most.
fn phase_span<const PREVIOUS: bool, const FINAL: bool>(
    device: &mut Device,
    (panels, scales, tail): (&dpanel::Staged, &scale::Scales<Chips>, &Tail<'_>),
    (resident, residuals, output, (next_rows, rank)): (
        &dpanel::Resident,
        &dpanel::Owned,
        &mut dpanel::Owned,
        (&mut NextRows, usize),
    ),
    carried: &mut SpanCarried,
    (j, slot): (usize, usize),
) {
    let register = dpanel::fill(device, panels, 2);
    dpanel::contract_span(
        device,
        (dpanel::span_of(resident, slot), &register),
        &mut carried.peer[0],
    );
    if FINAL {
        final_norms(
            device,
            (tail.next.0, tail.next.1),
            (&*carried, &mut *next_rows, rank),
            j - 2,
        );
    }
    let mut squared = None;
    if PREVIOUS {
        finish_span(device, scales, (&mut *carried, &mut *output), j - 1);
        squared = Some([
            gathered(device, (tail.next.0, &carried.output[0])),
            gathered(device, (tail.next.0, &carried.output[1])),
        ]);
    }
    dpanel::send_span(device, &carried.peer[0], &mut carried.received[0]);
    #[unroll]
    for p in 0..2 {
        residuals
            .view()
            .tile::<m![Blocks], 1, m![Blocks = 1 # Blocks, Hidden % 2560 / 80, Block, Hidden % 80]>(
                2 * j + p,
            )
            .to_dm_view(&mut device.tdma, carried.residual[p].view_mut());
    }
    let register3 = dpanel::fill(device, panels, 3);
    dpanel::contract_span(
        device,
        (dpanel::span_of(resident, slot), &register3),
        &mut carried.peer[1],
    );
    let mut local = None;
    if let Some([first, second]) = squared {
        local = Some([
            cluster_sums(device, (tail.next.0, &carried.output[0], first)),
            cluster_sums(device, (tail.next.0, &carried.output[1], second)),
        ]);
    }
    dpanel::send_span(device, &carried.peer[1], &mut carried.received[1]);
    let register0 = dpanel::fill(device, panels, 0);
    dpanel::contract_span(
        device,
        (dpanel::span_of(resident, slot), &register0),
        &mut carried.own[0],
    );
    if let Some(local) = &local {
        #[unroll]
        for p in 0..2 {
            local[p].chips_into(device, (&mut carried.total[p], &mut carried.crossed[p]));
        }
    }
    let register1 = dpanel::fill(device, panels, 1);
    dpanel::contract_span(
        device,
        (dpanel::span_of(resident, slot), &register1),
        &mut carried.own[1],
    );
}

/// The end of span `j`'s next-layer norms: both chips' row sums of each block, its rows (the
/// carried outputs) normalized and held.
fn final_norms(
    device: &mut Device,
    gains: (&PostNorm, &scale::InputScale<Chips, Cluster, Reduced>),
    (carried, rows, rank): (&SpanCarried, &mut NextRows, usize),
    j: usize,
) {
    #[unroll]
    for p in 0..2 {
        let input = Dm::<NormSpace, _> {
            value: carried.output[p].view(),
            tiles: Tiles::over(Block::SIZE),
        };
        let total = gains
            .0
            .total(device, (&input, (&carried.total[p], &carried.crossed[p])));
        let mut slots = Slots::new();
        total.complete::<Groups, Features>(device).value
            .vector_fp_div(gains.1.input_scale())
            .vector_widen_concat::<m![Hidden % 2560 / 1280, Block % 8, Hidden % 80 / 8], m![Hidden % 8]>()
            .vector_final()
            .cast::<f8e4m3, m![Hidden % 8 # 32]>()
            .commit_trim::<m![Hidden % 8]>()
            .commit_view(slots.view_mut().tile::<m![Slot], 1, m![Slot = 1 #{!} 2, Hidden % 2560 / 1280, { Time }, { Packet }]>(0));
        hold_own(device, &slots, (&mut *rows, 2 * j + p, rank));
    }
}

/// Span `j`'s finishes (each own place plus the peer's received partial) of both blocks, stored.
fn finish_span(
    device: &mut Device,
    scales: &scale::Scales<Chips>,
    (carried, output): (&mut SpanCarried, &mut dpanel::Owned),
    j: usize,
) {
    #[unroll]
    for p in 0..2 {
        #[unroll]
        for m in 0..2 {
            dpanel::finish_span(
                device,
                ((&carried.own[m], &carried.received[m], p), scales),
                (&carried.residual[p], &mut carried.output[p], m),
            );
        }
        dpanel::store(device, &carried.output[p], (&mut *output, 2 * j + p));
    }
}

/// Phase D two blocks a contraction, then the last span's next-layer norms. An odd count's last
/// span pairs its last block with the unused block after it, whose rows nothing reads.
fn down_spans(
    device: &mut Device,
    (panels, scales, tail_parts): (&dpanel::Staged, &scale::Scales<Chips>, &Tail<'_>),
    (product, resident, residuals, output): (
        &crate::models::nemo::serving::mlp::block::Product,
        &mut dpanel::Resident,
        &dpanel::Owned,
        &mut dpanel::Owned,
    ),
    (blocks, next_rows, chip): (usize, &mut NextRows, usize),
) {
    let spans = (blocks + 1) / 2;
    let ring = dpanel::RING;
    // The last block's product, stored by phase G's last iteration, lands under span 0 (two spans
    // or more) when the ring holds it.
    for _ in 0..(ring + 1 - blocks.min(ring + 1)).min(1) {
        dpanel::land_resident(device, product, (blocks - 1, blocks - 1), resident);
    }
    let mut carried = SpanCarried {
        own: [dpanel::Spanned::new(), dpanel::Spanned::new()],
        peer: [dpanel::Spanned::new(), dpanel::Spanned::new()],
        received: [dpanel::Spanned::new(), dpanel::Spanned::new()],
        residual: [dpanel::Own::new(), dpanel::Own::new()],
        output: [dpanel::Own::new(), dpanel::Own::new()],
        total: [RowSums::new(), RowSums::new()],
        crossed: [RowSums::new(), RowSums::new()],
    };
    let d = (panels, scales, tail_parts);
    phase_span::<false, false>(
        device,
        d,
        (&*resident, residuals, &mut *output, (&mut *next_rows, chip)),
        &mut carried,
        (0, 0),
    );
    for _ in 0..(spans - 1).min(1) {
        phase_span::<true, false>(
            device,
            d,
            (&*resident, residuals, &mut *output, (&mut *next_rows, chip)),
            &mut carried,
            (1, 1),
        );
    }
    for j in 2..spans.min(ring / 2) {
        phase_span::<true, true>(
            device,
            d,
            (&*resident, residuals, &mut *output, (&mut *next_rows, chip)),
            &mut carried,
            (j, j),
        );
    }
    // More blocks than the ring holds: the rest land once the ring's spans have read it.
    for _ in 0..(blocks - blocks.min(ring)).min(1) {
        for b in 0..blocks - blocks.min(ring) {
            dpanel::land_resident(device, product, (ring + b, b), resident);
        }
        for i in 0..spans - spans.min(ring / 2) {
            phase_span::<true, true>(
                device,
                d,
                (&*resident, residuals, &mut *output, (&mut *next_rows, chip)),
                &mut carried,
                (ring / 2 + i, i),
            );
        }
    }
    for _ in 0..(spans - 1).min(1) {
        final_norms(
            device,
            (tail_parts.next.0, tail_parts.next.1),
            (&carried, &mut *next_rows, chip),
            spans - 2,
        );
    }
    finish_span(device, scales, (&mut carried, &mut *output), spans - 1);
    // Block 0's next-layer half crosses the link while the last span's norms run; with one span
    // block 0 is that span's first, held before its send.
    for _ in 0..(spans - 1).min(1) {
        let peer = send_own(device, &*next_rows, (0, chip));
        span_norms(
            device,
            (&carried.output, (tail_parts.next.0, tail_parts.next.1)),
            (&mut *next_rows, 2 * spans - 2, chip),
        );
        peer.place_half(device, (&mut *next_rows, 0, chip));
    }
    for _ in 0..1 - (spans - 1).min(1) {
        span_norms(
            device,
            (&carried.output, (tail_parts.next.0, tail_parts.next.1)),
            (&mut *next_rows, 0, chip),
        );
        let peer = send_own(device, &*next_rows, (0, chip));
        peer.place_half(device, (&mut *next_rows, 0, chip));
    }
}

/// The next-layer norms of a span's two blocks (`outputs`, blocks `first` and `first + 1`) step by
/// step together, so the two chains share each transfer's wait, held into `rows`.
fn span_norms(
    device: &mut Device,
    (outputs, gains): (
        &[dpanel::Own; 2],
        (&PostNorm, &scale::InputScale<Chips, Cluster, Reduced>),
    ),
    (rows, first, chip): (&mut NextRows, usize, usize),
) {
    let [a, b] = [
        gathered(device, (gains.0, &outputs[0])),
        gathered(device, (gains.0, &outputs[1])),
    ];
    let local = [
        cluster_sums(device, (gains.0, &outputs[0], a)),
        cluster_sums(device, (gains.0, &outputs[1], b)),
    ];
    let inputs = [
        Dm::<NormSpace, _> {
            value: outputs[0].view(),
            tiles: Tiles::over(Block::SIZE),
        },
        Dm::<NormSpace, _> {
            value: outputs[1].view(),
            tiles: Tiles::over(Block::SIZE),
        },
    ];
    let [a, b] = local;
    let chips = [
        a.resume(gains.0, &inputs[0])
            .combine(device)
            .chips(device)
            .into_value(),
        b.resume(gains.0, &inputs[1])
            .combine(device)
            .chips(device)
            .into_value(),
    ];
    #[unroll]
    for p in 0..2 {
        let total = gains.0.total(device, (&inputs[p], chips[p].values()));
        let mut slots = Slots::new();
        total.complete::<Groups, Features>(device).value
            .vector_fp_div(gains.1.input_scale())
            .vector_widen_concat::<m![Hidden % 2560 / 1280, Block % 8, Hidden % 80 / 8], m![Hidden % 8]>()
            .vector_final()
            .cast::<f8e4m3, m![Hidden % 8 # 32]>()
            .commit_trim::<m![Hidden % 8]>()
            .commit_view(slots.view_mut().tile::<m![Slot], 1, m![Slot = 1 #{!} 2, Hidden % 2560 / 1280, { Time }, { Packet }]>(0));
        hold_own(device, &slots, (&mut *rows, first + p, chip));
    }
}
