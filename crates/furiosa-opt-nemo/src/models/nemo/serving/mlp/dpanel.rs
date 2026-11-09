//! A prompt block's down projection with the layer output D-sharded: chip `c` keeps hidden half
//! `c` (`Hidden / 2560`). PMLP's weight-stationary panels (`panel.rs`, 640 outputs by 64 K a
//! slice, the block's product streamed), with each cluster owning the panels of one parity
//! (`Hidden % 1280 / 640`) of both halves, so the panels at the same place of the two halves are
//! one pair whose half is an element digit: the reduce-scatter sends the peer's half of a pair
//! and keeps its own (`chip_slice` on `Hidden / 2560`), the only per-chip choice.
use crate::Module;
use furiosa_opt_std::prelude::*;

use super::block::{Block, Blocks, Product};
use super::{Hidden, Intermediate, Waves};
use crate::models::nemo::serving::{Chips, scale, stack, stack::Depth};

axes![Wide = 8192, Term = 2, Span = 2, Ring = 8];
/// Cluster `k` owns the 640-wide panels of parity `k` of every 1,280 features.
pub type Cluster = m![Hidden % 1280 / 640];
/// Two output groups of 320 by the 128 K groups of 64 (the 7,168 features padded to 8,192).
type Slice = m![
    Hidden % 640 / 320,
    Intermediate # 8192 / 4096,
    Intermediate # 8192 % 4096 / 2048,
    Intermediate # 8192 % 2048 / 256,
    Intermediate # 8192 % 256 / 64
];
/// `Raw` under the contraction's K axis.
type Landed = DmTensor<
    f8e4m3,
    Chips,
    Cluster,
    m![
        Hidden % 640 / 320,
        Intermediate # 8192 / 4096,
        Intermediate # 8192 % 4096 / 2048,
        Intermediate # 8192 % 2048 / 256,
        Hidden % 320 / 80
    ],
    m![Hidden % 80, Intermediate # 8192 % 256],
>;
type Lanes = m![Hidden % 8];
pub type Register = TrfTensor<
    f8e4m3,
    Chips,
    Cluster,
    Slice,
    Lanes,
    m![Hidden % 80 / 8, Hidden % 320 / 80, Intermediate # 8192 % 64],
>;
pub type Reduced = m![Hidden % 640 / 320, Block / 8, Hidden % 320 / 80];
pub type Time = m![Block % 8, Hidden % 80 / 8];
pub type Packet = m![Hidden % 8];
axes![Wave = 8, BothRows = 16];
type Space = crate::compose::Tile<Block, Wave>;
type SpanSpace = crate::compose::Tile<Block, BothRows>;

macro_rules! mapping {
    ($space:ty, $input:ty, $feed:ty, $flits:ty, $steps:ty, $output:ty) => {
        impl crate::ops::linear::contraction::Mapping for $space {
            type Cluster = Cluster;
            type Slices = Slice;
            type Reduced = Reduced;
            type Input = $input;
            type Lanes = Lanes;
            type Weight = m![Hidden % 80 / 8, Hidden % 320 / 80, Intermediate # 8192 % 64];
            type Feed = $feed;
            type Packet = m![Intermediate # 8192 % 64];
            type Flits = $flits;
            type Flit = m![Intermediate # 8192 % 32];
            type Steps = $steps;
            type Contracted = m![1];
            type Sums = $steps;
            type Groups = $steps;
            type Output = $output;
        }
    };
}

mapping!(
    Space,
    m![Intermediate # 8192 % 64 / 8, Block, Intermediate # 8192 % 8],
    m![Block],
    m![Block, Intermediate # 8192 % 64 / 32],
    m![Block, Hidden % 80 / 8, Hidden % 320 / 80],
    Time
);
mapping!(
    SpanSpace,
    SpanRows,
    m![Span, Block],
    m![Span, Block, Intermediate # 8192 % 64 / 32],
    m![Span, Block, Hidden % 80 / 8, Hidden % 320 / 80],
    m![Span, { Time }]
);
/// A block of a chip's own half: its two panels a cluster (`Hidden % 2560 / 1280`).
pub type Own =
    DmTensor<bf16, Chips, Cluster, Reduced, m![Hidden % 2560 / 1280, { Time }, { Packet }]>;
/// Every block of a chip's own half, each 80-feature group's 256 rows a 40-KiB run.
pub type Owned = HbmTensor<bf16, Chips, m![Blocks, Hidden % 2560 / 80, Block, Hidden % 80]>;

/// The layer's four down panels of a cluster as landed, once a layer: panel `q = Hidden / 1280`.
pub struct Staged([Landed; 4]);

impl Staged {
    /// Chip `rank`'s relative panel `panel` (its own half first) of layer `index`'s down as
    /// stored (`down::Stored`, absolute hidden order) into its buffer, one DMA.
    pub fn land_packed(
        &mut self,
        device: &mut Device,
        (layers, index): (&stack::Layers, usize),
        (panel, rank): (usize, usize),
    ) {
        let absolute = if panel < 2 {
            panel + 2 * rank
        } else {
            panel - 2 * rank
        };
        layers
            .down
            .at(index)
            .tile::<m![Hidden / 1280], 1, m![
                Depth = 1 # Depth,
                Hidden / 1280 = 1 # 4,
                Hidden % 1280 / 640,
                Hidden % 640 / 320,
                Intermediate # 8192 / 4096,
                Intermediate # 8192 % 4096 / 2048,
                Intermediate # 8192 % 2048 / 256,
                Hidden % 320 / 80,
                Hidden % 80,
                Intermediate # 8192 % 256
            ]>(absolute)
            .to_dm_view(&mut device.tdma, self.0[panel].view_mut());
    }

    /// Four panels' buffers, each landed on its own (`land_half`).
    pub fn new() -> Self {
        Self([Landed::new(), Landed::new(), Landed::new(), Landed::new()])
    }
}

/// Panel `panel` in the register file: the switch moves each 64-feature K group of the landed
/// 256-feature runs to its slice.
pub fn fill(device: &mut Device, staged: &Staged, panel: usize) -> Register {
    device
        .sub
        .begin(staged.0[panel].view())
        .fetch::<m![Hidden % 8, Hidden % 80 / 8, Intermediate # 8192 % 256 / 64], m![
            Intermediate # 8192 % 64
        ]>()
        .switch::<Slice, m![Hidden % 8, Hidden % 80 / 8, Hidden % 320 / 80]>(
            SwitchConfig::InterTranspose {
                slice0: 1,
                slice1: 4,
                time0: 1,
            },
        )
        .collect::<m![Hidden % 8, Hidden % 80 / 8, Hidden % 320 / 80, Intermediate # 8192 % 64 / 32], m![
            Intermediate # 8192 % 32
        ]>()
        .to_trf()
}

/// One panel over two blocks, unscaled BF16: a chip's own partial, a peer's partial on its way,
/// or the peer's partial of this chip's half as it lands.
pub type Spanned = DmTensor<bf16, Chips, Cluster, Reduced, m![Span, { Time }, { Packet }]>;
type SpanSlot = m![Span = 1 # 2, { Time }, { Packet }];

/// The product of up to `RING` blocks in both output groups' slices, each landed as soon as the
/// gate and up have stored it, so the down reads its spans in place.
pub type Resident = DmTensor<
    f8e4m3,
    Chips,
    Cluster,
    Slice,
    m![Intermediate # 8192 % 64 / 8, Ring, Block, Intermediate # 8192 % 8],
>;
type SpanRows = m![Intermediate # 8192 % 64 / 8, Span # Ring, Block, Intermediate # 8192 % 8];
pub type SpanView<'a> = DmTensorView<'a, f8e4m3, Chips, Cluster, Slice, SpanRows>;

/// The blocks `Resident` holds; a longer prompt's later blocks reuse its slots.
pub const RING: usize = Ring::SIZE;

/// Block `block` of `product` into slot `slot` of `resident`.
pub fn land_resident(
    device: &mut Device,
    product: &Product,
    (block, slot): (usize, usize),
    resident: &mut Resident,
) {
    type One = m![Intermediate # 8192 % 64 / 8, Ring = 1 #{!} Ring, Block, Intermediate # 8192 % 8];
    let rows = product.view().tile::<m![Blocks], 1, m![
        Blocks = 1 # Blocks,
        Intermediate / 8,
        Block,
        Intermediate % 8
    ]>(block);
    rows.to_dm_view(
        &mut device.tdma,
        resident.view_mut().tile::<m![Ring], 1, One>(slot),
    );
}

/// Slots `2 span` and `2 span + 1` of `resident`, one span.
pub fn span_of(resident: &Resident, span: usize) -> SpanView<'_> {
    let rows = resident.view().tile::<m![Ring], 2, m![
        Intermediate # 8192 % 64 / 8,
        Ring = 2 # Ring,
        Block,
        Intermediate # 8192 % 8
    ]>(2 * span);
    // SAFETY: the two blocks are consecutive; only the block label is renamed.
    unsafe { rows.reshape::<Chips, Cluster, Slice, SpanRows>() }
}

/// One panel over both blocks of a landed span, unscaled, rounded to BF16 into `partial`.
pub fn contract_span(
    device: &mut Device,
    (activation, register): (SpanView<'_>, &Register),
    partial: &mut Spanned,
) {
    crate::ops::linear::Affine::new(register, ())
        .forward((
            device,
            crate::ops::layout::Dm::<SpanSpace, _> {
                tiles: crate::compose::Tiles::over(Block::SIZE),
                value: (activation,),
            },
        ))
        .value
        .cast::<bf16, m![{ Packet } # 16]>()
        .commit_trim::<Packet>()
        .commit_view(partial.view_mut())
}

/// A peer-half panel's partial over the link into the peer's `received`.
pub fn send_span(device: &mut Device, partial: &Spanned, received: &mut Spanned) {
    crate::ops::collective::AllReduce::<crate::ops::layout::axes::Parallel<2>>::default().stage(
        device,
        crate::ops::layout::Dm::<SpanSpace, _> {
            tiles: crate::compose::Tiles::over(Block::SIZE),
            value: (partial.view(), received.view_mut()),
        },
    );
}

/// `finish_slot` of block `p` of a span: this chip's partial `own` plus the peer's `received`.
pub fn finish_span(
    device: &mut Device,
    ((own, received, p), scales): ((&Spanned, &Spanned, usize), &scale::Scales<Chips>),
    (residual, output, m): (&Own, &mut Own, usize),
) {
    let own = own.view().tile::<m![Span], 1, SpanSlot>(p);
    let received = received.view().tile::<m![Span], 1, SpanSlot>(p);
    finish_views(device, ((own, received), scales), (residual, output, m));
}

/// Both chips' partials of this chip's half (`slot` and `received`), scaled, rounded to BF16,
/// plus `residual` rounded again, into panel `m` of `output`.
fn finish_views<E: M>(
    device: &mut Device,
    ((slot, received), scales): (
        (
            DmTensorView<'_, bf16, Chips, Cluster, Reduced, E>,
            DmTensorView<'_, bf16, Chips, Cluster, Reduced, E>,
        ),
        &scale::Scales<Chips>,
    ),
    (residual, output, m): (&Own, &mut Own, usize),
) {
    let scales = scale::Epilogue {
        input: scales.get::<Cluster, Reduced>(device, 4),
        weight: scales.get::<Cluster, Reduced>(device, 5),
    };
    let mut sum = Own::new();
    let value =
        crate::ops::collective::AllReduce::<crate::ops::layout::axes::Parallel<2>>::default()
            .sum::<Space, Chips, Cluster, Reduced, _, Time, Packet>(
                device,
                crate::ops::layout::Dm::<Space, _> {
                    tiles: crate::compose::Tiles::over(Block::SIZE),
                    value: (slot, received),
                },
            )
            .value;
    crate::ops::linear::Affine::new((), &scales)
        .forward(crate::ops::layout::Dm::<Space, _> {
            tiles: crate::compose::Tiles::over(Block::SIZE),
            value,
        })
        .value
        .vector_widen_concat::<Time, Packet>()
        .vector_final()
        .cast::<bf16, m![{ Packet } # 16]>()
        .commit_trim::<Packet>()
        .commit_view(panel_mut(&mut sum, m));
    crate::ops::add::Add
        .stage::<Space, Chips, Cluster, Reduced, _, Time, Packet>(
            device,
            crate::ops::layout::Dm::<Space, _> {
                tiles: crate::compose::Tiles::over(Block::SIZE),
                value: (panel(&sum, m), panel(residual, m)),
            },
        )
        .value
        .commit_view(panel_mut(output, m))
}

/// Panel `m` of `a` plus panel `m` of `b`, rounded once to BF16 into panel `m` of `output`.
pub fn add(device: &mut Device, (a, b): (&Own, &Own), (output, m): (&mut Own, usize)) {
    crate::ops::add::Add
        .stage::<Space, Chips, Cluster, Reduced, _, Time, Packet>(
            device,
            crate::ops::layout::Dm::<Space, _> {
                tiles: crate::compose::Tiles::over(Block::SIZE),
                value: (panel(a, m), panel(b, m)),
            },
        )
        .value
        .commit_view(panel_mut(output, m))
}

type Panel = m![Hidden % 2560 / 1280 = 1 # 2, { Time }, { Packet }];

/// Panel `m` of a block of a chip's own half.
fn panel(own: &Own, m: usize) -> DmTensorView<'_, bf16, Chips, Cluster, Reduced, Panel> {
    own.view().tile::<m![Hidden % 2560 / 1280], 1, Panel>(m)
}

fn panel_mut(
    own: &mut Own,
    m: usize,
) -> DmTensorViewMut<
    '_,
    bf16,
    Chips,
    Cluster,
    Reduced,
    m![Hidden % 2560 / 1280 = 1 #{!} 2, { Time }, { Packet }],
> {
    own.view_mut()
        .tile::<m![Hidden % 2560 / 1280], 1, m![Hidden % 2560 / 1280 = 1 #{!} 2, { Time }, { Packet }]>(m)
}

/// Block `block` of a chip's own half from `owned`.
pub fn load(device: &mut Device, owned: &Owned, block: usize) -> Own {
    owned
        .view()
        .tile::<m![Blocks], 1, m![Blocks = 1 # Blocks, Hidden % 2560 / 80, Block, Hidden % 80]>(
            block,
        )
        .to_dm(&mut device.tdma)
}

/// `own` into block `block` of `owned`.
pub fn store(device: &mut Device, own: &Own, (owned, block): (&mut Owned, usize)) {
    own.view().to_hbm_view(
        &mut device.tdma,
        owned.view_mut().tile::<m![Blocks], 1, m![
            Blocks = 1 #{!} Blocks,
            Hidden % 2560 / 80,
            Block,
            Hidden % 80
        ]>(block),
    );
}

const _: () = assert!(Waves::SIZE == 4 * Blocks::SIZE && Hidden::SIZE == 5120);
