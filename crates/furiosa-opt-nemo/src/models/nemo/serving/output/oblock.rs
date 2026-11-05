//! The O projection of a 256-row block in the gate/up's geometry (`mlp::block`), for the
//! D-sharded layer (`step::dlayer`): the block's FP8 heads of both chips stay in the register file
//! (256 rows of one 64-feature K group a slice) while the chip's own hidden half streams past them
//! once, and the inter-slice reduction of the 64 K groups leaves each output in the block layout
//! the feed-forward's panels write (`mlp::dpanel::Own`), after one in-slice transpose and one
//! transfer: the reduction leaves the four kept output groups outermost.
//! - One command a block: 10,240 DPE steps, the oracle's count for its 256-row O tile.
//! - K part `p` holds chip `1 - p`'s heads, so a chip's weight is one window of the O copy both
//!   routes read (`output::Stored`).
use crate::Module;
use furiosa_opt_std::prelude::*;

use crate::models::nemo::serving::mlp::block::{Block, Blocks};
use crate::models::nemo::serving::mlp::dpanel;
use crate::models::nemo::serving::stack::{self, Depth};
use crate::models::nemo::serving::{Chips, Hidden, Rows, Width, scale};

axes![Part = 2];
pub type Cluster = dpanel::Cluster;
/// Four 320-output groups by the 64 K groups of 64 (the own heads' 32 groups, then the peer's):
/// the reduced K digit is innermost, where the reduction puts the block's row groups.
pub type Slice = m![Hidden % 640 / 320, Hidden % 320 / 160, Part, Width / 64];
/// The reduction's output: the kept output groups, then the row groups and the scattered output
/// digit.
pub type Reduced = m![
    Hidden % 640 / 320,
    Hidden % 320 / 160,
    Block / 8,
    Hidden % 160 / 80
];
/// A slice's 320 outputs by its 64 K features, streamed by Main fetch.
type WeightElement = m![
    Hidden % 2560 / 1280,
    Hidden % 160 / 80,
    Hidden % 80 / 8,
    Width % 64 / 8,
    Hidden % 8,
    Width % 8
];
pub type Weight = DmTensor<f8e4m3, Chips, Cluster, Slice, WeightElement>;
use crate::models::nemo::serving::output::Slot;
/// One hidden half's 2,560 outputs by one chip's 2,048 head features in `Weight`'s order.
type Half = m![
    { Cluster },
    Hidden % 640 / 320,
    Hidden % 320 / 160,
    Width / 64,
    { WeightElement }
];
/// Every layer's O weight on a chip, the one copy both routes read: slot `h - c + 1` holds hidden
/// half `h` by chip `c`'s heads, for the three pairs the chip reads. This route reads slots
/// `rank ..`: its own half by chip 1's heads (part 0), then chip 0's (part 1). Decode reads slots
/// `1 - rank ..`: its own heads by both halves.
pub type Stored = m![Slot, { Half }];
/// A block's FP8 heads of both chips in each output group's slices: 256 rows of 64 features.
pub type Heads = DmTensor<f8e4m3, Chips, Cluster, Slice, m![Block, Width % 64]>;
type Lanes = m![Block % 8];
pub type Register = TrfTensor<
    f8e4m3,
    Chips,
    Cluster,
    Slice,
    Lanes,
    m![Width % 64 / 16, Block % 32 / 8, Block / 32, Width % 16],
>;
type Feed = m![
    Hidden % 80 / 4,
    Hidden % 2560 / 1280,
    Hidden % 160 / 80,
    Width % 64 / 16
];
type Flit = m![Hidden % 4, Width % 16];
type Steps = m![{ Feed }, Block % 32 / 8, Block / 32];
type Sums = m![
    Hidden % 80 / 4,
    Hidden % 2560 / 1280,
    Hidden % 160 / 80,
    Block % 32 / 8,
    Block / 32
];
type Time = m![Hidden % 80 / 4, Hidden % 2560 / 1280, Hidden % 4];
type Packet = m![Block % 8];
/// The projection as the reduction leaves it: a slice's outputs, eight rows a packet.
pub type Output = DmTensor<bf16, Chips, Cluster, Reduced, m![{ Time }, { Packet }]>;
type Space = crate::compose::Tile<Block, Block>;

impl crate::ops::linear::contraction::Mapping for Space {
    type Cluster = Cluster;
    type Slices = Slice;
    type Reduced = Reduced;
    type Input = m![Width % 64 / 16, Block % 32 / 8, Block / 32, Width % 16];
    type Lanes = Lanes;
    type Weight = WeightElement;
    type Feed = Feed;
    type Packet = Flit;
    type Flits = m![{ Feed }, Hidden % 4 / 2];
    type Flit = m![Hidden % 2, Width % 16];
    type Steps = Steps;
    type Contracted = m![Hidden % 4];
    type Sums = Sums;
    type Groups = m![{ Sums }, Hidden % 4];
    type Output = Time;
}
/// One wave's heads of one chip, half the features a cluster: 4 rows of one 64-feature group a
/// slice.
type Halves = m![Width / 1024];
type Landing = m![Width % 1024 / 64, Rows / 4];
type Element = m![Rows % 4, Width % 64];
pub type Rows32 = DmTensor<f32, Chips, Halves, Landing, Element>;
pub type Rows8 = DmTensor<f8e4m3, Chips, Halves, Landing, Element>;
type QuantizeTime = m![Rows % 4, Width % 64 / 8];
type QuantizePacket = m![Width % 8];
/// A whole block's heads of one chip, half the features a cluster: 16 rows of one 64-feature
/// group a slice.
type BlockLanding = m![Width % 1024 / 64, Block / 16];
type BlockElement = m![Block % 16, Width % 64];
pub type Blocked32 = DmTensor<f32, Chips, Halves, BlockLanding, BlockElement>;
pub type Blocked8 = DmTensor<f8e4m3, Chips, Halves, BlockLanding, BlockElement>;
type BlockTime = m![Block % 16, Width % 64 / 8];

const _: () = {
    assert!(Slice::SIZE == 256);
    assert!(Reduced::SIZE == 256);
    assert!(Landing::SIZE == 256);
    assert!(Steps::SIZE == 10240);
    assert!(2 * Stored::SIZE == 3 * Width::SIZE * Hidden::SIZE);
};

/// The `Stored` position of O value (absolute `hidden` output, absolute head `feature`) on every
/// chip that holds it: its hidden half's chip and its heads' chip, the same slot on both.
pub fn places(hidden: usize, feature: usize) -> ([usize; 2], usize) {
    let (half, owner) = (hidden / 2560, feature / 2048);
    let (h, w) = (hidden % 2560, feature % 2048);
    let index = [
        (3, half + 1 - owner),
        (2, h % 1280 / 640),
        (2, h % 640 / 320),
        (2, h % 320 / 160),
        (32, w / 64),
        (2, h / 1280),
        (2, h % 160 / 80),
        (10, h % 80 / 8),
        (8, w % 64 / 8),
        (8, h % 8),
        (8, w % 8),
    ]
    .iter()
    .fold(0, |index, &(size, digit)| index * size + digit);
    ([half, owner], index)
}

/// Packs both chips' TP2 O shards (`output::Stored` of decode before, chip `s` holding its heads'
/// 2,048 features by all 5,120 hidden rows) into each chip's `Stored`.
pub fn pack<T: Copy + Default>(shards: [&[T]; 2]) -> [Vec<T>; 2] {
    // TP2 order: [Hidden / 2560, Hidden % 2560 / 80, Width / 256, Hidden % 8, Width % 256 / 32,
    // Hidden % 80 / 8, Width % 32].
    let place = |index: usize| {
        let (w32, rest) = (index % 32, index / 32);
        let (h80, rest) = (rest % 10, rest / 10);
        let (w256, rest) = (rest % 8, rest / 8);
        let (h8, rest) = (rest % 8, rest / 8);
        let (w, rest) = (rest % 8, rest / 8);
        let (group, half) = (rest % 32, rest / 32);
        (
            2560 * half + 80 * group + 8 * h80 + h8,
            256 * w + 32 * w256 + w32,
        )
    };
    let mut stored = [
        vec![T::default(); Stored::SIZE],
        vec![T::default(); Stored::SIZE],
    ];
    for (rank, shard) in shards.iter().enumerate() {
        for (at, &value) in shard[..Hidden::SIZE * Width::SIZE].iter().enumerate() {
            let (hidden, width) = place(at);
            let (chips, index) = places(hidden, rank * 2048 + width);
            for chip in chips {
                stored[chip][index] = value;
            }
        }
    }
    stored
}

/// Chip `rank`'s window of layer `index`'s O copy (slots `rank ..`) into `landed`, one 20-KiB
/// run a slice.
fn land_parts(
    device: &mut Device,
    (layers, index, rank): (&stack::Layers, usize, usize),
    landed: &mut Weight,
) {
    let window = layers
        .output
        .at(index)
        .tile::<m![Slot], 2, m![Depth = 1 # Depth, Slot = 2 # 3, { Half }]>(rank);
    // SAFETY: slot `rank + p` holds the own half by chip `1 - p`'s heads, part `p`, so naming the
    // window's slot axis by the part moves no data.
    let parts = unsafe { window.reshape::<Chips, m![Depth = 1 # Depth, Part # 3, { Half }]>() };
    parts.to_dm_view(&mut device.tdma, landed.view_mut());
}

/// The O weight of one layer and its scales in data memory.
pub struct Projection {
    weight: Weight,
    scales: scale::Scales<Chips>,
}

/// The O scales in registers: the heads' divisor and the projection's epilogue pair.
pub struct Factors {
    divisor: scale::InputScale<Chips, Halves, Landing>,
    /// `divisor` in a whole block's landing (`Blocked32`).
    block_divisor: scale::InputScale<Chips, Halves, BlockLanding>,
    epilogue: scale::Epilogue<Cluster, Reduced, Chips>,
}

impl Projection {
    /// The weight (one 20-KiB run a slice) and its packed scale row.
    pub fn land<F: M>(
        device: &mut Device,
        (weights, scales): (
            (&stack::Layers, usize, usize),
            HbmTensorView<'_, f32, Chips, F>,
        ),
    ) -> Self {
        let mut weight = Weight::new();
        land_parts(device, weights, &mut weight);
        Self {
            weight,
            scales: scale::Scales::land(device, scales),
        }
    }

    /// The scales alone; `land_weight` lands the weight behind the transfers the heads need first.
    pub fn with_scales<F: M>(
        device: &mut Device,
        scales: HbmTensorView<'_, f32, Chips, F>,
    ) -> Self {
        Self::of(scale::Scales::land(device, scales))
    }

    /// `with_scales` of scales already landed.
    pub fn of(scales: scale::Scales<Chips>) -> Self {
        Self {
            weight: Weight::new(),
            scales,
        }
    }

    pub fn land_weight(&mut self, device: &mut Device, weights: (&stack::Layers, usize, usize)) {
        land_parts(device, weights, &mut self.weight);
    }

    pub fn factors(&self, device: &mut Device) -> Factors {
        Factors {
            divisor: crate::ops::linear::Affine::new((), self.scales.get(device, 0)),
            block_divisor: crate::ops::linear::Affine::new((), self.scales.get(device, 0)),
            epilogue: scale::Epilogue {
                input: self.scales.get(device, 0),
                weight: self.scales.get(device, 1),
            },
        }
    }

    /// A filled block's projection, scaled and rounded once to BF16.
    pub fn project(&self, device: &mut Device, (heads, factors): (&Register, &Factors)) -> Output {
        crate::ops::linear::Affine::new(&self.weight, &factors.epilogue)
            .forward((
                device,
                crate::ops::layout::Dm::<Space, _> {
                    tiles: crate::compose::Tiles::over(Block::SIZE),
                    value: heads,
                },
            ))
            .value
    }
}

/// Wave `wave`'s attended rows of this chip's heads (of a chunk's attended rows), landed.
pub fn land(
    device: &mut Device,
    prompt: &crate::models::nemo::serving::attention::prompt::Attended,
    wave: usize,
) -> Rows32 {
    use crate::models::nemo::serving::attention::{
        Groups, Head, Heads as AttentionHeads, prompt::Prompt,
    };
    let attended = prompt.view().tile::<m![Prompt], 64, m![
        Prompt = 64 # Prompt,
        AttentionHeads,
        Groups,
        Head
    ]>(64 * wave);
    // SAFETY: the 64-row tile keeps the chunk's row stride, and its row-major heads flatten
    // Heads/Groups/Head in wire order.
    let attended = unsafe { attended.reshape::<Chips, m![Rows # Prompt, Width]>() };
    attended.to_dm(&mut device.tdma)
}

/// A wave's landed heads rounded once to BF16 and quantized by the O input scale.
pub fn quantize(device: &mut Device, rows: &Rows32, factors: &Factors) -> Rows8 {
    let rounded: DmTensor<bf16, Chips, Halves, Landing, m![{ QuantizeTime }, { QuantizePacket }]> =
        device
            .main
            .begin(rows.view())
            .fetch::<QuantizeTime, QuantizePacket>()
            .collect::<QuantizeTime, QuantizePacket>()
            .cast::<bf16, m![{ QuantizePacket } # 16]>()
            .commit_trim::<QuantizePacket>()
            .commit();
    let quantized = scale::quantize::<Halves, Landing, QuantizeTime, QuantizePacket, Chips>(
        device,
        &rounded,
        &factors.divisor,
    );
    // SAFETY: `[Rows % 4, Width % 64 / 8, Width % 8]` is the element in wire order.
    unsafe { quantized.reshape() }
}

/// Block `block`'s attended rows of this chip's heads (of a chunk's attended rows), landed.
pub fn land_rows(
    device: &mut Device,
    prompt: &crate::models::nemo::serving::attention::prompt::Attended,
    block: usize,
) -> Blocked32 {
    use crate::models::nemo::serving::attention::{
        Groups, Head, Heads as AttentionHeads, prompt::Prompt,
    };
    let attended = prompt.view().tile::<m![Prompt], 256, m![
        Prompt = 256 # Prompt,
        AttentionHeads,
        Groups,
        Head
    ]>(256 * block);
    // SAFETY: the 256-row tile keeps the chunk's row stride, and its row-major heads flatten
    // Heads/Groups/Head in wire order.
    let attended = unsafe { attended.reshape::<Chips, m![Block # Prompt, Width]>() };
    attended.to_dm(&mut device.tdma)
}

/// A block's landed heads rounded once to BF16 and quantized by the O input scale.
pub fn quantize_rows(device: &mut Device, rows: &Blocked32, factors: &Factors) -> Blocked8 {
    let rounded: DmTensor<
        bf16,
        Chips,
        Halves,
        BlockLanding,
        m![{ BlockTime }, { QuantizePacket }],
    > = device
        .main
        .begin(rows.view())
        .fetch::<BlockTime, QuantizePacket>()
        .collect::<BlockTime, QuantizePacket>()
        .cast::<bf16, m![{ QuantizePacket } # 16]>()
        .commit_trim::<QuantizePacket>()
        .commit();
    let quantized = scale::quantize::<Halves, BlockLanding, BlockTime, QuantizePacket, Chips>(
        device,
        &rounded,
        &factors.block_divisor,
    );
    // SAFETY: `[Block % 16, Width % 64 / 8, Width % 8]` is the element in wire order.
    unsafe { quantized.reshape() }
}

/// A block's quantized heads on their way to the peer chip.
pub fn send_rows(device: &mut Device, rows: &Blocked8) -> Blocked8 {
    rows.chip_shuffle([1, 0])
        .to_dm::<BlockElement>(&mut device.tdma)
}

/// A block's quantized heads into part `part` (0 own, 1 the peer's) of block `block` of
/// `staged`: one 1-KiB run a source slice.
pub fn store_rows(
    device: &mut Device,
    rows: &Blocked8,
    (staged, part, block): (&mut Staged, usize, usize),
) {
    rows.view().to_hbm_view(
        &mut device.tdma,
        staged
            .view_mut()
            .tile::<m![Blocks], 1, m![Blocks = 1 #{!} Blocks, Hidden % 1280 / 640, Part, Width / 64, Block, Width % 64]>(block)
            .tile::<m![Part], 1, m![Blocks = 1 #{!} Blocks, Hidden % 1280 / 640, Part = 1 #{!} 2, Width / 64, Block, Width % 64]>(part),
    );
}

/// A wave's quantized heads on their way to the peer chip.
pub fn send(device: &mut Device, rows: &Rows8) -> Rows8 {
    rows.chip_shuffle([1, 0]).to_dm::<Element>(&mut device.tdma)
}

/// Every block's FP8 heads of both chips in HBM, one copy a cluster, each 64-feature K group's 256
/// rows one 16-KiB run: each cluster's slices read their own copy, four readers a source.
pub type Staged = HbmTensor<
    f8e4m3,
    Chips,
    m![
        Blocks,
        Hidden % 1280 / 640,
        Part,
        Width / 64,
        Block,
        Width % 64
    ],
>;

/// A wave's quantized heads into part `part` (0 own, 1 the peer's) of rows `64 wave ..` of
/// block `block` of `staged`: one 256-byte run a source slice's row.
pub fn store(
    device: &mut Device,
    rows: &Rows8,
    (staged, part, (block, wave)): (&mut Staged, usize, (usize, usize)),
) {
    // SAFETY: a wave is rows `64 wave ..` of its block; only the row label changes.
    let rows = unsafe {
        rows.view()
            .reshape::<Chips, Halves, m![Width % 1024 / 64, Block % 64 / 4], m![Block % 4, Width % 64]>()
    };
    rows.to_hbm_view(
        &mut device.tdma,
        staged
            .view_mut()
            .tile::<m![Blocks], 1, m![Blocks = 1 #{!} Blocks, Hidden % 1280 / 640, Part, Width / 64, Block, Width % 64]>(block)
            .tile::<m![Part], 1, m![Blocks = 1 #{!} Blocks, Hidden % 1280 / 640, Part = 1 #{!} 2, Width / 64, Block, Width % 64]>(part)
            .tile::<m![Block / 64], 1, m![
                Blocks = 1 #{!} Blocks,
                Hidden % 1280 / 640,
                Part = 1 #{!} 2,
                Width / 64,
                Block / 64 = 1 #{!} 4,
                Block % 64,
                Width % 64
            ]>(wave),
    );
}

/// Block `block`'s staged heads into every output group's slices: one 16-KiB run a slice.
pub fn stage(device: &mut Device, (staged, block): (&Staged, usize), heads: &mut Heads) {
    staged
        .view()
        .tile::<m![Blocks], 1, m![Blocks = 1 # Blocks, Hidden % 1280 / 640, Part, Width / 64, Block, Width % 64]>(block)
        .to_dm_view(&mut device.tdma, heads.view_mut());
}

/// A block's staged heads in the register file.
pub fn fill(device: &mut Device, heads: &Heads) -> Register {
    device
        .sub
        .begin(heads.view())
        .fetch::<m![Block % 8, Width % 64 / 16, Block % 32 / 8, Block / 64], m![Block / 32 % 2, Width % 16]>()
        .collect::<m![Block % 8, Width % 64 / 16, Block % 32 / 8, Block / 64], m![Block / 32 % 2, Width % 16]>()
        .to_trf()
}

/// The projection transposed four outputs a packet: each row's outputs in runs of four.
type Quads = DmTensor<
    bf16,
    Chips,
    Cluster,
    Reduced,
    m![Hidden % 2560 / 1280, Hidden % 80 / 4, Block % 8, Hidden % 4],
>;
/// `dpanel::Own`'s element with each row's outputs in runs of four.
type Rowed = DmTensor<
    bf16,
    Chips,
    Cluster,
    Reduced,
    m![Hidden % 2560 / 1280, Block % 8, Hidden % 80 / 4, Hidden % 4],
>;

/// A projected block with each row's outputs innermost, moved into `dpanel::Own`: a BF16
/// transpose takes four rows at most, so four outputs a packet, then each row's runs gathered by
/// the fetch, then one 160-byte run a row and panel into the panels' slices.
pub fn transpose(device: &mut Device, output: &Output) -> dpanel::Own {
    let quads: Quads = device
        .main
        .begin(output.view())
        .fetch::<m![Hidden % 2560 / 1280, Hidden % 80 / 4, Hidden % 4], m![Block % 8 # 16]>()
        .collect::<m![Hidden % 2560 / 1280, Hidden % 80 / 4, Hidden % 4], m![Block % 8 # 16]>()
        .transpose::<m![Hidden % 2560 / 1280, Hidden % 80 / 4, Block % 8], m![Hidden % 4 # 16]>()
        .commit_trim::<m![Hidden % 4]>()
        .commit();
    let rowed: Rowed = device
        .main
        .begin(quads.view())
        .fetch::<m![Hidden % 2560 / 1280, Block % 8, Hidden % 80 / 4], m![Hidden % 4 # 16]>()
        .collect::<m![Hidden % 2560 / 1280, Block % 8, Hidden % 80 / 4], m![Hidden % 4 # 16]>()
        .commit_trim::<m![Hidden % 4]>()
        .commit();
    // SAFETY: `[M, Block % 8, Hidden % 80 / 4, Hidden % 4]` is `dpanel::Own`'s element in wire order.
    let rowed: DmTensor<
        bf16,
        Chips,
        Cluster,
        Reduced,
        m![Hidden % 2560 / 1280, { dpanel::Time }, { dpanel::Packet }],
    > = unsafe { rowed.reshape() };
    let mut own = dpanel::Own::new();
    rowed.view().to_dm_view(&mut device.tdma, own.view_mut());
    own
}
