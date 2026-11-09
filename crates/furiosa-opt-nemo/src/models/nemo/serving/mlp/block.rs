//! A prompt chunk's gate and up in the oracle's geometry: 256-row blocks, each block's activation
//! held in the register file (256 rows of one 80-feature K group a slice) while the cluster's
//! whole gate and up weights stream past it once, four commands a block; the K groups' partials
//! sum in the inter-slice reduction and the scales and SiLU run in the same pass, so a block's
//! product needs one more pass before it leaves for HBM.
use crate::Module;
use furiosa_opt_std::prelude::*;

use super::{Hidden, Intermediate, Rows, Waves};
use crate::models::nemo::serving::{Chips, scale, stack, stack::Depth};

axes![Block = 256, Blocks = 16, Projection = 2];
const _: () = assert!(Block::SIZE == 4 * Rows::SIZE && Blocks::SIZE * 4 == Waves::SIZE);

type Cluster = m![Intermediate / 3584];
/// One half of a cluster's outputs: four groups of 448 by the 64 K groups of 80 features.
pub type Slice = m![Intermediate % 1792 / 448, Hidden / 80];
/// A gate or up half in the layout its contraction streams: 448 outputs by 80 features a slice.
pub type Weight = DmTensor<
    f8e4m3,
    Chips,
    Cluster,
    Slice,
    m![
        Intermediate % 448 / 8,
        Hidden % 80 / 8,
        Intermediate % 8,
        Hidden % 8
    ],
>;
/// A block's FP8 activation in every output group's slices.
pub type Activation = DmTensor<f8e4m3, Chips, Cluster, Slice, m![Block, Hidden % 80]>;
type Lanes = m![Block % 8];
/// One 128-row half of a block in the register file: 8 rows a lane, 16 row steps.
pub type Register = TrfTensor<
    f8e4m3,
    Chips,
    Cluster,
    Slice,
    Lanes,
    m![Hidden % 80 / 16, Block % 32 / 8, Block / 32, Hidden % 16],
>;
/// A weight packet is eight outputs of eight features; it serves the half block's 16 row steps.
type Feed = m![
    Intermediate % 56 / 4,
    Intermediate % 448 / 56,
    Hidden % 80 / 16
];
type Flit = m![Intermediate % 4, Hidden % 16];
type Steps = m![{ Feed }, Block % 32 / 8, Block / 32];
type Sums = m![
    Intermediate % 56 / 4,
    Intermediate % 448 / 56,
    Block % 32 / 8,
    Block / 32
];
/// After the reduction a slice holds 56 outputs of 16 rows of a half block.
pub type Reduced = m![Intermediate % 1792 / 56, Block / 32];
type Time = m![Intermediate % 56 / 4, Block % 32 / 8, Intermediate % 4];
type Packet = m![Block % 8];
pub type Values<D> = DmTensor<D, Chips, Cluster, Reduced, m![{ Time }, { Packet }]>;
type Space = crate::compose::Tile<Block, Rows>;

impl crate::ops::linear::contraction::Mapping for Space {
    type Cluster = Cluster;
    type Slices = Slice;
    type Reduced = Reduced;
    type Input = m![Hidden % 80 / 16, Block % 32 / 8, Block / 32, Hidden % 16];
    type Lanes = Lanes;
    type Weight = m![
        Intermediate % 448 / 8,
        Hidden % 80 / 8,
        Intermediate % 8,
        Hidden % 8
    ];
    type Feed = Feed;
    type Packet = Flit;
    type Flits = m![{ Feed }, Intermediate % 4 / 2];
    type Flit = m![Intermediate % 2, Hidden % 16];
    type Steps = Steps;
    type Contracted = m![Intermediate % 4];
    type Sums = Sums;
    type Groups = m![{ Sums }, Intermediate % 4];
    type Output = Time;
}
/// A half block's product transposed: each row's eight features of a chunk one 8-byte run.
type Rowed = m![
    Intermediate % 56 / 8,
    Block % 32 / 8,
    Block % 8,
    Intermediate % 8
];
pub type Stored = DmTensor<f8e4m3, Chips, Cluster, Reduced, Rowed>;
/// Every block's E4M3 product: each eight-feature chunk's 256 rows one 2-KiB run.
pub type Product = HbmTensor<f8e4m3, Chips, m![Blocks, Intermediate / 8, Block, Intermediate % 8]>;

/// The cluster's four gate and up halves, staged once a layer.
pub struct Staged {
    gate: [Weight; 2],
    up: [Weight; 2],
}

impl Staged {
    /// The four staged halves: gate 0, up 0, gate 1, up 1.
    pub fn from([gate0, up0, gate1, up1]: [Weight; 4]) -> Self {
        Self {
            gate: [gate0, gate1],
            up: [up0, up1],
        }
    }

    /// Half `half` of layer `index`'s projection `projection` (0 gate, 1 up) as stored
    /// (`weight::Stored`), landed into its owner: a slice's 448 outputs by 80 features in eight
    /// 4,480-B runs.
    pub fn land_staged(
        &mut self,
        device: &mut Device,
        (layers, index): (&stack::Layers, usize),
        (projection, half): (usize, usize),
    ) {
        let (weight, stacked) = if projection == 0 {
            (&mut self.gate[half], &layers.gate)
        } else {
            (&mut self.up[half], &layers.up)
        };
        stacked
            .at(index)
            .tile::<m![Intermediate % 3584 / 1792], 1, m![
                Depth = 1 # Depth,
                Intermediate % 3584 / 1792 = 1 # 2,
                Intermediate / 3584,
                Intermediate % 1792 / 448,
                Intermediate % 448 / 56,
                Hidden / 80,
                Intermediate % 56 / 8,
                Hidden % 80 / 8,
                Intermediate % 8,
                Hidden % 8
            ]>(half)
            .to_dm_view(&mut device.tdma, weight.view_mut());
    }

    /// Lands four of a half's eight runs, preserving the other part in the same owner.
    pub fn land_part(
        &mut self,
        device: &mut Device,
        (layers, index): (&stack::Layers, usize),
        (projection, half): (usize, usize),
        part: usize,
    ) {
        let (weight, stacked) = if projection == 0 {
            (&mut self.gate[half], &layers.gate)
        } else {
            (&mut self.up[half], &layers.up)
        };
        stacked
            .at(index)
            .tile::<m![Intermediate % 3584 / 1792], 1, m![
                Depth = 1 # Depth,
                Intermediate % 3584 / 1792 = 1 # 2,
                Intermediate / 3584,
                Intermediate % 1792 / 448,
                Intermediate % 448 / 56,
                Hidden / 80,
                Intermediate % 56 / 8,
                Hidden % 80 / 8,
                Intermediate % 8,
                Hidden % 8
            ]>(half)
            .tile::<m![Intermediate % 448 / 224], 1, m![
                Depth = 1 # Depth,
                Intermediate % 3584 / 1792 = 1 # 2,
                Intermediate / 3584,
                Intermediate % 1792 / 448,
                Intermediate % 448 / 224 = 1 # 2,
                Intermediate % 224 / 56,
                Hidden / 80,
                Intermediate % 56 / 8,
                Hidden % 80 / 8,
                Intermediate % 8,
                Hidden % 8
            ]>(part)
            .to_dm_view(
                &mut device.tdma,
                weight
                    .view_mut()
                    .tile::<m![Intermediate % 448 / 224], 1, m![
                        Intermediate % 448 / 224 = 1 #{!} 2,
                        Intermediate % 224 / 8,
                        Hidden % 80 / 8,
                        Intermediate % 8,
                        Hidden % 8
                    ]>(part),
            );
    }
}

/// Every block's FP8 activation K80-blocked: each 80-feature group's 256 rows one 20-KiB run.
pub type Blocked = HbmTensor<f8e4m3, Chips, m![Blocks, Hidden / 80, Block, Hidden % 80]>;

/// A landed block in the register file.
pub fn fill(device: &mut Device, activation: &Activation) -> Register {
    device
        .sub
        .begin(activation.view())
        .fetch::<m![Block % 8, Hidden % 80 / 16, Block % 32 / 8, Block / 64], m![
            Block / 32 % 2,
            Hidden % 16
        ]>()
        .collect::<m![Block % 8, Hidden % 80 / 16, Block % 32 / 8, Block / 64], m![
            Block / 32 % 2,
            Hidden % 16
        ]>()
        .to_trf()
}

/// One half's contraction against a filled block, scaled (and for the gate, SiLU) in its pass.
fn contract(
    device: &mut Device,
    (weight, input): (&Weight, &Register),
    (scale, factor, silu): (&scale::Scales<Chips>, usize, bool),
) -> Values<f32> {
    let scales = scale::Epilogue {
        input: scale.get::<Cluster, Reduced>(device, factor),
        weight: scale.get::<Cluster, Reduced>(device, factor + 1),
    };
    let projection = crate::ops::linear::Affine::new(weight, &scales);
    let value = projection
        .forward((
            device,
            crate::ops::layout::Dm::<Space, _> {
                tiles: crate::compose::Tiles::over(Block::SIZE),
                value: (input,),
            },
        ))
        .value;
    if silu {
        value
            .vector_stash()
            .vector_fp_unary(FpUnaryOp::Sigmoid)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Fma), Stash)
            .vector_widen_concat::<Time, Packet>()
            .vector_final()
            .commit_trim::<Packet>()
            .commit()
    } else {
        value
            .vector_widen_concat::<Time, Packet>()
            .vector_final()
            .commit_trim::<Packet>()
            .commit()
    }
}

/// SiLU(gate) · up of one half, divided by the down input scale and rounded once to E4M3.
fn multiply(
    device: &mut Device,
    scales: &scale::Scales<Chips>,
    (gate, up): (&Values<f32>, &Values<f32>),
) -> Values<f8e4m3> {
    axes![Term = 2];
    let divisor = crate::ops::linear::Affine::new((), scales.get::<Cluster, Reduced>(device, 4));
    device
        .main
        .begin_interleaved::<Term, _, _, _, _, _>(gate.view(), up.view())
        .fetch::<m![{ Time }, Term], Packet>()
        .collect::<m![{ Time }, Term], Packet>()
        .vector_init()
        .vector_intra_slice_unzip::<Term, Time>()
        .vector_narrow_split::<m![{ Time }, { Packet } / 4], m![{ Packet } % 4]>()
        .vector_fp_zip(FpBinaryOp::MulF(FpMulAlu::Mul0))
        .vector_fp_div(divisor.input_scale())
        .vector_widen_concat::<Time, Packet>()
        .vector_final()
        .cast::<f8e4m3, m![{ Packet } # 32]>()
        .commit_trim::<Packet>()
        .commit()
}

/// A half's product with each row's features innermost, eight to a run, for the down's stream.
fn rowed(device: &mut Device, value: &Values<f8e4m3>) -> Stored {
    device
        .main
        .begin(value.view())
        .fetch::<m![Intermediate % 56 / 8, Block % 32 / 8, Intermediate % 8], m![Block % 8 # 32]>()
        .collect::<m![Intermediate % 56 / 8, Block % 32 / 8, Intermediate % 8], m![Block % 8 # 32]>()
        .transpose::<m![Intermediate % 56 / 8, Block % 32 / 8, Block % 8], m![Intermediate % 8 # 32]>()
        .commit_trim::<m![Intermediate % 8]>()
        .commit()
}

/// Up half `half` of a filled block.
pub fn up(
    device: &mut Device,
    (staged, scales): (&Staged, &scale::Scales<Chips>),
    (filled, half): (&Register, usize),
) -> Values<f32> {
    contract(device, (&staged.up[half], filled), (scales, 2, false))
}

/// SiLU of gate half `half` of a filled block.
pub fn gate(
    device: &mut Device,
    (staged, scales): (&Staged, &scale::Scales<Chips>),
    (filled, half): (&Register, usize),
) -> Values<f32> {
    contract(device, (&staged.gate[half], filled), (scales, 0, true))
}

/// Half `half`'s product of its gate and up into block `block` of `product`.
pub fn finish(
    device: &mut Device,
    scales: &scale::Scales<Chips>,
    (gate, up): (Values<f32>, Values<f32>),
    (product, block, half): (&mut Product, usize, usize),
) {
    let value = multiply(device, scales, (&gate, &up));
    let value = rowed(device, &value);
    store(device, &value, product_view(product, block, half));
}

type Tile<'a> = HbmTensorViewMut<
    'a,
    f8e4m3,
    Chips,
    m![
        Blocks = 1 #{!} Blocks,
        Intermediate / 3584,
        Intermediate % 3584 / 1792 = 1 #{!} 2,
        Intermediate % 1792 / 56,
        Intermediate % 56 / 8,
        Block,
        Intermediate % 8
    ],
>;

fn product_view(product: &mut Product, block: usize, half: usize) -> Tile<'_> {
    let tile = product
        .view_mut()
        .tile::<m![Blocks], 1, m![Blocks = 1 #{!} Blocks, Intermediate / 8, Block, Intermediate % 8]>(block);
    // SAFETY: 56 and 1,792 are multiples of 8, so the chunk index splits into these digits.
    let tile = unsafe {
        tile.reshape::<Chips, m![
            Blocks = 1 #{!} Blocks,
            Intermediate / 1792,
            Intermediate % 1792 / 56,
            Intermediate % 56 / 8,
            Block,
            Intermediate % 8
        ]>()
    };
    tile.tile::<m![Intermediate % 3584 / 1792], 1, m![
        Blocks = 1 #{!} Blocks,
        Intermediate / 3584,
        Intermediate % 3584 / 1792 = 1 #{!} 2,
        Intermediate % 1792 / 56,
        Intermediate % 56 / 8,
        Block,
        Intermediate % 8
    ]>(half)
}

/// One half's product into its tile of the product.
fn store(device: &mut Device, value: &Stored, tile: Tile<'_>) {
    value.view().to_hbm_view(&mut device.tdma, tile);
}
