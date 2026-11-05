//! A block's Q, K and V in one weight-stationary command, the oracle's prompt projection:
//! - each cluster holds every one of the chip's 3,072 outputs over one half of the features, a
//!   slice 96 outputs by 320 features of FP8 weight in its register file;
//! - the block's FP8 rows land once a cluster (the cluster's feature half, eight rows a slice) and
//!   the fetch network broadcasts them over the 32 output groups;
//! - the eight slices of an output group reduce their partials onto 32 rows a slice, and each
//!   cluster keeps the rows of one parity and receives the other cluster's partial of them;
//! - the summed rows, scaled, regroup onto whole heads a slice (one switch pass), where they
//!   rotate and leave for the step's row state and the pool's records.
use crate::Module;
use crate::ops::linear::contraction::staged::Term;
use furiosa_opt_std::prelude::*;

use super::super::query::{Heads, Panel};
use super::super::rotation::{self, Half as Freq, InputHalf, OutputHalf};
use super::super::{Coefficients, Head, Queries};
use super::{Block, Cache, Out};
use crate::models::nemo::serving::attention::{Head as PoolHead, Heads as PoolHeads, Kv as PoolKv};
use crate::models::nemo::serving::mlp::block::{Block as LayerBlock, Blocks as LayerBlocks};
use crate::models::nemo::serving::mlp::{Capacity, Waves};
use crate::models::nemo::serving::scale::{self, Copies, Factor, Ring};
use crate::models::nemo::serving::stack::{self, Depth};
use crate::models::nemo::serving::step::dlayer::NextRows;
use crate::models::nemo::serving::{Chips, Hidden, Rows};

axes![Part = 3];

/// The feature half a cluster contracts.
type Half = m![Hidden / 2560];
/// The rows a cluster keeps after the partials cross: the even or the odd ones.
type RowHalf = m![Block % 2];
/// An input slice's row group: the slices the fetch network broadcasts over the output groups.
type Groups = m![Block / 8];
type Features = m![Hidden % 2560 / 320];
/// Output group `Out / 32` holds a quarter (32 features) of a head in each of the three parts:
/// Q panel 0, Q panel 1 and the K/V weight regrouped by head (`land_pairs`).
type Owners = m![Out / 32];
type Slice = m![{ Owners }, { Features }];
type Lanes = m![Out % 8];
type Elements = m![
    Part,
    Hidden % 320 / 80,
    Hidden % 80 / 16,
    Out % 32 / 8,
    Hidden % 16
];
/// The fused weight of a cluster's feature half: 96 outputs by 320 features a slice, each lane's
/// 80-feature block of a part one 320-byte run.
pub(crate) type Weight = DmTensor<f8e4m3, Chips, Half, Slice, m![{ Lanes }, { Elements }]>;
/// The register file reads a lane's 64 features of one output a step.
type Filled = m![Part, Hidden % 320 / 64, Out % 32 / 8, Hidden % 64];
pub(super) type Register = TrfTensor<f8e4m3, Chips, Half, Slice, Lanes, Filled>;
/// A block's rows of a cluster's feature half: eight rows by 320 features a slice.
pub(super) type Activation = DmTensor<
    f8e4m3,
    Chips,
    Half,
    m![{ Groups }, { Features }],
    m![Hidden % 320 / 80, Block % 8, Hidden % 80],
>;
/// One row of 320 features, one packet of the broadcast: ten flits, few enough that the
/// pipeline behind the inter-slice reduce absorbs them.
type Packet = m![Hidden % 320];
type Switched = m![Block % 2, Block % 8 / 2, { Groups }];
type Steps = m![{ Switched }, Hidden % 320 / 64, Part, Out % 32 / 8];
type Sums = m![{ Switched }, Part, Out % 32 / 8];
/// After the reduce a slice holds 32 rows of its output group's three quarter heads.
type Reduced = m![{ Owners }, Block / 32];
type Kept = m![Block % 8 / 2, Block % 32 / 8, Part, Out % 32 / 8];
pub(super) type Partial =
    DmTensor<f32, Chips, Half, Reduced, m![{ m![Block % 2, { Kept }] }, { Lanes }]>;
type Own = DmTensor<f32, Chips, RowHalf, Reduced, m![{ Kept }, { Lanes }]>;
/// The summed and scaled rows, a quarter head's row of 32 features one run.
type Summed =
    DmTensor<bf16, Chips, RowHalf, Reduced, m![Part, Block % 32 / 8, Block % 8 / 2, Out % 32]>;
/// The fused weight's three parts: both Q panels, then the K/V weight regrouped by head.
pub(crate) fn land(
    device: &mut Device,
    weight: &mut Weight,
    (layers, index): (&stack::Layers, usize),
) {
    type Stored = super::super::contraction::Stored;
    type Into = m![Out % 8, Part = 1 #{!} 3, Hidden % 320 / 80, Hidden % 80 / 16, Out % 32 / 8, Hidden % 16];
    layers
        .query
        .at(index)
        .tile::<m![Panel], 1, m![Depth = 1 # Depth, Panel = 1 # 2, { Stored }]>(0)
        .to_dm_view(
            &mut device.tdma,
            weight.view_mut().tile::<m![Part], 1, Into>(0),
        );
    layers
        .query
        .at(index)
        .tile::<m![Panel], 1, m![Depth = 1 # Depth, Panel = 1 # 2, { Stored }]>(1)
        .to_dm_view(
            &mut device.tdma,
            weight.view_mut().tile::<m![Part], 1, Into>(1),
        );
    // SAFETY: relabels only, as `land_pairs` names the regrouped K/V weight by its stored axes;
    // only part 2 is written through these names.
    let grouped = unsafe {
        weight.view_mut().reshape::<Chips, Half, m![
            Out / 512,
            Out % 128 / 64,
            Out % 512 / 256,
            Out % 256 / 128,
            Out % 64 / 32,
            { Features }
        ], m![{ Lanes }, { Elements }]>()
    };
    layers
        .pairs
        .at(index)
        .to_dm_view(&mut device.tdma, grouped.tile::<m![Part], 1, Into>(2));
}

/// The fused weight in the register file, once for every block of the chunk.
pub(super) fn fill(device: &mut Device, weight: &Weight) -> Register {
    device
        .sub
        .begin(weight.view())
        .fetch::<m![Out % 8, Part, Hidden % 320 / 64, Out % 32 / 8], m![Hidden % 64 / 16, Hidden % 16]>()
        .collect::<m![Out % 8, Part, Hidden % 320 / 64, Out % 32 / 8, Hidden % 64 / 32], m![Hidden % 32]>()
        .to_trf()
}

/// Each part's weight scale in every slice of the summed rows: the Q scale for both Q panels, the
/// output group's K or V scale for the K/V part.
type Scales = DmTensor<f32, Chips, RowHalf, Reduced, m![Part, 1 # 8]>;

/// The K/V weight scale of one slice of every 16-slice ring, named by the axes of the stored
/// scale: `Out % 128 / 64` selects K or V and sits in the ring digit where `Reduced` has the
/// output group's `Out % 512 / 256` (output group `2 Kv + h % 2`).
pub(crate) type Seeds = DmTensor<
    f32,
    Chips,
    m![Out % 64 / 32],
    m![Out / 512, Out % 128 / 64, Out % 512 / 128, 1 # 16],
    m![1 # 8],
>;

/// The K/V weight scale seeds (`Seeds`): one transfer of 32 runs, so it can land under an earlier
/// phase.
pub(crate) fn land_scales(
    device: &mut Device,
    seeds: &mut Seeds,
    (layers, index): (&stack::Layers, usize),
) {
    layers
        .pair_scale
        .at(index)
        .tile::<m![Rows / 4], 1, m![
            Depth = 1 # Depth,
            Out / 512,
            Out % 512 / 128,
            Out % 128 / 64,
            Out % 64 / 32,
            Rows / 4 = 1 # 16,
            1 # 8
        ]>(0)
        .to_dm_view(&mut device.tdma, seeds.view_mut());
}

/// Each part's weight scale in every slice (`Scales`): the Q scale copied out of the spread row in
/// each slice, the K/V seeds spread over their rings.
fn spread(
    device: &mut Device,
    (row, seeds): (&scale::Scales<Chips>, &Seeds),
    weights: &mut Scales,
) {
    type Into = m![Part = 1 #{!} 3, 1 # 8];
    for part in 0..2 {
        device
            .main
            .begin(
                row.row::<RowHalf, Reduced>()
                    .tile::<m![Factor], 1, m![Factor = 1 # 8, 1 # 8]>(1)
                    .unpad::<m![Factor = 1, 1 # 8]>(),
            )
            .fetch::<m![Factor = 1], m![1 # 8]>()
            .collect::<m![Factor = 1], m![1 # 8]>()
            .commit_trim::<m![1 # 8]>()
            .commit_view(weights.view_mut().tile::<m![Part], 1, Into>(part));
    }
    // SAFETY: relabels only: cluster and ring keep their positions (`Seeds`).
    let seeds = unsafe {
        seeds
            .view()
            .reshape::<Chips, m![Copies / 256], m![Copies % 256 / 16, 1 # 16], m![1 # 8]>()
    };
    // SAFETY: as above, the ring and its 16 slices are `Reduced`'s slice digits in order.
    let into = unsafe {
        weights
            .view_mut()
            .reshape::<Chips, m![Copies / 256], m![Copies % 256 / 16, Ring], m![Part, 1 # 8]>()
    };
    device
        .main
        .begin(seeds)
        .fetch::<m![1], m![1 # 8]>()
        .switch::<m![Copies % 256 / 16, Ring], m![1]>(SwitchConfig::CustomBroadcast {
            ring_size: 16,
        })
        .collect::<m![1], m![1 # 8]>()
        .commit_trim::<m![1 # 8]>()
        .commit_view(into.tile::<m![Part], 1, m![Part = 1 #{!} 3, 1 # 8]>(2));
}

/// The input scale and each part's weight scale in registers, over the summed rows' slices.
pub(super) type Factors =
    crate::ops::linear::contraction::staged::Factors<Chips, RowHalf, Reduced, m![Part, 1 # 8]>;
pub(super) type Projection = crate::ops::linear::Affine<Register, Factors>;
type Space = crate::compose::Tile<Block, Block>;

impl Factors {
    pub(super) fn new(device: &mut Device, input: &scale::Scales<Chips>, seeds: &Seeds) -> Self {
        let mut weight = Scales::new();
        spread(device, (input, seeds), &mut weight);
        Self {
            input: input.get(device, 0),
            weight: device
                .sub
                .begin(weight.view())
                .fetch::<m![Part], m![1 # 8]>()
                .collect::<m![Part], m![1 # 8]>()
                .to_vrf(),
        }
    }
}

/// Block `block` of `rows` (both halves placed) into `activation`, each cluster its feature half.
pub(super) fn reload(
    device: &mut Device,
    rows: &NextRows,
    block: usize,
    activation: &mut Activation,
) {
    let tile = rows.view().tile::<m![LayerBlocks], 1, m![
        LayerBlocks = 1 # LayerBlocks,
        Hidden % 1280 / 640,
        Hidden % 640 / 320,
        LayerBlock / 8,
        Hidden % 320 / 80,
        Hidden / 2560,
        Hidden % 2560 / 1280,
        LayerBlock % 8,
        Hidden % 80 / 8,
        Hidden % 8
    ]>(block);
    // SAFETY: a block's 256 rows in order; only the row axis is renamed.
    let tile = unsafe {
        tile.reshape::<Chips, m![
            LayerBlocks = 1 # LayerBlocks,
            Hidden % 1280 / 640,
            Hidden % 640 / 320,
            Block / 8,
            Hidden % 320 / 80,
            Hidden / 2560,
            Hidden % 2560 / 1280,
            Block % 8,
            Hidden % 80 / 8,
            Hidden % 8
        ]>()
    };
    tile.to_dm_view(&mut device.tdma, activation.view_mut());
}

/// The block's Q/K/V command over the register, each slice its output group's partial of eight
/// rows over its cluster's feature half.
impl crate::ops::linear::contraction::staged::Split for Space {
    type Cluster = Half;
    type Slices = Slice;
    type InputSlices = m![{ Groups }, { Features }];
    type Input = m![Hidden % 320 / 80, Block % 8, Hidden % 80];
    type Reduced = Reduced;
    type Rows = RowHalf;
    type Lanes = Lanes;
    type Weight = Filled;
    type Feed = m![Block % 2, Block % 8 / 2];
    type Packet = Packet;
    type Switched = Switched;
    type Operand = m![Hidden % 64];
    const BROADCAST: SwitchConfig = SwitchConfig::Broadcast1 {
        slice1: 32,
        slice0: 8,
    };
    type Flits = m![{ Switched }, Hidden % 320 / 32];
    type Flit = m![Hidden % 32];
    type Steps = Steps;
    type Contracted = m![1];
    type Sums = Sums;
    type Partial = m![Block % 2, { Kept }];
    type Kept = Kept;
    type Interleaved = m![Block % 8 / 2, Block % 32 / 8, Part, Term, Out % 32 / 8];
    type Scale = m![Part, 1 # 8];
    type Output = m![Part, Block % 32 / 8, Block % 8 / 2, Out % 32];
}

/// Each cluster keeps its rows of its partial and receives the other cluster's partial of them.
pub(super) fn cross(device: &mut Device, partial: &Partial) -> Crossed {
    let remote = partial
        .cluster_swap()
        .cluster_slice::<m![Block % 2], m![{ Kept }, { Lanes }]>([0, 1])
        .to_dm::<m![{ Kept }, { Lanes }]>(&mut device.tdma);
    let local = partial.asymmetric_cluster_slice::<m![Block % 2], m![{ Kept }, { Lanes }]>(
        &mut device.sub,
        &[0, 1],
    );
    // SAFETY: cluster `c` holds the rows of parity `c`, so naming the cluster by it moves no data.
    unsafe {
        Crossed(
            local.reshape::<Chips, RowHalf, Reduced, m![{ Kept }, { Lanes }]>(),
            remote.reshape::<Chips, RowHalf, Reduced, m![{ Kept }, { Lanes }]>(),
        )
    }
}

/// Both partials summed, scaled by the input and each part's weight scale and rounded once to
/// BF16 into `summed`, a quarter head's row of 32 features one run.
fn finish(
    device: &mut Device,
    projection: &Projection,
    (local, remote): (&Own, &Own),
    summed: &mut Summed,
) {
    projection.forward((
        device,
        crate::ops::layout::Dm::<Space, _> {
            tiles: crate::compose::Tiles::over(Block::SIZE),
            value: (local, remote, summed.view_mut()),
        },
    ));
}

/// A block's partials crossed over the clusters: what `end` sums.
pub(super) struct Crossed(Own, Own);

/// A summed block's slices with whole heads: 32 rows of one head of each part a slice (this
/// cluster's parity), a head's row of 128 features one run.
type Headed = m![Out / 128, Block % 32 / 8, Block / 32];
/// A slice's rows in time: each part's head, of this cluster's parity.
type Lines = m![Part, Block % 8 / 2];
type Turned = DmTensor<bf16, Chips, RowHalf, Headed, m![{ Lines }, Head]>;
type Codes = DmTensor<f8e4m3, Chips, RowHalf, Headed, m![{ Lines }, Head]>;
/// A block's rotation coefficients in every slice of its rows; every part's head reads them.
type Spread =
    DmTensor<bf16, Chips, RowHalf, Headed, m![OutputHalf, Block % 8 / 2, InputHalf, Freq]>;
type Turning = rotation::Register<Chips, RowHalf, Headed, m![Block % 8 / 2]>;
/// A block's rows as the pool keeps a token's record (`super::Records`), each cluster the rows of
/// its parity, so the records build and scatter within the cluster that holds the rows.
type Records = DmTensor<
    f8e4m3,
    Chips,
    RowHalf,
    m![PoolHeads / 2, Block / 2],
    m![PoolHeads % 2, PoolKv, PoolHead],
>;

/// A block's outputs on their way to the step's row state and the pool: its rows with whole
/// heads, its rotation coefficients and its rows' pool offsets, then its rotated rows and its
/// K (rotated) and V in E4M3, which the write reads a block later.
pub(super) struct Outputs {
    headed: Turned,
    spread: Spread,
    entries: HbmTensor<i32, Chips, m![Block]>,
    turned: Turned,
    keys: Codes,
    values: Codes,
}

impl Outputs {
    pub(super) fn new() -> Self {
        Self {
            headed: Turned::new(),
            spread: Spread::new(),
            entries: HbmTensor::new(),
            turned: Turned::new(),
            keys: Codes::new(),
            values: Codes::new(),
        }
    }

    /// Lands the coefficients and pool offsets of the block of four waves from `wave` on.
    pub(super) fn land(
        &mut self,
        device: &mut Device,
        (table, offsets, wave): (&Coefficients, &HbmTensor<i32, Chips, m![Capacity]>, usize),
    ) {
        let rows = table
            .view()
            .tile::<m![Waves], 4, m![Waves = 4 # Waves, { rotation::Stored }]>(wave);
        // SAFETY: four consecutive waves of 64 rows are one block of 256; only the labels change.
        let rows: HbmTensorView<
            '_,
            bf16,
            Chips,
            m![Block / 64 # Waves, Block % 64, OutputHalf, InputHalf, Freq],
        > = unsafe { rows.reshape() };
        // SAFETY: the coefficients of a row are the same for every head; the literal digit the
        // transfer repeats them over is named as the head.
        let spread = unsafe {
            self.spread
                .view_mut()
                .reshape::<Chips, RowHalf, m![8, Block % 32 / 8, Block / 32], m![
                    OutputHalf,
                    Block % 8 / 2,
                    InputHalf,
                    Freq
                ]>()
        };
        rows.to_dm_view(&mut device.tdma, spread);
        let offsets = offsets
            .view()
            .tile::<m![Capacity], 256, m![Capacity = 256 # Capacity]>(64 * wave);
        // SAFETY: the offset tile names exactly the block's rows in order.
        let offsets = unsafe { offsets.reshape::<Chips, m![Block # Capacity]>() };
        offsets.to_hbm_view(&mut device.tdma, self.entries.view_mut());
    }

    /// Every head rotated against one coefficient fill, then K (rotated) and V cast for records.
    pub(super) fn rotate(&mut self, device: &mut Device) {
        let register: Turning = rotation::fill(device, self.spread.view());
        turn(device, &self.headed, &register, &mut self.turned);
        e4m3(device, &self.turned, &mut self.keys);
        e4m3(device, &self.headed, &mut self.values);
    }

    /// The block's rotated Q written into the step's row state at `wave` and its K/V records
    /// into the pool.
    pub(super) fn write(
        &self,
        device: &mut Device,
        wave: usize,
        (queries, cache): (&mut Queries, &mut Cache<'_>),
    ) {
        // SAFETY: output group `Out / 128` of a Q part is panel head `Heads`.
        let turned = unsafe {
            self.turned
                .view()
                .reshape::<Chips, RowHalf, m![Heads, Block % 32 / 8, Block / 32], m![Part, Block % 8 / 2, Head]>()
        };
        type Read = m![Part = 1 # 3, Block % 8 / 2, Head];
        type Into = m![Block #{!} Capacity, Heads, Panel = 1 #{!} 2, Head];
        for panel in 0..2 {
            // SAFETY: head `2 g + p` of a row is panel `p`'s head `g`.
            let heads = unsafe {
                queries
                    .view_mut()
                    .reshape::<Chips, m![Capacity, Heads, Panel, Head]>()
            };
            let tile = heads
                .tile::<m![Capacity], 256, m![Capacity = 256 #{!} Capacity, Heads, Panel, Head]>(
                    wave * 64,
                );
            // SAFETY: the 256-row tile keeps the capacity stride; only its row axis is renamed.
            let tile =
                unsafe { tile.reshape::<Chips, m![Block #{!} Capacity, Heads, Panel, Head]>() };
            turned
                .tile::<m![Part], 1, Read>(panel)
                .to_hbm_view(&mut device.tdma, tile.tile::<m![Panel], 1, Into>(panel));
        }
        let mut records = Records::new();
        copy(device, &self.keys, (&mut records, 0));
        copy(device, &self.values, (&mut records, 1));
        crate::ops::attention::cache::Scatter.forward((
            device,
            (
                &mut *cache.pool,
                (cache.entry, &self.entries),
                crate::ops::layout::Dm::<super::PairSpace, _> {
                    tiles: crate::compose::Tiles::over(Block::SIZE),
                    value: records.view(),
                },
            ),
        ));
    }
}

/// Each head of `rows` rotated by its row's position (`rotation::apply`) into `turned`.
fn turn(device: &mut Device, rows: &Turned, register: &Turning, turned: &mut Turned) {
    // SAFETY: Head is the contiguous product InputHalf x Half within each row.
    let rows = unsafe {
        rows.view()
            .reshape::<Chips, RowHalf, Headed, m![{ Lines }, InputHalf, Freq]>()
    };
    // SAFETY: concatenating OutputHalf x Half restores one contiguous Head per row.
    let turned = unsafe {
        turned
            .view_mut()
            .reshape::<Chips, RowHalf, Headed, m![{ Lines }, OutputHalf, Freq]>()
    };
    crate::ops::attention::rope::Rotary::<Head, Head, bf16, Chips>::new().forward((
        device,
        crate::ops::layout::Dm::<
            rotation::Space<RowHalf, Headed, Lines, m![Block % 8 / 2], Block>,
            _,
        > {
            tiles: crate::compose::Tiles::over(Block::SIZE),
            value: (rows, register, turned),
        },
    ));
}

/// Each head of the K/V part of `rows` in E4M3 into `codes`, the only part the records read.
fn e4m3(device: &mut Device, rows: &Turned, codes: &mut Codes) {
    type Kv = m![Part = 1, Block % 8 / 2, Head / 8];
    device
        .main
        .begin(
            rows.view()
                .tile::<m![Part], 1, m![Part = 1 # 3, Block % 8 / 2, Head]>(2)
                .unpad::<m![Part = 1, Block % 8 / 2, Head]>(),
        )
        .fetch::<Kv, m![Head % 8]>()
        .fetch_cast::<f32>()
        .collect::<Kv, m![Head % 8]>()
        .cast::<f8e4m3, m![Head % 8 # 32]>()
        .commit_trim::<m![Head % 8]>()
        .commit_view(
            codes
                .view_mut()
                .tile::<m![Part], 1, m![Part = 1 #{!} 3, Block % 8 / 2, Head]>(2),
        );
}

/// Half `kv` of every row's record (0 the keys, 1 the values) from the K/V part of `codes`.
fn copy(device: &mut Device, codes: &Codes, (records, kv): (&mut Records, usize)) {
    // SAFETY: K/V output group `Out / 128` is head `2 (Out / 512) + Out % 128 % 2`'s K or V
    // (`Out % 512 / 256`), under the pool's names.
    let codes = unsafe {
        codes.view().reshape::<Chips, RowHalf, m![
            PoolHeads / 2,
            PoolKv,
            PoolHeads % 2,
            Block % 32 / 8,
            Block / 32
        ], m![Part, Block % 8 / 2, PoolHead]>()
    };
    codes
        .slice_tile::<m![PoolKv], 1, m![PoolHeads / 2, PoolKv = 1 # 2, PoolHeads % 2, Block % 32 / 8, Block / 32]>(kv)
        .tile::<m![Part], 1, m![Part = 1 # 3, Block % 8 / 2, PoolHead]>(2)
        .to_dm_view(
            &mut device.tdma,
            records
                .view_mut()
                .tile::<m![PoolKv], 1, m![PoolHeads % 2, PoolKv = 1 #{!} 2, PoolHead]>(kv),
        );
}

/// The crossed partials summed and scaled by `factors`, then moved onto whole heads a slice.
pub(super) fn end(
    device: &mut Device,
    projection: &Projection,
    Crossed(local, remote): Crossed,
    outputs: &mut Outputs,
) {
    let mut summed = Summed::new();
    finish(device, projection, (&local, &remote), &mut summed);
    device
        .main
        .begin(summed.view())
        .fetch::<m![Part, Block % 32 / 8, Block % 8 / 2, Out % 32 / 16], m![Out % 16]>()
        .switch::<Headed, m![Part, Block % 8 / 2, Out % 32 / 16, Out % 128 / 32]>(
            SwitchConfig::InterTranspose {
                slice1: 4,
                slice0: 8,
                time0: 8,
            },
        )
        .collect::<m![Part, Block % 8 / 2, Out % 32 / 16, Out % 128 / 32], m![Out % 16]>()
        .commit_trim::<m![Out % 16]>()
        .commit_view(unsafe {
            // SAFETY: a head's 128 features in order.
            outputs
                .headed
                .view_mut()
                .reshape::<Chips, RowHalf, Headed, m![Part, Block % 8 / 2, Out % 128]>()
        });
}
