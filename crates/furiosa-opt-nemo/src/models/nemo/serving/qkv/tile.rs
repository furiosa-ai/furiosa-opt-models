//! Q, K and V of a prompt chunk in 256-row blocks, the activation stationary as Qwen3's E4M3
//! projection keeps it:
//! 1. every block lands and normalizes, half of its rows a cluster and a quarter of those a
//!    slice, and its FP8 rows go to HBM;
//! 2. each block's rows come back from HBM into every output group's slices, fill the register
//!    file once, and every Q and K/V weight byte streams once against them, in one K/V and one Q
//!    command whose inter-slice reduce scatters the rows, so a slice keeps one head of eight rows;
//! 3. a block's Q and K rotate in that layout under the next block's K/V command, and its Q, K, V
//!    and E4M3 pool records leave under the next block's Q command.
//!
//! The block loop issues no activation store, so its control stream never waits on a cluster
//! barrier for the rows.
use crate::Module;
use furiosa_opt_std::prelude::*;

use super::contraction::{self, Cluster, Out, Slice};
use super::query::{self, Heads, Panel};
use super::rotation::{self, Half, InputHalf, OutputHalf, Rotated};
use super::{Coefficients, Head, KvHeads, Queries, Weights};
use crate::models::nemo::serving::attention::{
    self, Head as PoolHead, Heads as PoolHeads, Kv as PoolKv, Page,
};
use crate::models::nemo::serving::mlp::block::{
    Block as LayerBlock, Blocked as LayerRows, Blocks as LayerBlocks,
};
use crate::models::nemo::serving::mlp::{Capacity, Waves};
use crate::models::nemo::serving::stack::{self, Depth};
use crate::models::nemo::serving::step::{HiddenRows, dlayer};
use crate::models::nemo::serving::{Blocked, Chips, Hidden, Rows, scale};

pub(crate) mod fused;

axes![Block = 256, Blocks = 16, Replica = 2];
const _: () = assert!(Block::SIZE == 4 * crate::models::nemo::serving::Rows::SIZE);
const _: () = assert!(Blocks::SIZE * Block::SIZE == Capacity::SIZE);

/// A weight packet: four adjacent outputs by 16 features, read as four 16-byte runs.
type Packet = m![Out % 4, Hidden % 16];
type Flit = m![Out % 2, Hidden % 16];
/// The row steps a weight packet feeds: the block's 32 groups of eight rows, as many partial sums
/// as the 1,024-cell accumulator holds for eight lanes of four outputs.
type Steps = m![Block / 8];
type Lanes = m![Block % 8];
/// A head's 128 features but the four in a packet, in feature order.
type Features = m![Out % 128 / 4];
type PairFeed = m![{ Features }, Hidden % 80 / 16];
type PairSums = m![{ Features }, { Steps }];
type QueryFeed = m![Panel, { Features }, Hidden % 80 / 16];
type QuerySums = m![Panel, { Features }, { Steps }];
/// After the reduce a slice keeps one head of eight rows: its output group's head, the rows, and
/// the panel (Q) or a copy (K/V, whose output group holds one head).
type QueryRows = m![Out % 512 / 128, Block / 8, Panel];
type PairRows = m![Out % 512 / 128, Block / 8, Replica];
/// Eight rows of one head, row-major.
type Row = m![Block % 8, { Features }, Out % 4];
type Queried = DmTensor<bf16, Chips, Cluster, QueryRows, Row>;
type Paired = DmTensor<bf16, Chips, Cluster, PairRows, Row>;
/// A block's rotation coefficients in every slice of its eight-row groups.
type Spread =
    DmTensor<bf16, Chips, Cluster, m![4, Block / 8, 2], m![OutputHalf, Block % 8, InputHalf, Half]>;
type Rotations = rotation::Register<Chips, Cluster, QueryRows, Lanes>;
type Turned = Rotated<Chips, Cluster, QueryRows, Lanes>;
/// A block's rows in the register file, eight a lane.
type Register = TrfTensor<
    f8e4m3,
    Chips,
    Cluster,
    Slice,
    Lanes,
    m![Block / 16, Hidden % 80 / 16, Block % 16 / 8, Hidden % 16],
>;
/// Slice names under which a block's K/V outputs read as the pool's: the output group is
/// `2 Kv + KvHeads % 2`, the cluster `KvHeads / 2`.
type PoolRows = m![PoolKv, PoolHeads % 2, Block / 8, Replica];
type Codes = DmTensor<f8e4m3, Chips, Cluster, PairRows, m![Block % 8, Head]>;
/// A block's rows as the pool keeps a token's record: one row a slice holding the keys, then the
/// values, of its cluster's two heads, so the scatter writes a row as one 512-byte run.
type Records =
    DmTensor<f8e4m3, Chips, m![PoolHeads / 2], m![Block], m![PoolHeads % 2, PoolKv, PoolHead]>;
const _: () = assert!(PoolHeads::SIZE == KvHeads::SIZE && PoolHead::SIZE == Head::SIZE);
type RowHalf = m![Block / 128];
/// A slice of a landed block: a quarter of its cluster's rows of one 80-feature group.
type Quarter = m![Block % 128 / 32, Hidden / 80];
type QuarterRows = m![Block % 32];
type QuarterTime = m![{ QuarterRows }, Hidden % 80 / 8];
type Landed = DmTensor<bf16, Chips, RowHalf, Quarter, m![{ QuarterRows }, Hidden % 80]>;
type Normalized = DmTensor<f8e4m3, Chips, RowHalf, Quarter, m![{ QuarterRows }, Hidden % 80]>;
/// Every row's normalized FP8 activation in HBM, a slice's rows of one 80-feature group one run.
type ChunkRows = HbmTensor<f8e4m3, Chips, m![Hidden / 80, Capacity, Hidden % 80]>;
type Activation = DmTensor<f8e4m3, Chips, Cluster, Slice, m![Block, Hidden % 80]>;
type Packet8 = m![Hidden % 8];
type PairSpace = crate::compose::Tile<Block, Block>;
type QuerySpace = crate::compose::Tile<Block, Rows>;

impl crate::ops::attention::cache::write::Mapping for PairSpace {
    type Page = m![Page];
    type Payload = m![PoolHeads, PoolKv, PoolHead];
}

macro_rules! stationary {
    ($space:ty, $weight:ty, $reduced:ty, $feed:ty, $sums:ty) => {
        impl crate::ops::linear::contraction::staged::Stationary for $space {
            type Cluster = Cluster;
            type Slices = Slice;
            type Reduced = $reduced;
            type Weight = $weight;
            type Register = m![Block / 16, Hidden % 80 / 16, Block % 16 / 8, Hidden % 16];
            type Lanes = Lanes;
            type Feed = $feed;
            type Packet = Packet;
            type Flits = m![{ $feed }, Out % 4 / 2];
            type Flit = Flit;
            type Steps = m![{ $feed }, { Steps }];
            type Contracted = m![Out % 4];
            type TransposeTime = m![{ Features }, Block % 8];
            type Sums = $sums;
            type Groups = m![{ $sums }, Out % 4];
            type Output = m![{ Features }, Out % 4];
            type Transposed = Row;
        }
    };
}
stationary!(
    PairSpace,
    m![{ contraction::Lanes }, { contraction::Elements }],
    PairRows,
    PairFeed,
    PairSums
);
stationary!(
    QuerySpace,
    m![{ contraction::Lanes }, Panel, { contraction::Elements }],
    QueryRows,
    QueryFeed,
    QuerySums
);

/// One block's rows on their way to HBM: landed, then normalized.
struct Prepared {
    landed: Landed,
    normalized: Normalized,
}

impl Prepared {
    fn new() -> Self {
        Self {
            landed: Landed::new(),
            normalized: Normalized::new(),
        }
    }
}

/// Both commands' epilogue scales. The K/V command streams `Weights::pairs` regrouped by head
/// (head `h`'s K or V in output group `2 Kv + h % 2` of cluster `h / 2`), so it reduces onto whole
/// heads; the Q command streams both panels of the decode waves' Q weight.
struct Projection {
    paired: scale::Epilogue<Cluster, PairRows, Chips>,
    queried: scale::Epilogue<Cluster, QueryRows, Chips>,
}

/// A K/V weight's scale as `land_pairs` lands it.
pub(crate) type PairScale = DmTensor<f32, Chips, m![Out / 512], m![Out % 128 / 64, 128], m![1 # 8]>;

/// Lands `layer`'s K/V weight regrouped by head into `weights.pairs`, a K80 block's 640-byte
/// runs, and its scale: transfers alone, so they can land under an earlier phase.
pub(crate) fn land_pairs(
    device: &mut Device,
    weights: &mut Weights,
    layer: (&stack::Layers, usize),
) -> PairScale {
    let mut scale = PairScale::new();
    land_pairs_into(device, weights, layer, &mut scale);
    scale
}

/// `land_pairs` into a scale allocated before the transfers are issued.
pub(crate) fn land_pairs_into(
    device: &mut Device,
    weights: &mut Weights,
    (layers, index): (&stack::Layers, usize),
    scale: &mut PairScale,
) {
    // SAFETY: relabels only: output group `2 Kv + KvHeads % 2` holds one K80 output group of
    // 128 features in feature order.
    let grouped = unsafe {
        weights
            .pairs
            .view_mut()
            .reshape::<Chips, m![Out / 512], m![Out % 128 / 64, Out % 512 / 256, Hidden / 80], m![
                Out % 8,
                Out % 256 / 128,
                Out % 64 / 32,
                Hidden % 80 / 16,
                Out % 32 / 8,
                Hidden % 16
            ]>()
    };
    layers.pairs.at(index).to_dm_view(&mut device.tdma, grouped);
    layers
        .pair_scale
        .at(index)
        .tile::<m![Out % 512 / 128], 1, m![
            Depth = 1 # Depth,
            Out / 512,
            Out % 512 / 128 = 1 # 4,
            Out % 128 / 64,
            Out % 64 / 32,
            Rows / 4,
            1 # 8
        ]>(0)
        .tile::<m![Out % 64 / 32], 1, m![
            Depth = 1 # Depth,
            Out / 512,
            Out % 512 / 128 = 1 # 4,
            Out % 128 / 64,
            Out % 64 / 32 = 1 # 2,
            Rows / 4,
            1 # 8
        ]>(0)
        .tile::<m![Rows / 4], 1, m![
            Depth = 1 # Depth,
            Out / 512,
            Out % 512 / 128 = 1 # 4,
            Out % 128 / 64,
            Out % 64 / 32 = 1 # 2,
            Rows / 4 = 1 # 16,
            1 # 8
        ]>(0)
        .to_dm_view(&mut device.tdma, scale.view_mut());
}

impl Projection {
    /// Lands `layer`'s K/V weight regrouped by head into `weights.pairs`, a K80 block's 640-byte
    /// runs, and its scales: this layer's decode waves have read the K80 form, and the next
    /// layer's lands after the chunk.
    fn land(device: &mut Device, weights: &mut Weights, layer: (&stack::Layers, usize)) -> Self {
        let scale = land_pairs(device, weights, layer);
        Self::landed(device, weights, &scale)
    }

    /// The projection's registers over a K/V weight and scale `land_pairs` landed.
    fn landed(device: &mut Device, weights: &Weights, scale: &PairScale) -> Self {
        // SAFETY: a scale's `Kv` is its output group's outer digit, every slice of the group holds it.
        let scale = unsafe {
            scale
                .view()
                .reshape::<Chips, Cluster, PairRows, m![1 # 8]>()
        };
        Self {
            paired: scale::Epilogue {
                input: weights.scales.get(device, 0),
                weight: device
                    .sub
                    .begin(scale)
                    .fetch::<m![1], m![1 # 8]>()
                    .collect::<m![1], m![1 # 8]>()
                    .to_vrf(),
            },
            queried: scale::Epilogue {
                input: weights.scales.get(device, 0),
                weight: weights.scales.get(device, 1),
            },
        }
    }

    /// The block's K/V command over the regrouped `pairs`, into `paired`.
    fn contract_pairs(
        &self,
        device: &mut Device,
        pairs: &contraction::Weight<Chips>,
        rows: &Register,
        paired: &mut Paired,
    ) {
        crate::ops::linear::Affine::new(pairs, &self.paired).forward((
            device,
            crate::ops::layout::Dm::<PairSpace, _> {
                tiles: crate::compose::Tiles::over(Block::SIZE),
                value: (rows, paired.view_mut()),
            },
        ));
    }

    /// The block's Q command over both panels of `query`, into `queried`.
    fn contract_queries(
        &self,
        device: &mut Device,
        query: &query::Weight<Chips>,
        rows: &Register,
        queried: &mut Queried,
    ) {
        crate::ops::linear::Affine::new(query, &self.queried).forward((
            device,
            crate::ops::layout::Dm::<QuerySpace, _> {
                tiles: crate::compose::Tiles::over(Block::SIZE),
                value: (rows, queried.view_mut()),
            },
        ));
    }
}

/// A block's outputs as its commands leave them, its rotation coefficients and its rows' pool
/// offsets, landed a block ahead of the scatter that reads them as indices.
struct Outputs {
    queried: Queried,
    paired: Paired,
    spread: Spread,
    entries: HbmTensor<i32, Chips, m![Block]>,
}

/// A block's rotated Q, and its rotated K and its V in E4M3 for its records.
struct Turns {
    queries: Turned,
    codes: (Codes, Codes),
}

/// Where the chunk's records go: layer `entry`'s pages of the pool, at each row's byte offset.
pub struct Cache<'a> {
    pub pool: &'a mut attention::Pool,
    pub entry: usize,
    pub offsets: &'a HbmTensor<i32, Chips, m![Capacity]>,
}

impl Outputs {
    fn new() -> Self {
        Self {
            queried: Queried::new(),
            paired: Paired::new(),
            spread: Spread::new(),
            entries: HbmTensor::new(),
        }
    }

    /// Lands the coefficients and pool offsets of the block of four waves from `wave` on.
    fn land(
        &mut self,
        device: &mut Device,
        (table, offsets, wave): (&Coefficients, &HbmTensor<i32, Chips, m![Capacity]>, usize),
    ) {
        coefficients(table, wave).to_dm_view(&mut device.tdma, self.spread.view_mut());
        let offsets = offsets
            .view()
            .tile::<m![Capacity], 256, m![Capacity = 256 # Capacity]>(64 * wave);
        // SAFETY: the offset tile names exactly the block's rows in order.
        let offsets = unsafe { offsets.reshape::<Chips, m![Block # Capacity]>() };
        offsets.to_hbm_view(&mut device.tdma, self.entries.view_mut());
    }

    /// The block's Q and K rotated against one coefficient fill, then its K and V cast for its
    /// records.
    fn rotate(&self, device: &mut Device) -> Turns {
        // SAFETY: the coefficients of a slice's eight rows sit in every slice of those rows, and
        // a K/V slice keeps its rows where a Q slice does; only the head digits are renamed.
        let (spread, paired) = unsafe {
            (
                self.spread
                    .view()
                    .reshape::<Chips, Cluster, QueryRows, m![OutputHalf, Block % 8, InputHalf, Half]>(),
                self.paired.view().reshape::<Chips, Cluster, QueryRows, m![Block % 8, Head]>(),
            )
        };
        let register: Rotations = rotation::fill(device, spread);
        // SAFETY: a head's 128 features in order.
        let queried = unsafe {
            self.queried
                .view()
                .reshape::<Chips, Cluster, QueryRows, m![Block % 8, Head]>()
        };
        let queries = rotation::apply(device, queried, &register);
        // SAFETY: back to the K/V outputs' names, as above.
        let keys: Paired = unsafe { rotation::apply(device, paired, &register).reshape() };
        let codes = (e4m3(device, &keys), e4m3(device, &self.paired));
        Turns { queries, codes }
    }

    /// The block's rotated Q written into the step's row state at `wave` and its records into the
    /// pool, all the prompt attention reads of it.
    fn write(
        &self,
        device: &mut Device,
        turns: Turns,
        wave: usize,
        (queries, cache): (&mut Queries, &mut Cache<'_>),
    ) {
        store_queries(device, &turns.queries, (queries, wave));
        let mut records = Records::new();
        copy(device, &turns.codes.0, (&mut records, 0));
        copy(device, &turns.codes.1, (&mut records, 1));
        crate::ops::attention::cache::Scatter.forward((
            device,
            (
                &mut *cache.pool,
                (cache.entry, &self.entries),
                crate::ops::layout::Dm::<PairSpace, _> {
                    tiles: crate::compose::Tiles::over(Block::SIZE),
                    value: records.view(),
                },
            ),
        ));
    }
}

/// Q, K and V of the `blocks` blocks of four waves from wave `start` on: Q projected and rotated
/// into the step's row state at their rows, K and V into `cache` as the pool's records.
/// When `land` is 1, `layer`'s Q weight lands here (its gain and scales already have), after
/// block 0's K/V command is issued, so it shares the queue with nothing block 0 waits on.
pub fn chunk(
    device: &mut Device,
    weights: &mut Weights,
    (layer, land): ((&stack::Layers, usize), usize),
    (hidden, start, blocks): (&HiddenRows, usize, usize),
    table: &Coefficients,
    queries: &mut Queries,
    mut cache: Cache<'_>,
) {
    let mut rows = ChunkRows::new();
    // One guard for both phases: their rows' publication then stays inside it.
    for _ in 0..blocks.min(1) {
        normalize_all(device, weights, (hidden, start, blocks), &mut rows);
        contract_all(
            device,
            (weights, (layer, land)),
            (&mut rows, start, blocks),
            (table, &mut *queries),
            &mut cache,
        );
    }
}

/// `chunk` whose K/V weight and `scale` the layer before landed (`land_pairs_into`) unless `own`
/// is 1, when the chunk lands them behind its first block's rows.
pub fn chunk_carried(
    device: &mut Device,
    weights: &mut Weights,
    (layer, land): ((&stack::Layers, usize), usize),
    (hidden, start, blocks): (&HiddenRows, usize, usize),
    (scale, own): (&mut PairScale, usize),
    table: &Coefficients,
    queries: &mut Queries,
    mut cache: Cache<'_>,
) {
    let mut rows = ChunkRows::new();
    for _ in 0..blocks.min(1) {
        normalize_all(device, weights, (hidden, start, blocks), &mut rows);
        contract_all(
            device,
            (weights, (layer, land)),
            (
                &mut Carried {
                    rows: &mut rows,
                    scale: &mut *scale,
                    own,
                },
                start,
                blocks,
            ),
            (table, &mut *queries),
            &mut cache,
        );
    }
}

/// Chunk rows whose K/V weight lands with them only if `own` is 1.
struct Carried<'a> {
    rows: &'a mut ChunkRows,
    scale: &'a mut PairScale,
    own: usize,
}

impl Source for Carried<'_> {
    type Pending = ();

    fn send(&mut self, _device: &mut Device, _wave: usize) {}

    fn reload(
        &mut self,
        device: &mut Device,
        pending: &mut (),
        wave: usize,
        activation: &mut Activation,
    ) {
        self.rows.reload(device, pending, wave, activation);
    }

    fn projection(
        &mut self,
        device: &mut Device,
        weights: &mut Weights,
        layer: (&stack::Layers, usize),
    ) -> Projection {
        for _ in 0..self.own {
            land_pairs_into(device, weights, layer, &mut *self.scale);
        }
        Projection::landed(device, weights, self.scale)
    }
}

/// `chunk` over rows another layer already normalized and gathered: `rows` block `b` holds the
/// normalized FP8 rows of the four waves from `start + 4 b` on.
pub fn chunk_gathered(
    device: &mut Device,
    weights: &mut Weights,
    (layer, land): ((&stack::Layers, usize), usize),
    (rows, start, blocks): (&LayerRows, usize, usize),
    table: &Coefficients,
    queries: &mut Queries,
    mut cache: Cache<'_>,
) {
    for _ in 0..blocks.min(1) {
        contract_all(
            device,
            (weights, (layer, land)),
            (&mut Gathered { rows, start }, start, blocks),
            (table, &mut *queries),
            &mut cache,
        );
    }
}

/// `chunk_gathered` over every block's FP8 half the D-sharded layer before left in data memory
/// (`dlayer::Halves`, block 0 the chunk's first): each block's halves are exchanged with the peer
/// chip and placed into the activation under the block before's commands, no HBM round trip.
pub fn chunk_resident(
    device: &mut Device,
    weights: &mut Weights,
    (layer, land): ((&stack::Layers, usize), usize),
    (halves, blocks): (&mut dlayer::Halves, usize),
    table: &Coefficients,
    queries: &mut Queries,
    mut cache: Cache<'_>,
) {
    for _ in 0..blocks.min(1) {
        contract_all(
            device,
            (weights, (layer, land)),
            (
                &mut Resident {
                    halves: &mut *halves,
                },
                0,
                blocks,
            ),
            (table, &mut *queries),
            &mut cache,
        );
    }
}

/// `chunk_gathered` over rows in the D-sharded layer's own order (`dlayer::NextRows`, block 0 at
/// wave 0): the reload regroups each block from HBM.
pub fn chunk_next(
    device: &mut Device,
    weights: &mut Weights,
    (layer, land): ((&stack::Layers, usize), usize),
    (rows, blocks): (&dlayer::NextRows, usize),
    table: &Coefficients,
    queries: &mut Queries,
    mut cache: Cache<'_>,
) {
    for _ in 0..blocks.min(1) {
        contract_all(
            device,
            (weights, (layer, land)),
            (&mut Next { rows }, 0, blocks),
            (table, &mut *queries),
            &mut cache,
        );
    }
}

/// A chunk's normalized FP8 rows, brought into a block's activation under the block before's
/// commands.
trait Source {
    /// The reload waits for a transfer `send` starts one block ahead, so it goes under the block
    /// before's Q command instead of its K/V command.
    const LATE: bool = false;
    /// What `send` leaves in flight for the reload.
    type Pending;
    /// Starts what the reload of the block from `wave` on waits for.
    fn send(&mut self, device: &mut Device, wave: usize) -> Self::Pending;
    /// The block of four waves from `wave` on into every output group's slices.
    fn reload(
        &mut self,
        device: &mut Device,
        pending: &mut Self::Pending,
        wave: usize,
        activation: &mut Activation,
    );
    /// The chunk's Q/K/V projection: its K/V weight landed here unless the source carries one.
    fn projection(
        &mut self,
        device: &mut Device,
        weights: &mut Weights,
        layer: (&stack::Layers, usize),
    ) -> Projection {
        Projection::land(device, weights, layer)
    }
    /// The chunk's first block (from `wave` on) into `activation`.
    fn first(&mut self, device: &mut Device, wave: usize, activation: &mut Activation) {
        let mut pending = self.send(device, wave);
        self.reload(device, &mut pending, wave, activation);
    }
}

impl Source for ChunkRows {
    type Pending = ();

    fn send(&mut self, _device: &mut Device, _wave: usize) {}

    fn reload(
        &mut self,
        device: &mut Device,
        _pending: &mut (),
        wave: usize,
        activation: &mut Activation,
    ) {
        let tile = self.view().tile::<m![Capacity], 256, m![
            Hidden / 80,
            Capacity = 256 # Capacity,
            Hidden % 80
        ]>(64 * wave);
        // SAFETY: the 256-row tile keeps the capacity stride; only its row axis is renamed.
        let tile =
            unsafe { tile.reshape::<Chips, m![Hidden / 80, Block # Capacity, Hidden % 80]>() };
        tile.to_dm_view(&mut device.tdma, activation.view_mut());
    }
}

/// K80-blocked rows whose block 0 is the chunk's wave `start`.
struct Gathered<'a> {
    rows: &'a LayerRows,
    start: usize,
}

impl Source for Gathered<'_> {
    type Pending = ();

    fn send(&mut self, _device: &mut Device, _wave: usize) {}

    fn reload(
        &mut self,
        device: &mut Device,
        _pending: &mut (),
        wave: usize,
        activation: &mut Activation,
    ) {
        let tile = self.rows.view().tile::<m![LayerBlocks], 1, m![
            LayerBlocks = 1 # LayerBlocks,
            Hidden / 80,
            LayerBlock,
            Hidden % 80
        ]>((wave - self.start) / 4);
        // SAFETY: a block's 256 rows in order; only the row axis is renamed.
        let tile = unsafe {
            tile.reshape::<Chips, m![LayerBlocks = 1 # LayerBlocks, Hidden / 80, Block, Hidden % 80]>()
        };
        tile.to_dm_view(&mut device.tdma, activation.view_mut());
    }
}

/// Rows in the D-sharded layer's own order, block 0 at wave 0.
struct Next<'a> {
    rows: &'a dlayer::NextRows,
}

impl Source for Next<'_> {
    type Pending = ();

    fn send(&mut self, _device: &mut Device, _wave: usize) {}

    fn reload(
        &mut self,
        device: &mut Device,
        _pending: &mut (),
        wave: usize,
        activation: &mut Activation,
    ) {
        let tile = self.rows.view().tile::<m![LayerBlocks], 1, m![
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
        ]>(wave / 4);
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
}

/// Rows in the D-sharded layer's own order whose halves cross the link on the way: block `j`'s
/// held half is sent to the peer one block ahead (`send`), both halves placed into `rows`
/// (`dlayer::Peer::place_next`) and reloaded under the block before's Q command.
struct Exchanging<'a> {
    rows: &'a mut dlayer::NextRows,
    chip: usize,
    scale: &'a PairScale,
}

impl Source for Exchanging<'_> {
    const LATE: bool = true;
    type Pending = dlayer::Peer;

    fn send(&mut self, device: &mut Device, wave: usize) -> Self::Pending {
        dlayer::send_own(device, &*self.rows, (wave / 4, self.chip))
    }

    fn reload(
        &mut self,
        device: &mut Device,
        peer: &mut Self::Pending,
        wave: usize,
        activation: &mut Activation,
    ) {
        peer.place_half(device, (&mut *self.rows, wave / 4, self.chip));
        Next { rows: &*self.rows }.reload(device, &mut (), wave, activation);
    }

    /// The first block's halves were placed by the layer before the chunk.
    fn first(&mut self, device: &mut Device, wave: usize, activation: &mut Activation) {
        Next { rows: &*self.rows }.reload(device, &mut (), wave, activation);
    }

    /// The K/V weight landed under the layer's down (`land_pairs`).
    fn projection(
        &mut self,
        device: &mut Device,
        weights: &mut Weights,
        _layer: (&stack::Layers, usize),
    ) -> Projection {
        Projection::landed(device, weights, self.scale)
    }
}

/// `chunk_next` with each block's next-layer half exchanged inside the chunk (`Exchanging`), but
/// the first block's, which the layer placed into `rows` already.
pub fn chunk_exchanged(
    device: &mut Device,
    weights: &mut Weights,
    (layer, land): ((&stack::Layers, usize), usize),
    (rows, chip, scale, blocks): (&mut dlayer::NextRows, usize, &PairScale, usize),
    table: &Coefficients,
    queries: &mut Queries,
    mut cache: Cache<'_>,
) {
    for _ in 0..blocks.min(1) {
        contract_all(
            device,
            (weights, (layer, land)),
            (
                &mut Exchanging {
                    rows: &mut *rows,
                    chip,
                    scale,
                },
                0,
                blocks,
            ),
            (table, &mut *queries),
            &mut cache,
        );
    }
}

/// The next layer's fused Q/K/V weight (`fused::land`) and norm, landed under the layer's down.
pub(crate) struct NextFused<'a> {
    pub(crate) weights: &'a mut Weights,
    pub(crate) fused: &'a mut fused::Weight,
    pub(crate) scales: &'a mut fused::Seeds,
    pub(crate) layer: (&'a stack::Layers<'a>, usize),
}

impl dlayer::Prefetch for NextFused<'_> {
    fn land(&mut self, device: &mut Device) {
        fused::land(device, self.fused, self.layer);
        fused::land_scales(device, self.scales, self.layer);
        self.weights.land_norm(device, self.layer);
    }
}

/// `chunk_exchanged` with Q, K and V of each block in one command over the fused weight in the
/// register file (`fused`): each block's halves land once a cluster, the cluster's feature half.
pub fn chunk_fused(
    device: &mut Device,
    (weights, (weight, scales)): (&Weights, (&fused::Weight, &fused::Seeds)),
    (rows, chip, start, blocks): (&mut dlayer::NextRows, usize, usize, usize),
    table: &Coefficients,
    queries: &mut Queries,
    mut cache: Cache<'_>,
) {
    for _ in 0..blocks.min(1) {
        let (mut even, mut odd) = (fused::Activation::new(), fused::Activation::new());
        let (mut first, mut second) = (fused::Outputs::new(), fused::Outputs::new());
        fused::reload(device, &*rows, 0, &mut even);
        let factors = fused::Factors::new(device, &weights.scales, scales);
        let register = fused::fill(device, weight);
        let projection = crate::ops::linear::Affine::new(register, factors);
        let partial = projection
            .forward((
                &mut *device,
                crate::ops::layout::Dm::<crate::compose::Tile<Block, Block>, _> {
                    tiles: crate::compose::Tiles::over(Block::SIZE),
                    value: &even,
                },
            ))
            .value;
        // A one-block chunk sends its block again, which no reload reads.
        let pending = dlayer::send_own(device, &*rows, ((blocks - 1).min(1), chip));
        first.land(device, (table, cache.offsets, start));
        let crossed = fused::cross(device, &partial);
        fused::end(device, &projection, crossed, &mut first);
        for _ in 0..(blocks - 1).min(1) {
            pending.place_half(device, (&mut *rows, 1, chip));
            fused::reload(device, &*rows, 1, &mut odd);
        }
        for _ in 0..1 - (blocks - 1).min(1) {
            first.rotate(device);
            first.write(device, start, (&mut *queries, &mut cache));
        }
        let run = &projection;
        for pair in 0..(blocks - 1).div_ceil(2) {
            let block = 1 + 2 * pair;
            let more = (blocks - block - 1).min(1);
            fused_block(
                device,
                run,
                (&odd, block),
                (&mut *rows, chip, start, &mut even, more),
                (&mut first, &mut second),
                (table, &mut *queries),
                &mut cache,
            );
            for _ in 0..more {
                let more = (blocks - block - 2).min(1);
                fused_block(
                    device,
                    run,
                    (&even, block + 1),
                    (&mut *rows, chip, start, &mut odd, more),
                    (&mut second, &mut first),
                    (table, &mut *queries),
                    &mut cache,
                );
            }
        }
    }
}

/// Block `block`'s fused command against `current`, the next block's halves crossing the link
/// and landing into `next` (when `more` is 1). Block `block - 2` (in `outputs`, whose buffers this
/// block reuses) is written behind the command and the send, so the command processor reaches
/// the next block's command right after its landing; the block `before` rotates under the
/// command's partials. The last block writes the one before and itself.
fn fused_block(
    device: &mut Device,
    projection: &fused::Projection,
    (current, block): (&fused::Activation, usize),
    (rows, chip, start, next, more): (
        &mut dlayer::NextRows,
        usize,
        usize,
        &mut fused::Activation,
        usize,
    ),
    (before, outputs): (&mut fused::Outputs, &mut fused::Outputs),
    (table, queries): (&Coefficients, &mut Queries),
    cache: &mut Cache<'_>,
) {
    let wave = start + 4 * block;
    let partial = projection
        .forward((
            &mut *device,
            crate::ops::layout::Dm::<crate::compose::Tile<Block, Block>, _> {
                tiles: crate::compose::Tiles::over(Block::SIZE),
                value: current,
            },
        ))
        .value;
    // Issued behind the command, so the command processor waits for its source under it. The
    // last block sends itself again, which no reload reads.
    let pending = dlayer::send_own(device, &*rows, (block + more, chip));
    for _ in 0..(block - 1).min(1) {
        outputs.write(device, wave - 8, (&mut *queries, &mut *cache));
    }
    outputs.land(device, (table, cache.offsets, wave));
    let crossed = fused::cross(device, &partial);
    before.rotate(device);
    fused::end(device, projection, crossed, outputs);
    for _ in 0..more {
        pending.place_half(device, (&mut *rows, block + 1, chip));
        fused::reload(device, &*rows, block + 1, &mut *next);
    }
    for _ in 0..1 - more {
        before.write(device, wave - 4, (&mut *queries, &mut *cache));
        outputs.rotate(device);
        outputs.write(device, wave, (&mut *queries, &mut *cache));
    }
}

/// Every block's FP8 half of this chip in data memory, block 0 at wave 0.
struct Resident<'a> {
    halves: &'a mut dlayer::Halves,
}

impl Source for Resident<'_> {
    type Pending = ();

    fn send(&mut self, _device: &mut Device, _wave: usize) {}

    fn reload(
        &mut self,
        device: &mut Device,
        _pending: &mut (),
        wave: usize,
        activation: &mut Activation,
    ) {
        let absolute = dlayer::exchanged(device, self.halves, wave / 4);
        // SAFETY: a block's 256 rows in order; only the row axis is renamed.
        let rows = unsafe {
            absolute.view().reshape::<Chips, m![Hidden % 1280 / 640], m![
                Hidden % 640 / 320,
                Block / 8,
                Hidden % 320 / 80
            ], m![Hidden / 2560, Hidden % 2560 / 1280, Block % 8, Hidden % 80 / 8, Hidden % 8]>()
        };
        rows.to_dm_view(&mut device.tdma, activation.view_mut());
    }
}

/// Phases 2 and 3: every block's K/V and Q commands against its rows in the register file, the
/// next block's rows reloading under them (even blocks in one buffer, odd blocks in the other);
/// each block rotates between the next block's two commands and is written under its Q command,
/// and the last block rotates and writes its own.
fn contract_all<R: Source>(
    device: &mut Device,
    (weights, (layer, land)): (&mut Weights, ((&stack::Layers, usize), usize)),
    (rows, start, blocks): (&mut R, usize, usize),
    (table, queries): (&Coefficients, &mut Queries),
    cache: &mut Cache<'_>,
) {
    let (mut even, mut odd) = (Activation::new(), Activation::new());
    let (mut first, mut second) = (Outputs::new(), Outputs::new());
    rows.first(device, start, &mut even);
    let projection = rows.projection(device, weights, layer);
    let mut pending = rows.send(device, start + 4 * (blocks - 1).min(1));
    // Block 0's K/V command is issued once its rows are back, ahead of a prompt-only step's Q
    // weight, which then lands alone instead of sharing the queue with those rows.
    let register = fill(device, &even);
    projection.contract_pairs(device, &weights.pairs, &register, &mut first.paired);
    for _ in 0..land {
        weights.land_query(device, layer);
    }
    let weights = &*weights;
    first.land(device, (table, cache.offsets, start));
    if !R::LATE {
        for _ in 0..(blocks - 1).min(1) {
            rows.reload(device, &mut pending, start + 4, &mut odd);
        }
    }
    projection.contract_queries(device, &weights.query, &register, &mut first.queried);
    if R::LATE {
        for _ in 0..(blocks - 1).min(1) {
            rows.reload(device, &mut pending, start + 4, &mut odd);
        }
    }
    for _ in 0..1 - (blocks - 1).min(1) {
        let turns = first.rotate(device);
        first.write(device, turns, start, (&mut *queries, &mut *cache));
    }
    for pair in 0..(blocks - 1).div_ceil(2) {
        let block = 1 + 2 * pair;
        let more = (blocks - block - 1).min(1);
        contract_block(
            device,
            (&projection, weights),
            (&odd, block),
            (&mut *rows, start, &mut even, more),
            (&first, &mut second),
            (table, &mut *queries),
            &mut *cache,
        );
        for _ in 0..more {
            let more = (blocks - block - 2).min(1);
            contract_block(
                device,
                (&projection, weights),
                (&even, block + 1),
                (&mut *rows, start, &mut odd, more),
                (&second, &mut first),
                (table, &mut *queries),
                &mut *cache,
            );
        }
    }
}

/// Block `block`'s K/V and Q commands against `current`, its coefficients and the next block's
/// rows (when `more` is 1, into `next`) landing under them; the block `before` rotates between the
/// two commands and is written under the Q command, and the last block rotates and writes its own.
fn contract_block<R: Source>(
    device: &mut Device,
    (projection, weights): (&Projection, &Weights),
    (current, block): (&Activation, usize),
    (rows, start, next, more): (&mut R, usize, &mut Activation, usize),
    (before, outputs): (&Outputs, &mut Outputs),
    (table, queries): (&Coefficients, &mut Queries),
    cache: &mut Cache<'_>,
) {
    let wave = start + 4 * block;
    // A late source's last block sends itself again, which no reload reads.
    let mut pending = rows.send(device, wave + 4 * more);
    let register = fill(device, current);
    projection.contract_pairs(device, &weights.pairs, &register, &mut outputs.paired);
    outputs.land(device, (table, cache.offsets, wave));
    if !R::LATE {
        for _ in 0..more {
            rows.reload(device, &mut pending, wave + 4, &mut *next);
        }
    }
    let turns = before.rotate(device);
    projection.contract_queries(device, &weights.query, &register, &mut outputs.queried);
    if R::LATE {
        for _ in 0..more {
            rows.reload(device, &mut pending, wave + 4, &mut *next);
        }
    }
    before.write(device, turns, wave - 4, (&mut *queries, &mut *cache));
    for _ in 0..1 - more {
        let turns = outputs.rotate(device);
        outputs.write(device, turns, wave, (&mut *queries, &mut *cache));
    }
}

/// A block's rows in the register file, eight a lane.
fn fill(device: &mut Device, rows: &Activation) -> Register {
    device
        .sub
        .begin(rows.view())
        .fetch::<m![Block % 8, Block / 16, Hidden % 80 / 16], m![Block % 16 / 8, Hidden % 16]>()
        .collect::<m![Block % 8, Block / 16, Hidden % 80 / 16], m![Block % 16 / 8, Hidden % 16]>()
        .to_trf()
}

/// A block's rotated Q written into its rows of the step's row state.
fn store_queries(device: &mut Device, rows: &Turned, (queries, wave): (&mut Queries, usize)) {
    // SAFETY: head `2 g + p` of a row is panel `p`'s head `g`.
    let heads = unsafe {
        queries
            .view_mut()
            .reshape::<Chips, m![Capacity, Heads, Panel, Head]>()
    };
    let tile = heads.tile::<m![Capacity], 256, m![
        Capacity = 256 #{!} Capacity,
        Heads,
        Panel,
        Head
    ]>(wave * 64);
    // SAFETY: the 256-row tile keeps the capacity stride; only its row axis is renamed.
    let tile = unsafe { tile.reshape::<Chips, m![Block #{!} Capacity, Heads, Panel, Head]>() };
    // SAFETY: a panel's 1,024 outputs are its eight heads of 128 features in order.
    let rows = unsafe {
        rows.view()
            .reshape::<Chips, m![Heads / 4], m![Heads % 4, Block / 8, Panel], m![Block % 8, Head]>()
    };
    rows.to_hbm_view(&mut device.tdma, tile);
}

/// A block's K/V outputs in E4M3.
fn e4m3(device: &mut Device, rows: &Paired) -> Codes {
    // SAFETY: a head's 128 features in order.
    let rows = unsafe {
        rows.view()
            .reshape::<Chips, Cluster, PairRows, m![Block % 8, Head]>()
    };
    device
        .main
        .begin(rows)
        .fetch::<m![Block % 8, Head / 8], m![Head % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Block % 8, Head / 8], m![Head % 8]>()
        .cast::<f8e4m3, m![Head % 8 # 32]>()
        .commit_trim::<m![Head % 8]>()
        .commit()
}

/// Half `kv` of every row's record (0 the keys, 1 the values) from a block's E4M3 K/V outputs.
fn copy(device: &mut Device, codes: &Codes, (records, kv): (&mut Records, usize)) {
    // SAFETY: output group `2 Kv + KvHeads % 2` of cluster `KvHeads / 2`, under the pool's names.
    let codes = unsafe {
        codes
            .view()
            .reshape::<Chips, m![PoolHeads / 2], PoolRows, m![Block % 8, PoolHead]>()
    };
    codes
        .slice_tile::<m![PoolKv], 1, m![PoolKv = 1 # 2, PoolHeads % 2, Block / 8, Replica]>(kv)
        .slice_tile::<m![Replica], 1, m![PoolKv = 1 # 2, PoolHeads % 2, Block / 8, Replica = 1 # 2]>(
            0,
        )
        .to_dm_view(
            &mut device.tdma,
            records
                .view_mut()
                .tile::<m![PoolKv], 1, m![PoolHeads % 2, PoolKv = 1 #{!} 2, PoolHead]>(kv),
        );
}

/// Phase 1: every block landed, normalized and stored into `rows`, the next block landing while
/// the current one normalizes: even blocks in one set of buffers, odd blocks in the other.
fn normalize_all(
    device: &mut Device,
    weights: &Weights,
    (hidden, start, blocks): (&HiddenRows, usize, usize),
    rows: &mut ChunkRows,
) {
    let (mut even, mut odd) = (Prepared::new(), Prepared::new());
    for _ in 0..blocks.min(1) {
        land(device, (hidden, start), &mut even.landed);
    }
    for pair in 0..blocks.div_ceil(2) {
        let block = 2 * pair;
        let wave = start + 4 * block;
        let more = (blocks - block - 1).min(1);
        prepare(
            device,
            weights,
            (hidden, wave),
            (&mut even, &mut odd, more),
            rows,
        );
        for _ in 0..more {
            let more = (blocks - block - 2).min(1);
            prepare(
                device,
                weights,
                (hidden, wave + 4),
                (&mut odd, &mut even, more),
                rows,
            );
        }
    }
}

/// The block of four waves from `wave` on normalized out of `current` and stored into `rows`, with
/// the next block (when `more` is 1) landing into `next` meanwhile.
fn prepare(
    device: &mut Device,
    weights: &Weights,
    (hidden, wave): (&HiddenRows, usize),
    (current, next, more): (&mut Prepared, &mut Prepared, usize),
    rows: &mut ChunkRows,
) {
    for _ in 0..more {
        land(device, (hidden, wave + 4), &mut next.landed);
    }
    normalize(device, weights, &current.landed, &mut current.normalized);
    store(device, &current.normalized, (rows, wave));
}

/// Phases 2 and 3: every block's K/V contraction into `pairs` and Q contraction into `queried`,
/// each block's rows brought back from `rows` while the block before contracts (even blocks in
/// one buffer, odd blocks in the other), and each block's outputs relaid, rotated and written

/// The block of four waves from `wave` on into `landed`, half of its rows a cluster: a slice holds
/// 32 rows of one 80-feature group, a single run of the slice-blocked state.
fn land(device: &mut Device, (hidden, wave): (&HiddenRows, usize), landed: &mut Landed) {
    let rows = hidden
        .view()
        .tile::<m![Waves], 4, m![Waves = 4 # Waves, { Blocked }]>(wave);
    // SAFETY: the four waves are consecutive, so their rows are one 256-row block; only the wave
    // and row labels are renamed.
    let rows = unsafe {
        rows.reshape::<Chips, m![Block / 64 # Waves, Hidden / 80, Block % 64, Hidden % 80]>()
    };
    rows.to_dm_view(&mut device.tdma, landed.view_mut());
}

/// RMS-normalizes the landed block, applies the gain and rounds straight to FP8 by the input scale
/// into `normalized`, each slice its quarter of the rows.
fn normalize(device: &mut Device, weights: &Weights, landed: &Landed, normalized: &mut Normalized) {
    // SAFETY: the gain depends only on a slice's features, whose digit both layouts keep innermost;
    // only the cluster and the outer digit of the slice index are renamed.
    let gain = unsafe {
        weights
            .gain
            .weight()
            .view()
            .reshape::<Chips, RowHalf, Quarter, m![Hidden % 80]>()
    };
    let norm = crate::ops::normalization::RmsNorm::<
        Hidden,
        { 1e-5_f32.to_bits() as usize },
        Chips,
        _,
    >::new(gain);
    let gain = norm.gain::<m![Hidden % 80 / 8], Packet8>(device);
    let divisor =
        crate::ops::linear::Affine::new((), weights.scales.get::<RowHalf, Quarter>(device, 0));
    let input = crate::ops::layout::Dm::<crate::compose::Tile<Block, Block>, _> {
        tiles: crate::compose::Tiles::over(Block::SIZE),
        value: landed.view(),
    };
    norm.replicated::<_, QuarterRows, m![Hidden % 80], m![Block % 128 / 32, 64], m![1 # 2], false, _>(device, &input)
        .inverse(device)
        .apply::<QuarterTime, Packet8, m![{ QuarterRows }, Hidden % 80 / 4]>(device, &gain)
        .value
        .vector_fp_div(divisor.input_scale())
        .vector_widen_concat::<QuarterTime, Packet8>()
        .vector_final()
        .cast::<f8e4m3, m![Hidden % 8 # 32]>()
        .commit_trim::<Packet8>()
        .commit_view(normalized.view_mut());
}

/// The cluster's normalized half of the block of four waves from `wave` on into `rows`, a
/// slice's quarter one 2.5-KiB run.
fn store(device: &mut Device, normalized: &Normalized, (rows, wave): (&mut ChunkRows, usize)) {
    let tile = rows.view_mut().tile::<m![Capacity], 256, m![
        Hidden / 80,
        Capacity = 256 #{!} Capacity,
        Hidden % 80
    ]>(64 * wave);
    // SAFETY: the 256-row tile keeps the capacity stride; only its row axis is renamed.
    let tile =
        unsafe { tile.reshape::<Chips, m![Hidden / 80, Block #{!} Capacity, Hidden % 80]>() };
    normalized.view().to_hbm_view(&mut device.tdma, tile);
}

/// The rotation coefficients of the block's 256 rows.
fn coefficients(
    table: &Coefficients,
    wave: usize,
) -> HbmTensorView<
    '_,
    bf16,
    Chips,
    m![
        Block / 64 # Waves,
        Block % 64,
        OutputHalf,
        InputHalf,
        Half
    ],
> {
    let rows = table
        .view()
        .tile::<m![Waves], 4, m![Waves = 4 # Waves, { rotation::Stored }]>(wave);
    // SAFETY: four consecutive waves of 64 rows are one block of 256; only the labels change.
    unsafe { rows.reshape() }
}
