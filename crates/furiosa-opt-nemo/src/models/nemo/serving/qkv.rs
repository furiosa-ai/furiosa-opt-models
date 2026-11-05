use crate::Module;
use furiosa_opt_std::prelude::*;

use crate::models::nemo::serving::mlp::{Capacity, Waves};
use crate::models::nemo::serving::{Blocked, Chips, Hidden, Rows, scale, stack};
mod activation;
pub mod contraction;
pub mod query;
pub mod rotation;
pub mod tile;

axes![QHeads = 16, KvHeads = 4, Head = 128, Kv = 2];
use contraction::{Cluster, Slice};

pub(crate) type PairCluster = m![KvHeads / 2];
// Four heads of 64 rows fill half a cluster; DM tensors must span all 256 slices.
pub(crate) type PairSlice = m![KvHeads % 2, Rows # 128];
type PairTime = m![1];
/// One wave's K or V rows in the rotation's layout, one head of one row a slice.
pub type WaveRows = rotation::Rotated<Chips, PairCluster, PairSlice, PairTime>;
type Pairs = DmTensor<
    bf16,
    Chips,
    PairCluster,
    m![KvHeads % 2, Head / 64, Kv, Head % 64 / 32, Rows / 4],
    m![Rows % 4, Head % 32],
>;

pub type PairScale = m![{ Cluster }, { contraction::Reduced }, 1 # 8];
/// One wave of the step's hidden state.
pub type Input<'a> = HbmTensorView<'a, bf16, Chips, m![Waves = 1 # Waves, { Blocked }]>;
pub type Coefficients = HbmTensor<bf16, Chips, m![Waves, { rotation::Stored }]>;
pub type Queries = HbmTensor<bf16, Chips, m![Capacity, QHeads, Head]>;
pub type Keys = HbmTensor<bf16, Chips, m![Capacity, KvHeads, Head]>;

fn coefficients(
    table: &Coefficients,
    wave: usize,
) -> HbmTensorView<'_, bf16, Chips, m![Waves = 1 # Waves, { rotation::Stored }]> {
    table
        .view()
        .tile::<m![Waves], 1, m![Waves = 1 # Waves, { rotation::Stored }]>(wave)
}

/// One wave's hidden rows in the normalization's layout.
pub type Landed = DmTensor<bf16, Chips, Cluster, Slice, m![Rows, Hidden % 80]>;

/// One wave's rotation coefficients landed beside its K rows and beside its Q panels' rows: the
/// same every layer, so a decode step's first wave lands them once.
pub struct Rotations {
    key: rotation::Landed<Chips, PairCluster, PairSlice, PairTime>,
    query: query::Coefficients<Chips>,
}

impl Rotations {
    /// Coefficients not yet landed, for `land_into`.
    pub fn new() -> Self {
        Self {
            key: DmTensor::new(),
            query: DmTensor::new(),
        }
    }

    pub fn land(device: &mut Device, table: &Coefficients, wave: usize) -> Self {
        let mut rotations = Self::new();
        rotations.land_into(device, table, wave);
        rotations
    }

    pub fn land_into(&mut self, device: &mut Device, table: &Coefficients, wave: usize) {
        coefficients(table, wave).to_dm_view(&mut device.tdma, self.key.view_mut());
        coefficients(table, wave).to_dm_view(&mut device.tdma, self.query.view_mut());
    }
}

/// One layer's Q/K/V weights and scales in data memory, landed once for every wave of a step.
pub struct Weights {
    scales: scale::Scales<Chips>,
    /// The scale row as landed, which `spread_scales` spreads into `scales`.
    seeds: scale::Seeds<Chips>,
    gain: crate::ops::normalization::RmsNorm<
        Hidden,
        { 1e-5_f32.to_bits() as usize },
        Chips,
        DmTensor<bf16, Chips, Cluster, Slice, m![Hidden % 80]>,
    >,
    pair_scale: DmTensor<f32, Chips, Cluster, contraction::Reduced, m![1 # 8]>,
    pairs: contraction::Weight<Chips>,
    query: query::Weight<Chips>,
}

impl Weights {
    /// Weights not yet landed, for `land_weights`, `prefetch` or `land_norm`.
    pub fn new() -> Self {
        Self {
            scales: scale::Scales::new(),
            seeds: scale::Seeds::new(),
            gain: crate::ops::normalization::RmsNorm::new(DmTensor::new()),
            pair_scale: DmTensor::new(),
            pairs: DmTensor::new(),
            query: DmTensor::new(),
        }
    }

    /// Spreads the scale row `land_norm` landed: Main work, so it runs at the layer it serves.
    pub fn spread_scales(&mut self, device: &mut Device) {
        self.scales.spread(device, &self.seeds);
    }

    /// The spread scales `begin` reads.
    pub fn scales(&self) -> &scale::Scales<Chips> {
        &self.scales
    }

    /// Lands `layer`'s scale row, gain and weights: transfers alone, so they can land under an
    /// earlier layer's Main work.
    pub fn land_weights(&mut self, device: &mut Device, layer: (&stack::Layers, usize)) {
        self.land_norm(device, layer);
        self.land_pairs(device, layer);
        self.land_query(device, layer);
    }

    /// `land_weights` but for the second Q panel, which `land_last` lands at the layer's start:
    /// the layer's Q fill follows its norm and K/V fill, so it lands in time. The K/V weight lands
    /// only if `pairs` is 1, else where the caller's queue idles.
    pub fn prefetch(&mut self, device: &mut Device, layer: (&stack::Layers, usize), pairs: usize) {
        self.land_norm(device, layer);
        for _ in 0..pairs {
            self.land_pairs(device, layer);
        }
        query::land_panel(device, layer, (0, &mut self.query));
    }

    pub fn land_last(&mut self, device: &mut Device, layer: (&stack::Layers, usize)) {
        query::land_panel(device, layer, (1, &mut self.query));
    }

    /// Lands `layer`'s scale row and norm gain, all a prompt chunk's normalization needs.
    pub fn land_norm(&mut self, device: &mut Device, (layers, index): (&stack::Layers, usize)) {
        self.seeds.land_into(device, layers.qkv_scales.at(index));
        layers
            .norm
            .at(index)
            .to_dm_view(&mut device.tdma, self.gain.weight_mut().view_mut());
    }

    /// Lands `layer`'s K/V weight and its scale.
    pub fn land_pairs(&mut self, device: &mut Device, (layers, index): (&stack::Layers, usize)) {
        layers
            .pair_scale
            .at(index)
            .to_dm_view(&mut device.tdma, self.pair_scale.view_mut());
        layers
            .pairs
            .at(index)
            .to_dm_view(&mut device.tdma, self.pairs.view_mut());
    }

    /// Lands both panels of `layer`'s Q weight.
    pub fn land_query(&mut self, device: &mut Device, layer: (&stack::Layers, usize)) {
        query::land_into(device, layer, &mut self.query);
    }
}

pub fn land(device: &mut Device, input: Input<'_>) -> Landed {
    input.to_dm(&mut device.tdma)
}

/// `land` into rows allocated before the transfer is issued.
pub fn land_into(device: &mut Device, input: Input<'_>, landed: &mut Landed) {
    input.to_dm_view(&mut device.tdma, landed.view_mut());
}

/// One 64-row wave's Q/K/V through K's rotation: the rotated K rows, the V rows, and both Q
/// panels' rows before their rotation.
pub struct Begun {
    pub key: WaveRows,
    pub value: WaveRows,
    queries: Panels,
}

/// Both Q panels' rows of one wave, one head of one row a slice, before their rotation.
pub struct Panels {
    even: query::Rotation<Chips>,
    odd: query::Rotation<Chips>,
}

/// Normalized Q, K and V of one 64-row wave for this chip's heads, through K's rotation; K/V,
/// then both Q panels in one command, contract against the same landed activation. The wave's K
/// and V rows stay in data memory, for the cache to take them there.
pub fn begin(
    device: &mut Device,
    (weights, scales): (&Weights, &scale::Scales<Chips>),
    rows: &Landed,
    rotations: &Rotations,
) -> Begun {
    let Weights {
        gain,
        pair_scale: weight,
        pairs,
        query,
        ..
    } = weights;
    let divisor = crate::ops::linear::Affine::new((), scales.get::<Cluster, Slice>(device, 0));
    let rows = activation::prepare(device, rows, (gain, &divisor));
    let panel = contraction::fill(device, pairs.view());
    let epilogue = scale::Epilogue::<Cluster, contraction::Reduced, Chips> {
        input: scales.get(device, 0),
        weight: device
            .sub
            .begin(weight.view())
            .fetch::<m![1], m![1 # 8]>()
            .collect::<m![1], m![1 # 8]>()
            .to_vrf(),
    };
    let contracted = crate::ops::linear::Affine::new(panel, epilogue)
        .forward((
            &mut *device,
            crate::ops::layout::Dm {
                value: &rows,
                tiles: crate::compose::Tiles::<crate::compose::Tile<Rows, Rows>>::over(Rows::SIZE),
            },
        ))
        .value;
    // The Q command issues ahead of the K/V copies, whose issue waits for the K/V contraction, so
    // its fill loads under that contraction; the copies and the Q relayouts then queue in order.
    let register = query::fill(device, query);
    let epilogue = scale::Epilogue::<Cluster, contraction::Reduced, Chips> {
        input: scales.get(device, 0),
        weight: scales.get(device, 1),
    };
    let both = crate::ops::linear::Affine::new(register, epilogue)
        .forward((
            &mut *device,
            crate::ops::layout::Dm {
                value: &rows,
                tiles: crate::compose::Tiles::<query::Space>::over(Rows::SIZE),
            },
        ))
        .value;
    // SAFETY: the K/V panel orders each PE group as K then V blocks of 64 features.
    let contracted: Pairs = unsafe { contracted.reshape() };
    let mut key: WaveRows = DmTensor::new();
    contracted
        .view()
        .slice_tile::<m![Kv], 1, m![KvHeads % 2, Head / 64, Kv = 1 # 2, Head % 64 / 32, Rows / 4]>(
            0,
        )
        .to_dm_view(&mut device.tdma, key.view_mut());
    let mut value: WaveRows = DmTensor::new();
    contracted
        .view()
        .slice_tile::<m![Kv], 1, m![KvHeads % 2, Head / 64, Kv = 1 # 2, Head % 64 / 32, Rows / 4]>(
            1,
        )
        .to_dm_view(&mut device.tdma, value.view_mut());
    let even = query::rows(device, &both, 0);
    let key = rotation::rotate(device, &key, &rotations.key);
    let odd = query::rows(device, &both, 1);
    Begun {
        key,
        value,
        queries: Panels { even, odd },
    }
}

impl Begun {
    /// The wave's Q panels, K rows and V rows.
    pub fn split(self) -> (Panels, WaveRows, WaveRows) {
        (self.queries, self.key, self.value)
    }
}

impl Panels {
    /// Both panels rotated and written into their heads of wave `wave`'s rows.
    pub fn store(
        self,
        device: &mut Device,
        (rotations, wave): (&Rotations, usize),
        queries: &mut Queries,
    ) {
        // Both panels' rows sit at the wave's positions: one coefficient fill rotates both.
        let filled = rotation::fill(device, rotations.query.view());
        store_query(device, &self.even, (&filled, wave), (queries, 0));
        store_query(device, &self.odd, (&filled, wave), (queries, 1));
    }
}

/// A wave's K and V rows written into the step's row state at wave `wave`'s rows.
pub fn store_rows(
    device: &mut Device,
    (key, value): (&WaveRows, &WaveRows),
    wave: usize,
    (keys, values): (&mut Keys, &mut Keys),
) {
    value
        .view()
        .to_hbm_view(&mut device.tdma, kv_rows(values, wave));
    key.view()
        .to_hbm_view(&mut device.tdma, kv_rows(keys, wave));
}

/// One Q panel's relaid rows rotated and written into its heads of the wave's rows.
fn store_query(
    device: &mut Device,
    rows: &query::Rotation<Chips>,
    (coefficients, wave): (&query::Filled<Chips>, usize),
    (queries, panel): (&mut Queries, usize),
) {
    let query = rotation::apply(device, rows.view(), coefficients);
    query
        .view()
        .to_hbm_view(&mut device.tdma, query::state(queries, (wave, panel)));
}

fn kv_rows(
    state: &mut Keys,
    wave: usize,
) -> HbmTensorViewMut<'_, bf16, Chips, m![Rows #{!} Capacity, KvHeads, Head]> {
    let tile = state
        .view_mut()
        .tile::<m![Capacity], 64, m![Capacity = 64 #{!} Capacity, KvHeads, Head]>(wave * 64);
    // SAFETY: the 64-row tile keeps the capacity stride; only its row axis is renamed.
    unsafe { tile.reshape::<Chips, m![Rows #{!} Capacity, KvHeads, Head]>() }
}
