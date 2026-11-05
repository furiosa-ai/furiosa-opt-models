use furiosa_opt_std::prelude::*;

use crate::Module;
use crate::models::nemo::serving::{
    Blocked, Chips, Hidden, Rows, Width, attention, head, mlp, norm, output, qkv, residual, scale,
    stack,
};
use attention::{
    Groups as AttentionGroups, Head as AttentionHead, Heads as AttentionHeads, Kv, Page,
    Rows as AttentionRows,
    prompt::{FirstSpans, Prompt, Segments},
};
use head::Vocabulary as ShardVocabulary;
use input::Embedded;
use mlp::{Capacity, Waves};
use qkv::{Head, KvHeads};
use stack::Depth;
use tail::Tail;

pub type Norm =
    crate::models::nemo::Norm<crate::models::nemo::Hidden, crate::ops::layout::axes::Parallel<2>>;

pub mod candidates;
#[cfg(not(feature = "trace"))]
pub(crate) mod dlayer;
mod exchange;
pub mod feed;
mod input;
// The piece entries call the tails without their trace arguments.
#[cfg(not(feature = "trace"))]
pub mod piece;
mod readout;
pub mod sample;
#[cfg(not(feature = "trace"))]
mod sharded;
mod tail;
#[cfg(feature = "trace")]
mod trace;
#[cfg(feature = "trace")]
pub use trace::{Recorded, Trace, Traced};

axes![
    Vocabulary = 131072,
    Pages = 1048576,
    Selected = 256,
    Tile = 128
];
/// The hidden state of every capacity row, one slice-blocked 64-row wave after another.
pub type HiddenRows = HbmTensor<bf16, Chips, m![Waves, { Blocked }]>;
pub type QueryRows = qkv::Queries;
type KvRows = qkv::Keys;
type Cluster = m![Hidden / 2560];
type Slices = m![Hidden % 2560 / 80, Rows / 8];
type Time = m![Rows % 8, Hidden % 80 / 8];
type Packet = m![Hidden % 8];
type Residual = DmTensor<bf16, Chips, Cluster, Slices, m![{ Time }, { Packet }]>;
type Offsets = HbmTensor<i32, Chips, m![Capacity]>;
type Attended<C, D> = DmTensor<D, Chips, C, output::Landing, output::Element>;
/// A wave's attended rows on both clusters, each projecting its own hidden half.
type Projected<D> = Attended<output::Cluster, D>;
#[cfg(feature = "trace")]
type Copied<D> = Attended<m![1 # 2], D>;
const _: () = {
    assert!(Rows::SIZE == AttentionRows::SIZE);
    assert!(Width::SIZE == AttentionHeads::SIZE * AttentionGroups::SIZE * AttentionHead::SIZE);
};

/// The 8-row groups of a wave its TP2 exchanges send: those holding its live rows, rounded up to
/// one, two, four or all eight. Each is a 0/1 trip count and exactly one of them is 1.
#[derive(Clone, Copy)]
pub(crate) struct Live {
    pub(crate) one: usize,
    pub(crate) two: usize,
    pub(crate) four: usize,
    pub(crate) all: usize,
}

impl Live {
    /// A wave whose live rows are its first `rows` (at least one).
    pub(crate) fn of(rows: usize) -> Self {
        let past_one = ((rows - 1) / 8).min(1);
        let (past_two, past_four) = (((rows - 1) / 16).min(1), ((rows - 1) / 32).min(1));
        Self {
            one: 1 - past_one,
            two: past_one * (1 - past_two),
            four: past_two * (1 - past_four),
            all: past_four,
        }
    }

    /// `partial` on its way to the peer chip, only the groups `self` holds.
    pub(crate) fn exchange(
        self,
        device: &mut Device,
        partial: output::Partial,
    ) -> residual::Exchanged<Cluster, Slices, Time, Packet> {
        let mut exchanged = residual::Exchanged::new(partial);
        for _ in 0..self.one {
            One::send(device, &mut exchanged);
        }
        for _ in 0..self.two {
            Two::send(device, &mut exchanged);
        }
        for _ in 0..self.four {
            Four::send(device, &mut exchanged);
        }
        for _ in 0..self.all {
            All::send(device, &mut exchanged);
        }
        exchanged
    }

    /// `hidden` in the norm's layout, only the groups `self` holds: the norm and the feed-forward
    /// keep every row apart, so rows outside them stay stale.
    pub(crate) fn copy(self, device: &mut Device, hidden: &Residual) -> norm::eight::Values {
        let mut rows = norm::eight::Values::new();
        for _ in 0..self.one {
            One::copy(device, hidden, &mut rows);
        }
        for _ in 0..self.two {
            Two::copy(device, hidden, &mut rows);
        }
        for _ in 0..self.four {
            Four::copy(device, hidden, &mut rows);
        }
        for _ in 0..self.all {
            All::copy(device, hidden, &mut rows);
        }
        rows
    }

    /// `activation` into wave `at` of `activations`, only the groups `self` holds: the
    /// feed-forward keeps every row apart, so rows outside them stay stale.
    pub(crate) fn store_activation(
        self,
        device: &mut Device,
        activation: &norm::eight::Normalized<f8e4m3>,
        (activations, at): (&mut mlp::Input, usize),
    ) {
        for _ in 0..self.one {
            activation
                .view()
                .cluster_tile::<m![Rows / 32], 1, m![Rows / 32 = 1 # 2]>(0)
                .slice_tile::<m![Rows % 32 / 8], 1, m![Rows % 32 / 8 = 1 # 4 # 8, Hidden / 160]>(0)
                .to_hbm_view(
                    &mut device.tdma,
                    activation_wave(activations, at).tile::<m![Rows / 8], 1, m![
                        Waves = 1 #{!} Waves,
                        Hidden / 160,
                        Rows / 8 = 1 #{!} 8,
                        Rows % 8,
                        Hidden % 160
                    ]>(0),
                );
        }
        for _ in 0..self.two {
            activation
                .view()
                .cluster_tile::<m![Rows / 32], 1, m![Rows / 32 = 1 # 2]>(0)
                .slice_tile::<m![Rows % 32 / 8], 2, m![Rows % 32 / 8 = 2 # 4 # 8, Hidden / 160]>(0)
                .to_hbm_view(
                    &mut device.tdma,
                    activation_wave(activations, at).tile::<m![Rows / 16], 1, m![
                        Waves = 1 #{!} Waves,
                        Hidden / 160,
                        Rows / 16 = 1 #{!} 4,
                        Rows % 16,
                        Hidden % 160
                    ]>(0),
                );
        }
        for _ in 0..self.four {
            activation
                .view()
                .cluster_tile::<m![Rows / 32], 1, m![Rows / 32 = 1 # 2]>(0)
                .to_hbm_view(
                    &mut device.tdma,
                    activation_wave(activations, at).tile::<m![Rows / 32], 1, m![
                        Waves = 1 #{!} Waves,
                        Hidden / 160,
                        Rows / 32 = 1 #{!} 2,
                        Rows % 32,
                        Hidden % 160
                    ]>(0),
                );
        }
        for _ in 0..self.all {
            activation
                .view()
                .to_hbm_view(&mut device.tdma, activation_wave(activations, at));
        }
    }

    /// `input`'s groups that `self` holds landed into `landed`, as `copy` leaves the others stale.
    pub(crate) fn land(self, device: &mut Device, input: qkv::Input<'_>, landed: &mut qkv::Landed) {
        for _ in 0..self.one {
            One::land(device, input.clone(), landed);
        }
        for _ in 0..self.two {
            Two::land(device, input.clone(), landed);
        }
        for _ in 0..self.four {
            Four::land(device, input.clone(), landed);
        }
        for _ in 0..self.all {
            All::land(device, input.clone(), landed);
        }
    }
}

/// A count of a wave's leading 8-row groups that a decode wave's per-row transfers move: its TP2
/// exchanges, its norm's copy, its hidden rows' landings and stores. Rows outside them stay stale.
pub(crate) trait Groups {
    fn send(
        device: &mut Device,
        exchanged: &mut residual::Exchanged<Cluster, Slices, Time, Packet>,
    );

    fn copy(device: &mut Device, hidden: &Residual, rows: &mut norm::eight::Values);

    fn land(device: &mut Device, input: qkv::Input<'_>, landed: &mut qkv::Landed);

    /// `hidden` into wave `at` of `rows`.
    fn store_hidden(device: &mut Device, hidden: &Residual, rows: (&mut HiddenRows, usize));

    /// Wave `at` of `rows` into `hidden`.
    fn land_hidden(device: &mut Device, rows: (&HiddenRows, usize), hidden: &mut Residual);
}

pub(crate) struct One;
pub(crate) struct Two;
pub(crate) struct Four;
pub(crate) struct All;

impl Groups for One {
    fn send(
        device: &mut Device,
        exchanged: &mut residual::Exchanged<Cluster, Slices, Time, Packet>,
    ) {
        exchanged.send_tile::<
            m![Rows / 8],
            m![Hidden % 2560 / 80, Rows / 8 = 1 # 8],
            m![Hidden % 2560 / 80, Rows / 8 = 1 #{!} 8],
        >(device);
    }

    fn copy(device: &mut Device, hidden: &Residual, rows: &mut norm::eight::Values) {
        hidden
            .view()
            .slice_tile::<m![Rows / 8], 1, m![Hidden % 2560 / 80, Rows / 8 = 1 # 8]>(0)
            .to_dm_view(
                &mut device.tdma,
                rows.view_mut()
                    .cluster_tile::<m![Rows / 32], 1, m![Rows / 32 = 1 #{!} 2]>(0)
                    .slice_tile::<m![Rows % 32 / 8], 1, m![Rows % 32 / 8 = 1 #{!} 4 # 8, Hidden / 160]>(0),
            );
    }

    fn land(device: &mut Device, input: qkv::Input<'_>, landed: &mut qkv::Landed) {
        input
            .tile::<m![Rows / 8], 1, m![
                Waves = 1 # Waves,
                Hidden / 80,
                Rows / 8 = 1 # 8,
                Rows % 8,
                Hidden % 80
            ]>(0)
            .to_dm_view(
                &mut device.tdma,
                landed
                    .view_mut()
                    .tile::<m![Rows / 8], 1, m![Rows / 8 = 1 #{!} 8, Rows % 8, Hidden % 80]>(0),
            );
    }

    fn store_hidden(device: &mut Device, hidden: &Residual, (rows, at): (&mut HiddenRows, usize)) {
        hidden
            .view()
            .slice_tile::<m![Rows / 8], 1, m![Hidden % 2560 / 80, Rows / 8 = 1 # 8]>(0)
            .to_hbm_view(
                &mut device.tdma,
                wave_mut(rows, at).tile::<m![Rows / 8], 1, m![
                    Waves = 1 #{!} Waves,
                    Hidden / 80,
                    Rows / 8 = 1 #{!} 8,
                    Rows % 8,
                    Hidden % 80
                ]>(0),
            );
    }

    fn land_hidden(device: &mut Device, (rows, at): (&HiddenRows, usize), hidden: &mut Residual) {
        wave(rows, at)
            .tile::<m![Rows / 8], 1, m![
                Waves = 1 # Waves,
                Hidden / 80,
                Rows / 8 = 1 # 8,
                Rows % 8,
                Hidden % 80
            ]>(0)
            .to_dm_view(
                &mut device.tdma,
                hidden
                    .view_mut()
                    .slice_tile::<m![Rows / 8], 1, m![Hidden % 2560 / 80, Rows / 8 = 1 #{!} 8]>(0),
            );
    }
}

impl Groups for Two {
    fn send(
        device: &mut Device,
        exchanged: &mut residual::Exchanged<Cluster, Slices, Time, Packet>,
    ) {
        exchanged.send_tile::<
            m![Rows / 16],
            m![Hidden % 2560 / 80, Rows / 16 = 1 # 4, Rows % 16 / 8],
            m![Hidden % 2560 / 80, Rows / 16 = 1 #{!} 4, Rows % 16 / 8],
        >(device);
    }

    fn copy(device: &mut Device, hidden: &Residual, rows: &mut norm::eight::Values) {
        hidden
            .view()
            .slice_tile::<m![Rows / 16], 1, m![
                Hidden % 2560 / 80,
                Rows / 16 = 1 # 4,
                Rows % 16 / 8
            ]>(0)
            .to_dm_view(
                &mut device.tdma,
                rows.view_mut()
                    .cluster_tile::<m![Rows / 32], 1, m![Rows / 32 = 1 #{!} 2]>(0)
                    .slice_tile::<m![Rows % 32 / 8], 2, m![Rows % 32 / 8 = 2 #{!} 4 # 8, Hidden / 160]>(0),
            );
    }

    fn land(device: &mut Device, input: qkv::Input<'_>, landed: &mut qkv::Landed) {
        input
            .tile::<m![Rows / 16], 1, m![
                Waves = 1 # Waves,
                Hidden / 80,
                Rows / 16 = 1 # 4,
                Rows % 16,
                Hidden % 80
            ]>(0)
            .to_dm_view(
                &mut device.tdma,
                landed
                    .view_mut()
                    .tile::<m![Rows / 16], 1, m![Rows / 16 = 1 #{!} 4, Rows % 16, Hidden % 80]>(0),
            );
    }

    fn store_hidden(device: &mut Device, hidden: &Residual, (rows, at): (&mut HiddenRows, usize)) {
        hidden
            .view()
            .slice_tile::<m![Rows / 16], 1, m![
                Hidden % 2560 / 80,
                Rows / 16 = 1 # 4,
                Rows % 16 / 8
            ]>(0)
            .to_hbm_view(
                &mut device.tdma,
                wave_mut(rows, at).tile::<m![Rows / 16], 1, m![
                    Waves = 1 #{!} Waves,
                    Hidden / 80,
                    Rows / 16 = 1 #{!} 4,
                    Rows % 16,
                    Hidden % 80
                ]>(0),
            );
    }

    fn land_hidden(device: &mut Device, (rows, at): (&HiddenRows, usize), hidden: &mut Residual) {
        wave(rows, at)
            .tile::<m![Rows / 16], 1, m![
                Waves = 1 # Waves,
                Hidden / 80,
                Rows / 16 = 1 # 4,
                Rows % 16,
                Hidden % 80
            ]>(0)
            .to_dm_view(
                &mut device.tdma,
                hidden.view_mut().slice_tile::<m![Rows / 16], 1, m![
                    Hidden % 2560 / 80,
                    Rows / 16 = 1 #{!} 4,
                    Rows % 16 / 8
                ]>(0),
            );
    }
}

impl Groups for Four {
    fn send(
        device: &mut Device,
        exchanged: &mut residual::Exchanged<Cluster, Slices, Time, Packet>,
    ) {
        exchanged.send_tile::<
            m![Rows / 32],
            m![Hidden % 2560 / 80, Rows / 32 = 1 # 2, Rows % 32 / 8],
            m![Hidden % 2560 / 80, Rows / 32 = 1 #{!} 2, Rows % 32 / 8],
        >(device);
    }

    fn copy(device: &mut Device, hidden: &Residual, rows: &mut norm::eight::Values) {
        hidden
            .view()
            .slice_tile::<m![Rows / 32], 1, m![
                Hidden % 2560 / 80,
                Rows / 32 = 1 # 2,
                Rows % 32 / 8
            ]>(0)
            .to_dm_view(
                &mut device.tdma,
                rows.view_mut()
                    .cluster_tile::<m![Rows / 32], 1, m![Rows / 32 = 1 #{!} 2]>(0),
            );
    }

    fn land(device: &mut Device, input: qkv::Input<'_>, landed: &mut qkv::Landed) {
        input
            .tile::<m![Rows / 32], 1, m![
                Waves = 1 # Waves,
                Hidden / 80,
                Rows / 32 = 1 # 2,
                Rows % 32,
                Hidden % 80
            ]>(0)
            .to_dm_view(
                &mut device.tdma,
                landed
                    .view_mut()
                    .tile::<m![Rows / 32], 1, m![Rows / 32 = 1 #{!} 2, Rows % 32, Hidden % 80]>(0),
            );
    }

    fn store_hidden(device: &mut Device, hidden: &Residual, (rows, at): (&mut HiddenRows, usize)) {
        hidden
            .view()
            .slice_tile::<m![Rows / 32], 1, m![
                Hidden % 2560 / 80,
                Rows / 32 = 1 # 2,
                Rows % 32 / 8
            ]>(0)
            .to_hbm_view(
                &mut device.tdma,
                wave_mut(rows, at).tile::<m![Rows / 32], 1, m![
                    Waves = 1 #{!} Waves,
                    Hidden / 80,
                    Rows / 32 = 1 #{!} 2,
                    Rows % 32,
                    Hidden % 80
                ]>(0),
            );
    }

    fn land_hidden(device: &mut Device, (rows, at): (&HiddenRows, usize), hidden: &mut Residual) {
        wave(rows, at)
            .tile::<m![Rows / 32], 1, m![
                Waves = 1 # Waves,
                Hidden / 80,
                Rows / 32 = 1 # 2,
                Rows % 32,
                Hidden % 80
            ]>(0)
            .to_dm_view(
                &mut device.tdma,
                hidden.view_mut().slice_tile::<m![Rows / 32], 1, m![
                    Hidden % 2560 / 80,
                    Rows / 32 = 1 #{!} 2,
                    Rows % 32 / 8
                ]>(0),
            );
    }
}

impl Groups for All {
    fn send(
        device: &mut Device,
        exchanged: &mut residual::Exchanged<Cluster, Slices, Time, Packet>,
    ) {
        exchanged.send(device);
    }

    fn copy(device: &mut Device, hidden: &Residual, rows: &mut norm::eight::Values) {
        hidden.view().to_dm_view(&mut device.tdma, rows.view_mut());
    }

    fn land(device: &mut Device, input: qkv::Input<'_>, landed: &mut qkv::Landed) {
        input.to_dm_view(&mut device.tdma, landed.view_mut());
    }

    fn store_hidden(device: &mut Device, hidden: &Residual, (rows, at): (&mut HiddenRows, usize)) {
        hidden
            .view()
            .to_hbm_view(&mut device.tdma, wave_mut(rows, at));
    }

    fn land_hidden(device: &mut Device, (rows, at): (&HiddenRows, usize), hidden: &mut Residual) {
        wave(rows, at).to_dm_view(&mut device.tdma, hidden.view_mut());
    }
}

#[derive(DeviceSend)]
pub struct Recipe<'a> {
    pub embedding: &'a crate::models::nemo::parameters::Embedding,
    pub layers: stack::Layers<'a>,
    pub head: head::Projection<'a>,
    pub greedy: &'a head::greedy::Greedy,
    pub sampler: &'a head::sample::Table,
}

#[derive(DeviceSend)]
pub struct Input {
    pub ids: Offsets,
    pub live: mlp::Count,
    pub layers: mlp::Count,
    pub pages: mlp::Count,
    pub rotation: qkv::Coefficients,
    /// Each row's cache record: its token's 1-KiB record, which holds every head's key and value.
    pub keys: Offsets,
    /// Unread: the record at `keys` holds the values too.
    pub values: Offsets,
    pub attention: attention::Plan,
    pub selected: HbmTensor<i32, Chips, m![Selected]>,
    pub count: mlp::Count,
    pub prompt: attention::prompt::Plan,
    pub short: attention::short::Plan,
    pub sampling: sample::Plan,
    /// This chip's index (0 or 1), as each chip's one-element extent.
    pub chip: mlp::Count,
    /// The chunk the D-sharded route runs, if any.
    pub routed: Routed,
}

/// A chunk the D-sharded route runs, which `prompt` and `live` then leave out: the TP2 body sees
/// only the rows before it. Both counts are 0 when the TP2 body runs the chunk.
#[derive(DeviceSend)]
pub struct Routed {
    /// The chunk's 256-row blocks.
    pub blocks: mlp::Count,
    /// The chunk's segments.
    pub segments: mlp::Count,
}

#[derive(DeviceSend)]
pub struct State {
    pub hidden: HiddenRows,
    pub queries: QueryRows,
    pub keys: KvRows,
    pub values: KvRows,
    pub pool: attention::Pool,
    pub partials: attention::Partials,
    pub logits: HbmTensor<bf16, Chips, m![Selected, ShardVocabulary]>,
    pub tokens: HbmTensor<i32, Chips, m![Selected, 1 # 4]>,
    pub prompt: attention::prompt::Attended,
}

pub(crate) fn wave(
    state: &HiddenRows,
    wave: usize,
) -> HbmTensorView<'_, bf16, Chips, m![Waves = 1 # Waves, { Blocked }]> {
    state
        .view()
        .tile::<m![Waves], 1, m![Waves = 1 # Waves, { Blocked }]>(wave)
}

pub(crate) fn wave_mut(
    state: &mut HiddenRows,
    wave: usize,
) -> HbmTensorViewMut<'_, bf16, Chips, m![Waves = 1 #{!} Waves, { Blocked }]> {
    state
        .view_mut()
        .tile::<m![Waves], 1, m![Waves = 1 #{!} Waves, { Blocked }]>(wave)
}

/// Wave `wave` of the feed-forward's activations.
fn activation_wave(
    activations: &mut mlp::Input,
    wave: usize,
) -> HbmTensorViewMut<'_, f8e4m3, Chips, m![Waves = 1 #{!} Waves, { mlp::Grouped }]> {
    activations
        .view_mut()
        .tile::<m![Waves], 1, m![Waves = 1 #{!} Waves, { mlp::Grouped }]>(wave)
}

/// A layer's pool records at their entry and the step's queries, which a wave's Q/K/V writes.
type Written<'a> = ((&'a mut attention::Pool, usize), &'a mut QueryRows);

/// Wave `at`'s Q/K/V from its `landed` rows and its own rotation coefficients, read from `input`
/// alone, so the step's wave-0 coefficients stay alive only to the layer's start.
fn later_wave(
    device: &mut Device,
    (input, weights): (&Input, (&qkv::Weights, &scale::Scales<Chips>)),
    (landed, at): (&qkv::Landed, usize),
    ((pool, entry), queries): Written<'_>,
) {
    let rotations = qkv::Rotations::land(device, &input.rotation, at);
    let entries = input::entries(device, input, at);
    let begun = qkv::begin(device, weights, landed, &rotations);
    let quantized = input::quantize(device, (&begun.key, &begun.value));
    let (panels, _, _) = begun.split();
    input::write_wave(device, (pool, entry), quantized, &entries);
    panels.store(device, (&rotations, at), queries);
}

/// The next layer's K/V weight of a one-wave decode step, and its Q/K/V of a two-wave step's first
/// wave (`issued` 1), issued by the feed-forward sweep on its finished rows; `count` is 0 on the last
/// layer, and a step of three waves or more projects at the next layer's start ([`qkv::tile`]). The
/// next layer's scales land by one broadcast transfer into `scales`: a spread would stage one more
/// switch table for the step, moving every buffer of the layer.
struct Ahead<'a> {
    input: &'a Input,
    weights: &'a mut qkv::Weights,
    layer: (&'a stack::Layers<'a>, usize),
    scales: scale::Scales<Chips>,
    written: Written<'a>,
    first: qkv::Landed,
    count: usize,
    issued: usize,
    /// A tiled step's next K/V weight scale and its count (1 on such a step): the transfers land
    /// in the sweep's idle queue, ahead of the tile pass that reads them.
    pair_scale: (&'a mut qkv::tile::PairScale, usize),
}

impl mlp::Ahead for Ahead<'_> {
    fn prefetch(&mut self, device: &mut Device) {
        let (layers, index) = self.layer;
        for _ in 0..self.count {
            self.weights.land_last(device, self.layer);
            self.scales.copy_into(device, layers.qkv_scales.at(index));
        }
        let (scale, tiled) = &mut self.pair_scale;
        for _ in 0..(self.count * *tiled).min(1) {
            qkv::tile::land_pairs_into(device, self.weights, self.layer, scale);
        }
    }

    fn land_pairs(&mut self, device: &mut Device) {
        for _ in 0..self.count {
            self.weights.land_pairs(device, self.layer);
        }
    }

    fn land(&mut self, device: &mut Device, hidden: &HiddenRows) {
        for _ in 0..(self.count * self.issued).min(1) {
            All::land(device, wave(hidden, 0), &mut self.first);
        }
    }

    fn issue(&mut self, device: &mut Device, (hidden, waves): (&HiddenRows, usize)) {
        let ((pool, entry), queries) = &mut self.written;
        for _ in 0..(self.count * self.issued).min(1) {
            later_wave(
                device,
                (self.input, (self.weights, &self.scales)),
                (&self.first, 0),
                ((&mut **pool, *entry), &mut **queries),
            );
            for at in 0..(waves - 2).min(Waves::SIZE) {
                let rows = qkv::land(device, wave(hidden, at + 1));
                later_wave(
                    device,
                    (self.input, (self.weights, &self.scales)),
                    (&rows, at + 1),
                    ((&mut **pool, *entry), &mut **queries),
                );
            }
        }
    }
}

/// The decode batches of a layer: its pool records at `pool`, the attention plan and positions,
/// and the step's live rows and decode batches.
#[cfg(not(feature = "trace"))]
struct Batches<'a> {
    pool: (&'a attention::Pool, usize),
    plan: &'a attention::Plan,
    positions: &'a attention::Positions,
    few: &'a attention::Few<'a>,
    rows: (usize, usize),
}

#[cfg(not(feature = "trace"))]
impl Batches<'_> {
    /// Batch `batch`'s history, `next` 1 if a batch attends after it.
    fn history(&self, (batch, next): (usize, usize)) -> attention::History<'_> {
        attention::History {
            pool: self.pool.0,
            entry: self.pool.1,
            plan: self.plan,
            batch,
            rows: (self.rows.0 - 64 * batch).min(Rows::SIZE),
            positions: self.positions,
            few: self.few,
            next,
        }
    }
}

/// The next layer's Q/K/V weights but its last Q panel, which the last decode batch lands under
/// its O projection (`count` 0 on the last layer).
#[cfg(not(feature = "trace"))]
struct Prefetched<'a> {
    weights: &'a mut qkv::Weights,
    layer: (&'a stack::Layers<'a>, usize),
    count: usize,
    /// 1 if the next layer's K/V weight lands here too, else 0: a one-wave step's lands in the
    /// sweep, a tiled step's in its tile pass.
    pairs: usize,
}

/// Batch `history.batch`'s attention of plan word `word`, merged and landed for its O projection.
#[cfg(not(feature = "trace"))]
fn attended<L: attention::Lead>(
    device: &mut Device,
    (history, word): (&attention::History<'_>, usize),
    (queries, partials): (&QueryRows, &mut attention::Partials),
    prefetch: attention::Prefetch<'_, L>,
    buffers: &mut attention::Buffers,
) -> Projected<f32> {
    let merging = attention::attend_word(
        device,
        (history, word),
        (queries, partials),
        prefetch,
        buffers,
    );
    let attended = merging.finish(device);
    // SAFETY: the completed row-major owner flattens Heads/Groups/Head in wire order.
    let attended = unsafe { attended.reshape::<Chips, m![Rows, Width]>() };
    attended.to_dm(&mut device.tdma)
}

/// A layer's decode batches through their tails, into `residuals` and `activations`. Every batch
/// but the last sends its O partial ahead of the next batch's attention and completes after it,
/// so that attention runs while the partial is in flight; the last sends only its live groups.
#[cfg(not(feature = "trace"))]
fn decode_batches(
    device: &mut Device,
    batches: &Batches<'_>,
    (queries, partials, stored): (&QueryRows, &mut attention::Partials, &HiddenRows),
    (tail, model, first, late): (
        &mut Tail<'_>,
        &mlp::Model<'_>,
        &mut mlp::weight::First,
        usize,
    ),
    next: Prefetched<'_>,
    (residuals, activations): (&mut HiddenRows, &mut mlp::Input),
) {
    let (rows, decode) = batches.rows;
    let mut buffers = attention::Buffers::new();
    // The tail's weights land in the first batch's last wave, behind the gate half (a one-row
    // batch's behind its merge copies); their scales spread after its merge.
    let mut seeds = tail::Seeds::new();
    let mut landed = Projected::<f32>::new();
    // A loop end built from a runtime scalar needs a finite envelope, and a subtraction inside one
    // stays unsaturated only where a loop index bounds it.
    for at in 0..decode.min(1) {
        landed = attended(
            device,
            (
                &batches.history((at, (decode - at - 1).min(1))),
                attention::batch_word(batches.plan, at),
            ),
            (queries, &mut *partials),
            attention::Prefetch {
                gate: (model.gate(), &mut *first, 1 - (decode - at - 1).min(1)),
                lead: (
                    &mut tail::Landing {
                        tail: &mut *tail,
                        seeds: &mut seeds,
                    },
                    1 - at.min(1),
                ),
            },
            &mut buffers,
        );
    }
    // Batch `at + 1` attends while batch `at`'s partial is in flight.
    for at in 0..decode - decode.min(1) {
        // Read behind the send, the next batch's plan word would wait for the transfer's data
        // sync: a read of a DMA issued after the transfer publishes it.
        let word = attention::batch_word(batches.plan, at + 1);
        // 1 if batch `at + 1` has a next batch.
        let after = ((decode - at - 1) / 2).min(1);
        let history = batches.history((at + 1, after));
        // The next batch's second wave lands under this batch's O projection and send, where the
        // transfer queue would idle.
        attention::gather_ahead(device, (&history, word), &mut buffers);
        let hidden: Residual = wave(stored, at).to_dm(&mut device.tdma);
        for _ in 0..1 - at.min(1) {
            tail.spread(device, &seeds);
        }
        let factors = tail.factors(device);
        let pending = tail.issue_whole(device, &factors, (&landed, hidden, at));
        landed = attended(
            device,
            (&history, word),
            (queries, &mut *partials),
            attention::Prefetch {
                gate: (model.gate(), &mut *first, 1 - after),
                lead: (&mut attention::Later, 0),
            },
            &mut buffers,
        );
        tail.complete_live(
            device,
            &factors,
            (pending, |_: &mut Device| {}),
            (&mut *residuals, &mut *activations),
        );
    }
    for at in 0..decode.min(1) {
        let last = decode - at - 1;
        let hidden: Residual = wave(stored, last).to_dm(&mut device.tdma);
        // The next layer's Q/K/V weights but its last Q panel land under the O projection: in the
        // attention they queued the partials' stores, under the merge its result's.
        for _ in 0..next.count {
            next.weights.prefetch(device, next.layer, next.pairs);
        }
        for _ in 0..1 - last.min(1) {
            tail.spread(device, &seeds);
        }
        let factors = tail.factors(device);
        let pending = tail.issue_kept(
            device,
            &factors,
            (&landed, hidden, last, Live::of(rows - 64 * last)),
        );
        // A one-batch step's second gate half lands behind the norm's copy, under the norm.
        let gate = model.gate();
        let high = &mut first.high;
        tail.complete_live(
            device,
            &factors,
            (pending, move |device: &mut Device| {
                for _ in 0..late {
                    gate.land_into(device, (1, &mut *high));
                }
            }),
            (&mut *residuals, &mut *activations),
        );
    }
}

impl Recipe<'_> {
    pub fn apply(
        &self,
        device: &mut Device,
        (norm, input, state): (&Norm, &Input, &mut State),
        #[cfg(feature = "trace")] trace: &mut Trace,
    ) {
        let rows = input.live.extent::<m![Capacity]>();
        // A routed chunk's rows are embedded with the rows before it.
        let routed = input.routed.blocks.extent::<m![Prompt / 256]>();
        let pairs = (rows + 256 * routed + 127) / 128;
        let waves = (rows + 63) / 64;
        // Rows before the prompt chunk, a whole number of waves, attend through the job plan.
        let start = input.prompt.start.extent::<m![Capacity]>();
        let decode = start / 64;
        let blocks = input.prompt.blocks.extent::<m![Prompt / 256]>();
        let segments = input.prompt.segments.extent::<m![Segments]>();
        // The first segment's spans in each geometry, for its first span's landing.
        let first_spans = FirstSpans::of(&input.prompt);
        // Scalar reads go ahead of the embedding: issued after it, they wait for its transfers.
        let pages = input.pages.extent::<m![Pages]>();
        let layers = input.layers.extent::<m![Depth]>();
        let few = attention::few(&input.attention);
        // A routed chunk runs its layers D-sharded (`sharded`) after the decode rows' TP2 layers.
        #[cfg(not(feature = "trace"))]
        let sharded = routed.min(1);
        #[cfg(feature = "trace")]
        let sharded = 0;
        // A decode-only step of one wave, whose feed-forward is one wave's down sweep.
        let alone = (1 - blocks.min(1)) * (1 - (waves - 1).min(1));
        self.embedding.forward((
            &mut *device,
            Embedded {
                ids: &input.ids,
                pairs,
            },
            &mut state.hidden,
        ));
        let mut weights = qkv::Weights::new();
        let positions = attention::positions(device, &input.attention);
        let first_wave = input::Wave::copy(device, input);
        let mut rotations = qkv::Rotations::new();
        for _ in 0..decode.min(1) {
            rotations.land_into(device, &input.rotation, 0);
        }
        // A tiled decode-only step's K/V weight scale, which each layer's sweep lands for the next.
        let mut pair_scale = qkv::tile::PairScale::new();
        for index in 0..layers * (1 - sharded * (1 - decode.min(1))) {
            let layer = (&self.layers, index);
            // A decode-only step of three waves or more, and a chunk's two decode waves or more,
            // project every decode wave in the chunk's 256-row tiles, each weight byte streaming once
            // against four waves. A decode-only two-wave step ran wave 0's Q/K/V under the previous
            // layer's last down exchange. Either finds this layer's weights landed by the previous
            // layer, with its last Q panel. The tile pass must end inside the capacity.
            let only = 1 - blocks.min(1);
            let fits = 1 - (4 * (blocks + (decode + 3) / 4) - Waves::SIZE).min(1);
            let tiled = (only * (decode - 2).min(1) + (1 - only) * (decode - 1).min(1)) * fits;
            let perwave = 1 - tiled;
            let carried = (index * only * (decode - 1).min(1) * perwave).min(1);
            let prefetched = (index * only * (decode - 1)).min(1);
            let fresh = 1 - carried;
            // A decode step's wave 0 lands ahead of the weights, so its norm runs under them.
            let mut first = qkv::Landed::new();
            for _ in 0..(decode * fresh * perwave).min(1) {
                Live::of(rows).land(device, wave(&state.hidden, 0), &mut first);
            }
            // A decode step's later layers find their scale row and weights landed by the
            // previous layer's last decode wave, so their first Main work spreads the scales. A
            // prompt-only step's projection weights land inside the chunk, after its norm.
            for _ in 0..1 - decode.min(1) {
                weights.land_norm(device, layer);
            }
            for _ in 0..decode.min(1) * (1 - index.min(1)) {
                weights.land_weights(device, layer);
            }
            weights.spread_scales(device);
            for _ in 0..(decode * index * (1 - prefetched)).min(1) {
                weights.land_last(device, layer);
            }
            // Each later decode wave's rows land a wave ahead, once the previous wave's norm has
            // read the buffer, so the transfer runs under that wave's Q/K/V work.
            let later = (perwave * (fresh * (decode - decode.min(1)) + carried)).min(Waves::SIZE);
            let mut ahead = qkv::Landed::new();
            for _ in 0..later.min(1) {
                qkv::land_into(device, wave(&state.hidden, 1), &mut ahead);
            }
            // Wave 0's Q rotations and stores issue ahead of its K/V cache records: the attention
            // waits for both tables, and the Q chain is the longer one.
            for _ in 0..(decode * fresh * perwave).min(1) {
                let begun = qkv::begin(device, (&weights, weights.scales()), &first, &rotations);
                let quantized = input::quantize(device, (&begun.key, &begun.value));
                let (panels, _, _) = begun.split();
                panels.store(device, (&rotations, 0), &mut state.queries);
                first_wave.write(
                    device,
                    (&mut state.pool, index * pages),
                    quantized,
                    Live::of(rows),
                );
            }
            for at in 0..later {
                let wave_at = at + 1;
                let rotations = qkv::Rotations::land(device, &input.rotation, wave_at);
                let entries = input::entries(device, input, wave_at);
                let begun = qkv::begin(device, (&weights, weights.scales()), &ahead, &rotations);
                for _ in 0..(later - at - 1).min(1) {
                    qkv::land_into(device, wave(&state.hidden, wave_at + 1), &mut ahead);
                }
                let quantized = input::quantize(device, (&begun.key, &begun.value));
                let (panels, _, _) = begun.split();
                input::write_wave(
                    device,
                    (&mut state.pool, index * pages),
                    quantized,
                    &entries,
                );
                panels.store(device, (&rotations, wave_at), &mut state.queries);
            }
            // The chunk, or a tiled step's every wave, projects in 256-row blocks, one K/V and one
            // Q command a block.
            qkv::tile::chunk_carried(
                device,
                &mut weights,
                (layer, 1 - decode.min(1)),
                (
                    &state.hidden,
                    decode * perwave,
                    blocks + tiled * (decode + 3) / 4,
                ),
                (&mut pair_scale, 1 - (index * only * tiled).min(1)),
                &input.rotation,
                &mut state.queries,
                qkv::tile::Cache {
                    pool: &mut state.pool,
                    entry: index * pages,
                    offsets: &input.keys,
                },
            );
            #[cfg(feature = "trace")]
            trace.queries(device, (&state.queries, index));
            // The chunk's first span lands under the decode rows' attention and tails; a routed
            // chunk's lands in its own layers.
            let landing = attention::prompt::land(
                device,
                (&state.pool, index * pages),
                (&input.prompt, &first_spans),
                blocks.min(1),
            );
            let mut activations = mlp::Input::new();
            // The post-attention residual of every wave, apart from the state the tails read, so a
            // wave's store never orders the next wave's load.
            let mut residuals = HiddenRows::new();
            // One O weight serves the layer's every wave, landed once the first attention is done.
            let mut tail = Tail::new(layer, input.chip.extent::<m![2]>());
            let model = mlp::Model {
                layers: &self.layers,
                index,
            };
            // The first gate half lands under the last decode batch's attention, or after a
            // chunk's attention on a step without decode rows: earlier batches' last waves land
            // the next batch's first. A decode-only step of two batches or more lands the second
            // gate half behind it, in the last wave's idle transfer queue.
            let ahead = (1 - blocks.min(1)) * (decode - 1).min(1);
            let mut first = mlp::weight::First::new(Some(ahead));
            // A decode-only step of one batch lands the second gate half behind its norm copy.
            #[cfg(not(feature = "trace"))]
            let late = (1 - blocks.min(1)) * (1 - (decode - 1).min(1));
            #[cfg(feature = "trace")]
            let late = 0;
            #[cfg(not(feature = "trace"))]
            decode_batches(
                device,
                &Batches {
                    pool: (&state.pool, index * pages),
                    plan: &input.attention,
                    positions: &positions,
                    few: &few,
                    rows: (rows, decode),
                },
                (&state.queries, &mut state.partials, &state.hidden),
                (&mut tail, &model, &mut first, late),
                Prefetched {
                    weights: &mut weights,
                    layer: (&self.layers, index + 1),
                    count: (layers - index - 1).min(1),
                    pairs: 1 - alone - tiled,
                },
                (&mut residuals, &mut activations),
            );
            #[cfg(feature = "trace")]
            let mut buffers = attention::Buffers::new();
            #[cfg(feature = "trace")]
            for at in 0..decode {
                let live = Live::of(rows - 64 * at);
                // The tail's weights land in the first batch's last wave, behind the gate half (a
                // one-row batch's behind its merge copies); their scales spread after its merge.
                let mut seeds = tail::Seeds::new();
                let merging = attention::attend(
                    device,
                    &attention::History {
                        pool: &state.pool,
                        entry: index * pages,
                        plan: &input.attention,
                        batch: at,
                        rows: (rows - 64 * at).min(Rows::SIZE),
                        positions: &positions,
                        few: &few,
                        next: (decode - at - 1).min(1),
                    },
                    (&state.queries, &mut state.partials),
                    attention::Prefetch {
                        gate: (model.gate(), &mut first, 1 - (decode - at - 1).min(1)),
                        lead: (
                            &mut tail::Landing {
                                tail: &mut tail,
                                seeds: &mut seeds,
                            },
                            1 - at.min(1),
                        ),
                    },
                    &mut buffers,
                );
                let attended = merging.finish(device);
                // SAFETY: the completed row-major owner flattens Heads/Groups/Head in wire order.
                let attended = unsafe { attended.reshape::<Chips, m![Rows, Width]>() };
                #[cfg(feature = "trace")]
                let copy: Copied<f32> = attended.view().to_dm(&mut device.tdma);
                let rows: Projected<f32> = attended.to_dm(&mut device.tdma);
                let hidden: Residual = wave(&state.hidden, at).to_dm(&mut device.tdma);
                // The next layer's Q/K/V weights but its last Q panel land under the O projection:
                // in the attention they queued the partials' stores, under the merge its result's.
                for _ in 0..(1 - (decode - at - 1).min(1)) * (layers - index - 1).min(1) {
                    weights.prefetch(device, (&self.layers, index + 1), 1 - alone - tiled);
                }
                for _ in 0..1 - at.min(1) {
                    tail.spread(device, &seeds);
                }
                // The wave's scale registers load ahead of its O fill: behind it, they would hold
                // the in-order stream until the fill ends.
                let factors = tail.factors(device);
                let pending = tail.issue(
                    device,
                    &factors,
                    (rows, hidden, at, live),
                    #[cfg(feature = "trace")]
                    (trace, copy),
                );
                tail.complete_live(
                    device,
                    &factors,
                    (pending, |_: &mut Device| {}),
                    (&mut residuals, &mut activations),
                    #[cfg(feature = "trace")]
                    trace,
                );
            }
            let mut product = mlp::Product::new();
            let mut partials = mlp::Output::new();
            // A step without a chunk: every wave's tail ran with its attention, and its
            // feed-forward runs here, ahead of the prompt paths, so the gate half landed ahead
            // is not live across them. A one-wave step's next K/V weight lands behind the down
            // panels, off the O exchange's queue.
            for _ in 0..1 - blocks.min(1) {
                let pass = model.begin(
                    device,
                    (
                        &activations,
                        (&first.low, Some((&mut first.high, ahead + late))),
                        tail.scales(),
                    ),
                    &mut product,
                );
                pass.sweep(
                    device,
                    (&activations, &mut product, &mut partials),
                    (&residuals, &mut state.hidden, waves, rows),
                    (
                        &mut Ahead {
                            input,
                            weights: &mut weights,
                            layer: (&self.layers, index + 1),
                            scales: scale::Scales::new(),
                            written: ((&mut state.pool, (index + 1) * pages), &mut state.queries),
                            first: qkv::Landed::new(),
                            count: (layers - index - 1).min(1),
                            issued: perwave,
                            pair_scale: (&mut pair_scale, tiled),
                        },
                        alone,
                    ),
                );
            }
            attention::prompt::attend(
                device,
                (&state.pool, index * pages),
                (&input.prompt, (start, segments), &state.queries),
                landing,
                &mut state.prompt,
            );
            for _ in 0..blocks.min(1) * (1 - decode.min(1)) {
                tail.land_into(device);
                model.land_first(device, &mut first);
            }
            // The chunk's first wave, and on a step with no decode rows its second, project before
            // wave 0's feed-forward; every later prompt wave's exchange runs under the previous
            // wave's gate and up.
            // On a step with no decode rows the first two waves land their attended rows, then
            // both project and send before either completes: the second's exchange queues right
            // behind the first's instead of behind its norm and its rows' landing.
            #[cfg(not(feature = "trace"))]
            let pair = blocks.min(1) * (1 - decode.min(1)) * (waves - 1).min(1);
            #[cfg(not(feature = "trace"))]
            for _ in 0..pair {
                // Fresh owners: see `third`.
                let rows = (
                    tail.land_prompt(device, (0, 0), &state.prompt),
                    tail.land_prompt(device, (0, 1), &state.prompt),
                );
                // The registers load after the rows' transfers are issued: a load waiting for Sub
                // holds back every later command.
                let factors = tail.factors(device);
                let first = tail.issue_landed(device, &factors, (&rows.0, 0), state);
                let second = tail.issue_landed(device, &factors, (&rows.1, 1), state);
                tail.complete(device, &factors, first, (&mut residuals, &mut activations));
                tail.complete(device, &factors, second, (&mut residuals, &mut activations));
            }
            #[cfg(not(feature = "trace"))]
            let single = blocks.min(1) - pair;
            #[cfg(feature = "trace")]
            let single = blocks.min(1);
            for _ in 0..single {
                let factors = tail.factors(device);
                let pending = tail.issue_prompt(
                    device,
                    &factors,
                    (decode, decode),
                    state,
                    #[cfg(feature = "trace")]
                    trace,
                );
                tail.complete(
                    device,
                    &factors,
                    pending,
                    (&mut residuals, &mut activations),
                    #[cfg(feature = "trace")]
                    trace,
                );
            }
            #[cfg(feature = "trace")]
            for _ in 0..(1 - decode.min(1)).min(waves - 1) {
                let factors = tail.factors(device);
                let pending = tail.issue_prompt(
                    device,
                    &factors,
                    (decode, decode + 1),
                    state,
                    #[cfg(feature = "trace")]
                    trace,
                );
                tail.complete(
                    device,
                    &factors,
                    pending,
                    (&mut residuals, &mut activations),
                    #[cfg(feature = "trace")]
                    trace,
                );
            }
            let many = (decode - decode.min(1)).min(1);
            #[cfg(not(feature = "trace"))]
            let fused = blocks.min(1) * (1 - many) * (waves - 4).min(1);
            // After two decode waves or more, a chunk of four waves or more fuses from its first.
            #[cfg(not(feature = "trace"))]
            let past = blocks.min(1) * many * ((waves - decode) / 4).min(1);
            // A fused chunk's wave 2 lands its attended rows ahead of the feed-forward's weights
            // and down panels, which its projection would otherwise wait behind.
            #[cfg(not(feature = "trace"))]
            let mut third = Projected::<f32>::new();
            // A fresh owner: the allocator may place an in-place landing's output away from its
            // input, which the stream compiler refuses.
            #[cfg(not(feature = "trace"))]
            for _ in 0..fused {
                third = tail.land_prompt(device, (decode, 2), &state.prompt);
            }
            // A step with a chunk: its feed-forward, the decode waves' among it.
            for _ in 0..blocks.min(1) {
                let pass = model.begin(
                    device,
                    (&activations, (&first.low, None), tail.scales()),
                    &mut product,
                );
                // A chunk of five waves or more after one decode wave at most: fused, one TP2
                // transfer a wave for its O and down partials.
                #[cfg(not(feature = "trace"))]
                for _ in 0..fused {
                    let swept = pass.start(device, (&activations, &mut product), waves);
                    // One load of the scale registers serves the chunk's every wave, issued behind
                    // the transfers `start` queues, which a load waiting for Sub would hold back.
                    let factors = tail.factors(device);
                    let pending = tail.issue_landed(device, &factors, (&third, 2), state);
                    tail.complete(
                        device,
                        &factors,
                        pending,
                        (&mut residuals, &mut activations),
                    );
                    let projection = tail.project_prompt(device, &factors, (decode, 3), state);
                    exchange::chunk(
                        device,
                        (&tail, &factors, &pass, &state.prompt, projection),
                        (&mut residuals, &mut activations, &mut product),
                        (
                            exchange::Hidden {
                                input: None,
                                output: &mut state.hidden,
                            },
                            &mut partials,
                        ),
                        (swept, exchange::Second, decode, waves),
                    );
                }
                // Fused past the decode waves: their tails ran with their attention, and the
                // chunk's second wave lands its attended rows ahead of the down panels.
                #[cfg(not(feature = "trace"))]
                for _ in 0..past {
                    let third = tail.land_prompt(device, (decode, decode + 1), &state.prompt);
                    let swept = pass.start(device, (&activations, &mut product), waves);
                    let factors = tail.factors(device);
                    let pending = tail.issue_landed(device, &factors, (&third, decode + 1), state);
                    tail.complete(
                        device,
                        &factors,
                        pending,
                        (&mut residuals, &mut activations),
                    );
                    let projection =
                        tail.project_prompt(device, &factors, (decode, decode + 2), state);
                    exchange::chunk(
                        device,
                        (&tail, &factors, &pass, &state.prompt, projection),
                        (&mut residuals, &mut activations, &mut product),
                        (
                            exchange::Hidden {
                                input: None,
                                output: &mut state.hidden,
                            },
                            &mut partials,
                        ),
                        (swept, exchange::Past(decode), decode, waves),
                    );
                }
                #[cfg(not(feature = "trace"))]
                let serial = blocks.min(1) - fused - past;
                #[cfg(feature = "trace")]
                let serial = blocks.min(1);
                for _ in 0..serial {
                    for at in 0..decode - decode.min(1) {
                        pass.project(device, &activations, (&mut product, at + 1));
                    }
                    let projected = decode + 1 - decode.min(1);
                    let rest = waves - projected;
                    // The prompt waves past the first, as one block a step without them skips:
                    // the activation table publishes once after their loop, and only when that
                    // loop ran.
                    for _ in 0..rest.min(1) {
                        let factors = tail.factors(device);
                        for at in 0..rest - rest.min(1) {
                            let wave = projected + at;
                            // The next wave's O projection runs under the last trip's activation
                            // store; the activation lands before the exchange, which runs under
                            // its contractions.
                            let projection = tail.project_prompt(
                                device,
                                &factors,
                                (decode, wave + 1),
                                state,
                                #[cfg(feature = "trace")]
                                trace,
                            );
                            let activation =
                                mlp::producer::Activation::land(device, &activations, wave);
                            let pending = tail.send(device, projection, state);
                            // The next wave's residual sum is issued before this wave's product is
                            // quantized, so the norm copy runs under that work.
                            let contracted = pass.contract(device, &activation);
                            let summed = tail.sum(device, pending);
                            pass.store(device, contracted, (&mut product, wave));
                            tail.normalize(
                                device,
                                &factors,
                                summed,
                                (&mut residuals, &mut activations),
                                #[cfg(feature = "trace")]
                                trace,
                            );
                        }
                        pass.project(device, &activations, (&mut product, waves - 1));
                    }
                    pass.finish(
                        device,
                        (&product, &mut partials),
                        (&residuals, &mut state.hidden, waves),
                    );
                }
            }
            #[cfg(feature = "trace")]
            trace.mlp(device, (&activations, &product, &partials), index);
            #[cfg(feature = "trace")]
            trace.output(device, (&state.hidden, index));
        }
        // The chunk's count is read where it runs and its first spans are the step's: each scalar
        // read at the top holds a scratchpad line through every decode layer.
        #[cfg(not(feature = "trace"))]
        for _ in 0..sharded {
            let chunk = sharded::Shape {
                start,
                blocks: routed,
                segments: input.routed.segments.extent::<m![Segments]>(),
                first: FirstSpans { ..first_spans },
                pages,
            };
            crate::Module::forward(
                self,
                (
                    &mut *device,
                    sharded::Prompt {
                        input,
                        state: &mut *state,
                        shape: &chunk,
                        layers,
                    },
                ),
            );
        }
        self.finish(device, (norm, input, waves + 4 * routed), state);
    }
}

/// Attended rows rounded once to BF16, the O projection's input.
fn round<C: M>(device: &mut Device, rows: &Attended<C, f32>) -> Attended<C, bf16> {
    device
        .main
        .begin(rows.view())
        .fetch::<m![Rows % 2, Width % 256 / 8], m![Width % 8]>()
        .collect::<m![Rows % 2, Width % 256 / 8], m![Width % 8]>()
        .cast::<bf16, m![Width % 8 # 16]>()
        .commit_trim::<m![Width % 8]>()
        .commit()
}

/// `round` into `output`.
#[cfg(not(feature = "trace"))]
fn round_into<C: M>(
    device: &mut Device,
    rows: &Attended<C, f32>,
    output: DmTensorViewMut<'_, bf16, Chips, C, output::Landing, output::Element>,
) {
    device
        .main
        .begin(rows.view())
        .fetch::<m![Rows % 2, Width % 256 / 8], m![Width % 8]>()
        .collect::<m![Rows % 2, Width % 256 / 8], m![Width % 8]>()
        .cast::<bf16, m![Width % 8 # 16]>()
        .commit_trim::<m![Width % 8]>()
        .commit_view(output)
}

#[cfg(not(feature = "trace"))]
#[device(chip = 2)]
pub fn forward(
    device: &mut Device,
    model: &crate::models::nemo::parameters::Model,
    input: (&Input, &mut State),
) {
    crate::Module::forward(model, (device, input));
}

#[cfg(not(feature = "trace"))]
impl crate::Module<(&mut Device, (&Norm, &Input, &mut State))> for Recipe<'_> {
    type Output = ();

    fn forward(&self, (device, input): (&mut Device, (&Norm, &Input, &mut State))) {
        self.apply(device, input);
    }
}

/// `forward` that also copies the first two layers' wave-0 piece outputs into `trace`; it is only
/// valid for at most two runtime layers.
#[cfg(feature = "trace")]
#[device(chip = 2)]
pub fn traced(
    device: &mut Device,
    model: &Recipe<'_>,
    (norm, input, state, trace): (&Norm, &Input, &mut State, &mut Trace),
) {
    model.apply(device, (norm, input, state), trace);
}
