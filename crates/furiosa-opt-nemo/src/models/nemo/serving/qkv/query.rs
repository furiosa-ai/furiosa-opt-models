//! Q as two weight-stationary panels in the K/V contraction's K80 form: each panel holds the
//! chip's even or odd heads, and contracts the activation where it lands, with no gather.
use furiosa_opt_std::prelude::*;

use super::contraction::{self, Elements, FillFlit, FillPacket, Lanes, Out};
use super::rotation::{self, Rotated};
use super::{Head, Queries};
use crate::models::nemo::serving::mlp::Capacity;
use crate::models::nemo::serving::stack::{self, Depth};
use crate::models::nemo::serving::{Chips, Hidden, Rows};

axes![Panel = 2, Heads = 8];
axes![Wave = 64];
pub(super) type Space = crate::compose::Tile<Rows, Wave>;
/// Both panels of one layer's Q weight; panel `p` holds heads `2 g + p` in the K/V layout.
pub type Stored = m![Panel, { contraction::Stored }];
/// Both panels in data memory, a lane's rows of the two panels side by side so one fill takes both.
pub type Weight<B> = DmTensor<
    f8e4m3,
    B,
    contraction::Cluster,
    contraction::Slice,
    m![{ Lanes }, Panel, { Elements }],
>;
/// Both panels in the register file, for a chunk's blocks to contract in one command each.
pub type Register<B> =
    TrfTensor<f8e4m3, B, contraction::Cluster, contraction::Slice, Lanes, m![Panel, { Elements }]>;
/// A wave's contraction of both panels, a panel's outputs where [`contraction::Values`] holds them.
pub type Both<B> = DmTensor<
    bf16,
    B,
    contraction::Cluster,
    contraction::Reduced,
    m![{ m![Panel, Rows % 4, Out % 32 / 8] }, { Lanes }],
>;
type Feed = m![Panel, Out % 128 / 32, Rows / 4, Hidden % 80 / 16];
type Sums = m![Panel, Out % 128 / 32, Rows / 4, Out % 32 / 8];
type Cluster = m![Heads / 4];
/// A panel's rows, one row of one head a slice.
pub type Rotation<B> = Rotated<B, Cluster, m![Heads % 4, Rows], m![1]>;
/// A wave's rotation coefficients landed beside a panel's rows.
pub type Coefficients<B> = rotation::Landed<B, Cluster, m![Heads % 4, Rows], m![1]>;
/// [`Coefficients`] in the register file.
pub type Filled<B> = rotation::Register<B, Cluster, m![Heads % 4, Rows], m![1]>;

/// Lands both panels of one layer's Q weight into `weight`.
pub fn land_into(
    device: &mut Device,
    (layers, index): (&stack::Layers, usize),
    weight: &mut Weight<Chips>,
) {
    layers
        .query
        .at(index)
        .to_dm_view(&mut device.tdma, weight.view_mut());
}

/// Lands panel `panel` of one layer's Q weight into its half of `weight`.
pub fn land_panel(
    device: &mut Device,
    (layers, index): (&stack::Layers, usize),
    (panel, weight): (usize, &mut Weight<Chips>),
) {
    layers
        .query
        .at(index)
        .tile::<m![Panel], 1, m![Depth = 1 # Depth, Panel = 1 # 2, { contraction::Stored }]>(panel)
        .to_dm_view(
            &mut device.tdma,
            weight
                .view_mut()
                .tile::<m![Panel], 1, m![{ Lanes }, Panel = 1 #{!} 2, { Elements }]>(panel),
        );
}

/// Both panels' register.
pub fn fill<B: M>(device: &mut Device, weight: &Weight<B>) -> Register<B> {
    device
        .sub
        .begin(weight.view())
        .fetch::<m![{ Lanes }, Panel, Out % 128 / 32, Hidden % 80 / 16], FillPacket>()
        .collect::<m![
            { Lanes },
            Panel,
            Out % 128 / 32,
            Hidden % 80 / 16,
            Out % 32 / 16
        ], FillFlit>()
        .to_trf()
}

impl crate::ops::linear::contraction::Mapping for Space {
    type Cluster = contraction::Cluster;
    type Slices = contraction::Slice;
    type Reduced = contraction::Reduced;
    type Input = m![Rows, Hidden % 80];
    type Lanes = Lanes;
    type Weight = m![Panel, { Elements }];
    type Feed = Feed;
    type Packet = m![Rows % 4, Hidden % 16];
    type Flits = m![{ Feed }, Rows % 4 / 2];
    type Flit = m![Rows % 2, Hidden % 16];
    type Steps = m![{ Feed }, Out % 32 / 8];
    type Contracted = m![Rows % 4];
    type Sums = Sums;
    type Groups = m![{ Sums }, Rows % 4];
    type Output = m![Panel, Rows % 4, Out % 32 / 8];
}

/// Panel `panel` of `both` relaid so that each slice holds whole heads of one row, for the rotation.
pub fn rows<B: M>(device: &mut Device, both: &Both<B>, panel: usize) -> Rotation<B> {
    let values = both
        .view()
        .tile::<m![Panel], 1, m![Panel = 1 # 2, Rows % 4, Out % 32 / 8, Out % 8]>(panel);
    // SAFETY: a panel's 1,024 outputs are its eight heads of 128 features in order.
    let values = unsafe {
        values.reshape::<B, Cluster, m![Heads % 4, Head / 64, Head % 64 / 32, Rows / 4], m![
            Panel = 1 # 2,
            Rows % 4,
            Head % 32 / 8,
            Head % 8
        ]>()
    };
    let mut rows = Rotation::new();
    values.to_dm_view(&mut device.tdma, rows.view_mut());
    rows
}

/// The step's row state at wave `wave`'s rows and panel `panel`'s heads.
pub fn state(
    queries: &mut Queries,
    (wave, panel): (usize, usize),
) -> HbmTensorViewMut<'_, bf16, Chips, m![Rows #{!} Capacity, Heads, Panel = 1 #{!} 2, Head]> {
    // SAFETY: head `2 g + p` of a row is panel `p`'s head `g`.
    let heads = unsafe {
        queries
            .view_mut()
            .reshape::<Chips, m![Capacity, Heads, Panel, Head]>()
    };
    let tile = heads
        .tile::<m![Capacity], 64, m![Capacity = 64 #{!} Capacity, Heads, Panel, Head]>(wave * 64)
        .tile::<m![Panel], 1, m![Capacity = 64 #{!} Capacity, Heads, Panel = 1 #{!} 2, Head]>(
            panel,
        );
    // SAFETY: the 64-row tile keeps the capacity stride; only its row axis is renamed.
    unsafe { tile.reshape::<Chips, m![Rows #{!} Capacity, Heads, Panel = 1 #{!} 2, Head]>() }
}
