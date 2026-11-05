//! How a block-scaled panel projection spreads over the slices: the panels, owners and slabs of
//! a shape, and the mappings its tensors take.

use super::super::GROUP;
use crate::ops::normalization::Chunk;
use furiosa_opt_std::prelude::*;
use std::marker::PhantomData;

/// The slices of a cluster.
const SLICES: usize = 256;

/// The most blocks a panel holds: eight lanes of a panel's codes then fill one TRF bank.
const DEPTH: usize = 32;

/// The weight bytes a slice holds of one slab: a double-buffered slab then takes four pages.
const SLAB: usize = 64 * 1024;

/// How a projection shape spreads over a cluster's slices.
#[derive(Clone, Copy)]
struct Layout {
    /// 128-input blocks a panel.
    blocks: usize,
    /// The slices one owner's panels take: its panels padded to a power of two, as the lowering
    /// counts a padded slice digit.
    ring: usize,
    /// Output rows an owner holds.
    rows: usize,
    /// Output rows of an owner a weight slab carries.
    slab: usize,
    /// The data networks an owner's panels alternate over.
    networks: usize,
    /// The inputs a slice stages contiguously for every row: the row carrier's feature chunk when
    /// whole chunks tile a panel along its blocks, else the whole panel.
    granule: usize,
    /// The outputs an owner keeps contiguously for every row: the output carrier's feature chunk
    /// when whole chunks tile the owner's rows, else all of them.
    piece: usize,
}

/// The fewest blocks a panel such that an owner's panels take at most a PE's slices (else two
/// PEs, else the cluster), and the largest slab of whole four-row tiles within `SLAB` bytes a slice.
const fn layout<In: AxisName, Out: AxisName>() -> Layout {
    let (input, output) = (In::SIZE, Out::SIZE);
    assert!(
        input.is_multiple_of(GROUP),
        "inputs come in whole 128-input groups"
    );
    let groups = input / GROUP;
    let mut most = SLICES / 4;
    let blocks = loop {
        let mut blocks = 1;
        while blocks <= DEPTH
            && (groups % blocks != 0 || (groups / blocks).next_power_of_two() > most)
        {
            blocks += 1;
        }
        if blocks <= DEPTH {
            break blocks;
        }
        assert!(
            most < SLICES,
            "no panel of at most 32 blocks fits the slices"
        );
        most *= 2;
    };
    let (panels, panel) = (groups / blocks, GROUP * blocks);
    let ring = panels.next_power_of_two();
    let owners = SLICES / ring;
    // An owner's four-row tiles come in pairs: its FP32 scales then fill whole 8-byte words.
    assert!(
        output.is_multiple_of(16 * owners),
        "an owner holds whole pairs of four-row tiles"
    );
    let rows = output / (2 * owners);
    let mut tiles = SLAB / (4 * panel);
    while (rows / 4) % tiles != 0 {
        tiles -= 1;
    }
    let (inputs, outputs) = (Chunk::<In>::SIZE, Chunk::<Out>::SIZE);
    let blocked = inputs.is_multiple_of(GROUP) || GROUP.is_multiple_of(inputs);
    Layout {
        blocks,
        ring,
        rows,
        slab: 4 * tiles,
        networks: if panels.is_multiple_of(2) { 2 } else { 1 },
        granule: if blocked && panel.is_multiple_of(inputs) {
            inputs
        } else {
            panel
        },
        piece: if rows.is_multiple_of(outputs) {
            outputs
        } else {
            rows
        },
    }
}

/// The inputs of one panel.
#[derive(Clone, Debug)]
pub struct Panel<In, Out>(PhantomData<(In, Out)>);

impl<In: AxisName, Out: AxisName> AxisName for Panel<In, Out> {
    const NAME: Ident = Ident::new("Panel");
    const SIZE: usize = layout::<In, Out>().blocks * GROUP;
}

/// The output rows of one owner.
#[derive(Clone, Debug)]
pub(super) struct Owned<In, Out>(PhantomData<(In, Out)>);

impl<In: AxisName, Out: AxisName> AxisName for Owned<In, Out> {
    const NAME: Ident = Ident::new("Owned");
    const SIZE: usize = layout::<In, Out>().rows;
}

/// The data networks an owner's panels alternate over: two for an even count, so that each
/// side of a PE, one data network each, holds half the panels.
#[derive(Clone, Debug)]
pub(super) struct Networks<In, Out>(PhantomData<(In, Out)>);

impl<In: AxisName, Out: AxisName> AxisName for Networks<In, Out> {
    const NAME: Ident = Ident::new("Networks");
    const SIZE: usize = layout::<In, Out>().networks;
}

/// The slices of an owner's panels on one side of its PE, padding included.
#[derive(Clone, Debug)]
pub(super) struct Side<In, Out>(PhantomData<(In, Out)>);

impl<In: AxisName, Out: AxisName> AxisName for Side<In, Out> {
    const NAME: Ident = Ident::new("Side");
    const SIZE: usize = {
        let layout = layout::<In, Out>();
        layout.ring / layout.networks
    };
}

/// The output rows of one owner a weight slab carries.
#[derive(Clone, Debug)]
pub(super) struct Slab<In, Out>(PhantomData<(In, Out)>);

impl<In: AxisName, Out: AxisName> AxisName for Slab<In, Out> {
    const NAME: Ident = Ident::new("Slab");
    const SIZE: usize = layout::<In, Out>().slab;
}

/// The inputs a slice stages contiguously for every row.
#[derive(Clone, Debug)]
pub(super) struct Granule<In, Out>(PhantomData<(In, Out)>);

impl<In: AxisName, Out: AxisName> AxisName for Granule<In, Out> {
    const NAME: Ident = Ident::new("Granule");
    const SIZE: usize = layout::<In, Out>().granule;
}

/// The outputs an owner keeps contiguously for every row.
#[derive(Clone, Debug)]
pub(super) struct Piece<In, Out>(PhantomData<(In, Out)>);

impl<In: AxisName, Out: AxisName> AxisName for Piece<In, Out> {
    const NAME: Ident = Ident::new("Piece");
    const SIZE: usize = layout::<In, Out>().piece;
}

pub(super) type Cluster<O> = m![O / (O / 2)];
pub(super) type Owners<I, O> = m![O % (O / 2) / ({ Symbol<Owned<I, O>> })];
pub(super) type Parts<I, O> = m![I / ({ Symbol<Panel<I, O>> })];
pub(super) type Slices<I, O> = m![
    { Owners<I, O> },
    { Parts<I, O> } % ({ Symbol<Networks<I, O>> }),
    { Parts<I, O> } / ({ Symbol<Networks<I, O>> }) # ({ Symbol<Side<I, O>> })
];
pub(super) type Reduced<I, O> = m![
    { Owners<I, O> },
    1 # ({ Symbol<Networks<I, O>> }),
    1 # ({ Symbol<Side<I, O>> })
];

/// An owner's output rows, the rows of them a weight slab carries, and the slabs.
pub(super) type Outputs<I, O> = m![O % ({ Symbol<Owned<I, O>> })];
pub(super) type Carried<I, O> = m![O % ({ Symbol<Slab<I, O>> })];
pub(super) type Slabs<I, O> = m![O % ({ Symbol<Owned<I, O>> }) / ({ Symbol<Slab<I, O>> })];

/// A panel's inputs and its 128-input blocks.
pub(super) type Inputs<I, O> = m![I % ({ Symbol<Panel<I, O>> })];
pub(super) type Blocks<I, O> = m![I % ({ Symbol<Panel<I, O>> }) / 128];

/// A tensor on the projection's slices.
pub(super) type Sliced<D, I, O, E> = DmTensor<D, m![1], Cluster<O>, Slices<I, O>, E>;
