use furiosa_opt_std::prelude::*;

use crate::ops::linear::query::Tile;
use crate::ops::rotary::group::{InputHalf, OutputHalf, Rotation};
use crate::tensor::axis;
use crate::tensor::parallel::Chip;

#[derive(Debug, Clone)]
pub struct Tiles<const N: usize>;

impl<const N: usize> AxisName for Tiles<N> {
    const NAME: Ident = Ident::new("Tiles");
    const SIZE: usize = N;
}

pub trait Group: AxisName {
    /// The rows of a period of this group, its tiles in order, as one axis.
    type Period: AxisName;
    type Copies: Count;
    type Slot: M;
    type Rows<All: AxisName>: M;

    fn stacked<S: AxisName, Half: AxisName>(
        rotation: &Rotation<S, Half>,
        group: usize,
    ) -> HbmTensorView<'_, bf16, Chip, m![Self # S, Tile, OutputHalf, InputHalf, Half]>;

    fn window<All: AxisName, Inner: M>(
        rows: HbmTensorViewMut<'_, bf16, Chip, m![All, { Inner }]>,
        start: usize,
    ) -> HbmTensorViewMut<'_, bf16, Chip, m![{ Self::Rows<All> }, { Inner }]>;

    fn slot<'l, C: M, P: M, E: M>(
        rows: DmTensorViewMut<'l, bf16, Chip, C, P, m![Self, { E }]>,
        wave: usize,
    ) -> DmTensorViewMut<'l, bf16, Chip, C, P, m![{ Self::Slot }, { E }]>;
}

/// [`Group::Copies`] of a single wave, named for bounds that a projection cannot carry.
pub(crate) type SingleCopies = m![512 / 128];

impl Group for Tiles<1> {
    type Period = Tile;
    type Copies = SingleCopies;
    type Slot = m![Self];
    type Rows<All: AxisName> = m![Tile #{!} All];

    fn stacked<S: AxisName, Half: AxisName>(
        rotation: &Rotation<S, Half>,
        group: usize,
    ) -> HbmTensorView<'_, bf16, Chip, m![Self # S, Tile, OutputHalf, InputHalf, Half]> {
        let waves = rotation
            .view()
            .tile::<m![S], 1, m![S = 1 # S, Tile, OutputHalf, InputHalf, Half]>(group);
        // SAFETY: one consecutive stack slot is one wave of this group.
        unsafe { waves.reshape() }
    }

    fn window<All: AxisName, Inner: M>(
        rows: HbmTensorViewMut<'_, bf16, Chip, m![All, { Inner }]>,
        start: usize,
    ) -> HbmTensorViewMut<'_, bf16, Chip, m![{ Self::Rows<All> }, { Inner }]> {
        let rows = rows
            .tile::<m![All], { Tile::SIZE }, m![All = { Tile::SIZE } #{!} All, { Inner }]>(start);
        // SAFETY: the contiguous row window preserves the wave's row order.
        unsafe { rows.reshape() }
    }

    fn slot<'l, C: M, P: M, E: M>(
        rows: DmTensorViewMut<'l, bf16, Chip, C, P, m![Self, { E }]>,
        _: usize,
    ) -> DmTensorViewMut<'l, bf16, Chip, C, P, m![Self, { E }]> {
        rows
    }
}

#[derive(Debug, Clone)]
pub struct TilePair;

impl AxisName for TilePair {
    const NAME: Ident = Ident::new("Tiles");
    const SIZE: usize = 2;
}

/// [`Group::Copies`] of a [`TilePair`], named for bounds that a projection cannot carry.
pub(crate) type PairCopies = m![256 / 128];

impl Group for TilePair {
    type Period = axis::Rows<256>;
    type Copies = PairCopies;
    type Slot = m![Self = 1 #{!} 2];
    type Rows<All: AxisName> = m![Self #{!} (All / Tile), Tile];

    fn stacked<S: AxisName, Half: AxisName>(
        rotation: &Rotation<S, Half>,
        group: usize,
    ) -> HbmTensorView<'_, bf16, Chip, m![Self # S, Tile, OutputHalf, InputHalf, Half]> {
        let waves = rotation
            .view()
            .tile::<m![S], 2, m![S = 2 # S, Tile, OutputHalf, InputHalf, Half]>(2 * group);
        // SAFETY: the pair's waves are consecutive slots of the stack.
        unsafe { waves.reshape() }
    }

    fn window<All: AxisName, Inner: M>(
        rows: HbmTensorViewMut<'_, bf16, Chip, m![All, { Inner }]>,
        start: usize,
    ) -> HbmTensorViewMut<'_, bf16, Chip, m![{ Self::Rows<All> }, { Inner }]> {
        let rows = rows
            .tile::<m![All / Tile], 2, m![All / Tile = 2 #{!} (All / Tile), All % Tile, { Inner }]>(
                start / Tile::SIZE,
            );
        // SAFETY: the pair's consecutive waves keep their original row order.
        unsafe { rows.reshape() }
    }

    fn slot<'l, C: M, P: M, E: M>(
        rows: DmTensorViewMut<'l, bf16, Chip, C, P, m![Self, { E }]>,
        wave: usize,
    ) -> DmTensorViewMut<'l, bf16, Chip, C, P, m![Self = 1 #{!} 2, { E }]> {
        rows.tile::<m![Self], 1, m![Self = 1 #{!} 2, { E }]>(wave)
    }
}
