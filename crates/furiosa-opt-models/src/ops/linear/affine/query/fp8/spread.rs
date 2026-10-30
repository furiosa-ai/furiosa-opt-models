use super::{Input, Quantize, RowGroups, activation::Copies};
use crate::Op;
use crate::ops::linear::affine::query::Tile;
use crate::ops::linear::query::group::TilePair;
use crate::ops::vector::Quantization;
use crate::tensor::parallel::Chip;
use core::marker::PhantomData;
use furiosa_opt_std::prelude::*;

pub(crate) type Spread<Hidden, All> = DmTensor<
    bf16,
    Chip,
    m![All % 64 / 32],
    m![All % 32 / 8, Hidden / 64],
    m![All / 64, Hidden % 64 / 16, All % 8, Hidden % 16],
>;
type SpreadInput<Hidden, All> = DmTensor<
    f8e4m3,
    Chip,
    m![All % 64 / 32],
    m![All % 32 / 8, Hidden / 64],
    m![All / 64, All % 8, Hidden % 64],
>;
type SpreadScales<All> =
    DmTensor<bf16, Chip, m![All % 64 / 32], m![All % 32 / 8, Copies], m![All / 64, All % 8]>;
pub(super) type Staged<Hidden, All> = HbmTensor<f8e4m3, Chip, m![Hidden / 64, All, Hidden % 64]>;
pub(super) type StagedScales<All> = HbmTensor<bf16, Chip, m![All % 64 / 8, All / 64, All % 8]>;
/// The quantization of the spread rows.
type Rows<Hidden, All> = Quantization<
    m![All / 64, All % 8, Hidden % 64 / 8],
    m![All / 64, All % 8, Hidden % 64 / 4],
    Hidden,
    m![All / 64, All % 8],
    m![All % 32 / 8, Copies],
    m![All / 64, All % 8 / 4],
    m![All % 4],
    m![All / 64, All % 8, Hidden % 64 / 16, Hidden % 16 / 8],
>;

/// Spread rows as E4M3 rows and their BF16 row scales, the first `blocks` blocks of 64 rows.
impl<'a, Hidden: AxisName, All: AxisName> Op<(&'a Spread<Hidden, All>, usize)> for Quantize {
    type Output = (SpreadInput<Hidden, All>, SpreadScales<All>);

    fn forward(
        &self,
        context: &mut Device,
        (rows, blocks): (&'a Spread<Hidden, All>, usize),
    ) -> Self::Output {
        let mut scales = SpreadScales::<All>::new();
        Rows::<Hidden, All>::scales(&mut context.main, rows.view())
            .commit_prefix::<m![All / 64], _, _>(scales.view_mut(), blocks);
        // SAFETY: the reduction broadcasts each row's scale to every slice of its slice group.
        let broadcast = unsafe {
            scales.view().reshape::<Chip, m![All % 64 / 32], m![All % 32 / 8, Hidden / 64],
                m![All / 64, All % 8]>()
        };
        let scale: VrfTensor<
            f32,
            Chip,
            m![All % 64 / 32],
            m![All % 32 / 8, Hidden / 64],
            m![All / 64, All % 8],
        > = context
            .sub
            .begin(broadcast)
            .fetch::<m![All / 64], m![All % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![All / 64], m![All % 8]>()
            .to_vrf();
        let mut quantized = SpreadInput::<Hidden, All>::new();
        Rows::<Hidden, All>::rows(&mut context.main, rows.view(), &scale)
            .commit_prefix::<m![All / 64], _, _>(quantized.view_mut(), blocks);
        (quantized, scales)
    }
}

/// Quantized spread rows and their scales, stored in HBM in row order for the wave pairs.
pub(crate) struct Stage;

impl<'a, Hidden: AxisName, All: AxisName> Op<(&'a SpreadInput<Hidden, All>, &'a SpreadScales<All>)>
    for Stage
{
    type Output = (Staged<Hidden, All>, StagedScales<All>);

    fn forward(
        &self,
        context: &mut Device,
        (rows, scales): (&'a SpreadInput<Hidden, All>, &'a SpreadScales<All>),
    ) -> Self::Output {
        let mut staged = Staged::<Hidden, All>::new();
        rows.view()
            .to_hbm_view(&mut context.tdma, staged.view_mut());
        let mut stored = StagedScales::<All>::new();
        scales
            .view()
            .slice_tile::<m![Copies], 1, m![All % 32 / 8, Copies = 1 # 64]>(0)
            .to_hbm_view(&mut context.tdma, stored.view_mut());
        (staged, stored)
    }
}

/// Pair `pair`'s quantized rows, written from the staged rows.
pub(crate) struct PairRows;

impl<'a, Hidden: AxisName, All: AxisName> Op<(&'a Staged<Hidden, All>, usize)> for PairRows {
    type Output = Input<Hidden, TilePair>;

    fn forward(
        &self,
        context: &mut Device,
        (staged, pair): (&'a Staged<Hidden, All>, usize),
    ) -> Self::Output {
        let rows = staged.view().tile::<m![All / Tile], 2,
            m![Hidden / 64, All / Tile = 2 # (All / Tile), All % Tile, Hidden % 64]>(2 * pair);
        // SAFETY: paired waves do not overlap, so the pair's rows are its waves' in order.
        let rows = unsafe {
            rows.reshape::<Chip, m![Hidden / 64, TilePair # (All / Tile), Tile, Hidden % 64]>()
        };
        rows.to_dm(&mut context.tdma)
    }
}

/// Pair `pair`'s row scales, written from the staged scales in the slices of each head partition.
pub(crate) struct PairScales<Heads>(pub(crate) PhantomData<Heads>);

impl<'a, All: AxisName, Heads: AxisName> Op<(&'a StagedScales<All>, usize)> for PairScales<Heads> {
    type Output = RowGroups<TilePair, Heads>;

    fn forward(
        &self,
        context: &mut Device,
        (scales, pair): (&'a StagedScales<All>, usize),
    ) -> Self::Output {
        // SAFETY: the same order with the 64-row block split into its wave and half.
        let scales = unsafe {
            scales
                .view()
                .reshape::<Chip, m![All % 64 / 8, All / Tile, All % Tile / 64, All % 8]>()
        };
        let rows = scales.tile::<m![All / Tile], 2,
            m![All % 64 / 8, All / Tile = 2 # (All / Tile), All % Tile / 64, All % 8]>(2 * pair);
        // SAFETY: a pair's rows are its waves' in order: wave, half of 64 rows, block, row.
        let rows = unsafe {
            rows.reshape::<Chip, m![Tile % 64 / 8, TilePair # (All / Tile), Tile / 64, Tile % 8]>()
        };
        rows.to_dm(&mut context.tdma)
    }
}
