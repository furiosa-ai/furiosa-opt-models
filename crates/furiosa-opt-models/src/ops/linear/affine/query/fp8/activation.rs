use super::*;
use crate::Op;
use crate::ops::vector::Quantization;
use core::marker::PhantomData;

pub(super) type InDm<Hidden, G> =
    DmTensor<bf16, Chip, m![2], m![4, Hidden / 64], m![G, Tile, Hidden % 64]>;
axes![Copies = 64];
type Scales<G> = DmTensor<bf16, Chip, m![2], m![4, Copies], m![G, Tile]>;
type Activation<Hidden, G> = (Input<Hidden, G>, Scales<G>);
/// The quantization of a group's written rows.
type Rows<Hidden, G> = Quantization<
    m![G, Tile, Hidden % 64 / 8],
    m![G, Tile, Hidden % 64 / 4],
    Hidden,
    m![G, Tile],
    m![4, Copies],
    m![G, Tile / 4],
    m![Tile % 4],
>;

/// A group's written rows as E4M3 rows and their BF16 row scales, the first `groups` row groups
/// alone when the group is one tile.
pub(crate) struct Quantize;

impl<'a, Hidden: AxisName, G: Group> Op<(&'a InDm<Hidden, G>, usize)> for Quantize {
    type Output = Activation<Hidden, G>;

    fn forward(
        &self,
        context: &mut Device,
        (input, groups): (&'a InDm<Hidden, G>, usize),
    ) -> Self::Output {
        let mut scales = Scales::<G>::new();
        let maxima = Rows::<Hidden, G>::scales(&mut context.main, input.view());
        if const { G::SIZE == 1 } {
            maxima.commit_prefix::<m![Tile / 8], _, _>(scales.view_mut(), groups)
        } else {
            maxima.commit_view(scales.view_mut())
        }
        // SAFETY: the reduction broadcasts each row's scale to every slice of its cluster.
        let broadcast = unsafe {
            scales
                .view()
                .reshape::<Chip, m![2], m![4, Hidden / 64], m![G, Tile]>()
        };
        let scale: VrfTensor<f32, Chip, m![2], m![4, Hidden / 64], m![G, Tile]> = context
            .sub
            .begin(broadcast)
            .fetch::<m![G, Tile / 8], m![Tile % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![G, Tile / 8], m![Tile % 8]>()
            .to_vrf();
        let mut quantized = Input::<Hidden, G>::new();
        let coded = Rows::<Hidden, G>::rows(&mut context.main, input.view(), &scale);
        if const { G::SIZE == 1 } {
            coded.commit_prefix::<m![Tile / 8], _, _>(quantized.view_mut(), groups)
        } else {
            coded.commit_view(quantized.view_mut())
        }
        (quantized, scales)
    }
}

/// Each row's scale, in the slices of its head partition.
pub(crate) struct RowScales<Hidden, Heads>(pub(crate) PhantomData<(Hidden, Heads)>);

impl<'a, Hidden: AxisName, G: Group, Heads: AxisName> Op<&'a Scales<G>>
    for RowScales<Hidden, Heads>
{
    type Output = RowGroups<G, Heads>;

    fn forward(&self, context: &mut Device, scales: &'a Scales<G>) -> Self::Output {
        // SAFETY: the reduction broadcasts each row's scale to every slice of its cluster.
        let broadcast = unsafe {
            scales
                .view()
                .reshape::<Chip, Cluster<Heads>, Slice<Hidden, Heads>, m![G, Tile]>()
        };
        let mut rows = RowGroups::<G, Heads>::new();
        broadcast
            .slice_tile::<m![Hidden / 64], 1, m![Heads % (Heads / 2) / (Heads / 4),
            Heads % (Heads / 4) / (Heads / 8), Hidden / 64 = 1 # 64]>(0)
            .to_dm_view(&mut context.tdma, rows.view_mut());
        rows
    }
}

/// The quantized rows in the TRF of each head partition, the first `groups` row groups alone when
/// the group is one tile.
pub(crate) struct Fill<Heads>(pub(crate) PhantomData<Heads>);

impl<'a, Hidden: AxisName, G: Group, Heads: AxisName> Op<(&'a Input<Hidden, G>, usize)>
    for Fill<Heads>
{
    type Output = Lanes<Hidden, G, Heads>;

    fn forward(
        &self,
        context: &mut Device,
        (input, groups): (&'a Input<Hidden, G>, usize),
    ) -> Self::Output {
        // SAFETY: both head partitions rename replicated coordinates; no activation moves.
        let input = unsafe {
            input
                .view()
                .reshape::<Chip, Cluster<Heads>, Slice<Hidden, Heads>, m![G, Tile, Hidden % 64]>()
        };
        let fetched = context
            .sub
            .begin(input)
            .fetch::<m![Tile % 8, G, Tile / 8, Hidden % 64 / 32], m![Hidden % 32]>()
            .collect::<m![Tile % 8, G, Tile / 8, Hidden % 64 / 32], m![Hidden % 32]>();
        if const { G::SIZE == 1 } {
            fetched.to_trf_prefix::<m![G, Tile / 8], _, _>(groups)
        } else {
            fetched.to_trf()
        }
    }
}
