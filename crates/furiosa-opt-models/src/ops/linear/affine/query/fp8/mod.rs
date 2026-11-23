use super::launch::Normalized;
use super::{Cluster, Output, Rows, Tile};
pub(crate) use super::{
    PairLocal, PairSlices, Pairs, Queries, QueryLocal, QuerySlices, QuerySpread, Reduced,
};
use crate::ops::linear::query::group::Group;
use crate::ops::normalization::Scaled;
use crate::tensor::parallel::Chip;
use crate::{Op, ops::linear::Affine};
pub(crate) use activation::{Fill, Quantize, RowScales};
use furiosa_opt_std::prelude::*;
pub(crate) use spread::{PairRows, PairScales, Stage};
use std::marker::PhantomData;
pub(crate) use to_dm::ToDm;

mod activation;
mod produce;
mod sink;
pub(crate) mod source;
mod spread;
mod to_dm;
pub(crate) mod value;

pub(crate) type Slice<Hidden, Heads> = m![
    Heads % (Heads / 2) / (Heads / 4),
    Heads % (Heads / 4) / (Heads / 8),
    Hidden / 64
];
pub(crate) type Weight<Hidden, Heads, Head> = DmTensor<
    f8e4m3,
    Chip,
    Cluster<Heads>,
    Slice<Hidden, Heads>,
    m![Head / 64, Heads % (Heads / 8), Head % 64, Hidden % 64],
>;
pub type Layout<Hidden, Heads, Head> = m![
    Heads / (Heads / 2),
    Heads % (Heads / 2) / (Heads / 4),
    Heads % (Heads / 4) / (Heads / 8),
    Hidden / 64,
    Head / 64,
    Heads % (Heads / 8),
    Head % 64,
    Hidden % 64
];
pub type Stored<Hidden, Heads, Head> = HbmTensor<f8e4m3, Chip, Layout<Hidden, Heads, Head>>;
/// A head projection's stored weight: its codes, with `S` the row scales it keeps, if any.
pub type Weights<Hidden, Heads, Head, S = ()> =
    crate::tensor::form::fp8::Rows<m![Hidden], m![Heads, Head], Stored<Hidden, Heads, Head>, S>;
/// The value projection's row scales, its heads' scales.
pub type HeadScales<Heads, Head> = HbmTensor<f32, Chip, m![Heads, Head]>;
/// A query or key projection's row scales: each head's square, for each row of a tile.
pub type HeadSquares<Heads> = crate::tensor::form::fp8::Squares<Squared<Heads>>;
/// Each head's squared scale, for each row of a tile.
pub type Squared<Heads> = HbmTensor<f32, Chip, m![Heads, Tile]>;
pub(crate) type Input<Hidden, G> =
    DmTensor<f8e4m3, Chip, m![2], m![4, Hidden / 64], m![G, Tile, Hidden % 64]>;
pub(crate) type Lanes<Hidden, G, Heads> = TrfTensor<
    f8e4m3,
    Chip,
    Cluster<Heads>,
    Slice<Hidden, Heads>,
    m![Tile % 8],
    m![G, Tile / 8, Hidden % 64],
>;
pub(crate) type ValueScale<Hidden, Heads, Head> =
    DmTensor<f32, Chip, Cluster<Heads>, Slice<Hidden, Heads>, m![Heads % (Heads / 8), Head]>;
pub(crate) type RowGroups<G, Heads> =
    DmTensor<bf16, Chip, Cluster<Heads>, Reduced<G, Heads, <G as Group>::Copies>, m![Tile % 8]>;
pub(crate) type HeadScale<G, Heads, Head> = VrfTensor<
    f32,
    Chip,
    Cluster<Heads>,
    Reduced<G, Heads, <G as Group>::Copies>,
    m![Heads % (Heads / 8), Head],
>;

type Outputs<Heads, Head> = m![Heads % (Heads / 8), Head / 64, Head % 64];

/// Each row's factor of a projection's output, `1 / (c s)²`, in its slices: one for each of its
/// `Local` heads.
pub(crate) type Inverse<G, Heads, Spread, Local> =
    VrfTensor<f32, Chip, Cluster<Heads>, Rows<G, Heads, Spread>, m![{ Local }, { m![Tile % 8] }]>;

pub(crate) struct Counted<'a, Hidden: AxisName, G: Group, Heads: AxisName, Spread: M, Local: M> {
    pub(crate) lanes: &'a Lanes<Hidden, G, Heads>,
    /// The scale of each row of `lanes`, in the slices of the output's heads.
    pub(crate) rows:
        DmTensorView<'a, bf16, Chip, Cluster<Heads>, Reduced<G, Heads, Spread>, m![Tile % 8]>,
    pub(crate) groups: usize,
    pub(crate) output: PhantomData<(Spread, Local)>,
}

/// Each head's squared weight scale, for each row of a tile, in the slices of a projection's
/// output that keep `Local` of its heads.
pub(crate) type Squares<G, Heads, Spread, Local> =
    VrfTensor<f32, Chip, Cluster<Heads>, Rows<G, Heads, Spread>, m![{ Local }, { m![Tile % 8] }]>;

/// [`Squares`] written in data memory, before the register file reads them.
pub(crate) type Written<G, Heads, Spread, Local> =
    DmTensor<f32, Chip, Cluster<Heads>, Rows<G, Heads, Spread>, m![{ Local }, Tile % 8]>;

/// The projection of a period's rows; its heads' squares, written beforehand, are fetched once the
/// contraction is issued, so the contraction does not wait for them.
impl<Hidden: AxisName, Head: AxisName, G: Group, Heads: AxisName, Spread: M, Local: M>
    Op<(
        Counted<'_, Hidden, G, Heads, Spread, Local>,
        &Written<G, Heads, Spread, Local>,
    )> for Affine<&Weight<Hidden, Heads, Head>, ()>
{
    type Output = Scaled<Output<G, Heads, Head, Spread, Local>, Inverse<G, Heads, Spread, Local>>;

    fn forward(
        &self,
        context: &mut Device,
        (input, written): (
            Counted<'_, Hidden, G, Heads, Spread, Local>,
            &Written<G, Heads, Spread, Local>,
        ),
    ) -> Self::Output {
        let (weight, lanes, rows, groups) = (self.weight, input.lanes, input.rows, input.groups);
        let mut output: DmTensor<
            bf16,
            Chip,
            Cluster<Heads>,
            Reduced<G, Heads, Spread>,
            m![{ Local }, Tile % 8, Head],
        > = DmTensor::new();
        let products = context.main.begin(weight.view())
            .fetch::<Outputs<Heads, Head>, m![Hidden % 64]>()
            .collect::<m![{ Outputs<Heads, Head> }, Hidden % 64 / 32], m![Hidden % 32]>()
            .contract_outer::<
                m![{ Outputs<Heads, Head> }, G, Tile / 8],
                m![Hidden % 64],
                _,
                _,
                _,
            >(lanes)
            .contract_packet::<m![1]>()
            .contract_time::<m![{ Outputs<Heads, Head> }, G, Tile / 8]>()
            .contract_lane::<
                m![{ Outputs<Heads, Head> }, G, Tile / 8],
                m![Tile % 8],
            >(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<Reduced<G, Heads, Spread>,
                m![{ Local }, Head / 64, Head % 64]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![Tile % 8 # 16]>()
            .transpose::<m![{ Local }, Head / 64, Head % 64 / 4, Tile % 8], m![Head % 4 # 16]>()
            .commit_trim::<m![Head % 4]>();
        if const { G::SIZE == 1 } {
            products.commit_prefix::<m![Tile / 8], _, _>(output.view_mut(), groups)
        } else {
            products.commit_view(output.view_mut())
        }
        let squares: Squares<G, Heads, Spread, Local> = context
            .sub
            .begin(written.view())
            .fetch::<m![{ Local }], m![Tile % 8]>()
            .collect::<m![{ Local }], m![Tile % 8]>()
            .to_vrf();
        // SAFETY: BF16's slice names for Reduced's slices, for the products and their rows' scales
        // alike; no sum moves.
        Scaled::new(
            context,
            unsafe { output.reshape() },
            unsafe { rows.reshape() },
            &squares,
        )
    }
}

/// The squares of a projection's head scales, written into the slices of its output.
pub(crate) fn squares<G: Group, Heads: AxisName, Spread: M, Local: M>(
    context: &mut Device,
    stored: &Squared<Heads>,
) -> Written<G, Heads, Spread, Local> {
    let written: DmTensor<
        f32,
        Chip,
        Cluster<Heads>,
        Reduced<G, Heads, Spread>,
        m![{ Local }, Tile % 8],
    > = stored.to_dm(&mut context.tdma);
    // SAFETY: BF16's slice names for Reduced's slices, as the products' are; row digits only
    // replicate each head's square.
    unsafe { written.reshape() }
}
