use super::{Cluster, Output, Tile};
use crate::compose::Scaled;
use crate::ops::layout::axes::Chip;
use crate::ops::linear::query::group::Group;
use crate::{Module, ops::linear::Affine};
use furiosa_opt_std::prelude::*;

mod activation;
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
pub(crate) type Lanes<Hidden, G, Heads> = TrfTensor<
    f8e4m3,
    Chip,
    Cluster<Heads>,
    Slice<Hidden, Heads>,
    m![Tile % 8],
    m![G, Tile / 8, Hidden % 64],
>;
pub(crate) type Reduced<G, Heads, Spread> = m![
    Heads % (Heads / 2) / (Heads / 4),
    Heads % (Heads / 4) / (Heads / 8),
    G,
    Tile / 8,
    { Spread }
];
pub(crate) type RowGroups<G, Heads> =
    DmTensor<bf16, Chip, Cluster<Heads>, Reduced<G, Heads, <G as Group>::Copies>, m![Tile % 8]>;
pub(crate) type HeadScale<G, Heads, Head> = VrfTensor<
    f32,
    Chip,
    Cluster<Heads>,
    Reduced<G, Heads, <G as Group>::Copies>,
    m![Heads % (Heads / 8), Head],
>;

macro_rules! commit_live {
    ($chain:expr, $output:expr, $groups:expr) => {
        if const { G::SIZE == 1 } {
            $chain.commit_prefix::<m![Tile / 8], _, _>($output, $groups)
        } else {
            $chain.commit_view($output)
        }
    };
}
pub(crate) use commit_live;

type Outputs<Heads, Head> = m![Heads % (Heads / 8), Head / 64, Head % 64];

pub(crate) struct Counted<'a, Hidden: AxisName, G: Group, Heads: AxisName, Spread: M, Local: M> {
    pub(crate) lanes: &'a Lanes<Hidden, G, Heads>,
    pub(crate) groups: usize,
    pub(crate) output: std::marker::PhantomData<(Spread, Local)>,
}

impl<Hidden: AxisName, Head: AxisName, G: Group, Heads: AxisName, Spread: M, Local: M>
    Module<(&mut Device, Counted<'_, Hidden, G, Heads, Spread, Local>)>
    for Affine<Weight<Hidden, Heads, Head>, ()>
{
    type Output = Scaled<Output<G, Heads, Head, Spread, Local>>;

    fn forward(
        &self,
        (context, input): (&mut Device, Counted<'_, Hidden, G, Heads, Spread, Local>),
    ) -> Self::Output {
        let (weight, lanes, groups) = (&self.weight, input.lanes, input.groups);
        let mut output: DmTensor<
            bf16,
            Chip,
            Cluster<Heads>,
            Reduced<G, Heads, Spread>,
            m![{ Local }, Tile % 8, Head],
        > = DmTensor::new();
        commit_live!(context.main.begin(weight.view())
            .fetch::<Outputs<Heads, Head>, m![Hidden % 64]>()
            .collect::<m![{ Outputs<Heads, Head> }, Hidden % 64 / 32], m![Hidden % 32]>()
            .contract_outer::<m![{ Outputs<Heads, Head> }, G, Tile / 8], m![Hidden % 64], _, _, _>(lanes)
            .contract_packet::<m![1]>()
            .contract_time::<m![{ Outputs<Heads, Head> }, G, Tile / 8]>()
            .contract_lane::<m![{ Outputs<Heads, Head> }, G, Tile / 8], m![Tile % 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<Reduced<G, Heads, Spread>,
                m![{ Local }, Head / 64, Head % 64]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![Tile % 8 # 16]>()
            .transpose::<m![{ Local }, Head / 64, Head % 64 / 4, Tile % 8], m![Head % 4 # 16]>()
            .commit_trim::<m![Head % 4]>(), output.view_mut(), groups);
        // SAFETY: BF16's slice names for Reduced's slices; no sum moves.
        Scaled::new(unsafe { output.reshape() })
    }
}
