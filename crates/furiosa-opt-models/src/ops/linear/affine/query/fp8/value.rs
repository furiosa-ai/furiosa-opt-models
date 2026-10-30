use super::*;
use crate::Op;
use core::marker::PhantomData;
use std::borrow::Borrow;

pub(crate) struct Counted<'a, Hidden: AxisName, G: Group, Heads: AxisName> {
    pub(crate) lanes: &'a Lanes<Hidden, G, Heads>,
    pub(crate) rows: &'a RowGroups<G, Heads>,
    pub(crate) groups: usize,
}

/// The value projection with each head's scales fetched into the VRF of the group geometry `G`;
/// the weight stays where it was written.
pub(crate) struct Stage<G>(pub(crate) PhantomData<G>);

impl<'a, Hidden: AxisName, Heads: AxisName, Head: AxisName, G: Group>
    Op<&'a Affine<(Weight<Hidden, Heads, Head>, ValueScale<Hidden, Heads, Head>), ()>>
    for Stage<G>
{
    type Output = Affine<(&'a Weight<Hidden, Heads, Head>, HeadScale<G, Heads, Head>), ()>;

    fn forward(
        &self,
        context: &mut Device,
        value: &'a Affine<(Weight<Hidden, Heads, Head>, ValueScale<Hidden, Heads, Head>), ()>,
    ) -> Self::Output {
        let head_scale = |context: &mut Device,
                          scale: &ValueScale<Hidden, Heads, Head>|
         -> HeadScale<G, Heads, Head> {
            // SAFETY: both partitions hold a head's scales in every slice of its head digits.
            let scale = unsafe {
                scale.view().reshape::<
                    Chip,
                    Cluster<Heads>,
                    Reduced<G, Heads, <G as Group>::Copies,
                >,
                    m![Heads % (Heads / 8), Head]>()
            };
            context
                .sub
                .begin(scale)
                .fetch::<m![Heads % (Heads / 8), Head / 8], m![Head % 8]>()
                .collect::<m![Heads % (Heads / 8), Head / 8], m![Head % 8]>()
                .to_vrf()
        };
        Affine::new((&value.weight.0, head_scale(context, &value.weight.1)), ())
    }
}

impl<Hidden: AxisName, Head: AxisName, Heads: AxisName, G: Group, W>
    Op<Counted<'_, Hidden, G, Heads>> for Affine<(W, HeadScale<G, Heads, Head>), ()>
where
    W: Borrow<Weight<Hidden, Heads, Head>>,
{
    type Output = Pairs<G, Heads, Head>;

    fn forward(&self, context: &mut Device, input: Counted<'_, Hidden, G, Heads>) -> Self::Output {
        let (weight, head) = (self.weight.0.borrow(), &self.weight.1);
        let (lanes, rows, groups) = (input.lanes, input.rows, input.groups);
        let row: VrfTensor<f32, Chip, Cluster<Heads>, Reduced<G, Heads, G::Copies>, m![Tile % 8]> =
            context
                .sub
                .begin(rows.view())
                .fetch::<m![1], m![Tile % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![1], m![Tile % 8]>()
                .to_vrf();
        let mut output: DmTensor<
            bf16,
            Chip,
            Cluster<Heads>,
            Reduced<G, Heads, G::Copies>,
            m![Heads % (Heads / 8), Tile % 8, Head],
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
            .vector_inter_slice_reduce::<Reduced<G, Heads, G::Copies>,
                m![Heads % (Heads / 8), Head / 64, Head % 64]>(InterSliceReduceOpF32::Add)
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Heads % (Heads / 8), Head / 64, Head % 64, Tile % 8 / 4],
                m![Tile % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &row)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), head)
            .vector_widen_concat::<m![Heads % (Heads / 8), Head / 64, Head % 64], m![Tile % 8]>()
            .vector_final()
            .cast::<bf16, m![Tile % 8 # 16]>()
            .transpose::<m![Heads % (Heads / 8), Head / 64, Head % 64 / 4, Tile % 8],
                m![Head % 4 # 16]>()
            .commit_trim::<m![Head % 4]>();
        if const { G::SIZE == 1 } {
            products.commit_prefix::<m![Tile / 8], _, _>(output.view_mut(), groups)
        } else {
            products.commit_view(output.view_mut())
        }
        // SAFETY: BF16's slice names for Reduced's slices; no sum moves.
        unsafe { output.reshape() }
    }
}
