//! Unrounded projection streams lent to a fused consumer.

use furiosa_opt_std::prelude::*;

use super::{Mapping, Scales};
use crate::Module;
use crate::ops::layout::Dm;
use crate::ops::linear::Affine;

impl<'d, 'x, 'w, 's, B: M, T: Mapping>
    Module<(
        &'d mut Device,
        Dm<T, (&'x TrfTensor<f8e4m3, B, T::Cluster, T::Slices, T::Lanes, T::Input>,)>,
    )>
    for Affine<
        &'w DmTensor<f8e4m3, B, T::Cluster, T::Slices, T::Weight>,
        &'s Scales<T::Cluster, T::Reduced, B>,
    >
where
    'w: 'd,
{
    type Output = Dm<
        T,
        VectorFpTensor<
            'd,
            { Tu::Main },
            f32,
            B,
            T::Cluster,
            T::Reduced,
            m![{ T::Output }, { T::Lanes } / 4],
            m![{ T::Lanes } % 4],
            Fresh,
            { stage::VeOrder::InterFirst },
        >,
    >;

    fn forward(
        &self,
        (device, input): (
            &'d mut Device,
            Dm<T, (&'x TrfTensor<f8e4m3, B, T::Cluster, T::Slices, T::Lanes, T::Input>,)>,
        ),
    ) -> Self::Output {
        Dm {
            tiles: input.tiles,
            value: device
                .main
                .begin(self.weight.view())
                .fetch::<T::Feed, T::Packet>()
                .collect::<T::Flits, T::Flit>()
                .contract_outer::<T::Steps, T::Packet, _, _, _>(input.value.0)
                .contract_packet::<T::Contracted>()
                .contract_time::<T::Sums>()
                .contract_lane::<T::Groups, T::Lanes>(LaneMode::Interleaved)
                .vector_init()
                .vector_inter_slice_reduce::<T::Reduced, T::Output>(InterSliceReduceOpF32::Add)
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![{ T::Output }, { T::Lanes } / 4], m![{ T::Lanes } % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &self.bias.input)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &self.bias.weight),
        }
    }
}
