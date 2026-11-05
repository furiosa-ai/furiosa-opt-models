//! Projection partials and scales executed at separate composer positions.

use furiosa_opt_std::prelude::*;

use super::{Mapping, Scales};
use crate::Module;
use crate::ops::layout::Dm;
use crate::ops::linear::Affine;

impl<'d, 'x, 'w, B: M, T: Mapping>
    Module<(
        &'d mut Device,
        Dm<T, (DmTensorView<'x, f8e4m3, B, T::Cluster, T::Slices, T::Input>,)>,
    )> for Affine<&'w TrfTensor<f8e4m3, B, T::Cluster, T::Slices, T::Lanes, T::Weight>, ()>
where
    'x: 'd,
{
    type Output = Dm<
        T,
        VectorFinalTensor<'d, { Tu::Main }, f32, B, T::Cluster, T::Reduced, T::Output, T::Lanes>,
    >;

    fn forward(
        &self,
        (device, input): (
            &'d mut Device,
            Dm<T, (DmTensorView<'x, f8e4m3, B, T::Cluster, T::Slices, T::Input>,)>,
        ),
    ) -> Self::Output {
        Dm {
            tiles: input.tiles,
            value: device
                .main
                .begin(input.value.0)
                .fetch::<T::Feed, T::Packet>()
                .collect::<T::Flits, T::Flit>()
                .contract_outer::<T::Steps, T::Packet, _, _, _>(self.weight)
                .contract_packet::<T::Contracted>()
                .contract_time::<T::Sums>()
                .contract_lane::<T::Groups, T::Lanes>(LaneMode::Interleaved)
                .vector_init()
                .vector_inter_slice_reduce::<T::Reduced, T::Output>(InterSliceReduceOpF32::Add)
                .vector_final(),
        }
    }
}

impl<'d, 'x, 'w, B: M, T: Mapping>
    Module<(
        &'d mut Device,
        Dm<T, (&'x TrfTensor<f8e4m3, B, T::Cluster, T::Slices, T::Lanes, T::Input>,)>,
    )> for Affine<DmTensorView<'w, f8e4m3, B, T::Cluster, T::Slices, T::Weight>, ()>
where
    'w: 'd,
{
    type Output = Dm<
        T,
        VectorFinalTensor<'d, { Tu::Main }, f32, B, T::Cluster, T::Reduced, T::Output, T::Lanes>,
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
                .begin(self.weight.clone())
                .fetch::<T::Feed, T::Packet>()
                .collect::<T::Flits, T::Flit>()
                .contract_outer::<T::Steps, T::Packet, _, _, _>(input.value.0)
                .contract_packet::<T::Contracted>()
                .contract_time::<T::Sums>()
                .contract_lane::<T::Groups, T::Lanes>(LaneMode::Interleaved)
                .vector_init()
                .vector_inter_slice_reduce::<T::Reduced, T::Output>(InterSliceReduceOpF32::Add)
                .vector_final(),
        }
    }
}

impl<'d, 's, B: M, C: M, S: M, U: M, P: M, T: crate::compose::Space, V, F>
    Module<
        Dm<
            T,
            VectorTensor<
                'd,
                { Tu::Main },
                V,
                f32,
                B,
                C,
                S,
                U,
                P,
                Fresh,
                { stage::VeOrder::IntraFirst },
                F,
                { stage::Way::Way4 },
            >,
        >,
    > for Affine<(), &'s Scales<C, S, B>>
where
    V: stage::Stage + stage::CanTransitionTo<stage::Fp>,
    F: stage::VeTensorContext,
{
    type Output = Dm<
        T,
        VectorFpTensor<'d, { Tu::Main }, f32, B, C, S, U, P, Fresh, { stage::VeOrder::IntraFirst }>,
    >;

    fn forward(
        &self,
        input: Dm<
            T,
            VectorTensor<
                'd,
                { Tu::Main },
                V,
                f32,
                B,
                C,
                S,
                U,
                P,
                Fresh,
                { stage::VeOrder::IntraFirst },
                F,
                { stage::Way::Way4 },
            >,
        >,
    ) -> Self::Output {
        Dm {
            tiles: input.tiles,
            value: input
                .value
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &self.bias.input)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &self.bias.weight),
        }
    }
}
