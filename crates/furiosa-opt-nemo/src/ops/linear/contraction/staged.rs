//! Projection stages that retain their parameters across composer-placed transfers.

use furiosa_opt_std::prelude::*;

use super::Scales;
use crate::Module;
use crate::compose::Space;
use crate::ops::layout::Dm;
use crate::ops::linear::Affine;
axes![Term = 2];

/// Mappings for a projection whose input rows stay in registers.
pub trait Stationary: Space {
    type Cluster: M;
    type Slices: M;
    type Reduced: M;
    type Weight: M;
    type Register: M;
    type Lanes: M;
    type Feed: M;
    type Packet: M;
    type Flits: M;
    type Flit: M;
    type Steps: M;
    type Contracted: M;
    type TransposeTime: M;
    type Sums: M;
    type Groups: M;
    type Output: M;
    type Transposed: M;
}

impl<'d, 'x, 'w, 's, B: M, T: Stationary>
    Module<(
        &'d mut Device,
        Dm<
            T,
            (
                &'x TrfTensor<f8e4m3, B, T::Cluster, T::Slices, T::Lanes, T::Register>,
                DmTensorViewMut<'x, bf16, B, T::Cluster, T::Reduced, T::Transposed>,
            ),
        >,
    )>
    for Affine<
        &'w DmTensor<f8e4m3, B, T::Cluster, T::Slices, T::Weight>,
        &'s Scales<T::Cluster, T::Reduced, B>,
    >
{
    type Output = ();

    fn forward(
        &self,
        (device, input): (
            &'d mut Device,
            Dm<
                T,
                (
                    &'x TrfTensor<f8e4m3, B, T::Cluster, T::Slices, T::Lanes, T::Register>,
                    DmTensorViewMut<'x, bf16, B, T::Cluster, T::Reduced, T::Transposed>,
                ),
            >,
        ),
    ) {
        let value = device
            .main
            .begin(self.weight.view())
            .fetch::<T::Feed, T::Packet>()
            .collect::<T::Flits, T::Flit>()
            .contract_outer::<T::Steps, T::Packet, _, _, _>(input.value.0)
            .contract_packet::<T::Contracted>()
            .contract_time::<T::Sums>()
            .contract_lane::<T::Groups, T::Lanes>(LaneMode::Interleaved)
            .vector_init();
        self.bias
            .apply::<T::Slices, T::Groups, T::Output, T::Lanes>(value)
            .cast::<bf16, m![{ T::Lanes } # 16]>()
            .transpose::<T::TransposeTime, m![{ T::Contracted } # 16]>()
            .commit_trim::<T::Contracted>()
            .commit_view(input.value.1);
    }
}

/// Mappings for a projection reduced across clusters between contraction and scaling.
pub trait Split: Space {
    type Cluster: M;
    type Slices: M;
    type InputSlices: M;
    type Input: M;
    type Reduced: M;
    type Rows: M;
    type Lanes: M;
    type Weight: M;
    type Feed: M;
    type Packet: M;
    type Switched: M;
    type Operand: M;
    const BROADCAST: SwitchConfig;
    type Flits: M;
    type Flit: M;
    type Steps: M;
    type Contracted: M;
    type Sums: M;
    type Partial: M;
    type Kept: M;
    type Interleaved: M;
    type Scale: M;
    type Output: M;
}

/// Register scales retained until the projection's cluster partials are available.
pub struct Factors<B: M, C: M, S: M, W: M> {
    pub input: VrfTensor<f32, B, C, S, m![1 # 8]>,
    pub weight: VrfTensor<f32, B, C, S, W>,
}

impl<'d, 'x, B: M, T: Split>
    Module<(
        &'d mut Device,
        Dm<T, &'x DmTensor<f8e4m3, B, T::Cluster, T::InputSlices, T::Input>>,
    )>
    for Affine<
        TrfTensor<f8e4m3, B, T::Cluster, T::Slices, T::Lanes, T::Weight>,
        Factors<B, T::Rows, T::Reduced, T::Scale>,
    >
{
    type Output = Dm<T, DmTensor<f32, B, T::Cluster, T::Reduced, m![{ T::Partial }, { T::Lanes }]>>;

    fn forward(
        &self,
        (device, input): (
            &'d mut Device,
            Dm<T, &'x DmTensor<f8e4m3, B, T::Cluster, T::InputSlices, T::Input>>,
        ),
    ) -> Self::Output {
        Dm {
            tiles: input.tiles,
            value: device
                .main
                .begin(input.value.view())
                .fetch::<T::Feed, T::Packet>()
                .switch::<T::Slices, T::Switched>(T::BROADCAST)
                .collect::<T::Flits, T::Flit>()
                .contract_outer::<T::Steps, T::Operand, _, _, _>(&self.weight)
                .contract_packet::<T::Contracted>()
                .contract_time::<T::Sums>()
                .contract_lane::<T::Sums, T::Lanes>(LaneMode::Interleaved)
                .vector_init()
                .vector_inter_slice_reduce::<T::Reduced, T::Partial>(InterSliceReduceOpF32::Add)
                .vector_final()
                .commit_trim::<T::Lanes>()
                .commit(),
        }
    }
}

impl<'d, 'x, B: M, T: Split>
    Module<(
        &'d mut Device,
        Dm<
            T,
            (
                &'x DmTensor<f32, B, T::Rows, T::Reduced, m![{ T::Kept }, { T::Lanes }]>,
                &'x DmTensor<f32, B, T::Rows, T::Reduced, m![{ T::Kept }, { T::Lanes }]>,
                DmTensorViewMut<'x, bf16, B, T::Rows, T::Reduced, T::Output>,
            ),
        >,
    )>
    for Affine<
        TrfTensor<f8e4m3, B, T::Cluster, T::Slices, T::Lanes, T::Weight>,
        Factors<B, T::Rows, T::Reduced, T::Scale>,
    >
{
    type Output = ();

    fn forward(
        &self,
        (device, input): (
            &'d mut Device,
            Dm<
                T,
                (
                    &'x DmTensor<f32, B, T::Rows, T::Reduced, m![{ T::Kept }, { T::Lanes }]>,
                    &'x DmTensor<f32, B, T::Rows, T::Reduced, m![{ T::Kept }, { T::Lanes }]>,
                    DmTensorViewMut<'x, bf16, B, T::Rows, T::Reduced, T::Output>,
                ),
            >,
        ),
    ) {
        device
            .main
            .begin_interleaved::<Term, _, _, _, _, _>(input.value.0.view(), input.value.1.view())
            .fetch::<T::Interleaved, T::Lanes>()
            .collect::<T::Interleaved, T::Lanes>()
            .vector_init()
            .vector_intra_slice_unzip::<Term, T::Kept>()
            .vector_narrow_split::<m![{ T::Kept }, { T::Lanes } / 4], m![{ T::Lanes } % 4]>()
            .vector_fp_zip(FpBinaryOp::AddF)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &self.bias.input)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &self.bias.weight)
            .vector_widen_concat::<T::Kept, T::Lanes>()
            .vector_final()
            .cast::<bf16, m![{ T::Lanes } # 16]>()
            .commit_trim::<T::Lanes>()
            .commit_view(input.value.2);
    }
}
