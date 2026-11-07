//! Resident E4M3 projection contractions with retained input and weight scales.

use furiosa_opt_std::prelude::*;

use super::Affine;
use crate::Module;
use crate::compose::Space;
use crate::ops::layout::Dm;

pub mod broadcast;
pub mod deferred;
pub mod resident;
pub mod staged;
pub mod stationary;
pub mod stream;

/// A projection's input scale, retained before its coefficient fill.
pub type InputScale<B, C, S> = Affine<(), VrfTensor<f32, B, C, S, m![1 # 8]>>;

impl<B: M, C: M, S: M> InputScale<B, C, S> {
    pub fn input_scale(&self) -> &VrfTensor<f32, B, C, S, m![1 # 8]> {
        &self.bias
    }
}

/// The physical mappings of one projection tile, including its reduction and output coordinates.
pub trait Mapping: Space {
    type Cluster: M;
    type Slices: M;
    type Reduced: M;
    type Input: M;
    type Lanes: M;
    type Weight: M;
    type Feed: M;
    type Packet: M;
    type Flits: M;
    type Flit: M;
    type Steps: M;
    type Contracted: M;
    type Sums: M;
    type Groups: M;
    type Output: M;
}

/// Scales retained by a projection after their register landing.
pub struct Scales<C: M, S: M, B: M> {
    pub input: VrfTensor<f32, B, C, S, m![1 # 8]>,
    pub weight: VrfTensor<f32, B, C, S, m![1 # 8]>,
}

impl<C: M, S: M, B: M> Scales<C, S, B> {
    pub fn apply<'l, I: M, T: M, U: M, P: M>(
        &self,
        value: VectorInitTensor<'l, { Tu::Main }, f32, B, C, I, T, P>,
    ) -> VectorFinalTensor<'l, { Tu::Main }, f32, B, C, S, U, P> {
        const { assert!(P::SIZE == 8) };
        value
            .vector_inter_slice_reduce::<S, U>(InterSliceReduceOpF32::Add)
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ U }, { P } / 4], m![{ P } % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &self.input)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &self.weight)
            .vector_widen_concat::<U, P>()
            .vector_final()
    }
}

/// Contract a resident tile and restore both scales after reducing its feature partitions.
impl<'d, 'x, B: M, T: Mapping>
    Module<(
        &'d mut Device,
        Dm<T, &'x DmTensor<f8e4m3, B, T::Cluster, T::Slices, T::Input>>,
    )>
    for Affine<
        TrfTensor<f8e4m3, B, T::Cluster, T::Slices, T::Lanes, T::Weight>,
        Scales<T::Cluster, T::Reduced, B>,
    >
{
    type Output = Dm<T, DmTensor<bf16, B, T::Cluster, T::Reduced, m![{ T::Output }, { T::Lanes }]>>;

    fn forward(
        &self,
        (device, input): (
            &'d mut Device,
            Dm<T, &'x DmTensor<f8e4m3, B, T::Cluster, T::Slices, T::Input>>,
        ),
    ) -> Self::Output {
        let value = device
            .main
            .begin(input.value.view())
            .fetch::<T::Feed, T::Packet>()
            .collect::<T::Flits, T::Flit>()
            .contract_outer::<T::Steps, T::Packet, _, _, _>(&self.weight)
            .contract_packet::<T::Contracted>()
            .contract_time::<T::Sums>()
            .contract_lane::<T::Groups, T::Lanes>(LaneMode::Interleaved)
            .vector_init();
        Dm {
            tiles: input.tiles,
            value: self
                .bias
                .apply::<T::Slices, T::Groups, T::Output, T::Lanes>(value)
                .cast::<bf16, m![{ T::Lanes } # 16]>()
                .commit_trim::<T::Lanes>()
                .commit(),
        }
    }
}
