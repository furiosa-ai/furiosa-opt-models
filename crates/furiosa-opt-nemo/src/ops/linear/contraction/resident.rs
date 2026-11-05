//! BF16 projection partials with activation rows retained in registers.

use furiosa_opt_std::prelude::*;

use crate::Module;
use crate::compose::Space;
use crate::ops::layout::Dm;
use crate::ops::linear::Affine;

/// A projection tile's contraction and transpose coordinates, with no cross-slice reduction.
pub trait Mapping: Space {
    type Cluster: M;
    type Slices: M;
    type Lanes: M;
    type Input: M;
    type Weight: M;
    type Feed: M;
    type Packet: M;
    type Flits: M;
    type Flit: M;
    type Steps: M;
    type Contracted: M;
    type Sums: M;
    type Groups: M;
    type Transpose: M;
    type Columns: M;
    type Output: M;
}

impl<'d, 'x, 'w, B: M, T: Mapping>
    Module<(
        &'d mut Device,
        Dm<T, &'x TrfTensor<bf16, B, T::Cluster, T::Slices, T::Lanes, T::Input>>,
    )> for Affine<&'w DmTensor<bf16, B, T::Cluster, T::Slices, T::Weight>, ()>
{
    type Output = Dm<T, DmTensor<f32, B, T::Cluster, T::Slices, T::Output>>;

    fn forward(
        &self,
        (device, input): (
            &'d mut Device,
            Dm<T, &'x TrfTensor<bf16, B, T::Cluster, T::Slices, T::Lanes, T::Input>>,
        ),
    ) -> Self::Output {
        Dm {
            tiles: input.tiles,
            value: device
                .main
                .begin(self.weight.view())
                .fetch::<T::Feed, T::Packet>()
                .collect::<T::Flits, T::Flit>()
                .contract_outer::<T::Steps, T::Packet, _, _, _>(input.value)
                .contract_packet::<T::Contracted>()
                .contract_time::<T::Sums>()
                .contract_lane::<T::Groups, T::Lanes>(LaneMode::Interleaved)
                .transpose::<T::Transpose, T::Columns>()
                .commit_trim::<T::Contracted>()
                .commit(),
        }
    }
}
