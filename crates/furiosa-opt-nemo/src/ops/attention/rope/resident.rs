use furiosa_opt_std::prelude::*;

use super::Rotary;
use crate::Module;
use crate::compose::Space;
use crate::ops::layout::Dm;

/// Physical mappings for feature-pair rotation with coefficients retained in TRF.
pub trait Mapping: Space {
    type Cluster: M;
    type Slices: M;
    type Input: M;
    type Output: M;
    type Coefficients: M;
    type Lanes: M;
    type Feed: M;
    type Packet: M;
    type Flits: M;
    type Flit: M;
    type Steps: M;
    type Sums: M;
    type Groups: M;
    type Result: M;
}

macro_rules! rotate {
    ($device:ident, $input:ident) => {
        $device
            .main
            .begin($input.value.0)
            .fetch::<T::Feed, T::Packet>()
            .collect::<T::Flits, T::Flit>()
            .contract_outer::<T::Steps, T::Packet, _, _, _>($input.value.1)
            .contract_packet::<T::Packet>()
            .contract_time::<T::Sums>()
            .contract_lane::<T::Groups, T::Result>(LaneMode::Sequential)
            .commit_trim::<T::Result>()
            .commit_cast::<bf16>()
    };
}

impl<'d, 'x, Head: AxisName, B: M, T: Mapping>
    Module<(
        &'d mut Device,
        Dm<
            T,
            (
                DmTensorView<'x, bf16, B, T::Cluster, T::Slices, T::Input>,
                &'x TrfTensor<bf16, B, T::Cluster, T::Slices, T::Lanes, T::Coefficients>,
            ),
        >,
    )> for Rotary<Head, Head, bf16, B>
{
    type Output = Dm<T, DmTensor<bf16, B, T::Cluster, T::Slices, T::Output>>;

    fn forward(
        &self,
        (device, input): (
            &'d mut Device,
            Dm<
                T,
                (
                    DmTensorView<'x, bf16, B, T::Cluster, T::Slices, T::Input>,
                    &'x TrfTensor<bf16, B, T::Cluster, T::Slices, T::Lanes, T::Coefficients>,
                ),
            >,
        ),
    ) -> Self::Output {
        Dm {
            value: rotate!(device, input).commit(),
            tiles: input.tiles,
        }
    }
}

impl<'d, 'x, Head: AxisName, B: M, T: Mapping>
    Module<(
        &'d mut Device,
        Dm<
            T,
            (
                DmTensorView<'x, bf16, B, T::Cluster, T::Slices, T::Input>,
                &'x TrfTensor<bf16, B, T::Cluster, T::Slices, T::Lanes, T::Coefficients>,
                DmTensorViewMut<'x, bf16, B, T::Cluster, T::Slices, T::Output>,
            ),
        >,
    )> for Rotary<Head, Head, bf16, B>
{
    type Output = ();

    fn forward(
        &self,
        (device, input): (
            &'d mut Device,
            Dm<
                T,
                (
                    DmTensorView<'x, bf16, B, T::Cluster, T::Slices, T::Input>,
                    &'x TrfTensor<bf16, B, T::Cluster, T::Slices, T::Lanes, T::Coefficients>,
                    DmTensorViewMut<'x, bf16, B, T::Cluster, T::Slices, T::Output>,
                ),
            >,
        ),
    ) {
        rotate!(device, input).commit_view(input.value.2);
    }
}
