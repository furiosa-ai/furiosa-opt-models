//! Projection contractions that broadcast input rows across the weight partitions.

use furiosa_opt_std::prelude::*;

use super::{Mapping, Scales};
use crate::Module;
use crate::ops::layout::Dm;
use crate::ops::linear::Affine;

pub trait Rows: Mapping {
    type Result: M;
    type Landing: M;
    type InputPacket: M;
    type Gather: M;
    const BROADCAST: SwitchConfig;
}

impl<'d, 'x, 'w, 's, B: M, T: Rows>
    Module<(
        &'d mut Device,
        Dm<T, &'x DmTensor<f8e4m3, B, T::Cluster, T::Landing, T::Input>>,
    )>
    for Affine<
        &'w TrfTensor<f8e4m3, B, T::Cluster, T::Slices, T::Lanes, T::Weight>,
        &'s Scales<T::Cluster, T::Reduced, B>,
    >
{
    type Output = Dm<T, DmTensor<bf16, B, T::Cluster, T::Reduced, T::Result>>;

    fn forward(
        &self,
        (device, input): (
            &'d mut Device,
            Dm<T, &'x DmTensor<f8e4m3, B, T::Cluster, T::Landing, T::Input>>,
        ),
    ) -> Self::Output {
        let value = device
            .main
            .begin(input.value.view())
            .fetch::<T::Feed, T::InputPacket>()
            .switch::<T::Slices, T::Gather>(T::BROADCAST)
            .collect::<T::Flits, T::Flit>()
            .contract_outer::<T::Steps, T::Packet, _, _, _>(self.weight)
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

impl<'d, 'x, 'w, 's, B: M, T: Rows, E: M>
    Module<(
        &'d mut Device,
        Dm<
            T,
            (
                &'x DmTensor<f8e4m3, B, T::Cluster, T::Landing, T::Input>,
                DmTensorViewMut<'x, bf16, B, T::Cluster, T::Reduced, E>,
            ),
        >,
    )>
    for Affine<
        &'w TrfTensor<f8e4m3, B, T::Cluster, T::Slices, T::Lanes, T::Weight>,
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
                    &'x DmTensor<f8e4m3, B, T::Cluster, T::Landing, T::Input>,
                    DmTensorViewMut<'x, bf16, B, T::Cluster, T::Reduced, E>,
                ),
            >,
        ),
    ) {
        let value = device
            .main
            .begin(input.value.0.view())
            .fetch::<T::Feed, T::InputPacket>()
            .switch::<T::Slices, T::Gather>(T::BROADCAST)
            .collect::<T::Flits, T::Flit>()
            .contract_outer::<T::Steps, T::Packet, _, _, _>(self.weight)
            .contract_packet::<T::Contracted>()
            .contract_time::<T::Sums>()
            .contract_lane::<T::Groups, T::Lanes>(LaneMode::Interleaved)
            .vector_init();
        self.bias
            .apply::<T::Slices, T::Groups, T::Output, T::Lanes>(value)
            .cast::<bf16, m![{ T::Lanes } # 16]>()
            .commit_trim::<T::Lanes>()
            .commit_view(input.value.1);
    }
}
