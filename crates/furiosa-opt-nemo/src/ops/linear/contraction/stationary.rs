//! Projection contractions with activation rows retained in registers.

use furiosa_opt_std::prelude::*;

use super::{Mapping, Scales};
use crate::Module;
use crate::ops::layout::Dm;
use crate::ops::linear::Affine;

impl<'d, 'x, 'w, 's, B: M, T: Mapping>
    Module<(
        &'d mut Device,
        Dm<T, &'x TrfTensor<f8e4m3, B, T::Cluster, T::Slices, T::Lanes, T::Input>>,
    )>
    for Affine<
        &'w DmTensor<f8e4m3, B, T::Cluster, T::Slices, T::Weight>,
        &'s Scales<T::Cluster, T::Reduced, B>,
    >
{
    type Output = Dm<T, DmTensor<bf16, B, T::Cluster, T::Reduced, m![{ T::Output }, { T::Lanes }]>>;

    fn forward(
        &self,
        (device, input): (
            &'d mut Device,
            Dm<T, &'x TrfTensor<f8e4m3, B, T::Cluster, T::Slices, T::Lanes, T::Input>>,
        ),
    ) -> Self::Output {
        let value = device
            .main
            .begin(self.weight.view())
            .fetch::<T::Feed, T::Packet>()
            .collect::<T::Flits, T::Flit>()
            .contract_outer::<T::Steps, T::Packet, _, _, _>(input.value)
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
