//! Broadcast scalar packets: one value repeated over a vector packet, and the main-engine passes
//! that compute on it.

use crate::ops::layout::{BroadcastClusters as Cluster, BroadcastSlices as Slices, axes::Chip};
use furiosa_opt_std::prelude::*;

/// One I32 row index or position, a copy on each chip of `Chips`.
pub type Value<Chips = Chip> = HbmTensor<i32, Chips, m![1]>;

/// Scalar repeated over the eight cells of a vector packet in both clusters and 256 slices.
pub type Packet<D = i32, Chips = Chip> = DmTensor<D, Chips, Cluster, Slices, m![1 # 8]>;

/// A packet in the main vector engine at stage `S`; it enters as I32 at the tag stage.
pub type Stream<'l, S = stage::Tag, D = i32, Stash = Fresh, Chips = Chip> = VectorTensor<
    'l,
    { Tu::Main },
    S,
    D,
    Chips,
    Cluster,
    Slices,
    m![1],
    m![1 # 8],
    Stash,
    { stage::VeOrder::IntraFirst },
>;

/// Stage a scalar as an eight-cell DM packet for vector arithmetic.
pub(in crate::ops) fn packet<Chips: M>(
    device: &mut Device,
    input: &Value<Chips>,
) -> Packet<i32, Chips> {
    let packed: HbmTensor<i32, Chips, m![1 # 8]> = input.to_hbm(&mut device.tdma);
    packed.to_dm(&mut device.tdma)
}

/// Load a broadcast packet into vector registers.
pub(in crate::ops) fn register<D: VeScalar, Chips: M, Cl: M, S: M>(
    device: &mut Device,
    input: &DmTensor<D, Chips, Cl, S, m![1 # 8]>,
) -> VrfTensor<D, Chips, Cl, S, m![1 # 8]> {
    device
        .sub
        .begin(input.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}

/// Apply `stages` to a packet in the main vector engine and commit their result.
pub(in crate::ops) fn map<S: stage::Stage, D: VeScalar, Stash: StashState, Chips: M>(
    device: &mut Device,
    input: &Packet<i32, Chips>,
    stages: impl FnOnce(Stream<'_, stage::Tag, i32, Fresh, Chips>) -> Stream<'_, S, D, Stash, Chips>,
) -> Packet<D, Chips> {
    stages(
        device
            .main
            .begin(input.view())
            .fetch::<m![1], m![1 # 8]>()
            .collect::<m![1], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero),
    )
    .vector_final()
    .commit_trim::<m![1 # 8]>()
    .commit()
}

/// Convert a row index to a byte offset for rows shaped S with elements D.
pub(in crate::ops) fn offset<D: Scalar, S: M, O: M, Chips: M>(
    device: &mut Device,
    input: &Value<Chips>,
) -> HbmTensor<i32, Chips, O> {
    let input = packet(device, input);
    map(device, &input, |index| {
        index.vector_fxp(
            FxpBinaryOp::MulInt,
            const { (S::SIZE * D::BITS / 8) as i32 },
        )
    })
    .to_hbm(&mut device.tdma)
}
