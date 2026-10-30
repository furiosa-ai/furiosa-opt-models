//! Broadcast scalar packets: one value repeated over a vector packet, and the main-engine passes
//! that compute on it.

use crate::load::{Context, Load, LoadError};
use crate::tensor::{
    mapping::{BroadcastClusters as Cluster, BroadcastSlices as Slices},
    parallel::Chip,
};
use furiosa_opt_std::prelude::*;

/// One I32 row index or position, a copy on each chip of `Chips`.
pub type Value<Chips = Chip> = HbmTensor<i32, Chips, m![1]>;

/// An FP32 fixed when the kernel compiles, as its bits: a score multiplier or a norm epsilon.
#[derive(Clone, Copy, Debug, DeviceSend, Load)]
pub struct F32<const BITS: usize>;

impl<const BITS: usize> From<F32<BITS>> for f32 {
    fn from(_: F32<BITS>) -> Self {
        f32::from_bits(
            const {
                assert!(BITS <= u32::MAX as usize);
                BITS as u32
            },
        )
    }
}

impl<const BITS: usize> F32<BITS> {
    /// The value in every element on every chip of `C`, for a leaf a load places, as no
    /// checkpoint stores it.
    pub fn host<C: M + 'static, S: M + 'static>() -> HostTensor<f32, m![{ C }, { S }]> {
        HostTensor::from_vec(vec![f32::from(Self); C::SIZE * S::SIZE])
    }
}

/// A placed value of a constant leaf loads as the constant itself, which the kernel compiles in.
impl<const BITS: usize, S: M> Load<HostTensor<f32, S>> for F32<BITS> {
    fn load(_: &mut Context<'_>, _: HostTensor<f32, S>) -> Result<Self, LoadError> {
        Ok(Self)
    }
}

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

/// Copy the one I32 element a view selects into a scalar of its own.
pub(crate) fn read<Chips: M, E: M>(
    device: &mut Device,
    input: HbmTensorView<'_, i32, Chips, E>,
) -> Value<Chips> {
    let mut output = Value::new();
    input.to_hbm_view(&mut device.tdma, output.view_mut());
    output
}

/// Stage a scalar as an eight-cell DM packet for vector arithmetic.
pub(crate) fn packet<Chips: M>(device: &mut Device, input: &Value<Chips>) -> Packet<i32, Chips> {
    let packed: HbmTensor<i32, Chips, m![1 # 8]> = input.to_hbm(&mut device.tdma);
    packed.to_dm(&mut device.tdma)
}

/// Load a broadcast packet into vector registers.
pub(crate) fn register<D: VeScalar, Chips: M, Cluster: M, S: M>(
    device: &mut Device,
    input: &DmTensor<D, Chips, Cluster, S, m![1 # 8]>,
) -> VrfTensor<D, Chips, Cluster, S, m![1 # 8]> {
    device
        .sub
        .begin(input.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}

/// Apply `stages` to a packet in the main vector engine and commit their result.
pub(crate) fn map<S: stage::Stage, D: VeScalar, Stash: StashState, Chips: M>(
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
