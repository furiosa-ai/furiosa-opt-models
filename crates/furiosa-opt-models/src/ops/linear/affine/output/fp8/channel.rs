//! Token-dynamic E4M3 output projection. Input halves stay in their clusters; only their FP32
//! token-scale maxima and the projection's partial sums cross clusters.

use super::*;
use crate::ops::vector;
use crate::tensor::form::fp8::Channel;

pub type Stored<Hidden, QWidth> = super::Stored<Hidden, QWidth, Channel, bf16>;

/// Select complete eight-row owners statically: the switch fixes their position inside time,
/// where a runtime prefix is unsupported. Physical owner coordinates and weights stay unchanged.
pub(super) fn contract<H: AxisName, Q: AxisName, R: AxisName, const G: usize>(
    device: &mut Device,
    weight: &Resident<H, Q>,
    input: &Source<H, Q, R>,
    output: &mut tiles::Partial<H, Q, R>,
) {
    device.main.begin(input.view()
        .slice_tile::<m![R / (R / 4)], G,
            m![H / 1024, R / (R / 4) = {G} # 4, Q % 2048 / 128]>(0)
        .tile::<m![R % (R / 4) / 8], 1,
        m![R % (R / 4) / 8 = 1 # (R / 32), R % 8, Q % 128]>(0))
        .fetch::<m![R % 8, Q % 128 / 64], m![Q % 64]>()
        .switch::<WeightSlices<H, Q>, m![R % 8, Q % 128 / 64, R / (R / 4) = {G}]>(
            SwitchConfig::CustomBroadcast { ring_size: 64 })
        .collect::<m![R % 8, Q % 128 / 64, R / (R / 4) = {G}, Q % 64 / 32], m![Q % 32]>()
        .contract_outer::<m![R % 8, Q % 128 / 64, R / (R / 4) = {G}, H % 256 / 8],
            m![Q % 64], _, _, _>(weight)
        .contract_packet::<m![1]>()
        .contract_time::<m![R % 8, R / (R / 4) = {G}, H % 256 / 8]>()
        .contract_lane::<m![R % 8, R / (R / 4) = {G}, H % 256 / 8], m![H % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<m![H / 16], m![R % 8, R / (R / 4) = {G}, H % 16 / 8]>(
            InterSliceReduceOpF32::Add)
        .vector_final().cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit_view(output.view_mut().tile::<m![R % (R / 4) / 8], 1,
            m![R % (R / 4) / 8 = 1 #{!} (R / 32), R / (R / 4), R % 8, H % 16]>(0)
            .tile::<m![R / (R / 4)], G,
                m![R % (R / 4) / 8 = 1 #{!} (R / 32), R / (R / 4) = {G} #{!} 4, R % 8, H % 16]>(0));
}

pub(crate) fn quantize<Hidden: AxisName, QWidth: AxisName, Capacity: AxisName>(
    device: &mut Device,
    input: &InDm<Hidden, QWidth, Capacity>,
    rows: usize,
) -> (
    Source<Hidden, QWidth, Capacity>,
    ReplicatedScales<Hidden, QWidth, Capacity, f32>,
) {
    let mut local = ReplicatedScales::<Hidden, QWidth, Capacity, f32>::new();
    quantize::Quarter::<Hidden, QWidth, Capacity>::scales_f32(&mut device.main, input.view())
        .commit_prefix::<m![Capacity % (Capacity / 4) / 4], _, _>(
            local.view_mut(),
            rows.div_ceil(32) * 2,
        );
    // The other cluster owns the other 2048 input features of these same tokens.
    let scales = vector::cluster_max(device, &local, rows.div_ceil(32) * 8);
    // SAFETY: each token scale has been replicated to all feature slices of its row owner.
    let broadcast = unsafe {
        scales.view().reshape::<Chip, Cluster<QWidth>,
        RowWise<Hidden, QWidth, Capacity>, m![Capacity % (Capacity / 4)]>()
    };
    let factor: VrfTensor<
        f32,
        Chip,
        Cluster<QWidth>,
        RowWise<Hidden, QWidth, Capacity>,
        m![Capacity % (Capacity / 4)],
    > = device
        .sub
        .begin(broadcast)
        .fetch::<m![Capacity % (Capacity / 4) / 8], m![Capacity % 8]>()
        .collect::<m![Capacity % (Capacity / 4) / 8], m![Capacity % 8]>()
        .to_vrf();
    let mut codes = Source::<Hidden, QWidth, Capacity>::new();
    quantize::Quarter::<Hidden, QWidth, Capacity>::rows(&mut device.main, input.view(), &factor)
        .commit_prefix::<m![Capacity % (Capacity / 4)], _, _>(
            codes.view_mut(),
            rows.div_ceil(32) * 8,
        );
    (codes, scales)
}
