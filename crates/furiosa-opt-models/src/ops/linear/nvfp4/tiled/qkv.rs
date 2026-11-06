//! Qwen-style input partitioning with BF16 operands: four output groups and
//! 64 input slices per cluster. Paired waves fill a 32-KiB TRF operand; a final
//! direct launches use a 32-row tile; larger launches restore BF16 weights once.
use super::{Scales, block};
use crate::ops::normalization::span::Owned;
use crate::tensor::parallel::Chip;
use furiosa_opt_std::prelude::*;
type Cluster<O> = m![O / (O / 2)];
/// Heads in the layout consumed by Llama's RoPE/cache path.
pub(crate) type WaveOutput<O, R> =
    DmTensor<bf16, Chip, Cluster<O>, m![O % (O / 2) / 128 # 16, R % 128 / 8], m![R % 8, O % 128]>;

pub type Stored<H, O> = m![O / (O / 8), H / 64, O % (O / 8), H % 64];
pub type Bytes<H, O> = m![O / (O / 8), H / 64, O % (O / 8), H % 64 / 2];
pub type StoredScales<H, O> = m![O / (O / 8), H / 64, O % (O / 8), H % 64 / 16];

/// Packed checkpoint tensors, reordered once on the host into contraction slices.
#[derive(DeviceSend, crate::load::Load)]
#[load(tree)]
pub struct QkvProjection<H: AxisName, O: AxisName> {
    pub weight: HbmTensor<f4e2m1, Chip, Stored<H, O>>,
    pub weight_scale: HbmTensor<f8e4m3, Chip, StoredScales<H, O>>,
    #[load(rename = "")]
    pub scales: Scales,
}
impl<H: AxisName, O: AxisName> QkvProjection<H, O> {
    pub fn reorder<const PACK: usize, T: Copy>(values: &[T]) -> Vec<T> {
        assert!(H::SIZE.is_multiple_of(64) && O::SIZE.is_multiple_of(8));
        assert!(matches!(PACK, 1 | 2 | 16));
        assert_eq!(values.len(), H::SIZE * O::SIZE / PACK);
        let mut result = Vec::with_capacity(values.len());
        for owner in 0..8 {
            for input in (0..H::SIZE).step_by(64) {
                for output in owner * (O::SIZE / 8)..(owner + 1) * (O::SIZE / 8) {
                    let start = (output * H::SIZE + input) / PACK;
                    result.extend_from_slice(&values[start..start + 64 / PACK]);
                }
            }
        }
        result
    }
}

axes![Owner = 8];
type Cl = m![Owner / 4];
type Sl<H> = m![Owner % 4, H / 64];
/// Exactly one representation is populated, selected at the prefetch boundary.
pub(crate) struct Weight<H: AxisName, O: AxisName> {
    packed: Staged<H, O>,
    restored: Restored<H, O>,
}
type Restored<H, O> = DmTensor<bf16, Chip, Cl, Sl<H>, m![O % (O / 8), H % 64]>;
impl<H: AxisName, O: AxisName> Weight<H, O> {
    pub(crate) fn empty() -> Self {
        Self {
            packed: (DmTensor::new(), DmTensor::new()),
            restored: DmTensor::new(),
        }
    }
}
pub(crate) fn prepared<H: AxisName, O: AxisName>(
    device: &mut Device,
    packed: Staged<H, O>,
    rows: usize,
) -> Weight<H, O> {
    if rows <= 32 {
        Weight {
            packed,
            restored: DmTensor::new(),
        }
    } else {
        let restored = block::restore(device, packed.0, packed.1);
        Weight {
            packed: (DmTensor::new(), DmTensor::new()),
            restored,
        }
    }
}
pub(crate) fn prepare<H: AxisName, O: AxisName>(
    device: &mut Device,
    projection: &QkvProjection<H, O>,
    rows: usize,
) -> Weight<H, O> {
    let packed = stage(device, projection);
    prepared(device, packed, rows)
}
type Input<H, R, const W: usize> = DmTensor<bf16, Chip, Cl, Sl<H>, m![R % { W }, H % 64]>;
type Operand<H, R, const W: usize> =
    TrfTensor<bf16, Chip, Cl, Sl<H>, m![R % 8], m![R % { W } / 8, H % 64]>;
// Reduce input slices into row and output ownership, as in Qwen's QKV path.
type RowGroups<R, const W: usize> = m![R % { W } / 8];
type LocalWidth<O, R, const W: usize> = m![O / (512 / ({RowGroups<R, W>}))];
type Reduced<O, R, const W: usize> =
    m![Owner % 4, R % {W} / 8, O % (O / 8) / ({LocalWidth<O, R, W>})];
type Values<O, R, const W: usize> =
    DmTensor<bf16, Chip, Cl, Reduced<O, R, W>, m![R % 8, O % ({LocalWidth<O, R, W>})]>;

pub(crate) type Staged<H, O> = (
    DmTensor<f4e2m1, Chip, Cl, Sl<H>, m![O % (O / 8), H % 64]>,
    DmTensor<f8e4m3, Chip, Cl, Sl<H>, m![O % (O / 8), H % 64 / 16]>,
);

pub(crate) fn stage<H: AxisName, O: AxisName>(
    device: &mut Device,
    projection: &QkvProjection<H, O>,
) -> Staged<H, O> {
    let codes: DmTensor<
        f4e2m1,
        Chip,
        Cluster<O>,
        m![O % (O / 2) / (O / 8), H / 64],
        m![O % (O / 8), H % 64],
    > = projection.weight.to_dm(&mut device.tdma);
    let scales: DmTensor<
        f8e4m3,
        Chip,
        Cluster<O>,
        m![O % (O / 2) / (O / 8), H / 64],
        m![O % (O / 8), H % 64 / 16],
    > = projection.weight_scale.to_dm(&mut device.tdma);
    // SAFETY: rename the same eight output owners, preserving physical order.
    unsafe { (codes.reshape(), scales.reshape()) }
}
type StoredRows<H, R> = HbmTensor<bf16, Chip, m![H / 64, R, H % 64]>;

type PackRows<R, const N: usize> = m![R % { N } / 4];

fn stage_rows<H: AxisName, R: AxisName, const N: usize>(
    device: &mut Device,
    input: &Owned<H, R>,
) -> StoredRows<H, R> {
    // Exchange four 16-feature owners for four row owners before the HBM
    // store. Each input slice then loads contiguous rows of 64 features.
    let selected = input
        .view()
        .tile::<m![R % 256], N, m![R % 256 = {N} # 256, H % 16]>(0)
        .unpad::<m![R % 256 = { N }, H % 16]>();
    // SAFETY: rename only the selected local row prefix.
    let selected =
        unsafe { selected.reshape::<Chip, m![R / 256], m![H / 16], m![R % { N }, H % 16]>() };
    let packed: DmTensor<
        bf16,
        Chip,
        m![R / 256],
        m![H / 64, R % { N } / ({ PackRows<R, N> })],
        m![R % ({ PackRows<R, N> }), H % 64],
    > = device
        .sub
        .begin(selected)
        .fetch::<m![R % ({ PackRows<R, N> }), R % { N } / ({ PackRows<R, N> })], m![H % 16]>()
        .switch::<m![H / 64, R % { N } / ({ PackRows<R, N> })], m![R % ({ PackRows<R, N> }), H % 64 / 16]>(
            SwitchConfig::InterTranspose {
                slice1: 4,
                slice0: 1,
                time0: 1,
            },
        )
        .collect::<m![R % ({ PackRows<R, N> }), H % 64 / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();
    let mut stored = StoredRows::new();
    packed.view().to_hbm_view(
        &mut device.tdma,
        stored
            .view_mut()
            .tile::<m![R % 256], N, m![H / 64, R / 256, R % 256 = {N} #{!} 256, H % 64]>(0),
    );
    stored
}

fn load_rows<H: AxisName, R: AxisName, const N: usize, const W: usize>(
    device: &mut Device,
    stored: &StoredRows<H, R>,
    wave: usize,
) -> Input<H, R, W> {
    let mut input = Input::new();
    stored
        .view()
        .tile::<m![R / { W }], 1, m![H / 64, R / {W} = 1 # (R / {W}), R % {W}, H % 64]>(wave)
        .tile::<m![R % { W }], N, m![H / 64, R / {W} = 1 # (R / {W}), R % {W} = {N} # {W}, H % 64]>(
            0,
        )
        .to_dm_view(
            &mut device.tdma,
            input
                .view_mut()
                .tile::<m![R % { W }], N, m![R % {W} = {N} #{!} {W}, H % 64]>(0),
        );
    input
}

fn fill<H: AxisName, R: AxisName, const W: usize>(
    device: &mut Device,
    input: &Input<H, R, W>,
    groups: usize,
) -> Operand<H, R, W> {
    device
        .sub
        .begin(input.view())
        .fetch::<m![R % 8, R % { W } / 8, H % 64 / 16], m![H % 16]>()
        .collect::<m![R % 8, R % { W } / 8, H % 64 / 16], m![H % 16]>()
        .to_trf_prefix::<m![R % { W } / 8], _, _>(groups)
}

/// Broadcast a scalar into DM for a vector immediate operand, leaving the full
/// VRF available for block scales. Direct contraction applies it to each FP32
/// scaled block sum before local and inter-slice reduction, without output spills.
fn output_factor<H: AxisName, O: AxisName>(
    device: &mut Device,
    projection: &QkvProjection<H, O>,
) -> DmTensor<f32, Chip, Cl, Sl<H>, m![1 # 8]> {
    let landed: DmTensor<f32, Chip, m![2], m![1 # 256], m![1 # 8]> =
        projection.scales.output.to_dm(&mut device.tdma);
    let factor: DmTensor<f32, Chip, m![2], m![256], m![1 # 8]> = device
        .sub
        .begin(landed.view())
        .fetch::<m![1], m![1 # 8]>()
        .switch::<m![256], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![1], m![1 # 8]>()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // SAFETY: rename the same two clusters and 256 slices carrying uniform scalars.
    unsafe { factor.reshape() }
}

fn project<H: AxisName, O: AxisName, R: AxisName, const W: usize>(
    device: &mut Device,
    projection: &QkvProjection<H, O>,
    weight: &Weight<H, O>,
    input: &Operand<H, R, W>,
) -> Values<O, R, W> {
    let factor = output_factor(device, projection);
    let scales: VrfTensor<f32, Chip, Cl, Sl<H>, m![O % (O / 8), H % 64 / 16]> = device
        .sub
        .begin(weight.packed.1.view())
        .fetch::<m![O % (O / 8) / 2], m![O % 2, H % 64 / 16]>()
        .fetch_cast::<f32>()
        .collect::<m![O % (O / 8) / 2], m![O % 2, H % 64 / 16]>()
        .to_vrf();
    device
        .main
        .begin(weight.packed.0.view())
        // Four row groups reuse each weight block and fill the eight FP32
        // reduction accumulators. Direct launches use exactly one 32-row tile.
        // Each single tile computes all 32 rows; the consumer limits valid rows.
        .fetch::<m![O % (O / 8), R % { W } / 32], m![H % 64]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![O % (O / 8), R % { W } / 32, H % 64 / 16], m![H % 16]>()
        .contract_outer::<m![O % (O / 8), R % { W } / 32, H % 64 / 16, R % 32 / 8], m![H % 16], _, _, _>(input)
        .contract_packet::<m![1]>()
        .contract_time::<m![O % (O / 8), R % { W } / 32, H % 64 / 16, R % 32 / 8]>()
        .contract_lane::<m![O % (O / 8), R % { W } / 32, H % 64 / 16, R % 32 / 8], m![R % 8]>(
            LaneMode::Interleaved,
        )
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![O % (O / 8), R % { W } / 32, H % 64 / 16, R % 32 / 8, R % 8 / 4], m![R % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scales)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &factor)
        .vector_intra_slice_reduce::<H, m![O % (O / 8), R % { W } / 8, R % 8 / 4], m![R % 4]>(
            IntraSliceReduceOpF32::Add,
        )
        .vector_widen_concat::<m![O % (O / 8), R % { W } / 8], m![R % 8]>()
        .vector_inter_slice_reduce::<Reduced<O, R, W>, m![O % ({LocalWidth<O, R, W>})]>(
            InterSliceReduceOpF32::Add,
        )
        .vector_final()
        .cast::<bf16, m![R % 8 # 16]>()
        .transpose::<m![O % ({LocalWidth<O, R, W>}) / 4, R % 8], m![O % 4 # 16]>()
        .commit_trim::<m![O % 4]>()
        .commit()
}

fn project_restored<H: AxisName, O: AxisName, R: AxisName, const W: usize>(
    device: &mut Device,
    projection: &QkvProjection<H, O>,
    weight: &Restored<H, O>,
    input: &Operand<H, R, W>,
    groups: usize,
) -> Values<O, R, W> {
    let factor = block::scalar::<Cl, Reduced<O, R, W>>(device, &projection.scales.output);
    let mut output = Values::new();
    let products = device
        .main
        .begin(weight.view())
        .fetch::<m![O % (O / 8), H % 64 / 16], m![H % 16]>()
        .collect::<m![O % (O / 8), H % 64 / 16], m![H % 16]>()
        .contract_outer::<m![O % (O / 8), H % 64 / 32, R % { W } / 8], m![H % 32], _, _, _>(input)
        .contract_packet::<m![1]>()
        .contract_time::<m![O % (O / 8), R % { W } / 8]>()
        .contract_lane::<m![O % (O / 8), R % { W } / 8], m![R % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<Reduced<O, R, W>, m![O % ({LocalWidth<O, R, W>})]>(
            InterSliceReduceOpF32::Add,
        )
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![O % ({LocalWidth<O, R, W>}), R % 8 / 4], m![R % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &factor)
        .vector_widen_concat::<m![O % ({LocalWidth<O, R, W>})], m![R % 8]>()
        .vector_final()
        .cast::<bf16, m![R % 8 # 16]>()
        .transpose::<m![O % ({LocalWidth<O, R, W>}) / 4, R % 8], m![O % 4 # 16]>()
        .commit_trim::<m![O % 4]>();
    if const { W == 128 } {
        products.commit_prefix::<m![R % { W } / 8], _, _>(output.view_mut(), groups);
    } else {
        products.commit_view(output.view_mut());
    }
    output
}

type WaveCount<R, const W: usize> = m![R % { W } / 128];

fn land<O: AxisName, R: AxisName, const W: usize, const HALF: usize>(
    device: &mut Device,
    values: &Values<O, R, W>,
) -> WaveOutput<O, R> {
    // SAFETY: give the eight output owners their width coordinates without moving data.
    let values = unsafe {
        values.view().reshape::<Chip, Cluster<O>,
            m![O % (O / 2) / (O / 8), R % {W} / 8, O % (O / 8) / ({LocalWidth<O, R, W>})],
            m![R % 8, O % ({LocalWidth<O, R, W>})]>()
    };
    let mut output = WaveOutput::new();
    values.slice_tile::<m![R % {W} / 128], 1,
        m![O % (O / 2) / (O / 8), R % {W} / 128 = 1 # ({WaveCount<R, W>}), R % 128 / 8, O % (O / 8) / ({LocalWidth<O, R, W>})]>(HALF)
        .to_dm_view(&mut device.tdma, output.view_mut());
    output
}

pub(crate) fn consume<H: AxisName, Q: AxisName, K: AxisName, R: AxisName, F>(
    device: &mut Device,
    projections: (
        &QkvProjection<H, Q>,
        &QkvProjection<H, K>,
        &QkvProjection<H, K>,
    ),
    weights: (&Weight<H, Q>, &Weight<H, K>, &Weight<H, K>),
    input: &Owned<H, R>,
    rows: usize,
    mut consume: F,
) where
    F: FnMut(&mut Device, WaveOutput<Q, R>, WaveOutput<K, R>, WaveOutput<K, R>, usize),
{
    const {
        assert!(H::SIZE == 4096 && Q::SIZE == 4096 && K::SIZE == 1024 && R::SIZE == 512);
        assert!(256 * 64 * size_of::<bf16>() == 32768);
    }
    // Stage normalized rows once; each pair's activation is shared by all three projections.
    let stored = if rows <= 8 {
        stage_rows::<H, R, 8>(device, input)
    } else if rows <= 16 {
        stage_rows::<H, R, 16>(device, input)
    } else if rows <= 32 {
        stage_rows::<H, R, 32>(device, input)
    } else if rows <= 64 {
        stage_rows::<H, R, 64>(device, input)
    } else if rows <= 128 {
        stage_rows::<H, R, 128>(device, input)
    } else {
        stage_rows::<H, R, 256>(device, input)
    };
    if rows <= 32 {
        single::<H, Q, K, R, F>(device, projections, weights, &stored, 0, rows, &mut consume);
        return;
    }
    let waves = rows.div_ceil(128);
    for pair in 0..waves / 2 {
        run_restored::<H, Q, K, R, 256, F>(
            device,
            projections,
            weights,
            &stored,
            pair,
            (rows - pair * 256).min(256),
            &mut consume,
        );
    }
    if waves % 2 == 1 {
        let wave = waves - 1;
        run_restored::<H, Q, K, R, 128, F>(
            device,
            projections,
            weights,
            &stored,
            wave,
            rows - wave * 128,
            &mut consume,
        );
    }
}

// The summed reduction cannot lower a runtime row prefix on this compiler.
// A direct launch computes one complete 32-row tile. Input transfers still use
// live prefixes, and downstream RoPE/cache writes consume only live rows.
fn single<H: AxisName, Q: AxisName, K: AxisName, R: AxisName, F>(
    device: &mut Device,
    projections: (
        &QkvProjection<H, Q>,
        &QkvProjection<H, K>,
        &QkvProjection<H, K>,
    ),
    weights: (&Weight<H, Q>, &Weight<H, K>, &Weight<H, K>),
    stored: &StoredRows<H, R>,
    wave: usize,
    live: usize,
    consume: &mut F,
) where
    F: FnMut(&mut Device, WaveOutput<Q, R>, WaveOutput<K, R>, WaveOutput<K, R>, usize),
{
    let mut output = (WaveOutput::new(), WaveOutput::new(), WaveOutput::new());
    single_tile::<H, Q, K, R, 0>(
        device,
        projections,
        weights,
        stored,
        wave,
        live,
        &mut output,
    );
    consume(device, output.0, output.1, output.2, wave);
}

fn single_tile<H: AxisName, Q: AxisName, K: AxisName, R: AxisName, const QUARTER: usize>(
    device: &mut Device,
    projections: (
        &QkvProjection<H, Q>,
        &QkvProjection<H, K>,
        &QkvProjection<H, K>,
    ),
    weights: (&Weight<H, Q>, &Weight<H, K>, &Weight<H, K>),
    stored: &StoredRows<H, R>,
    wave: usize,
    live: usize,
    output: &mut (WaveOutput<Q, R>, WaveOutput<K, R>, WaveOutput<K, R>),
) {
    let live = (live - QUARTER * 32).min(32);
    let tile = wave * 4 + QUARTER;
    let input = if live <= 8 {
        load_rows::<H, R, 8, 32>(device, stored, tile)
    } else if live <= 16 {
        load_rows::<H, R, 16, 32>(device, stored, tile)
    } else {
        load_rows::<H, R, 32, 32>(device, stored, tile)
    };
    let groups = live.div_ceil(8);
    let lanes = fill(device, &input, groups);
    let q = project(device, projections.0, weights.0, &lanes);
    let k = project(device, projections.1, weights.1, &lanes);
    let v = project(device, projections.2, weights.2, &lanes);
    land_tile::<Q, R, QUARTER>(device, &q, &mut output.0);
    land_tile::<K, R, QUARTER>(device, &k, &mut output.1);
    land_tile::<K, R, QUARTER>(device, &v, &mut output.2);
}

fn land_tile<O: AxisName, R: AxisName, const QUARTER: usize>(
    device: &mut Device,
    values: &Values<O, R, 32>,
    output: &mut WaveOutput<O, R>,
) {
    // SAFETY: rename the same eight output owners without moving data.
    let values = unsafe {
        values.view().reshape::<Chip, Cluster<O>,
            m![O % (O / 2) / (O / 8), R % 32 / 8, O % (O / 8) / ({LocalWidth<O, R, 32>})],
            m![R % 8, O % ({LocalWidth<O, R, 32>})]>()
    };
    values.to_dm_view(&mut device.tdma,
        output.view_mut().slice_tile::<m![R % 128 / 32], 1,
            m![O % (O / 2) / 128 # 16, R % 128 / 32 = 1 #{!} 4, R % 32 / 8]>(QUARTER));
}

fn run_restored<H: AxisName, Q: AxisName, K: AxisName, R: AxisName, const W: usize, F>(
    device: &mut Device,
    projections: (
        &QkvProjection<H, Q>,
        &QkvProjection<H, K>,
        &QkvProjection<H, K>,
    ),
    weights: (&Weight<H, Q>, &Weight<H, K>, &Weight<H, K>),
    stored: &StoredRows<H, R>,
    wave: usize,
    live: usize,
    consume: &mut F,
) where
    F: FnMut(&mut Device, WaveOutput<Q, R>, WaveOutput<K, R>, WaveOutput<K, R>, usize),
{
    const {
        assert!(W == 128 || W == 256);
    }
    let groups = live.div_ceil(8);
    let input = if live <= 8 {
        load_rows::<H, R, 8, W>(device, stored, wave)
    } else if live <= 16 {
        load_rows::<H, R, 16, W>(device, stored, wave)
    } else if live <= 32 {
        load_rows::<H, R, 32, W>(device, stored, wave)
    } else if live <= 64 {
        load_rows::<H, R, 64, W>(device, stored, wave)
    } else if const { W == 128 } {
        load_rows::<H, R, 128, W>(device, stored, wave)
    } else {
        load_rows::<H, R, 256, W>(device, stored, wave)
    };
    let lanes = fill(device, &input, groups);
    let q = project_restored(device, projections.0, &weights.0.restored, &lanes, groups);
    let k = project_restored(device, projections.1, &weights.1.restored, &lanes, groups);
    let v = project_restored(device, projections.2, &weights.2.restored, &lanes, groups);
    let qo = land::<Q, R, W, 0>(device, &q);
    let ko = land::<K, R, W, 0>(device, &k);
    let vo = land::<K, R, W, 0>(device, &v);
    consume(device, qo, ko, vo, wave * (W / 128));
    if const { W == 256 } {
        let qo = land::<Q, R, W, 1>(device, &q);
        let ko = land::<K, R, W, 1>(device, &k);
        let vo = land::<K, R, W, 1>(device, &v);
        consume(device, qo, ko, vo, wave * 2 + 1);
    }
}

#[cfg(feature = "kernel-diagnostics")]
pub type Output<O, R> = HbmTensor<bf16, Chip, m![R, O]>;

#[cfg(feature = "kernel-diagnostics")]
pub(crate) fn store_wave<O: AxisName, R: AxisName>(
    device: &mut Device,
    values: &WaveOutput<O, R>,
    output: &mut Output<O, R>,
    wave: usize,
) {
    values.view().to_hbm_view(
        &mut device.tdma,
        output
            .view_mut()
            .tile::<m![R / 128], 1, m![R / 128 = 1 #{!} (R / 128), R % 128, O]>(wave),
    );
}

crate::ops::linear::nvfp4::tiled::loading!(QkvProjection);
