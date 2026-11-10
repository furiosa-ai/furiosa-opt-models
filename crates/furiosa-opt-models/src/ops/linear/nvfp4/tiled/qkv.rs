//! Qwen-style input partitioning with BF16 operands: four output groups and
//! 64 input slices per cluster. Paired waves fill a 32-KiB TRF operand, with
//! the next pair or final single wave prefetched into a separate TRF half.
//! Direct launches use 16/32-row tiles; larger launches restore BF16 weights once.
use super::prefetch::{Points, Stage};
use super::{Scales, block};
use crate::ops::normalization::span::Owned;
use crate::tensor::parallel::Chip;
use furiosa_opt_std::prelude::*;
type Cluster<O> = m![O / (O / 2)];
/// Heads in the layout consumed by Llama's RoPE/cache path.
/// Only the live prefix is valid within this 128-row allocation.
pub(crate) type WaveOutput<O, R> =
    DmTensor<bf16, Chip, Cluster<O>, m![O % (O / 2) / 128 # 16, R % 128 / 8], m![R % 8, O % 128]>;

#[cfg(any(test, feature = "kernel-diagnostics"))]
pub type Output<O, R> = HbmTensor<bf16, Chip, m![R, O]>;

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

const DIRECT_MAX_ROWS: usize = 32;

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

type StoredRows<H, R> = HbmTensor<bf16, Chip, m![H / 64, R, H % 64]>;

type PackRows<R, const N: usize> = m![R % { N } / 4];

type DirectTiles<R, const W: usize> = m![R % 128 / { W }];
type WaveCount<R, const W: usize> = m![R % { W } / 128];

loading!(QkvProjection);

/// Select the representation for this span. Consumption must use the same row
/// classification (direct through DIRECT_MAX_ROWS, restored above it).
pub(crate) fn prepared<H: AxisName, O: AxisName>(
    device: &mut Device,
    packed: Staged<H, O>,
    rows: usize,
) -> Weight<H, O> {
    if rows <= DIRECT_MAX_ROWS {
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
    let products = device
        .main
        .begin(weight.packed.0.view())
        // Two or four row groups reuse each weight block. Static
        // 16/32-row geometry uses four or eight reduction slots.
        .fetch::<m![O % (O / 8)], m![H % 64]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![O % (O / 8), H % 64 / 16], m![H % 16]>()
        .contract_outer::<m![O % (O / 8), H % 64 / 16, R % { W } / 8], m![H % 16], _, _, _>(input)
        .contract_packet::<m![1]>()
        .contract_time::<m![O % (O / 8), H % 64 / 16, R % { W } / 8]>()
        .contract_lane::<m![O % (O / 8), H % 64 / 16, R % { W } / 8], m![R % 8]>(
            LaneMode::Interleaved,
        )
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![O % (O / 8), H % 64 / 16, R % { W } / 8, R % 8 / 4], m![R % 4]>()
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
        .cast::<bf16, m![R % 8 # 16]>();
    products
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

/// Consume 1..=512 live rows into complete 128-row scratch tiles.
/// Each weight must come from its corresponding projection and be prepared with
/// the same direct/restored row classification. Padding is not a valid result;
/// attention must mask padded keys and zero padded values before consuming it.
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
    } else if rows <= DIRECT_MAX_ROWS {
        stage_rows::<H, R, 32>(device, input)
    } else if rows <= 64 {
        stage_rows::<H, R, 64>(device, input)
    } else if rows <= 128 {
        stage_rows::<H, R, 128>(device, input)
    } else {
        stage_rows::<H, R, 256>(device, input)
    };
    if rows <= DIRECT_MAX_ROWS {
        single::<H, Q, K, R, F>(device, projections, weights, &stored, 0, rows, &mut consume);
        return;
    }
    // A span has at most two pairs. Keep both operands in the same scheduling
    // unit so the next DMA and sub-context TRF fill can overlap the first Q.
    // A partial second wave still uses the paired contraction geometry.
    if rows > 384 {
        two_restored::<H, Q, K, R, 256, 256, F>(
            device,
            projections,
            weights,
            &stored,
            rows,
            &mut consume,
        );
    } else if rows > 320 {
        two_restored::<H, Q, K, R, 128, 128, F>(
            device,
            projections,
            weights,
            &stored,
            rows,
            &mut consume,
        );
    } else if rows > 288 {
        two_restored::<H, Q, K, R, 128, 64, F>(
            device,
            projections,
            weights,
            &stored,
            rows,
            &mut consume,
        );
    } else if rows > 272 {
        two_restored::<H, Q, K, R, 128, 32, F>(
            device,
            projections,
            weights,
            &stored,
            rows,
            &mut consume,
        );
    } else if rows > 264 {
        two_restored::<H, Q, K, R, 128, 16, F>(
            device,
            projections,
            weights,
            &stored,
            rows,
            &mut consume,
        );
    } else if rows > 256 {
        two_restored::<H, Q, K, R, 128, 8, F>(
            device,
            projections,
            weights,
            &stored,
            rows,
            &mut consume,
        );
    } else if rows > 128 {
        let lanes = prepare_rows::<H, R, 256>(device, &stored, 0, rows);
        run_restored::<H, Q, K, R, 256, F, _>(
            device,
            projections,
            weights,
            lanes,
            0,
            rows,
            &mut consume,
            (),
        );
    } else {
        let lanes = prepare_rows::<H, R, 128>(device, &stored, 0, rows);
        run_restored::<H, Q, K, R, 128, F, _>(
            device,
            projections,
            weights,
            lanes,
            0,
            rows,
            &mut consume,
            (),
        );
    }
}

// Static 16/32-row direct contractions avoid runtime summed-reduction
// prefixes. The consumer writes the 128-row scratch allocation; only live rows
// are valid, and attention masks/zeros its padded key/value positions.
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
    // W=16 leaves four BF16 KV columns per slice, the minimum contiguous
    // output packet for the current transpose/commit layout.
    if live <= 16 {
        single_tile::<H, Q, K, R, 16>(
            device,
            projections,
            weights,
            stored,
            wave,
            live,
            &mut output,
        );
    } else {
        single_tile::<H, Q, K, R, 32>(
            device,
            projections,
            weights,
            stored,
            wave,
            live,
            &mut output,
        );
    }
    consume(device, output.0, output.1, output.2, wave);
}

fn single_tile<H: AxisName, Q: AxisName, K: AxisName, R: AxisName, const W: usize>(
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
    let tile = wave * (128 / W);
    let input = if live <= 8 {
        load_rows::<H, R, 8, W>(device, stored, tile)
    } else {
        load_rows::<H, R, W, W>(device, stored, tile)
    };
    let lanes = fill(device, &input, live.div_ceil(8));
    let q = project(device, projections.0, weights.0, &lanes);
    let k = project(device, projections.1, weights.1, &lanes);
    let v = project(device, projections.2, weights.2, &lanes);
    land_tile::<Q, R, W>(device, &q, &mut output.0);
    land_tile::<K, R, W>(device, &k, &mut output.1);
    land_tile::<K, R, W>(device, &v, &mut output.2);
}

fn land_tile<O: AxisName, R: AxisName, const W: usize>(
    device: &mut Device,
    values: &Values<O, R, W>,
    output: &mut WaveOutput<O, R>,
) {
    // SAFETY: rename the same eight output owners without moving data.
    let values = unsafe {
        values.view().reshape::<Chip, Cluster<O>,
            m![O % (O / 2) / (O / 8), R % {W} / 8, O % (O / 8) / ({LocalWidth<O, R, W>})],
            m![R % 8, O % ({LocalWidth<O, R, W>})]>()
    };
    values.to_dm_view(&mut device.tdma,
        output.view_mut().slice_tile::<m![R % 128 / {W}], 1,
            m![O % (O / 2) / 128 # 16, R % 128 / {W} = 1 #{!} ({DirectTiles<R, W>}), R % {W} / 8]>(0));
}

fn prepare_rows<H: AxisName, R: AxisName, const W: usize>(
    device: &mut Device,
    stored: &StoredRows<H, R>,
    wave: usize,
    live: usize,
) -> Operand<H, R, W> {
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
    fill(device, &input, live.div_ceil(8))
}

/// The full first pair and either another pair or a final single wave. Keeping
/// the next operand live while K/V read the first makes TRF allocation disjoint.
fn two_restored<
    H: AxisName,
    Q: AxisName,
    K: AxisName,
    R: AxisName,
    const W: usize,
    const N: usize,
    F,
>(
    device: &mut Device,
    projections: (
        &QkvProjection<H, Q>,
        &QkvProjection<H, K>,
        &QkvProjection<H, K>,
    ),
    weights: (&Weight<H, Q>, &Weight<H, K>, &Weight<H, K>),
    stored: &StoredRows<H, R>,
    rows: usize,
    consume: &mut F,
) where
    F: FnMut(&mut Device, WaveOutput<Q, R>, WaveOutput<K, R>, WaveOutput<K, R>, usize),
{
    const {
        assert!(R::SIZE == 512 && (W == 128 || W == 256));
        assert!(N <= W && N.is_power_of_two());
    }
    let first = prepare_rows::<H, R, 256>(device, stored, 0, 256);
    // Select N before issuing the first pair, keeping the next DMA and fill
    // in the same branch-free scheduling unit as the current Q.
    let next = Stage(|device: &mut Device| {
        let input = load_rows::<H, R, N, W>(device, stored, 256 / W);
        fill(device, &input, (rows - 256).div_ceil(8))
    });
    let second = run_restored::<H, Q, K, R, 256, F, _>(
        device,
        projections,
        weights,
        first,
        0,
        256,
        consume,
        next,
    );
    run_restored::<H, Q, K, R, W, F, _>(
        device,
        projections,
        weights,
        second,
        256 / W,
        rows - 256,
        consume,
        (),
    );
}

fn run_restored<
    H: AxisName,
    Q: AxisName,
    K: AxisName,
    R: AxisName,
    const W: usize,
    F,
    N: Points<Device>,
>(
    device: &mut Device,
    projections: (
        &QkvProjection<H, Q>,
        &QkvProjection<H, K>,
        &QkvProjection<H, K>,
    ),
    weights: (&Weight<H, Q>, &Weight<H, K>, &Weight<H, K>),
    lanes: Operand<H, R, W>,
    wave: usize,
    live: usize,
    consume: &mut F,
    next: N,
) -> N::Output
where
    F: FnMut(&mut Device, WaveOutput<Q, R>, WaveOutput<K, R>, WaveOutput<K, R>, usize),
{
    const {
        assert!(W == 128 || W == 256);
    }
    let groups = live.div_ceil(8);
    let q = project_restored(device, projections.0, &weights.0.restored, &lanes, groups);
    // Q is issued first; its factor is ready before the next activation fill
    // uses the sub context. K and V still need the current TRF operand.
    let next = next.at(device);
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
    next.finish(device)
}

#[cfg(any(test, feature = "kernel-diagnostics"))]
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

#[cfg(test)]
mod tests {
    use super::{self as qkv, QkvProjection};
    use crate::ops::linear::nvfp4::tiled::tests as fixtures;
    use crate::tensor::scalar::Value as Count;
    use fixtures::*;
    use furiosa_opt_std::{prelude::*, runtime::launch};
    axes![KvWidth = 1024];
    #[device(chip = 1)]
    fn shared(
        device: &mut Device,
        q: &QkvProjection<Hidden, OutputWidth>,
        k: &QkvProjection<Hidden, KvWidth>,
        v: &QkvProjection<Hidden, KvWidth>,
        count: &Count,
        input: &HbmTensor<bf16, m![1], m![Rows, Hidden]>,
    ) -> (
        HbmTensor<bf16, m![1], m![Rows, OutputWidth]>,
        HbmTensor<bf16, m![1], m![Rows, KvWidth]>,
        HbmTensor<bf16, m![1], m![Rows, KvWidth]>,
    ) {
        let own = input.to_dm(&mut device.tdma);
        let (q, k, v) = crate::models::decoder::attention::llama::nvfp4::inspect_qkv_heads::<
            Hidden,
            OutputWidth,
            KvWidth,
            Rows,
        >(
            device,
            (q, k, v),
            &own,
            crate::tensor::live::Live::<(), Rows>::read(count).rows(),
        );
        let (mut qo, mut ko, mut vo) = (HbmTensor::new(), HbmTensor::new(), HbmTensor::new());
        q.view().to_hbm_view(&mut device.tdma, qo.view_mut());
        k.view().to_hbm_view(&mut device.tdma, ko.view_mut());
        v.view().to_hbm_view(&mut device.tdma, vo.view_mut());
        (qo, ko, vo)
    }
    #[tokio::test(flavor = "current_thread")]
    #[ignore = "opens a device"]
    async fn projects_live_rows_into_qkv_heads() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(shared.topology()).unwrap();
        let qn = projection::<OutputWidth>(&mut device).await;
        let kn = projection::<KvWidth>(&mut device).await;
        let mut vn = projection::<KvWidth>(&mut device).await;
        vn.scales.output =
            fixtures::upload::<f32, m![1 # 8]>(&mut device.pdma, vec![-0.5; 8]).await;
        let input = input::<Hidden>(&mut device).await;
        let original = fixtures::read::<_, m![Rows, Hidden]>(&mut device.pdma, &input).await;
        // DMA capacities, direct/restored cutover, paired waves, and every tail capacity.
        for rows in [
            1, 8, 9, 16, 17, 32, 33, 64, 65, 128, 129, 256, 257, 264, 265, 272, 273, 288, 289, 320,
            321, 384, 385, 512,
        ] {
            let mut padded = original.clone();
            padded[rows * Hidden::SIZE..].fill(bf16::from_f32(f32::NAN));
            let input = fixtures::upload::<bf16, m![Rows, Hidden]>(&mut device.pdma, padded).await;
            let count = count(&mut device, rows).await;
            let (qo, ko, vo) = launch(shared, (&mut device, &qn, &kn, &vn, &count, &input))
                .await
                .unwrap();
            let kh = fixtures::read::<_, m![Rows, KvWidth]>(&mut device.pdma, &ko).await;
            let vh = fixtures::read::<_, m![Rows, KvWidth]>(&mut device.pdma, &vo).await;
            assert!(
                kh.iter()
                    .zip(&vh)
                    .take(rows * KvWidth::SIZE)
                    .all(|(k, v)| k.to_f32() == -v.to_f32()),
                "V must use its own output scale"
            );
            check::<Hidden, OutputWidth>(&mut device, qo, rows).await;
            check::<Hidden, KvWidth>(&mut device, ko, rows).await;
        }
    }
    #[tokio::test(flavor = "current_thread")]
    #[ignore = "opens a device"]
    async fn sums_scaled_blocks_before_rounding() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(shared.topology()).unwrap();
        let mut qn = projection::<OutputWidth>(&mut device).await;
        let kn = projection::<KvWidth>(&mut device).await;
        let vn = projection::<KvWidth>(&mut device).await;
        // Opposite-signed adjacent blocks must be scaled before their sums combine.
        // In every 64 inputs, 16 * (.25 - .5 + 1 - 2) = -20;
        // across 4096 inputs, with tensor factor .5, that is -640.
        // Rotating scales by one output flips the sign, detecting owner misalignment.
        let codes: Vec<_> = (0..Hidden::SIZE * OutputWidth::SIZE)
            .map(|i| {
                f4e2m1::from_bits(if (i % Hidden::SIZE / 16) % 2 == 0 {
                    2
                } else {
                    10
                })
            })
            .collect();
        let scales: Vec<_> = (0..Hidden::SIZE * OutputWidth::SIZE / 16)
            .map(|i| {
                f8e4m3::from_f32(
                    [0.25, 0.5, 1.0, 2.0][(i / (Hidden::SIZE / 16) + i % (Hidden::SIZE / 16)) % 4],
                )
            })
            .collect();
        qn.weight = fixtures::upload::<f4e2m1, qkv::Stored<Hidden, OutputWidth>>(
            &mut device.pdma,
            QkvProjection::<Hidden, OutputWidth>::reorder::<1, _>(&codes),
        )
        .await;
        qn.weight_scale = fixtures::upload::<f8e4m3, qkv::StoredScales<Hidden, OutputWidth>>(
            &mut device.pdma,
            QkvProjection::<Hidden, OutputWidth>::reorder::<16, _>(&scales),
        )
        .await;
        // Distinct row values detect reuse of the first operand in a prefetched tail.
        for rows in [32, 33, 257] {
            let input = fixtures::upload::<bf16, m![Rows, Hidden]>(
                &mut device.pdma,
                (0..Rows::SIZE * Hidden::SIZE).map(|i| {
                    let row = i / Hidden::SIZE;
                    bf16::from_f32(if row < rows {
                        (row % 3 + 1) as f32 * 0.5
                    } else {
                        f32::NAN
                    })
                }),
            )
            .await;
            let count = fixtures::count(&mut device, rows).await;
            let (q, _, _) = launch(shared, (&mut device, &qn, &kn, &vn, &count, &input))
                .await
                .unwrap();
            let values = fixtures::read::<_, m![Rows, OutputWidth]>(&mut device.pdma, &q).await;
            for (i, value) in values.iter().take(rows * OutputWidth::SIZE).enumerate() {
                let expected = (if i % OutputWidth::SIZE % 2 == 0 {
                    -320.0
                } else {
                    320.0
                }) * ((i / OutputWidth::SIZE) % 3 + 1) as f32;
                assert_eq!(
                    value.to_f32(),
                    expected,
                    "unequal scale blocks: rows={rows}, element={i}"
                );
            }
        }
    }
    #[tokio::test(flavor = "current_thread")]
    #[ignore = "opens a device"]
    async fn applies_the_tensor_factor_to_the_complete_dot() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(shared.topology()).unwrap();
        let mut qn = projection::<OutputWidth>(&mut device).await;
        let kn = projection::<KvWidth>(&mut device).await;
        let vn = projection::<KvWidth>(&mut device).await;
        qn.weight = fixtures::upload::<f4e2m1, qkv::Stored<Hidden, OutputWidth>>(
            &mut device.pdma,
            vec![f4e2m1::from_bits(2); Hidden::SIZE * OutputWidth::SIZE],
        )
        .await;
        qn.weight_scale = fixtures::upload::<f8e4m3, qkv::StoredScales<Hidden, OutputWidth>>(
            &mut device.pdma,
            vec![f8e4m3::from_f32(0.125); Hidden::SIZE * OutputWidth::SIZE / 16],
        )
        .await;
        let input = fixtures::upload::<bf16, m![Rows, Hidden]>(
            &mut device.pdma,
            vec![bf16::from_f32(6.0); Rows::SIZE * Hidden::SIZE],
        )
        .await;
        // Non-power-of-two factors expose scaling changes hidden by exact powers of two;
        // a small normal factor also checks that a representable nonzero result survives.
        // The FP64 oracle uses the stored FP32 factor and permits final BF16 rounding.
        for factor in [1.3_f32, -0.7, 2.0_f32.powi(-120)] {
            qn.scales.output =
                fixtures::upload::<f32, m![1 # 8]>(&mut device.pdma, vec![factor; 8]).await;
            let expected = 3072.0_f64 * factor as f64;
            let rounded = bf16::from_f32(expected as f32);
            assert_ne!(rounded.to_f32(), 0.0);
            let ulp = 2.0_f64.powi(((rounded.to_bits() >> 7) & 255) as i32 - 127 - 7);
            let tolerance = 0.5 * ulp + 1e-6 * expected.abs();
            for rows in [32, 33] {
                let count = fixtures::count(&mut device, rows).await;
                let (q, _, _) = launch(shared, (&mut device, &qn, &kn, &vn, &count, &input))
                    .await
                    .unwrap();
                let values = fixtures::read::<_, m![Rows, OutputWidth]>(&mut device.pdma, &q).await;
                for (index, value) in values.iter().take(rows * OutputWidth::SIZE).enumerate() {
                    let actual = value.to_f32() as f64;
                    assert!(
                        actual.is_finite() && (actual - expected).abs() <= tolerance,
                        "tensor factor oracle: rows={rows} factor={factor} element={index} actual={actual} expected={expected} tolerance={tolerance}"
                    );
                }
            }
        }
    }
    async fn projection<O: AxisName>(device: &mut Device) -> QkvProjection<Hidden, O> {
        let (weight, weight_scale, scales) = weights::<Hidden, O, _, _>(
            device,
            QkvProjection::<Hidden, O>::reorder::<1, _>,
            QkvProjection::<Hidden, O>::reorder::<16, _>,
        )
        .await;
        QkvProjection {
            weight,
            weight_scale,
            scales,
        }
    }

    #[test]
    fn checkpoint_reorder_preserves_packed_weight_and_scale_coordinates() {
        fn check<const PACK: usize>() {
            let original: Vec<_> = (0..Hidden::SIZE * KvWidth::SIZE / PACK).collect();
            let reordered = QkvProjection::<Hidden, KvWidth>::reorder::<PACK, _>(&original);
            // Literal landmarks independently pin output-row, input-slice, and owner transitions.
            if PACK == 2 {
                assert_eq!(
                    [
                        reordered[31],
                        reordered[32],
                        reordered[4095],
                        reordered[4096],
                        reordered[262144]
                    ],
                    [31, 2048, 260127, 32, 262144]
                );
            } else {
                assert_eq!(
                    [
                        reordered[3],
                        reordered[4],
                        reordered[511],
                        reordered[512],
                        reordered[32768]
                    ],
                    [3, 256, 32515, 4, 32768]
                );
            }

            for owner in 0..8 {
                for slice in 0..64 {
                    for out in 0..128 {
                        for column in 0..64 / PACK {
                            let target = ((owner * 64 + slice) * 128 + out) * (64 / PACK) + column;
                            let source = ((owner * 128 + out) * 4096 + slice * 64) / PACK + column;
                            assert_eq!(reordered[target], original[source]);
                        }
                    }
                }
            }
        }
        check::<2>();
        check::<16>();
    }
}
