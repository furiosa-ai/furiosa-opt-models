//! Resident, shared BF16 activation preparation and blocked QKV projection outputs.
use super::*;
use crate::ops::normalization::span::Owned;
type Input<H, R> = HbmTensor<bf16, Chip, m![H / 128, R, H % 128]>;
pub(crate) type Resident<H, R> =
    DmTensor<bf16, Chip, m![2], m![H % 1024 / 32, R % 64 / 8], m![H / 1024, R / 64, R % 8, H % 32]>;
type Local<R> = m![R % 128 / 64, R % 8];
axes![QkvOwner = 64];
type PhysicalOwners = m![QkvOwner % 32];
type QkvCluster = m![QkvOwner / 32];
type Reduced<R> = m![{ PhysicalOwners }, R % 64 / 8];
type Placement<H> = m![{ PhysicalOwners }, H % 1024 / 128];
type Lanes<H, R> = TrfTensor<
    bf16,
    Chip,
    QkvCluster,
    Placement<H>,
    m![R % 8],
    m![R % 128 / 64, H % 32 / 16, H % 128 / 32, R % 64 / 8, H % 16],
>;
type Partial<O, R> = DmTensor<f32, Chip, QkvCluster, Reduced<R>, m![{ Local<R> }, O % (O / 64)]>;
pub(crate) type WaveOutput<O, R> =
    DmTensor<bf16, Chip, QkvCluster, Reduced<R>, m![{ Local<R> }, O % (O / 64)]>;

type Triple<Q, K, R> = (Partial<Q, R>, Partial<K, R>, Partial<K, R>);

fn pack_rows<H: AxisName, R: AxisName>(device: &mut Device, input: &Owned<H, R>) -> Input<H, R> {
    let mut output = Input::new();
    // Exchange eight 16-feature owners for eight 32-row owners on the switch.
    // The resident DM redistribution reads contiguous 128-feature blocks from HBM.
    let packed: DmTensor<
        bf16,
        Chip,
        m![R / (R / 2)],
        m![H / 128, R % (R / 2) / 32],
        m![R % 32, H % 128],
    > = device
        .sub
        .begin(input.view())
        .fetch::<m![R % 32, R % (R / 2) / 32], m![H % 16]>()
        .switch::<m![H / 128, R % (R / 2) / 32], m![R % 32, H % 128 / 16]>(
            SwitchConfig::InterTranspose {
                slice1: 8,
                slice0: 1,
                time0: 1,
            },
        )
        .collect::<m![R % 32, H % 128 / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();
    packed
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
    output
}

// One bounded HBM redistribution per span, then all four slabs' activations
// remain in DM. Every TRF refill uses the switch, not another replicated HBM read.
fn resident<H: AxisName, R: AxisName, const N: usize, const CHUNKS: usize>(
    device: &mut Device,
    input: &Input<H, R>,
) -> Resident<H, R> {
    let mut output = Resident::new();
    input.view().tile::<m![R], N, m![H / 128, R = {N} # R, H % 128]>(0)
        .to_dm_view(&mut device.tdma, output.view_mut()
            .tile::<m![R / 64], CHUNKS,
                m![H / 1024, R / 64 = {CHUNKS} #{!} (R / 64), R % 8, H % 32]>(0));
    output
}

fn lanes<H: AxisName, R: AxisName>(
    device: &mut Device,
    input: &Resident<H, R>,
    panel: usize,
    wave: usize,
    rows: usize,
) -> Lanes<H, R> {
    // SAFETY: rename two physical cluster owners without moving data.
    let input = unsafe {
        input.view().reshape::<Chip, QkvCluster,
        m![H % 1024 / 32, R % 64 / 8], m![H / 1024, R / 64, R % 8, H % 32]>()
    };
    device
        .sub
        .begin(
            input
                .tile::<m![H / 1024], 1, m![H / 1024 = 1 # 4, R / 64, R % 8, H % 32]>(panel)
                .tile::<m![R / 128], 1, m![H / 1024 = 1 # 4, R / 128 = 1 # (R / 128),
            R % 128 / 64, R % 8, H % 32]>(wave),
        )
        .fetch::<m![R % 8, R % 128 / 64, H % 32 / 16], m![H % 16]>()
        .switch::<Placement<H>, m![R % 8, R % 128 / 64, H % 32 / 16, H % 128 / 32, R % 64 / 8]>(
            SwitchConfig::TransposedBroadcast1 {
                slice1: 8,
                slice0: 32,
            },
        )
        .collect::<m![R % 8, R % 128 / 64, H % 32 / 16, H % 128 / 32, R % 64 / 8], m![H % 16]>()
        .to_trf_prefix::<m![R % 128 / 64], _, _>(
            rows.div_ceil(64) - (2 * wave).min(rows.div_ceil(64)),
        )
}

fn single<H: AxisName, O: AxisName, R: AxisName, E: M>(
    device: &mut Device,
    weight: &Prepared<H, O>,
    lanes: &Lanes<H, R>,
    output: DmTensorViewMut<'_, f32, Chip, QkvCluster, Reduced<R>, E>,
    groups: usize,
    panel: usize,
) {
    // SAFETY: Q and KV each have 64 output owners, in the same physical order.
    // Only the local output width differs (64 for Q, 16 for K/V).
    let weight = unsafe {
        weight
            .view()
            .reshape::<Chip, QkvCluster, Placement<H>, m![H / 1024, O % (O / 64), H % 128]>()
    };
    device
        .main
        .begin(
            weight
                .tile::<m![H / 1024], 1, m![H / 1024 = 1 # (H / 1024), O % (O / 64), H % 128]>(
                    panel,
                )
                .unpad::<m![O % (O / 64), H % 128]>(),
        )
        .fetch::<m![
            R % 128 / 64,
            O % (O / 64) / 2,
            H % 32 / 16,
            H % 128 / 32,
            O % 2
        ], m![H % 16]>()
        .collect::<m![
            R % 128 / 64,
            O % (O / 64) / 2,
            H % 32 / 16,
            H % 128 / 32,
            O % 2
        ], m![H % 16]>()
        .contract_outer::<m![
            R % 128 / 64,
            O % (O / 64) / 2,
            H % 32 / 16,
            H % 128 / 32,
            R % 64 / 8
        ], m![O % 2, H % 16], _, _, _>(lanes)
        .contract_packet::<m![O % 2]>()
        .contract_time::<m![R % 128 / 64, O % (O / 64) / 2, R % 64 / 8]>()
        .contract_lane::<m![R % 128 / 64, O % (O / 64) / 2, R % 64 / 8, O % 2], m![R % 8]>(
            LaneMode::Interleaved,
        )
        .vector_init()
        .vector_inter_slice_reduce::<Reduced<R>, m![R % 128 / 64, O % (O / 64)]>(
            InterSliceReduceOpF32::Add,
        )
        .vector_final()
        .transpose::<m![R % 128 / 64, O % (O / 64) / 2, R % 8], m![O % 2 # 8]>()
        .commit_trim::<m![O % 2]>()
        .commit_prefix::<m![R % 128 / 64], _, _>(output, groups);
}

pub(crate) fn activation<H: AxisName, R: AxisName>(
    device: &mut Device,
    input: &Owned<H, R>,
    rows: usize,
) -> Resident<H, R> {
    let packed = pack_rows(device, input);
    if rows <= 64 {
        resident::<H, R, 64, 1>(device, &packed)
    } else if rows <= 128 {
        resident::<H, R, 128, 2>(device, &packed)
    } else {
        resident::<H, R, 512, 8>(device, &packed)
    }
}

// Add the final slab in FP32, then multiply the tensor factor in the same
// stream. This removes an FP32 DM round trip without distributing scaling
// over partial sums or changing their accumulation order.
fn finish_wave<O: AxisName, R: AxisName>(
    device: &mut Device,
    previous: &Partial<O, R>,
    last: &Partial<O, R>,
    factor: &VrfTensor<f32, Chip, QkvCluster, Reduced<R>, m![1 # 8]>,
    groups: usize,
) -> WaveOutput<O, R> {
    let mut values: DmTensor<bf16, Chip, QkvCluster, Reduced<R>, m![{Local<R>}, O % (O / 64)]> =
        DmTensor::new();
    device
        .main
        .begin_interleaved::<Terms, _, _, _, _, _>(previous.view(), last.view())
        .fetch::<m![{Local<R>}, Terms, O % (O / 64) / 8], m![O % 8]>()
        .collect::<m![{Local<R>}, Terms, O % (O / 64) / 8], m![O % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Terms, m![{Local<R>}, O % (O / 64) / 8]>()
        .vector_narrow_split::<m![{Local<R>}, O % (O / 64) / 4], m![O % 4]>()
        .vector_fp_zip(FpBinaryOp::AddF)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), factor)
        .vector_widen_concat::<m![{Local<R>}, O % (O / 64) / 8], m![O % 8]>()
        .vector_final()
        .cast::<bf16, m![O % 8 # 16]>()
        .commit_trim::<m![O % 8]>()
        .commit_prefix::<m![R % 128 / 64], _, _>(values.view_mut(), groups);
    values
}

fn add<O: AxisName, R: AxisName>(
    device: &mut Device,
    a: &Partial<O, R>,
    b: &Partial<O, R>,
    out: &mut Partial<O, R>,
    groups: usize,
) {
    device
        .main
        .begin_interleaved::<Terms, _, _, _, _, _>(a.view(), b.view())
        .fetch::<m![{Local<R>},Terms,O % (O / 64) / 8], m![O % 8]>()
        .collect::<m![{Local<R>},Terms,O % (O / 64) / 8], m![O % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Terms, m![{Local<R>},O % (O / 64) / 8]>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_final()
        .commit_trim::<m![O % 8]>()
        .commit_prefix::<m![R % 128 / 64], _, _>(out.view_mut(), groups);
}

/// One activation fill serves Q, K and V before that TRF buffer is recycled.
/// The other 32-KiB buffer prepares the following 1024-feature slab.
pub(crate) fn consume_qkv<H: AxisName, Q: AxisName, K: AxisName, R: AxisName, F>(
    device: &mut Device,
    projections: (&Panels<H, Q>, &Panels<H, K>, &Panels<H, K>),
    weights: (&Prepared<H, Q>, &Prepared<H, K>, &Prepared<H, K>),
    input: &Resident<H, R>,
    rows: usize,
    mut consume: F,
) where
    F: FnMut(&mut Device, WaveOutput<Q, R>, WaveOutput<K, R>, WaveOutput<K, R>, usize),
{
    const {
        assert!(H::SIZE == 4096 && Q::SIZE == 4096 && K::SIZE == 1024 && R::SIZE == 512);
    }
    let fq = block::scalar::<QkvCluster, Reduced<R>>(device, &projections.0.scales.output);
    let fk = block::scalar::<QkvCluster, Reduced<R>>(device, &projections.1.scales.output);
    let fv = block::scalar::<QkvCluster, Reduced<R>>(device, &projections.2.scales.output);
    for wave in 0..rows.div_ceil(128) {
        let groups = rows.div_ceil(64) - (2 * wave).min(rows.div_ceil(64));
        let l0 = lanes(device, input, 0, wave, rows);
        let l1 = lanes(device, input, 1, wave, rows);
        let p0 = triple(device, weights, &l0, groups, 0);
        let l2 = lanes(device, input, 2, wave, rows);
        let p1 = triple(device, weights, &l1, groups, 1);
        let s1 = sum_triple(device, &p0, &p1, groups);
        let l3 = lanes(device, input, 3, wave, rows);
        let p2 = triple(device, weights, &l2, groups, 2);
        let s2 = sum_triple(device, &s1, &p2, groups);
        let p3 = triple(device, weights, &l3, groups, 3);
        let q = finish_wave(device, &s2.0, &p3.0, &fq, groups);
        let k = finish_wave(device, &s2.1, &p3.1, &fk, groups);
        let v = finish_wave(device, &s2.2, &p3.2, &fv, groups);
        consume(device, q, k, v, wave);
    }
}

fn triple<H: AxisName, Q: AxisName, K: AxisName, R: AxisName>(
    device: &mut Device,
    weights: (&Prepared<H, Q>, &Prepared<H, K>, &Prepared<H, K>),
    lanes: &Lanes<H, R>,
    groups: usize,
    panel: usize,
) -> Triple<Q, K, R> {
    let (mut q, mut k, mut v) = (Partial::new(), Partial::new(), Partial::new());
    single(device, weights.0, lanes, q.view_mut(), groups, panel);
    single(device, weights.1, lanes, k.view_mut(), groups, panel);
    single(device, weights.2, lanes, v.view_mut(), groups, panel);
    (q, k, v)
}
fn sum_triple<Q: AxisName, K: AxisName, R: AxisName>(
    device: &mut Device,
    a: &Triple<Q, K, R>,
    b: &Triple<Q, K, R>,
    groups: usize,
) -> Triple<Q, K, R> {
    let (mut q, mut k, mut v) = (Partial::new(), Partial::new(), Partial::new());
    add(device, &a.0, &b.0, &mut q, groups);
    add(device, &a.1, &b.1, &mut k, groups);
    add(device, &a.2, &b.2, &mut v, groups);
    (q, k, v)
}

/// Select compact decode preparation/contraction without changing the DM head contract.
pub(crate) fn consume_normalized<H: AxisName, Q: AxisName, K: AxisName, R: AxisName, F>(
    device: &mut Device,
    projections: (&Panels<H, Q>, &Panels<H, K>, &Panels<H, K>),
    weights: (&Prepared<H, Q>, &Prepared<H, K>, &Prepared<H, K>),
    input: &Owned<H, R>,
    rows: usize,
    consume: F,
) where
    F: FnMut(&mut Device, WaveOutput<Q, R>, WaveOutput<K, R>, WaveOutput<K, R>, usize),
{
    if rows <= 8 {
        compact_qkv::<H, Q, K, R, 8, F>(device, projections, weights, input, consume);
    } else if rows <= 16 {
        compact_qkv::<H, Q, K, R, 16, F>(device, projections, weights, input, consume);
    } else {
        let input = activation(device, input, rows);
        consume_qkv(device, projections, weights, &input, rows, consume);
    }
}

fn compact_qkv<H: AxisName, Q: AxisName, K: AxisName, R: AxisName, const W: usize, F>(
    device: &mut Device,
    projections: (&Panels<H, Q>, &Panels<H, K>, &Panels<H, K>),
    weights: (&Prepared<H, Q>, &Prepared<H, K>, &Prepared<H, K>),
    input: &Owned<H, R>,
    mut consume: F,
) where
    F: FnMut(&mut Device, WaveOutput<Q, R>, WaveOutput<K, R>, WaveOutput<K, R>, usize),
{
    use super::super::compact as tile;
    fn project<H: AxisName, O: AxisName, R: AxisName, const W: usize>(
        device: &mut Device,
        weight: &Prepared<H, O>,
        lanes: &tile::Lanes<H, R, W>,
        panel: usize,
    ) -> tile::Partial<O, R, W> {
        let weight = weight
            .view()
            .tile::<m![H / 1024], 1, m![H / 1024 = 1 # 4, O % (O / 64), H % 128]>(panel)
            .unpad::<m![O % (O / 64), H % 128]>();
        tile::single::<H, O, R, W, _, _, _>(device, weight, lanes)
    }
    fn land<O: AxisName, R: AxisName, const W: usize>(
        device: &mut Device,
        values: tile::Output<O, R, W>,
    ) -> WaveOutput<O, R> {
        // SAFETY: identical 64 output owners and one reduction slice per owner.
        let values = unsafe {
            values.reshape::<Chip, QkvCluster, m![{PhysicalOwners}, 1 # 8],
            m![R % {W}, O % (O / 64)]>()
        };
        let mut output = WaveOutput::new();
        // The consumer retains its 128-row allocation, but only live slice owners
        // and the first local row group receive values from this compact producer.
        if const { W == 8 } {
            values.view().to_dm_view(
                &mut device.tdma,
                output
                    .view_mut()
                    .slice_tile::<m![R % 64 / 8], 1, m![{PhysicalOwners}, R % 64 / 8 = 1 #{!} 8]>(0)
                    .tile::<m![R % 128 / 64], 1, m![R % 128 / 64 = 1 #{!} 2, R % 8, O % (O / 64)]>(
                        0,
                    ),
            );
        } else {
            values.view().to_dm_view(
                &mut device.tdma,
                output
                    .view_mut()
                    .slice_tile::<m![R % 64 / 8], 2, m![{PhysicalOwners}, R % 64 / 8 = 2 #{!} 8]>(0)
                    .tile::<m![R % 128 / 64], 1, m![R % 128 / 64 = 1 #{!} 2, R % 8, O % (O / 64)]>(
                        0,
                    ),
            );
        }
        output
    }
    let input = tile::input::<H, R, W>(device, input);
    let triple = |device: &mut Device, lanes: &tile::Lanes<H, R, W>, panel| {
        (
            project(device, weights.0, lanes, panel),
            project(device, weights.1, lanes, panel),
            project(device, weights.2, lanes, panel),
        )
    };
    let l0 = tile::lanes(device, &input, 0);
    let l1 = tile::lanes(device, &input, 1);
    let p0 = triple(device, &l0, 0);
    let l2 = tile::lanes(device, &input, 2);
    let p1 = triple(device, &l1, 1);
    let s1 = (
        tile::add(device, p0.0.view(), p1.0.view()),
        tile::add(device, p0.1.view(), p1.1.view()),
        tile::add(device, p0.2.view(), p1.2.view()),
    );
    let l3 = tile::lanes(device, &input, 3);
    let p2 = triple(device, &l2, 2);
    let s2 = (
        tile::add(device, s1.0.view(), p2.0.view()),
        tile::add(device, s1.1.view(), p2.1.view()),
        tile::add(device, s1.2.view(), p2.2.view()),
    );
    let p3 = triple(device, &l3, 3);
    let q = tile::add(device, s2.0.view(), p3.0.view());
    let k = tile::add(device, s2.1.view(), p3.1.view());
    let v = tile::add(device, s2.2.view(), p3.2.view());
    let q = block::finish(device, &projections.0.scales, q);
    let k = block::finish(device, &projections.1.scales, k);
    let v = block::finish(device, &projections.2.scales, v);
    let q = land::<Q, R, W>(device, q);
    let k = land::<K, R, W>(device, k);
    let v = land::<K, R, W>(device, v);
    consume(device, q, k, v, 0);
}
