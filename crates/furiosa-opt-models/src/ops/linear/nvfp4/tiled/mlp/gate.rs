use super::*;

axes![Term = 2];
type Partition<H, I> = m![{ Owners<I> }, H % 1024 / 128];
type Weight<H, I> = DmTensor<bf16, Chip, Cluster<I>, Partition<H, I>, m![I % 224, H % 128]>;
type Resident<H, I, R> = DmTensor<
    bf16,
    Chip,
    Cluster<I>,
    m![H % 1024 / 32, R % 64 / 8],
    m![H / 1024, R / 64, R % 8, H % 32],
>;
type Lanes<H, I, R> = TrfTensor<
    bf16,
    Chip,
    Cluster<I>,
    Partition<H, I>,
    m![R % 8],
    m![R % 128 / 64, H % 32 / 16, H % 128 / 32, R % 64 / 8, H % 16],
>;
type Both<I, R> =
    DmTensor<f32, Chip, Cluster<I>, Slices<I, R>, m![R / 128, { LocalRows<R> }, Term, I % 224]>;
type Part<I, R> =
    DmTensor<f32, Chip, Cluster<I>, Slices<I, R>, m![{ LocalRows<R> }, Term, I % 224]>;

// Keep all eight row owners, but shorten their local row dimension before packing.
// N is the capacity per cluster; the HBM allocation retains the full span stride.
fn pack_rows<H: AxisName, R: AxisName, const N: usize>(
    device: &mut Device,
    input: &Owned<H, R>,
) -> Input<H, R> {
    type Local<R, const N: usize> = m![R % { N } / 8];
    let selected = input
        .view()
        .tile::<m![R % 256], N, m![R % 256 = {N} # 256, H % 16]>(0)
        .unpad::<m![R % 256 = { N }, H % 16]>();
    // SAFETY: rename the selected local prefix; clusters and slice owners are unchanged.
    let selected =
        unsafe { selected.reshape::<Chip, m![R / 256], m![H / 16], m![R % { N }, H % 16]>() };
    let packed: DmTensor<
        bf16,
        Chip,
        m![R / 256],
        m![H / 128, R % {N} / ({Local<R, N>})],
        m![R % ({Local<R, N>}), H % 128],
    > = device
        .sub
        .begin(selected)
        .fetch::<m![R % ({Local<R, N>}), R % {N} / ({Local<R, N>})], m![H % 16]>()
        .switch::<m![H / 128, R % {N} / ({Local<R, N>})], m![R % ({Local<R, N>}), H % 128 / 16]>(
            SwitchConfig::InterTranspose {
                slice1: 8,
                slice0: 1,
                time0: 1,
            },
        )
        .collect::<m![R % ({Local<R, N>}), H % 128 / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();
    let mut output = Input::new();
    packed.view().to_hbm_view(
        &mut device.tdma,
        output
            .view_mut()
            .tile::<m![R % 256], N, m![H / 128, R / 256, R % 256 = {N} #{!} 256, H % 128]>(0),
    );
    output
}

// One bounded HBM redistribution per span, then all four slabs' activations
// remain in DM. Every TRF refill uses the switch, not another replicated HBM read.
fn resident<H: AxisName, I: AxisName, R: AxisName, const N: usize, const CHUNKS: usize>(
    device: &mut Device,
    input: &Input<H, R>,
) -> Resident<H, I, R> {
    let mut output = Resident::new();
    input.view().tile::<m![R], N, m![H / 128, R = {N} # R, H % 128]>(0)
        .to_dm_view(&mut device.tdma, output.view_mut()
            .tile::<m![R / 64], CHUNKS,
                m![H / 1024, R / 64 = {CHUNKS} #{!} (R / 64), R % 8, H % 32]>(0));
    output
}

fn lanes<H: AxisName, I: AxisName, R: AxisName>(
    device: &mut Device,
    input: &Resident<H, I, R>,
    panel: usize,
    wave: usize,
    rows: usize,
) -> Lanes<H, I, R> {
    device
        .sub
        .begin(
            input
                .view()
                .tile::<m![H / 1024], 1, m![H / 1024 = 1 # 4, R / 64, R % 8, H % 32]>(panel)
                .tile::<m![R / 128], 1, m![H / 1024 = 1 # 4, R / 128 = 1 # (R / 128),
            R % 128 / 64, R % 8, H % 32]>(wave),
        )
        .fetch::<m![R % 8, R % 128 / 64, H % 32 / 16], m![H % 16]>()
        .switch::<Partition<H, I>, m![R % 8, R % 128 / 64, H % 32 / 16, H % 128 / 32, R % 64 / 8]>(
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

// Two output features share each BF16 packet. The eight row groups are
// TRF-only time broadcasts, reusing each fetched weight packet across them.
fn paired<H: AxisName, I: AxisName, R: AxisName, E: M>(
    device: &mut Device,
    weights: (&Weight<H, I>, &Weight<H, I>),
    lanes: &Lanes<H, I, R>,
    output: DmTensorViewMut<'_, f32, Chip, Cluster<I>, Slices<I, R>, E>,
    groups: usize,
) {
    device
        .main
        .begin_interleaved::<Term, _, _, _, _, _>(weights.0.view(), weights.1.view())
        .fetch::<m![
            R % 128 / 64,
            Term,
            I % 224 / 2,
            H % 32 / 16,
            H % 128 / 32,
            I % 2
        ], m![H % 16]>()
        .collect::<m![
            R % 128 / 64,
            Term,
            I % 224 / 2,
            H % 32 / 16,
            H % 128 / 32,
            I % 2
        ], m![H % 16]>()
        .contract_outer::<m![
            R % 128 / 64,
            Term,
            I % 224 / 2,
            H % 32 / 16,
            H % 128 / 32,
            R % 64 / 8
        ], m![I % 2, H % 16], _, _, _>(lanes)
        .contract_packet::<m![I % 2]>()
        .contract_time::<m![R % 128 / 64, Term, I % 224 / 2, R % 64 / 8]>()
        .contract_lane::<m![R % 128 / 64, Term, I % 224 / 2, R % 64 / 8, I % 2], m![R % 8]>(
            LaneMode::Interleaved,
        )
        .vector_init()
        .vector_inter_slice_reduce::<Slices<I, R>, m![R % 128 / 64, Term, I % 224]>(
            InterSliceReduceOpF32::Add,
        )
        .vector_final()
        .transpose::<m![R % 128 / 64, Term, I % 224 / 2, R % 8], m![I % 2 # 8]>()
        .commit_trim::<m![I % 2]>()
        .commit_prefix::<m![R % 128 / 64], _, _>(output, groups);
}

fn single<H: AxisName, I: AxisName, R: AxisName, E: M>(
    device: &mut Device,
    weight: &Weight<H, I>,
    lanes: &Lanes<H, I, R>,
    output: DmTensorViewMut<'_, f32, Chip, Cluster<I>, Slices<I, R>, E>,
    groups: usize,
) {
    device
        .main
        .begin(weight.view())
        .fetch::<m![R % 128 / 64, I % 224 / 2, H % 32 / 16, H % 128 / 32, I % 2], m![H % 16]>()
        .collect::<m![R % 128 / 64, I % 224 / 2, H % 32 / 16, H % 128 / 32, I % 2], m![H % 16]>()
        .contract_outer::<m![
            R % 128 / 64,
            I % 224 / 2,
            H % 32 / 16,
            H % 128 / 32,
            R % 64 / 8
        ], m![I % 2, H % 16], _, _, _>(lanes)
        .contract_packet::<m![I % 2]>()
        .contract_time::<m![R % 128 / 64, I % 224 / 2, R % 64 / 8]>()
        .contract_lane::<m![R % 128 / 64, I % 224 / 2, R % 64 / 8, I % 2], m![R % 8]>(
            LaneMode::Interleaved,
        )
        .vector_init()
        .vector_inter_slice_reduce::<Slices<I, R>, m![R % 128 / 64, I % 224]>(
            InterSliceReduceOpF32::Add,
        )
        .vector_final()
        .transpose::<m![R % 128 / 64, I % 224 / 2, R % 8], m![I % 2 # 8]>()
        .commit_trim::<m![I % 2]>()
        .commit_prefix::<m![R % 128 / 64], _, _>(output, groups);
}

fn panel<H: AxisName, I: AxisName, R: AxisName, const ADD: bool>(
    device: &mut Device,
    input: &Resident<H, I, R>,
    weights: (&Weight<H, I>, &Weight<H, I>),
    previous: Option<&Both<I, R>>,
    output: &mut Both<I, R>,
    mut current: Lanes<H, I, R>,
    index: usize,
    rows: usize,
) -> Lanes<H, I, R> {
    // Fixed wave offsets avoid loop-carried buffer copies. Zero prefixes skip
    // inactive contractions/refills while the last slot carries the next slab.
    let waves = 4;
    #[unroll]
    for wave in 0..4 {
        let mut part = Part::new();
        let last = (wave + 5 - waves) / 4;
        let next = lanes(device, input, index + last, wave + 1 - last * waves, rows);
        let groups = rows.div_ceil(64) - (2 * wave).min(rows.div_ceil(64));
        let target = output.view_mut().tile::<m![R / 128], 1,
            m![R / 128 = 1 #{!} (R / 128), { LocalRows<R> }, Term, I % 224]>(wave);
        if ADD {
            paired(device, weights, &current, part.view_mut(), groups);
            let old = previous.unwrap().view().tile::<m![R / 128], 1,
                m![R / 128 = 1 # (R / 128), { LocalRows<R> }, Term, I % 224]>(wave)
                .unpad::<m![{ LocalRows<R> }, Term, I % 224]>();
            super::sum_into::<_, _, m![{ LocalRows<R> }, Term], m![I % 224], m![R % 128 / 64], _, _>(
                device,
                old,
                part.view(),
                target,
                groups,
            );
        } else {
            paired(device, weights, &current, target, groups);
        }
        current = next;
    }
    current
}

fn last<H: AxisName, I: AxisName, R: AxisName>(
    device: &mut Device,
    input: &Resident<H, I, R>,
    weight: &Weight<H, I>,
    previous: &Both<I, R>,
    term: usize,
    scales: &super::super::Scales,
    mut current: Lanes<H, I, R>,
    rows: usize,
) -> (Ready<I, R>, Lanes<H, I, R>) {
    axes![Side = 2];
    let factor = block::scalar::<Cluster<I>, Slices<I, R>>(device, &scales.output);
    let mut part = Part::<I, R>::new();
    let mut output = Ready::new();
    let waves = rows.div_ceil(128);
    for wave in 0..waves {
        // Gate's final refill becomes up's first operand; the final up refill
        // is unused, matching the existing Qwen sweep boundary.
        let last = (wave + 5 - waves) / 4;
        let next = lanes(device, input, 3, wave + 1 - last * waves, rows);
        let groups = rows.div_ceil(64) - (2 * wave).min(rows.div_ceil(64));
        single(
            device,
            weight,
            &current,
            part.view_mut()
                .tile::<m![Term], 1, m![{ LocalRows<R> }, Term = 1 #{!} 2, I % 224]>(term),
            groups,
        );
        let old = previous.view().tile::<m![R / 128], 1,
            m![R / 128 = 1 # (R / 128), { LocalRows<R> }, Term, I % 224]>(wave)
            .tile::<m![Term], 1, m![R / 128 = 1 # (R / 128), { LocalRows<R> }, Term = 1 # 2, I % 224]>(term);
        // Keep term padding in both views, so no copy is needed to interleave them.
        let old = old.unpad::<m![{ LocalRows<R> }, Term = 1 # 2, I % 224]>();
        let part = part
            .view()
            .tile::<m![Term], 1, m![{ LocalRows<R> }, Term = 1 # 2, I % 224]>(term);
        // FP32 add precedes the separate FP32 multiply in one stream.
        device
            .main
            .begin_interleaved::<Side, _, _, _, _, _>(old, part)
            .fetch::<m![{ LocalRows<R> }, Side, I % 224 / 8], m![I % 8]>()
            .collect::<m![{ LocalRows<R> }, Side, I % 224 / 8], m![I % 8]>()
            .vector_init()
            .vector_intra_slice_unzip::<Side, m![{ LocalRows<R> }, I % 224 / 8]>()
            .vector_narrow_split::<m![{ LocalRows<R> }, I % 224 / 4], m![I % 4]>()
            .vector_fp_zip(FpBinaryOp::AddF)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &factor)
            .vector_widen_concat::<m![{ LocalRows<R> }, I % 224 / 8], m![I % 8]>()
            .vector_final().cast::<bf16, m![I % 8 # 16]>()
            .commit_trim::<m![I % 8]>()
            .commit_prefix::<m![R % 128 / 64], _, _>(output.view_mut()
                .tile::<m![R / 128], 1, m![R / 128 = 1 #{!} (R / 128), { LocalRows<R> }, I % 224]>(wave), groups);
        current = next;
    }
    (output, current)
}

pub(super) fn sweep<H: AxisName, I: AxisName, R: AxisName>(
    device: &mut Device,
    gate: &Panels<H, I>,
    up: &Panels<H, I>,
    down: &Down<I, H>,
    normalized: &Owned<H, R>,
    rows: usize,
    entry: super::Entry<H, I>,
) -> (Ready<I, R>, Ready<I, R>, down::Ahead<I, H>) {
    const {
        assert!(I::SIZE / 64 == 224);
    }
    let (g0, u0, down0) = entry;
    let packed = if rows <= 64 {
        pack_rows::<H, R, 64>(device, normalized)
    } else if rows <= 128 {
        pack_rows::<H, R, 128>(device, normalized)
    } else {
        pack_rows::<H, R, 256>(device, normalized)
    };
    let input = if rows <= 64 {
        resident::<H, I, R, 64, 1>(device, &packed)
    } else if rows <= 128 {
        resident::<H, I, R, 128, 2>(device, &packed)
    } else if rows <= 256 {
        resident::<H, I, R, 256, 4>(device, &packed)
    } else {
        resident::<H, I, R, 512, 8>(device, &packed)
    };
    let restore =
        |device: &mut Device, packed: super::super::panels::Packed<H, I>| -> Weight<H, I> {
            let value = block::restore(device, packed.0, packed.1);
            // SAFETY: I / 64 is 224; ownership and element order are identical.
            unsafe { value.reshape() }
        };
    let (g0, u0) = (restore(device, g0), restore(device, u0));
    let mut b0 = Both::new();
    let first = lanes(device, &input, 0, 0, rows);
    let first = panel::<H, I, R, false>(device, &input, (&g0, &u0), None, &mut b0, first, 0, rows);
    // Issue the next packed slab after the current contractions, so it overlaps
    // their execution without staying resident through two BF16 restores.
    let g1 = gate.load(device, 1);
    let u1 = up.load(device, 1);
    let (g1, u1) = (restore(device, g1), restore(device, u1));
    let mut b1 = Both::new();
    let first = panel::<H, I, R, true>(
        device,
        &input,
        (&g1, &u1),
        Some(&b0),
        &mut b1,
        first,
        1,
        rows,
    );
    let g2 = gate.load(device, 2);
    let u2 = up.load(device, 2);
    let (g2, u2) = (restore(device, g2), restore(device, u2));
    let mut b2 = Both::new();
    let first = panel::<H, I, R, true>(
        device,
        &input,
        (&g2, &u2),
        Some(&b1),
        &mut b2,
        first,
        2,
        rows,
    );
    let g3 = gate.load(device, 3);
    let u3 = up.load(device, 3);
    let down1 = down.stage(device, 1);
    let g3 = restore(device, g3);
    let (g, first) = last(device, &input, &g3, &b2, 0, &gate.scales, first, rows);
    let u3 = restore(device, u3);
    let (u, _) = last(device, &input, &u3, &b2, 1, &up.scales, first, rows);
    (g, u, (down0, down1))
}
