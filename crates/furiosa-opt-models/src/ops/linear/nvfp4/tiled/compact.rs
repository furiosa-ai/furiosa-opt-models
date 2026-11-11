//! Eight-/sixteen-row contractions retaining the four 1024-feature input slabs.
//! Reduction keeps one owner per eight input slices instead of distributing 64 rows.
use crate::ops::normalization::span::Owned;
use crate::tensor::parallel::Chip;
use furiosa_opt_std::prelude::*;

axes![Owner = 64, Term = 2];
pub(super) type Cluster = m![Owner / 32];
pub(super) type Slices<H> = m![Owner % 32, H % 1024 / 128];
pub(super) type Reduced = m![Owner % 32, 1 # 8];
pub(super) type Input<H, R, const W: usize> =
    DmTensor<bf16, Chip, Cluster, Slices<H>, m![H / 1024, R % { W }, H % 128]>;
pub(super) type Lanes<H, R, const W: usize> = TrfTensor<
    bf16,
    Chip,
    Cluster,
    Slices<H>,
    m![R % 8],
    m![R % { W } / 8, H % 32 / 16, H % 128 / 32, H % 16],
>;
pub(super) type Partial<O, R, const W: usize> =
    DmTensor<f32, Chip, Cluster, Reduced, m![R % { W }, O % (O / 64)]>;
pub(super) type Paired<O, R, const W: usize> =
    DmTensor<f32, Chip, Cluster, Reduced, m![R % { W }, Term, O % (O / 64)]>;
pub(super) type Output<O, R, const W: usize> =
    DmTensor<bf16, Chip, Cluster, Reduced, m![R % { W }, O % (O / 64)]>;

type NormRows<R, const W: usize> = m![R % { W } / 8];

pub(super) fn input<H: AxisName, R: AxisName, const W: usize>(
    device: &mut Device,
    input: &Owned<H, R>,
) -> Input<H, R, W> {
    const {
        assert!(matches!(W, 8 | 16));
    }
    // Borrow the live normalized BF16 rows; no quantizer buffer or copy is needed.
    // SAFETY: rename the selected row prefix, retaining cluster/slice ownership.
    let values = unsafe {
        input
            .view()
            .tile::<m![R % (R / 2)], W, m![R % (R / 2) = {W} # (R / 2), H % 16]>(0)
            .unpad::<m![R % (R / 2) = { W }, H % 16]>()
            .reshape::<Chip, m![R / (R / 2)], m![H / 16], m![R % { W }, H % 16]>()
    };
    let packed: DmTensor<bf16, Chip, m![R / (R / 2)], m![H / 128, R % {W} / ({ NormRows<R, W> })],
        m![R % ({ NormRows<R, W> }), H % 128]> = device.sub.begin(values)
        .fetch::<m![R % ({ NormRows<R, W> }), R % {W} / ({ NormRows<R, W> })], m![H % 16]>()
        .switch::<m![H / 128, R % {W} / ({ NormRows<R, W> })], m![R % ({ NormRows<R, W> }), H % 128 / 16]>(
            SwitchConfig::InterTranspose { slice1: 8, slice0: 1, time0: 1 })
        .collect::<m![R % ({ NormRows<R, W> }), H % 128 / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>().commit();
    let mut stored: HbmTensor<bf16, Chip, m![H / 128, R, H % 128]> = HbmTensor::new();
    packed.view().to_hbm_view(&mut device.tdma, stored.view_mut()
        .tile::<m![R % (R / 2)], W, m![H / 128, R / (R / 2), R % (R / 2) = {W} #{!} (R / 2), H % 128]>(0));
    stored
        .view()
        .tile::<m![R], W, m![H / 128, R = {W} # R, H % 128]>(0)
        .to_dm(&mut device.tdma)
}

pub(super) fn lanes<H: AxisName, R: AxisName, const W: usize>(
    device: &mut Device,
    input: &Input<H, R, W>,
    panel: usize,
) -> Lanes<H, R, W> {
    device
        .sub
        .begin(
            input
                .view()
                .tile::<m![H / 1024], 1, m![H / 1024 = 1 # 4, R % {W}, H % 128]>(panel),
        )
        .fetch::<m![R % 8, R % { W } / 8, H % 32 / 16, H % 128 / 32], m![H % 16]>()
        .collect::<m![R % 8, R % { W } / 8, H % 32 / 16, H % 128 / 32], m![H % 16]>()
        .to_trf()
}

pub(super) fn single<H: AxisName, O: AxisName, R: AxisName, const W: usize, C: M, S: M, E: M>(
    device: &mut Device,
    weight: DmTensorView<'_, bf16, Chip, C, S, E>,
    lanes: &Lanes<H, R, W>,
) -> Partial<O, R, W> {
    // SAFETY: slab weights retain 64 output owners and eight 128-feature input parts.
    let weight = unsafe { weight.reshape::<Chip, Cluster, Slices<H>, m![O % (O / 64), H % 128]>() };
    device.main.begin(weight)
        .fetch::<m![R % {W} / 8, O % (O / 64) / 2, H % 32 / 16, H % 128 / 32, O % 2], m![H % 16]>()
        .collect::<m![R % {W} / 8, O % (O / 64) / 2, H % 32 / 16, H % 128 / 32, O % 2], m![H % 16]>()
        .contract_outer::<m![R % {W} / 8, O % (O / 64) / 2, H % 32 / 16, H % 128 / 32], m![O % 2, H % 16], _, _, _>(lanes)
        .contract_packet::<m![O % 2]>()
        .contract_time::<m![R % {W} / 8, O % (O / 64) / 2]>()
        .contract_lane::<m![R % {W} / 8, O % (O / 64) / 2, O % 2], m![R % 8]>(LaneMode::Interleaved)
        .vector_init().vector_inter_slice_reduce::<Reduced, m![R % {W} / 8, O % (O / 64)]>(InterSliceReduceOpF32::Add)
        .vector_final().transpose::<m![R % {W} / 8, O % (O / 64) / 2, R % 8], m![O % 2 # 8]>()
        .commit_trim::<m![O % 2]>().commit()
}

pub(super) fn pair<H: AxisName, O: AxisName, R: AxisName, const W: usize, C: M, S: M, E: M>(
    device: &mut Device,
    gate: DmTensorView<'_, bf16, Chip, C, S, E>,
    up: DmTensorView<'_, bf16, Chip, C, S, E>,
    lanes: &Lanes<H, R, W>,
) -> Paired<O, R, W> {
    let gate = unsafe { gate.reshape::<Chip, Cluster, Slices<H>, m![O % (O / 64), H % 128]>() };
    let up = unsafe { up.reshape::<Chip, Cluster, Slices<H>, m![O % (O / 64), H % 128]>() };
    device
        .main
        .begin_interleaved::<Term, _, _, _, _, _>(gate, up)
        .fetch::<m![
            R % { W } / 8,
            Term,
            O % (O / 64) / 2,
            H % 32 / 16,
            H % 128 / 32,
            O % 2
        ], m![H % 16]>()
        .collect::<m![
            R % { W } / 8,
            Term,
            O % (O / 64) / 2,
            H % 32 / 16,
            H % 128 / 32,
            O % 2
        ], m![H % 16]>()
        .contract_outer::<m![
            R % { W } / 8,
            Term,
            O % (O / 64) / 2,
            H % 32 / 16,
            H % 128 / 32
        ], m![O % 2, H % 16], _, _, _>(lanes)
        .contract_packet::<m![O % 2]>()
        .contract_time::<m![R % { W } / 8, Term, O % (O / 64) / 2]>()
        .contract_lane::<m![R % { W } / 8, Term, O % (O / 64) / 2, O % 2], m![R % 8]>(
            LaneMode::Interleaved,
        )
        .vector_init()
        .vector_inter_slice_reduce::<Reduced, m![R % { W } / 8, Term, O % (O / 64)]>(
            InterSliceReduceOpF32::Add,
        )
        .vector_final()
        .transpose::<m![R % { W } / 8, Term, O % (O / 64) / 2, R % 8], m![O % 2 # 8]>()
        .commit_trim::<m![O % 2]>()
        .commit()
}

pub(super) fn term<O: AxisName, R: AxisName, const W: usize>(
    device: &mut Device,
    pair: &Paired<O, R, W>,
    term: usize,
) -> Partial<O, R, W> {
    let mut result = Partial::new();
    pair.view()
        .tile::<m![Term], 1, m![R % {W}, Term = 1 # 2, O % (O / 64)]>(term)
        .to_dm_view(&mut device.tdma, result.view_mut());
    result
}
