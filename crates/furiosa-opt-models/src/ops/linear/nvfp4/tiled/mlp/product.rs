use super::*;
use crate::ops::vector::{Silu as _, Zip as _};

pub(super) fn multiply<C: M, S: M, Time: M, Feature: M>(
    device: &mut Device,
    gate: DmTensorView<'_, bf16, Chip, C, S, m![{ Time }, { Feature }]>,
    up: DmTensorView<'_, bf16, Chip, C, S, m![{ Time }, { Feature }]>,
) -> DmTensor<bf16, Chip, C, S, m![{ Time }, { Feature }]> {
    axes![Term = 2];
    // Preserve the reference's BF16 rounding between SiLU and its product with up.
    let activated: DmTensor<bf16, Chip, C, S, m![{ Time }, { Feature }]> = device
        .main
        .begin(gate)
        .fetch::<m![{ Time }, { Feature } / 8], m![{ Feature } % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![{ Time }, { Feature } / 8], m![{ Feature } % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ Time }, { Feature } / 4], m![{ Feature } % 4]>()
        .vector_silu()
        .vector_widen_concat::<m![{ Time }, { Feature } / 8], m![{ Feature } % 8]>()
        .vector_final()
        .cast::<bf16, m![{Feature} % 8 # 16]>()
        .commit_trim::<m![{ Feature } % 8]>()
        .commit();
    device
        .main
        .begin_interleaved::<Term, _, _, _, _, _>(activated.view(), up)
        .fetch::<m![{ Time }, Term, { Feature } / 8], m![{ Feature } % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![{ Time }, Term, { Feature } / 8], m![{ Feature } % 8]>()
        .vector_init()
        .vector_zip::<Term, m![{ Time }, { Feature } / 8], m![{ Time }, { Feature } / 4], m![{ Feature } % 4]>(
            FpBinaryOp::MulF(FpMulAlu::Fma),
        )
        .vector_final()
        .cast::<bf16, m![{Feature} % 8 # 16]>()
        .commit_trim::<m![{ Feature } % 8]>()
        .commit()
}

pub(super) fn stage<I: AxisName, R: AxisName>(
    device: &mut Device,
    g: &Ready<I, R>,
    u: &Ready<I, R>,
    rows: usize,
    output: &mut Products<I, R>,
) {
    for wave in 0..rows.div_ceil(128) {
        let gate = g
            .view()
            .tile::<m![R / 128], 1, m![R / 128 = 1 # (R / 128), { LocalRows<R> }, I % 224]>(wave)
            .unpad::<m![{ LocalRows<R> }, I % 224]>();
        let up = u
            .view()
            .tile::<m![R / 128], 1, m![R / 128 = 1 # (R / 128), { LocalRows<R> }, I % 224]>(wave)
            .unpad::<m![{ LocalRows<R> }, I % 224]>();
        let product = multiply::<_, _, LocalRows<R>, m![I % 224]>(device, gate, up);
        // Seven 32-feature panels cover each 224-feature output owner exactly.
        // Transpose once on-chip, then each slice writes 1 KiB contiguous runs.
        let packed: DmTensor<
            bf16,
            Chip,
            Cluster<I>,
            Slices<I, R>,
            m![I % 224 / 32, { LocalRows<R> }, I % 32],
        > = device
            .sub
            .begin(product.view())
            .fetch::<m![I % 224 / 32, { LocalRows<R> }, I % 32 / 16], m![I % 16]>()
            .collect::<m![I % 224 / 32, { LocalRows<R> }, I % 32 / 16], m![I % 16]>()
            .commit_trim::<m![I % 16]>()
            .commit();
        packed.view().to_hbm_view(&mut device.tdma, output.view_mut()
            .tile::<m![R / 128], 1, m![I / 224, I % 224 / 32, R / 128 = 1 #{!} (R / 128), R % 128, I % 32]>(wave));
    }
}

/// Preserve the compact SwiGLU product as BF16 across the down layout bridge.
pub(super) fn compact<I: AxisName, R: AxisName, const W: usize>(
    device: &mut Device,
    g: &super::super::compact::Output<I, R, W>,
    u: &super::super::compact::Output<I, R, W>,
    output: &mut Products<I, R>,
) {
    let values = multiply::<_, _, m![R % { W }], m![I % (I / 64)]>(device, g.view(), u.view());
    // SAFETY: rename the identical 64 feature owners and their reduction slices.
    let values = unsafe {
        values.reshape::<Chip, Cluster<I>, m![{Owners<I>}, 1 # 8], m![R % { W }, I % 224]>()
    };
    let packed: DmTensor<
        bf16,
        Chip,
        Cluster<I>,
        m![{Owners<I>}, 1 # 8],
        m![I % 224 / 32, R % { W }, I % 32],
    > = device
        .sub
        .begin(values.view())
        .fetch::<m![I % 224 / 32, R % { W }, I % 32 / 16], m![I % 16]>()
        .collect::<m![I % 224 / 32, R % { W }, I % 32 / 16], m![I % 16]>()
        .commit_trim::<m![I % 16]>()
        .commit();
    packed.view().to_hbm_view(
        &mut device.tdma,
        output
            .view_mut()
            .tile::<m![R], W, m![I / 224, I % 224 / 32, R = {W} #{!} R, I % 32]>(0),
    );
}
