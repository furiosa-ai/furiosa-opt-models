use super::*;

pub(super) type Halves<Intermediate, Rows> = DmTensor<
    bf16,
    Chip,
    Cluster<Intermediate>,
    Slice<Intermediate, Rows>,
    m![
        Intermediate % 192 / 64,
        Rows % (Rows / 8) / 16,
        Rows % 16 / 8,
        Rows % 8,
        Intermediate % 64
    ],
>;

pub(super) fn halves<Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    operands: (
        &Projection<Intermediate, Rows>,
        &Projection<Intermediate, Rows>,
    ),
    rows: usize,
    output: &mut Halves<Intermediate, Rows>,
) {
    let product = product::multiply::<Intermediate, Rows>(context, operands, Some(rows));
    context
        .main
        .begin(product.view())
        .fetch::<m![
            Intermediate % 192 / 64,
            Rows % (Rows / 8) / 16,
            Rows % 16 / 8,
            Rows % 8,
            Intermediate % 64 / 16
        ], m![Intermediate % 16]>()
        .collect::<m![
            Intermediate % 192 / 64,
            Rows % (Rows / 8) / 16,
            Rows % 16 / 8,
            Rows % 8,
            Intermediate % 64 / 16
        ], m![Intermediate % 16]>()
        .commit_trim::<m![Intermediate % 16]>()
        .commit_view(output.view_mut());
}

pub(super) fn activate<Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    gate: &Projection<Intermediate, Rows>,
    rows: usize,
) -> Projection<Intermediate, Rows> {
    let mut output: Projection<Intermediate, Rows> = DmTensor::new();
    context
        .main
        .begin(gate.view())
        .fetch::<m![Rows % (Rows / 8), Intermediate % 192 / 8], m![Intermediate % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Rows % (Rows / 8), Intermediate % 192 / 8], m![Intermediate % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Rows % (Rows / 8), Intermediate % 192 / 4], m![Intermediate % 4]>(
        )
        .vector_stash()
        .vector_fp_unary(FpUnaryOp::Sigmoid)
        .vector_fp_binary_with_mode(
            FpBinaryOp::MulF(FpMulAlu::Mul0),
            BinaryArgMode::Mode10,
            Stash,
        )
        .vector_widen_concat::<m![Rows % (Rows / 8), Intermediate % 192 / 8], m![Intermediate % 8]>(
        )
        .vector_final()
        .cast::<bf16, m![Intermediate % 8 # 16]>()
        .commit_trim::<m![Intermediate % 8]>()
        .commit_prefix::<m![Rows % (Rows / 8)], _, _>(output.view_mut(), (rows + 63) / 64 * 8);
    output
}

pub(super) fn halves_panel<Intermediate: AxisName, Rows: AxisName, const P: usize>(
    context: &mut Device,
    (activated, up): (
        &Projection<Intermediate, Rows>,
        &Projection<Intermediate, Rows>,
    ),
    output: &mut Halves<Intermediate, Rows>,
) {
    context.main.begin_interleaved::<Term, _, _, _, _, _>(
        activated.view().tile::<m![Intermediate % 192 / 64], 1,
            m![Rows % (Rows / 8), Intermediate % 64 # 192]>(P),
        up.view().tile::<m![Intermediate % 192 / 64], 1,
            m![Rows % (Rows / 8), Intermediate % 64 # 192]>(P))
        .fetch::<m![Rows % (Rows / 8), Term, Intermediate % 64 / 8], m![Intermediate % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Rows % (Rows / 8), Term, Intermediate % 64 / 8], m![Intermediate % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Term, m![Rows % (Rows / 8), Intermediate % 64 / 8]>()
        .vector_narrow_split::<m![Rows % (Rows / 8), Intermediate % 64 / 4], m![Intermediate % 4]>()
        .vector_fp_zip(FpBinaryOp::MulF(FpMulAlu::Fma))
        .vector_widen_concat::<m![Rows % (Rows / 8), Intermediate % 64 / 8], m![Intermediate % 8]>()
        .vector_final().cast::<bf16, m![Intermediate % 8 # 16]>()
        .commit_trim::<m![Intermediate % 8]>()
        .commit_view(output.view_mut().tile::<m![Intermediate % 192 / 64], 1,
            m![Intermediate % 192 / 64 = 1 #{!} 3, Rows % (Rows / 8) / 16,
                Rows % 16 / 8, Rows % 8, Intermediate % 64]>(P));
}

fn panel_scales<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    product: &Product<Hidden, Intermediate, Rows>,
    (rows, scales): (usize, &mut Scale<Hidden, Intermediate, Rows>),
) {
    // SAFETY: both mappings name every slice of the cluster; the reduction writes each one.
    let output = unsafe {
        scales
            .view_mut()
            .reshape::<Chip, Cluster<Intermediate>, m![Hidden / 512, Copies], m![Rows]>()
    };
    context
        .main
        .begin(product.view())
        .fetch::<m![Rows, Intermediate % 64 / 8], m![Intermediate % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Rows, Intermediate % 64 / 8], m![Intermediate % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Rows, Intermediate % 64 / 4], m![Intermediate % 4]>()
        .vector_fp_binary_with_mode(
            FpBinaryOp::MulF(FpMulAlu::Mul0),
            BinaryArgMode::Mode00,
            0.0f32,
        )
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_intra_slice_reduce::<Intermediate, m![Rows], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_fp_div(LARGEST)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, SMALLEST)
        .vector_inter_slice_reduce::<m![Hidden / 512, Copies], m![Rows]>(InterSliceReduceOpF32::Max)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Rows / 4], m![Rows % 4 # 16]>()
        .commit_trim::<m![Rows % 4]>()
        .commit_prefix::<m![Rows / 4], _, _>(output, rows.div_ceil(4));
}

pub(super) fn quantize_panel<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    product: &Product<Hidden, Intermediate, Rows>,
    rows: usize,
    (quantized, scales): (
        &mut Activation<Hidden, Intermediate, Rows>,
        &mut Scale<Hidden, Intermediate, Rows>,
    ),
) {
    panel_scales::<Hidden, Intermediate, Rows>(context, product, (rows, &mut *scales));
    // SAFETY: every slice of a cluster holds the same scales.
    let broadcast = unsafe {
        scales
            .view()
            .reshape::<Chip, Cluster<Intermediate>, DownSlice<Hidden, Intermediate>, m![Rows]>()
    };
    let scale: VrfTensor<
        f32,
        Chip,
        Cluster<Intermediate>,
        DownSlice<Hidden, Intermediate>,
        m![Rows],
    > = context
        .sub
        .begin(broadcast)
        .fetch::<m![Rows / 8], m![Rows % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Rows / 8], m![Rows % 8]>()
        .to_vrf();
    context
        .main
        .begin(product.view())
        .fetch::<m![Rows, Intermediate % 64 / 8], m![Intermediate % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Rows, Intermediate % 64 / 8], m![Intermediate % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Rows, Intermediate % 64 / 4], m![Intermediate % 4]>()
        .vector_fp_div(&scale)
        .vector_widen_concat::<m![Rows, Intermediate % 64 / 8], m![Intermediate % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Intermediate % 8 # 32]>()
        .commit_trim::<m![Intermediate % 8]>()
        .commit_prefix::<m![Rows], _, _>(quantized.view_mut(), rows);
}
