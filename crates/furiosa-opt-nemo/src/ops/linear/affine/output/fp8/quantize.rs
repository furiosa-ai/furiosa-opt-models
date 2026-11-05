use super::*;

pub(super) fn quantize_first<Hidden: AxisName, QWidth: AxisName, Capacity: AxisName>(
    context: &mut Device,
    input: &Landing8<Hidden, QWidth, Capacity>,
    (quantized, scales): (
        &mut Source8<Hidden, QWidth, Capacity>,
        &mut Scales8<Hidden, QWidth, Capacity>,
    ),
) {
    let window = || {
        input.view().tile::<m![Capacity % (Capacity / 4) / 32], 1,
        m![Capacity % (Capacity / 4) / 32 = 1 # 4, Capacity % 32, QWidth % 128]>(0)
    };
    context.main.begin(window())
        .fetch::<m![Capacity % 32, QWidth % 128 / 8],
            m![QWidth % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Capacity % 32, QWidth % 128 / 8],
            m![QWidth % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Capacity % 32,
            QWidth % 128 / 4], m![QWidth % 4]>()
        .vector_fp_binary_with_mode(FpBinaryOp::MulF(FpMulAlu::Mul0), BinaryArgMode::Mode00, 0.0f32)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_intra_slice_reduce::<QWidth, m![Capacity % 32],
            m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_fp_div(LARGEST)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, SMALLEST)
        .vector_inter_slice_reduce::<m![Hidden / 1024, Capacity / (Capacity / 4), Replicas],
            m![Capacity % 32]>(InterSliceReduceOpF32::Max)
        .vector_final().cast::<bf16, m![1 # 16]>()
        .transpose::<m![Capacity % 32 / 4],
            m![Capacity % 4 # 16]>()
        .commit_trim::<m![Capacity % 4]>()
        .commit_view(scales.view_mut().tile::<m![Capacity % (Capacity / 4) / 32], 1,
                m![Capacity % (Capacity / 4) / 32 = 1 #{!} 4, Capacity % 32]>(0));
    // SAFETY: the reduction broadcasts each row's scale to every chunk's slice of its quarter.
    let broadcast = unsafe {
        scales
            .view()
            .reshape::<Chip, Cluster<QWidth>, Rowwise<Hidden, QWidth, Capacity>, m![Capacity % (Capacity / 4)]>()
    };
    let scale: VrfTensor<f32, Chip, Cluster<QWidth>, Rowwise<Hidden, QWidth, Capacity>,
        m![Capacity % 32]> =
        context.sub.begin(broadcast.tile::<m![Capacity % (Capacity / 4) / 32], 1,
            m![Capacity % (Capacity / 4) / 32 = 1 # 4, Capacity % 32]>(0))
        .fetch::<m![Capacity % 32 / 8], m![Capacity % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Capacity % 32 / 8], m![Capacity % 8]>()
        .to_vrf();
    context.main.begin(window())
        .fetch::<m![Capacity % 32, QWidth % 128 / 8],
            m![QWidth % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Capacity % 32, QWidth % 128 / 8],
            m![QWidth % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Capacity % 32,
            QWidth % 128 / 4], m![QWidth % 4]>()
        .vector_fp_div(&scale)
        .vector_widen_concat::<m![Capacity % 32,
            QWidth % 128 / 8], m![QWidth % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![QWidth % 8 # 32]>()
        .commit_trim::<m![QWidth % 8]>()
        .commit_view(quantized.view_mut().tile::<m![Capacity % (Capacity / 4) / 32], 1,
                m![Capacity % (Capacity / 4) / 32 = 1 #{!} 4, Capacity % 32,
                    QWidth % 128]>(0));
}

pub(super) fn quantize_all<Hidden: AxisName, QWidth: AxisName, Capacity: AxisName>(
    context: &mut Device,
    input: &Landing8<Hidden, QWidth, Capacity>,
    (quantized, scales): (
        &mut Source8<Hidden, QWidth, Capacity>,
        &mut Scales8<Hidden, QWidth, Capacity>,
    ),
    waves: usize,
) {
    context.main.begin(input.view())
        .fetch::<m![Capacity % (Capacity / 4), QWidth % 128 / 8],
            m![QWidth % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Capacity % (Capacity / 4), QWidth % 128 / 8],
            m![QWidth % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Capacity % (Capacity / 4),
            QWidth % 128 / 4], m![QWidth % 4]>()
        .vector_fp_binary_with_mode(FpBinaryOp::MulF(FpMulAlu::Mul0), BinaryArgMode::Mode00, 0.0f32)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_intra_slice_reduce::<QWidth, m![Capacity % (Capacity / 4)],
            m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_fp_div(LARGEST)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, SMALLEST)
        .vector_inter_slice_reduce::<m![Hidden / 1024, Capacity / (Capacity / 4), Replicas],
            m![Capacity % (Capacity / 4)]>(InterSliceReduceOpF32::Max)
        .vector_final().cast::<bf16, m![1 # 16]>()
        .transpose::<m![Capacity % (Capacity / 4) / 4],
            m![Capacity % 4 # 16]>()
        .commit_trim::<m![Capacity % 4]>()
        .commit_prefix::<m![Capacity % (Capacity / 4) / 4], _, _>(scales.view_mut(), 2 * waves);
    // SAFETY: the reduction broadcasts each row's scale to every chunk's slice of its quarter.
    let broadcast = unsafe {
        scales
            .view()
            .reshape::<Chip, Cluster<QWidth>, Rowwise<Hidden, QWidth, Capacity>, m![Capacity % (Capacity / 4)]>()
    };
    let scale: VrfTensor<
        f32,
        Chip,
        Cluster<QWidth>,
        Rowwise<Hidden, QWidth, Capacity>,
        m![Capacity % (Capacity / 4)],
    > = context
        .sub
        .begin(broadcast)
        .fetch::<m![Capacity % (Capacity / 4) / 8], m![Capacity % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Capacity % (Capacity / 4) / 8], m![Capacity % 8]>()
        .to_vrf();
    context
        .main
        .begin(input.view())
        .fetch::<m![Capacity % (Capacity / 4), QWidth % 128 / 8], m![QWidth % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Capacity % (Capacity / 4), QWidth % 128 / 8], m![QWidth % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Capacity % (Capacity / 4), QWidth % 128 / 4], m![QWidth % 4]>()
        .vector_fp_div(&scale)
        .vector_widen_concat::<m![Capacity % (Capacity / 4), QWidth % 128 / 8], m![QWidth % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![QWidth % 8 # 32]>()
        .commit_trim::<m![QWidth % 8]>()
        .commit_prefix::<m![Capacity % (Capacity / 4)], _, _>(quantized.view_mut(), 8 * waves);
}
