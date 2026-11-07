use super::*;

pub(crate) fn compute_rows<Width: AxisName, C: M, P: M, T: M, E: M, N: Tiling>(
    context: &mut Device,
    input: (
        &DmTensor<bf16, Chip, C, P, m![{ T }, { E }]>,
        DmTensorView<'_, bf16, Chip, C, P, E>,
    ),
) -> DmTensor<bf16, Chip, C, P, m![{ T }, { E }]> {
    const { assert!(N::CAPACITY == capacity::<E>() && N::LAST == last::<T, E>()) };
    let gain: VrfTensor<f32, Chip, C, P, E> = context
        .sub
        .begin(input.1)
        .fetch::<m![{ E } / 8], m![{ E } % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![{ E } / 8], m![{ E } % 8]>()
        .to_vrf();
    let variance: DmTensor<f32, Chip, C, P, m![{ T }, 1 # 8]> = context
        .main
        .begin(input.0.view())
        .fetch::<m![{ T }, { E } / 8], m![{ E } % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![{ T }, { E } / 8], m![{ E } % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ T }, { E } / 4], m![{ E } % 4]>()
        .vector_fp_binary_with_mode(
            FpBinaryOp::MulF(FpMulAlu::Mul0),
            BinaryArgMode::Mode00,
            0.0f32,
        )
        .vector_intra_slice_reduce::<Width, T, m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(const { Width::SIZE as f32 })
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(
            ClipBinaryOpF32::Add,
            const { f32::from_bits(N::EPSILON as u32) },
        )
        .vector_final()
        .commit_trim::<m![1 # 2]>()
        .commit();
    N::finish::<C, P, T, E>(context, (input.0, &variance, &gain))
}
