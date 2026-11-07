use super::*;

pub(crate) fn multiply<Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    operands: (
        &Projection<Intermediate, Rows>,
        &Projection<Intermediate, Rows>,
    ),
    live: Option<usize>,
) -> Projection<Intermediate, Rows> {
    macro_rules! commit {
        ($chain:expr) => {
            match live {
                None => $chain.commit(),
                Some(rows) => {
                    let mut output: Projection<Intermediate, Rows> = DmTensor::new();
                    $chain.commit_prefix::<m![Rows % (Rows / 8)], _, _>(
                        output.view_mut(),
                        (rows + 63) / 64 * 8,
                    );
                    output
                }
            }
        };
    }
    let activated: Projection<Intermediate, Rows> = commit!(context.main.begin(operands.0.view())
        .fetch::<m![Rows % (Rows / 8), Intermediate % 192 / 8], m![Intermediate % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Rows % (Rows / 8), Intermediate % 192 / 8], m![Intermediate % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Rows % (Rows / 8), Intermediate % 192 / 4], m![Intermediate % 4]>()
        .vector_stash().vector_fp_unary(FpUnaryOp::Sigmoid)
        .vector_fp_binary_with_mode(FpBinaryOp::MulF(FpMulAlu::Mul0), BinaryArgMode::Mode10, Stash)
        .vector_widen_concat::<m![Rows % (Rows / 8), Intermediate % 192 / 8], m![Intermediate % 8]>()
        .vector_final().cast::<bf16, m![Intermediate % 8 # 16]>()
        .commit_trim::<m![Intermediate % 8]>());
    commit!(context.main
        .begin_interleaved::<Term, _, _, _, _, _>(activated.view(), operands.1.view())
        .fetch::<m![Rows % (Rows / 8), Term, Intermediate % 192 / 8], m![Intermediate % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Rows % (Rows / 8), Term, Intermediate % 192 / 8], m![Intermediate % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Term, m![Rows % (Rows / 8), Intermediate % 192 / 8]>()
        .vector_narrow_split::<m![Rows % (Rows / 8), Intermediate % 192 / 4], m![Intermediate % 4]>()
        .vector_fp_zip(FpBinaryOp::MulF(FpMulAlu::Fma))
        .vector_widen_concat::<m![Rows % (Rows / 8), Intermediate % 192 / 8], m![Intermediate % 8]>()
        .vector_final().cast::<bf16, m![Intermediate % 8 # 16]>()
        .commit_trim::<m![Intermediate % 8]>())
}
