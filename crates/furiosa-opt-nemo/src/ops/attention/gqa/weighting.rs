use super::{PagedAttention, completion};
use crate::Module;
use crate::ops::layout::Dm;
use furiosa_opt_std::prelude::*;

const CLEARED: TagGuard =
    TagGuard::matches([BitReq::Zero, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]);
const SECOND_CLEARED: TagGuard =
    TagGuard::matches([BitReq::Ignore, BitReq::Zero, BitReq::Ignore, BitReq::Ignore]);

impl<
    'x,
    H: AxisName,
    G: AxisName,
    D,
    const SCALE: usize,
    B: M,
    C: M,
    L: M,
    E: M,
    S: completion::Mapping,
>
    Module<(
        &mut Device,
        Dm<
            S,
            (
                DmTensorView<'x, f32, B, C, L, E>,
                (
                    &'x VrfTensor<i4, B, C, L, m![H, G, 1 # 8]>,
                    &'x VrfTensor<f32, B, C, L, m![H, G, 1 # 8]>,
                ),
            ),
        >,
    )> for PagedAttention<H, G, D, SCALE, B>
{
    type Output = Dm<S, DmTensor<f32, B, C, L, m![H, G, 1 # 8]>>;

    fn forward(
        &self,
        (device, input): (
            &mut Device,
            Dm<
                S,
                (
                    DmTensorView<'x, f32, B, C, L, E>,
                    (
                        &'x VrfTensor<i4, B, C, L, m![H, G, 1 # 8]>,
                        &'x VrfTensor<f32, B, C, L, m![H, G, 1 # 8]>,
                    ),
                ),
            >,
        ),
    ) -> Self::Output {
        let (stats, (tags, scales)) = input.value;
        Dm {
            tiles: input.tiles,
            value: device
                .main
                .begin(stats)
                .fetch::<m![H, G], m![1 # 8]>()
                .collect::<m![H, G], m![1 # 8]>()
                .vector_init()
                .vector_tagged(tags)
                .vector_logic(LogicBinaryOpF32::BitAnd, Branched::imm(CLEARED, 0.0f32))
                .vector_narrow_trim::<m![1 # 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scales)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit(),
        }
    }
}

impl<
    'x,
    H: AxisName,
    G: AxisName,
    D: AxisName,
    const SCALE: usize,
    B: M,
    C: M,
    L: M,
    S: completion::Mapping,
>
    Module<(
        &mut Device,
        Dm<
            S,
            (
                DmTensorView<'x, bf16, B, C, L, m![H, G, D]>,
                &'x VrfTensor<f32, B, C, L, m![H, G, 1 # 8]>,
            ),
        >,
    )> for PagedAttention<H, G, D, SCALE, B>
{
    type Output = Dm<S, DmTensor<f32, B, C, L, m![H, G, D]>>;

    fn forward(
        &self,
        (device, input): (
            &mut Device,
            Dm<
                S,
                (
                    DmTensorView<'x, bf16, B, C, L, m![H, G, D]>,
                    &'x VrfTensor<f32, B, C, L, m![H, G, 1 # 8]>,
                ),
            >,
        ),
    ) -> Self::Output {
        let (values, scales) = input.value;
        Dm {
            tiles: input.tiles,
            value: device
                .main
                .begin(values)
                .fetch::<m![H, G, D / 8], m![D % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![H, G, D / 8], m![D % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Comparison([
                    Cmp::Less(f32::INFINITY),
                    Cmp::Greater(f32::NEG_INFINITY),
                    Cmp::False,
                    Cmp::False,
                ]))
                .vector_logic(
                    LogicBinaryOpF32::BitAnd,
                    Branched::imm(CLEARED, 0.0f32).imm(SECOND_CLEARED, 0.0f32),
                )
                .vector_narrow_split::<m![H, G, D / 4], m![D % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scales)
                .vector_widen_concat::<m![H, G, D / 8], m![D % 8]>()
                .vector_final()
                .commit_trim::<m![D % 8]>()
                .commit(),
        }
    }
}
