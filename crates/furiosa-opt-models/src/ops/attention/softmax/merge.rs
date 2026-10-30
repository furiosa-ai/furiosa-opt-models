//! The log-sum-exp merge: attentions over disjoint keys, each normalized over its own keys and
//! carrying its log-sum-exp `lse`, combined into the attention over all of their keys.
//!
//! - A row's summands `S` merge into `sum(e^(lse - top) o) / (sum(e^(lse - top)) + eps)`, `top`
//!   their largest `lse` and `eps` the regularizer, which keeps an empty row's divisor positive.
//! - The summand axis lies partly in a slice's time, `SliceTime`, the intra-slice reducer's part,
//!   and partly over the slices a row's group spans, the inter-slice reducer's part.
//! - A summand of log-sum-exp negative infinity has no keys and no share; its partial may be any
//!   bits, so a non-finite partial is cleared before it is weighted, as zero times NaN is NaN.

use furiosa_opt_std::prelude::*;

/// The tag of a stream whose first comparison failed.
const FIRST: TagGuard =
    TagGuard::matches([BitReq::Zero, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]);
/// The tag of a stream whose second comparison failed.
const SECOND: TagGuard =
    TagGuard::matches([BitReq::Ignore, BitReq::Zero, BitReq::Ignore, BitReq::Ignore]);

/// Each row `R` of `partials` merged over its summands `S` onto the `Merged` slices, statistics
/// agreed over a group's `Agreed` slices, each slice adding `regularizer` to its group's divisor.
pub(crate) fn merge<
    S: AxisName,
    SliceTime: M,
    R: M,
    Head: M,
    Agreed: M,
    Chip: M,
    Cluster: M,
    Slices: M,
    Merged: M,
    P: M,
    L: M,
    O: M,
>(
    device: &mut Device,
    (partials, lses): (
        DmTensorView<'_, f32, Chip, Cluster, Slices, P>,
        DmTensorView<'_, f32, Chip, Cluster, Slices, L>,
    ),
    regularizer: f32,
) -> DmTensor<bf16, Chip, Cluster, Merged, O> {
    let top: VrfTensor<f32, Chip, Cluster, Agreed, m![{ R }, 1 # 8]> = device
        .main
        .begin(lses.clone())
        .fetch::<m![{ R }, { SliceTime }], m![1 # 8]>()
        .collect::<m![{ R }, { SliceTime }], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<S, R, m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<Agreed, R>(InterSliceReduceOpF32::Max)
        .vector_final()
        .to_vrf(&mut device.sub);
    // SAFETY: the collective gives each of a group's slices the group's largest lse.
    let top: VrfTensor<f32, Chip, Cluster, Slices, m![{ R }, 1 # 8]> = unsafe { top.reshape() };
    let total: VrfTensor<f32, Chip, Cluster, Agreed, m![{ R }, 1 # 8]> = device
        .main
        .begin(lses.clone())
        .fetch::<m![{ R }, { SliceTime }], m![1 # 8]>()
        .collect::<m![{ R }, { SliceTime }], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Comparison([Cmp::Equal(f32::NEG_INFINITY); 4]))
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, Branched::rf(FIRST, &top))
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_intra_slice_reduce::<S, R, m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, regularizer)
        .vector_inter_slice_reduce::<Agreed, R>(InterSliceReduceOpF32::Add)
        .vector_final()
        .to_vrf(&mut device.sub);
    // SAFETY: the collective gives each of a group's slices the group's divisor.
    let total: VrfTensor<f32, Chip, Cluster, Slices, m![{ R }, 1 # 8]> = unsafe { total.reshape() };
    let shares: VrfTensor<f32, Chip, Cluster, Slices, m![{ SliceTime }, { R }, 1 # 8]> = device
        .main
        .begin(lses)
        .fetch::<m![{ SliceTime }, { R }], m![1 # 8]>()
        .collect::<m![{ SliceTime }, { R }], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Comparison([Cmp::Equal(f32::NEG_INFINITY); 4]))
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, Branched::rf(FIRST, &top))
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_fp_div(&total)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .to_vrf(&mut device.sub);
    let weighted: DmTensor<f32, Chip, Cluster, Slices, m![{ SliceTime }, { R }, { Head }]> = device
        .main
        .begin(partials)
        .fetch::<m![{ SliceTime }, { R }, { Head } / 8], m![{ Head } % 8]>()
        .collect::<m![{ SliceTime }, { R }, { Head } / 8], m![{ Head } % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Comparison([
            Cmp::Less(f32::INFINITY),
            Cmp::Greater(f32::NEG_INFINITY),
            Cmp::False,
            Cmp::False,
        ]))
        .vector_logic(
            LogicBinaryOpF32::BitAnd,
            Branched::imm(FIRST, 0.0f32).imm(SECOND, 0.0f32),
        )
        .vector_narrow_split::<m![{ SliceTime }, { R }, { Head } / 4], m![{ Head } % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &shares)
        .vector_widen_concat::<m![{ SliceTime }, { R }, { Head } / 8], m![{ Head } % 8]>()
        .vector_final()
        .commit_trim::<m![{ Head } % 8]>()
        .commit();
    device
        .main
        .begin(weighted.view())
        .fetch::<m![{ R }, { Head } / 8, { SliceTime }], m![{ Head } % 8]>()
        .collect::<m![{ R }, { Head } / 8, { SliceTime }], m![{ Head } % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<
            m![{ R }, { Head } / 8, { SliceTime }, { Head } % 8 / 4],
            m![{ Head } % 4],
        >(
        )
        .vector_intra_slice_reduce::<S, m![{ R }, { Head } / 4], m![{ Head } % 4]>(
            IntraSliceReduceOpF32::Add,
        )
        .vector_widen_concat::<m![{ R }, { Head } / 8], m![{ Head } % 8]>()
        .vector_inter_slice_reduce::<Merged, m![{ R }, { Head } / 8]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![{ Head } % 8 # 16]>()
        .commit_trim::<m![{ Head } % 8]>()
        .commit()
}
