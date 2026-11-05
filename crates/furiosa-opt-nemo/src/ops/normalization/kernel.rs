use crate::ops::normalization::{Chunk, Cluster, FeatureSlices, Learned, Spread};
use crate::ops::vector::{MeanSquare, scalar};
use furiosa_opt_std::prelude::*;

type Reduced = m![256];

/// A norm's rows as the stream its caller commits.
pub(super) type Commit<'l, Width, Rows, Chips> = CommitTrimTensor<
    'l,
    { Tu::Main },
    bf16,
    Chips,
    Cluster,
    FeatureSlices<Width>,
    m![{ Rows }, Width % ({ Symbol<Chunk<Width>> }) / 8],
    m![Width % 8],
>;

/// Each of the `Rows` rows of every feature slice divided by its root mean square and scaled by
/// `gain`, whose lanes land once the inverse is in the vector register file.
pub(super) fn rms_norm<'l, Width: AxisName, Rows: M, const EPSILON: usize, Chips: M, E: M>(
    device: &'l mut Device,
    input: DmTensorView<'l, bf16, Chips, Cluster, FeatureSlices<Width>, E>,
    gain: &impl Learned<Width, Chips>,
) -> Commit<'l, Width, Rows, Chips> {
    let variance: DmTensor<f32, Chips, Cluster, Reduced, m![{ Rows }, 1 # 8]> = device
        .main
        .begin(input.clone())
        .fetch::<m![{ Rows }, Width % ({ Symbol<Chunk<Width>> }) / 8], m![Width % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![{ Rows }, Width % ({ Symbol<Chunk<Width>> }) / 8], m![Width % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ Rows }, Width % ({ Symbol<Chunk<Width>> }) / 8, Width % 8 / 4 % 2], m![Width % 4]>()
        .vector_mean_square::<Width, Rows, m![Width / ({ Symbol<Chunk<Width>> })], EPSILON>()
        .vector_inter_slice_reduce::<Reduced, Rows>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // Every slice holds the same row scalar; each row keeps its original address.
    let inverse: DmTensor<f32, Chips, Cluster, FeatureSlices<Width>, m![{ Rows }, 1 # 8]> =
        unsafe { variance.reshape() };
    let inverse: VrfTensor<f32, Chips, Cluster, FeatureSlices<Width>, m![{ Rows }, 1 # 8]> = device
        .sub
        .begin(inverse.view())
        .fetch::<Rows, m![1 # 8]>()
        .fetch_cast::<f32>()
        .collect::<Rows, m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_fp_div_with_mode(BinaryArgMode::Mode10, 1.0f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .to_vrf();
    // The learned scale is shared by every row.
    let scale = gain.lanes(device);
    device
        .main
        .begin(input)
        .fetch::<m![{ Rows }, Width % ({ Symbol<Chunk<Width>> }) / 8], m![Width % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![{ Rows }, Width % ({ Symbol<Chunk<Width>> }) / 8], m![Width % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ Rows }, Width % ({ Symbol<Chunk<Width>> }) / 8, Width % 8 / 4 % 2], m![Width % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &inverse)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &scale)
        .vector_widen_concat::<m![{ Rows }, Width % ({ Symbol<Chunk<Width>> }) / 8], m![Width % 8]>()
        .vector_final()
        .cast::<bf16, m![Width % 8 # 16]>()
        .commit_trim::<m![Width % 8]>()
}

pub(super) fn heads<S: M, P: M, Head: AxisName, const EPSILON: usize>(
    device: &mut Device,
    input: &DmTensor<bf16, m![1], m![1 # 2], S, m![{ P }, Head]>,
    weight: &HbmTensor<bf16, m![1], m![Head]>,
) -> DmTensor<bf16, m![1], m![1 # 2], S, m![{ P }, Head]> {
    let mean: DmTensor<f32, m![1], m![1 # 2], S, m![{ P }, 1 # 8]> = device
        .main
        .begin(input.view())
        .fetch::<m![{ P }, Head / 16], m![Head % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![{ P }, Head / 8], m![Head % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ P }, Head / 4], m![Head % 4]>()
        .vector_mean_square::<Head, m![{ P }], m![1], EPSILON>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: VrfTensor<f32, m![1], m![1 # 2], S, m![{ P }, 1 # 8]> = device
        .sub
        .begin(mean.view())
        .fetch::<m![{ P }], m![1 # 8]>()
        .collect::<m![{ P }], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .to_vrf();
    let weight: DmTensor<bf16, m![1], m![1 # 2], S, m![Head]> = weight.to_dm(&mut device.tdma);
    let weight: VrfTensor<f32, m![1], m![1 # 2], S, m![Head]> = device
        .sub
        .begin(weight.view())
        .fetch::<m![Head / 16], m![Head % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Head / 8], m![Head % 8]>()
        .to_vrf();
    device
        .main
        .begin(input.view())
        .fetch::<m![{ P }, Head / 16], m![Head % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![{ P }, Head / 8], m![Head % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ P }, Head / 4], m![Head % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight)
        .vector_fp_div(&rms)
        .vector_widen_concat::<m![{ P }, Head / 8], m![Head % 8]>()
        .vector_final()
        .cast::<bf16, m![Head % 8 # 16]>()
        .commit_trim::<m![Head % 8]>()
        .commit()
}

/// Per-feature FP32 operands of the normalized rows' pass.
pub(super) type Features<Width, Chips> =
    VrfTensor<f32, Chips, Cluster, FeatureSlices<Width>, m![Width % ({ Symbol<Chunk<Width>> })]>;

/// The stream shape a gain multiplies in: each row's features, four lanes at a time.
type Normalized<Width, Rows, Chips> = VeTensorShape<
    Chips,
    Cluster,
    FeatureSlices<Width>,
    m![Rows, Width % ({ Symbol<Chunk<Width>> }) / 8, Width % 8 / 4 % 2],
    m![Width % 4],
>;

/// The FP32 gain `1 + offset` of every feature, formed before any rounding.
pub(super) fn centered<Width: AxisName, Chips: M>(
    device: &mut Device,
    offset: &HbmTensor<bf16, Chips, m![Width]>,
) -> Features<Width, Chips> {
    let offset: DmTensor<
        bf16,
        Chips,
        Cluster,
        FeatureSlices<Width>,
        m![Width % ({ Symbol<Chunk<Width>> })],
    > = offset.to_dm(&mut device.tdma);
    device
        .sub
        .begin(offset.view())
        .fetch::<m![Width % ({ Symbol<Chunk<Width>> }) / 8], m![Width % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Width % ({ Symbol<Chunk<Width>> }) / 8], m![Width % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Width % ({ Symbol<Chunk<Width>> }) / 8, Width % 8 / 4 % 2], m![Width % 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, 1.0_f32)
        .vector_widen_concat::<m![Width % ({ Symbol<Chunk<Width>> }) / 8], m![Width % 8]>()
        .vector_final()
        .to_vrf()
}

/// `input * gain / sqrt(mean(input²) + epsilon)` per row with FP32 statistics and one BF16
/// rounding; epsilon joins the whole row's mean square once, as a device operand.
pub(super) fn normalize<Width: AxisName, Rows: AxisName, Chips: M, Gain>(
    device: &mut Device,
    input: &Spread<Width, Rows, Chips>,
    epsilon: &HbmTensor<f32, Chips, m![1 # 8]>,
    gain: Gain,
) -> Spread<Width, Rows, Chips>
where
    Gain: IntoBranchedOperand<f32, Normalized<Width, Rows, Chips>>
        + StashTransition<Fresh, f32, { stage::Way::Way4 }, Next = Fresh>,
{
    let epsilon: DmTensor<f32, Chips, Cluster, FeatureSlices<Width>, m![1 # 8]> =
        epsilon.to_dm(&mut device.tdma);
    let epsilon = scalar::register(device, &epsilon);
    let variance: DmTensor<f32, Chips, Cluster, Reduced, m![Rows, 1 # 8]> = device
        .main
        .begin(input.view())
        .fetch::<m![Rows, Width % ({ Symbol<Chunk<Width>> }) / 8], m![Width % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Rows, Width % ({ Symbol<Chunk<Width>> }) / 8], m![Width % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Rows, Width % ({ Symbol<Chunk<Width>> }) / 8, Width % 8 / 4 % 2], m![Width % 4]>()
        .vector_mean_square::<Width, m![Rows], m![Width / ({ Symbol<Chunk<Width>> })], 0>()
        .vector_inter_slice_reduce::<Reduced, m![Rows]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // Every slice holds the same row scalar; each row keeps its original address.
    let inverse: DmTensor<f32, Chips, Cluster, FeatureSlices<Width>, m![Rows, 1 # 8]> =
        unsafe { variance.reshape() };
    let inverse: VrfTensor<f32, Chips, Cluster, FeatureSlices<Width>, m![Rows, 1 # 8]> = device
        .sub
        .begin(inverse.view())
        .fetch::<m![Rows], m![1 # 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Rows], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, &epsilon)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_fp_div_with_mode(BinaryArgMode::Mode10, 1.0f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .to_vrf();
    device
        .main
        .begin(input.view())
        .fetch::<m![Rows, Width % ({ Symbol<Chunk<Width>> }) / 8], m![Width % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Rows, Width % ({ Symbol<Chunk<Width>> }) / 8], m![Width % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Rows, Width % ({ Symbol<Chunk<Width>> }) / 8, Width % 8 / 4 % 2], m![Width % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &inverse)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), gain)
        .vector_widen_concat::<m![Rows, Width % ({ Symbol<Chunk<Width>> }) / 8], m![Width % 8]>()
        .vector_final()
        .cast::<bf16, m![Width % 8 # 16]>()
        .commit_trim::<m![Width % 8]>()
        .commit()
}
