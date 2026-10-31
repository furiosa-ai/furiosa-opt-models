//! NVFP4 block einsums against a BF16 operand in the TRF, without restoring the weights: the fetch
//! table lookup decodes each E2M1 code straight to BF16 into the contraction.

use furiosa_opt_std::prelude::*;

/// Each 16-input block's sum of E2M1 `codes` times `input`, multiplied by the block's scale, in
/// FP32: the vector-engine stream of one value per row of `Rows` and block of `Inputs` (time
/// `Sums`), on four lanes. The caller reduces the blocks and commits the result.
///
/// - `In` is the input axis: a fetch packet holds 64 of its codes (32 bytes) and a contraction
///   packet one block of 16 BF16 values. `Inputs`, the inputs a slice holds of each row, divides it.
/// - `Sums` is the contraction's time: `Rows`, `Inputs / 16`, and the axes of `input`'s element
///   that the contraction keeps.
///
/// The packet reducer sums each block, so each block sum meets its scale as a broadcast VRF read.
pub(crate) fn scaled_block_sums<
    'l,
    In: AxisName,
    Rows: M,
    Inputs: M,
    Sums: M,
    Element: M,
    Chip: M,
    Cluster: M,
    Slice: M,
>(
    device: &'l mut Device,
    codes: DmTensorView<'l, f4e2m1, Chip, Cluster, Slice, m![{ Rows }, { Inputs }]>,
    input: &TrfTensor<bf16, Chip, Cluster, Slice, m![1], Element>,
    scales: &VrfTensor<f32, Chip, Cluster, Slice, m![{ Rows }, { Inputs } / 16]>,
) -> VectorFpTensor<
    'l,
    { Tu::Main },
    f32,
    Chip,
    Cluster,
    Slice,
    Sums,
    m![1 # 4],
    Fresh,
    { stage::VeOrder::IntraFirst },
> {
    device
        .main
        .begin(codes)
        .fetch::<m![{ Rows }, { Inputs } / 64], m![In % 64]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![{ Rows }, { Inputs } / 16], m![In % 16]>()
        .contract_outer::<Sums, m![In % 16], _, _, _>(input)
        .contract_packet::<m![1]>()
        .contract_time::<Sums>()
        .contract_lane::<Sums, m![1 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), scales)
}
