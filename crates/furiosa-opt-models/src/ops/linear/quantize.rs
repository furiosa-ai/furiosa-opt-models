//! Quantize: rounds values into NVFP4, E2M1 codes with one E4M3 scale per 16 consecutive
//! values, bit for bit as the reference quantizer does.

use furiosa_opt_std::prelude::BitReq::{Ignore, One};
use furiosa_opt_std::prelude::*;

use crate::ops::pointwise::cast::Finite;
use crate::ops::vector::Restore as _;
use core::marker::PhantomData;

// The 16 values one E4M3 scale covers, apart from the block they belong to.
axes![Group = 16];

/// An NVFP4 quantization's passes over BF16 values in DM: `Units` 16-input blocks of `Blocks`,
/// values `Grouped` in `Eighths` split into `Fourths`, scales in `Octets` and `Quads` packets.
pub(super) struct Quantization<Units, Grouped, Eighths, Fourths, Octets, Quads, Blocks>(
    PhantomData<(Units, Grouped, Eighths, Fourths, Octets, Quads, Blocks)>,
);

impl<Units: M, Grouped: M, Eighths: M, Fourths: M, Octets: M, Quads: M, Blocks: M>
    Quantization<Units, Grouped, Eighths, Fourths, Octets, Quads, Blocks>
{
    /// Each element of the `input` view as its E2M1 value in FP32, and each block's E4M3 scale in
    /// the vector registers: `global` puts block scales in E4M3 units, `restore` returns them.
    pub(super) fn codes<'i, X: M, C: M, S: M, E: M>(
        device: &mut Device,
        input: impl Fn() -> DmTensorView<'i, bf16, X, C, S, E>,
        (global, restore): (
            &VrfTensor<f32, X, C, S, m![1 # 8]>,
            &VrfTensor<f32, X, C, S, m![1 # 8]>,
        ),
    ) -> (
        DmTensor<f32, X, C, S, Grouped>,
        VrfTensor<f32, X, C, S, Units>,
    ) {
        // Rounding is monotone, so the largest rounded `|x| / 6 * global` of a block is the
        // reference's `global * (amax / 6)`.
        let scale: DmTensor<f8e4m3, X, C, S, Units> = device
            .main
            .begin(input())
            .fetch::<Units, m![Group]>()
            .fetch_cast::<f32>()
            .collect::<Eighths, m![Group % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_reinterpret::<i32>()
            .vector_logic(LogicBinaryOpI32::BitAnd, i32::MAX)
            .vector_reinterpret::<f32>()
            .vector_narrow_split::<Fourths, m![Group % 4]>()
            .vector_fp_binary(
                FpBinaryOp::MulF(FpMulAlu::Mul0),
                const { 1.0f32 / f4e2m1::LARGEST },
            )
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), global)
            .vector_intra_slice_reduce::<Group, Units, m![1 # 4]>(IntraSliceReduceOpF32::Max)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_clip(ClipBinaryOpF32::Min, f8e4m3::LARGEST)
            .vector_final()
            .cast::<f8e4m3, m![1 # 32]>()
            .transpose::<Octets, m![{ Blocks } % 8 # 32]>()
            .commit_trim::<m![{ Blocks } % 8]>()
            .commit();
        // A zero scale gives an infinite reciprocal; capping it keeps `0 * reciprocal` finite.
        let reciprocal: DmTensor<f32, X, C, S, Units> = device
            .main
            .begin(scale.view())
            .fetch::<Octets, m![{ Blocks } % 8]>()
            .fetch_cast::<f32>()
            .collect::<Octets, m![{ Blocks } % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<Quads, m![{ Blocks } % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), restore)
            .vector_fp_div_with_mode(BinaryArgMode::Mode10, 1.0f32)
            .vector_widen_concat::<Octets, m![{ Blocks } % 8]>()
            .vector_clip(ClipBinaryOpF32::Min, f32::MAX)
            .vector_final()
            .commit_trim::<m![{ Blocks } % 8]>()
            .commit();
        let reciprocal: VrfTensor<f32, X, C, S, Units> = device
            .sub
            .begin(reciprocal.view())
            .fetch::<Octets, m![{ Blocks } % 8]>()
            .collect::<Octets, m![{ Blocks } % 8]>()
            .to_vrf();
        let scaled: DmTensor<f32, X, C, S, Grouped> = device
            .main
            .begin(input())
            .fetch::<Units, m![Group]>()
            .fetch_cast::<f32>()
            .collect::<Eighths, m![Group % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<Fourths, m![Group % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &reciprocal)
            .vector_widen_concat::<Eighths, m![Group % 8]>()
            .vector_clip(ClipBinaryOpF32::Max, -f4e2m1::LARGEST)
            .vector_clip(ClipBinaryOpF32::Min, f4e2m1::LARGEST)
            .vector_final()
            .commit_trim::<m![Group % 8]>()
            .commit();
        // Adding 1.5 * 2^(22 + k) rounds to the E2M1 spacing 2^(k - 1) of the value's binade,
        // to nearest even; subtracting it again is exact.
        let below = TagGuard::matches([One, One, Ignore, Ignore]);
        let within = TagGuard::matches([Ignore, Ignore, One, One]);
        let rounded: DmTensor<f32, X, C, S, Grouped> = device
            .main
            .begin(scaled.view())
            .fetch::<Eighths, m![Group % 8]>()
            .collect::<Eighths, m![Group % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Comparison([
                Cmp::Less(2.0f32),
                Cmp::Greater(-2.0f32),
                Cmp::Less(4.0f32),
                Cmp::Greater(-4.0f32),
            ]))
            .vector_narrow_split::<Fourths, m![Group % 4]>()
            .vector_fp_binary(
                FpBinaryOp::AddF,
                Branched::imm(below, ROUND[0])
                    .imm(within, ROUND[1])
                    .imm(TagGuard::all(), ROUND[2]),
            )
            .vector_widen_concat::<Eighths, m![Group % 8]>()
            .vector_clip(
                ClipBinaryOpF32::Add,
                Branched::imm(below, -ROUND[0])
                    .imm(within, -ROUND[1])
                    .imm(TagGuard::all(), -ROUND[2]),
            )
            .vector_final()
            .commit_trim::<m![Group % 8]>()
            .commit();
        let scale: VrfTensor<f32, X, C, S, Units> = device
            .sub
            .begin(scale.view())
            .fetch::<Octets, m![{ Blocks } % 8]>()
            .fetch_cast::<f32>()
            .collect::<Octets, m![{ Blocks } % 8]>()
            .to_vrf();
        (rounded, scale)
    }

    /// The E2M1 `codes` times their blocks' `scale` in BF16, as the stream a caller commits.
    pub(super) fn restored<'l, X: M, C: M, S: M, E: M>(
        main: &'l mut TuContext<{ Tu::Main }>,
        codes: DmTensorView<'l, f32, X, C, S, E>,
        scale: &VrfTensor<f32, X, C, S, Units>,
    ) -> CommitTrimTensor<'l, { Tu::Main }, bf16, X, C, S, Eighths, m![Group % 8]> {
        main.begin(codes)
            .fetch::<Eighths, m![Group % 8]>()
            .collect::<Eighths, m![Group % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<Fourths, m![Group % 4]>()
            .vector_restore::<Eighths, m![Group % 8], _, _>(scale, |values| values)
    }
}

/// Magic constants rounding magnitudes below 2, below 4 and up to 6 to E2M1 steps.
const ROUND: [f32; 3] = [
    1.5 * (1 << 22) as f32,
    1.5 * (1 << 23) as f32,
    1.5 * (1 << 24) as f32,
];
