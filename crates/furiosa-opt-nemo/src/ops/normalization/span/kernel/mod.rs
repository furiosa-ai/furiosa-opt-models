use furiosa_opt_std::prelude::*;

use crate::ops::layout::axes::Chip;

mod rows;
mod tiles;
pub(crate) use rows::compute_rows;
pub(crate) use tiles::{Tiles, Tiling};

pub(crate) type Owned<Width, All> =
    DmTensor<bf16, Chip, m![All / (All / 2)], m![Width / 16], m![All % (All / 2), Width % 16]>;

/// Vector register file bytes per slice.
const VRF: usize = 8192;
/// VRF bytes of one row's inverse: eight FP32 lanes.
const ROW: usize = 8 * 4;
/// VRF bytes the preceding residual add can still hold when a norm starts: one flit.
const OPERAND: usize = 32;

/// Rows per tile when tiles alternate, so two inverses fit beside the gain and the operand.
pub(crate) const fn capacity<E: M>() -> usize {
    (VRF - OPERAND - E::SIZE * 4) / (2 * ROW)
}

/// Full tiles before the last one; none when a single inverse holds every row.
const fn whole<T: M, E: M>() -> usize {
    if OPERAND + E::SIZE * 4 + T::SIZE * ROW <= VRF {
        0
    } else {
        (T::SIZE - 1) / capacity::<E>()
    }
}

/// Rows of the last tile, the only one that may be partial.
pub(crate) const fn last<T: M, E: M>() -> usize {
    T::SIZE - whole::<T, E>() * capacity::<E>()
}

/// Normalize the first `rows` rows of every slice; rows past them keep stale values.
pub(crate) fn compute_live<Width: AxisName, C: M, P: M, R: M, T: M, E: M, PART: M, N: Tiling>(
    context: &mut Device,
    input: (
        &DmTensor<bf16, Chip, C, P, m![{ T }, { E }]>,
        &DmTensor<bf16, Chip, C, P, E>,
    ),
    rows: usize,
) -> DmTensor<bf16, Chip, C, P, m![{ T }, { E }]> {
    const { assert!(N::CAPACITY == capacity::<E>() && N::LAST == last::<T, E>()) };
    let gain: VrfTensor<f32, Chip, C, P, E> = context
        .sub
        .begin(input.1.view())
        .fetch::<m![{ E } / 8], m![{ E } % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![{ E } / 8], m![{ E } % 8]>()
        .to_vrf();
    let mut variance: DmTensor<f32, Chip, C, R, m![{ T }, 1 # 8]> = DmTensor::new();
    context
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
            const { f32::from_bits(N::EPSILON as u32) / PART::SIZE as f32 },
        )
        .vector_inter_slice_reduce::<R, T>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit_trim::<m![1 # 2]>()
        .commit_prefix::<T, _, _>(variance.view_mut(), rows);
    // SAFETY: the reduction broadcasts each scalar to the original feature partitions.
    let variance: DmTensor<f32, Chip, C, P, m![{ T }, 1 # 8]> = unsafe { variance.reshape() };
    N::finish_live::<C, P, T, E>(context, (input.0, &variance, &gain), rows)
}
