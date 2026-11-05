use super::{Input, Normalized, RowGroups, activation::Copies};
use crate::ops::layout::axes::Chip;
use crate::ops::linear::affine::query::Tile;
use crate::ops::linear::query::group::WavePair;
use crate::ops::swiglu::spans::fp8::{LARGEST, SMALLEST};
use furiosa_opt_std::prelude::*;

pub(crate) type Spread<Hidden, All> = DmTensor<
    bf16,
    Chip,
    m![All % 64 / 32],
    m![All % 32 / 8, Hidden / 64],
    m![All / 64, Hidden % 64 / 16, All % 8, Hidden % 16],
>;
type SpreadInput<Hidden, All> = DmTensor<
    f8e4m3,
    Chip,
    m![All % 64 / 32],
    m![All % 32 / 8, Hidden / 64],
    m![All / 64, All % 8, Hidden % 64],
>;
type SpreadScales<All> =
    DmTensor<bf16, Chip, m![All % 64 / 32], m![All % 32 / 8, Copies], m![All / 64, All % 8]>;
type Staged<Hidden, All> = HbmTensor<f8e4m3, Chip, m![Hidden / 64, All, Hidden % 64]>;
type StagedScales<All> = HbmTensor<bf16, Chip, m![All % 64 / 8, All / 64, All % 8]>;

pub(crate) fn land_spread<Hidden: AxisName, All: AxisName>(
    context: &mut Device,
    normalized: &Normalized<Hidden, All>,
) -> Spread<Hidden, All> {
    normalized.to_dm(&mut context.tdma)
}

pub(crate) fn quantize_spread<Hidden: AxisName, All: AxisName>(
    context: &mut Device,
    rows: &Spread<Hidden, All>,
    blocks: usize,
) -> (SpreadInput<Hidden, All>, SpreadScales<All>) {
    let mut scales = SpreadScales::<All>::new();
    context
        .main
        .begin(rows.view())
        .fetch::<m![All / 64, All % 8, Hidden % 64 / 16, Hidden % 16 / 8], m![Hidden % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![All / 64, All % 8, Hidden % 64 / 16, Hidden % 16 / 8], m![Hidden % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![All / 64, All % 8, Hidden % 64 / 4], m![Hidden % 4]>()
        .vector_fp_binary_with_mode(
            FpBinaryOp::MulF(FpMulAlu::Mul0),
            BinaryArgMode::Mode00,
            0.0f32,
        )
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_intra_slice_reduce::<Hidden, m![All / 64, All % 8], m![1 # 4]>(
            IntraSliceReduceOpF32::Max,
        )
        .vector_fp_div(LARGEST)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, SMALLEST)
        .vector_inter_slice_reduce::<m![All % 32 / 8, Copies], m![All / 64, All % 8]>(
            InterSliceReduceOpF32::Max,
        )
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![All / 64, All % 8 / 4], m![All % 4 # 16]>()
        .commit_trim::<m![All % 4]>()
        .commit_prefix::<m![All / 64], _, _>(scales.view_mut(), blocks);
    // SAFETY: the reduction broadcasts each row's scale to every slice of its slice group.
    let broadcast = unsafe {
        scales.view().reshape::<Chip, m![All % 64 / 32], m![All % 32 / 8, Hidden / 64],
            m![All / 64, All % 8]>()
    };
    let scale: VrfTensor<
        f32,
        Chip,
        m![All % 64 / 32],
        m![All % 32 / 8, Hidden / 64],
        m![All / 64, All % 8],
    > = context
        .sub
        .begin(broadcast)
        .fetch::<m![All / 64], m![All % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![All / 64], m![All % 8]>()
        .to_vrf();
    let mut quantized = SpreadInput::<Hidden, All>::new();
    context
        .main
        .begin(rows.view())
        .fetch::<m![All / 64, All % 8, Hidden % 64 / 16, Hidden % 16 / 8], m![Hidden % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![All / 64, All % 8, Hidden % 64 / 16, Hidden % 16 / 8], m![Hidden % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![All / 64, All % 8, Hidden % 64 / 4], m![Hidden % 4]>()
        .vector_fp_div(&scale)
        .vector_widen_concat::<m![All / 64, All % 8, Hidden % 64 / 8], m![Hidden % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Hidden % 8 # 32]>()
        .commit_trim::<m![Hidden % 8]>()
        .commit_prefix::<m![All / 64], _, _>(quantized.view_mut(), blocks);
    (quantized, scales)
}

pub(crate) fn stage<Hidden: AxisName, All: AxisName>(
    context: &mut Device,
    (rows, scales): (&SpreadInput<Hidden, All>, &SpreadScales<All>),
) -> (Staged<Hidden, All>, StagedScales<All>) {
    let mut staged = Staged::<Hidden, All>::new();
    rows.view()
        .to_hbm_view(&mut context.tdma, staged.view_mut());
    let mut stored = StagedScales::<All>::new();
    scales
        .view()
        .slice_tile::<m![Copies], 1, m![All % 32 / 8, Copies = 1 # 64]>(0)
        .to_hbm_view(&mut context.tdma, stored.view_mut());
    (staged, stored)
}

pub(crate) fn land_pair<Hidden: AxisName, All: AxisName>(
    context: &mut Device,
    staged: &Staged<Hidden, All>,
    pair: usize,
) -> Input<Hidden, WavePair> {
    let rows = staged.view().tile::<m![All / Tile], 2,
        m![Hidden / 64, All / Tile = 2 # (All / Tile), All % Tile, Hidden % 64]>(2 * pair);
    // SAFETY: paired waves do not overlap, so the pair's rows are its waves' in order.
    let rows = unsafe {
        rows.reshape::<Chip, m![Hidden / 64, WavePair # (All / Tile), Tile, Hidden % 64]>()
    };
    rows.to_dm(&mut context.tdma)
}

pub(crate) fn pair_scales<All: AxisName, Heads: AxisName>(
    context: &mut Device,
    scales: &StagedScales<All>,
    pair: usize,
) -> RowGroups<WavePair, Heads> {
    // SAFETY: the same order with the 64-row block split into its wave and half.
    let scales = unsafe {
        scales
            .view()
            .reshape::<Chip, m![All % 64 / 8, All / Tile, All % Tile / 64, All % 8]>()
    };
    let rows = scales.tile::<m![All / Tile], 2,
        m![All % 64 / 8, All / Tile = 2 # (All / Tile), All % Tile / 64, All % 8]>(2 * pair);
    // SAFETY: a pair's rows are its waves' in order: wave, half of 64 rows, block, row.
    let rows = unsafe {
        rows.reshape::<Chip, m![Tile % 64 / 8, WavePair # (All / Tile), Tile / 64, Tile % 8]>()
    };
    rows.to_dm(&mut context.tdma)
}
