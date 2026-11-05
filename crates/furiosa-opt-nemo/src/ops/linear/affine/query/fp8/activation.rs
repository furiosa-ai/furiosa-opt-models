use super::*;
use crate::ops::swiglu::spans::fp8::{LARGEST, SMALLEST};

pub(super) type Landed<Hidden, G> =
    DmTensor<bf16, Chip, m![2], m![4, Hidden / 64], m![G, Tile, Hidden % 64]>;
axes![Copies = 64];
type Scales<G> = DmTensor<bf16, Chip, m![2], m![4, Copies], m![G, Tile]>;
type Activation<Hidden, G> = (Input<Hidden, G>, Scales<G>);

pub(crate) fn quantize<Hidden: AxisName, G: Group>(
    context: &mut Device,
    input: &Landed<Hidden, G>,
    groups: usize,
) -> Activation<Hidden, G> {
    let mut scales = Scales::<G>::new();
    commit_live!(
        context
            .main
            .begin(input.view())
            .fetch::<m![G, Tile, Hidden % 64 / 8], m![Hidden % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![G, Tile, Hidden % 64 / 8], m![Hidden % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![G, Tile, Hidden % 64 / 4], m![Hidden % 4]>()
            .vector_fp_binary_with_mode(
                FpBinaryOp::MulF(FpMulAlu::Mul0),
                BinaryArgMode::Mode00,
                0.0f32
            )
            .vector_fp_unary(FpUnaryOp::Sqrt)
            .vector_intra_slice_reduce::<Hidden, m![G, Tile], m![1 # 4]>(IntraSliceReduceOpF32::Max)
            .vector_fp_div(LARGEST)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_clip(ClipBinaryOpF32::Max, SMALLEST)
            .vector_inter_slice_reduce::<m![4, Copies], m![G, Tile]>(InterSliceReduceOpF32::Max)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![G, Tile / 4], m![Tile % 4 # 16]>()
            .commit_trim::<m![Tile % 4]>(),
        scales.view_mut(),
        groups
    );
    // SAFETY: the reduction broadcasts each row's scale to every slice of its cluster.
    let broadcast = unsafe {
        scales
            .view()
            .reshape::<Chip, m![2], m![4, Hidden / 64], m![G, Tile]>()
    };
    let scale: VrfTensor<f32, Chip, m![2], m![4, Hidden / 64], m![G, Tile]> = context
        .sub
        .begin(broadcast)
        .fetch::<m![G, Tile / 8], m![Tile % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![G, Tile / 8], m![Tile % 8]>()
        .to_vrf();
    let mut quantized = Input::<Hidden, G>::new();
    commit_live!(
        context
            .main
            .begin(input.view())
            .fetch::<m![G, Tile, Hidden % 64 / 8], m![Hidden % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![G, Tile, Hidden % 64 / 8], m![Hidden % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![G, Tile, Hidden % 64 / 4], m![Hidden % 4]>()
            .vector_fp_div(&scale)
            .vector_widen_concat::<m![G, Tile, Hidden % 64 / 8], m![Hidden % 8]>()
            .vector_final()
            .cast::<f8e4m3, m![Hidden % 8 # 32]>()
            .commit_trim::<m![Hidden % 8]>(),
        quantized.view_mut(),
        groups
    );
    (quantized, scales)
}

pub(crate) fn row_groups<Hidden: AxisName, G: Group, Heads: AxisName>(
    context: &mut Device,
    scales: &Scales<G>,
) -> RowGroups<G, Heads> {
    // SAFETY: the reduction broadcasts each row's scale to every slice of its cluster.
    let broadcast = unsafe {
        scales
            .view()
            .reshape::<Chip, Cluster<Heads>, Slice<Hidden, Heads>, m![G, Tile]>()
    };
    let mut rows = RowGroups::<G, Heads>::new();
    broadcast
        .slice_tile::<m![Hidden / 64], 1, m![Heads % (Heads / 2) / (Heads / 4),
        Heads % (Heads / 4) / (Heads / 8), Hidden / 64 = 1 # 64]>(0)
        .to_dm_view(&mut context.tdma, rows.view_mut());
    rows
}

pub(crate) fn fill<Hidden: AxisName, G: Group, Heads: AxisName>(
    context: &mut Device,
    input: &Input<Hidden, G>,
    groups: usize,
) -> Lanes<Hidden, G, Heads> {
    // SAFETY: both head partitions rename replicated coordinates; no activation moves.
    let input = unsafe {
        input
            .view()
            .reshape::<Chip, Cluster<Heads>, Slice<Hidden, Heads>, m![G, Tile, Hidden % 64]>()
    };
    let fetched = context
        .sub
        .begin(input)
        .fetch::<m![Tile % 8, G, Tile / 8, Hidden % 64 / 32], m![Hidden % 32]>()
        .collect::<m![Tile % 8, G, Tile / 8, Hidden % 64 / 32], m![Hidden % 32]>();
    if const { G::SIZE == 1 } {
        fetched.to_trf_prefix::<m![G, Tile / 8], _, _>(groups)
    } else {
        fetched.to_trf()
    }
}
