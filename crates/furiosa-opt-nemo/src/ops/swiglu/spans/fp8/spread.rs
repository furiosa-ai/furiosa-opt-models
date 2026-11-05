use super::*;

pub(super) type SpreadScale<Intermediate, Rows> = DmTensor<
    bf16,
    Chip,
    Cluster<Intermediate>,
    m![Rows / (Rows / 8), Copies],
    m![Panels, Rows % (Rows / 8)],
>;
pub(super) type ChunkMax<Intermediate, Rows, P> =
    DmTensor<bf16, Chip, Cluster<Intermediate>, P, m![Panels, Rows % (Rows / 8)]>;
pub(super) type SpreadSlices<Intermediate, Rows> = m![Intermediate % 6144 / 192, Rows / (Rows / 8)];

pub(super) fn panel_maxima<Intermediate: AxisName, Rows: AxisName, const P: usize>(
    context: &mut Device,
    product: &Halves<Intermediate, Rows>,
    chunks: &mut ChunkMax<Intermediate, Rows, SpreadSlices<Intermediate, Rows>>,
) {
    // SAFETY: a chunk's panel digit and a slice's row digits name the same coordinates.
    let spread = unsafe {
        product.view().reshape::<Chip, Cluster<Intermediate>, SpreadSlices<Intermediate, Rows>,
            m![Panels, Rows % (Rows / 8), Intermediate % 64]>()
    };
    context
        .main
        .begin(
            spread
                .tile::<m![Panels], 1, m![Panels = 1 # 3, Rows % (Rows / 8), Intermediate % 64]>(P),
        )
        .fetch::<m![Rows % (Rows / 8), Intermediate % 64 / 8], m![Intermediate % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Rows % (Rows / 8), Intermediate % 64 / 8], m![Intermediate % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Rows % (Rows / 8), Intermediate % 64 / 4], m![Intermediate % 4]>()
        .vector_fp_binary_with_mode(
            FpBinaryOp::MulF(FpMulAlu::Mul0),
            BinaryArgMode::Mode00,
            0.0f32,
        )
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_intra_slice_reduce::<Intermediate, m![Rows % (Rows / 8)], m![1 # 4]>(
            IntraSliceReduceOpF32::Max,
        )
        .vector_fp_div(LARGEST)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, SMALLEST)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Rows % (Rows / 8) / 4], m![Rows % 4 # 16]>()
        .commit_trim::<m![Rows % 4]>()
        .commit_view(
            chunks
                .view_mut()
                .tile::<m![Panels], 1, m![Panels = 1 #{!} 3, Rows % (Rows / 8)]>(P),
        );
}

pub(super) fn spread_scales<Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    chunks: &ChunkMax<Intermediate, Rows, SpreadSlices<Intermediate, Rows>>,
    groups: &mut SpreadScale<Intermediate, Rows>,
) -> ChunkMax<Intermediate, Rows, SpreadSlices<Intermediate, Rows>> {
    let mut inner =
        ChunkMax::<Intermediate, Rows, m![Rows / (Rows / 8), Intermediate % 6144 / 192]>::new();
    chunks
        .view()
        .to_dm_view(&mut context.tdma, inner.view_mut());
    context.main.begin(inner.view())
        .fetch::<m![Panels, Rows % (Rows / 8) / 8], m![Rows % 8]>().fetch_cast::<f32>()
        .collect::<m![Panels, Rows % (Rows / 8) / 8], m![Rows % 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![Rows / (Rows / 8), Copies],
            m![Panels, Rows % (Rows / 8) / 8]>(InterSliceReduceOpF32::Max)
        .vector_final().cast::<bf16, m![Rows % 8 # 16]>()
        .commit_trim::<m![Rows % 8]>().commit_view(groups.view_mut());
    // SAFETY: the reduction copies each row group's scales to every slice of the group.
    let copies = unsafe {
        groups.view().reshape::<Chip, Cluster<Intermediate>,
            m![Rows / (Rows / 8), Intermediate % 6144 / 192], m![Panels, Rows % (Rows / 8)]>()
    };
    let mut broadcast = ChunkMax::<Intermediate, Rows, SpreadSlices<Intermediate, Rows>>::new();
    copies.to_dm_view(&mut context.tdma, broadcast.view_mut());
    broadcast
}

pub(super) fn quantize_spread<Intermediate: AxisName, Rows: AxisName, const P: usize>(
    context: &mut Device,
    product: &Halves<Intermediate, Rows>,
    broadcast: &ChunkMax<Intermediate, Rows, SpreadSlices<Intermediate, Rows>>,
    quantized: &mut Spread<Intermediate, Rows, f8e4m3>,
) {
    let scale: VrfTensor<
        f32,
        Chip,
        Cluster<Intermediate>,
        SpreadSlices<Intermediate, Rows>,
        m![Rows % (Rows / 8)],
    > = context
        .sub
        .begin(
            broadcast
                .view()
                .tile::<m![Panels], 1, m![Panels = 1 # 3, Rows % (Rows / 8)]>(P),
        )
        .fetch::<m![Rows % (Rows / 8) / 8], m![Rows % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Rows % (Rows / 8) / 8], m![Rows % 8]>()
        .to_vrf();
    // SAFETY: a chunk's panel digit and a slice's row digits name the same coordinates.
    let spread = unsafe {
        product.view().reshape::<Chip, Cluster<Intermediate>, SpreadSlices<Intermediate, Rows>,
            m![Panels, Rows % (Rows / 8), Intermediate % 64]>()
    };
    context
        .main
        .begin(
            spread
                .tile::<m![Panels], 1, m![Panels = 1 # 3, Rows % (Rows / 8), Intermediate % 64]>(P),
        )
        .fetch::<m![Rows % (Rows / 8), Intermediate % 64 / 8], m![Intermediate % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Rows % (Rows / 8), Intermediate % 64 / 8], m![Intermediate % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Rows % (Rows / 8), Intermediate % 64 / 4], m![Intermediate % 4]>()
        .vector_fp_div(&scale)
        .vector_widen_concat::<m![Rows % (Rows / 8), Intermediate % 64 / 8], m![Intermediate % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Intermediate % 8 # 32]>()
        .commit_trim::<m![Intermediate % 8]>()
        .commit_view(
            quantized
                .view_mut()
                .tile::<m![Panels], 1, m![Panels = 1 #{!} 3, Rows % (Rows / 8), Intermediate % 64]>(
                    P,
                ),
        );
}
