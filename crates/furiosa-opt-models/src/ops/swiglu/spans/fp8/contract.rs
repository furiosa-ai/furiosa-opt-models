use super::*;
use crate::Op;

/// Short contractions retain the eight normalization row owners but only their first
/// eight local rows: a 64-row, 8 KiB operand rather than a 128-row, 16 KiB operand.
type CompactLanes<H, I, R> = TrfTensor<
    f8e4m3,
    Chip,
    Cluster<I>,
    Partition<H, I>,
    m![R % 8],
    m![H % 128 / 32, R / (R / 8), H % 32],
>;

/// A panel's contraction with a wave's TRF lanes into the wave's rows of the term's sums, the
/// wave's first live row groups alone.
pub(super) struct Contract;

impl<'a, Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>
    Op<(
        (
            &'a Panel<Hidden, Intermediate>,
            &'a Lanes<Hidden, Intermediate, Rows>,
        ),
        (&'a mut Both<Intermediate, Rows>, usize, usize),
        usize,
    )> for Contract
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        ((weight, lanes), (output, term, wave), rows): (
            (
                &'a Panel<Hidden, Intermediate>,
                &'a Lanes<Hidden, Intermediate, Rows>,
            ),
            (&'a mut Both<Intermediate, Rows>, usize, usize),
            usize,
        ),
    ) {
        context.main.begin(weight.view())
            .fetch::<m![Rows % 16 / 8, { Out<Intermediate> } / 4, { Out<Intermediate> } % 4 / 2,
                Rows / (Rows / 4), { K<Hidden> } / 32],
                m![{ Out<Intermediate> } % 2, { K<Hidden> } % 32]>()
            .collect::<m![Rows % 16 / 8, { Out<Intermediate> } / 2, Rows / (Rows / 4),
                { K<Hidden> } / 32, { Out<Intermediate> } % 2], m![{ K<Hidden> } % 32]>()
            .contract_outer::<m![Rows % 16 / 8, { Out<Intermediate> } / 2, Rows / (Rows / 4),
                { K<Hidden> } / 32, Rows % (Rows / 4) / (Rows / 8)],
                m![{ Out<Intermediate> } % 2, { K<Hidden> } % 32], _, _, _>(lanes)
            .contract_packet::<m![{ Out<Intermediate> } % 2]>()
            .contract_time::<m![Rows % 16 / 8, { Out<Intermediate> } / 2, Rows / (Rows / 8)]>()
            .contract_lane::<m![Rows % 16 / 8, { Out<Intermediate> } / 2, Rows / (Rows / 8),
                { Out<Intermediate> } % 2], m![Rows % 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<Slice<Intermediate, Rows>,
                m![Rows % 16 / 8, { Out<Intermediate> }]>(InterSliceReduceOpF32::Add)
            .vector_final().cast::<bf16, m![Rows % 8 # 16]>()
            .transpose::<m![Rows % 16 / 8, { Out<Intermediate> } / 4, Rows % 8],
                m![{ Out<Intermediate> } % 4 # 16]>().commit_trim::<m![{ Out<Intermediate> } % 4]>()
            .commit_prefix::<m![Rows % 16 / 8], _, _>(output.view_mut()
                .tile::<m![Term], 1,
                    m![Rows % (Rows / 8), Term = 1 #{!} 2, { Out<Intermediate> }]>(term)
                .tile::<m![Rows % (Rows / 8) / 16], 1,
                    m![Rows % (Rows / 8) / 16 = 1 #{!} (Rows / 128), Rows % 16,
                        Term = 1 #{!} 2, { Out<Intermediate> }]>(wave),
                (rows + 63) / 64 - 2 * wave);
    }
}

/// A gate/up panel pair's contraction with a wave's TRF lanes into the wave's rows of both terms'
/// sums.
impl<'a, Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>
    Op<(
        (
            &'a GateUp<Hidden, Intermediate>,
            &'a Lanes<Hidden, Intermediate, Rows>,
        ),
        (&'a mut Both<Intermediate, Rows>, usize),
        usize,
    )> for Contract
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        ((weight, lanes), (output, wave), rows): (
            (
                &'a GateUp<Hidden, Intermediate>,
                &'a Lanes<Hidden, Intermediate, Rows>,
            ),
            (&'a mut Both<Intermediate, Rows>, usize),
            usize,
        ),
    ) {
        context.main.begin(weight.view())
            .fetch::<m![Rows % 16 / 8, Term, { Out<Intermediate> } / 4,
                { Out<Intermediate> } % 4 / 2, Rows / (Rows / 4), { K<Hidden> } / 32],
                m![{ Out<Intermediate> } % 2, { K<Hidden> } % 32]>()
            .collect::<m![Rows % 16 / 8, Term, { Out<Intermediate> } / 2, Rows / (Rows / 4),
                { K<Hidden> } / 32, { Out<Intermediate> } % 2], m![{ K<Hidden> } % 32]>()
            .contract_outer::<m![Rows % 16 / 8, Term, { Out<Intermediate> } / 2,
                Rows / (Rows / 4), { K<Hidden> } / 32, Rows % (Rows / 4) / (Rows / 8)],
                m![{ Out<Intermediate> } % 2, { K<Hidden> } % 32], _, _, _>(lanes)
            .contract_packet::<m![{ Out<Intermediate> } % 2]>()
            .contract_time::<
                m![Rows % 16 / 8, Term, { Out<Intermediate> } / 2, Rows / (Rows / 8)]
            >()
            .contract_lane::<m![Rows % 16 / 8, Term, { Out<Intermediate> } / 2,
                Rows / (Rows / 8), { Out<Intermediate> } % 2], m![Rows % 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<Slice<Intermediate, Rows>,
                m![Rows % 16 / 8, Term, { Out<Intermediate> }]>(InterSliceReduceOpF32::Add)
            .vector_final().cast::<bf16, m![Rows % 8 # 16]>()
            .transpose::<m![Rows % 16 / 8, Term, { Out<Intermediate> } / 4, Rows % 8],
                m![{ Out<Intermediate> } % 4 # 16]>().commit_trim::<m![{ Out<Intermediate> } % 4]>()
            .commit_prefix::<m![Rows % 16 / 8], _, _>(output.view_mut()
                .tile::<m![Rows % (Rows / 8) / 16], 1,
                    m![Rows % (Rows / 8) / 16 = 1 #{!} (Rows / 128), Rows % 16,
                        Term, { Out<Intermediate> }]>(wave), (rows + 63) / 64 - 2 * wave);
    }
}

/// A term's two partial sums added in FP32 and scaled by the row and channel scales, the first
/// `rows` rows' row groups.
pub(super) struct Combine;

/// The two partial sums with the term they hold, and the row and channel scales.
type Terms<'a, Intermediate, Rows> = (
    (
        &'a Both<Intermediate, Rows>,
        &'a Both<Intermediate, Rows>,
        usize,
    ),
    (
        &'a RowScale<Intermediate, Rows>,
        &'a ChannelScale<Intermediate, Rows>,
    ),
    usize,
);

impl<'a, Intermediate: AxisName, Rows: AxisName> Op<Terms<'a, Intermediate, Rows>> for Combine {
    type Output = Projection<Intermediate, Rows>;

    fn forward(
        &self,
        context: &mut Device,
        ((left, right, term), (row, channel), rows): Terms<'a, Intermediate, Rows>,
    ) -> Self::Output {
        let side = |both: &'a Both<Intermediate, Rows>, term: usize| {
            both.view()
                .tile::<m![Term], 1, m![Rows % (Rows / 8), Term = 1 # 2, { Out<Intermediate> }]>(
                    term,
                )
        };
        let live = (rows + 63) / 64 * 8;
        let mut sum: DmTensor<
            f32,
            Chip,
            Cluster<Intermediate>,
            Slice<Intermediate, Rows>,
            m![Rows % (Rows / 8), { Out<Intermediate> }],
        > = DmTensor::new();
        context.main.begin_interleaved::<Side, _, _, _, _, _>(
            side(left, term), side(right, term))
            .fetch::<m![Rows % (Rows / 8), Side, { Out<Intermediate> } / 8],
                m![{ Out<Intermediate> } % 8]>().fetch_cast::<f32>()
            .collect::<m![Rows % (Rows / 8), Side, { Out<Intermediate> } / 8],
                m![{ Out<Intermediate> } % 8]>()
            .vector_init().vector_intra_slice_unzip::<Side,
                m![Rows % (Rows / 8), { Out<Intermediate> } / 8]>()
            .vector_clip_zip(ClipBinaryOpF32::Add).vector_final()
            .commit_trim::<m![{ Out<Intermediate> } % 8]>()
            .commit_prefix::<m![Rows % (Rows / 8)], _, _>(sum.view_mut(), live);
        let mut output: Projection<Intermediate, Rows> = DmTensor::new();
        context.main.begin(sum.view())
            .fetch::<
                m![Rows % (Rows / 8), { Out<Intermediate> } / 8],
                m![{ Out<Intermediate> } % 8],
            >()
            .collect::<
                m![Rows % (Rows / 8), { Out<Intermediate> } / 8],
                m![{ Out<Intermediate> } % 8],
            >()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Rows % (Rows / 8), { Out<Intermediate> } / 4],
                m![{ Out<Intermediate> } % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), row)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), channel)
            .vector_widen_concat::<m![Rows % (Rows / 8), { Out<Intermediate> } / 8],
                m![{ Out<Intermediate> } % 8]>()
            .vector_final().cast::<bf16, m![{ Out<Intermediate> } % 8 # 16]>()
            .commit_trim::<m![{ Out<Intermediate> } % 8]>()
            .commit_prefix::<m![Rows % (Rows / 8)], _, _>(output.view_mut(), live);
        output
    }
}

/// The first one or two eight-row owners, using the existing slab and TRF placement.
/// Static row groups retain the existing two-output, 64-byte weight packet.
fn short<H: AxisName, I: AxisName, R: AxisName, E: M, const G: usize, const T: usize>(
    context: &mut Device,
    weight: DmTensorView<'_, f8e4m3, Chip, Cluster<I>, Partition<H, I>, E>,
    lanes: &CompactLanes<H, I, R>,
    output: DmTensorViewMut<
        '_,
        bf16,
        Chip,
        Cluster<I>,
        Slice<I, R>,
        m![R % (R / 8), Term = {T} #{!} 2, {Out<I>}],
    >,
) {
    const {
        assert!(G == 1 || G == 2);
        assert!(T == 1 || T == 2);
    }
    // SAFETY: T is two for a paired slab and one for the separate final gate/up slab.
    let weight = unsafe {
        weight.reshape::<Chip, Cluster<I>, Partition<H, I>, m![Term = {T}, {Out<I>}, {K<H>}]>()
    };
    let target = output
        .slice_tile::<m![R / (R / 8)], G, m![{Blocks<I>}, R / (R / 8) = {G} #{!} 8]>(0)
        .tile::<m![R % (R / 8) / 8], 1,
            m![R % (R / 8) / 8 = 1 #{!} (R / 64), R % 8, Term = {T} #{!} 2, {Out<I>}]>(0);
    let products = context.main.begin(weight)
        .fetch::<m![R % (R / 4) / (R / 8) = {G}, Term = {T}, {Out<I>} / 2, {K<H>} / 32], m![{Out<I>} % 2, {K<H>} % 32]>()
        .collect::<m![R % (R / 4) / (R / 8) = {G}, Term = {T}, {Out<I>} / 2, {K<H>} / 32, {Out<I>} % 2], m![{K<H>} % 32]>()
        .contract_outer::<m![R % 16 / 8 = 1, R / (R / 4) = 1, R % (R / 4) / (R / 8) = {G}, Term = {T}, {Out<I>} / 2,
            {K<H>} / 32], m![{Out<I>} % 2, {K<H>} % 32], _, _, _>(lanes)
        .contract_packet::<m![{Out<I>} % 2]>()
        .contract_time::<m![R % 16 / 8 = 1, R / (R / 4) = 1, R % (R / 4) / (R / 8) = {G}, Term = {T}, {Out<I>} / 2]>()
        .contract_lane::<m![R % 16 / 8 = 1, R / (R / 4) = 1, R % (R / 4) / (R / 8) = {G}, Term = {T}, {Out<I>} / 2, {Out<I>} % 2], m![R % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<m![{Blocks<I>}, 1 # 8], m![R % (R / 4) / (R / 8) = {G}, Term = {T}, {Out<I>}]>(InterSliceReduceOpF32::Add)
        .vector_final().cast::<bf16, m![R % 8 # 16]>()
        .transpose::<m![R % (R / 4) / (R / 8) = {G}, Term = {T}, {Out<I>} / 4, R % 8], m![{Out<I>} % 4 # 16]>()
        .commit_trim::<m![{Out<I>} % 4]>();
    // One eight-row group lands directly. For sixteen rows, keeping the groups local
    // avoids the compiler's unsupported partially occupied reduction slices. The DMA
    // only redistributes the BF16 slab result into its established row-owner layout.
    if const { G == 1 } {
        products.commit_view(target);
    } else {
        let contracted: DmTensor<
            bf16,
            Chip,
            Cluster<I>,
            m![{Blocks<I>},1 # 8],
            m![R % (R / 4) / (R / 8) = {G},R % 8,Term = {T},{Out<I>}],
        > = products.commit();
        // SAFETY: rename the two row groups to the destination's slice-owner coordinate.
        let contracted = unsafe {
            contracted.view().reshape::<Chip,Cluster<I>,m![{Blocks<I>},1 # 8],m![R / (R / 8) = {G}, R % 8, Term = {T}, {Out<I>}]>()
        };
        contracted.to_dm_view(&mut context.tdma, target);
    }
}

pub(super) fn short_pair<H: AxisName, I: AxisName, R: AxisName>(
    context: &mut Device,
    weight: &GateUp<H, I>,
    input: &Input<H, I, R>,
    panel: usize,
    output: &mut Both<I, R>,
    rows: usize,
) {
    let lanes = compact_lanes(context, input, panel);
    if rows <= 8 {
        short::<H, I, R, _, 1, 2>(
            context,
            weight.view(),
            &lanes,
            output
                .view_mut()
                .tile::<m![Term], 2, m![R % (R / 8), Term = 2 #{!} 2, {Out<I>}]>(0),
        );
    } else {
        short::<H, I, R, _, 2, 2>(
            context,
            weight.view(),
            &lanes,
            output
                .view_mut()
                .tile::<m![Term], 2, m![R % (R / 8), Term = 2 #{!} 2, {Out<I>}]>(0),
        );
    }
}

pub(super) fn short_single<H: AxisName, I: AxisName, R: AxisName>(
    context: &mut Device,
    weight: &Panel<H, I>,
    input: &Input<H, I, R>,
    panel: usize,
    output: &mut Both<I, R>,
    term: usize,
    rows: usize,
) {
    let lanes = compact_lanes(context, input, panel);
    let target = output
        .view_mut()
        .tile::<m![Term], 1, m![R % (R / 8), Term = 1 #{!} 2, {Out<I>}]>(term);
    if rows <= 8 {
        short::<H, I, R, _, 1, 1>(context, weight.view(), &lanes, target);
    } else {
        short::<H, I, R, _, 2, 1>(context, weight.view(), &lanes, target);
    }
}

fn compact_lanes<H: AxisName, I: AxisName, R: AxisName>(
    context: &mut Device,
    input: &Input<H, I, R>,
    panel: usize,
) -> CompactLanes<H, I, R> {
    context.sub.begin(input.view()
        .tile::<m![H / 1024],1,m![H / 1024 = 1 # (H / 1024), R % (R / 8), H % 32]>(panel)
        .tile::<m![R % (R / 8) / 8],1,
            m![H / 1024 = 1 # (H / 1024), R % (R / 8) / 8 = 1 # (R / 64), R % 8, H % 32]>(0))
        .fetch::<m![R % 8],m![H % 32]>()
        .switch::<Partition<H,I>,m![R % 8,H % 128 / 32,R / (R / 8)]>(
            SwitchConfig::TransposedBroadcast1 {slice1:8,slice0:32})
        .collect::<m![R % 8,H % 128 / 32,R / (R / 8)],m![H % 32]>().to_trf()
}
