use super::super::compact as tile;
use super::*;
use crate::Op;
use crate::ops::pointwise::add::Add;

axes![Term = 2];
type Partial<H, I, R, const W: usize> =
    DmTensor<f32, Chip, tile::Cluster, tile::Slices<H>, m![R % { W }, I % 224]>;
type Paired<H, I, R, const W: usize> =
    DmTensor<f32, Chip, tile::Cluster, tile::Slices<H>, m![R % { W }, Term, I % 224]>;

pub(super) fn prepare<H: AxisName, I: AxisName, R: AxisName, const W: usize>(
    device: &mut Device,
    gate: &Panels<H, I>,
    up: &Panels<H, I>,
    down: &Down<I, H>,
    input: &Owned<H, R>,
    entry: Entry<H, I>,
    product: &mut Products<I, R>,
) -> down::Ahead<I, H> {
    const {
        assert!(W == 8);
    }
    let input = tile::input::<H, R, W>(device, input);
    let (g0, u0, d0) = entry;
    let g1 = gate.load(device, 1);
    let u1 = up.load(device, 1);
    let l0 = tile::lanes(device, &input, 0);
    let l1 = tile::lanes(device, &input, 1);
    let p0 = pair::<H, I, R, W>(device, &g0, &u0, &l0);
    let g2 = gate.load(device, 2);
    let u2 = up.load(device, 2);
    let l2 = tile::lanes(device, &input, 2);
    let p1 = pair::<H, I, R, W>(device, &g1, &u1, &l1);
    let g3 = gate.load(device, 3);
    let u3 = up.load(device, 3);
    let s1 = Add.forward(device, (p0.view(), p1.view()));
    let l3 = tile::lanes(device, &input, 3);
    let p2 = pair::<H, I, R, W>(device, &g2, &u2, &l2);
    let d1 = down.stage(device, 1);
    let s2 = Add.forward(device, (s1.view(), p2.view()));
    let g3 = single::<H, I, R, W>(device, &g3, &l3);
    let g = term(device, &s2, 0);
    let g = Add.forward(device, (g.view(), g3.view()));
    let g = finish(device, &gate.weights.contract, &g);
    let u3 = single::<H, I, R, W>(device, &u3, &l3);
    let u = term(device, &s2, 1);
    let u = Add.forward(device, (u.view(), u3.view()));
    let u = finish(device, &up.weights.contract, &u);
    product::compact::<I, R, W>(device, &g, &u, product);
    (d0, d1)
}

// Keep packed codes resident; only the 16-value packet is expanded before its dot.
fn single<H: AxisName, I: AxisName, R: AxisName, const W: usize>(
    device: &mut Device,
    weight: &super::super::panels::Packed<H, I>,
    lanes: &tile::Lanes<H, R, W>,
) -> Partial<H, I, R, W> {
    // SAFETY: MLP has 224 features per owner; physical ownership is unchanged.
    let codes = unsafe {
        weight
            .0
            .view()
            .reshape::<Chip, tile::Cluster, tile::Slices<H>, m![I % 224, H % 128]>()
    };
    let scales = unsafe {
        weight
            .1
            .view()
            .reshape::<Chip, tile::Cluster, tile::Slices<H>, m![I % 224, H % 128 / 16]>()
    };
    let scales: VrfTensor<f32, Chip, tile::Cluster, tile::Slices<H>, m![I % 224, H % 128 / 16]> =
        device
            .sub
            .begin(scales)
            .fetch::<m![I % 224], m![H % 128 / 16]>()
            .fetch_cast::<f32>()
            .collect::<m![I % 224], m![H % 128 / 16]>()
            .to_vrf();
    device.main.begin(codes)
        .fetch::<m![R % {W} / 8, I % 224 / 2, H % 32 / 16, H % 128 / 32, I % 2], m![H % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![R % {W} / 8, I % 224 / 2, H % 32 / 16, H % 128 / 32, I % 2], m![H % 16]>()
        .contract_outer::<m![R % {W} / 8, I % 224 / 2, H % 32 / 16, H % 128 / 32], m![I % 2, H % 16], _, _, _>(lanes)
        .contract_packet::<m![I % 2]>()
        .contract_time::<m![R % {W} / 8, I % 224 / 2, H % 32 / 16, H % 128 / 32]>()
        .contract_lane::<m![R % {W} / 8, I % 224 / 2, H % 32 / 16, H % 128 / 32, I % 2], m![R % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![R % {W} / 8, I % 224 / 2, H % 32 / 16, H % 128 / 32, I % 2, R % 8 / 4], m![R % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scales)
        .vector_intra_slice_reduce::<H, m![R % {W} / 8, I % 224 / 2, I % 2, R % 8 / 4], m![R % 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_concat::<m![R % {W} / 8, I % 224 / 2, I % 2], m![R % 8]>()
        .vector_final().transpose::<m![R % {W} / 8, I % 224 / 2, R % 8], m![I % 2 # 8]>()
        .commit_trim::<m![I % 2]>().commit()
}

fn pair<H: AxisName, I: AxisName, R: AxisName, const W: usize>(
    device: &mut Device,
    gate: &super::super::panels::Packed<H, I>,
    up: &super::super::panels::Packed<H, I>,
    lanes: &tile::Lanes<H, R, W>,
) -> Paired<H, I, R, W> {
    // SAFETY: MLP's 64 owners each carry 224 consecutive intermediate features.
    let g = unsafe {
        gate.0
            .view()
            .reshape::<Chip, tile::Cluster, tile::Slices<H>, m![I % 224, H % 128]>()
    };
    let u = unsafe {
        up.0.view()
            .reshape::<Chip, tile::Cluster, tile::Slices<H>, m![I % 224, H % 128]>()
    };
    let gs = unsafe {
        gate.1
            .view()
            .reshape::<Chip, tile::Cluster, tile::Slices<H>, m![I % 224, H % 128 / 16]>()
    };
    let us = unsafe {
        up.1.view()
            .reshape::<Chip, tile::Cluster, tile::Slices<H>, m![I % 224, H % 128 / 16]>()
    };
    let mut output = Paired::new();
    #[unroll]
    for half in 0..2 {
        let g = g
            .tile::<m![I % 224 / 112], 1, m![I % 224 / 112 = 1 # 2, I % 112, H % 128]>(half)
            .unpad::<m![I % 112, H % 128]>();
        let u = u
            .tile::<m![I % 224 / 112], 1, m![I % 224 / 112 = 1 # 2, I % 112, H % 128]>(half)
            .unpad::<m![I % 112, H % 128]>();
        let gs = gs
            .tile::<m![I % 224 / 112], 1, m![I % 224 / 112 = 1 # 2, I % 112, H % 128 / 16]>(half)
            .unpad::<m![I % 112, H % 128 / 16]>();
        let us = us
            .tile::<m![I % 224 / 112], 1, m![I % 224 / 112 = 1 # 2, I % 112, H % 128 / 16]>(half)
            .unpad::<m![I % 112, H % 128 / 16]>();
        // Interleave the compact FP8 metadata; the fetch-only VRF loader accepts one DM source.
        let mut blocks: DmTensor<
            f8e4m3,
            Chip,
            tile::Cluster,
            tile::Slices<H>,
            m![Term, I % 112, H % 128 / 16],
        > = DmTensor::new();
        gs.to_dm_view(
            &mut device.tdma,
            blocks
                .view_mut()
                .tile::<m![Term], 1, m![Term = 1 #{!} 2, I % 112, H % 128 / 16]>(0),
        );
        us.to_dm_view(
            &mut device.tdma,
            blocks
                .view_mut()
                .tile::<m![Term], 1, m![Term = 1 #{!} 2, I % 112, H % 128 / 16]>(1),
        );
        let scales: VrfTensor<
            f32,
            Chip,
            tile::Cluster,
            tile::Slices<H>,
            m![Term, I % 112, H % 128 / 16],
        > = device
            .sub
            .begin(blocks.view())
            .fetch::<m![Term, I % 112], m![H % 128 / 16]>()
            .fetch_cast::<f32>()
            .collect::<m![Term, I % 112], m![H % 128 / 16]>()
            .to_vrf();
        device.main.begin_interleaved::<Term, _, _, _, _, _>(g, u)
            .fetch::<m![R % {W} / 8, Term, I % 112 / 2, H % 32 / 16, H % 128 / 32, I % 2], m![H % 16]>()
            .fetch_table_lookup::<bf16>()
            .collect::<m![R % {W} / 8, Term, I % 112 / 2, H % 32 / 16, H % 128 / 32, I % 2], m![H % 16]>()
            .contract_outer::<m![R % {W} / 8, Term, I % 112 / 2, H % 32 / 16, H % 128 / 32], m![I % 2, H % 16], _, _, _>(lanes)
            .contract_packet::<m![I % 2]>()
            .contract_time::<m![R % {W} / 8, Term, I % 112 / 2, H % 32 / 16, H % 128 / 32]>()
            .contract_lane::<m![R % {W} / 8, Term, I % 112 / 2, H % 32 / 16, H % 128 / 32, I % 2], m![R % 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![R % {W} / 8, Term, I % 112 / 2, H % 32 / 16, H % 128 / 32, I % 2, R % 8 / 4], m![R % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scales)
            .vector_intra_slice_reduce::<H, m![R % {W} / 8, Term, I % 112 / 2, I % 2, R % 8 / 4], m![R % 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_concat::<m![R % {W} / 8, Term, I % 112 / 2, I % 2], m![R % 8]>()
            .vector_final()
            .transpose::<m![R % {W} / 8, Term, I % 112 / 2, R % 8], m![I % 2 # 8]>()
            .commit_trim::<m![I % 2]>()
            .commit_view(output.view_mut()
                .tile::<m![I % 224 / 112], 1, m![R % {W}, Term, I % 224 / 112 = 1 #{!} 2, I % 112]>(half));
    }
    output
}

fn term<H: AxisName, I: AxisName, R: AxisName, const W: usize>(
    device: &mut Device,
    pair: &Paired<H, I, R, W>,
    index: usize,
) -> Partial<H, I, R, W> {
    let mut output = Partial::new();
    pair.view()
        .tile::<m![Term], 1, m![R % {W}, Term = 1 # 2, I % 224]>(index)
        .to_dm_view(&mut device.tdma, output.view_mut());
    output
}

/// Apply this projection's factor once to the complete FP32 slab sum, before
/// the single inter-slice reduction and BF16 cast. The final slab add remains a
/// separate stream because FP AddF and inter-slice Add share a hardware ALU.
fn finish<H: AxisName, I: AxisName, R: AxisName, const W: usize>(
    device: &mut Device,
    scales: &super::super::Scales,
    sums: &Partial<H, I, R, W>,
) -> tile::Output<I, R, W> {
    let factor = block::scalar::<tile::Cluster, tile::Slices<H>>(device, &scales.output);
    device
        .main
        .begin(sums.view())
        .fetch::<m![R % { W }, I % 224 / 8], m![I % 8]>()
        .collect::<m![R % { W }, I % 224 / 8], m![I % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![R % { W }, I % 224 / 4], m![I % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &factor)
        .vector_widen_concat::<m![R % { W }, I % 224 / 8], m![I % 8]>()
        .vector_inter_slice_reduce::<tile::Reduced, m![R % { W }, I % 224 / 8]>(
            InterSliceReduceOpF32::Add,
        )
        .vector_final()
        .cast::<bf16, m![I % 8 # 16]>()
        .commit_trim::<m![I % 8]>()
        .commit()
}
