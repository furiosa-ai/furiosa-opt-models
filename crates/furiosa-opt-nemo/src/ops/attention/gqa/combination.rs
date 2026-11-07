use super::{PagedAttention, reduction};
use crate::Module;
use crate::ops::layout::Dm;
use furiosa_opt_std::prelude::*;

type Tags<'x, B, C, L, H, G, S: reduction::Mapping> =
    &'x VrfTensor<i4, B, C, L, m![H, G, { S::Entries }, 1 # 8]>;
type Peak<'x, B, C, L, H, G> = &'x VrfTensor<f32, B, C, L, m![H, G, 1 # 8]>;
const CLEARED: TagGuard =
    TagGuard::matches([BitReq::Zero, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]);
const ADMITTED: TagGuard =
    TagGuard::matches([BitReq::One, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]);

impl<
    'x,
    H: AxisName,
    G: AxisName,
    D,
    const SCALE: usize,
    B: M,
    C: M,
    L: M,
    E: M,
    S: reduction::Mapping,
>
    Module<(
        &mut Device,
        Dm<
            S,
            (
                DmTensorView<'x, f32, B, C, L, E>,
                Tags<'x, B, C, L, H, G, S>,
            ),
        >,
    )> for PagedAttention<H, G, D, SCALE, B>
{
    type Output = Dm<S, DmTensor<f32, B, C, L, m![H, G, 1 # 8]>>;

    fn forward(
        &self,
        (device, input): (
            &mut Device,
            Dm<
                S,
                (
                    DmTensorView<'x, f32, B, C, L, E>,
                    Tags<'x, B, C, L, H, G, S>,
                ),
            >,
        ),
    ) -> Self::Output {
        let (stats, tags) = input.value;
        Dm {
            tiles: input.tiles,
            value: device
                .main
                .begin(stats)
                .fetch::<m![H, G, { S::Entries }], m![1 # 8]>()
                .collect::<m![H, G, { S::Entries }], m![1 # 8]>()
                .vector_init()
                .vector_tagged(tags)
                .vector_logic(LogicBinaryOpF32::BitAnd, Branched::imm(CLEARED, 0.0f32))
                .vector_narrow_trim::<m![1 # 4]>()
                .vector_fp_binary(FpBinaryOp::AddF, Branched::imm(CLEARED, f32::NEG_INFINITY))
                .vector_intra_slice_reduce::<S::Entry, m![H, G], m![1 # 4]>(
                    IntraSliceReduceOpF32::Max,
                )
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit(),
        }
    }
}

impl<
    'x,
    H: AxisName,
    G: AxisName,
    D,
    const SCALE: usize,
    B: M,
    C: M,
    L: M,
    E: M,
    S: reduction::Mapping,
>
    Module<(
        &mut Device,
        Dm<
            S,
            (
                (DmTensorView<'x, f32, B, C, L, m![H, G, 1 # 8]>, (), ()),
                DmTensorViewMut<'x, f32, B, C, L, E>,
            ),
        >,
    )> for PagedAttention<H, G, D, SCALE, B>
{
    type Output = Dm<S, ()>;

    fn forward(
        &self,
        (device, input): (
            &mut Device,
            Dm<
                S,
                (
                    (DmTensorView<'x, f32, B, C, L, m![H, G, 1 # 8]>, (), ()),
                    DmTensorViewMut<'x, f32, B, C, L, E>,
                ),
            >,
        ),
    ) -> Self::Output {
        let ((local, (), ()), peak) = input.value;
        device
            .main
            .begin(local)
            .fetch::<m![H, G], m![1 # 8]>()
            .collect::<m![H, G], m![1 # 8]>()
            .vector_init()
            .vector_inter_slice_reduce_span(InterSliceReduceOpF32::Max, S::SPAN)
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Max, -1e30f32)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit_view(peak);
        Dm {
            tiles: input.tiles,
            value: (),
        }
    }
}

impl<
    'x,
    H: AxisName,
    G: AxisName,
    D,
    const SCALE: usize,
    B: M,
    C: M,
    L: M,
    E: M,
    S: reduction::Mapping,
> Module<(&mut Device, Dm<S, ((), DmTensorView<'x, f32, B, C, L, E>)>)>
    for PagedAttention<H, G, D, SCALE, B>
{
    type Output = Dm<S, VrfTensor<f32, B, C, L, m![H, G, 1 # 8]>>;

    fn forward(
        &self,
        (device, input): (&mut Device, Dm<S, ((), DmTensorView<'x, f32, B, C, L, E>)>),
    ) -> Self::Output {
        Dm {
            tiles: input.tiles,
            value: device
                .sub
                .begin(input.value.1)
                .fetch::<m![H, G], m![1 # 8]>()
                .collect::<m![H, G], m![1 # 8]>()
                .to_vrf(),
        }
    }
}

impl<
    'x,
    H: AxisName,
    G: AxisName,
    D,
    const SCALE: usize,
    B: M,
    C: M,
    L: M,
    E: M,
    S: reduction::Mapping,
>
    Module<(
        &mut Device,
        Dm<
            S,
            (
                DmTensorView<'x, f32, B, C, L, E>,
                Tags<'x, B, C, L, H, G, S>,
                Peak<'x, B, C, L, H, G>,
            ),
        >,
    )> for PagedAttention<H, G, D, SCALE, B>
{
    type Output = Dm<S, VrfTensor<f32, B, C, L, m![H, G, { S::Entries }, 1 # 8]>>;

    fn forward(
        &self,
        (device, input): (
            &mut Device,
            Dm<
                S,
                (
                    DmTensorView<'x, f32, B, C, L, E>,
                    Tags<'x, B, C, L, H, G, S>,
                    Peak<'x, B, C, L, H, G>,
                ),
            >,
        ),
    ) -> Self::Output {
        let (stats, tags, peak) = input.value;
        let scales: DmTensor<f32, B, C, L, m![H, G, { S::Entries }, 1 # 8]> = device
            .main
            .begin(stats)
            .fetch::<m![H, G, { S::Entries }], m![1 # 8]>()
            .collect::<m![H, G, { S::Entries }], m![1 # 8]>()
            .vector_init()
            .vector_tagged(tags)
            .vector_logic(LogicBinaryOpF32::BitAnd, Branched::imm(CLEARED, 0.0f32))
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(
                FpBinaryOp::SubF,
                Branched::imm(CLEARED, f32::INFINITY).rf(ADMITTED, peak),
            )
            .vector_fp_unary(FpUnaryOp::Exp)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        Dm {
            tiles: input.tiles,
            value: device
                .sub
                .begin(scales.view())
                .fetch::<m![H, G, { S::Entries }], m![1 # 8]>()
                .collect::<m![H, G, { S::Entries }], m![1 # 8]>()
                .to_vrf(),
        }
    }
}
