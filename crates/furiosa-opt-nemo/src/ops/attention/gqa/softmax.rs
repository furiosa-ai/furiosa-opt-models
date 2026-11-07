use super::{PagedAttention, visibility};
use crate::Module;
use crate::ops::layout::Dm;
use furiosa_opt_std::prelude::*;

mod weights;

impl<'x, H, G, D, const SCALE: usize, B: M, C: M, L: M, E: M, S: Mapping>
    Module<(
        &mut Device,
        Dm<
            S,
            (
                DmTensorView<'x, bf16, B, C, L, m![{ S::Rows }, { S::Groups }, { S::Keys } / 16]>,
                DmTensorView<'x, f32, B, C, L, E>,
                DmTensorViewMut<
                    'x,
                    bf16,
                    B,
                    C,
                    L,
                    m![{ S::Rows }, { S::Groups }, { S::Keys } / 16],
                >,
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
                    DmTensorView<
                        'x,
                        bf16,
                        B,
                        C,
                        L,
                        m![{ S::Rows }, { S::Groups }, { S::Keys } / 16],
                    >,
                    DmTensorView<'x, f32, B, C, L, E>,
                    DmTensorViewMut<
                        'x,
                        bf16,
                        B,
                        C,
                        L,
                        m![{ S::Rows }, { S::Groups }, { S::Keys } / 16],
                    >,
                ),
            >,
        ),
    ) -> Self::Output {
        let (scores, peak, weights) = input.value;
        let top: VrfTensor<f32, B, C, L, m![{ S::Rows }, { S::Groups }, 1 # 8]> = device
            .sub
            .begin(peak)
            .fetch::<m![{ S::Rows }, { S::Groups }], m![1 # 8]>()
            .collect::<m![{ S::Rows }, { S::Groups }], m![1 # 8]>()
            .to_vrf();
        self.forward((
            device,
            Dm {
                tiles: input.tiles,
                value: (scores, &top, weights),
            },
        ))
    }
}

pub trait Mapping: visibility::Mapping {
    type Key: AxisName;
}

impl<'x, H, G, D, const SCALE: usize, B: M, C: M, L: M, E: M, S: Mapping>
    Module<(
        &mut Device,
        Dm<
            S,
            (
                DmTensorView<'x, f32, B, C, L, m![{ S::Rows }, { S::Groups }, 1 # 8]>,
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
                    DmTensorView<'x, f32, B, C, L, m![{ S::Rows }, { S::Groups }, 1 # 8]>,
                    DmTensorViewMut<'x, f32, B, C, L, E>,
                ),
            >,
        ),
    ) -> Self::Output {
        let (local, peak) = input.value;
        device
            .main
            .begin(local)
            .fetch::<m![{ S::Rows }, { S::Groups }], m![1 # 8]>()
            .collect::<m![{ S::Rows }, { S::Groups }], m![1 # 8]>()
            .vector_init()
            .vector_inter_slice_reduce_span(InterSliceReduceOpF32::Max, 4)
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

impl<'x, H, G, D, const SCALE: usize, B: M, C: M, L: M, S: Mapping>
    Module<(
        &mut Device,
        Dm<
            S,
            (
                DmTensorView<'x, bf16, B, C, L, m![{ S::Rows }, { S::Groups }, { S::Keys } / 16]>,
                &'x VrfTensor<f32, B, C, L, m![{ S::Rows }, { S::Groups }, 1 # 8]>,
                DmTensorViewMut<
                    'x,
                    bf16,
                    B,
                    C,
                    L,
                    m![{ S::Rows }, { S::Groups }, { S::Keys } / 16],
                >,
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
                    DmTensorView<
                        'x,
                        bf16,
                        B,
                        C,
                        L,
                        m![{ S::Rows }, { S::Groups }, { S::Keys } / 16],
                    >,
                    &'x VrfTensor<f32, B, C, L, m![{ S::Rows }, { S::Groups }, 1 # 8]>,
                    DmTensorViewMut<
                        'x,
                        bf16,
                        B,
                        C,
                        L,
                        m![{ S::Rows }, { S::Groups }, { S::Keys } / 16],
                    >,
                ),
            >,
        ),
    ) -> Self::Output {
        let (scores, top, weights) = input.value;
        device
            .main
            .begin(scores)
            .fetch::<m![{ S::Rows }, { S::Groups }, { S::Keys } / 128], m![{ S::Keys } / 16 % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![{ S::Rows }, { S::Groups }, { S::Keys } / 128], m![{ S::Keys } / 16 % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ S::Rows }, { S::Groups }, { S::Keys } / 64], m![{ S::Keys } / 16 % 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, top)
            .vector_fp_unary(FpUnaryOp::Exp)
            .vector_widen_concat::<m![{ S::Rows }, { S::Groups }, { S::Keys } / 128], m![{ S::Keys } / 16 % 8]>()
            .vector_final()
            .cast::<bf16, m![{ S::Keys } / 16 % 8 # 16]>()
            .commit_trim::<m![{ S::Keys } / 16 % 8]>()
            .commit_view(weights);
        Dm {
            tiles: input.tiles,
            value: (),
        }
    }
}

impl<'x, H, G, D, const SCALE: usize, B: M, C: M, L: M, S: Mapping>
    Module<(
        &mut Device,
        Dm<S, DmTensorView<'x, bf16, B, C, L, m![{ S::Rows }, { S::Groups }, { S::Keys } / 16]>>,
    )> for PagedAttention<H, G, D, SCALE, B>
{
    type Output = Dm<S, DmTensor<f32, B, C, L, m![{ S::Rows }, { S::Groups }, 1 # 8]>>;

    fn forward(
        &self,
        (device, input): (
            &mut Device,
            Dm<
                S,
                DmTensorView<'x, bf16, B, C, L, m![{ S::Rows }, { S::Groups }, { S::Keys } / 16]>,
            >,
        ),
    ) -> Self::Output {
        Dm {
            tiles: input.tiles,
            value: device
                .main
                .begin(input.value)
                .fetch::<m![{ S::Rows }, { S::Groups }, { S::Keys } / 128], m![{ S::Keys } / 16 % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![{ S::Rows }, { S::Groups }, { S::Keys } / 128], m![{ S::Keys } / 16 % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![{ S::Rows }, { S::Groups }, { S::Keys } / 64], m![{ S::Keys } / 16 % 4]>()
                .vector_intra_slice_reduce::<S::Key, m![{ S::Rows }, { S::Groups }], m![1 # 4]>(
                    IntraSliceReduceOpF32::Max,
                )
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit(),
        }
    }
}
