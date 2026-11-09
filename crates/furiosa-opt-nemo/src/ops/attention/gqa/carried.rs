use super::{PagedAttention, visibility::Mapping};
use crate::Module;
use crate::ops::layout::Dm;
use furiosa_opt_std::prelude::*;

type Local<'x, B, C, L, S> =
    DmTensorView<'x, f32, B, C, L, m![{ <S as Mapping>::Rows }, { <S as Mapping>::Groups }, 1 # 8]>;
type Target<'x, B, C, L, S> = DmTensorViewMut<
    'x,
    f32,
    B,
    C,
    L,
    m![{ <S as Mapping>::Rows }, { <S as Mapping>::Groups }, 1 # 8],
>;
type Operand<B, C, L, S> =
    VrfTensor<f32, B, C, L, m![{ <S as Mapping>::Rows }, { <S as Mapping>::Groups }, 1 # 8]>;
type Peak<B, C, L, S> =
    DmTensor<f32, B, C, L, m![{ <S as Mapping>::Rows }, { <S as Mapping>::Groups }, 1 # 8]>;

axes![Term = 2];

type Cells<'x, B, C, L, E, S> =
    DmTensorView<'x, f32, B, C, L, m![{ <S as Mapping>::Rows }, { <S as Mapping>::Groups }, E]>;
type CellsMut<'x, B, C, L, E, S> =
    DmTensorViewMut<'x, f32, B, C, L, m![{ <S as Mapping>::Rows }, { <S as Mapping>::Groups }, E]>;

impl<'x, H, G, D, const SCALE: usize, B: M, C: M, L: M, S: Mapping>
    Module<(
        &mut Device,
        Dm<
            S,
            (
                (Local<'x, B, C, L, S>, Local<'x, B, C, L, S>),
                (&'x mut Peak<B, C, L, S>, Target<'x, B, C, L, S>),
            ),
        >,
    )> for PagedAttention<H, G, D, SCALE, B>
{
    type Output = Dm<S, Operand<B, C, L, S>>;

    fn forward(
        &self,
        (device, input): (
            &mut Device,
            Dm<
                S,
                (
                    (Local<'x, B, C, L, S>, Local<'x, B, C, L, S>),
                    (&'x mut Peak<B, C, L, S>, Target<'x, B, C, L, S>),
                ),
            >,
        ),
    ) -> Self::Output {
        let ((local, carried), (peak, scale)) = input.value;
        let before: Operand<B, C, L, S> = device
            .sub
            .begin(carried.clone())
            .fetch::<m![{ S::Rows }, { S::Groups }], m![1 # 8]>()
            .collect::<m![{ S::Rows }, { S::Groups }], m![1 # 8]>()
            .to_vrf();
        device
            .main
            .begin(local)
            .fetch::<m![{ S::Rows }, { S::Groups }], m![1 # 8]>()
            .collect::<m![{ S::Rows }, { S::Groups }], m![1 # 8]>()
            .vector_init()
            .vector_inter_slice_reduce_span(InterSliceReduceOpF32::Max, 4)
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Max, &before)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit_view(peak.view_mut());
        let top: Operand<B, C, L, S> = device
            .sub
            .begin(peak.view())
            .fetch::<m![{ S::Rows }, { S::Groups }], m![1 # 8]>()
            .collect::<m![{ S::Rows }, { S::Groups }], m![1 # 8]>()
            .to_vrf();
        device
            .main
            .begin(carried)
            .fetch::<m![{ S::Rows }, { S::Groups }], m![1 # 8]>()
            .collect::<m![{ S::Rows }, { S::Groups }], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, &top)
            .vector_fp_unary(FpUnaryOp::Exp)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit_view(scale);
        Dm {
            tiles: input.tiles,
            value: top,
        }
    }
}

impl<'x, H, G, D, const SCALE: usize, B: M, C: M, L: M, E: AxisName, S: Mapping>
    Module<(
        &mut Device,
        Dm<
            S,
            (
                (Cells<'x, B, C, L, E, S>, Cells<'x, B, C, L, E, S>),
                (&'x Operand<B, C, L, S>, CellsMut<'x, B, C, L, E, S>),
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
                    (Cells<'x, B, C, L, E, S>, Cells<'x, B, C, L, E, S>),
                    (&'x Operand<B, C, L, S>, CellsMut<'x, B, C, L, E, S>),
                ),
            >,
        ),
    ) -> Self::Output {
        let ((carried, local), (factor, mass)) = input.value;
        device
            .main
            .begin_interleaved::<Term, _, _, _, _, _>(carried, local)
            .fetch::<m![{ S::Rows }, { S::Groups }, Term], m![E]>()
            .collect::<m![{ S::Rows }, { S::Groups }, Term], m![E]>()
            .vector_init()
            .vector_intra_slice_unzip::<Term, m![{ S::Rows }, { S::Groups }]>()
            .vector_narrow_split::<m![{ S::Rows }, { S::Groups }, E / 4], m![E % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), factor, ())
            .vector_fp_zip(FpBinaryOp::AddF)
            .vector_widen_concat::<m![{ S::Rows }, { S::Groups }], m![E]>()
            .vector_final()
            .commit_trim::<m![E]>()
            .commit_view(mass);
        Dm {
            tiles: input.tiles,
            value: (),
        }
    }
}

impl<'x, H, G, D: AxisName, const SCALE: usize, B: M, C: M, L: M, S: Mapping>
    Module<(
        &mut Device,
        Dm<
            S,
            (
                (Cells<'x, B, C, L, D, S>, Cells<'x, B, C, L, D, S>),
                &'x Operand<B, C, L, S>,
                CellsMut<'x, B, C, L, D, S>,
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
                    (Cells<'x, B, C, L, D, S>, Cells<'x, B, C, L, D, S>),
                    &'x Operand<B, C, L, S>,
                    CellsMut<'x, B, C, L, D, S>,
                ),
            >,
        ),
    ) -> Self::Output {
        let ((carried, weighted), factor, sum) = input.value;
        device
            .main
            .begin_interleaved::<Term, _, _, _, _, _>(carried, weighted)
            .fetch::<m![{ S::Rows }, { S::Groups }, D / 8, Term], m![D % 8]>()
            .collect::<m![{ S::Rows }, { S::Groups }, D / 8, Term], m![D % 8]>()
            .vector_init()
            .vector_intra_slice_unzip::<Term, m![{ S::Rows }, { S::Groups }, D / 8]>()
            .vector_narrow_split::<m![{ S::Rows }, { S::Groups }, D / 4], m![D % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), factor, ())
            .vector_fp_zip(FpBinaryOp::AddF)
            .vector_widen_concat::<m![{ S::Rows }, { S::Groups }, D / 8], m![D % 8]>()
            .vector_final()
            .commit_trim::<m![D % 8]>()
            .commit_view(sum);
        Dm {
            tiles: input.tiles,
            value: (),
        }
    }
}
