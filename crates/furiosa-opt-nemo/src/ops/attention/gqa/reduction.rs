use super::{PagedAttention, completion};
use crate::Module;
use crate::ops::layout::Dm;
use furiosa_opt_std::prelude::*;

pub trait Mapping: completion::Mapping {
    type Entries: M;
    type Entry: AxisName;
}

type Terms<'x, B, C, L, H, G, S> =
    DmTensorView<'x, f32, B, C, L, m![H, G, { <S as Mapping>::Entries }, 1 # 8]>;
type Values<'x, B, C, L, H, G, D, S> =
    DmTensorView<'x, f32, B, C, L, m![H, G, { <S as Mapping>::Entries }, D]>;

impl<'x, H: AxisName, G: AxisName, D: AxisName, const SCALE: usize, B: M, C: M, L: M, S: Mapping>
    Module<(
        &mut Device,
        Dm<S, ((DmTensorView<'x, f32, B, C, L, m![H, G, D]>,), ())>,
    )> for PagedAttention<H, G, D, SCALE, B>
{
    type Output = Dm<S, DmTensor<bf16, B, C, L, m![H, G, D]>>;

    fn forward(
        &self,
        (device, input): (
            &mut Device,
            Dm<S, ((DmTensorView<'x, f32, B, C, L, m![H, G, D]>,), ())>,
        ),
    ) -> Self::Output {
        Dm {
            tiles: input.tiles,
            value: device
                .main
                .begin(input.value.0.0)
                .fetch::<m![H, G, D / 8], m![D % 8]>()
                .collect::<m![H, G, D / 8], m![D % 8]>()
                .vector_init()
                .vector_inter_slice_reduce_span(InterSliceReduceOpF32::Add, S::SPAN)
                .vector_final()
                .cast::<bf16, m![D % 8 # 16]>()
                .commit_trim::<m![D % 8]>()
                .commit(),
        }
    }
}

impl<'x, H: AxisName, G: AxisName, D, const SCALE: usize, B: M, C: M, L: M, S: Mapping>
    Module<(&mut Device, Dm<S, (Terms<'x, B, C, L, H, G, S>, (), ())>)>
    for PagedAttention<H, G, D, SCALE, B>
{
    type Output = Dm<S, DmTensor<f32, B, C, L, m![H, G, 1 # 8]>>;

    fn forward(
        &self,
        (device, input): (&mut Device, Dm<S, (Terms<'x, B, C, L, H, G, S>, (), ())>),
    ) -> Self::Output {
        let (terms, (), ()) = input.value;
        let local: DmTensor<f32, B, C, L, m![H, G, 1 # 8]> = device
            .main
            .begin(terms)
            .fetch::<m![H, G, { S::Entries }], m![1 # 8]>()
            .collect::<m![H, G, { S::Entries }], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_intra_slice_reduce::<S::Entry, m![H, G], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        Dm {
            tiles: input.tiles,
            value: local,
        }
    }
}

impl<'x, H: AxisName, G: AxisName, D, const SCALE: usize, B: M, C: M, L: M, E: M, S: Mapping>
    Module<(
        &mut Device,
        Dm<
            S,
            (
                (DmTensorView<'x, f32, B, C, L, m![H, G, 1 # 8]>, ()),
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
                    (DmTensorView<'x, f32, B, C, L, m![H, G, 1 # 8]>, ()),
                    DmTensorViewMut<'x, f32, B, C, L, E>,
                ),
            >,
        ),
    ) -> Self::Output {
        let ((local, ()), mass) = input.value;
        device
            .main
            .begin(local)
            .fetch::<m![H, G], m![1 # 8]>()
            .collect::<m![H, G], m![1 # 8]>()
            .vector_init()
            .vector_inter_slice_reduce_span(InterSliceReduceOpF32::Add, S::SPAN)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit_view(mass);
        Dm {
            tiles: input.tiles,
            value: (),
        }
    }
}

impl<'x, H: AxisName, G: AxisName, D: AxisName, const SCALE: usize, B: M, C: M, L: M, S: Mapping>
    Module<(&mut Device, Dm<S, ((Values<'x, B, C, L, H, G, D, S>, ()),)>)>
    for PagedAttention<H, G, D, SCALE, B>
{
    type Output = Dm<S, DmTensor<f32, B, C, L, m![H, G, D]>>;

    fn forward(
        &self,
        (device, input): (&mut Device, Dm<S, ((Values<'x, B, C, L, H, G, D, S>, ()),)>),
    ) -> Self::Output {
        Dm {
            tiles: input.tiles,
            value: device
                .main
                .begin(input.value.0.0)
                .fetch::<m![H, G, D / 8, { S::Entries }], m![D % 8]>()
                .collect::<m![H, G, D / 8, { S::Entries }], m![D % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![H, G, D / 8, { S::Entries }, D % 8 / 4], m![D % 4]>()
                .vector_intra_slice_reduce::<S::Entry, m![H, G, D / 4], m![D % 4]>(
                    IntraSliceReduceOpF32::Add,
                )
                .vector_widen_concat::<m![H, G, D / 8], m![D % 8]>()
                .vector_final()
                .commit_trim::<m![D % 8]>()
                .commit(),
        }
    }
}
