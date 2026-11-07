use super::{PagedAttention, softmax::Mapping};
use crate::Module;
use crate::ops::layout::Dm;
use furiosa_opt_std::prelude::*;

impl<'x, H, G, D, const SCALE: usize, B: M, C: M, L: M, S: Mapping>
    Module<(
        &mut Device,
        Dm<S, (DmTensorView<'x, bf16, B, C, L, m![{ S::Rows }, { S::Groups }, { S::Keys } / 16]>,)>,
    )> for PagedAttention<H, G, D, SCALE, B>
{
    type Output = Dm<S, DmTensor<f32, B, C, L, m![{ S::Rows }, { S::Groups }, 1 # 8]>>;

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
                ),
            >,
        ),
    ) -> Self::Output {
        Dm {
            tiles: input.tiles,
            value: device
                .main
                .begin(input.value.0)
                .fetch::<m![{ S::Rows }, { S::Groups }, { S::Keys } / 128], m![{ S::Keys } / 16 % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![{ S::Rows }, { S::Groups }, { S::Keys } / 128], m![{ S::Keys } / 16 % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![{ S::Rows }, { S::Groups }, { S::Keys } / 64], m![{ S::Keys } / 16 % 4]>()
                .vector_intra_slice_reduce::<S::Key, m![{ S::Rows }, { S::Groups }], m![1 # 4]>(
                    IntraSliceReduceOpF32::Add,
                )
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit(),
        }
    }
}

impl<'x, H, G, D, const SCALE: usize, B: M, C: M, L: M, E: M, S: Mapping>
    Module<(
        &mut Device,
        Dm<
            S,
            (
                (DmTensorView<'x, f32, B, C, L, m![{ S::Rows }, { S::Groups }, 1 # 8]>,),
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
                    (DmTensorView<'x, f32, B, C, L, m![{ S::Rows }, { S::Groups }, 1 # 8]>,),
                    DmTensorViewMut<'x, f32, B, C, L, E>,
                ),
            >,
        ),
    ) -> Self::Output {
        let ((local,), mass) = input.value;
        device
            .main
            .begin(local)
            .fetch::<m![{ S::Rows }, { S::Groups }], m![1 # 8]>()
            .collect::<m![{ S::Rows }, { S::Groups }], m![1 # 8]>()
            .vector_init()
            .vector_inter_slice_reduce_span(InterSliceReduceOpF32::Add, 4)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit_view(mass);
        Dm {
            tiles: input.tiles,
            value: (),
        }
    }
}
