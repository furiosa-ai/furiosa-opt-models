use super::{PagedAttention, visibility::Mapping};
use crate::Module;
use crate::ops::layout::Dm;
use furiosa_opt_std::prelude::*;

pub mod product;

impl<'x, H, G, D, const SCALE: usize, B: M, C: M, L: M, S: Mapping>
    Module<(
        &mut Device,
        Dm<
            S,
            (
                DmTensorView<'x, bf16, B, C, L, m![{ S::Rows }, { S::Groups }, { S::Keys } / 16]>,
                (),
            ),
        >,
    )> for PagedAttention<H, G, D, SCALE, B>
{
    type Output =
        Dm<S, TrfTensor<bf16, B, C, L, m![{ S::Groups }], m![{ S::Rows }, { S::Keys } / 16]>>;

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
                    (),
                ),
            >,
        ),
    ) -> Self::Output {
        Dm {
            tiles: input.tiles,
            value: device
                .sub
                .begin(input.value.0)
                .fetch::<m![{ S::Groups }, { S::Rows }, { S::Keys } / 256], m![{ S::Keys } / 16 % 16]>()
                .collect::<m![{ S::Groups }, { S::Rows }, { S::Keys } / 256], m![{ S::Keys } / 16 % 16]>()
                .to_trf(),
        }
    }
}

impl<'x, H, G, D: AxisName, const SCALE: usize, B: M, C: M, L: M, S: Mapping>
    Module<(
        &mut Device,
        Dm<
            S,
            ((
                DmTensorView<'x, bf16, B, C, L, m![{ S::Rows }, { S::Keys } / 16, D]>,
                &'x TrfTensor<bf16, B, C, L, m![{ S::Groups }], m![{ S::Rows }, { S::Keys } / 16]>,
            ),),
        >,
    )> for PagedAttention<H, G, D, SCALE, B>
{
    type Output = Dm<S, DmTensor<f32, B, C, L, m![{ S::Rows }, { S::Groups }, D]>>;

    fn forward(
        &self,
        (device, input): (
            &mut Device,
            Dm<
                S,
                ((
                    DmTensorView<'x, bf16, B, C, L, m![{ S::Rows }, { S::Keys } / 16, D]>,
                    &'x TrfTensor<
                        bf16,
                        B,
                        C,
                        L,
                        m![{ S::Groups }],
                        m![{ S::Rows }, { S::Keys } / 16],
                    >,
                ),),
            >,
        ),
    ) -> Self::Output {
        let ((value, weights),) = input.value;
        Dm {
            tiles: input.tiles,
            value: device
                .main
                .begin(value)
                .fetch::<m![{ S::Rows }, D / 32, { S::Keys } / 16], m![D % 32]>()
                .collect::<m![{ S::Rows }, D / 32, { S::Keys } / 16, D % 32 / 16], m![D % 16]>()
                .contract_outer::<m![{ S::Rows }, D / 32, { S::Keys } / 16], m![D % 32], _, _, _>(
                    weights,
                )
                .contract_packet::<m![D % 32]>()
                .contract_time::<m![{ S::Rows }, D / 32]>()
                .contract_lane::<m![{ S::Rows }, D / 32, { S::Groups }, D % 32 / 8], m![D % 8]>(
                    LaneMode::Sequential,
                )
                .vector_init()
                .vector_inter_slice_reduce_span(InterSliceReduceOpF32::Add, 4)
                .vector_final()
                .commit_trim::<m![D % 8]>()
                .commit(),
        }
    }
}

impl<'x, H, G, D: AxisName, const SCALE: usize, B: M, C: M, L: M, E: M, S: Mapping>
    Module<(
        &mut Device,
        Dm<
            S,
            (
                DmTensorView<'x, f32, B, C, L, m![{ S::Rows }, { S::Groups }, D]>,
                (DmTensorViewMut<'x, bf16, B, C, L, E>,),
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
                    DmTensorView<'x, f32, B, C, L, m![{ S::Rows }, { S::Groups }, D]>,
                    (DmTensorViewMut<'x, bf16, B, C, L, E>,),
                ),
            >,
        ),
    ) -> Self::Output {
        let (partial, (rounded,)) = input.value;
        // Main completion makes the rounded contraction visible to the following DMA.
        device
            .main
            .begin(partial)
            .fetch::<m![{ S::Rows }, { S::Groups }, D / 8], m![D % 8]>()
            .collect::<m![{ S::Rows }, { S::Groups }, D / 8], m![D % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Add, 0.0f32)
            .vector_final()
            .cast::<bf16, m![D % 8 # 16]>()
            .commit_trim::<m![D % 8]>()
            .commit_view(rounded);
        Dm {
            tiles: input.tiles,
            value: (),
        }
    }
}
