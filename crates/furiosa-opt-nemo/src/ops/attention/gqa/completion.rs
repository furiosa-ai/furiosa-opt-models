use super::PagedAttention;
use crate::Module;
use crate::compose::Space;
use crate::ops::layout::Dm;
use furiosa_opt_std::prelude::*;

pub trait Mapping: Space {
    const SPAN: usize;
}

pub trait Accumulation: Space {
    type Groups: M;
    type Channel: M;
    type Term: AxisName;
    type Terms: M;
}

impl<'x, H, G, D, const SCALE: usize, B: M, C: M, L: M, E: M, S: Accumulation>
    Module<(
        &mut Device,
        Dm<
            S,
            ((
                DmTensorView<'x, f32, B, C, L, E>,
                DmTensorView<'x, f32, B, C, L, E>,
            ),),
        >,
    )> for PagedAttention<H, G, D, SCALE, B>
{
    type Output = Dm<S, DmTensor<f32, B, C, L, E>>;

    fn forward(
        &self,
        (device, input): (
            &mut Device,
            Dm<
                S,
                ((
                    DmTensorView<'x, f32, B, C, L, E>,
                    DmTensorView<'x, f32, B, C, L, E>,
                ),),
            >,
        ),
    ) -> Self::Output {
        let ((earlier, product),) = input.value;
        Dm {
            tiles: input.tiles,
            value: device
                .main
                .begin_interleaved::<S::Term, _, _, _, _, _>(earlier, product)
                .fetch::<m![{ S::Groups }, { S::Terms }, { S::Channel } / 8], m![{ S::Channel } % 8]>()
                .collect::<m![{ S::Groups }, { S::Terms }, { S::Channel } / 8], m![{ S::Channel } % 8]>()
                .vector_init()
                .vector_intra_slice_unzip::<S::Term, m![{ S::Groups }, { S::Channel } / 8]>()
                .vector_clip_zip(ClipBinaryOpF32::Add)
                .vector_final()
                .commit_trim::<m![{ S::Channel } % 8]>()
                .commit(),
        }
    }
}

type Product<'d, B: M, C: M, L: M, T: M, P: M> = VectorTensor<
    'd,
    { Tu::Main },
    stage::Narrow,
    f32,
    B,
    C,
    L,
    T,
    P,
    Fresh,
    { stage::VeOrder::IntraFirst },
    stage::Standalone,
    { stage::Way::Way4 },
>;

impl<'d, 'x, H, G, D, const SCALE: usize, B: M, C: M, L: M, T: M, P: M, E: M, S: Space>
    Module<
        Dm<
            S,
            (
                Product<'d, B, C, L, T, P>,
                (&'x VrfTensor<f32, B, C, L, E>,),
            ),
        >,
    > for PagedAttention<H, G, D, SCALE, B>
{
    type Output = Dm<
        S,
        VectorFpDivTensor<
            'd,
            { Tu::Main },
            f32,
            B,
            C,
            L,
            T,
            P,
            Fresh,
            { stage::VeOrder::IntraFirst },
        >,
    >;

    fn forward(
        &self,
        input: Dm<
            S,
            (
                Product<'d, B, C, L, T, P>,
                (&'x VrfTensor<f32, B, C, L, E>,),
            ),
        >,
    ) -> Self::Output {
        let (product, (mass,)) = input.value;
        Dm {
            tiles: input.tiles,
            value: product.vector_fp_div(mass),
        }
    }
}

impl<'x, H: AxisName, G: AxisName, D, const SCALE: usize, B: M, C: M, L: M, E: M, S: Mapping>
    Module<(&mut Device, Dm<S, (DmTensorView<'x, f32, B, C, L, E>, ())>)>
    for PagedAttention<H, G, D, SCALE, B>
{
    type Output = Dm<S, VrfTensor<f32, B, C, L, m![H, G, 1 # 8]>>;

    fn forward(
        &self,
        (device, input): (&mut Device, Dm<S, (DmTensorView<'x, f32, B, C, L, E>, ())>),
    ) -> Self::Output {
        Dm {
            tiles: input.tiles,
            value: device
                .main
                .begin(input.value.0)
                .fetch::<m![H, G], m![1 # 8]>()
                .collect::<m![H, G], m![1 # 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_clip(ClipBinaryOpF32::Add, 1e-20f32)
                .vector_final()
                .to_vrf(&mut device.sub),
        }
    }
}

impl<'x, H: AxisName, G: AxisName, D: AxisName, const SCALE: usize, B: M, C: M, L: M, S: Mapping>
    Module<(
        &mut Device,
        Dm<
            S,
            (
                DmTensorView<'x, f32, B, C, L, m![H, G, D]>,
                &'x VrfTensor<f32, B, C, L, m![H, G, 1 # 8]>,
            ),
        >,
    )> for PagedAttention<H, G, D, SCALE, B>
{
    type Output = Dm<S, DmTensor<f32, B, C, L, m![H, G, D]>>;

    fn forward(
        &self,
        (device, input): (
            &mut Device,
            Dm<
                S,
                (
                    DmTensorView<'x, f32, B, C, L, m![H, G, D]>,
                    &'x VrfTensor<f32, B, C, L, m![H, G, 1 # 8]>,
                ),
            >,
        ),
    ) -> Self::Output {
        let (partial, mass) = input.value;
        Dm {
            tiles: input.tiles,
            value: device
                .main
                .begin(partial)
                .fetch::<m![H, G, D / 8], m![D % 8]>()
                .collect::<m![H, G, D / 8], m![D % 8]>()
                .vector_init()
                .vector_inter_slice_reduce_span(InterSliceReduceOpF32::Add, S::SPAN)
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![H, G, D / 4], m![D % 4]>()
                .vector_fp_div(mass)
                .vector_widen_concat::<m![H, G, D / 8], m![D % 8]>()
                .vector_final()
                .commit_trim::<m![D % 8]>()
                .commit(),
        }
    }
}
