use super::{PagedAttention, visibility::Mapping};
use crate::Module;
use crate::compose::Space;
use crate::ops::layout::Dm;
use furiosa_opt_std::prelude::*;

pub mod query;

impl<'x, H, G, D: AxisName, const SCALE: usize, B: M, C: M, L: M, J: M, S: Mapping>
    Module<(
        &mut Device,
        Dm<
            S,
            (
                DmTensorView<'x, bf16, B, C, L, m![D / 8, { S::Keys } % 4, D % 8]>,
                &'x TrfTensor<
                    bf16,
                    B,
                    C,
                    J,
                    m![{ S::Rows } % 2, { S::Groups }],
                    m![{ S::Rows } % 4 / 2, D],
                >,
                DmTensorViewMut<
                    'x,
                    f32,
                    B,
                    C,
                    J,
                    m![{ S::Rows } % 2, { S::Groups }, { S::Rows } % 4 / 2, {
                        S::Keys
                    }],
                >,
            ),
        >,
    )> for PagedAttention<H, G, D, SCALE, B>
{
    type Output = ();

    fn forward(
        &self,
        (device, input): (
            &mut Device,
            Dm<
                S,
                (
                    DmTensorView<'x, bf16, B, C, L, m![D / 8, { S::Keys } % 4, D % 8]>,
                    &'x TrfTensor<
                        bf16,
                        B,
                        C,
                        J,
                        m![{ S::Rows } % 2, { S::Groups }],
                        m![{ S::Rows } % 4 / 2, D],
                    >,
                    DmTensorViewMut<
                        'x,
                        f32,
                        B,
                        C,
                        J,
                        m![{ S::Rows } % 2, { S::Groups }, { S::Rows } % 4 / 2, {
                            S::Keys
                        }],
                    >,
                ),
            >,
        ),
    ) {
        let (key, query, scores) = input.value;
        device
            .main
            .begin(key)
            .fetch::<m![D / 8], m![{ S::Keys } % 4, D % 8]>()
            .switch::<J, m![{ S::Keys } / 4, D / 8]>(SwitchConfig::Broadcast01 {
                slice1: S::Keys::SIZE / 4,
                slice0: 1,
                time0: D::SIZE / 8,
            })
            .collect::<m![{ S::Keys } / 4, D / 8, { S::Keys } % 4 / 2], m![{ S::Keys } % 2, D % 8]>()
            .contract_outer::<m![{ S::Keys } / 4, D / 8, { S::Rows } % 4 / 2], m![{ S::Keys } % 4, D % 8], _, _, _>(query)
            .contract_packet::<m![{ S::Keys } % 4]>()
            .contract_time::<m![{ S::Keys } / 4, { S::Rows } % 4 / 2]>()
            .contract_lane::<m![{ S::Keys } / 4, { S::Rows } % 4 / 2, { S::Rows } % 2, { S::Groups }], m![{ S::Keys } % 4 # 8]>(LaneMode::Sequential)
            .commit_trim::<m![{ S::Keys } % 4]>()
            .commit_view(scores);
    }
}

impl<'d, H, G, D, const SCALE: usize, B: M, C: M, L: M, T: M, P: M, S: Space>
    Module<
        Dm<
            S,
            VectorTensor<
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
            >,
        >,
    > for PagedAttention<H, G, D, SCALE, B>
{
    type Output = Dm<
        S,
        VectorFpTensor<'d, { Tu::Main }, f32, B, C, L, T, P, Fresh, { stage::VeOrder::IntraFirst }>,
    >;

    fn forward(
        &self,
        input: Dm<
            S,
            VectorTensor<
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
            >,
        >,
    ) -> Self::Output {
        Dm {
            tiles: input.tiles,
            value: input.value.vector_fp_binary(
                FpBinaryOp::MulF(FpMulAlu::Mul0),
                f32::from_bits(SCALE as u32),
            ),
        }
    }
}

impl<'x, H, G, D: AxisName, const SCALE: usize, B: M, C: M, L: M, S: Mapping>
    Module<(
        &mut Device,
        Dm<
            S,
            (
                DmTensorView<'x, bf16, B, C, L, m![{ S::Rows }, { S::Keys } / 16, D]>,
                &'x TrfTensor<bf16, B, C, L, m![{ S::Groups }], m![{ S::Rows }, D]>,
            ),
        >,
    )> for PagedAttention<H, G, D, SCALE, B>
{
    type Output = Dm<S, DmTensor<f32, B, C, L, m![{ S::Rows }, { S::Groups }, { S::Keys } / 16]>>;

    fn forward(
        &self,
        (device, input): (
            &mut Device,
            Dm<
                S,
                (
                    DmTensorView<'x, bf16, B, C, L, m![{ S::Rows }, { S::Keys } / 16, D]>,
                    &'x TrfTensor<bf16, B, C, L, m![{ S::Groups }], m![{ S::Rows }, D]>,
                ),
            >,
        ),
    ) -> Self::Output {
        let (key, query) = input.value;
        Dm {
            tiles: input.tiles,
            value: device
                .main
                .begin(key)
                .fetch::<m![{ S::Rows }, { S::Keys } / 16, D / 16], m![D % 16]>()
                .collect::<m![{ S::Rows }, { S::Keys } / 16, D / 16], m![D % 16]>()
                .contract_outer::<m![{ S::Rows }, { S::Keys } / 16, D / 32], m![D % 32], _, _, _>(query)
                .contract_packet::<m![1]>()
                .contract_time::<m![{ S::Rows }, { S::Keys } / 16]>()
                .contract_lane::<m![{ S::Rows }, { S::Keys } / 16], m![{ S::Groups } # 8]>(
                    LaneMode::Interleaved,
                )
                .transpose::<m![{ S::Rows }, { S::Keys } / 32, { S::Groups }], m![{ S::Keys } / 16 % 2 # 8]>()
                .commit_trim::<m![{ S::Keys } / 16 % 2]>()
                .commit(),
        }
    }
}
