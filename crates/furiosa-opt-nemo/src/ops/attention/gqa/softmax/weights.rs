use super::*;
use crate::compose::Space;

type Scores<'d, B: M, C: M, L: M, T: M, P: M> = VectorTensor<
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
    Module<Dm<S, (Scores<'d, B, C, L, T, P>, &'x VrfTensor<f32, B, C, L, E>)>>
    for PagedAttention<H, G, D, SCALE, B>
{
    type Output = Dm<
        S,
        VectorFpTensor<'d, { Tu::Main }, f32, B, C, L, T, P, Fresh, { stage::VeOrder::IntraFirst }>,
    >;

    fn forward(
        &self,
        input: Dm<S, (Scores<'d, B, C, L, T, P>, &'x VrfTensor<f32, B, C, L, E>)>,
    ) -> Self::Output {
        let (scores, peak) = input.value;
        Dm {
            tiles: input.tiles,
            value: scores
                .vector_fp_ternary_with_mode(
                    FpTernaryOp::FmaF,
                    TernaryArgMode::Mode021,
                    (peak, -f32::from_bits(SCALE as u32)),
                )
                .vector_fp_unary(FpUnaryOp::NegExp),
        }
    }
}

type Weights<'d, B: M, C: M, L: M, T: M, P: M> = VectorTensor<
    'd,
    { Tu::Main },
    stage::Widen,
    f32,
    B,
    C,
    L,
    T,
    P,
    Fresh,
    { stage::VeOrder::IntraFirst },
    stage::Standalone,
    { stage::Way::Way8 },
>;

impl<'d, H, G, D, const SCALE: usize, B: M, C: M, L: M, T: M, P: M, S: Space>
    Module<Dm<S, (Weights<'d, B, C, L, T, P>, f32)>> for PagedAttention<H, G, D, SCALE, B>
{
    type Output = Dm<
        S,
        VectorClipTensor<
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

    fn forward(&self, input: Dm<S, (Weights<'d, B, C, L, T, P>, f32)>) -> Self::Output {
        let (weights, ceiling) = input.value;
        Dm {
            tiles: input.tiles,
            value: weights.vector_clip(ClipBinaryOpF32::Min, ceiling),
        }
    }
}
