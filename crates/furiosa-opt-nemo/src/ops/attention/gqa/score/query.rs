use super::*;

pub trait Mapping: super::Mapping {
    type Pair: M;
    type RowGroup: M;
    type Span: M;
}

type Sums<S> = m![
    { <S as Mapping>::Pair },
    { <S as Mapping>::Span } / 16,
    { <S as Mapping>::RowGroup },
    { <S as super::Mapping>::Groups },
    { <S as super::Mapping>::Keys } % 16 / 8
];
type Stream<S, D> = m![
    { <S as Mapping>::Pair },
    { <S as Mapping>::Span } / 16,
    { <S as Mapping>::RowGroup },
    { <S as super::Mapping>::Groups },
    D / 16
];
type Products<'d, B, C, L, S> = VectorTensor<
    'd,
    { Tu::Main },
    stage::Narrow,
    f32,
    B,
    C,
    L,
    m![{ Sums<S> }, { <S as super::Mapping>::Rows } % 2, { <S as super::Mapping>::Keys } % 8 / 4],
    m![{ <S as super::Mapping>::Keys } % 4],
    Fresh,
    { stage::VeOrder::IntraFirst },
    stage::Standalone,
    { stage::Way::Way4 },
>;

impl<'d, 'x: 'd, H, G, D: AxisName, const SCALE: usize, B: M, C: M, L: M, J: M, E: M, S: Mapping>
    Module<(
        &'d mut Device,
        Dm<
            S,
            (
                DmTensorView<'x, bf16, B, C, J, E>,
                &'x TrfTensor<
                    bf16,
                    B,
                    C,
                    L,
                    m![{ S::Keys } % 8],
                    m![{ S::Span } / 16, D / 16, { S::Keys } % 16 / 8, D % 16],
                >,
            ),
        >,
    )> for PagedAttention<H, G, D, SCALE, B>
{
    type Output = Dm<S, Products<'d, B, C, L, S>>;

    fn forward(
        &self,
        (device, input): (
            &'d mut Device,
            Dm<
                S,
                (
                    DmTensorView<'x, bf16, B, C, J, E>,
                    &'x TrfTensor<
                        bf16,
                        B,
                        C,
                        L,
                        m![{ S::Keys } % 8],
                        m![{ S::Span } / 16, D / 16, { S::Keys } % 16 / 8, D % 16],
                    >,
                ),
            >,
        ),
    ) -> Self::Output {
        let (query, key) = input.value;
        Dm {
            tiles: input.tiles,
            value: device
                .main
                .begin(query)
                .fetch::<m![{ S::Pair }, { S::Span } / 16, D / 16], m![{ S::Rows } % 2, D % 16]>()
                .switch::<L, Stream<S, D>>(SwitchConfig::Broadcast01 {
                    slice1: S::Keys::SIZE / S::Span::SIZE,
                    slice0: 1,
                    time0: D::SIZE / 16,
                })
                .collect::<m![{ Stream<S, D> }, { S::Rows } % 2], m![D % 16]>()
                .contract_outer::<m![{ Stream<S, D> }, { S::Keys } % 16 / 8], m![{ S::Rows } % 2, D % 16], _, _, _>(key)
                .contract_packet::<m![{ S::Rows } % 2]>()
                .contract_time::<Sums<S>>()
                .contract_lane::<m![{ Sums<S> }, { S::Rows } % 2], m![{ S::Keys } % 8]>(LaneMode::Interleaved)
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split(),
        }
    }
}
