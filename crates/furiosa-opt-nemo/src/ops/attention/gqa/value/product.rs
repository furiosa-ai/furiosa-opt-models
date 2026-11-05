use super::*;

pub trait Mapping: super::Mapping {
    type Channel: M;
}

pub trait Partitioned: Mapping {
    type Row: M;
    type RowGroup: M;
    type Span: M;
}

type Stream<S: Mapping> = m![{ S::Channel } / 32, { S::Keys } / 4, { S::Keys } % 4];

impl<'x, H, G, D, const SCALE: usize, B: M, C: M, L: M, J: M, E: M, S: Mapping>
    Module<(
        &mut Device,
        Dm<
            S,
            (
                DmTensorView<'x, bf16, B, C, L, m![{ S::Keys } % 4, { S::Channel }]>,
                &'x TrfTensor<
                    bf16,
                    B,
                    C,
                    J,
                    m![{ S::Rows } % 2, { S::Groups }],
                    m![{ S::Rows } % 4 / 2, { S::Keys }],
                >,
                (DmTensorViewMut<'x, f32, B, C, J, E>,),
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
                    DmTensorView<'x, bf16, B, C, L, m![{ S::Keys } % 4, { S::Channel }]>,
                    &'x TrfTensor<
                        bf16,
                        B,
                        C,
                        J,
                        m![{ S::Rows } % 2, { S::Groups }],
                        m![{ S::Rows } % 4 / 2, { S::Keys }],
                    >,
                    (DmTensorViewMut<'x, f32, B, C, J, E>,),
                ),
            >,
        ),
    ) -> Self::Output {
        let (value, weights, (product,)) = input.value;
        device
            .main
            .begin(value)
            .fetch::<m![{ S::Channel } / 32, { S::Keys } % 4], m![{ S::Channel } % 32]>()
            .switch::<J, Stream<S>>(SwitchConfig::Broadcast01 {
                slice1: S::Keys::SIZE / 4,
                slice0: 1,
                time0: 4,
            })
            .collect::<m![{ Stream<S> }, { S::Channel } % 32 / 16], m![{ S::Channel } % 16]>()
            .contract_outer::<m![{ Stream<S> }, { S::Rows } % 4 / 2], m![{ S::Channel } % 32], _, _, _>(weights)
            .contract_packet::<m![{ S::Channel } % 32]>()
            .contract_time::<m![{ S::Channel } / 32, { S::Rows } % 4 / 2]>()
            .contract_lane::<m![
                { S::Channel } / 32,
                { S::Rows } % 4 / 2,
                { S::Rows } % 2,
                { S::Groups },
                { S::Channel } % 32 / 8
            ], m![{ S::Channel } % 8]>(LaneMode::Sequential)
            .commit_trim::<m![{ S::Channel } % 8]>()
            .commit_view(product);
        Dm {
            tiles: input.tiles,
            value: (),
        }
    }
}

type Fetch<S: Partitioned> = m![
    { S::Row },
    { S::Channel } / 16,
    { S::RowGroup },
    { S::Groups } / 2,
    { S::Span } / 16
];
type Sums<S: Partitioned> = m![
    { S::Row },
    { S::Channel } / 16,
    { S::RowGroup },
    { S::Groups } / 2,
    { S::Channel } % 16 / 8
];
type Values<B: M, C: M, L: M, S: Partitioned> = TrfTensor<
    bf16,
    B,
    C,
    L,
    m![{ S::Channel } % 8],
    m![
        { S::Channel } / 16,
        { S::Span } / 16,
        { S::Channel } % 16 / 8,
        { S::Keys } % 16
    ],
>;

impl<'x, H, G, D, const SCALE: usize, B: M, C: M, L: M, J: M, E: M, F: M, S: Partitioned>
    Module<(
        &mut Device,
        Dm<
            S,
            (
                DmTensorView<'x, bf16, B, C, L, E>,
                Values<B, C, L, S>,
                (DmTensorViewMut<'x, f32, B, C, J, F>,),
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
                    DmTensorView<'x, bf16, B, C, L, E>,
                    Values<B, C, L, S>,
                    (DmTensorViewMut<'x, f32, B, C, J, F>,),
                ),
            >,
        ),
    ) -> Self::Output {
        let (weights, value, (product,)) = input.value;
        device
            .main
            .begin(weights)
            .fetch::<Fetch<S>, m![{ S::Groups } % 2, { S::Keys } % 16]>()
            .collect::<m![{ Fetch<S> }, { S::Groups } % 2], m![{ S::Keys } % 16]>()
            .contract_outer::<m![{ Fetch<S> }, { S::Channel } % 16 / 8], m![{ S::Groups } % 2, { S::Keys } % 16], _, _, _>(&value)
            .contract_packet::<m![{ S::Groups } % 2]>()
            .contract_time::<Sums<S>>()
            .contract_lane::<m![{ Sums<S> }, { S::Groups } % 2], m![{ S::Channel } % 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<J, m![{ S::Row }, { S::Channel } / 8]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .commit_trim::<m![{ S::Channel } % 8]>()
            .commit_view(product);
        Dm {
            tiles: input.tiles,
            value: (),
        }
    }
}
