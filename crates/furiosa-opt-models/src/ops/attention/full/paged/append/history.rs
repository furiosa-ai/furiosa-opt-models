use super::*;
use crate::Op;
use crate::ops::index::gather::Gather;
use crate::ops::index::gather::{Prefix, Read};
use std::marker::PhantomData;

pub(super) type DmSlices<KvHeads> = m![KvHeads % 4, Cache / Block # 64];
pub(super) type Tokens<KvHeads, Head> =
    DmTensor<bf16, Chip, Cluster<KvHeads>, DmSlices<KvHeads>, m![Cache % Block, Head]>;
/// Each token's distance to the end of its keys, negative where it lies before the end.
pub(super) type Distance<KvHeads> =
    DmTensor<i32, Chip, Cluster<KvHeads>, DmSlices<KvHeads>, m![Cache % Block]>;
/// Tokens tagged where their distance is not negative, across every head channel.
pub(super) type Ended<KvHeads, Head> =
    VrfTensor<i4, Chip, Cluster<KvHeads>, DmSlices<KvHeads>, m![Cache % Block, Head]>;

pub(crate) struct LastPanel;

impl<KvHeads: AxisName, Head: AxisName>
    Op<(
        (
            (&Gather, &Blocks<KvHeads, Head>),
            &HbmTensor<i32, Chip, m![Cache / Block]>,
            &Value,
        ),
        usize,
    )> for LastPanel
{
    type Output = (Tokens<KvHeads, Head>, Tokens<KvHeads, Head>);

    fn forward(
        &self,
        context: &mut Device,
        (input, at): (
            (
                (&Gather, &Blocks<KvHeads, Head>),
                &HbmTensor<i32, Chip, m![Cache / Block]>,
                &Value,
            ),
            usize,
        ),
    ) -> Self::Output {
        let (pool, entries, length) = input;
        // SAFETY: the caller bounds every block offset by the pool and initializes the live prefix.
        let packed: DmTensor<
            bf16,
            Chip,
            Cluster<KvHeads>,
            DmSlices<KvHeads>,
            m![Block, KeyValue, Head],
        > = Op::forward(
            pool.0,
            &mut *context,
            Prefix {
                read: Read {
                    pool: pool.1,
                    indices: entries,
                    output: PhantomData,
                },
                lengths: length,
                offset: (at * 128) as i32,
                sparse: PhantomData::<m![Cache]>,
            },
        );
        // SAFETY: entries are ordered by cache block, so each pool row is a Cache % Block row.
        let rows = unsafe {
            packed
                .view()
                .reshape::<
                    Chip,
                    Cluster<KvHeads>,
                    DmSlices<KvHeads>,
                    m![Cache % Block, KeyValue, Head],
                >()
        };
        let mut keys = Tokens::<KvHeads, Head>::new();
        let mut values = Tokens::<KvHeads, Head>::new();
        rows.tile::<m![KeyValue], 1, m![Cache % Block, KeyValue = 1 # 2, Head]>(0)
            .to_dm_view(&mut context.tdma, keys.view_mut());
        rows.tile::<m![KeyValue], 1, m![Cache % Block, KeyValue = 1 # 2, Head]>(1)
            .to_dm_view(&mut context.tdma, values.view_mut());
        (keys, values)
    }
}

/// The panel axis of a `P`-wide block, and one panel's tile of each block owner.
type Panels<P> = m![{ Symbol<<P as Panel>::Span> }];
type PanelOf<P> = m![{ Symbol<<P as Panel>::Span> } = 1 # ({ Symbol<<P as Panel>::Span> })];
type PanelSlot<P> = m![{ Symbol<<P as Panel>::Span> } = 1 #{!} ({ Symbol<<P as Panel>::Span> })];
type KeyTile<Head, P> = m![{ PanelOf<P> }, Cache % 2, Head];
type ValueTile<Head, P> = m![{ PanelOf<P> }, Cache % 16, Head % 16];
type Keys<P> = m![{ Symbol<<P as Panel>::Keys> }];
type PanelCount<P, W> = m![{ Symbol<<P as Panel>::Count<W>> }];
type Tiles<T> = m![{ Symbol<T> }];
pub(crate) type Group<KvHeads, Groups, Head, T> = DmTensor<
    bf16,
    Chip,
    Cluster<KvHeads>,
    Slice<KvHeads, Groups>,
    m![{ Tiles<T> }, Rows % 8, Head],
>;

pub(crate) type InDm<KvHeads, Head, P> = DmTensor<
    bf16,
    Chip,
    Cluster<KvHeads>,
    m![KvHeads % 4, { Keys<P> } % 128 / Block # 64],
    m![{ Keys<P> } / 128, Block, KeyValue, Head],
>;
type Staged<KvHeads, Head, P> = DmTensor<
    bf16,
    Chip,
    Cluster<KvHeads>,
    fresh::KeyStage<KvHeads>,
    m![{ Panels<P> }, Cache % 2, Head],
>;
/// A value for every key of a `P`-wide block in each row of `T` query tiles.
pub(crate) type Keyed<KvHeads, Groups, P, T> = DmTensor<
    bf16,
    Chip,
    Cluster<KvHeads>,
    Slice<KvHeads, Groups>,
    m![{ Tiles<T> }, Rows % 8, { Keys<P> }],
>;

pub(crate) struct Gathered<P>(pub(crate) PhantomData<P>);

/// Whole history block `at` of width `P`, every row of it committed history: read whole, so no
/// length is staged for it.
impl<KvHeads: AxisName, Head: AxisName, P: Panel, W: AxisName>
    Op<((&Table<W>, (&Gather, &Blocks<KvHeads, Head>)), usize)> for Gathered<P>
{
    type Output = InDm<KvHeads, Head, P>;

    fn forward(
        &self,
        context: &mut Device,
        ((table, pool), at): ((&Table<W>, (&Gather, &Blocks<KvHeads, Head>)), usize),
    ) -> Self::Output {
        // SAFETY: a whole block lies below h, so every gathered row is committed history.
        Op::forward(
            pool.0,
            context,
            Read {
                pool: pool.1,
                indices: &Self::entries(table, at),
                output: PhantomData,
            },
        )
    }
}

/// History block `at` of width `P` up to the history's `length`: the rows past it are unspecified.
impl<KvHeads: AxisName, Head: AxisName, P: Panel, W: AxisName>
    Op<(
        (&Table<W>, (&Gather, &Blocks<KvHeads, Head>), &Value),
        usize,
    )> for Gathered<P>
{
    type Output = InDm<KvHeads, Head, P>;

    fn forward(
        &self,
        context: &mut Device,
        ((table, pool, length), at): (
            (&Table<W>, (&Gather, &Blocks<KvHeads, Head>), &Value),
            usize,
        ),
    ) -> Self::Output {
        // SAFETY: the prefix reads committed history only; the table stays readable past it.
        Op::forward(
            pool.0,
            context,
            Prefix {
                read: Read {
                    pool: pool.1,
                    indices: &Self::entries(table, at),
                    output: PhantomData,
                },
                lengths: length,
                offset: (at * P::Span::SIZE * Cache::SIZE) as i32,
                sparse: PhantomData::<Keys<P>>,
            },
        )
    }
}

impl<P: Panel> Gathered<P> {
    /// The table's entries of block `at` of width `P`.
    fn entries<W: AxisName>(
        table: &Table<W>,
        at: usize,
    ) -> HbmTensor<i32, Chip, m![{ Keys<P> } / Block]> {
        // SAFETY: the table's rows, read as consecutive blocks of this width, keep the entry order;
        // read in place, so no device copy of them makes the processor wait before checking them.
        unsafe {
            table
                .view()
                .reshape::<Chip, m![{ PanelCount<P, W> }, { Keys<P> } / Block]>()
                .tile::<
                    PanelCount<P, W>,
                    1,
                    m![{ PanelCount<P, W> } = 1 # ({ PanelCount<P, W> }), { Keys<P> } / Block],
                >(at)
                .alias()
        }
    }
}

/// Each of `T` tiles' scores over a `P`-wide block, one row of 128 keys per panel.
pub(crate) type PanelScores<KvHeads, Groups, P, T> = DmTensor<
    bf16,
    Chip,
    Cluster<KvHeads>,
    Slice<KvHeads, Groups>,
    m![{ Tiles<T> }, Rows % 8, { Panels<P> }, Cache],
>;
/// Each of `T` tiles' weighted values of every panel of a `P`-wide block.
pub(crate) type Partials<KvHeads, Groups, Head, P, T> = DmTensor<
    f32,
    Chip,
    Cluster<KvHeads>,
    Slice<KvHeads, Groups>,
    m![{ Tiles<T> }, { Panels<P> }, Rows % 8, Head],
>;

/// Each of `T` tiles' scores over a `P`-wide block of staged keys.
struct ScoreStaged<P>(PhantomData<P>);

impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, P: Panel, T: AxisName>
    Op<(
        &Attention<KvHeads, Groups, Head>,
        &Staged<KvHeads, Head, P>,
        &Group<KvHeads, Groups, Head, T>,
    )> for ScoreStaged<P>
{
    type Output = PanelScores<KvHeads, Groups, P, T>;

    fn forward(
        &self,
        context: &mut Device,
        (attention, keys, query): (
            &Attention<KvHeads, Groups, Head>,
            &Staged<KvHeads, Head, P>,
            &Group<KvHeads, Groups, Head, T>,
        ),
    ) -> Self::Output {
        let fill = |context: &mut Device, s| {
            Op::forward(
                &fresh::FillKeys::<Groups, Head>(PhantomData),
                context,
                keys.view().tile::<Panels<P>, 1, KeyTile<Head, P>>(s),
            )
        };
        let mut scores = PanelScores::<KvHeads, Groups, P, T>::new();
        // Each panel's fill is issued before the previous panel's contractions, so the two overlap.
        let first = fill(context, 0);
        Op::forward(
            &ScoreBlock::<P>(PhantomData),
            context,
            (attention, (first, fill), query, &mut scores),
        );
        scores
    }
}

/// Each tile's scores exponentiated against the tile's running lse.
struct Probabilities<P>(PhantomData<P>);

impl<KvHeads: AxisName, Groups: AxisName, P: Panel, T: AxisName>
    Op<(
        &PanelScores<KvHeads, Groups, P, T>,
        &state::Running<KvHeads, Groups, T>,
    )> for Probabilities<P>
{
    type Output = Keyed<KvHeads, Groups, P, T>;

    fn forward(
        &self,
        context: &mut Device,
        (scores, running): (
            &PanelScores<KvHeads, Groups, P, T>,
            &state::Running<KvHeads, Groups, T>,
        ),
    ) -> Self::Output {
        // SAFETY: the panel axis is outer to Cache, so each row keeps every key in its block order.
        let rows = unsafe {
            scores
                .view()
                .reshape::<
                    Chip,
                    Cluster<KvHeads>,
                    Slice<KvHeads, Groups>,
                    m![{ Tiles<T> }, Rows % 8, { Keys<P> }],
                >()
        };
        let mut probabilities = Keyed::<KvHeads, Groups, P, T>::new();
        #[unroll]
        for t in 0..T::SIZE {
            // The running lse replaces the block maximum as the shift: only a score 88 above
            // overflows.
            let top = Op::forward(
                &state::Operand,
                &mut *context,
                running
                    .view()
                    .tile::<m![{ Symbol<T> }], 1, TileStatistic<T>>(t),
            );
            Op::forward(
                &panel::Exponents,
                &mut *context,
                ((rows.clone(), &top), &mut probabilities, t),
            );
        }
        probabilities
    }
}

/// Each tile row's total over a `P`-wide block of probabilities, one contraction against ones: a
/// vector-engine sum takes a pass of four lanes a cycle per tile.
struct Totals<P>(PhantomData<P>);

impl<KvHeads: AxisName, Groups: AxisName, P: Panel, T: AxisName> Op<&Keyed<KvHeads, Groups, P, T>>
    for Totals<P>
{
    type Output = state::Running<KvHeads, Groups, T>;

    fn forward(
        &self,
        context: &mut Device,
        probabilities: &Keyed<KvHeads, Groups, P, T>,
    ) -> Self::Output {
        let mut ones = DmTensor::<
            bf16,
            Chip,
            Cluster<KvHeads>,
            Slice<KvHeads, Groups>,
            m![{ Keys<P> } % 32],
        >::new();
        ones.view_mut()
            .memset(const { bf16::from_f32(1.0) }, &mut context.sub);
        let ones: TrfTensor<
            bf16,
            Chip,
            Cluster<KvHeads>,
            Slice<KvHeads, Groups>,
            m![1],
            m![{ Keys<P> } % 32],
        > = context
            .sub
            .begin(ones.view())
            .fetch::<m![{ Keys<P> } % 32 / 16], m![{ Keys<P> } % 16]>()
            .collect::<m![{ Keys<P> } % 32 / 16], m![{ Keys<P> } % 16]>()
            .to_trf();
        let mut total = state::Running::<KvHeads, Groups, T>::new();
        context.main.begin(probabilities.view())
            .fetch::<m![{ Tiles<T> }, Rows % 8, { Keys<P> } / 32], m![{ Keys<P> } % 32]>()
            .collect::<m![{ Tiles<T> }, Rows % 8, { Keys<P> } / 32, { Keys<P> } % 32 / 16],
                m![{ Keys<P> } % 16]>()
            .contract_outer::<m![{ Tiles<T> }, Rows % 8, { Keys<P> } / 32], m![{ Keys<P> } % 32],
                _, _, _>(&ones)
            .contract_packet::<m![1]>()
            .contract_time::<m![{ Tiles<T> }, Rows % 8]>()
            .contract_lane::<m![{ Tiles<T> }, Rows % 8], m![1 # 8]>(LaneMode::Sequential)
            .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit_view(total.view_mut());
        total
    }
}

/// Folds full history block `at` of width `P` into each of `T` query tiles' state: one gather
/// and one register-file fill per panel serve every tile, and each tile normalizes one row.
pub(crate) struct FoldBlock<P>(pub(crate) PhantomData<P>);

impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, P: Panel, T: AxisName, W: AxisName>
    Op<(
        &Attention<KvHeads, Groups, Head>,
        (&Table<W>, (&Gather, &Blocks<KvHeads, Head>)),
        usize,
        &Group<KvHeads, Groups, Head, T>,
        (
            &mut state::Carry<KvHeads, Groups, Head, T>,
            &mut state::Running<KvHeads, Groups, T>,
        ),
    )> for FoldBlock<P>
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (attention, input, at, query, state): (
            &Attention<KvHeads, Groups, Head>,
            (&Table<W>, (&Gather, &Blocks<KvHeads, Head>)),
            usize,
            &Group<KvHeads, Groups, Head, T>,
            (
                &mut state::Carry<KvHeads, Groups, Head, T>,
                &mut state::Running<KvHeads, Groups, T>,
            ),
        ),
    ) {
        let (carry, running) = state;
        let in_dm = Op::forward(&Gathered::<P>(PhantomData), &mut *context, (input, at));
        // SAFETY: the gather's key digits relabelled as panel and in-panel row, in the same order.
        let rows = unsafe {
            in_dm
                .view()
                .reshape::<
                    Chip,
                    Cluster<KvHeads>,
                    DmSlices<KvHeads>,
                    m![{ Panels<P> }, Cache % Block, KeyValue, Head],
                >()
        };
        let mut keys = Staged::<KvHeads, Head, P>::new();
        let mut values = DmTensor::<
            bf16,
            Chip,
            Cluster<KvHeads>,
            fresh::ValueSlice<KvHeads, Head>,
            m![{ Panels<P> }, Cache % 16, Head % 16],
        >::new();
        rows.clone()
            .tile::<m![KeyValue], 1, m![{ Panels<P> }, Cache % Block, KeyValue = 1 # 2, Head]>(0)
            .to_dm_view(&mut context.tdma, keys.view_mut());
        rows.tile::<m![KeyValue], 1, m![{ Panels<P> }, Cache % Block, KeyValue = 1 # 2, Head]>(1)
            .to_dm_view(&mut context.tdma, values.view_mut());
        let scores = Op::forward(
            &ScoreStaged::<P>(PhantomData),
            &mut *context,
            (attention, &keys, query),
        );
        let probabilities = Op::forward(
            &Probabilities::<P>(PhantomData),
            &mut *context,
            (&scores, &*running),
        );
        let partials = Op::forward(
            &WeighBlock::<P>(PhantomData),
            &mut *context,
            (
                |context: &mut Device, s| {
                    Op::forward(
                        &fresh::FillValues::<Groups>(PhantomData),
                        context,
                        values.view().tile::<Panels<P>, 1, ValueTile<Head, P>>(s),
                    )
                },
                &probabilities,
            ),
        );
        let total = Op::forward(&Totals::<P>(PhantomData), &mut *context, &probabilities);
        Op::forward(
            &state::Absorb,
            context,
            ((carry, running), &partials, &total),
        );
    }
}

/// Where a 1024-key block's values wait: 16 keys of every head dimension a slice.
type Runs<KvHeads> = m![KvHeads % 4, Span1024, Cache / 16];
/// Sixteen head dimensions of every key of a 1024-key block.
type Dims<KvHeads, Groups, Head> = TrfTensor<
    bf16,
    Chip,
    Cluster<KvHeads>,
    Slice<KvHeads, Groups>,
    m![Head % 8],
    m![Head % 16 / 8, Span1024, Cache],
>;
type DimsTile<Head> = m![Head / 16 = 1 # 8, Head % 16, Cache % 16];
type Weighted<KvHeads, Groups, Head, T> = DmTensor<
    f32,
    Chip,
    Cluster<KvHeads>,
    Slice<KvHeads, Groups>,
    m![{ Tiles<T> }, Span128, Rows % 8, Head],
>;

/// [`FoldBlock`] for a 1024-key [`Gathered`] block, each value fill 16 head dimensions of every
/// key, so each contraction sums the whole block and no per-panel partials remain.
pub(crate) struct FoldWide;

impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, T: AxisName>
    Op<(
        &Attention<KvHeads, Groups, Head>,
        &InDm<KvHeads, Head, Wide>,
        &Group<KvHeads, Groups, Head, T>,
        (
            &mut state::Carry<KvHeads, Groups, Head, T>,
            &mut state::Running<KvHeads, Groups, T>,
        ),
    )> for FoldWide
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (attention, in_dm, query, state): (
            &Attention<KvHeads, Groups, Head>,
            &InDm<KvHeads, Head, Wide>,
            &Group<KvHeads, Groups, Head, T>,
            (
                &mut state::Carry<KvHeads, Groups, Head, T>,
                &mut state::Running<KvHeads, Groups, T>,
            ),
        ),
    ) {
        let (carry, running) = state;
        // SAFETY: the gather's key digits relabelled as panel and in-panel row, in the same order.
        let rows = unsafe {
            in_dm
                .view()
                .reshape::<
                    Chip,
                    Cluster<KvHeads>,
                    DmSlices<KvHeads>,
                    m![{ Panels<Wide> }, Cache % Block, KeyValue, Head],
                >()
        };
        let mut keys = Staged::<KvHeads, Head, Wide>::new();
        let mut values =
            DmTensor::<bf16, Chip, Cluster<KvHeads>, Runs<KvHeads>, m![Cache % 16, Head]>::new();
        rows.clone()
            .tile::<m![KeyValue], 1, m![Span1024, Cache % Block, KeyValue = 1 # 2, Head]>(0)
            .to_dm_view(&mut context.tdma, keys.view_mut());
        rows.tile::<m![KeyValue], 1, m![Span1024, Cache % Block, KeyValue = 1 # 2, Head]>(1)
            .to_dm_view(&mut context.tdma, values.view_mut());
        let scores = Op::forward(
            &ScoreStaged::<Wide>(PhantomData),
            &mut *context,
            (attention, &keys, query),
        );
        let mut transposed = DmTensor::<
            bf16,
            Chip,
            Cluster<KvHeads>,
            Runs<KvHeads>,
            m![Head / 16, Head % 16, Cache % 16],
        >::new();
        context
            .main
            .begin(values.view())
            .fetch::<m![Head / 16, Cache % 16], m![Head % 16]>()
            .collect::<m![Head / 16, Cache % 16], m![Head % 16]>()
            .transpose::<m![Head / 16, Cache % 16 / 4, Head % 16], m![Cache % 4 # 16]>()
            .commit_trim::<m![Cache % 4]>()
            .commit_view(transposed.view_mut());
        let fill_dims = |context: &mut Device,
                         values: DmTensorView<
            '_,
            bf16,
            Chip,
            Cluster<KvHeads>,
            Runs<KvHeads>,
            DimsTile<Head>,
        >|
         -> Dims<KvHeads, Groups, Head> {
            context
                .sub
                .begin(values)
                .fetch::<m![Head % 8, Head % 16 / 8], m![Cache % 16]>()
                .switch::<
                    Slice<KvHeads, Groups>,
                    m![Head % 8, Head % 16 / 8, Span1024, Cache / 16],
                >(
                    SwitchConfig::Broadcast01 {
                        slice1: 8,
                        slice0: 8,
                        time0: 1,
                    },
                )
                .collect::<m![Head % 8, Head % 16 / 8, Span1024, Cache / 16], m![Cache % 16]>()
                .to_trf()
        };
        // The first fill runs under the softmax, which leaves the register file idle.
        let mut value = fill_dims(
            context,
            transposed
                .view()
                .tile::<m![Head / 16], 1, DimsTile<Head>>(0),
        );
        let probabilities = Op::forward(
            &Probabilities::<Wide>(PhantomData),
            &mut *context,
            (&scores, &*running),
        );
        // SAFETY: the inverse relabel of the score rows.
        let panels = unsafe {
            probabilities
                .view()
                .reshape::<
                    Chip,
                    Cluster<KvHeads>,
                    Slice<KvHeads, Groups>,
                    m![{ Tiles<T> }, Rows % 8, Span1024, Cache],
                >()
        };
        let mut weighted = Weighted::<KvHeads, Groups, Head, T>::new();
        // Every tile's probabilities over a 1024-key block times the block's head dimensions `d`,
        // one contraction, into those dimensions of `weighted`.
        let weigh_dims = |context: &mut Device,
                          probabilities: &DmTensorView<
            '_,
            bf16,
            Chip,
            Cluster<KvHeads>,
            Slice<KvHeads, Groups>,
            m![{ Tiles<T> }, Rows % 8, Span1024, Cache],
        >,
                          value: &Dims<KvHeads, Groups, Head>,
                          weighted: &mut Weighted<KvHeads, Groups, Head, T>,
                          d: usize| {
            // Two rows of 16 keys a packet: the fetch reads 32 contiguous bytes a row, as fast as
            // the DPE consumes them, where 8-key pieces halve the rate.
            context.main
                .begin(probabilities.clone())
                .fetch::<
                    m![{ Tiles<T> }, Rows % 8 / 2, Span1024, Cache / 16],
                    m![Rows % 2, Cache % 16],
                >()
                .collect::<m![{ Tiles<T> }, Rows % 8 / 2, Span1024, Cache / 16, Rows % 2],
                    m![Cache % 16]>()
                .contract_outer::<m![
                    { Tiles<T> },
                    Rows % 8 / 2,
                    Span1024,
                    Cache / 16,
                    Head % 16 / 8
                ],
                    m![Rows % 2, Cache % 16], _, _, _>(value)
                .contract_packet::<m![Rows % 2]>()
                .contract_time::<m![{ Tiles<T> }, Rows % 8 / 2, Head % 16 / 8]>()
                .contract_lane::<m![{ Tiles<T> }, Rows % 8 / 2, Head % 16 / 8, Head % 8],
                    m![Rows % 2 # 8]>(LaneMode::Sequential)
                .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_final()
                .transpose::<
                    m![{ Tiles<T> }, Rows % 8 / 2, Head % 16 / 2, Rows % 2],
                    m![Head % 2 # 8],
                >()
                .commit_trim::<m![Head % 2]>()
                .commit_view(weighted.view_mut().tile::<m![Head / 16], 1,
                    m![{ Tiles<T> }, Span128, Rows % 8, Head / 16 = 1 #{!} 8, Head % 16]>(d));
        };
        #[unroll]
        for d in 1..Head::SIZE / 16 {
            let next = fill_dims(
                context,
                transposed
                    .view()
                    .tile::<m![Head / 16], 1, DimsTile<Head>>(d),
            );
            weigh_dims(context, &panels, &value, &mut weighted, d - 1);
            value = next;
        }
        weigh_dims(context, &panels, &value, &mut weighted, Head::SIZE / 16 - 1);
        let total = Op::forward(&Totals::<Wide>(PhantomData), &mut *context, &probabilities);
        Op::forward(
            &state::Absorb,
            context,
            ((carry, running), &weighted, &total),
        );
    }
}

/// Panel `s`'s scaled scores of every tile of the group, one contraction for all tiles.
pub(crate) struct Score<P>(pub(crate) PhantomData<P>);

impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, P: Panel, T: AxisName, Q: M>
    Op<(
        &Attention<KvHeads, Groups, Head>,
        DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, Q>,
        &fresh::Keys<KvHeads, Groups, Head>,
        &mut DmTensor<
            bf16,
            Chip,
            Cluster<KvHeads>,
            Slice<KvHeads, Groups>,
            m![{ Tiles<T> }, Rows % 8, { Panels<P> }, Cache],
        >,
        usize,
    )> for Score<P>
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (attention, query, key, scores, s): (
            &Attention<KvHeads, Groups, Head>,
            DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, Q>,
            &fresh::Keys<KvHeads, Groups, Head>,
            &mut DmTensor<
                bf16,
                Chip,
                Cluster<KvHeads>,
                Slice<KvHeads, Groups>,
                m![{ Tiles<T> }, Rows % 8, { Panels<P> }, Cache],
            >,
            usize,
        ),
    ) {
        Op::forward(
            &ScoreTiles::<T>(PhantomData),
            context,
            (
                attention,
                query,
                key,
                scores
                    .view_mut()
                    .tile::<Panels<P>, 1, m![{ Tiles<T> }, Rows % 8, { PanelSlot<P> }, Cache]>(s),
            ),
        );
    }
}

/// One panel's scaled scores of the `T` tiles `query` holds, one contraction, into `scores`.
pub(crate) struct ScoreTiles<T>(pub(crate) PhantomData<T>);

impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, T: AxisName, Q: M, E: M>
    Op<(
        &Attention<KvHeads, Groups, Head>,
        DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, Q>,
        &fresh::Keys<KvHeads, Groups, Head>,
        DmTensorViewMut<'_, bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, E>,
    )> for ScoreTiles<T>
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (attention, query, key, scores): (
            &Attention<KvHeads, Groups, Head>,
            DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, Q>,
            &fresh::Keys<KvHeads, Groups, Head>,
            DmTensorViewMut<'_, bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, E>,
        ),
    ) {
        context.main.begin(query)
        .fetch::<m![{ Tiles<T> }, Rows % 8 / 2, Cache / 8 % 8, Head / 16],
            m![Rows % 2, Head % 16]>()
        .collect::<m![{ Tiles<T> }, Rows % 8 / 2, Cache / 8 % 8, Head / 16, Rows % 2],
            m![Head % 16]>()
        .contract_outer::<m![{ Tiles<T> }, Rows % 8 / 2, Cache / 8 % 8, Head / 16, Cache / 64],
            m![Rows % 2, Head % 16], _, _, _>(key)
        .contract_packet::<m![Rows % 2]>()
        .contract_time::<m![{ Tiles<T> }, Rows % 8 / 2, Cache / 8 % 8, Cache / 64]>()
        .contract_lane::<m![{ Tiles<T> }, Rows % 8 / 2, Cache / 8 % 8, Cache / 64, Rows % 2],
            m![Cache % 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ Tiles<T> }, Rows % 8 / 2, Cache / 8 % 8, Cache / 64, Rows % 2,
            Cache % 8 / 4], m![Cache % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), attention.scale)
        .vector_widen_concat::<m![{ Tiles<T> }, Rows % 8 / 2, Cache / 8 % 8, Cache / 64, Rows % 2],
            m![Cache % 8]>()
        .vector_final()
        .commit_trim::<m![Cache % 8]>()
        .commit_cast::<bf16>()
        .commit_view(scores);
    }
}

/// Panel `s`'s probabilities times its values for every tile of the group, one contraction.
pub(crate) struct Weigh<P>(pub(crate) PhantomData<P>);

impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, P: Panel, T: AxisName>
    Op<(
        &DmTensorView<
            '_,
            bf16,
            Chip,
            Cluster<KvHeads>,
            Slice<KvHeads, Groups>,
            m![{ Tiles<T> }, Rows % 8, { Panels<P> }, Cache],
        >,
        &fresh::Values<KvHeads, Groups, Head>,
        &mut DmTensor<
            f32,
            Chip,
            Cluster<KvHeads>,
            Slice<KvHeads, Groups>,
            m![{ Tiles<T> }, { Panels<P> }, Rows % 8, Head],
        >,
        usize,
    )> for Weigh<P>
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (probabilities, value, partials, s): (
            &DmTensorView<
                '_,
                bf16,
                Chip,
                Cluster<KvHeads>,
                Slice<KvHeads, Groups>,
                m![{ Tiles<T> }, Rows % 8, { Panels<P> }, Cache],
            >,
            &fresh::Values<KvHeads, Groups, Head>,
            &mut DmTensor<
                f32,
                Chip,
                Cluster<KvHeads>,
                Slice<KvHeads, Groups>,
                m![{ Tiles<T> }, { Panels<P> }, Rows % 8, Head],
            >,
            usize,
        ),
    ) {
        Op::forward(
            &WeighTiles::<T>(PhantomData),
            context,
            (
                probabilities
                    .clone()
                    .tile::<Panels<P>, 1, m![{ Tiles<T> }, Rows % 8, { PanelOf<P> }, Cache]>(s),
                value,
                partials
                    .view_mut()
                    .tile::<Panels<P>, 1, m![{ Tiles<T> }, { PanelSlot<P> }, Rows % 8, Head]>(s),
            ),
        );
    }
}

/// One panel's probabilities of the `T` tiles `probabilities` holds times its values, one
/// contraction, into `partials`.
pub(crate) struct WeighTiles<T>(pub(crate) PhantomData<T>);

impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, T: AxisName, Q: M, E: M>
    Op<(
        DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, Q>,
        &fresh::Values<KvHeads, Groups, Head>,
        DmTensorViewMut<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, E>,
    )> for WeighTiles<T>
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (probabilities, value, partials): (
            DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, Q>,
            &fresh::Values<KvHeads, Groups, Head>,
            DmTensorViewMut<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, E>,
        ),
    ) {
        context.main
            .begin(probabilities)
            .fetch::<
                m![{ Tiles<T> }, Rows % 2, Head / 32, Cache / 8],
                m![Rows % 8 / 2, Cache % 8],
            >()
            .collect::<m![{ Tiles<T> }, Rows % 2, Head / 32, Cache / 8, Rows % 8 / 4],
                m![Rows % 4 / 2, Cache % 8]>()
            .contract_outer::<m![{ Tiles<T> }, Rows % 2, Head / 32, Cache / 8, Head % 32 / 8],
                m![Rows % 8 / 2, Cache % 8], _, _, _>(value)
            .contract_packet::<m![Rows % 8 / 2]>()
            .contract_time::<m![{ Tiles<T> }, Rows % 2, Head / 32, Head % 32 / 8]>()
            .contract_lane::<m![{ Tiles<T> }, Rows % 2, Head / 32, Head % 32 / 8, Head % 8],
                m![Rows % 8 / 2 # 8]>(LaneMode::Sequential)
            .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_final()
            .transpose::<m![{ Tiles<T> }, Rows % 2, Head / 2, Rows % 8 / 2], m![Head % 2 # 8]>()
            .commit_trim::<m![Head % 2]>()
            .commit_view(partials);
    }
}

/// The partial last panel's key mask: tagged where a key lies at or past the committed history.
pub(crate) type TailMask<KvHeads, Groups> =
    VrfTensor<i4, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, m![Cache]>;

pub(crate) struct LastKeys<KvHeads, Groups>(pub(crate) PhantomData<(KvHeads, Groups)>);

impl<KvHeads: AxisName, Groups: AxisName, W: AxisName>
    Op<(&HbmTensor<i32, Chip, m![Cache]>, &SlotTable<W>, usize)> for LastKeys<KvHeads, Groups>
{
    type Output = TailMask<KvHeads, Groups>;

    fn forward(
        &self,
        context: &mut Device,
        (length, slots, at): (&HbmTensor<i32, Chip, m![Cache]>, &SlotTable<W>, usize),
    ) -> Self::Output {
        let length: DmTensor<i32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, m![Cache]> =
            length.to_dm(&mut context.tdma);
        let length: VrfTensor<i32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, m![Cache]> =
            context
                .sub
                .begin(length.view())
                .fetch::<m![Cache / 8], m![Cache % 8]>()
                .collect::<m![Cache / 8], m![Cache % 8]>()
                .to_vrf();
        let slots: DmTensor<i32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, m![Cache]> = slots
            .view()
            .tile::<m![{ Symbol<Chunks<W>> }], 1,
                m![{ Symbol<Chunks<W>> } = 1 # ({ Symbol<Chunks<W>> }), Cache]>(at)
            .to_dm(&mut context.tdma);
        let distance: DmTensor<i32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, m![Cache]> =
            context
                .main
                .begin(slots.view())
                .fetch::<m![Cache / 8], m![Cache % 8]>()
                .collect::<m![Cache / 8], m![Cache % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_fxp(FxpBinaryOp::SubFxpSat, &length)
                .vector_final()
                .commit_trim::<m![Cache % 8]>()
                .commit();
        context
            .main
            .begin(distance.view())
            .fetch::<m![Cache / 8], m![Cache % 8]>()
            .collect::<m![Cache / 8], m![Cache % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Comparison([
                Cmp::Less(0i32),
                Cmp::False,
                Cmp::False,
                Cmp::False,
            ]))
            .vector_log(&mut context.sub)
    }
}

/// `scores` with every key `mask` does not tag set to −∞.
pub(crate) struct MaskScores;

impl<KvHeads: AxisName, Groups: AxisName> Op<(&Scores<KvHeads, Groups>, &TailMask<KvHeads, Groups>)>
    for MaskScores
{
    type Output = Scores<KvHeads, Groups>;

    fn forward(
        &self,
        context: &mut Device,
        (scores, mask): (&Scores<KvHeads, Groups>, &TailMask<KvHeads, Groups>),
    ) -> Self::Output {
        let mut output = Scores::<KvHeads, Groups>::new();
        for row in 0..8 {
            context
                .main
                .begin(
                    scores
                        .view()
                        .tile::<m![Rows % 8], 1, m![Rows % 8 = 1 # 8, Cache]>(row),
                )
                .fetch::<m![Cache / 16], m![Cache % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![Cache / 8], m![Cache % 8]>()
                .vector_init()
                .vector_tagged(mask)
                .vector_logic(
                    LogicBinaryOpF32::BitAnd,
                    Branched::imm(state::PRESENT, 0.0f32),
                )
                .vector_clip(
                    ClipBinaryOpF32::Add,
                    Branched::imm(state::PRESENT, f32::NEG_INFINITY),
                )
                .vector_final()
                .cast::<bf16, m![Cache % 8 # 16]>()
                .commit_trim::<m![Cache % 8]>()
                .commit_view(
                    output
                        .view_mut()
                        .tile::<m![Rows % 8], 1, m![Rows % 8 = 1 #{!} 8, Cache]>(row),
                );
        }
        output
    }
}

/// A partial last panel's keys and values in HBM, each token at or past the committed history
/// zeroed: both read one mask.
pub(crate) struct Clean;

impl<KvHeads: AxisName, Head: AxisName, W: AxisName>
    Op<(
        (&Tokens<KvHeads, Head>, &Tokens<KvHeads, Head>),
        (&SlotTable<W>, &HbmTensor<i32, Chip, m![Cache]>, usize),
    )> for Clean
{
    type Output = (
        HbmTensor<bf16, Chip, m![Cache, KvHeads, Head]>,
        HbmTensor<bf16, Chip, m![Cache, KvHeads, Head]>,
    );

    fn forward(
        &self,
        context: &mut Device,
        ((key, value), input): (
            (&Tokens<KvHeads, Head>, &Tokens<KvHeads, Head>),
            (&SlotTable<W>, &HbmTensor<i32, Chip, m![Cache]>, usize),
        ),
    ) -> Self::Output {
        let (slots, length, at) = input;
        let length: DmTensor<i32, Chip, Cluster<KvHeads>, DmSlices<KvHeads>, m![Cache % Block]> =
            length.to_dm(&mut context.tdma);
        let length: VrfTensor<i32, Chip, Cluster<KvHeads>, DmSlices<KvHeads>, m![Cache % Block]> =
            context
                .sub
                .begin(length.view())
                .fetch::<m![Cache % Block / 8], m![Cache % 8]>()
                .collect::<m![Cache % Block / 8], m![Cache % 8]>()
                .to_vrf();
        let slots: DmTensor<i32, Chip, Cluster<KvHeads>, DmSlices<KvHeads>, m![Cache % Block]> =
            slots
                .view()
                .tile::<m![{ Symbol<Chunks<W>> }], 1,
                    m![{ Symbol<Chunks<W>> } = 1 # ({ Symbol<Chunks<W>> }), Cache]>(at)
                .to_dm(&mut context.tdma);
        let distance: Distance<KvHeads> = context
            .main
            .begin(slots.view())
            .fetch::<m![Cache % Block / 8], m![Cache % 8]>()
            .collect::<m![Cache % Block / 8], m![Cache % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::SubFxpSat, &length)
            .vector_final()
            .commit_trim::<m![Cache % 8]>()
            .commit();
        let ended = Op::forward(&End::<Head>(PhantomData), &mut *context, &distance);
        (
            Op::forward(&Zero, &mut *context, (key, &ended)),
            Op::forward(&Zero, context, (value, &ended)),
        )
    }
}

/// Tags each token whose distance is not negative, across every head channel.
pub(super) struct End<Head>(pub(super) PhantomData<Head>);

impl<KvHeads: AxisName, Head: AxisName> Op<&Distance<KvHeads>> for End<Head> {
    type Output = Ended<KvHeads, Head>;

    fn forward(&self, context: &mut Device, distance: &Distance<KvHeads>) -> Self::Output {
        let distance: VrfTensor<
            i32,
            Chip,
            Cluster<KvHeads>,
            DmSlices<KvHeads>,
            m![Cache % Block, 1 # 8],
        > = context
            .sub
            .begin(distance.view())
            .fetch::<m![Cache % Block], m![1 # 8]>()
            .collect::<m![Cache % Block], m![1 # 8]>()
            .to_vrf();
        let mut zeros: DmTensor<
            i32,
            Chip,
            Cluster<KvHeads>,
            DmSlices<KvHeads>,
            m![Cache % Block, Head],
        > = DmTensor::new();
        zeros.view_mut().memset(0, &mut context.sub);
        let expanded: DmTensor<
            i32,
            Chip,
            Cluster<KvHeads>,
            DmSlices<KvHeads>,
            m![Cache % Block, Head],
        > = context
            .main
            .begin(zeros.view())
            .fetch::<m![Cache % Block, Head / 8], m![Head % 8]>()
            .collect::<m![Cache % Block, Head / 8], m![Head % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::AddFxpSat, &distance)
            .vector_final()
            .commit_trim::<m![Head % 8]>()
            .commit();
        context
            .main
            .begin(expanded.view())
            .fetch::<m![Cache % Block, Head / 8], m![Head % 8]>()
            .collect::<m![Cache % Block, Head / 8], m![Head % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Comparison([
                Cmp::Less(0i32),
                Cmp::False,
                Cmp::False,
                Cmp::False,
            ]))
            .vector_log(&mut context.sub)
    }
}

/// `tokens` in HBM, every token `ended` tags zero.
pub(super) struct Zero;

impl<KvHeads: AxisName, Head: AxisName> Op<(&Tokens<KvHeads, Head>, &Ended<KvHeads, Head>)>
    for Zero
{
    type Output = HbmTensor<bf16, Chip, m![Cache, KvHeads, Head]>;

    fn forward(
        &self,
        context: &mut Device,
        (tokens, ended): (&Tokens<KvHeads, Head>, &Ended<KvHeads, Head>),
    ) -> Self::Output {
        let zeroed: Tokens<KvHeads, Head> = context
            .main
            .begin(tokens.view())
            .fetch::<m![Cache % Block, Head / 16], m![Head % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![Cache % Block, Head / 8], m![Head % 8]>()
            .vector_init()
            .vector_tagged(ended)
            .vector_logic(
                LogicBinaryOpF32::BitAnd,
                Branched::imm(state::PRESENT, 0.0f32),
            )
            .vector_final()
            .cast::<bf16, m![Head % 8 # 16]>()
            .commit_trim::<m![Head % 8]>()
            .commit();
        let mut output = HbmTensor::new();
        zeroed
            .view()
            .to_hbm_view(&mut context.tdma, output.view_mut());
        output
    }
}
