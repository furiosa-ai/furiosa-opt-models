use furiosa_opt_std::prelude::*;

use super::Attention;
use super::chunks::{Cache, Chunks, Epoch, Record, Run, Runs, SlotTable};
use crate::ops::index::gather::Gather;
use crate::specialize::attention;
use crate::tensor::cache::Block;
use crate::tensor::cache::KeyValue;
use crate::tensor::parallel::Chip;
use crate::tensor::scalar::Value;
use crate::{Module, Op};

use super::chunks::Query;
use std::marker::PhantomData;

use crate::tensor::cache::Blocks;
use causal::CausalBlock;

mod causal;
mod fresh;
mod history;
mod panel;
mod state;

axes![Rows = 128];
// Query tiles that share each history block's fills, and the one tile of a chunk's remainder.
axes![TileGroup = 4, SingleTile = 1, Span128 = 1];
// Two tile groups of a 1024-row chunk that share every history block.
axes![TilePair = 8, Halves = 2];

/// A number of query tiles attended together, and the block their own fresh keys form.
trait Group<KvHeads: AxisName, Groups: AxisName, Head: AxisName>: AxisName {
    type Fresh: Panel;
    /// The causal mask over the group's own fresh panels: below a tile's diagonal every key
    /// counts, on it the causal tile; a tile reads no panel past its own, so those stay visible.
    fn mask<All: AxisName, W: AxisName>(
        context: &mut Device,
        input: &Input<'_, KvHeads, Groups, Head, All, W>,
    ) -> GroupMask<KvHeads, Groups, Head, Self>;
    /// Writes the group's query tiles, from batch row `row`, in data memory.
    fn to_dm<All: AxisName>(
        context: &mut Device,
        query: &HbmTensor<bf16, Chip, m![All, KvHeads, Groups, Head]>,
        row: usize,
    ) -> history::Group<KvHeads, Groups, Head, Self>;
    /// The scaled scores of the group's causal block, whose first panel's keys are filled.
    fn scores(
        attention: &Attention<KvHeads, Groups, Head>,
        context: &mut Device,
        source: (
            &mut impl KeySource<KvHeads, Groups, Head>,
            fresh::Keys<KvHeads, Groups, Head>,
        ),
        queries: &impl QueryTiles<KvHeads, Groups, Head>,
    ) -> history::PanelScores<KvHeads, Groups, Self::Fresh, Self>;
    /// Masks and normalizes the causal block's scores and seeds the group's state with the
    /// block's weighted values.
    fn seed<S: ValueSource<KvHeads, Groups, Head>>(
        context: &mut Device,
        block: (
            &history::PanelScores<KvHeads, Groups, Self::Fresh, Self>,
            &GroupMask<KvHeads, Groups, Head, Self>,
        ),
        source: &mut S,
        state: (
            &mut impl state::Tiles<KvHeads, Groups, Head>,
            &mut impl state::Tiles<KvHeads, Groups, Head>,
        ),
    );
    /// Folds the chunk's own panels between the last whole block before the record's first tile
    /// and that tile; four tiles always start at a whole block, so they have none.
    fn within<All: AxisName, W: AxisName>(
        _: &mut Device,
        _: &Input<'_, KvHeads, Groups, Head, All, W>,
        _: Record,
        _: &history::Group<KvHeads, Groups, Head, Self>,
        _: (
            &mut state::Carry<KvHeads, Groups, Head, Self>,
            &mut state::Running<KvHeads, Groups, Self>,
        ),
    ) {
    }
}
impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName> Group<KvHeads, Groups, Head>
    for TileGroup
{
    type Fresh = Half;
    fn mask<All: AxisName, W: AxisName>(
        context: &mut Device,
        input: &Input<'_, KvHeads, Groups, Head, All, W>,
    ) -> GroupMask<KvHeads, Groups, Head, Self> {
        // A tile's prefix is visible panels and its causal tile, so only that tile arrives, a
        // sixteenth of the group mask's bytes, and is placed on the diagonal of a zeroed mask.
        let diagonal = Op::forward(&Diagonal, &mut *context, input);
        let mut mask = GroupMask::<KvHeads, Groups, Head, Self>::new();
        mask.view_mut().memset(0, &mut context.sub);
        #[unroll]
        for t in 0..Self::SIZE {
            context.main.begin(diagonal.view())
                .fetch::<m![Rows % 8, Cache / 32], m![Cache % 32]>()
                .collect::<m![Rows % 8, Cache / 32], m![Cache % 32]>()
                .commit_trim::<m![Cache % 32]>()
                .commit_view(mask.view_mut()
                    .tile::<m![{ Symbol<Self> }], 1,
                        m![{ TileSlot<Self> }, Rows % 8, { Panels<Half> }, Cache]>(t)
                    .tile::<Panels<Half>, 1,
                        m![{ TileSlot<Self> }, Rows % 8, { KeptSlot<Half> }, Cache]>(t));
        }
        mask
    }
    fn to_dm<All: AxisName>(
        context: &mut Device,
        query: &HbmTensor<bf16, Chip, m![All, KvHeads, Groups, Head]>,
        row: usize,
    ) -> history::Group<KvHeads, Groups, Head, Self> {
        Self::queries::<KvHeads, Groups, Head, All>(query, row).to_dm(&mut context.tdma)
    }
    /// Panel s for the tiles from s on, the tiles that see it; the scores of a tile before a
    /// panel stay unwritten. A head wider than one fill scores every panel for all four tiles, in
    /// one contraction a part.
    fn scores(
        attention: &Attention<KvHeads, Groups, Head>,
        context: &mut Device,
        (source, first): (
            &mut impl KeySource<KvHeads, Groups, Head>,
            fresh::Keys<KvHeads, Groups, Head>,
        ),
        queries: &impl QueryTiles<KvHeads, Groups, Head>,
    ) -> history::PanelScores<KvHeads, Groups, Self::Fresh, Self> {
        let mut scores = history::PanelScores::<KvHeads, Groups, Half, Self>::new();
        if const { Part::<Head>::WHOLE } {
            let second = source.key(context, 1);
            Op::forward(
                &history::Score::<Half>(PhantomData),
                &mut *context,
                (attention, queries.group(), &first, &mut scores, 0),
            );
            let third = source.key(context, 2);
            Op::forward(
                &TailScore::<Tail3, { Tail3::SIZE }>(PhantomData),
                &mut *context,
                (attention, queries, &second, &mut scores, 1),
            );
            let fourth = source.key(context, 3);
            Op::forward(
                &TailScore::<Tail2, { Tail2::SIZE }>(PhantomData),
                &mut *context,
                (attention, queries, &third, &mut scores, 2),
            );
            Op::forward(
                &TailScore::<SingleTile, { SingleTile::SIZE }>(PhantomData),
                &mut *context,
                (attention, queries, &fourth, &mut scores, 3),
            );
        } else {
            Op::forward(
                &ScoreBlock::<Half>(PhantomData),
                &mut *context,
                (attention, (first, &mut *source), queries, &mut scores),
            );
        }
        scores
    }
    /// Tile t sees panels 0..=t, so its softmax, weighing and sum read only those; a head wider
    /// than one fill weighs every panel for all four tiles, and the sums leave the rest unread.
    fn seed<S: ValueSource<KvHeads, Groups, Head>>(
        context: &mut Device,
        (scores, mask): (
            &history::PanelScores<KvHeads, Groups, Self::Fresh, Self>,
            &GroupMask<KvHeads, Groups, Head, Self>,
        ),
        source: &mut S,
        (carry, running): (
            &mut impl state::Tiles<KvHeads, Groups, Head>,
            &mut impl state::Tiles<KvHeads, Groups, Head>,
        ),
    ) {
        // SAFETY: the panel axis is outer to Cache, so each row keeps every key in its block order.
        let (score_rows, mask_rows) = unsafe {
            (
                scores.view().reshape::<Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>,
                    m![{ Symbol<Self> }, Rows % 8, { Keys<Half> }]>(),
                mask.view().reshape::<Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>,
                    m![{ Symbol<Self> }, Rows % 8, { Keys<Half> }]>(),
            )
        };
        let mut probabilities = history::Keyed::<KvHeads, Groups, Half, Self>::new();
        let mut total = state::Running::<KvHeads, Groups, Self>::new();
        Op::forward(
            &panel::CausalSoftmax::<Cache, { Cache::SIZE }>(PhantomData),
            &mut *context,
            (
                (score_rows.clone(), mask_rows.clone()),
                (&mut probabilities, &mut total, running.tile_mut(0)),
                0,
            ),
        );
        Op::forward(
            &panel::CausalSoftmax::<Keys256, { Keys256::SIZE }>(PhantomData),
            &mut *context,
            (
                (score_rows.clone(), mask_rows.clone()),
                (&mut probabilities, &mut total, running.tile_mut(1)),
                1,
            ),
        );
        Op::forward(
            &panel::CausalSoftmax::<Keys384, { Keys384::SIZE }>(PhantomData),
            &mut *context,
            (
                (score_rows.clone(), mask_rows.clone()),
                (&mut probabilities, &mut total, running.tile_mut(2)),
                2,
            ),
        );
        Op::forward(
            &panel::CausalSoftmax::<Keys512, { Keys512::SIZE }>(PhantomData),
            &mut *context,
            (
                (score_rows.clone(), mask_rows.clone()),
                (&mut probabilities, &mut total, running.tile_mut(3)),
                3,
            ),
        );
        let partials = if const { Part::<Head>::WHOLE } {
            // SAFETY: the inverse relabel of the key rows.
            let probabilities = unsafe {
                probabilities
                    .view()
                    .reshape::<Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>,
                        m![{ Symbol<Self> }, Rows % 8, { Panels<Half> }, Cache]>()
            };
            let mut partials = history::Partials::<KvHeads, Groups, Head, Half, Self>::new();
            let first = source.value(context, 0);
            let second = source.value(context, 1);
            Op::forward(
                &history::Weigh::<Half>(PhantomData),
                &mut *context,
                (&probabilities, &first, &mut partials, (0, 0)),
            );
            let third = source.value(context, 2);
            Op::forward(
                &TailWeigh::<Tail3, { Tail3::SIZE }>(PhantomData),
                &mut *context,
                (&probabilities, &second, &mut partials, 1),
            );
            let fourth = source.value(context, 3);
            Op::forward(
                &TailWeigh::<Tail2, { Tail2::SIZE }>(PhantomData),
                &mut *context,
                (&probabilities, &third, &mut partials, 2),
            );
            Op::forward(
                &TailWeigh::<SingleTile, { SingleTile::SIZE }>(PhantomData),
                &mut *context,
                (&probabilities, &fourth, &mut partials, 3),
            );
            partials
        } else {
            // SAFETY: the inverse relabel of the key rows.
            let probabilities = unsafe {
                probabilities
                    .view()
                    .reshape::<Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>,
                        m![{ Symbol<Self> }, Rows % 8, { Panels<Half> }, Cache]>()
            };
            let mut partials = history::Partials::<KvHeads, Groups, Head, Half, Self>::new();
            // Each panel by its own index: a fresh panel's cut tests a tile known when compiling.
            let first = source.value(context, 0);
            let second = Op::forward(
                &WeighPanel::<Half>(PhantomData),
                &mut *context,
                (
                    &probabilities,
                    (first, &mut *source, 0),
                    &mut partials,
                    |source: &mut S, context: &mut Device| source.value(context, 1),
                ),
            );
            let third = Op::forward(
                &WeighPanel::<Half>(PhantomData),
                &mut *context,
                (
                    &probabilities,
                    (second, &mut *source, 1),
                    &mut partials,
                    |source: &mut S, context: &mut Device| source.value(context, 2),
                ),
            );
            let fourth = Op::forward(
                &WeighPanel::<Half>(PhantomData),
                &mut *context,
                (
                    &probabilities,
                    (third, &mut *source, 2),
                    &mut partials,
                    |source: &mut S, context: &mut Device| source.value(context, 3),
                ),
            );
            Op::forward(
                &WeighPanel::<Half>(PhantomData),
                &mut *context,
                (
                    &probabilities,
                    (fourth, &mut *source, 3),
                    &mut partials,
                    |_: &mut S, _: &mut Device| (),
                ),
            );
            partials
        };
        Op::forward(
            &state::SeedPrefix::<Span128, 1>(PhantomData),
            &mut *context,
            (&mut *carry, &partials, &total, 0),
        );
        Op::forward(
            &state::SeedPrefix::<Span256, 2>(PhantomData),
            &mut *context,
            (&mut *carry, &partials, &total, 1),
        );
        Op::forward(
            &state::SeedPrefix::<Span384, 3>(PhantomData),
            &mut *context,
            (&mut *carry, &partials, &total, 2),
        );
        Op::forward(
            &state::SeedPrefix::<Span512, 4>(PhantomData),
            &mut *context,
            (&mut *carry, &partials, &total, 3),
        );
    }
}
/// Panel `s`'s scaled scores of a four-tile group's `R` tiles from `s` on, the tiles that see it;
/// `N` is `R`'s size.
struct TailScore<R, const N: usize>(PhantomData<R>);

impl<
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    R: AxisName,
    Q: QueryTiles<KvHeads, Groups, Head>,
    const N: usize,
>
    Op<(
        &Attention<KvHeads, Groups, Head>,
        &Q,
        &fresh::Keys<KvHeads, Groups, Head>,
        &mut history::PanelScores<KvHeads, Groups, Half, TileGroup>,
        usize,
    )> for TailScore<R, N>
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (attention, queries, key, scores, s): (
            &Attention<KvHeads, Groups, Head>,
            &Q,
            &fresh::Keys<KvHeads, Groups, Head>,
            &mut history::PanelScores<KvHeads, Groups, Half, TileGroup>,
            usize,
        ),
    ) {
        Op::forward(
            &history::ScoreTiles::<R>(PhantomData),
            context,
            (
                attention,
                queries.from::<R, N>(s),
                key,
                // SAFETY: the tiles from `s` keep their positions in the group's footprint.
                unsafe {
                    scores.view_mut()
                    .tile::<Panels<Half>, 1,
                        m![{ Symbol<TileGroup> }, Rows % 8, { KeptSlot<Half> }, Cache]>(s)
                    .tile::<m![{ Symbol<TileGroup> }], N,
                        m![{ Symbol<TileGroup> } = { N } #{!} ({ Symbol<TileGroup> }),
                            Rows % 8, { KeptSlot<Half> }, Cache]>(s)
                    .reshape::<Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>,
                        m![{ Symbol<R> } #{!} ({ Symbol<TileGroup> }), Rows % 8,
                            { KeptSlot<Half> }, Cache]>()
                },
            ),
        );
    }
}

/// Panel `s`'s probabilities times its values for a four-tile group's `R` tiles from `s` on, the
/// tiles that see it; `N` is `R`'s size.
struct TailWeigh<R, const N: usize>(PhantomData<R>);

impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, R: AxisName, const N: usize>
    Op<(
        &DmTensorView<
            '_,
            bf16,
            Chip,
            Cluster<KvHeads>,
            Slice<KvHeads, Groups>,
            m![{ Symbol<TileGroup> }, Rows % 8, { Panels<Half> }, Cache],
        >,
        &fresh::Values<KvHeads, Groups, Head>,
        &mut history::Partials<KvHeads, Groups, Head, Half, TileGroup>,
        usize,
    )> for TailWeigh<R, N>
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
                m![{ Symbol<TileGroup> }, Rows % 8, { Panels<Half> }, Cache],
            >,
            &fresh::Values<KvHeads, Groups, Head>,
            &mut history::Partials<KvHeads, Groups, Head, Half, TileGroup>,
            usize,
        ),
    ) {
        Op::forward(
            &history::WeighTiles::<R>(PhantomData),
            context,
            (
                // SAFETY: the tiles from `s` keep their positions in the group's footprint.
                unsafe {
                    probabilities.clone()
                    .tile::<Panels<Half>, 1,
                        m![{ Symbol<TileGroup> }, Rows % 8, { KeptPanel<Half> }, Cache]>(s)
                    .tile::<m![{ Symbol<TileGroup> }], N,
                        m![{ Symbol<TileGroup> } = { N } # ({ Symbol<TileGroup> }),
                            Rows % 8, { KeptPanel<Half> }, Cache]>(s)
                    .reshape::<Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>,
                        m![{ Symbol<R> } # ({ Symbol<TileGroup> }), Rows % 8,
                            { KeptPanel<Half> }, Cache]>()
                },
                value,
                // SAFETY: the tiles from `s` keep their positions in the group's footprint.
                unsafe {
                    partials.view_mut()
                    .tile::<Panels<Half>, 1,
                        m![{ Symbol<TileGroup> }, { KeptSlot<Half> }, Rows % 8, Head]>(s)
                    .tile::<m![{ Symbol<TileGroup> }], N,
                        m![{ Symbol<TileGroup> } = { N } #{!} ({ Symbol<TileGroup> }),
                            { KeptSlot<Half> }, Rows % 8, Head]>(s)
                    .reshape::<Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>,
                        m![{ Symbol<R> } #{!} ({ Symbol<TileGroup> }), { KeptSlot<Half> },
                            Rows % 8, Head]>()
                },
            ),
        );
    }
}

/// The causal tile of a query tile over its own keys, written for every query head.
struct Diagonal;

impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, All: AxisName, W: AxisName>
    Op<&Input<'_, KvHeads, Groups, Head, All, W>> for Diagonal
{
    type Output = DmTensor<i8, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, m![Rows % 8, Cache]>;

    fn forward(
        &self,
        context: &mut Device,
        input: &Input<'_, KvHeads, Groups, Head, All, W>,
    ) -> Self::Output {
        // SAFETY: the two row axes have the same contiguous 128-element order.
        unsafe { input.causal.view().reshape::<Chip, m![Rows, Cache]>() }.to_dm(&mut context.tdma)
    }
}

impl TileGroup {
    /// The group's four query tiles from batch row `row`, in place.
    fn queries<KvHeads: AxisName, Groups: AxisName, Head: AxisName, All: AxisName>(
        query: &HbmTensor<bf16, Chip, m![All, KvHeads, Groups, Head]>,
        row: usize,
    ) -> HbmTensorView<
        '_,
        bf16,
        Chip,
        m![{ Symbol<TileGroup> } # (All / Rows), Rows, KvHeads, Groups, Head],
    > {
        // SAFETY: the group's tiles are four contiguous 128-row query panels from any row.
        unsafe {
            query.view().tile::<m![All], 512,
            m![All = 512 # All, KvHeads, Groups, Head]>(row)
            .reshape::<
                Chip,
                m![{ Symbol<TileGroup> } # (All / Rows), Rows, KvHeads, Groups, Head],
            >()
        }
    }
}
impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName> Group<KvHeads, Groups, Head>
    for SingleTile
{
    type Fresh = Narrow;
    fn mask<All: AxisName, W: AxisName>(
        context: &mut Device,
        input: &Input<'_, KvHeads, Groups, Head, All, W>,
    ) -> GroupMask<KvHeads, Groups, Head, Self> {
        // SAFETY: unit tile and panel axes leave the tile's bytes in place.
        unsafe { Op::forward(&Diagonal, context, input).reshape() }
    }
    fn to_dm<All: AxisName>(
        context: &mut Device,
        query: &HbmTensor<bf16, Chip, m![All, KvHeads, Groups, Head]>,
        row: usize,
    ) -> history::Group<KvHeads, Groups, Head, Self> {
        // SAFETY: the tile is one contiguous 128-row query panel.
        let query = unsafe {
            query
                .view()
                .tile::<m![All], 128, m![All = 128 # All, KvHeads, Groups, Head]>(row)
                .reshape::<Chip, m![Rows # All, KvHeads, Groups, Head]>()
        };
        let in_dm: DmTensor<
            bf16,
            Chip,
            Cluster<KvHeads>,
            Slice<KvHeads, Groups>,
            m![Rows % 8, Head],
        > = query.to_dm(&mut context.tdma);
        // SAFETY: a unit tile axis leaves the tile's bytes in place.
        unsafe { in_dm.reshape() }
    }
    fn scores(
        attention: &Attention<KvHeads, Groups, Head>,
        context: &mut Device,
        source: (
            &mut impl KeySource<KvHeads, Groups, Head>,
            fresh::Keys<KvHeads, Groups, Head>,
        ),
        queries: &impl QueryTiles<KvHeads, Groups, Head>,
    ) -> history::PanelScores<KvHeads, Groups, Self::Fresh, Self> {
        let (source, first) = source;
        let mut scores = history::PanelScores::<KvHeads, Groups, Narrow, Self>::new();
        Op::forward(
            &ScoreBlock::<Narrow>(PhantomData),
            context,
            (attention, (first, source), queries, &mut scores),
        );
        scores
    }
    fn seed<S: ValueSource<KvHeads, Groups, Head>>(
        context: &mut Device,
        (scores, mask): (
            &history::PanelScores<KvHeads, Groups, Self::Fresh, Self>,
            &GroupMask<KvHeads, Groups, Head, Self>,
        ),
        source: &mut S,
        (carry, running): (
            &mut impl state::Tiles<KvHeads, Groups, Head>,
            &mut impl state::Tiles<KvHeads, Groups, Head>,
        ),
    ) {
        // SAFETY: the panel axis is outer to Cache, so each row keeps every key in its block order.
        let (score_rows, mask_rows) = unsafe {
            (
                scores.view().reshape::<Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>,
                    m![{ Symbol<Self> }, Rows % 8, { Keys<Narrow> }]>(),
                mask.view().reshape::<Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>,
                    m![{ Symbol<Self> }, Rows % 8, { Keys<Narrow> }]>(),
            )
        };
        let mut masked = history::Keyed::<KvHeads, Groups, Narrow, Self>::new();
        let mut probabilities = history::Keyed::<KvHeads, Groups, Narrow, Self>::new();
        let mut total = state::Running::<KvHeads, Groups, Self>::new();
        #[unroll]
        for t in 0..Self::SIZE {
            Op::forward(
                &panel::Mask,
                &mut *context,
                ((score_rows.clone(), mask_rows.clone()), &mut masked, t),
            );
            let top = Op::forward(&panel::Maximum::<Cache>(PhantomData), &mut *context, masked
                    .view()
                    .tile::<
                        m![{ Symbol<Self> }],
                        1,
                        m![{ TileOf<Self> }, Rows % 8, { Symbol<Cache> }],
                    >(t));
            Op::forward(
                &panel::Exponentiate,
                &mut *context,
                ((masked.view(), &top), (&mut probabilities, &mut total), t),
            );
            let lse = running.tile_mut(t);
            Op::forward(
                &panel::Lse,
                &mut *context,
                (
                    (
                        total
                            .view()
                            .tile::<m![{ Symbol<Self> }], 1, TileStatistic<Self>>(t),
                        &top,
                    ),
                    lse,
                ),
            );
        }
        let partials = Op::forward(
            &WeighBlock::<Narrow, Head>(PhantomData),
            &mut *context,
            (source, &probabilities),
        );
        Op::forward(&state::Seed, context, (carry, &partials, &total));
    }
    /// The panels of the tiles a single tile follows inside a block, each its own softmax.
    fn within<All: AxisName, W: AxisName>(
        context: &mut Device,
        input: &Input<'_, KvHeads, Groups, Head, All, W>,
        record: Record,
        queries: &history::Group<KvHeads, Groups, Head, Self>,
        (carry, running): (
            &mut state::Carry<KvHeads, Groups, Head, Self>,
            &mut state::Running<KvHeads, Groups, Self>,
        ),
    ) {
        let within = record.within();
        for r in 0..within.count() {
            let j = within.begin() + r;
            let mut keys: fresh::KeyPanel<KvHeads, Head> =
                Cache::rows(input.key, input.start + j * 128).to_dm(&mut context.tdma);
            let key = keys.key(context, 0);
            let mut scores = Scores::<KvHeads, Groups>::new();
            Op::forward(
                &fresh::TileScores,
                &mut *context,
                (
                    input.attention,
                    (queries, 0),
                    (key, &mut keys),
                    scores.view_mut(),
                ),
            );
            let (probabilities, lse) = Op::forward(&panel::Normalize, &mut *context, scores.view());
            let mut values = fresh::Staged::<KvHeads, Head>::new();
            Op::forward(
                &fresh::StageValues,
                &mut *context,
                (Cache::rows(input.value, input.start + j * 128), &mut values),
            );
            let partial = Op::forward(
                &panel::WeighParts::<Head>(PhantomData),
                &mut *context,
                (probabilities.view(), &mut values),
            );
            Op::forward(
                &state::Merge,
                &mut *context,
                (
                    (&mut *carry, &mut *running, 0),
                    (partial.view(), lse.view()),
                ),
            );
        }
    }
}
const _: () = assert!(TileGroup::SIZE == Span512::SIZE && SingleTile::SIZE == Span128::SIZE);
axes![Span512 = 4, Keys512 = 512, Span1024 = 8, Keys1024 = 1024];
axes![Span2048 = 16, Keys2048 = 2048];
// What a four-tile group's causal block's tiles see: the keys and panels before their own end,
// and the tiles from a panel's own on.
axes![
    Keys256 = 256,
    Keys384 = 384,
    Span256 = 2,
    Span384 = 3,
    Tail2 = 2,
    Tail3 = 3
];

/// A history block width: `Span` 128-key panels normalized as one `Keys`-wide row, of which
/// the table of `W` windows of history holds `Count<W>`.
pub(crate) trait Panel {
    type Span: AxisName;
    type Keys: AxisName;
    type Count<W: AxisName>: AxisName;
}
pub(crate) struct Half;
impl Panel for Half {
    type Span = Span512;
    type Keys = Keys512;
    type Count<W: AxisName> = Whole<W, 512>;
}
pub(crate) struct Wide;
impl Panel for Wide {
    type Span = Span1024;
    type Keys = Keys1024;
    type Count<W: AxisName> = Whole<W, 1024>;
}
pub(crate) struct Narrow;
impl Panel for Narrow {
    type Span = Span128;
    type Keys = Cache;
    type Count<W: AxisName> = Chunks<W>;
}
pub(crate) struct Double;
impl Panel for Double {
    type Span = Span2048;
    type Keys = Keys2048;
    type Count<W: AxisName> = Whole<W, 2048>;
}
const _: () = assert!(
    Keys2048::SIZE == Span2048::SIZE * Cache::SIZE
        && Keys512::SIZE == Span512::SIZE * Cache::SIZE
        && Keys1024::SIZE == Span1024::SIZE * Cache::SIZE
);

/// The whole `KEYS`-wide blocks of `W` windows of history.
#[derive(Clone, Debug)]
pub(crate) struct Whole<W, const KEYS: usize>(PhantomData<W>);

impl<W: AxisName, const KEYS: usize> AxisName for Whole<W, KEYS> {
    const NAME: Ident = match KEYS {
        512 => Ident::new("Count512"),
        1024 => Ident::new("Count1024"),
        _ => Ident::new("Count2048"),
    };
    const SIZE: usize = Chunks::<W>::SIZE * Cache::SIZE / KEYS;
}
/// The panel and key axes of a `P`-wide block.
type Panels<P> = m![{ Symbol<<P as Panel>::Span> }];
type Keys<P> = m![{ Symbol<<P as Panel>::Keys> }];
/// One tile of an owner over a group's query-tile axis `T`: read, and written in place.
type TileOf<T> = m![{ Symbol<T> } = 1 # ({ Symbol<T> })];
type TileSlot<T> = m![{ Symbol<T> } = 1 #{!} ({ Symbol<T> })];
type TileRows<Head, T> = m![{ TileOf<T> }, Rows % 8, Head];
type TileStatistic<T> = m![{ TileOf<T> }, Rows % 8, 1 # 8];
type TileStatisticSlot<T> = m![{ TileSlot<T> }, Rows % 8, 1 # 8];
type CachePanel<KvHeads, Head> = HbmTensor<bf16, Chip, m![Cache, KvHeads, Head]>;
type Cluster<KvHeads> = m![KvHeads / 4];
/// A slice's KV head of its cluster's four, its query group of four, and its eight rows of a tile;
/// fewer query groups leave the slices of the absent ones empty.
type Slice<KvHeads, Groups> = m![KvHeads % 4, Groups # 4, Rows / 8];
/// The channels of a head one register fill of a panel holds: the head, or a part of it.
#[derive(Clone, Debug)]
pub(crate) struct Part<Head>(PhantomData<Head>);
impl<Head: AxisName> AxisName for Part<Head> {
    const NAME: Ident = Ident::new("Part");
    const SIZE: usize = attention::channels(Head::SIZE, Cache::SIZE);
}
impl<Head: AxisName> Part<Head> {
    /// The parts of a head.
    const COUNT: usize = Head::SIZE / <Self as AxisName>::SIZE;
    /// Whether one fill holds the whole head. A wider head doubles each owner per head, so its
    /// groups of four tiles attend apart, not paired, over history blocks of at most 1024 keys.
    const WHOLE: bool = Self::COUNT == 1;
}
/// A part's channels of a head, and the parts.
type Channels<Head> = m![Head % ({ Symbol<Part<Head>> })];
type Parts<Head> = m![Head / ({ Symbol<Part<Head>> })];
/// One part of an owner over a head's parts: read, and written in place.
type PartOf<Head> = m![{ Parts<Head> } = 1 # ({ Parts<Head> })];
type PartSlot<Head> = m![{ Parts<Head> } = 1 #{!} ({ Parts<Head> })];
type Scalar<KvHeads, Groups> =
    VrfTensor<f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, m![Rows % 8, 1 # 8]>;
type Local<KvHeads, Groups> =
    DmTensor<f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, m![Rows % 8, 1 # 8]>;
type Scores<KvHeads, Groups> =
    DmTensor<bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, m![Rows % 8, Cache]>;
type Partial<KvHeads, Groups, Head> =
    DmTensor<f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, m![Rows % 8, Head]>;
pub(crate) type Table<W> = HbmTensor<i32, Chip, m![{ Symbol<Chunks<W>> }, Cache / Block]>;

pub(crate) type Causal = HbmTensor<i8, Chip, m![Cache, Cache]>;
/// Each key's column less each row's within a tile, `kk - ii`: where a band's edge crosses a tile.
pub(crate) type Offsets = HbmTensor<i32, Chip, m![Cache, Cache]>;
/// Each panel's first key, `128 p`, in the first of a flit's eight words.
pub(crate) type Starts<W> = HbmTensor<i32, Chip, m![{ Symbol<Chunks<W>> }, 1 # 8]>;
/// A chunk's tiles' first-row window keys, a tile's in the first of a flit's eight words.
pub(crate) type TileBases = HbmTensor<i32, Chip, m![TilePair, 1 # 8]>;
/// A group's tiles' first-row window keys, a tile's in the first of a flit's eight words.
type Bases<KvHeads, Groups, T> =
    DmTensor<i32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, m![{ Symbol<T> }, 1 # 8]>;
/// An I32 of every row and key of a tile.
type Grid<KvHeads, Groups> =
    DmTensor<i32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, m![Rows % 8, Cache]>;
/// One I32 on every row's slice, in the first of a flit's eight words.
type Word<KvHeads, Groups> =
    DmTensor<i32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, m![1 # 8]>;
/// A [`Word`] as a vector-engine operand.
type WordOperand<KvHeads, Groups> =
    VrfTensor<i32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, m![1 # 8]>;
type GroupMask<KvHeads, Groups, Head, T> = DmTensor<
    i8,
    Chip,
    Cluster<KvHeads>,
    Slice<KvHeads, Groups>,
    m![{ Symbol<T> }, Rows % 8, { Panels<<T as Group<KvHeads, Groups, Head>>::Fresh> }, Cache],
>;

pub(crate) struct Input<
    'a,
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    All: AxisName,
    W: AxisName,
> {
    pub attention: &'a Attention<KvHeads, Groups, Head>,
    pub query: &'a HbmTensor<bf16, Chip, m![All, KvHeads, Groups, Head]>,
    pub key: &'a HbmTensor<bf16, Chip, m![All, KvHeads, Head]>,
    pub value: &'a HbmTensor<bf16, Chip, m![All, KvHeads, Head]>,
    pub start: usize,
    pub causal: &'a Causal,
    pub table: &'a Table<W>,
    /// The history key count as the history gathers read it.
    pub history: &'a Value,
    /// The committed history length in every lane: the partial last history panel keeps only the
    /// keys the history gather defines, since the fresh rows are attended from `key` and `value`.
    pub lanes: &'a HbmTensor<i32, Chip, m![Cache]>,
    pub slots: &'a SlotTable<W>,
    /// Where the chunk's window binds, each tile's first row's first window key in the history
    /// it reads, the first of a flit's eight words.
    pub bases: &'a TileBases,
    pub offsets: &'a Offsets,
    pub starts: &'a Starts<W>,
}

/// The attended rows of a batch `All`.
pub(crate) type Output<KvHeads, Groups, Head, All> =
    HbmTensor<bf16, Chip, m![All, KvHeads, Groups, Head]>;

/// Attends a chunk's query records over its fresh keys, not their pool slots, and the pool's
/// history, pairs to single tiles; with `keeps` 1 the one-guest record runs `boundary`.
pub(super) struct Attend;

impl<
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    All: AxisName,
    Sequences: AxisName,
    W: AxisName,
    H,
    S,
    Env: Copy,
>
    Module<
        (
            (
                Input<'_, KvHeads, Groups, Head, All, W>,
                &Epoch<'_, Sequences, W>,
            ),
            (&Gather, &Blocks<KvHeads, Head>),
            &mut Output<KvHeads, Groups, Head, All>,
            ((&H, &mut S), usize),
        ),
        Env,
        &mut (),
    > for Attend
where
    H: for<'d, 's> Module<&'s mut S, Env, &'d mut (), Output = ()>,
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        env: Env,
        _: &mut (),
        ((input, epoch), pool, output, ((boundary, state), keeps)): (
            (
                Input<'_, KvHeads, Groups, Head, All, W>,
                &Epoch<'_, Sequences, W>,
            ),
            (&Gather, &Blocks<KvHeads, Head>),
            &mut Output<KvHeads, Groups, Head, All>,
            ((&H, &mut S), usize),
        ),
    ) {
        let runs = epoch.runs();
        let mut last_key = CachePanel::<KvHeads, Head>::new();
        let mut last_value = CachePanel::<KvHeads, Head>::new();
        let partial = runs.partial();
        for _ in 0..partial.count() {
            let p = partial.begin();
            // SAFETY: one immutable run of the table's entries, read in place.
            let entries: HbmTensor<i32, Chip, m![Cache / Block]> = unsafe {
                input
                    .table
                    .view()
                    .tile::<m![{ Symbol<Chunks<W>> }], 1,
                        m![{ Symbol<Chunks<W>> } = 1 # ({ Symbol<Chunks<W>> }), Cache / Block]>(p)
                    .alias()
            };
            let (key, value) = Module::forward(
                &history::LastPanel,
                &mut *context,
                env,
                &mut (),
                ((pool, &entries, input.history), p),
            );
            (last_key, last_value) = Module::forward(
                &history::Clean,
                &mut *context,
                env,
                &mut (),
                ((&key, &value), (input.slots, input.lanes, p)),
            );
        }
        let committed = (runs, (&last_key, &last_value));
        const { assert!(All::SIZE >= TilePair::SIZE * Rows::SIZE) };
        const {
            assert!(
                Groups::SIZE <= 4,
                "a KV head's query groups fill at most four slices"
            )
        };
        if const { Part::<Head>::WHOLE } {
            // Pairs cover whole chunks with or without history: a pair stores its tiles after
            // both groups, so the second group's loads do not queue behind the first's stores.
            for r in 0..runs.records::<TilePair>() {
                Pairs.forward(
                    &mut *context,
                    env,
                    &mut (),
                    (
                        &input,
                        (pool, &mut *output),
                        (committed, epoch.record::<TilePair>(r)),
                        ((boundary, &mut *state), keeps),
                    ),
                );
            }
        } else {
            // A pair's two groups attend apart, the second after the first's panels; only the
            // first runs `boundary`.
            for r in 0..runs.records::<TilePair>() {
                let record = epoch.record::<TilePair>(r);
                let mut kept = (DmTensor::new(), DmTensor::new());
                let mut own = (DmTensor::new(), DmTensor::new());
                Grouped::<TileGroup, 0>(PhantomData).forward(
                    &mut *context,
                    env,
                    &mut (),
                    (
                        &input,
                        (pool, &mut *output),
                        committed,
                        record,
                        ((boundary, &mut *state), keeps),
                        (&mut kept, &own),
                    ),
                );
                Grouped::<TileGroup, 1>(PhantomData).forward(
                    &mut *context,
                    env,
                    &mut (),
                    (
                        &input,
                        (pool, &mut *output),
                        committed,
                        record,
                        ((boundary, &mut *state), 0),
                        (&mut own, &kept),
                    ),
                );
            }
        }
        // A chunk holds at most two groups, so the block before any group or remainder tile is the
        // last group's own fresh panels: their fill layouts carry over from it.
        const _: () = assert!(Query::SIZE <= 2 * TileGroup::SIZE * Rows::SIZE);
        let mut previous: Kept<KvHeads, Head, Half> = (DmTensor::new(), DmTensor::new());
        for r in 0..runs.records::<TileGroup>() {
            let mut own = (DmTensor::new(), DmTensor::new());
            Grouped::<TileGroup, 0>(PhantomData).forward(
                &mut *context,
                env,
                &mut (),
                (
                    &input,
                    (pool, &mut *output),
                    committed,
                    epoch.record::<TileGroup>(r),
                    ((boundary, &mut *state), keeps),
                    (&mut own, &previous),
                ),
            );
            previous = own;
        }
        for r in 0..runs.records::<SingleTile>() {
            let mut own = (DmTensor::new(), DmTensor::new());
            Grouped::<SingleTile, 0>(PhantomData).forward(
                &mut *context,
                env,
                &mut (),
                (
                    &input,
                    (pool, &mut *output),
                    committed,
                    epoch.record::<SingleTile>(r),
                    ((boundary, &mut *state), keeps),
                    (&mut own, &previous),
                ),
            );
        }
    }
}

/// A chunk's committed history as a record reads it: the runs of its whole blocks, and the cleaned
/// keys and values of its partial last panel.
type Committed<'a, KvHeads, Head, W> = (
    Runs<W>,
    (&'a CachePanel<KvHeads, Head>, &'a CachePanel<KvHeads, Head>),
);

/// Attends `T` consecutive query tiles from the first of the record's group `G`: each tile's fresh
/// keys, then every history block and full panel once for all of them, then the partial last panel
/// per tile.
struct Grouped<T, const G: usize>(PhantomData<T>);

impl<T: AxisName, const G: usize> Grouped<T, G> {
    /// The group's first tile: group `G` of a record starts `G` groups past its first tile.
    fn first(record: Record) -> usize {
        if const { G == 0 } {
            record.first()
        } else {
            record.first() + G * T::SIZE
        }
    }

    /// The whole fresh blocks before the group's tiles: the record's, then its groups before `G`.
    fn preceding(record: Record) -> usize {
        if const { G == 0 } {
            record.preceding()
        } else {
            record.preceding() + G
        }
    }
}

impl<
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    T: Group<KvHeads, Groups, Head>,
    All: AxisName,
    W: AxisName,
    H,
    S,
    Env: Copy,
    const G: usize,
>
    Module<
        (
            &Input<'_, KvHeads, Groups, Head, All, W>,
            (
                (&Gather, &Blocks<KvHeads, Head>),
                &mut Output<KvHeads, Groups, Head, All>,
            ),
            Committed<'_, KvHeads, Head, W>,
            Record,
            ((&H, &mut S), usize),
            (
                &mut Kept<KvHeads, Head, T::Fresh>,
                &Kept<KvHeads, Head, Half>,
            ),
        ),
        Env,
        &mut (),
    > for Grouped<T, G>
where
    H: for<'d, 's> Module<&'s mut S, Env, &'d mut (), Output = ()>,
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        env: Env,
        _: &mut (),
        (input, (pool, output), (runs, last), record, boundary, (own, previous)): (
            &Input<'_, KvHeads, Groups, Head, All, W>,
            (
                (&Gather, &Blocks<KvHeads, Head>),
                &mut Output<KvHeads, Groups, Head, All>,
            ),
            Committed<'_, KvHeads, Head, W>,
            Record,
            ((&H, &mut S), usize),
            (
                &mut Kept<KvHeads, Head, T::Fresh>,
                &Kept<KvHeads, Head, Half>,
            ),
        ),
    ) {
        let first = Self::first(record);
        let mut source = Load {
            input,
            start: input.start + first * 128,
            kept: own,
            record,
            tile: const { G * T::SIZE },
        };
        // The first key panel loads ahead of the queries, so its fill runs under their copy.
        let key = source.key(context, 0);
        let queries = T::to_dm::<All>(context, input.query, input.start + first * 128);
        let mut carry = state::Carry::<KvHeads, Groups, Head, T>::new();
        let mut running = state::Running::<KvHeads, Groups, T>::new();
        let (boundary, keeps) = boundary;
        Module::forward(
            &CausalBlock::<T>(PhantomData),
            &mut *context,
            env,
            &mut (),
            (
                input,
                (key, &mut source),
                &queries,
                (&mut carry, &mut running),
                (|| record.guest(), boundary, keeps),
            ),
        );
        // At most one block precedes a group's tiles: the previous group's panels.
        for _ in 0..Self::preceding(record) {
            Module::forward(
                &Preceding::<Half, T>(PhantomData),
                &mut *context,
                env,
                &mut (),
                (
                    input.attention,
                    &mut Reuse::<KvHeads, Head, Half>(previous),
                    &queries,
                    (&mut carry, &mut running),
                ),
            );
        }
        T::within::<All, W>(context, input, record, &queries, (&mut carry, &mut running));
        let double = runs.double();
        for b in 0..double.count() {
            if const { Part::<Head>::WHOLE } {
                Module::forward(
                    &history::FoldBlock::<Double>(PhantomData),
                    &mut *context,
                    env,
                    &mut (),
                    (
                        input.attention,
                        (input.table, pool),
                        double.begin() + b,
                        &queries,
                        (&mut carry, &mut running),
                    ),
                );
            } else {
                // A wider head's 2048-key block is gathered and folded a 1024-key half at a time.
                #[unroll]
                for half in 0..Keys2048::SIZE / Keys1024::SIZE {
                    let block = Op::forward(
                        &history::Gathered::<Wide>(PhantomData),
                        &mut *context,
                        (
                            (input.table, pool),
                            (double.begin() + b) * (Keys2048::SIZE / Keys1024::SIZE) + half,
                        ),
                    );
                    Op::forward(
                        &history::FoldWide,
                        &mut *context,
                        (
                            input.attention,
                            &block,
                            &queries,
                            (&mut carry, &mut running),
                        ),
                    );
                }
            }
        }
        Module::forward(
            &Fold,
            &mut *context,
            env,
            &mut (),
            (
                input,
                pool,
                (runs.wide(), (runs, last)),
                (&queries, &mut carry, &mut running),
            ),
        );
        // A chunk whose window binds reads its history as panels, each attended per tile with the
        // keys before each row's window masked.
        let window = runs.window();
        for _ in 0..window.count().min(1) {
            // SAFETY: the two row axes have the same contiguous 128-element order.
            let offsets = unsafe { input.offsets.view().reshape::<Chip, m![Rows, Cache]>() };
            let offsets = offsets.to_dm(&mut context.tdma);
            let mut bases = Bases::<KvHeads, Groups, T>::new();
            #[unroll]
            for t in 0..T::SIZE {
                // SAFETY: one tile's flit of a table nothing writes.
                let base: HbmTensor<i32, Chip, m![1 # 8]> = unsafe {
                    input
                        .bases
                        .view()
                        .tile::<m![TilePair], 1, m![TilePair = 1 # TilePair, 1 # 8]>(first + t)
                        .alias()
                };
                base.view().to_dm_view(
                    &mut context.tdma,
                    bases
                        .view_mut()
                        .tile::<m![{ Symbol<T> }], 1, m![{ TileSlot<T> }, 1 # 8]>(t),
                );
            }
            for p in 0..window.count() {
                // SAFETY: one immutable run of the table's entries, read in place.
                let entries: HbmTensor<i32, Chip, m![Cache / Block]> = unsafe {
                    input
                        .table
                        .view()
                        .tile::<m![{ Symbol<Chunks<W>> }], 1, m![
                            { Symbol<Chunks<W>> } = 1 # ({ Symbol<Chunks<W>> }), Cache / Block
                        ]>(p)
                        .alias()
                };
                let (key, value) = Module::forward(
                    &history::LastPanel,
                    &mut *context,
                    env,
                    &mut (),
                    ((pool, &entries, input.history), p),
                );
                let panel = Module::forward(
                    &history::Clean,
                    &mut *context,
                    env,
                    &mut (),
                    ((&key, &value), (input.slots, input.lanes, p)),
                );
                // SAFETY: one panel's flit of a table nothing writes.
                let start: HbmTensor<i32, Chip, m![1 # 8]> = unsafe {
                    input
                        .starts
                        .view()
                        .tile::<m![{ Symbol<Chunks<W>> }], 1,
                            m![{ Symbol<Chunks<W>> } = 1 # ({ Symbol<Chunks<W>> }), 1 # 8]>(p)
                        .alias()
                };
                let start: Word<KvHeads, Groups> = start.to_dm(&mut context.tdma);
                let band = Banded {
                    offsets: &offsets,
                    bases: &bases,
                    start: context
                        .sub
                        .begin(start.view())
                        .fetch::<m![1], m![1 # 8]>()
                        .collect::<m![1], m![1 # 8]>()
                        .to_vrf(),
                };
                Op::forward(
                    &Masked,
                    &mut *context,
                    (
                        (input, (&band, p)),
                        (&panel.0, &panel.1),
                        (&queries, &mut carry, &mut running),
                    ),
                );
            }
        }
        #[unroll]
        for i in 0..T::SIZE {
            let attended: DmTensor<
                bf16,
                Chip,
                Cluster<KvHeads>,
                Slice<KvHeads, Groups>,
                m![Rows % 8, Head],
            > = context
                .main
                .begin(
                    carry
                        .view()
                        .tile::<m![{ Symbol<T> }], 1, TileRows<Head, T>>(i),
                )
                .fetch::<m![Rows % 8, Head / 8], m![Head % 8]>()
                .collect::<m![Rows % 8, Head / 8], m![Head % 8]>()
                .cast::<bf16, m![Head % 8 # 16]>()
                .commit_trim::<m![Head % 8]>()
                .commit();
            // SAFETY: a unit tile axis leaves the tile's bytes in place.
            let tile: DmTensorView<
                '_,
                bf16,
                Chip,
                Cluster<KvHeads>,
                Slice<KvHeads, Groups>,
                m![{ Symbol<SingleTile> }, Rows % 8, Head],
            > = unsafe { attended.view().reshape() };
            let row = input.start + (first + i) * 128;
            tile.to_hbm_view(
                &mut context.tdma,
                // SAFETY: the run's rows from `row` are its tiles' rows in order.
                unsafe {
                    output
                        .view_mut()
                        .tile::<m![All], 128, m![All = 128 #{!} All, KvHeads, Groups, Head]>(row)
                        .reshape::<Chip, m![
                            { Symbol<SingleTile> } #{!} (All / Rows),
                            Rows,
                            KvHeads,
                            Groups,
                            Head
                        ]>()
                },
            );
        }
    }
}

/// A pair's record's two tile groups, each history block (at most 1024 keys: no pair owner holds
/// a 2048-key block) read once for both; with `keeps` 1 group 0 runs `boundary` `guest` times.
struct Pairs;

impl<
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    All: AxisName,
    W: AxisName,
    H,
    S,
    Env: Copy,
>
    Module<
        (
            &Input<'_, KvHeads, Groups, Head, All, W>,
            (
                (&Gather, &Blocks<KvHeads, Head>),
                &mut Output<KvHeads, Groups, Head, All>,
            ),
            (Committed<'_, KvHeads, Head, W>, Record),
            ((&H, &mut S), usize),
        ),
        Env,
        &mut (),
    > for Pairs
where
    H: for<'d, 's> Module<&'s mut S, Env, &'d mut (), Output = ()>,
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        env: Env,
        _: &mut (),
        (input, (pool, output), ((runs, last), record), (boundary, keeps)): (
            &Input<'_, KvHeads, Groups, Head, All, W>,
            (
                (&Gather, &Blocks<KvHeads, Head>),
                &mut Output<KvHeads, Groups, Head, All>,
            ),
            (Committed<'_, KvHeads, Head, W>, Record),
            ((&H, &mut S), usize),
        ),
    ) {
        let mut queries = Halved::<KvHeads, Groups, bf16, m![Head]>::new();
        let mut carry = Halved::<KvHeads, Groups, f32, m![Head]>::new();
        let mut running = Halved::<KvHeads, Groups, f32, m![1 # 8]>::new();
        // Group 0's own panels are group 1's preceding block, so their fill layouts are kept.
        let mut kept = (DmTensor::new(), DmTensor::new());
        let mut own = (DmTensor::new(), DmTensor::new());
        // Both groups' causal masks are the same tiles: group 1 reads the one group 0 writes.
        let mask = {
            let (key, mut source) = Module::forward(
                &ToDm::<0>,
                &mut *context,
                env,
                &mut (),
                (input, &mut kept, record, &mut queries),
            );
            Module::forward(
                &CausalBlock::<TileGroup>(PhantomData),
                &mut *context,
                env,
                &mut (),
                (
                    input,
                    (key, &mut source),
                    &Paired::<_, 0>(&queries),
                    (
                        &mut Paired::<_, 0>(&mut carry),
                        &mut Paired::<_, 0>(&mut running),
                    ),
                    (|| record.guest(), boundary, keeps),
                ),
            )
        };
        {
            let (key, mut source) = Module::forward(
                &ToDm::<1>,
                &mut *context,
                env,
                &mut (),
                (input, &mut own, record, &mut queries),
            );
            let queries = Paired::<_, 1>(&queries);
            let scores = <TileGroup as Group<KvHeads, Groups, Head>>::scores(
                input.attention,
                context,
                (&mut source, key),
                &queries,
            );
            <TileGroup as Group<KvHeads, Groups, Head>>::seed(
                context,
                (&scores, &mask),
                &mut source,
                (
                    &mut Paired::<_, 1>(&mut carry),
                    &mut Paired::<_, 1>(&mut running),
                ),
            );
        }
        // The first 1024-key block is written under group 1's preceding block, not after it; with
        // less history the gather's prefix reads only committed rows, which no block then uses.
        let early = Module::forward(
            &history::Gathered::<Wide>(PhantomData),
            &mut *context,
            env,
            &mut (),
            ((input.table, pool, input.history), 0),
        );
        Module::forward(
            &Preceding::<Half, TileGroup>(PhantomData),
            &mut *context,
            env,
            &mut (),
            (
                input.attention,
                &mut Reuse::<KvHeads, Head, Half>(&kept),
                &Paired::<_, 1>(&queries),
                (
                    &mut Paired::<_, 1>(&mut carry),
                    &mut Paired::<_, 1>(&mut running),
                ),
            ),
        );
        // SAFETY: the groups' tiles, in order, are the pair's tiles.
        let (queries, mut carry, mut running): (
            history::Group<KvHeads, Groups, Head, TilePair>,
            state::Carry<KvHeads, Groups, Head, TilePair>,
            state::Running<KvHeads, Groups, TilePair>,
        ) = unsafe { (queries.reshape(), carry.reshape(), running.reshape()) };
        for _ in 0..runs.early().count() {
            Module::forward(
                &history::FoldWide,
                &mut *context,
                env,
                &mut (),
                (
                    input.attention,
                    &early,
                    &queries,
                    (&mut carry, &mut running),
                ),
            );
        }
        Module::forward(
            &Fold,
            &mut *context,
            env,
            &mut (),
            (
                input,
                pool,
                (runs.rest(), (runs, last)),
                (&queries, &mut carry, &mut running),
            ),
        );
        // One cast and one store of the pair's 1024 rows: a store per tile held each next cast
        // behind it, and each transfer queues before its bytes.
        let attended: DmTensor<
            bf16,
            Chip,
            Cluster<KvHeads>,
            Slice<KvHeads, Groups>,
            m![{ Symbol<TilePair> }, Rows % 8, Head],
        > = context
            .main
            .begin(carry.view())
            .fetch::<m![{ Symbol<TilePair> }, Rows % 8, Head / 8], m![Head % 8]>()
            .collect::<m![{ Symbol<TilePair> }, Rows % 8, Head / 8], m![Head % 8]>()
            .cast::<bf16, m![Head % 8 # 16]>()
            .commit_trim::<m![Head % 8]>()
            .commit();
        let row = input.start + record.first() * 128;
        attended.view().to_hbm_view(
            &mut context.tdma,
            // SAFETY: the run's rows from `row` are its tiles' rows in order.
            unsafe {
                output
                    .view_mut()
                    .tile::<m![All], 1024, m![All = 1024 #{!} All, KvHeads, Groups, Head]>(row)
                    .reshape::<Chip,
                        m![{ Symbol<TilePair> } #{!} (All / Rows), Rows, KvHeads, Groups, Head]>()
            },
        );
    }
}

/// A pair's owner of per-row values over its two groups' tiles.
type Halved<KvHeads, Groups, D, E> = DmTensor<
    D,
    Chip,
    Cluster<KvHeads>,
    Slice<KvHeads, Groups>,
    m![{ Symbol<Halves> }, { Symbol<TileGroup> }, Rows % 8, { E }],
>;
/// One group's half of a pair's owner, read and written in place, and one tile of that half.
type HalfOf<E> = m![{ Symbol<Halves> } = 1 # ({ Symbol<Halves> }), { Symbol<TileGroup> },
    Rows % 8, { E }];
type HalfSlot<E> = m![{ Symbol<Halves> } = 1 #{!} ({ Symbol<Halves> }), { Symbol<TileGroup> },
    Rows % 8, { E }];
type HalfTile<E> = m![{ Symbol<Halves> } = 1 # ({ Symbol<Halves> }), { TileOf<TileGroup> },
    Rows % 8, { E }];
type HalfTileSlot<E> = m![{ Symbol<Halves> } = 1 #{!} ({ Symbol<Halves> }),
    { TileSlot<TileGroup> }, Rows % 8, { E }];

/// A view of query tiles in their slices.
type QueryView<'a, KvHeads, Groups, E> =
    DmTensorView<'a, bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, E>;

/// A group's queries: an owner of its own, or its half of a pair's.
trait QueryTiles<KvHeads: AxisName, Groups: AxisName, Head: AxisName> {
    type Group: M;
    type From<R: AxisName>: M;
    type GroupPart: M;
    type FromPart<R: AxisName>: M;
    fn group(
        &self,
    ) -> DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, Self::Group>;
    /// Tiles `s..s + N` as a run of `R`, `N` = `R::SIZE`.
    fn from<R: AxisName, const N: usize>(
        &self,
        s: usize,
    ) -> DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, Self::From<R>>;
    /// Channel part `j` of every tile.
    fn group_part(&self, j: usize) -> QueryView<'_, KvHeads, Groups, Self::GroupPart>;
    /// Channel part `j` of tiles `s..s + N` as a run of `R`.
    fn part_from<R: AxisName, const N: usize>(
        &self,
        s: usize,
        j: usize,
    ) -> QueryView<'_, KvHeads, Groups, Self::FromPart<R>>;
}
impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, T: AxisName>
    QueryTiles<KvHeads, Groups, Head> for history::Group<KvHeads, Groups, Head, T>
{
    type Group = m![{ Symbol<T> }, Rows % 8, Head];
    type From<R: AxisName> = m![{ Symbol<R> } # ({ Symbol<T> }), Rows % 8, Head];
    type GroupPart = m![{ Symbol<T> }, Rows % 8, { PartOf<Head> }, { Channels<Head> }];
    type FromPart<R: AxisName> =
        m![{ Symbol<R> } # ({ Symbol<T> }), Rows % 8, { PartOf<Head> }, { Channels<Head> }];
    fn group(
        &self,
    ) -> DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, Self::Group> {
        self.view()
    }
    fn from<R: AxisName, const N: usize>(
        &self,
        s: usize,
    ) -> DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, Self::From<R>> {
        // SAFETY: the tiles from `s` keep their positions in the group's footprint.
        unsafe {
            self.view()
            .tile::<
                m![{ Symbol<T> }],
                N,
                m![{ Symbol<T> } = { N } # ({ Symbol<T> }), Rows % 8, Head],
            >(s)
            .reshape()
        }
    }
    fn group_part(&self, j: usize) -> QueryView<'_, KvHeads, Groups, Self::GroupPart> {
        self.view().tile::<Parts<Head>, 1, Self::GroupPart>(j)
    }
    fn part_from<R: AxisName, const N: usize>(
        &self,
        s: usize,
        j: usize,
    ) -> QueryView<'_, KvHeads, Groups, Self::FromPart<R>> {
        self.from::<R, N>(s)
            .tile::<Parts<Head>, 1, Self::FromPart<R>>(j)
    }
}
impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, const G: usize>
    QueryTiles<KvHeads, Groups, Head> for Paired<&Halved<KvHeads, Groups, bf16, m![Head]>, G>
{
    type Group = HalfOf<m![Head]>;
    type From<R: AxisName> = m![{ Symbol<Halves> } = 1 # ({ Symbol<Halves> }),
        { Symbol<R> } # ({ Symbol<TileGroup> }), Rows % 8, Head];
    type GroupPart = HalfOf<m![{ PartOf<Head> }, { Channels<Head> }]>;
    type FromPart<R: AxisName> = m![{ Symbol<Halves> } = 1 # ({ Symbol<Halves> }),
        { Symbol<R> } # ({ Symbol<TileGroup> }), Rows % 8, { PartOf<Head> }, { Channels<Head> }];
    fn group(
        &self,
    ) -> DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, Self::Group> {
        self.0
            .view()
            .tile::<m![{ Symbol<Halves> }], 1, HalfOf<m![Head]>>(G)
    }
    fn from<R: AxisName, const N: usize>(
        &self,
        s: usize,
    ) -> DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, Self::From<R>> {
        // SAFETY: the tiles from `s` keep their positions in the group's footprint.
        unsafe {
            self.group()
            .tile::<m![{ Symbol<TileGroup> }], N, m![{ Symbol<Halves> } = 1 # ({ Symbol<Halves> }),
                { Symbol<TileGroup> } = { N } # ({ Symbol<TileGroup> }), Rows % 8, Head]>(s)
            .reshape()
        }
    }
    fn group_part(&self, j: usize) -> QueryView<'_, KvHeads, Groups, Self::GroupPart> {
        self.group().tile::<Parts<Head>, 1, Self::GroupPart>(j)
    }
    fn part_from<R: AxisName, const N: usize>(
        &self,
        s: usize,
        j: usize,
    ) -> QueryView<'_, KvHeads, Groups, Self::FromPart<R>> {
        self.from::<R, N>(s)
            .tile::<Parts<Head>, 1, Self::FromPart<R>>(j)
    }
}

/// Group `G`'s tiles of a pair's owner, through a shared or a mutable reference `O`.
struct Paired<O, const G: usize>(O);
impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, E: M, const G: usize>
    state::Tiles<KvHeads, Groups, Head> for Paired<&mut Halved<KvHeads, Groups, f32, E>, G>
{
    type Group = HalfOf<E>;
    type GroupSlot = HalfSlot<E>;
    type Tile = HalfTile<E>;
    type TileSlot = HalfTileSlot<E>;
    type TilePart = HalfTile<m![{ PartOf<Head> }, { Channels<Head> }]>;
    type TilePartSlot = HalfTileSlot<m![{ PartSlot<Head> }, { Channels<Head> }]>;
    fn group(
        &self,
    ) -> DmTensorView<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, HalfOf<E>> {
        self.0
            .view()
            .tile::<m![{ Symbol<Halves> }], 1, HalfOf<E>>(G)
    }
    fn group_mut(
        &mut self,
    ) -> DmTensorViewMut<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, HalfSlot<E>> {
        self.0
            .view_mut()
            .tile::<m![{ Symbol<Halves> }], 1, HalfSlot<E>>(G)
    }
    fn tile(
        &self,
        t: usize,
    ) -> DmTensorView<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, HalfTile<E>> {
        <Self as state::Tiles<KvHeads, Groups, Head>>::group(self)
            .tile::<m![{ Symbol<TileGroup> }], 1, HalfTile<E>>(t)
    }
    fn tile_mut(
        &mut self,
        t: usize,
    ) -> DmTensorViewMut<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, HalfTileSlot<E>>
    {
        <Self as state::Tiles<KvHeads, Groups, Head>>::group_mut(self)
            .tile::<m![{ Symbol<TileGroup> }], 1, HalfTileSlot<E>>(t)
    }
    fn tile_part(
        &self,
        t: usize,
        j: usize,
    ) -> DmTensorView<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, Self::TilePart> {
        <Self as state::Tiles<KvHeads, Groups, Head>>::tile(self, t)
            .tile::<Parts<Head>, 1, Self::TilePart>(j)
    }
    fn tile_part_mut(
        &mut self,
        t: usize,
        j: usize,
    ) -> DmTensorViewMut<'_, f32, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, Self::TilePartSlot>
    {
        <Self as state::Tiles<KvHeads, Groups, Head>>::tile_mut(self, t)
            .tile::<Parts<Head>, 1, Self::TilePartSlot>(j)
    }
}

/// Writes group `G` of a pair's record: the first panel of its fresh keys, its fill layouts kept
/// in `own`, then its queries into their half of `queries`.
struct ToDm<const G: usize>;

impl<
    'a,
    'b,
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    All: AxisName,
    W: AxisName,
    const G: usize,
>
    Op<(
        &'a Input<'b, KvHeads, Groups, Head, All, W>,
        &'a mut Kept<KvHeads, Head, Half>,
        Record,
        &mut Halved<KvHeads, Groups, bf16, m![Head]>,
    )> for ToDm<G>
{
    type Output = (
        fresh::Keys<KvHeads, Groups, Head>,
        Load<'a, 'b, KvHeads, Groups, Head, All, W, Half>,
    );

    fn forward(
        &self,
        context: &mut Device,
        (input, own, record, queries): (
            &'a Input<'b, KvHeads, Groups, Head, All, W>,
            &'a mut Kept<KvHeads, Head, Half>,
            Record,
            &mut Halved<KvHeads, Groups, bf16, m![Head]>,
        ),
    ) -> Self::Output {
        let row = input.start + (record.first() + G * TileGroup::SIZE) * 128;
        let mut source = Load {
            input,
            start: row,
            kept: own,
            record,
            tile: G * TileGroup::SIZE,
        };
        // The first key panel loads ahead of the queries, so its fill runs under their copy.
        let key = source.key(context, 0);
        TileGroup::queries::<KvHeads, Groups, Head, All>(input.query, row).to_dm_view(
            &mut context.tdma,
            queries
                .view_mut()
                .tile::<m![{ Symbol<Halves> }], 1, HalfSlot<m![Head]>>(G),
        );
        (key, source)
    }
}

/// Folds the run `wide` of 1024-key blocks, then the chunk's 512-key block and full 128-key panels,
/// into all `T` query tiles at once, then the partial last panel into each tile.
struct Fold;

impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, T: AxisName, All: AxisName, W: AxisName>
    Op<(
        &Input<'_, KvHeads, Groups, Head, All, W>,
        (&Gather, &Blocks<KvHeads, Head>),
        (Run, Committed<'_, KvHeads, Head, W>),
        (
            &history::Group<KvHeads, Groups, Head, T>,
            &mut state::Carry<KvHeads, Groups, Head, T>,
            &mut state::Running<KvHeads, Groups, T>,
        ),
    )> for Fold
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (input, pool, (wide, (runs, last)), (queries, carry, running)): (
            &Input<'_, KvHeads, Groups, Head, All, W>,
            (&Gather, &Blocks<KvHeads, Head>),
            (Run, Committed<'_, KvHeads, Head, W>),
            (
                &history::Group<KvHeads, Groups, Head, T>,
                &mut state::Carry<KvHeads, Groups, Head, T>,
                &mut state::Running<KvHeads, Groups, T>,
            ),
        ),
    ) {
        let blocks = (input.table, pool);
        for b in 0..wide.count() {
            let block = Op::forward(
                &history::Gathered::<Wide>(PhantomData),
                &mut *context,
                (blocks, wide.begin() + b),
            );
            Op::forward(
                &history::FoldWide,
                &mut *context,
                (
                    input.attention,
                    &block,
                    queries,
                    (&mut *carry, &mut *running),
                ),
            );
        }
        let half = runs.half();
        for b in 0..half.count() {
            Op::forward(
                &history::FoldBlock::<Half>(PhantomData),
                &mut *context,
                (
                    input.attention,
                    blocks,
                    half.begin() + b,
                    queries,
                    (&mut *carry, &mut *running),
                ),
            );
        }
        let narrow = runs.narrow();
        for r in 0..narrow.count() {
            Op::forward(
                &history::FoldBlock::<Narrow>(PhantomData),
                &mut *context,
                (
                    input.attention,
                    blocks,
                    narrow.begin() + r,
                    queries,
                    (&mut *carry, &mut *running),
                ),
            );
        }
        // Only the partial last panel, masked to the committed keys, is attended per tile.
        let partial = runs.partial();
        for _ in 0..partial.count() {
            Op::forward(
                &Masked,
                &mut *context,
                (
                    (input, (&(), partial.begin())),
                    last,
                    (queries, &mut *carry, &mut *running),
                ),
            );
        }
    }
}

/// What masks each tile's scores of a history panel past what the panel's end mask does.
trait Band<KvHeads: AxisName, Groups: AxisName> {
    /// Tile `tile`'s scores of the panel, masked.
    fn mask(
        &self,
        context: &mut Device,
        scores: Scores<KvHeads, Groups>,
        tile: usize,
    ) -> Scores<KvHeads, Groups>;
}

/// Each row sees every key of the panel before the history's end.
impl<KvHeads: AxisName, Groups: AxisName> Band<KvHeads, Groups> for () {
    fn mask(
        &self,
        _: &mut Device,
        scores: Scores<KvHeads, Groups>,
        _: usize,
    ) -> Scores<KvHeads, Groups> {
        scores
    }
}

/// A window's band over a panel: row `ii` of a group's tile `t` sees the panel's keys from
/// `bases[t] + ii` on, the first key its window holds in the history the chunk reads.
struct Banded<'a, KvHeads: AxisName, Groups: AxisName, T: AxisName> {
    offsets: &'a Grid<KvHeads, Groups>,
    bases: &'a Bases<KvHeads, Groups, T>,
    /// The panel's first key.
    start: WordOperand<KvHeads, Groups>,
}

impl<KvHeads: AxisName, Groups: AxisName, T: AxisName> Band<KvHeads, Groups>
    for Banded<'_, KvHeads, Groups, T>
{
    fn mask(
        &self,
        context: &mut Device,
        scores: Scores<KvHeads, Groups>,
        tile: usize,
    ) -> Scores<KvHeads, Groups> {
        let base: WordOperand<KvHeads, Groups> = context
            .sub
            .begin(
                self.bases
                    .view()
                    .tile::<m![{ Symbol<T> }], 1, m![{ TileOf<T> }, 1 # 8]>(tile),
            )
            .fetch::<m![1], m![1 # 8]>()
            .collect::<m![1], m![1 # 8]>()
            .to_vrf();
        // Key `kk` of the panel lies in row `ii`'s window where `start + kk - ii - base` is not
        // negative.
        let distance: Grid<KvHeads, Groups> = context
            .main
            .begin(self.offsets.view())
            .fetch::<m![Rows % 8, Cache / 8], m![Cache % 8]>()
            .collect::<m![Rows % 8, Cache / 8], m![Cache % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::SubFxp, &base)
            .vector_clip(ClipBinaryOpI32::AddFxp, &self.start)
            .vector_final()
            .commit_trim::<m![Cache % 8]>()
            .commit();
        let seen: history::BandMask<KvHeads, Groups> = context
            .main
            .begin(distance.view())
            .fetch::<m![Rows % 8, Cache / 8], m![Cache % 8]>()
            .collect::<m![Rows % 8, Cache / 8], m![Cache % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Comparison([
                Cmp::Greater(-1i32),
                Cmp::False,
                Cmp::False,
                Cmp::False,
            ]))
            .vector_log(&mut context.sub);
        Op::forward(&history::MaskBand, context, (&scores, &seen))
    }
}

/// A history panel attended by each tile of a group: its cleaned keys and values, the scores
/// masked to the history's keys and as `band` says of each tile. The panel's key and value fills
/// and its end mask serve every tile; a head wider than one fill keeps the panel staged and fills
/// a tile's parts one after another.
struct Masked;

impl<
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    T: AxisName,
    All: AxisName,
    W: AxisName,
    B: Band<KvHeads, Groups>,
>
    Op<(
        (&Input<'_, KvHeads, Groups, Head, All, W>, (&B, usize)),
        (&CachePanel<KvHeads, Head>, &CachePanel<KvHeads, Head>),
        (
            &history::Group<KvHeads, Groups, Head, T>,
            &mut state::Carry<KvHeads, Groups, Head, T>,
            &mut state::Running<KvHeads, Groups, T>,
        ),
    )> for Masked
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        ((input, (band, at)), last, (queries, carry, running)): (
            (&Input<'_, KvHeads, Groups, Head, All, W>, (&B, usize)),
            (&CachePanel<KvHeads, Head>, &CachePanel<KvHeads, Head>),
            (
                &history::Group<KvHeads, Groups, Head, T>,
                &mut state::Carry<KvHeads, Groups, Head, T>,
                &mut state::Running<KvHeads, Groups, T>,
            ),
        ),
    ) {
        if const { Part::<Head>::WHOLE } {
            let key = Op::forward(
                &fresh::LoadKeys::<KvHeads, Groups, Head>(PhantomData),
                &mut *context,
                last.0.view(),
            );
            let mask = Op::forward(
                &history::LastKeys::<KvHeads, Groups>(PhantomData),
                &mut *context,
                (input.lanes, input.slots, at),
            );
            let value = Op::forward(
                &fresh::LoadValues::<KvHeads, Groups, Head>(PhantomData),
                &mut *context,
                last.1.view(),
            );
            #[unroll]
            for i in 0..T::SIZE {
                let query = queries
                    .view()
                    .tile::<m![{ Symbol<T> }], 1, TileRows<Head, T>>(i);
                let mut scores = Scores::<KvHeads, Groups>::new();
                Op::forward(
                    &fresh::FreshScores,
                    &mut *context,
                    (input.attention, query, &key, scores.view_mut()),
                );
                let scores = Op::forward(&history::MaskScores, &mut *context, (&scores, &mask));
                let scores = band.mask(context, scores, i);
                let (probabilities, lse) =
                    Op::forward(&panel::Normalize, &mut *context, scores.view());
                let mut partial = Partial::<KvHeads, Groups, Head>::new();
                Op::forward(
                    &panel::Weight,
                    &mut *context,
                    (probabilities.view(), &value, partial.view_mut()),
                );
                Op::forward(
                    &state::Merge,
                    &mut *context,
                    (
                        (&mut *carry, &mut *running, i),
                        (partial.view(), lse.view()),
                    ),
                );
            }
        } else {
            let mut keys: fresh::KeyPanel<KvHeads, Head> = last.0.view().to_dm(&mut context.tdma);
            let mask = Op::forward(
                &history::LastKeys::<KvHeads, Groups>(PhantomData),
                &mut *context,
                (input.lanes, input.slots, at),
            );
            let mut values = fresh::Staged::<KvHeads, Head>::new();
            Op::forward(
                &fresh::StageValues,
                &mut *context,
                (last.1.view(), &mut values),
            );
            #[unroll]
            for i in 0..T::SIZE {
                let key = keys.key(context, 0);
                let mut scores = Scores::<KvHeads, Groups>::new();
                Op::forward(
                    &fresh::TileScores,
                    &mut *context,
                    (
                        input.attention,
                        (queries, i),
                        (key, &mut keys),
                        scores.view_mut(),
                    ),
                );
                let scores = Op::forward(&history::MaskScores, &mut *context, (&scores, &mask));
                let scores = band.mask(context, scores, i);
                let (probabilities, lse) =
                    Op::forward(&panel::Normalize, &mut *context, scores.view());
                let partial = Op::forward(
                    &panel::WeighParts::<Head>(PhantomData),
                    &mut *context,
                    (probabilities.view(), &mut values),
                );
                Op::forward(
                    &state::Merge,
                    &mut *context,
                    (
                        (&mut *carry, &mut *running, i),
                        (partial.view(), lse.view()),
                    ),
                );
            }
        }
    }
}

impl Cache {
    /// The 128 fresh key or value rows from `row` as one cache panel.
    fn rows<KvHeads: AxisName, Head: AxisName, All: AxisName>(
        tensor: &HbmTensor<bf16, Chip, m![All, KvHeads, Head]>,
        row: usize,
    ) -> HbmTensorView<'_, bf16, Chip, m![Cache # All, KvHeads, Head]> {
        // SAFETY: the row relabel keeps the batch padding and every live row coordinate.
        unsafe {
            tensor
                .view()
                .tile::<m![All], 128, m![All = 128 # All, KvHeads, Head]>(row)
                .reshape::<Chip, m![Cache # All, KvHeads, Head]>()
        }
    }
}

/// Folds a block of `P::Span` fresh panels, all before the group's first row, into every tile of
/// the group as one unmasked block.
struct Preceding<P, T>(PhantomData<(P, T)>);

impl<
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    P: Panel,
    T: AxisName,
    S: KeySource<KvHeads, Groups, Head> + ValueSource<KvHeads, Groups, Head>,
    Q: QueryTiles<KvHeads, Groups, Head>,
    C: state::Tiles<KvHeads, Groups, Head>,
    R: state::Tiles<KvHeads, Groups, Head>,
>
    Op<(
        &Attention<KvHeads, Groups, Head>,
        &mut S,
        &Q,
        (&mut C, &mut R),
    )> for Preceding<P, T>
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (attention, source, queries, (carry, running)): (
            &Attention<KvHeads, Groups, Head>,
            &mut S,
            &Q,
            (&mut C, &mut R),
        ),
    ) {
        let key = source.key(context, 0);
        let mut scores = history::PanelScores::<KvHeads, Groups, P, T>::new();
        Op::forward(
            &ScoreBlock::<P>(PhantomData),
            &mut *context,
            (attention, (key, &mut *source), queries, &mut scores),
        );
        // SAFETY: the panel axis is outer to Cache, so each row keeps every key in its block order.
        let score_rows = unsafe {
            scores.view().reshape::<Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>,
                m![{ Symbol<T> }, Rows % 8, { Keys<P> }]>()
        };
        let mut probabilities = history::Keyed::<KvHeads, Groups, P, T>::new();
        let mut total = state::Running::<KvHeads, Groups, T>::new();
        #[unroll]
        for t in 0..T::SIZE {
            let top = Op::forward(&state::Operand, &mut *context, running.tile(t));
            Op::forward(
                &panel::Exponentiate,
                &mut *context,
                (
                    (score_rows.clone(), &top),
                    (&mut probabilities, &mut total),
                    t,
                ),
            );
        }
        let partials = Op::forward(
            &WeighBlock::<P, Head>(PhantomData),
            &mut *context,
            (source, &probabilities),
        );
        Op::forward(
            &state::Absorb,
            context,
            ((carry, running), &partials, &total),
        );
    }
}

/// A block's panels in the layouts their register-file fills read, each part's apart, kept for
/// another group.
type Kept<KvHeads, Head, P> = (
    DmTensor<
        bf16,
        Chip,
        Cluster<KvHeads>,
        fresh::Switched<KvHeads, Head>,
        m![{ Panels<P> }, { Parts<Head> }, Cache % 8, Head % 32],
    >,
    DmTensor<
        bf16,
        Chip,
        Cluster<KvHeads>,
        fresh::ValueSlice<KvHeads, Head>,
        m![{ Panels<P> }, { Parts<Head> }, Head % 16, Cache % 16],
    >,
);
type KeptPanel<P> = m![{ Symbol<<P as Panel>::Span> } = 1 # ({ Symbol<<P as Panel>::Span> })];
type KeptSlot<P> = m![{ Symbol<<P as Panel>::Span> } = 1 #{!} ({ Symbol<<P as Panel>::Span> })];

/// Where a block's keys come from: one register-file fill a channel part of a panel.
trait KeySource<KvHeads: AxisName, Groups: AxisName, Head: AxisName> {
    /// Panel `s`'s first part, the panel relaid first where it is fresh.
    fn key(&mut self, context: &mut Device, s: usize) -> fresh::Keys<KvHeads, Groups, Head>;
    /// Panel `s`'s part `j`, once its first is filled.
    fn key_part(
        &mut self,
        context: &mut Device,
        s: usize,
        j: usize,
    ) -> fresh::Keys<KvHeads, Groups, Head>;
}

/// Where a block's values come from: one register-file fill a channel part of a panel.
trait ValueSource<KvHeads: AxisName, Groups: AxisName, Head: AxisName> {
    /// Panel `s`'s first part, the panel relaid first where it is fresh.
    fn value(&mut self, context: &mut Device, s: usize) -> fresh::Values<KvHeads, Groups, Head>;
    /// Panel `s`'s part `j`, once its first is filled.
    fn value_part(
        &mut self,
        context: &mut Device,
        s: usize,
        j: usize,
    ) -> fresh::Values<KvHeads, Groups, Head>;
}

/// The fresh rows from `start`, loaded and relaid once, their fill layouts kept in `kept` for a
/// later block to [`Reuse`]; panel `s` is tile `tile + s` of `record`.
struct Load<
    'a,
    'b,
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    All: AxisName,
    W: AxisName,
    P: Panel,
> {
    input: &'a Input<'b, KvHeads, Groups, Head, All, W>,
    start: usize,
    kept: &'a mut Kept<KvHeads, Head, P>,
    record: Record,
    tile: usize,
}
impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, All: AxisName, W: AxisName, P: Panel>
    KeySource<KvHeads, Groups, Head> for Load<'_, '_, KvHeads, Groups, Head, All, W, P>
{
    fn key(&mut self, context: &mut Device, s: usize) -> fresh::Keys<KvHeads, Groups, Head> {
        let staged: fresh::KeyPanel<KvHeads, Head> =
            Cache::rows(self.input.key, self.start + s * 128).to_dm(&mut context.tdma);
        #[unroll]
        for j in 0..Part::<Head>::COUNT {
            Op::forward(
                &fresh::SwitchKeys,
                &mut *context,
                (
                    staged.view().tile::<
                        Parts<Head>,
                        1,
                        m![Cache % 2, { PartOf<Head> }, { Channels<Head> }],
                    >(j),
                    self.kept
                        .0
                        .view_mut()
                        .tile::<Panels<P>, 1,
                            m![{ KeptSlot<P> }, { Parts<Head> }, Cache % 8, Head % 32]>(s)
                        .tile::<Parts<Head>, 1,
                            m![{ KeptSlot<P> }, { PartSlot<Head> }, Cache % 8, Head % 32]>(j),
                ),
            );
        }
        Reuse::<KvHeads, Head, P>(&*self.kept).key(context, s)
    }
    fn key_part(
        &mut self,
        context: &mut Device,
        s: usize,
        j: usize,
    ) -> fresh::Keys<KvHeads, Groups, Head> {
        Reuse::<KvHeads, Head, P>(&*self.kept).key_part(context, s, j)
    }
}
impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, All: AxisName, W: AxisName, P: Panel>
    ValueSource<KvHeads, Groups, Head> for Load<'_, '_, KvHeads, Groups, Head, All, W, P>
{
    fn value(&mut self, context: &mut Device, s: usize) -> fresh::Values<KvHeads, Groups, Head> {
        let values = Cache::rows(self.input.value, self.start + s * 128);
        let mut staged = fresh::Staged::<KvHeads, Head>::new();
        Op::forward(
            &fresh::StageValues,
            &mut *context,
            (values.clone(), &mut staged),
        );
        // Rows past the chunk's last may hold any bits, and a key's zero probability times NaN is
        // NaN; only the chunk's last tile holds such rows.
        for _ in 0..self.record.cut(self.tile + s) {
            Op::forward(
                &fresh::Cut,
                &mut *context,
                (
                    (values.clone(), (self.input.causal, self.record.last())),
                    &mut staged,
                ),
            );
        }
        #[unroll]
        for j in 0..Part::<Head>::COUNT {
            Op::forward(
                &fresh::TransposeValues,
                &mut *context,
                (
                    staged
                        .view()
                        .tile::<Parts<Head>, 1, m![{ PartOf<Head> }, Cache % 16, Head % 16]>(j),
                    self.kept
                        .1
                        .view_mut()
                        .tile::<Panels<P>, 1,
                            m![{ KeptSlot<P> }, { Parts<Head> }, Head % 16, Cache % 16]>(s)
                        .tile::<Parts<Head>, 1,
                            m![{ KeptSlot<P> }, { PartSlot<Head> }, Head % 16, Cache % 16]>(j),
                ),
            );
        }
        Reuse::<KvHeads, Head, P>(&*self.kept).value(context, s)
    }
    fn value_part(
        &mut self,
        context: &mut Device,
        s: usize,
        j: usize,
    ) -> fresh::Values<KvHeads, Groups, Head> {
        Reuse::<KvHeads, Head, P>(&*self.kept).value_part(context, s, j)
    }
}

/// Panels another group's block already relaid, filled straight from their kept layouts.
struct Reuse<'a, KvHeads: AxisName, Head: AxisName, P: Panel>(&'a Kept<KvHeads, Head, P>);
impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, P: Panel> KeySource<KvHeads, Groups, Head>
    for Reuse<'_, KvHeads, Head, P>
{
    fn key(&mut self, context: &mut Device, s: usize) -> fresh::Keys<KvHeads, Groups, Head> {
        self.key_part(context, s, 0)
    }
    fn key_part(
        &mut self,
        context: &mut Device,
        s: usize,
        j: usize,
    ) -> fresh::Keys<KvHeads, Groups, Head> {
        Op::forward(
            &fresh::RegisterKeys::<Groups>(PhantomData),
            context,
            self.0
                .0
                .view()
                .tile::<Panels<P>, 1, m![{ KeptPanel<P> }, { Parts<Head> }, Cache % 8, Head % 32]>(s)
                .tile::<Parts<Head>, 1,
                    m![{ KeptPanel<P> }, { PartOf<Head> }, Cache % 8, Head % 32]>(j),
        )
    }
}
impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, P: Panel>
    ValueSource<KvHeads, Groups, Head> for Reuse<'_, KvHeads, Head, P>
{
    fn value(&mut self, context: &mut Device, s: usize) -> fresh::Values<KvHeads, Groups, Head> {
        self.value_part(context, s, 0)
    }
    fn value_part(
        &mut self,
        context: &mut Device,
        s: usize,
        j: usize,
    ) -> fresh::Values<KvHeads, Groups, Head> {
        Op::forward(
            &fresh::RegisterValues::<Groups>(PhantomData),
            context,
            self.0
                .1
                .view()
                .tile::<Panels<P>, 1, m![{ KeptPanel<P> }, { Parts<Head> }, Head % 16, Cache % 16]>(s)
                .tile::<Parts<Head>, 1,
                    m![{ KeptPanel<P> }, { PartOf<Head> }, Head % 16, Cache % 16]>(j),
        )
    }
}

/// The scaled scores of a block's `P::Span` panels for every tile of the group, into `scores`,
/// the first panel's first fill already made: a head one fill holds takes a contraction a panel,
/// a wider one a contraction a part, whose FP32 dot products sum before one rounding.
struct ScoreBlock<P>(PhantomData<P>);

impl<
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    P: Panel,
    T: AxisName,
    S: KeySource<KvHeads, Groups, Head>,
    Q: QueryTiles<KvHeads, Groups, Head>,
>
    Op<(
        &Attention<KvHeads, Groups, Head>,
        (fresh::Keys<KvHeads, Groups, Head>, &mut S),
        &Q,
        &mut history::PanelScores<KvHeads, Groups, P, T>,
    )> for ScoreBlock<P>
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (attention, (first, source), queries, scores): (
            &Attention<KvHeads, Groups, Head>,
            (fresh::Keys<KvHeads, Groups, Head>, &mut S),
            &Q,
            &mut history::PanelScores<KvHeads, Groups, P, T>,
        ),
    ) {
        let mut key = first;
        if const { Part::<Head>::WHOLE } {
            #[unroll]
            for s in 1..P::Span::SIZE {
                let next = source.key(context, s);
                Op::forward(
                    &history::Score::<P>(PhantomData),
                    &mut *context,
                    (attention, queries.group(), &key, &mut *scores, s - 1),
                );
                key = next;
            }
            Op::forward(
                &history::Score::<P>(PhantomData),
                context,
                (attention, queries.group(), &key, scores, P::Span::SIZE - 1),
            );
        } else {
            let parts = move |j| queries.group_part(j);
            #[unroll]
            for s in 0..P::Span::SIZE - 1 {
                let (sum, last, next) = Op::forward(
                    &history::Dots::<T>(PhantomData),
                    &mut *context,
                    (
                        attention,
                        parts,
                        (key, &mut *source, s),
                        move |source: &mut S, context: &mut Device| source.key(context, s + 1),
                    ),
                );
                Op::forward(
                    &history::RoundPanel::<P>(PhantomData),
                    &mut *context,
                    ((&sum, &last), &mut *scores, s),
                );
                key = next;
            }
            let (sum, last, ()) = Op::forward(
                &history::Dots::<T>(PhantomData),
                &mut *context,
                (
                    attention,
                    parts,
                    (key, &mut *source, P::Span::SIZE - 1),
                    |_: &mut S, _: &mut Device| (),
                ),
            );
            Op::forward(
                &history::RoundPanel::<P>(PhantomData),
                context,
                ((&sum, &last), scores, P::Span::SIZE - 1),
            );
        }
    }
}

/// Each tile's per-panel weighted values of a block's panels under the block's probabilities:
/// one value fill and one contraction per channel part of a panel, each into that part's
/// channels.
struct WeighBlock<P, Head>(PhantomData<(P, Head)>);

impl<
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    P: Panel,
    T: AxisName,
    S: ValueSource<KvHeads, Groups, Head>,
> Op<(&mut S, &history::Keyed<KvHeads, Groups, P, T>)> for WeighBlock<P, Head>
{
    type Output = history::Partials<KvHeads, Groups, Head, P, T>;

    fn forward(
        &self,
        context: &mut Device,
        (source, probabilities): (&mut S, &history::Keyed<KvHeads, Groups, P, T>),
    ) -> Self::Output {
        // SAFETY: the inverse relabel of the key rows.
        let probabilities = unsafe {
            probabilities.view().reshape::<Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>,
                m![{ Symbol<T> }, Rows % 8, { Panels<P> }, Cache]>()
        };
        let mut partials = history::Partials::<KvHeads, Groups, Head, P, T>::new();
        let mut value = source.value(context, 0);
        #[unroll]
        for s in 1..P::Span::SIZE {
            value = Op::forward(
                &WeighPanel::<P>(PhantomData),
                &mut *context,
                (
                    &probabilities,
                    (value, &mut *source, s - 1),
                    &mut partials,
                    move |source: &mut S, context: &mut Device| source.value(context, s),
                ),
            );
        }
        Op::forward(
            &WeighPanel::<P>(PhantomData),
            &mut *context,
            (
                &probabilities,
                (value, source, P::Span::SIZE - 1),
                &mut partials,
                |_: &mut S, _: &mut Device| (),
            ),
        );
        partials
    }
}

/// Panel `s`'s probabilities times its values for every tile of the group, one contraction a
/// channel part into that part's channels, the first part's fill already made; `next` makes the
/// following fill before the last contraction.
struct WeighPanel<P>(PhantomData<P>);

impl<
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    P: Panel,
    T: AxisName,
    S: ValueSource<KvHeads, Groups, Head>,
    N,
    F: FnOnce(&mut S, &mut Device) -> N,
>
    Op<(
        &DmTensorView<
            '_,
            bf16,
            Chip,
            Cluster<KvHeads>,
            Slice<KvHeads, Groups>,
            m![{ Symbol<T> }, Rows % 8, { Panels<P> }, Cache],
        >,
        (fresh::Values<KvHeads, Groups, Head>, &mut S, usize),
        &mut history::Partials<KvHeads, Groups, Head, P, T>,
        F,
    )> for WeighPanel<P>
{
    type Output = N;

    fn forward(
        &self,
        context: &mut Device,
        (probabilities, (first, source, s), partials, next): (
            &DmTensorView<
                '_,
                bf16,
                Chip,
                Cluster<KvHeads>,
                Slice<KvHeads, Groups>,
                m![{ Symbol<T> }, Rows % 8, { Panels<P> }, Cache],
            >,
            (fresh::Values<KvHeads, Groups, Head>, &mut S, usize),
            &mut history::Partials<KvHeads, Groups, Head, P, T>,
            F,
        ),
    ) -> Self::Output {
        let mut value = first;
        #[unroll]
        for j in 1..Part::<Head>::COUNT {
            let later = source.value_part(context, s, j);
            Op::forward(
                &history::Weigh::<P>(PhantomData),
                &mut *context,
                (probabilities, &value, &mut *partials, (s, j - 1)),
            );
            value = later;
        }
        let next = next(source, context);
        Op::forward(
            &history::Weigh::<P>(PhantomData),
            context,
            (
                probabilities,
                &value,
                partials,
                (s, Part::<Head>::COUNT - 1),
            ),
        );
        next
    }
}

#[cfg(test)]
mod tests {
    use super::super::chunks;
    use super::super::gqa::items::tests::{QueryPairs, UNIT, WINDOW, Wide, place};
    use super::*;
    use crate::tensor::axis::Rows;
    use crate::tensor::cache::Blocks;
    use crate::tensor::scalar::F32;
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;
    axes![KvHeads = 8, Groups = 4, Pairs = 2, Head = 128, Windows = 80];
    type FixtureRows = Rows<1024>;
    type Sequences = Rows<1>;
    type Queries<H, G, D> = HbmTensor<bf16, Chip, m![Query, H, G, D]>;
    type Fresh<H, D> = HbmTensor<bf16, Chip, m![FixtureRows, H, D]>;
    type Output<H, G, D> = HbmTensor<bf16, Chip, m![FixtureRows, H, G, D]>;

    /// The score multiplier of 128-wide heads, `1 / sqrt(128)` rounded to BF16 (0.08837890625), as
    /// FP32 bits.
    const SCALE: usize = 0x3db5_0000;

    /// One chunk's rows and the launch arrays its plan binds.
    #[derive(DeviceSend)]
    struct Fixture<H: AxisName, G: AxisName, D: AxisName> {
        query: Queries<H, G, D>,
        key: Fresh<H, D>,
        value: Fresh<H, D>,
        chunks: chunks::Invocation<Sequences, Windows>,
    }

    #[device(chip = 1)]
    fn attend_chunk<H: AxisName, G: AxisName, D: AxisName, const SCALE: usize>(
        device: &mut Device,
        scale: &F32<SCALE>,
        fixture: &Fixture<H, G, D>,
        blocks: &mut Blocks<H, D>,
        output: &mut Output<H, G, D>,
    ) {
        // SAFETY: fixture query rows are contiguous and share the same length.
        let query = unsafe {
            fixture
                .query
                .view()
                .reshape::<Chip, m![FixtureRows, H, G, D]>()
                .alias()
        };
        let attention = Attention {
            heads: PhantomData,
            scale: f32::from(*scale),
        };
        let (invocation, chunk) = (&fixture.chunks, fixture.chunks.chunk(0));
        Attend.forward(
            device,
            (),
            &mut (),
            (
                (
                    Input::<H, G, D, FixtureRows, Windows> {
                        attention: &attention,
                        query: &query,
                        key: &fixture.key,
                        value: &fixture.value,
                        start: 0,
                        causal: &invocation.causal,
                        table: &chunk.table,
                        history: &chunk.h,
                        lanes: &chunk.lanes,
                        slots: &invocation.slots,
                        bases: &chunk.bases,
                        offsets: &invocation.offsets,
                        starts: &invocation.starts,
                    },
                    &Epoch::new(&invocation.work, 0),
                ),
                (&Gather, &*blocks),
                output,
                ((&(), &mut ()), 0),
            ),
        );
    }

    fn kv(k: usize, head: usize) -> f32 {
        ((k * 3 + head * 5) % 29) as f32 / 32.0 - 0.4375
    }

    fn value(k: usize, head: usize) -> f32 {
        ((k * 7 + head * 3) % 31) as f32 / 16.0 - 0.9375
    }

    /// Exact in bf16, and distinct for every 16-channel run of a head.
    fn scale(channel: usize) -> f32 {
        (1 + channel / 16) as f32 / (8 << (channel % 4)) as f32
    }

    fn query_value(j: usize, head: usize, group: usize) -> f32 {
        ((j + head * 3 + group * 5) % 15) as f32 / 32.0 - 0.21875
    }

    /// A query's and a key's factors on channel `channel`, constant over 16-channel runs of
    /// periods 3 and 5, so no two 128-channel parts of a head are alike.
    fn factors(channel: usize) -> (f32, f32) {
        let run = channel / 16;
        ((1 + run % 3) as f32, (1 + run % 5) as f32)
    }

    /// The power of two each key of `D`-wide heads is divided by, which keeps every score under
    /// the multiplier of bits `SCALE` below 1.6, and that score per unit query and key.
    fn gain<D: AxisName, const SCALE: usize>() -> (f32, f64) {
        let products: f64 = (0..D::SIZE)
            .map(|channel| {
                let (query, key) = factors(channel);
                f64::from(query * key)
            })
            .sum();
        let gain = f64::from(f32::from(F32::<SCALE>)) * products;
        let shrink = 2.0f32.powi(gain.log2().floor() as i32 - 3);
        (shrink, gain / f64::from(shrink))
    }

    /// The first key row `j` of a chunk over `h` keys attends in `window`.
    fn lowest(h: usize, j: usize, window: Option<usize>) -> usize {
        window.map_or(0, |window| (h + j + 1).saturating_sub(window))
    }

    /// The attended rows of a chunk of `q` rows over `h` keys of a poisoned, shuffled pool of `H`
    /// KV heads of `D` channels, `G` query groups each, each row attending its last `window` keys,
    /// under the score multiplier of bits `SCALE`; scratch rows past the chunk's hold `past`, or
    /// else a NaN query and the next key's key and value, and live row `nan` holds a NaN value.
    /// Poison fills the keys below the chunk's first row's window, and one poisoned block takes
    /// the place of every block wholly below it, as a framework names its null block there.
    /// Attending leaves the pool's bytes.
    async fn attend<H: AxisName, G: AxisName, D: AxisName, const SCALE: usize>(
        context: &mut Device,
        (q, h, window): (usize, usize, Option<usize>),
        (past, nan): (Option<f32>, Option<usize>),
    ) -> Vec<bf16> {
        let scratch = |row: usize, own: f32, next: f32| {
            bf16::from_f32(if row < q { own } else { past.unwrap_or(next) })
        };
        let (heads, groups, width) = (H::SIZE, G::SIZE, D::SIZE);
        let (shrink, _) = gain::<D, SCALE>();
        let key = |k: usize, head: usize, channel: usize| kv(k, head) * factors(channel).1 / shrink;
        let blocks = 256;
        let physical = |logical: usize| (logical * 17 + 7) % blocks;
        let mut allocation = Blocks::<H, D> {
            tokens: (0..blocks)
                .map(|_| HbmTensor::alloc(&mut *context))
                .collect::<Result<_, _>>()
                .unwrap(),
        };
        let row = KeyValue::SIZE * heads * width;
        let mut data: Vec<bf16> = (0..blocks * Block::SIZE * row)
            .map(|at| bf16::from_f32([f32::NAN, f32::INFINITY, f32::NEG_INFINITY][at % 3]))
            .collect();
        let first = lowest(h, 0, window);
        for k in first..q + h {
            let token = physical(k / Block::SIZE) * Block::SIZE + k % Block::SIZE;
            for head in 0..heads {
                for channel in 0..width {
                    let at = token * row + head * width + channel;
                    data[at] = bf16::from_f32(key(k, head, channel));
                    data[at + heads * width] = bf16::from_f32(value(k, head) * scale(channel));
                }
            }
        }
        allocation
            .write(&mut *context, 0..blocks, data.iter().copied())
            .await
            .unwrap();
        let null = |i: usize| match (i + 1) * Block::SIZE <= first {
            true => 0,
            false => i,
        };
        let mut plan = chunks::Plan::<Windows>::new::<Sequences>();
        (plan.count, plan.q[0], plan.h[0]) = (1, q as i32, h as i32);
        plan.window = window;
        plan.lanes.fill(h as i32);
        for (i, entry) in plan.tables.iter_mut().enumerate() {
            *entry =
                i32::try_from(physical(null(i) % blocks)).unwrap() * Blocks::<H, D>::BLOCK_STRIDE;
        }
        let reads = plan.reads().unwrap();
        let reads = reads.iter().map(|(read, words)| (&read[..], *words));
        let mut buffers = place(&mut *context, reads).await;
        let masks = chunks::Masks::<Windows>::new(&mut *context).await.unwrap();
        let next = || Ok(buffers.next().expect("an array a buffer"));
        let invocation = chunks::Invocation::bind(next, &masks).unwrap();
        let query: Queries<H, G, D> = HostTensor::<bf16, m![Query, H, G, D]>::from_vec(
            (0..Query::SIZE * heads * groups * width).map(|at| {
                let j = at / (heads * groups * width);
                let (head, group) = (at / (groups * width) % heads, at / width % groups);
                let own = query_value(j, head, group) * factors(at % width).0;
                scratch(j, own, f32::NAN)
            }),
        )
        .to_hbm(&mut context.pdma)
        .await
        .expect("host transfer");
        let mut output: Output<H, G, D> =
            HostTensor::<bf16, m![FixtureRows, H, G, D]>::from_vec(vec![
                bf16::from_f32(
                    f32::INFINITY
                );
                FixtureRows::SIZE
                    * heads
                    * groups
                    * width
            ])
            .to_hbm(&mut context.pdma)
            .await
            .expect("host transfer");
        let fresh: Fresh<H, D> = HostTensor::<bf16, m![FixtureRows, H, D]>::from_vec(
            (0..FixtureRows::SIZE * heads * width).map(|at| {
                let (row, head, channel) = (at / (heads * width), at / width % heads, at % width);
                let own = key(h + row, head, channel);
                scratch(row, own, own)
            }),
        )
        .to_hbm(&mut context.pdma)
        .await
        .expect("host transfer");
        let values: Fresh<H, D> = HostTensor::<bf16, m![FixtureRows, H, D]>::from_vec(
            (0..FixtureRows::SIZE * heads * width).map(|at| {
                let (row, head, channel) = (at / (heads * width), at / width % heads, at % width);
                let own = match nan == Some(row) {
                    true => f32::NAN,
                    false => value(h + row, head) * scale(channel),
                };
                scratch(row, own, own)
            }),
        )
        .to_hbm(&mut context.pdma)
        .await
        .expect("host transfer");
        let fixture = Fixture {
            query,
            key: fresh,
            value: values,
            chunks: invocation,
        };
        let arguments = (
            &mut *context,
            &F32::<SCALE>,
            &fixture,
            &mut allocation,
            &mut output,
        );
        launch(attend_chunk, arguments)
            .await
            .expect("device launch");
        let cache = allocation.read(&mut *context, 0..blocks).await.unwrap();
        let written = cache
            .iter()
            .zip(&data)
            .position(|(found, put)| found.to_bits() != put.to_bits());
        assert_eq!(written, None, "q={q} h={h}: attending wrote the pool");
        output
            .to_host::<m![FixtureRows, H, G, D]>(&mut context.pdma)
            .await
            .expect("host transfer")
            .into_vec()
    }

    /// Each of the `q` live rows of `attend`'s chunk over `h` keys attends every key of its last
    /// `window` up to its own as an f64 softmax does, to 0.004 of each channel's magnitude, at
    /// least 1.
    fn check<H: AxisName, G: AxisName, D: AxisName, const SCALE: usize>(
        (q, h, window): (usize, usize, Option<usize>),
        got: &[bf16],
    ) {
        let (_, gain) = gain::<D, SCALE>();
        let mut max_error = 0.0f64;
        for j in 0..q {
            let keys = lowest(h, j, window)..=h + j;
            for head in 0..H::SIZE {
                for group in 0..G::SIZE {
                    let query = f64::from(query_value(j, head, group)) * gain;
                    let score = |k: usize| query * f64::from(kv(k, head));
                    let top = keys.clone().map(score).fold(f64::NEG_INFINITY, f64::max);
                    let (weighted, mass) = keys.clone().fold((0.0, 0.0), |(weighted, mass), k| {
                        let weight = (score(k) - top).exp();
                        (weighted + weight * f64::from(value(k, head)), mass + weight)
                    });
                    for channel in 0..D::SIZE {
                        let want = weighted / mass * f64::from(scale(channel));
                        let at = ((j * H::SIZE + head) * G::SIZE + group) * D::SIZE + channel;
                        let found = f64::from(got[at].to_f32());
                        let error = (found - want).abs();
                        max_error = max_error.max(error);
                        assert!(
                            error <= 0.004 * want.abs().max(1.0),
                            "q={q} h={h} window={window:?} row={j} head={head} group={group} \
                             channel={channel}: {found} != {want}"
                        );
                    }
                }
            }
        }
        println!("q={q} h={h} window={window:?} max_error={max_error}");
    }

    /// Scratch rows of NaN or infinities past live rows that end inside a 128-key panel give the
    /// finite outputs zeroed rows give, and a NaN value in the last live row reaches its output.
    #[tokio::test(flavor = "current_thread")]
    async fn ignores_scratch_rows_past_the_live_ones() {
        let _lease = DEVICE.lock().await;
        let mut context = Device::new(attend_chunk.topology()).unwrap();
        ignores_scratch_rows::<Groups>(&mut context).await;
        ignores_scratch_rows::<Pairs>(&mut context).await;
    }

    /// The scratch rows' check at `G` query groups a KV head.
    async fn ignores_scratch_rows<G: AxisName>(context: &mut Device) {
        let cases = [
            (1, 0),
            (127, 130),
            (129, 0),
            (300, 1100),
            (384, 0),
            (613, 2049),
            (1000, 0),
        ];
        let row = <m![KvHeads, G, Head]>::SIZE;
        let attend = attend::<KvHeads, G, Head, SCALE>;
        for (q, h) in cases {
            let zeroed = attend(&mut *context, (q, h, None), (Some(0.0), None)).await;
            let runs = [
                (f32::NAN, None),
                (f32::INFINITY, None),
                (f32::NAN, Some(q - 1)),
            ];
            for (past, nan) in runs {
                let got = attend(&mut *context, (q, h, None), (Some(past), nan)).await;
                let clean = nan.unwrap_or(q) * row;
                for (at, (got, zeroed)) in got[..clean].iter().zip(&zeroed[..clean]).enumerate() {
                    let (got, zeroed) = (got.to_f32(), zeroed.to_f32());
                    assert!(
                        got.is_finite() && got == zeroed,
                        "{} groups q={q} h={h} past={past} row={}: {got} != {zeroed}",
                        G::SIZE,
                        at / row
                    );
                }
                assert!(
                    got[clean..q * row].iter().all(|got| got.to_f32().is_nan()),
                    "{} groups q={q} h={h}: the last row's NaN value does not reach its output",
                    G::SIZE
                );
            }
        }
    }

    /// A chunk of any tile count attends causally over its own rows and every history block
    /// width (2048, 1024, 512 and 128 keys, a partial panel), below, at and above each width's
    /// edge, the pool holding NaN or infinities past the history and its blocks out of order: at
    /// four query groups a KV head, and at two, whose slices are half empty.
    #[tokio::test(flavor = "current_thread")]
    async fn attends_causally_with_poisoned_padding() {
        let _lease = DEVICE.lock().await;
        let mut context = Device::new(attend_chunk.topology()).unwrap();
        attends_causally::<Groups>(&mut context).await;
        attends_causally::<Pairs>(&mut context).await;
    }

    /// The causal check at `G` query groups a KV head.
    async fn attends_causally<G: AxisName>(context: &mut Device) {
        let cases = [
            (128, 0),
            (256, 0),
            (896, 0),
            (1024, 0),
            (1024, 1024),
            (128, 3000),
            (384, 1100),
            (1024, 2100),
            (2, 2048),
            (129, 127),
            (257, 513),
            (513, 1023),
            (641, 1024),
            (769, 2047),
            (897, 128),
            (1023, 2049),
        ];
        for (q, h) in cases {
            let chunk = (q, h, None);
            let got = attend::<KvHeads, G, Head, SCALE>(&mut *context, chunk, (None, None));
            check::<KvHeads, G, Head, SCALE>(chunk, &got.await);
        }
    }

    /// Chunks of 256-wide heads, two query groups to each of eight KV heads, attend causally under
    /// a unit score multiplier over their own rows, a partial last tile among them, and over every
    /// history block width, a 2048-key block among them: four tiles, two groups of a pair's
    /// eight, two single tiles and a group before a single tile.
    #[tokio::test(flavor = "current_thread")]
    async fn attends_causally_over_wide_heads() {
        let _lease = DEVICE.lock().await;
        let mut context = Device::new(attend_chunk.topology()).unwrap();
        let cases = [
            (300, 2866),
            (1024, 1100),
            (129, 2049),
            (513, 3000),
            (1023, 0),
        ];
        for (q, h) in cases {
            let chunk = (q, h, None);
            let got = attend::<KvHeads, QueryPairs, Wide, UNIT>(&mut context, chunk, (None, None));
            check::<KvHeads, QueryPairs, Wide, UNIT>(chunk, &got.await);
        }
    }

    /// Chunks of 128-wide heads in four query groups and of 256-wide heads in two attend only
    /// each row's last [`WINDOW`] keys, causally, as an f64 softmax does: a row whose window opens
    /// inside a block, rows whose windows reach exactly the history's first key and one past it,
    /// whole tiles whose first row sees all of a short history and whose last row sees none, and
    /// chunks of every tile domain, though poison fills the keys below the first row's window and
    /// one poisoned block every block wholly below it.
    #[tokio::test(flavor = "current_thread")]
    async fn attends_only_each_row_window() {
        let _lease = DEVICE.lock().await;
        let mut context = Device::new(attend_chunk.topology()).unwrap();
        let cases = [
            (1, 2 * WINDOW + 16),
            (2, WINDOW),
            (17, WINDOW - 17),
            (128, WINDOW),
            (300, 2866),
            (1024, 1100),
            (129, 2049),
            (513, 3000),
        ];
        for (q, h) in cases {
            let chunk = (q, h, Some(WINDOW));
            let got = attend::<KvHeads, Groups, Head, SCALE>(&mut context, chunk, (None, None));
            check::<KvHeads, Groups, Head, SCALE>(chunk, &got.await);
            let got = attend::<KvHeads, QueryPairs, Wide, UNIT>(&mut context, chunk, (None, None));
            check::<KvHeads, QueryPairs, Wide, UNIT>(chunk, &got.await);
        }
    }
}
