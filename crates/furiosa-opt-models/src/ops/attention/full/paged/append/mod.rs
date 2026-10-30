use furiosa_opt_std::prelude::*;

use super::Attention;
use super::chunks::{Cache, Chunks, Epoch, Record, Run, Runs, SlotTable};
use crate::ops::index::gather::Gather;
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
            &mut impl Source<KvHeads, Groups, Head, Self::Fresh>,
            fresh::Keys<KvHeads, Groups, Head>,
        ),
        queries: &impl QueryTiles<KvHeads, Groups, Head>,
    ) -> history::PanelScores<KvHeads, Groups, Self::Fresh, Self>;
    /// Masks and normalizes the causal block's scores and seeds the group's state with the
    /// block's weighted values.
    fn seed(
        context: &mut Device,
        block: (
            &history::PanelScores<KvHeads, Groups, Self::Fresh, Self>,
            &GroupMask<KvHeads, Groups, Head, Self>,
        ),
        source: &mut impl Source<KvHeads, Groups, Head, Self::Fresh>,
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
    /// panel stay unwritten.
    fn scores(
        attention: &Attention<KvHeads, Groups, Head>,
        context: &mut Device,
        (source, first): (
            &mut impl Source<KvHeads, Groups, Head, Self::Fresh>,
            fresh::Keys<KvHeads, Groups, Head>,
        ),
        queries: &impl QueryTiles<KvHeads, Groups, Head>,
    ) -> history::PanelScores<KvHeads, Groups, Self::Fresh, Self> {
        let mut scores = history::PanelScores::<KvHeads, Groups, Half, Self>::new();
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
        scores
    }
    /// Tile t sees panels 0..=t, so its softmax, weighing and sum read only those.
    fn seed(
        context: &mut Device,
        (scores, mask): (
            &history::PanelScores<KvHeads, Groups, Self::Fresh, Self>,
            &GroupMask<KvHeads, Groups, Head, Self>,
        ),
        source: &mut impl Source<KvHeads, Groups, Head, Self::Fresh>,
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
        let partials = {
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
                (&probabilities, &first, &mut partials, 0),
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
            &mut impl Source<KvHeads, Groups, Head, Self::Fresh>,
            fresh::Keys<KvHeads, Groups, Head>,
        ),
        queries: &impl QueryTiles<KvHeads, Groups, Head>,
    ) -> history::PanelScores<KvHeads, Groups, Self::Fresh, Self> {
        let (source, first) = source;
        let mut scores = history::PanelScores::<KvHeads, Groups, Narrow, Self>::new();
        Op::forward(
            &ScoreBlock::<Narrow>(PhantomData),
            context,
            (
                attention,
                (first, |context: &mut Device, s| source.key(context, s)),
                queries,
                &mut scores,
            ),
        );
        scores
    }
    fn seed(
        context: &mut Device,
        (scores, mask): (
            &history::PanelScores<KvHeads, Groups, Self::Fresh, Self>,
            &GroupMask<KvHeads, Groups, Head, Self>,
        ),
        source: &mut impl Source<KvHeads, Groups, Head, Self::Fresh>,
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
            &WeighBlock::<Narrow>(PhantomData),
            &mut *context,
            (
                |context: &mut Device, s| source.value(context, s),
                &probabilities,
            ),
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
        let query = queries
            .view()
            .tile::<m![{ Symbol<Self> }], 1, TileRows<Head, Self>>(0);
        let within = record.within();
        for r in 0..within.count() {
            let j = within.begin() + r;
            let key = Op::forward(
                &fresh::LoadKeys::<KvHeads, Groups, Head>(PhantomData),
                &mut *context,
                Cache::rows(input.key, input.start + j * 128),
            );
            let mut scores = Scores::<KvHeads, Groups>::new();
            Op::forward(
                &fresh::FreshScores,
                &mut *context,
                (input.attention, query.clone(), &key, scores.view_mut()),
            );
            let (probabilities, lse) = Op::forward(&panel::Normalize, &mut *context, scores.view());
            let value = Op::forward(
                &fresh::LoadValues::<KvHeads, Groups, Head>(PhantomData),
                &mut *context,
                Cache::rows(input.value, input.start + j * 128),
            );
            let partial = Op::forward(
                &panel::Weight,
                &mut *context,
                (probabilities.view(), &value),
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
type TileRowSlot<Head, T> = m![{ TileSlot<T> }, Rows % 8, Head];
type TileStatistic<T> = m![{ TileOf<T> }, Rows % 8, 1 # 8];
type TileStatisticSlot<T> = m![{ TileSlot<T> }, Rows % 8, 1 # 8];
type CachePanel<KvHeads, Head> = HbmTensor<bf16, Chip, m![Cache, KvHeads, Head]>;
type Cluster<KvHeads> = m![KvHeads / 4];
/// A slice's KV head of its cluster's four, its query group of four, and its eight rows of a tile;
/// fewer query groups leave the slices of the absent ones empty.
type Slice<KvHeads, Groups> = m![KvHeads % 4, Groups # 4, Rows / 8];
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
        // Pairs cover whole chunks with or without history: a pair stores its tiles after both
        // groups, so the second group's loads do not queue behind the first group's stores.
        const { assert!(All::SIZE >= TilePair::SIZE * Rows::SIZE) };
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
        // A chunk holds at most two groups, so the block before any group or remainder tile is the
        // last group's own fresh panels: their fill layouts carry over from it.
        const _: () = assert!(Query::SIZE <= 2 * TileGroup::SIZE * Rows::SIZE);
        let mut previous: Kept<KvHeads, Head, Half> = (DmTensor::new(), DmTensor::new());
        for r in 0..runs.records::<TileGroup>() {
            let mut own = (DmTensor::new(), DmTensor::new());
            Grouped::<TileGroup>(PhantomData).forward(
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
            Grouped::<SingleTile>(PhantomData).forward(
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

/// Attends `T` consecutive query tiles from the record's first: each tile's fresh keys, then every
/// history block and full panel once for all of them, then the partial last panel per tile.
struct Grouped<T>(PhantomData<T>);

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
    > for Grouped<T>
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
        let first = record.first();
        let mut source = Load {
            input,
            start: input.start + first * 128,
            kept: own,
            record,
            tile: 0,
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
        for _ in 0..record.preceding() {
            Module::forward(
                &Preceding::<Half, T>(PhantomData),
                &mut *context,
                env,
                &mut (),
                (
                    input.attention,
                    &mut Reuse::<KvHeads, Head, _>(previous),
                    &queries,
                    (&mut carry, &mut running),
                ),
            );
        }
        T::within::<All, W>(context, input, record, &queries, (&mut carry, &mut running));
        let double = runs.double();
        for b in 0..double.count() {
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
                &mut Reuse::<KvHeads, Head, _>(&kept),
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

/// A group's queries: an owner of its own, or its half of a pair's.
trait QueryTiles<KvHeads: AxisName, Groups: AxisName, Head: AxisName> {
    type Group: M;
    type From<R: AxisName>: M;
    fn group(
        &self,
    ) -> DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, Self::Group>;
    /// Tiles `s..s + N` as a run of `R`, `N` = `R::SIZE`.
    fn from<R: AxisName, const N: usize>(
        &self,
        s: usize,
    ) -> DmTensorView<'_, bf16, Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>, Self::From<R>>;
}
impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, T: AxisName>
    QueryTiles<KvHeads, Groups, Head> for history::Group<KvHeads, Groups, Head, T>
{
    type Group = m![{ Symbol<T> }, Rows % 8, Head];
    type From<R: AxisName> = m![{ Symbol<R> } # ({ Symbol<T> }), Rows % 8, Head];
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
}
impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, const G: usize>
    QueryTiles<KvHeads, Groups, Head> for Paired<&Halved<KvHeads, Groups, bf16, m![Head]>, G>
{
    type Group = HalfOf<m![Head]>;
    type From<R: AxisName> = m![{ Symbol<Halves> } = 1 # ({ Symbol<Halves> }),
        { Symbol<R> } # ({ Symbol<TileGroup> }), Rows % 8, Head];
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
        // Only the partial last panel, masked to the committed keys, is attended per tile; its key
        // and value fills and its mask serve every tile.
        let partial = runs.partial();
        for _ in 0..partial.count() {
            let key = Op::forward(
                &fresh::LoadKeys::<KvHeads, Groups, Head>(PhantomData),
                &mut *context,
                last.0.view(),
            );
            let mask = Op::forward(
                &history::LastKeys::<KvHeads, Groups>(PhantomData),
                &mut *context,
                (input.lanes, input.slots, partial.begin()),
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
                let (probabilities, lse) =
                    Op::forward(&panel::Normalize, &mut *context, scores.view());
                let partial = Op::forward(
                    &panel::Weight,
                    &mut *context,
                    (probabilities.view(), &value),
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
    S: Source<KvHeads, Groups, Head, P>,
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
            (
                attention,
                (key, |context: &mut Device, s| source.key(context, s)),
                queries,
                &mut scores,
            ),
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
            &WeighBlock::<P>(PhantomData),
            &mut *context,
            (
                |context: &mut Device, s| source.value(context, s),
                &probabilities,
            ),
        );
        Op::forward(
            &state::Absorb,
            context,
            ((carry, running), &partials, &total),
        );
    }
}

/// A block's panels in the layouts their register-file fills read, kept for another group.
type Kept<KvHeads, Head, P> = (
    DmTensor<
        bf16,
        Chip,
        Cluster<KvHeads>,
        fresh::Switched<KvHeads, Head>,
        m![{ Panels<P> }, Cache % 8, Head % 32],
    >,
    DmTensor<
        bf16,
        Chip,
        Cluster<KvHeads>,
        fresh::ValueSlice<KvHeads, Head>,
        m![{ Panels<P> }, Head % 16, Cache % 16],
    >,
);
type KeptPanel<P> = m![{ Symbol<<P as Panel>::Span> } = 1 # ({ Symbol<<P as Panel>::Span> })];
type KeptSlot<P> = m![{ Symbol<<P as Panel>::Span> } = 1 #{!} ({ Symbol<<P as Panel>::Span> })];

/// Where a fresh block's panel `s` comes from, as one register-file fill of keys or values.
trait Source<KvHeads: AxisName, Groups: AxisName, Head: AxisName, P: Panel> {
    fn key(&mut self, context: &mut Device, s: usize) -> fresh::Keys<KvHeads, Groups, Head>;
    fn value(&mut self, context: &mut Device, s: usize) -> fresh::Values<KvHeads, Groups, Head>;
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
    Source<KvHeads, Groups, Head, P> for Load<'_, '_, KvHeads, Groups, Head, All, W, P>
{
    fn key(&mut self, context: &mut Device, s: usize) -> fresh::Keys<KvHeads, Groups, Head> {
        let staged: DmTensor<
            bf16,
            Chip,
            Cluster<KvHeads>,
            fresh::KeyStage<KvHeads>,
            m![Cache % 2, Head],
        > = Cache::rows(self.input.key, self.start + s * 128).to_dm(&mut context.tdma);
        Op::forward(
            &fresh::SwitchKeys,
            &mut *context,
            (
                staged.view(),
                self.kept
                    .0
                    .view_mut()
                    .tile::<Panels<P>, 1, m![{ KeptSlot<P> }, Cache % 8, Head % 32]>(s),
            ),
        );
        Reuse::<KvHeads, Head, P>(&*self.kept).key(context, s)
    }
    fn value(&mut self, context: &mut Device, s: usize) -> fresh::Values<KvHeads, Groups, Head> {
        let values = Cache::rows(self.input.value, self.start + s * 128);
        let mut staged = fresh::Staged::<KvHeads, Head>::new();
        Op::forward(
            &fresh::StageValues,
            &mut *context,
            (values.clone(), staged.view_mut()),
        );
        // Rows past the chunk's last may hold any bits, and a key's zero probability times NaN is
        // NaN; only the chunk's last tile holds such rows.
        for _ in 0..self.record.cut(self.tile + s) {
            Op::forward(
                &fresh::Cut,
                &mut *context,
                (
                    (values.clone(), (self.input.causal, self.record.last())),
                    staged.view_mut(),
                ),
            );
        }
        Op::forward(
            &fresh::TransposeValues,
            &mut *context,
            (
                staged.view(),
                self.kept
                    .1
                    .view_mut()
                    .tile::<Panels<P>, 1, m![{ KeptSlot<P> }, Head % 16, Cache % 16]>(s),
            ),
        );
        Reuse::<KvHeads, Head, P>(&*self.kept).value(context, s)
    }
}

/// Panels another group's block already relaid, filled straight from their kept layouts.
struct Reuse<'a, KvHeads: AxisName, Head: AxisName, P: Panel>(&'a Kept<KvHeads, Head, P>);
impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, P: Panel> Source<KvHeads, Groups, Head, P>
    for Reuse<'_, KvHeads, Head, P>
{
    fn key(&mut self, context: &mut Device, s: usize) -> fresh::Keys<KvHeads, Groups, Head> {
        Op::forward(
            &fresh::RegisterKeys::<Groups>(PhantomData),
            context,
            self.0
                .0
                .view()
                .tile::<Panels<P>, 1, m![{ KeptPanel<P> }, Cache % 8, Head % 32]>(s),
        )
    }
    fn value(&mut self, context: &mut Device, s: usize) -> fresh::Values<KvHeads, Groups, Head> {
        Op::forward(
            &fresh::RegisterValues::<Groups>(PhantomData),
            context,
            self.0
                .1
                .view()
                .tile::<Panels<P>, 1, m![{ KeptPanel<P> }, Head % 16, Cache % 16]>(s),
        )
    }
}

/// The scaled scores of a block's `P::Span` panels for every tile of the group, into `scores`:
/// one fill, the first already made, and one contraction per panel.
struct ScoreBlock<P>(PhantomData<P>);

impl<
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    P: Panel,
    T: AxisName,
    F: FnMut(&mut Device, usize) -> fresh::Keys<KvHeads, Groups, Head>,
    Q: QueryTiles<KvHeads, Groups, Head>,
>
    Op<(
        &Attention<KvHeads, Groups, Head>,
        (fresh::Keys<KvHeads, Groups, Head>, F),
        &Q,
        &mut history::PanelScores<KvHeads, Groups, P, T>,
    )> for ScoreBlock<P>
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (attention, (first, mut fill), queries, scores): (
            &Attention<KvHeads, Groups, Head>,
            (fresh::Keys<KvHeads, Groups, Head>, F),
            &Q,
            &mut history::PanelScores<KvHeads, Groups, P, T>,
        ),
    ) {
        let mut key = first;
        #[unroll]
        for s in 1..P::Span::SIZE {
            let next = fill(context, s);
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
    }
}

/// Each tile's per-panel weighted values of a block's panels under the block's probabilities:
/// one value fill and one contraction per panel.
struct WeighBlock<P>(PhantomData<P>);

impl<
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    P: Panel,
    T: AxisName,
    F: FnMut(&mut Device, usize) -> fresh::Values<KvHeads, Groups, Head>,
> Op<(F, &history::Keyed<KvHeads, Groups, P, T>)> for WeighBlock<P>
{
    type Output = history::Partials<KvHeads, Groups, Head, P, T>;

    fn forward(
        &self,
        context: &mut Device,
        (mut fill, probabilities): (F, &history::Keyed<KvHeads, Groups, P, T>),
    ) -> Self::Output {
        // SAFETY: the inverse relabel of the key rows.
        let probabilities = unsafe {
            probabilities.view().reshape::<Chip, Cluster<KvHeads>, Slice<KvHeads, Groups>,
                m![{ Symbol<T> }, Rows % 8, { Panels<P> }, Cache]>()
        };
        let mut partials = history::Partials::<KvHeads, Groups, Head, P, T>::new();
        let mut value = fill(context, 0);
        #[unroll]
        for s in 1..P::Span::SIZE {
            let next = fill(context, s);
            Op::forward(
                &history::Weigh::<P>(PhantomData),
                &mut *context,
                (&probabilities, &value, &mut partials, s - 1),
            );
            value = next;
        }
        Op::forward(
            &history::Weigh::<P>(PhantomData),
            &mut *context,
            (&probabilities, &value, &mut partials, P::Span::SIZE - 1),
        );
        partials
    }
}

#[cfg(test)]
mod tests {
    use super::super::chunks::{self, Words};
    use super::*;
    use crate::tensor::axis::Rows;
    use crate::tensor::cache::Blocks;
    use crate::tensor::live::tests::count;
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;
    axes![KvHeads = 8, Groups = 4, Pairs = 2, Head = 128, Windows = 80];
    type FixtureRows = Rows<1024>;
    type Sequences = Rows<1>;
    type Queries<G> = HbmTensor<bf16, Chip, m![Query, KvHeads, G, Head]>;
    type Fresh = HbmTensor<bf16, Chip, m![FixtureRows, KvHeads, Head]>;
    type Output<G> = HbmTensor<bf16, Chip, m![FixtureRows, KvHeads, G, Head]>;

    #[derive(DeviceSend)]
    struct Chunk<G: AxisName> {
        query: Queries<G>,
        key: Fresh,
        value: Fresh,
        causal: Causal,
        table: Table<Windows>,
        h: Value,
        work: HbmTensor<i32, Chip, m![{ Symbol<Words<Sequences, Windows>> }]>,
        lanes: HbmTensor<i32, Chip, m![Cache]>,
        slots: SlotTable<Windows>,
    }

    #[device(chip = 1)]
    fn attend_chunk<G: AxisName>(
        device: &mut Device,
        chunk: &Chunk<G>,
        blocks: &mut Blocks<KvHeads, Head>,
        output: &mut Output<G>,
    ) {
        // SAFETY: fixture query rows are contiguous and share the same length.
        let query = unsafe {
            chunk
                .query
                .view()
                .reshape::<Chip, m![FixtureRows, KvHeads, G, Head]>()
                .alias()
        };
        // A local: device code cannot read the constant that `&Attention {..}` is promoted to.
        let attention = Attention {
            heads: PhantomData,
            scale: 0.08837890625f32,
        };
        Attend.forward(
            device,
            (),
            &mut (),
            (
                (
                    Input::<KvHeads, G, Head, FixtureRows, Windows> {
                        attention: &attention,
                        query: &query,
                        key: &chunk.key,
                        value: &chunk.value,
                        start: 0,
                        causal: &chunk.causal,
                        table: &chunk.table,
                        history: &chunk.h,
                        lanes: &chunk.lanes,
                        slots: &chunk.slots,
                    },
                    &Epoch::new(&chunk.work, 0),
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

    /// The attended rows of a chunk of `q` rows of `G` query groups a KV head over `h` keys of a
    /// poisoned, shuffled pool; scratch rows past the chunk's hold `past`, or else a NaN query and
    /// the next key's key and value, and live row `nan` holds a NaN value.
    async fn attend<G: AxisName>(
        context: &mut Device,
        (q, h): (usize, usize),
        (past, nan): (Option<f32>, Option<usize>),
    ) -> Vec<bf16> {
        let scratch = |row: usize, own: f32, next: f32| {
            bf16::from_f32(if row < q { own } else { past.unwrap_or(next) })
        };
        let blocks = 256;
        let physical = |logical: usize| (logical * 17 + 7) % blocks;
        let mut allocation = Blocks::<KvHeads, Head> {
            tokens: (0..blocks)
                .map(|_| HbmTensor::alloc(&mut *context))
                .collect::<Result<_, _>>()
                .unwrap(),
        };
        let mut data: Vec<bf16> = (0..blocks * 16 * 2 * 8 * 128)
            .map(|at| bf16::from_f32([f32::NAN, f32::INFINITY, f32::NEG_INFINITY][at % 3]))
            .collect();
        for k in 0..q + h {
            let token = physical(k / 16) * 16 + k % 16;
            for head in 0..8 {
                for channel in 0..128 {
                    let at = (token * 2 * 8 + head) * 128 + channel;
                    data[at] = bf16::from_f32(kv(k, head));
                    data[at + 8 * 128] = bf16::from_f32(value(k, head) * scale(channel));
                }
            }
        }
        allocation
            .write(&mut *context, 0..blocks, data.iter().copied())
            .await
            .unwrap();
        let table: Table<Windows> =
            HostTensor::<i32, m![{ Symbol<Chunks<Windows>> }, Cache / Block]>::from_vec(
                (0..Chunks::<Windows>::SIZE * 8)
                    .map(|i| i32::try_from(physical(i % blocks) * 16 * 2 * 8 * 128 * 2).unwrap()),
            )
            .to_hbm(&mut context.pdma)
            .await
            .expect("host transfer");
        let slots = HostTensor::<i32, m![{ Symbol<Chunks<Windows>> }, Cache]>::from_vec(0..40960)
            .to_hbm(&mut context.pdma)
            .await
            .expect("host transfer");
        let groups = G::SIZE;
        let query: Queries<G> = HostTensor::<bf16, m![Query, KvHeads, G, Head]>::from_vec(
            (0..1024 * 8 * groups * 128).map(|at| {
                let j = at / (8 * groups * 128);
                let own = query_value(j, at / (groups * 128) % 8, at / 128 % groups);
                scratch(j, own, f32::NAN)
            }),
        )
        .to_hbm(&mut context.pdma)
        .await
        .expect("host transfer");
        let lanes: HbmTensor<i32, Chip, m![Cache]> =
            HostTensor::<i32, m![Cache]>::from_vec([i32::try_from(h).unwrap(); 128])
                .to_hbm(&mut context.pdma)
                .await
                .expect("host transfer");
        let mut output: Output<G> =
            HostTensor::<bf16, m![FixtureRows, KvHeads, G, Head]>::from_vec(vec![
                bf16::from_f32(
                    f32::INFINITY
                );
                1024 * 8
                    * groups
                    * 128
            ])
            .to_hbm(&mut context.pdma)
            .await
            .expect("host transfer");
        let fresh: HbmTensor<bf16, Chip, m![FixtureRows, KvHeads, Head]> =
            HostTensor::<bf16, m![FixtureRows, KvHeads, Head]>::from_vec((0..1024 * 8 * 128).map(
                |at| {
                    let (row, head) = (at / (8 * 128), at / 128 % 8);
                    scratch(row, kv(h + row, head), kv(h + row, head))
                },
            ))
            .to_hbm(&mut context.pdma)
            .await
            .expect("host transfer");
        let values: HbmTensor<bf16, Chip, m![FixtureRows, KvHeads, Head]> =
            HostTensor::<bf16, m![FixtureRows, KvHeads, Head]>::from_vec((0..1024 * 8 * 128).map(
                |at| {
                    let (row, head) = (at / (8 * 128), at / 128 % 8);
                    let own = match nan == Some(row) {
                        true => f32::NAN,
                        false => value(h + row, head) * scale(at % 128),
                    };
                    scratch(row, own, own)
                },
            ))
            .to_hbm(&mut context.pdma)
            .await
            .expect("host transfer");
        let causal: Causal = HostTensor::<i8, m![Cache, Cache]>::from_vec(
            (0..128 * 128).map(|at| i8::from(at % 128 > at / 128)),
        )
        .to_hbm(&mut context.pdma)
        .await
        .expect("host transfer");
        let mut plan = chunks::Plan::<Windows>::new::<Sequences>();
        (plan.count, plan.q[0], plan.h[0]) = (1, q as i32, h as i32);
        let [.., (work, _)] = plan.reads().unwrap();
        let chunk = Chunk {
            query,
            key: fresh,
            value: values,
            causal,
            table,
            h: count(h)
                .to_hbm(&mut context.pdma)
                .await
                .expect("host transfer"),
            work: HostTensor::<i32, m![{ Symbol<Words<Sequences, Windows>> }]>::from_vec(work)
                .to_hbm(&mut context.pdma)
                .await
                .expect("host transfer"),
            lanes,
            slots,
        };
        launch(
            attend_chunk,
            (&mut *context, &chunk, &mut allocation, &mut output),
        )
        .await
        .expect("device launch");
        output
            .to_host::<m![FixtureRows, KvHeads, G, Head]>(&mut context.pdma)
            .await
            .expect("host transfer")
            .into_vec()
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
        let row = 8 * G::SIZE * 128;
        for (q, h) in cases {
            let zeroed = attend::<G>(&mut *context, (q, h), (Some(0.0), None)).await;
            let runs = [
                (f32::NAN, None),
                (f32::INFINITY, None),
                (f32::NAN, Some(q - 1)),
            ];
            for (past, nan) in runs {
                let got = attend::<G>(&mut *context, (q, h), (Some(past), nan)).await;
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
            let got = attend::<G>(&mut *context, (q, h), (None, None)).await;
            let mut max_error = 0.0f32;
            for j in 0..q {
                for head in 0..8 {
                    for group in 0..G::SIZE {
                        let scores: Vec<f32> = (0..=h + j)
                            .map(|k| {
                                query_value(j, head, group) * kv(k, head) * 128.0 / 128.0f32.sqrt()
                            })
                            .collect();
                        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let mut sum = 0.0f32;
                        let mut weighted = 0.0f32;
                        for (k, score) in scores.into_iter().enumerate() {
                            let weight = (score - max).exp();
                            sum += weight;
                            weighted += weight * value(k, head);
                        }
                        for channel in 0..128 {
                            let expected = weighted / sum * scale(channel);
                            let at = ((j * 8 + head) * G::SIZE + group) * 128 + channel;
                            let actual = got[at].to_f32();
                            assert!(
                                actual.is_finite(),
                                "{} groups q={q} h={h} row={j} head={head} group={group}",
                                G::SIZE
                            );
                            let error = (actual - expected).abs();
                            max_error = max_error.max(error);
                            assert!(
                                error <= 0.004,
                                "{} groups q={q} h={h} row={j}: {actual} != {expected}",
                                G::SIZE
                            );
                        }
                    }
                }
            }
            println!("{} groups q={q} h={h} max_error={max_error}", G::SIZE);
        }
    }
}
