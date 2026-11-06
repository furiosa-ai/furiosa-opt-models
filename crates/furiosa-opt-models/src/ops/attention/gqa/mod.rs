use crate::Op;
use crate::tensor::{axis::Lanes, live::Live, mapping::Rows as Layout};
use furiosa_opt_std::prelude::*;

use super::{Attention, Chip, KeyValue, TileGroups};
use crate::ops::index::table::{BlockCounts, BlockIndex};
use crate::specialize::attention::{self, Strategy};
use crate::tensor::cache::{self, Block, Blocks};
use crate::tensor::scalar::{F32, Value};
use std::marker::PhantomData;

use crate::ops::linear;
pub use paged::{Counted, Pages, Read, Wrap, Wrapped, Wrapping};

mod paged;
mod partition;
pub(crate) mod shard;

axes![
    Cache = 512, // tokens one attention window holds; histories and block tables tile by it
];

/// The shards of `Count` row groups, every group of the envelope unless one is named.
pub(crate) type Partition<
    Rows,
    KvHeads,
    Groups,
    Head,
    const SCALE: usize,
    Count = RowGroups<Rows>,
> = partition::Geometry<
    Chip,
    Cluster<Rows>,
    CacheSlices<Rows, KvHeads>,
    Agreed<Rows, KvHeads>,
    Merged<Rows, KvHeads>,
    Count,
    m![Groups],
    m![Head],
    ShardKeys<KvHeads>,
    Cache,
    SCALE,
>;

/// The shards each head's window splits into, one a slice: as many as a cluster's slices hold
/// beside the heads and their four row lanes, each at least one cache block.
#[derive(Clone, Debug)]
pub struct Shards<KvHeads>(PhantomData<KvHeads>);

impl<KvHeads: AxisName> AxisName for Shards<KvHeads> {
    const NAME: Ident = Ident::new("Shards");
    const SIZE: usize = attention::shards(KvHeads::SIZE, Cache::SIZE, Block::SIZE);
}

/// The keys of a window one shard holds.
#[derive(Clone, Debug)]
pub struct Keys<KvHeads>(PhantomData<KvHeads>);

impl<KvHeads: AxisName> AxisName for Keys<KvHeads> {
    const NAME: Ident = Ident::new("Keys");
    const SIZE: usize = Cache::SIZE / Shards::<KvHeads>::SIZE;
}

/// The consecutive blocks of a row's table one shard holds.
#[derive(Clone, Debug)]
pub struct ShardBlocks<KvHeads>(PhantomData<KvHeads>);

impl<KvHeads: AxisName> AxisName for ShardBlocks<KvHeads> {
    const NAME: Ident = Ident::new("ShardBlocks");
    const SIZE: usize = Keys::<KvHeads>::SIZE / Block::SIZE;
}

/// The heads a cluster's slices hold at a head's shards of four row lanes; the slots past a
/// layer's heads stay empty.
#[derive(Clone, Debug)]
pub struct HeadSlots<KvHeads>(PhantomData<KvHeads>);

impl<KvHeads: AxisName> AxisName for HeadSlots<KvHeads> {
    const NAME: Ident = Ident::new("HeadSlots");
    const SIZE: usize = attention::heads(Shards::<KvHeads>::SIZE);
}

/// Two blocks' pairs a shard holds, or the shards' pairs where a shard holds one block: the units
/// of a window's head, its first two blocks.
#[derive(Clone, Debug)]
pub struct ShardPairs<KvHeads>(PhantomData<KvHeads>);

impl<KvHeads: AxisName> AxisName for ShardPairs<KvHeads> {
    const NAME: Ident = Ident::new("ShardPairs");
    const SIZE: usize = if ShardBlocks::<KvHeads>::SIZE >= 2 {
        ShardBlocks::<KvHeads>::SIZE / 2
    } else {
        Shards::<KvHeads>::SIZE / 2
    };
}

type ShardKeys<KvHeads> = m![Cache % ({ Symbol<Keys<KvHeads>> })];

/// The shards of the row groups of one `N`-row tile, which one pass attends.
type TileShards<Rows, KvHeads, Groups, Head, const SCALE: usize, const N: usize> =
    Partition<Rows, KvHeads, Groups, Head, SCALE, TileRows<Rows, N>>;

/// Windows a history may span after the first, which the bounded loop carries: the peeled first
/// window is why the loop's envelope is one short of a history's.
#[derive(Clone, Debug)]
pub(crate) struct Rest<Windows>(PhantomData<Windows>);

impl<Windows: AxisName> AxisName for Rest<Windows> {
    const NAME: Ident = Ident::new("Rest");
    const SIZE: usize = Windows::SIZE - 1;
}

/// Four lanes per cluster fill the chip with eight rows; fewer rows leave padded lanes.
pub(super) type Cluster<Rows> = m![{ Layout<Rows>::Lanes } # 8 / 4];

/// A head's shards are consecutive slices so the inter-slice reducer can combine them; each of
/// its four row lanes holds shards of its own.
type CacheSlices<Rows, KvHeads> = m![
    KvHeads # ({ Symbol<HeadSlots<KvHeads>> }),
    { Layout<Rows>::Lanes } # 8 % 4,
    Cache / ({ Symbol<Keys<KvHeads>> })
];

/// All shard slices of a head hold a copy of the same reduced statistic.
type Agreed<Rows, KvHeads> = m![
    KvHeads # ({ Symbol<HeadSlots<KvHeads>> }),
    { Layout<Rows>::Lanes } # 8 % 4,
    { Symbol<Shards<KvHeads>> }
];

/// One slice per row and head holds the merged attention, `Heads` naming the head.
pub(super) type Merged<Rows, KvHeads, Heads = m![KvHeads]> = m![
    { Heads } # ({ Symbol<HeadSlots<KvHeads>> }),
    { Layout<Rows>::Lanes } # 8 % 4,
    1 # ({ Symbol<Shards<KvHeads>> })
];

type RowGroups<Rows> = linear::Groups<Rows>;

/// The row groups of an `N`-row tile; at 32 rows the four independent row groups whose 128-wide
/// keys fill the 64 KiB register file, the most one pass attends.
type TileRows<Rows, const N: usize = 32> =
    m![{ RowGroups<Rows> } % ({ Symbol<TileGroups<Rows, N>> })];

/// The `N`-row tile a row group belongs to.
type Tiles<Rows, const N: usize = 32> = m![{ RowGroups<Rows> } / ({ Symbol<TileGroups<Rows, N>> })];

/// Pass `tile` over an envelope's `N`-row tiles, `rows` of whose rows are live: the row groups of
/// that tile, attended together.
#[derive(Clone, Copy)]
pub struct Pass<const N: usize> {
    tile: usize,
    rows: usize,
}

impl<const N: usize> Pass<N> {
    /// Tile `tile` of the envelope whose rows `live` are, which reach into it.
    fn of<R: AxisName>(tile: usize, live: Live<(), R>) -> Self {
        Self {
            tile,
            rows: (live.rows() - tile * N).min(N),
        }
    }
}

/// Every window of every row's history, for rows that carry it outright; in DRAM, since a slice's
/// sixteen-chunk data memory would otherwise bound the windows a history spans.
pub(super) type Envelope<Rows, KvHeads, Head, Windows> =
    HbmTensor<bf16, Chip, m![Rows, Windows, Cache, KeyValue, KvHeads, Head]>;

/// One chunk as the gather writes it, needing no repartition: each table entry's token block on its
/// shard and its heads over their slices, the cells of [`ChunkView`], each a `D` the pool holds.
type InDm<Rows, KvHeads, Head, Count = RowGroups<Rows>, D = bf16> = DmTensor<
    D,
    Chip,
    Cluster<Rows>,
    m![
        KvHeads # ({ Symbol<HeadSlots<KvHeads>> }),
        { Layout<Rows>::Lanes } # 8 % 4,
        BlockIndex / ({ Symbol<ShardBlocks<KvHeads>> })
    ],
    m![{ Count }, KeyValue, BlockIndex % ({ Symbol<ShardBlocks<KvHeads>> }), Block, Head],
>;

/// One chunk's entries of a step's table, the only shape a gather takes an index in: a window's
/// tile of the table, whose rows lie together.
type Entries<Rows> = HbmTensor<i32, Chip, m![Rows, BlockIndex]>;

/// One chunk's entries of the row groups of one `N`-row tile, their lanes in order.
type TileEntries<Rows, const N: usize> =
    HbmTensor<i32, Chip, m![{ TileRows<Rows, N> }, { Layout<Rows>::Lanes }, BlockIndex]>;

/// A layer's cache blocks of `D`, the table of blocks each row reads by window, the scales its
/// codes are read under (nothing for BF16 blocks, the layer's [`cache::Landed`] scales for E4M3
/// ones), and the tokens each row holds past its last window: a [`Wrap`] with the times it lands,
/// one if any row holds one, else none, or nothing for histories that hold none.
pub struct Paged<
    'l,
    Rows: AxisName,
    KvHeads: AxisName,
    Head: AxisName,
    Windows: AxisName,
    D: MaterializableScalar = bf16,
    S = (),
    W = (),
> {
    pub blocks: &'l Blocks<KvHeads, Head, D>,
    pub table: &'l HbmTensor<i32, Chip, m![Windows, Rows, BlockIndex]>,
    pub lengths: &'l BlockCounts<Rows, Windows>,
    pub scales: &'l S,
    pub wrap: W,
}

/// The tokens of the rows' histories past their last window, landed into the head of their first
/// window's cells: the slots before each row's lowest visible token, which its oldest blocks hold.
#[derive(Clone, Copy)]
pub(crate) struct Wraps;

/// The cache rows a step reads, a shard's whole rows of one head per slice, each row its token's
/// key and value; `Count` row groups of them, as attention reads a landing's cells.
type ChunkView<'a, Rows, KvHeads, Head, Count, D = bf16> = DmTensorView<
    'a,
    D,
    Chip,
    Cluster<Rows>,
    CacheSlices<Rows, KvHeads>,
    m![{ Count }, KeyValue, { ShardKeys<KvHeads> }, Head],
>;

/// The same cells as a landing writes a copy of a chunk into them.
type ChunkViewMut<'a, Rows, KvHeads, Head, Count, D = bf16> = DmTensorViewMut<
    'a,
    D,
    Chip,
    Cluster<Rows>,
    CacheSlices<Rows, KvHeads>,
    m![{ Count }, KeyValue, { ShardKeys<KvHeads> }, Head],
>;

/// One window as it lands, its cells of `D` and its bias: two alternate, so a window lands while
/// the one before it is attended.
struct Landing<
    Rows: AxisName,
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    Count: M,
    D: Scalar = bf16,
> {
    cells: InDm<Rows, KvHeads, Head, Count, D>,
    bias: Bias<Rows, KvHeads, Groups, Count>,
}

impl<Rows: AxisName, KvHeads: AxisName, Groups: AxisName, Head: AxisName, Count: M, D: Scalar>
    Landing<Rows, KvHeads, Groups, Head, Count, D>
{
    fn new() -> Self {
        Self {
            cells: DmTensor::new(),
            bias: DmTensor::new(),
        }
    }
}

/// The cells of `cells` as attention reads a chunk.
fn chunk<Rows: AxisName, KvHeads: AxisName, Head: AxisName, Count: M, D: Scalar>(
    cells: &InDm<Rows, KvHeads, Head, Count, D>,
) -> ChunkView<'_, Rows, KvHeads, Head, Count, D> {
    // SAFETY: a shard's blocks are its consecutive slots, so both nests name the same cells in the
    // same order.
    unsafe { cells.view().reshape() }
}

/// A history as its windows land and as attention reads them: BF16 cells as they landed, or a
/// pool's E4M3 codes, whose keys widen to BF16 once a window is prepared (which leaves the widening
/// behind no gather in flight), whose values widen in their transpose or, where a window streams
/// them, once it is scored, and whose scales the scores and the weighted sum take.
pub(crate) trait Landed<Rows: AxisName, KvHeads: AxisName, Head: AxisName> {
    /// The element a window lands in.
    type Element: shard::Value;
    /// Where a window's keys, and its values where it streams them, of `Count` row groups widen to
    /// BF16: nothing for BF16 cells.
    type Restored<Count: M>;

    /// A place to widen a window's keys and values of `Count` row groups into.
    fn place<Count: M>() -> Self::Restored<Count>;

    /// The window landed in `cells` as the key fill reads it, its keys widened into `place`
    /// first where they are codes; only its keys' half is read.
    fn keys<'c, Count: M>(
        &self,
        device: &mut Device,
        cells: &'c InDm<Rows, KvHeads, Head, Count, Self::Element>,
        place: &'c mut Self::Restored<Count>,
    ) -> ChunkView<'c, Rows, KvHeads, Head, Count>;

    /// The window landed in `cells` as its weighted sum streams it, its values widened into `place`
    /// first where they are codes; only its values' half is read.
    fn values<'c, Count: M>(
        &self,
        device: &mut Device,
        cells: &'c InDm<Rows, KvHeads, Head, Count, Self::Element>,
        place: &'c mut Self::Restored<Count>,
    ) -> ChunkView<'c, Rows, KvHeads, Head, Count>;

    /// The same, once [`Self::keys`] widened them.
    fn widened<'c, Count: M>(
        cells: &'c InDm<Rows, KvHeads, Head, Count, Self::Element>,
        place: &'c Self::Restored<Count>,
    ) -> ChunkView<'c, Rows, KvHeads, Head, Count>;

    /// The scales of a window's codes in the registers of partition `G`, none for BF16 cells.
    fn scales<G>(&self, device: &mut Device) -> Option<shard::Scales<G>>
    where
        G: shard::Partition<
                Chip = Chip,
                Cluster = Cluster<Rows>,
                Slices = CacheSlices<Rows, KvHeads>,
            >;
}

/// The cells a window of `Count` row groups of `Source`'s history lands in.
type Cells<Source, Rows, KvHeads, Head, Count = RowGroups<Rows>> =
    InDm<Rows, KvHeads, Head, Count, <Source as Landed<Rows, KvHeads, Head>>::Element>;

/// A history in BF16 is read as it landed.
macro_rules! landed_as_is {
    () => {
        type Element = bf16;
        type Restored<Count: M> = ();

        fn place<Count: M>() {}

        fn keys<'c, Count: M>(
            &self,
            _: &mut Device,
            cells: &'c InDm<Rows, KvHeads, Head, Count>,
            _: &'c mut (),
        ) -> ChunkView<'c, Rows, KvHeads, Head, Count> {
            chunk(cells)
        }

        fn values<'c, Count: M>(
            &self,
            _: &mut Device,
            cells: &'c InDm<Rows, KvHeads, Head, Count>,
            _: &'c mut (),
        ) -> ChunkView<'c, Rows, KvHeads, Head, Count> {
            chunk(cells)
        }

        fn widened<'c, Count: M>(
            cells: &'c InDm<Rows, KvHeads, Head, Count>,
            _: &'c (),
        ) -> ChunkView<'c, Rows, KvHeads, Head, Count> {
            chunk(cells)
        }

        fn scales<G>(&self, _: &mut Device) -> Option<shard::Scales<G>>
        where
            G: shard::Partition<
                    Chip = Chip,
                    Cluster = Cluster<Rows>,
                    Slices = CacheSlices<Rows, KvHeads>,
                >,
        {
            None
        }
    };
}

impl<Rows: AxisName, KvHeads: AxisName, Head: AxisName, Windows: AxisName>
    Landed<Rows, KvHeads, Head> for Envelope<Rows, KvHeads, Head, Windows>
{
    landed_as_is!();
}

impl<Rows: AxisName, KvHeads: AxisName, Head: AxisName, Windows: AxisName, W>
    Landed<Rows, KvHeads, Head> for Paged<'_, Rows, KvHeads, Head, Windows, bf16, (), W>
{
    landed_as_is!();
}

impl<Rows: AxisName, KvHeads: AxisName, Head: AxisName, Windows: AxisName, W>
    Landed<Rows, KvHeads, Head>
    for Paged<'_, Rows, KvHeads, Head, Windows, f8e4m3, cache::Landed, W>
{
    type Element = f8e4m3;
    type Restored<Count: M> = InDm<Rows, KvHeads, Head, Count>;

    fn place<Count: M>() -> InDm<Rows, KvHeads, Head, Count> {
        DmTensor::new()
    }

    /// Each key code widened to BF16, exactly, in one pass without the vector stage.
    fn keys<'c, Count: M>(
        &self,
        device: &mut Device,
        cells: &'c InDm<Rows, KvHeads, Head, Count, f8e4m3>,
        place: &'c mut InDm<Rows, KvHeads, Head, Count>,
    ) -> ChunkView<'c, Rows, KvHeads, Head, Count> {
        widen(device, cells, place, 0)
    }

    /// Each value code widened to BF16 as [`Self::keys`] widens the keys.
    fn values<'c, Count: M>(
        &self,
        device: &mut Device,
        cells: &'c InDm<Rows, KvHeads, Head, Count, f8e4m3>,
        place: &'c mut InDm<Rows, KvHeads, Head, Count>,
    ) -> ChunkView<'c, Rows, KvHeads, Head, Count> {
        widen(device, cells, place, 1)
    }

    fn widened<'c, Count: M>(
        _: &'c InDm<Rows, KvHeads, Head, Count, f8e4m3>,
        place: &'c InDm<Rows, KvHeads, Head, Count>,
    ) -> ChunkView<'c, Rows, KvHeads, Head, Count> {
        chunk(place)
    }

    fn scales<G>(&self, device: &mut Device) -> Option<shard::Scales<G>>
    where
        G: shard::Partition<
                Chip = Chip,
                Cluster = Cluster<Rows>,
                Slices = CacheSlices<Rows, KvHeads>,
            >,
    {
        let (key, value) = (self.scales).registers(device);
        Some(shard::Scales(key, value))
    }
}

/// Window `at` of every row, into the cells of a chunk.
impl<Rows: AxisName, KvHeads: AxisName, Head: AxisName, Windows: AxisName, E>
    crate::Module<(usize, Live<(), Rows>), E, &mut InDm<Rows, KvHeads, Head>>
    for Envelope<Rows, KvHeads, Head, Windows>
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        _: E,
        cells: &mut InDm<Rows, KvHeads, Head>,
        (at, _): (usize, Live<(), Rows>),
    ) {
        // SAFETY: as in `Landing::chunk`.
        let chunk: ChunkViewMut<'_, Rows, KvHeads, Head, RowGroups<Rows>> =
            unsafe { cells.view_mut().reshape() };
        self.view()
            .tile::<m![Windows], 1, m![Rows, Windows = 1 # Windows, Cache, KeyValue, KvHeads, Head]>(at)
            .to_dm_view(&mut device.tdma, chunk);
    }
}

/// A history carried outright holds every window whole: nothing wraps.
impl<Rows: AxisName, KvHeads: AxisName, Head: AxisName, Windows: AxisName, E, X, Count: M>
    crate::Module<(Wraps, X), E, &mut InDm<Rows, KvHeads, Head, Count>>
    for Envelope<Rows, KvHeads, Head, Windows>
{
    type Output = ();

    fn forward(
        &self,
        _: &mut Device,
        _: E,
        _: &mut InDm<Rows, KvHeads, Head, Count>,
        _: (Wraps, X),
    ) {
    }
}

/// Window `at` of the row groups of `pass` alone.
impl<Rows: AxisName, KvHeads: AxisName, Head: AxisName, Windows: AxisName, E, const N: usize>
    crate::Module<(usize, Pass<N>), E, &mut InDm<Rows, KvHeads, Head, TileRows<Rows, N>>>
    for Envelope<Rows, KvHeads, Head, Windows>
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        _: E,
        cells: &mut InDm<Rows, KvHeads, Head, TileRows<Rows, N>>,
        (at, pass): (usize, Pass<N>),
    ) {
        // SAFETY: as in `Landing::chunk`.
        let chunk: ChunkViewMut<'_, Rows, KvHeads, Head, TileRows<Rows, N>> =
            unsafe { cells.view_mut().reshape() };
        self.view()
            .tile::<m![Windows], 1, m![Rows, Windows = 1 # Windows, Cache, KeyValue, KvHeads, Head]>(at)
            .tile::<Tiles<Rows, N>, 1, m![
                { Tiles<Rows, N> } = 1 # ({ Tiles<Rows, N> }), { TileRows<Rows, N> },
                { Layout<Rows>::Lanes }, Windows = 1 # Windows, Cache, KeyValue, KvHeads, Head
            ]>(pass.tile)
            .to_dm_view(&mut device.tdma, chunk);
    }
}

impl<
    Rows: AxisName,
    KvHeads: AxisName,
    Head: AxisName,
    Windows: AxisName,
    D: MaterializableScalar,
    R,
    W,
> Paged<'_, Rows, KvHeads, Head, Windows, D, R, W>
{
    /// Gathers the pool blocks `entries` name into `chunk`, each row's up to its count in
    /// `lengths`: `SHARED`, in one plan up to the longest count, for a tile whose rows are all live;
    /// else each row its own, so a row past the live ones reads none. Cells past a count stay
    /// unspecified, for the bias to mask.
    fn land<E: M, L: M, Count: M, const SHARED: bool>(
        &self,
        device: &mut Device,
        (entries, lengths): (&HbmTensor<i32, Chip, E>, &HbmTensor<i32, Chip, L>),
        chunk: &mut InDm<Rows, KvHeads, Head, Count, D>,
    ) {
        // The plan is one expression a branch: one bound to a name loses its slice table.
        if const { SHARED } {
            self.blocks
                .tokens
                .gather::<m![1], m![Block, KeyValue, KvHeads, Head]>()
                .by_byte_offsets(entries)
                .prefix::<m![BlockIndex], _>(lengths, 0)
                .overfetch()
                .to_dm_view(&mut device.tdma, chunk.view_mut());
        } else {
            self.blocks
                .tokens
                .gather::<m![1], m![Block, KeyValue, KvHeads, Head]>()
                .by_byte_offsets(entries)
                .prefix::<m![BlockIndex], _>(lengths, 0)
                .to_dm_view(&mut device.tdma, chunk.view_mut());
        }
    }
}

/// Window `at` of every row, the live ones `live`, into the cells of a chunk.
impl<
    Rows: AxisName,
    KvHeads: AxisName,
    Head: AxisName,
    Windows: AxisName,
    D: MaterializableScalar,
    R,
    W,
    E,
> crate::Module<(usize, Live<(), Rows>), E, &mut InDm<Rows, KvHeads, Head, RowGroups<Rows>, D>>
    for Paged<'_, Rows, KvHeads, Head, Windows, D, R, W>
{
    type Output = ();

    /// A gather takes its index and counts as whole tensors, which a window's entries and counts
    /// are where they lie.
    fn forward(
        &self,
        device: &mut Device,
        _: E,
        cells: &mut InDm<Rows, KvHeads, Head, RowGroups<Rows>, D>,
        (at, live): (usize, Live<(), Rows>),
    ) {
        // SAFETY: a window's rows are consecutive cells of the table and of the counts, which
        // nothing writes while a step attends over them.
        let (entries, lengths): (Entries<Rows>, HbmTensor<i32, Chip, m![Rows]>) = unsafe {
            (
                (self.table.view())
                    .tile::<m![Windows], 1, m![Windows = 1 # Windows, Rows, BlockIndex]>(at)
                    .alias(),
                (self.lengths.view())
                    .tile::<m![Windows], 1, m![Windows = 1 # Windows, Rows]>(at)
                    .alias(),
            )
        };
        let whole = live.rows() / Rows::SIZE;
        for _ in 0..whole {
            self.land::<_, _, _, true>(device, (&entries, &lengths), cells);
        }
        for _ in 0..1 - whole {
            self.land::<_, _, _, false>(device, (&entries, &lengths), cells);
        }
    }
}

/// Window `at` of the row groups of `pass` alone, into the cells of a chunk: only their entries
/// address the pool.
impl<
    Rows: AxisName,
    KvHeads: AxisName,
    Head: AxisName,
    Windows: AxisName,
    D: MaterializableScalar,
    R,
    W,
    E,
    const N: usize,
> crate::Module<(usize, Pass<N>), E, &mut InDm<Rows, KvHeads, Head, TileRows<Rows, N>, D>>
    for Paged<'_, Rows, KvHeads, Head, Windows, D, R, W>
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        _: E,
        cells: &mut InDm<Rows, KvHeads, Head, TileRows<Rows, N>, D>,
        (at, pass): (usize, Pass<N>),
    ) {
        type Lengths<Rows, const N: usize> =
            HbmTensor<i32, Chip, m![{ TileRows<Rows, N> }, { Layout<Rows>::Lanes }]>;
        // SAFETY: as over every row; a tile's rows are consecutive rows of the window.
        let (entries, lengths): (TileEntries<Rows, N>, Lengths<Rows, N>) = unsafe {
            (
                (self.table.view())
                    .tile::<m![Windows], 1, m![Windows = 1 # Windows, Rows, BlockIndex]>(at)
                    .tile::<Tiles<Rows, N>, 1, m![
                        Windows = 1 # Windows, { Tiles<Rows, N> } = 1 # ({ Tiles<Rows, N> }),
                        { TileRows<Rows, N> }, { Layout<Rows>::Lanes }, BlockIndex
                    ]>(pass.tile)
                    .alias(),
                (self.lengths.view())
                    .tile::<m![Windows], 1, m![Windows = 1 # Windows, Rows]>(at)
                    .tile::<Tiles<Rows, N>, 1, m![
                        Windows = 1 # Windows, { Tiles<Rows, N> } = 1 # ({ Tiles<Rows, N> }),
                        { TileRows<Rows, N> }, { Layout<Rows>::Lanes }
                    ]>(pass.tile)
                    .alias(),
            )
        };
        // An envelope of fewer rows than the tile never fills it.
        if const { Rows::SIZE >= N } {
            for _ in 0..pass.rows / N {
                self.land::<_, _, _, true>(device, (&entries, &lengths), cells);
            }
        }
        for _ in 0..1 - pass.rows / N {
            self.land::<_, _, _, false>(device, (&entries, &lengths), cells);
        }
    }
}

/// Histories that hold no tokens past their last window land none.
impl<
    Rows: AxisName,
    KvHeads: AxisName,
    Head: AxisName,
    Windows: AxisName,
    D: MaterializableScalar,
    R,
    E,
    X,
    Count: M,
> crate::Module<(Wraps, X), E, &mut InDm<Rows, KvHeads, Head, Count, D>>
    for Paged<'_, Rows, KvHeads, Head, Windows, D, R>
{
    type Output = ();

    fn forward(
        &self,
        _: &mut Device,
        _: E,
        _: &mut InDm<Rows, KvHeads, Head, Count, D>,
        _: (Wraps, X),
    ) {
    }
}

impl<
    'l,
    Rows: AxisName,
    KvHeads: AxisName,
    Head: AxisName,
    Windows: AxisName,
    D: MaterializableScalar,
    R,
> Paged<'l, Rows, KvHeads, Head, Windows, D, R, (&'l Wrap<Rows>, usize)>
{
    /// Gathers the token rows `offsets` names into the head of `cells`, its first two blocks, each
    /// row's up to its count in `counts`: a row's tokens past its last window take the slots
    /// before its lowest visible token there.
    fn wraps<E: M, L: M, Count: M>(
        &self,
        device: &mut Device,
        (offsets, counts): (&HbmTensor<i32, Chip, E>, &HbmTensor<i32, Chip, L>),
        cells: &mut InDm<Rows, KvHeads, Head, Count, D>,
    ) {
        // A shard of two blocks or more holds the head; else it spans the first two shards.
        if const { ShardBlocks::<KvHeads>::SIZE >= 2 } {
            self.blocks
                .tokens
                .gather::<m![Block], m![KeyValue, KvHeads, Head]>()
                .by_byte_offsets(offsets)
                .prefix::<m![BlockIndex % 2, Block], _>(counts, 0)
                .to_dm_view(
                    &mut device.tdma,
                    cells
                        .view_mut()
                        .slice_tile::<m![BlockIndex / ({ Symbol<ShardBlocks<KvHeads>> })], 1, m![
                            KvHeads # ({ Symbol<HeadSlots<KvHeads>> }),
                            { Layout<Rows>::Lanes } # 8 % 4,
                            BlockIndex / ({ Symbol<ShardBlocks<KvHeads>> })
                                = 1 #{!} ({ Symbol<Shards<KvHeads>> })
                        ]>(0)
                        .tile::<m![BlockIndex % ({ Symbol<ShardBlocks<KvHeads>> }) / 2], 1, m![
                            { Count },
                            KeyValue,
                            BlockIndex % ({ Symbol<ShardBlocks<KvHeads>> }) / 2
                                = 1 #{!} ({ Symbol<ShardPairs<KvHeads>> }),
                            BlockIndex % 2,
                            Block,
                            Head
                        ]>(0),
                );
        } else {
            self.blocks
                .tokens
                .gather::<m![Block], m![KeyValue, KvHeads, Head]>()
                .by_byte_offsets(offsets)
                .prefix::<m![BlockIndex % 2, Block], _>(counts, 0)
                .to_dm_view(
                    &mut device.tdma,
                    cells
                        .view_mut()
                        .slice_tile::<
                            m![BlockIndex / ({ Symbol<ShardBlocks<KvHeads>> }) / 2],
                            1,
                            m![
                                KvHeads # ({ Symbol<HeadSlots<KvHeads>> }),
                                { Layout<Rows>::Lanes } # 8 % 4,
                                BlockIndex / ({ Symbol<ShardBlocks<KvHeads>> }) / 2
                                    = 1 #{!} ({ Symbol<ShardPairs<KvHeads>> }),
                                BlockIndex / ({ Symbol<ShardBlocks<KvHeads>> }) % 2
                            ]
                        >(0),
                );
        }
    }
}

/// Every row's tokens past its last window, into the head of the cells of its first window.
impl<
    'l,
    Rows: AxisName,
    KvHeads: AxisName,
    Head: AxisName,
    Windows: AxisName,
    D: MaterializableScalar,
    R,
    E,
> crate::Module<(Wraps, Live<(), Rows>), E, &mut InDm<Rows, KvHeads, Head, RowGroups<Rows>, D>>
    for Paged<'l, Rows, KvHeads, Head, Windows, D, R, (&'l Wrap<Rows>, usize)>
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        _: E,
        cells: &mut InDm<Rows, KvHeads, Head, RowGroups<Rows>, D>,
        _: (Wraps, Live<(), Rows>),
    ) {
        let (wrap, times) = self.wrap;
        for _ in 0..times {
            self.wraps(device, (&wrap.offsets, &wrap.counts), cells);
        }
    }
}

/// The tokens past their last window of the rows of `pass` alone, into the head of the cells of
/// their first window.
impl<
    'l,
    Rows: AxisName,
    KvHeads: AxisName,
    Head: AxisName,
    Windows: AxisName,
    D: MaterializableScalar,
    R,
    E,
    const N: usize,
> crate::Module<(Wraps, Pass<N>), E, &mut InDm<Rows, KvHeads, Head, TileRows<Rows, N>, D>>
    for Paged<'l, Rows, KvHeads, Head, Windows, D, R, (&'l Wrap<Rows>, usize)>
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        _: E,
        cells: &mut InDm<Rows, KvHeads, Head, TileRows<Rows, N>, D>,
        (_, pass): (Wraps, Pass<N>),
    ) {
        type Offsets<Rows, const N: usize> = HbmTensor<
            i32,
            Chip,
            m![{ TileRows<Rows, N> }, { Layout<Rows>::Lanes }, BlockIndex % 2, Block],
        >;
        type Counts<Rows, const N: usize> =
            HbmTensor<i32, Chip, m![{ TileRows<Rows, N> }, { Layout<Rows>::Lanes }]>;
        let (wrap, times) = self.wrap;
        for _ in 0..times {
            // SAFETY: a tile's rows are consecutive rows of the offsets and of the counts, which
            // nothing writes while a step attends over them.
            let (offsets, counts): (Offsets<Rows, N>, Counts<Rows, N>) = unsafe {
                (
                    (wrap.offsets.view())
                        .tile::<Tiles<Rows, N>, 1, m![
                            { Tiles<Rows, N> } = 1 # ({ Tiles<Rows, N> }), { TileRows<Rows, N> },
                            { Layout<Rows>::Lanes }, BlockIndex % 2, Block
                        ]>(pass.tile)
                        .alias(),
                    (wrap.counts.view())
                        .tile::<Tiles<Rows, N>, 1, m![
                            { Tiles<Rows, N> } = 1 # ({ Tiles<Rows, N> }), { TileRows<Rows, N> },
                            { Layout<Rows>::Lanes }
                        ]>(pass.tile)
                        .alias(),
                )
            };
            self.wraps(device, (&offsets, &counts), cells);
        }
    }
}

/// One of the chunk's two halves, the keys or the values of its tokens.
type Half<Count, KvHeads, Head> = m![{ Count }, KeyValue = 1 # 2, { ShardKeys<KvHeads> }, Head];

/// The query rows of the head a slice owns, for `Count` row groups.
type Query<Rows, KvHeads, Groups, Head, Count = RowGroups<Rows>> =
    DmTensor<bf16, Chip, Cluster<Rows>, CacheSlices<Rows, KvHeads>, m![{ Count }, Groups, Head]>;

/// One query per row and KV head, ready for its local shards to consume.
pub type Heads<Rows, KvHeads, Groups, Head> = DmTensor<
    bf16,
    Chip,
    Cluster<Rows>,
    Merged<Rows, KvHeads>,
    m![{ RowGroups<Rows> }, Groups, Head],
>;

/// The query rows in HBM where their store left them, each row's heads and groups in order.
pub type Stored<Rows, KvHeads, Groups, Head> =
    HbmTensor<bf16, Chip, m![Rows, KvHeads, Groups, Head]>;

/// Where attention reads the query rows it attends: a copy of every row group in data memory, or
/// the rows where their store left them, each pass landing its own, so that an envelope's queries
/// never take data memory at once.
pub(crate) trait Queries<Rows: AxisName, KvHeads: AxisName, Groups: AxisName, Head: AxisName> {
    /// Every row group's query rows on the slices of their heads' shards.
    fn whole(&self, device: &mut Device) -> Query<Rows, KvHeads, Groups, Head>;

    /// The query rows of `pass`'s row groups on the slices of their heads' shards.
    fn pass<const N: usize>(
        &self,
        device: &mut Device,
        pass: Pass<N>,
    ) -> Query<Rows, KvHeads, Groups, Head, TileRows<Rows, N>>;
}

/// The copy in data memory: a pass reads its row groups where they lie.
impl<Rows: AxisName, KvHeads: AxisName, Groups: AxisName, Head: AxisName>
    Queries<Rows, KvHeads, Groups, Head> for Heads<Rows, KvHeads, Groups, Head>
{
    fn whole(&self, device: &mut Device) -> Query<Rows, KvHeads, Groups, Head> {
        broadcast::<_, _, Groups, Head, RowGroups<Rows>, _>(device, self.view())
    }

    fn pass<const N: usize>(
        &self,
        device: &mut Device,
        pass: Pass<N>,
    ) -> Query<Rows, KvHeads, Groups, Head, TileRows<Rows, N>> {
        let rows = self.view().tile::<Tiles<Rows, N>, 1, m![
            { Tiles<Rows, N> } = 1 # ({ Tiles<Rows, N> }), { TileRows<Rows, N> }, Groups, Head
        ]>(pass.tile);
        broadcast::<_, _, Groups, Head, TileRows<Rows, N>, _>(device, rows)
    }
}

/// The rows in HBM: a pass lands its own row groups first, the whole envelope every one.
impl<Rows: AxisName, KvHeads: AxisName, Groups: AxisName, Head: AxisName>
    Queries<Rows, KvHeads, Groups, Head> for Stored<Rows, KvHeads, Groups, Head>
{
    fn whole(&self, device: &mut Device) -> Query<Rows, KvHeads, Groups, Head> {
        let heads: Heads<Rows, KvHeads, Groups, Head> = self.to_dm(&mut device.tdma);
        heads.whole(device)
    }

    fn pass<const N: usize>(
        &self,
        device: &mut Device,
        pass: Pass<N>,
    ) -> Query<Rows, KvHeads, Groups, Head, TileRows<Rows, N>> {
        let mut rows = DmTensor::<
            bf16,
            Chip,
            Cluster<Rows>,
            Merged<Rows, KvHeads>,
            m![{ TileRows<Rows, N> }, Groups, Head],
        >::new();
        (self.view())
            .tile::<Tiles<Rows, N>, 1, m![
                { Tiles<Rows, N> } = 1 # ({ Tiles<Rows, N> }), { TileRows<Rows, N> },
                { Layout<Rows>::Lanes }, KvHeads, Groups, Head
            ]>(pass.tile)
            .to_dm_view(&mut device.tdma, rows.view_mut());
        broadcast::<_, _, Groups, Head, TileRows<Rows, N>, _>(device, rows.view())
    }
}

/// `Count` row groups' query rows from the slices of their heads' merged attention to every shard
/// slice of their heads: each head's rows read once, then carried round its shards' ring.
fn broadcast<
    Rows: AxisName,
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    Count: M,
    E: M,
>(
    device: &mut Device,
    rows: DmTensorView<'_, bf16, Chip, Cluster<Rows>, Merged<Rows, KvHeads>, E>,
) -> Query<Rows, KvHeads, Groups, Head, Count> {
    device
        .sub
        .begin(rows)
        .fetch::<m![{ Count }, Groups, Head / 16], m![Head % 16]>()
        .switch::<CacheSlices<Rows, KvHeads>, m![{ Count }, Groups, Head / 16]>(
            SwitchConfig::CustomBroadcast {
                ring_size: Shards::<KvHeads>::SIZE,
            },
        )
        .collect::<m![{ Count }, Groups, Head / 16], m![Head % 16]>()
        .commit_trim::<m![Head % 16]>()
        .commit()
}

/// Each shard's bias of its slots, held per query row: a register operand repeats a single flit,
/// so an operand eight flits wide names the `Groups` axis of the logits it multiplies.
type Bias<Rows, KvHeads, Groups, Count = RowGroups<Rows>> = DmTensor<
    f32,
    Chip,
    Cluster<Rows>,
    CacheSlices<Rows, KvHeads>,
    m![{ Count }, Groups, { ShardKeys<KvHeads> }],
>;

pub(super) type Local<Rows, KvHeads, Groups, Head, const SCALE: usize> =
    shard::Local<Partition<Rows, KvHeads, Groups, Head, SCALE>, RowGroups<Rows>>;

pub(super) type Partial<Rows, KvHeads, Groups, Head, const SCALE: usize> =
    shard::Partial<Partition<Rows, KvHeads, Groups, Head, SCALE>, RowGroups<Rows>>;

/// Every head's attention over its values, one slice per lane and head.
pub type Attended<Rows, KvHeads, Groups, Head> = DmTensor<
    bf16,
    Chip,
    Cluster<Rows>,
    Merged<Rows, KvHeads>,
    m![{ RowGroups<Rows> }, Groups, Head],
>;

/// Grouped-query attention of `KvHeads` KV heads of `Groups` query heads of `Head` channels over
/// the windows of each row's paged history, every score multiplied by the FP32 of bits `SCALE`.
type Windowed<KvHeads, Groups, Head, const SCALE: usize> =
    Attention<PhantomData<(KvHeads, Groups, Head)>, F32<SCALE>, { Strategy::Windows }>;

/// The queries, the history they read, its mask, the windows after the first and the live rows.
pub type History<'a, Q, Rows, Windows, Source> = (
    &'a Q,
    &'a Source,
    &'a HbmTensor<f32, Chip, m![Rows, Windows, Cache]>,
    &'a Value,
    Live<(), Rows>,
);

impl<
    'a,
    Rows: AxisName,
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    Windows: AxisName,
    Source,
    Q: Queries<Rows, KvHeads, Groups, Head>,
    const SCALE: usize,
> Op<History<'a, Q, Rows, Windows, Source>> for Windowed<KvHeads, Groups, Head, SCALE>
where
    Source: Landed<Rows, KvHeads, Head>
        + for<'c> crate::Module<
            (usize, Live<(), Rows>),
            (),
            &'c mut Cells<Source, Rows, KvHeads, Head>,
        > + for<'c> crate::Module<
            (usize, Pass<32>),
            (),
            &'c mut Cells<Source, Rows, KvHeads, Head, TileRows<Rows, 32>>,
        > + for<'c> crate::Module<
            (usize, Pass<16>),
            (),
            &'c mut Cells<Source, Rows, KvHeads, Head, TileRows<Rows, 16>>,
        > + for<'c> crate::Module<
            (usize, Pass<8>),
            (),
            &'c mut Cells<Source, Rows, KvHeads, Head, TileRows<Rows, 8>>,
        > + for<'c> crate::Module<
            (Wraps, Live<(), Rows>),
            (),
            &'c mut Cells<Source, Rows, KvHeads, Head>,
        > + for<'c> crate::Module<
            (Wraps, Pass<32>),
            (),
            &'c mut Cells<Source, Rows, KvHeads, Head, TileRows<Rows, 32>>,
        > + for<'c> crate::Module<
            (Wraps, Pass<16>),
            (),
            &'c mut Cells<Source, Rows, KvHeads, Head, TileRows<Rows, 16>>,
        > + for<'c> crate::Module<
            (Wraps, Pass<8>),
            (),
            &'c mut Cells<Source, Rows, KvHeads, Head, TileRows<Rows, 8>>,
        >,
{
    type Output = Attended<Rows, KvHeads, Groups, Head>;

    fn forward(
        &self,
        device: &mut Device,
        history: History<'a, Q, Rows, Windows, Source>,
    ) -> Self::Output {
        self.forward(device, (history, (&(), &mut ())))
    }
}

/// The same history attended, `boundary` running on its state once in the first pass: before its
/// last window is prepared where windows overlap and no next pass may land under it, else after
/// its first window lands.
impl<
    'a,
    Rows: AxisName,
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    Windows: AxisName,
    Source,
    Q: Queries<Rows, KvHeads, Groups, Head>,
    H,
    G,
    const SCALE: usize,
> Op<(History<'a, Q, Rows, Windows, Source>, (&H, &mut G))>
    for Windowed<KvHeads, Groups, Head, SCALE>
where
    Source: Landed<Rows, KvHeads, Head>
        + for<'c> crate::Module<
            (usize, Live<(), Rows>),
            (),
            &'c mut Cells<Source, Rows, KvHeads, Head>,
        > + for<'c> crate::Module<
            (usize, Pass<32>),
            (),
            &'c mut Cells<Source, Rows, KvHeads, Head, TileRows<Rows, 32>>,
        > + for<'c> crate::Module<
            (usize, Pass<16>),
            (),
            &'c mut Cells<Source, Rows, KvHeads, Head, TileRows<Rows, 16>>,
        > + for<'c> crate::Module<
            (usize, Pass<8>),
            (),
            &'c mut Cells<Source, Rows, KvHeads, Head, TileRows<Rows, 8>>,
        > + for<'c> crate::Module<
            (Wraps, Live<(), Rows>),
            (),
            &'c mut Cells<Source, Rows, KvHeads, Head>,
        > + for<'c> crate::Module<
            (Wraps, Pass<32>),
            (),
            &'c mut Cells<Source, Rows, KvHeads, Head, TileRows<Rows, 32>>,
        > + for<'c> crate::Module<
            (Wraps, Pass<16>),
            (),
            &'c mut Cells<Source, Rows, KvHeads, Head, TileRows<Rows, 16>>,
        > + for<'c> crate::Module<
            (Wraps, Pass<8>),
            (),
            &'c mut Cells<Source, Rows, KvHeads, Head, TileRows<Rows, 8>>,
        >,
    H: for<'g, 'x> crate::Module<&'g mut G, (), &'x mut (), Output = ()>,
{
    type Output = Attended<Rows, KvHeads, Groups, Head>;

    /// Attends each row's history and merges its shard-local softmax terms, window zero seeding the
    /// carries; row groups run in shared passes when all hold a live row and a pass holds 32 rows,
    /// else only the live ones, in the widest passes a slice's registers hold.
    fn forward(
        &self,
        device: &mut Device,
        ((query, history, bias, rest, live), (boundary, state)): (
            History<'a, Q, Rows, Windows, Source>,
            (&H, &mut G),
        ),
    ) -> Self::Output {
        // The pass that runs first takes the boundary once, the others none.
        let mut once = |device: &mut Device, times: usize| {
            for _ in 0..times {
                crate::Module::forward(boundary, &mut *device, (), &mut (), &mut *state);
            }
        };
        if const { Self::ROWS == 32 } && live.count::<Lanes<Rows>>() == <RowGroups<Rows> as M>::SIZE
        {
            let projected = query.whole(device);
            let rest = rest.extent::<m![{ Symbol<Rest<Windows>> }]>();
            type Transposed<Rows, KvHeads, Groups, Head, const SCALE: usize> =
                shard::Transposed<Partition<Rows, KvHeads, Groups, Head, SCALE>, RowGroups<Rows>>;
            // Staged per window, not before the loop, a slice holds one or two windows' bias
            // however many a history spans, and a skipped window moves nothing.
            type Whole<Rows, KvHeads, Groups, Head, Source> = Landing<
                Rows,
                KvHeads,
                Groups,
                Head,
                RowGroups<Rows>,
                <Source as Landed<Rows, KvHeads, Head>>::Element,
            >;
            type Restored<Source, Rows, KvHeads, Head> =
                <Source as Landed<Rows, KvHeads, Head>>::Restored<RowGroups<Rows>>;
            let mut lands =
                |device: &mut Device,
                 at: usize,
                 landing: &mut Whole<Rows, KvHeads, Groups, Head, Source>| {
                    crate::Module::forward(
                        history,
                        &mut *device,
                        (),
                        &mut landing.cells,
                        (at, live),
                    );
                    bias.view()
                        .tile::<m![Windows], 1, m![Rows, Windows = 1 # Windows, Cache]>(at)
                        .to_dm_view(&mut device.tdma, landing.bias.view_mut());
                };
            let wraps =
                |device: &mut Device, landing: &mut Whole<Rows, KvHeads, Groups, Head, Source>| {
                    crate::Module::forward(
                        history,
                        &mut *device,
                        (),
                        &mut landing.cells,
                        (Wraps, live),
                    );
                };
            // The place a window's keys widen into holds them until they fill the registers.
            let prepares = |device: &mut Device,
                            landing: &Whole<Rows, KvHeads, Groups, Head, Source>,
                            values: &mut Transposed<Rows, KvHeads, Groups, Head, SCALE>|
             -> Restored<Source, Rows, KvHeads, Head> {
                let mut restored = Source::place();
                history.keys(device, &landing.cells, &mut restored);
                transposes::<Rows, KvHeads, Groups, Head, SCALE, _>(
                    device,
                    chunk(&landing.cells),
                    values,
                );
                restored
            };
            // One tile's keys fill the tensor registers at a time, so the attention reads its place.
            let attends =
                |device: &mut Device,
                 (landing, restored): (
                    &Whole<Rows, KvHeads, Groups, Head, Source>,
                    Restored<Source, Rows, KvHeads, Head>,
                ),
                 values: &Transposed<Rows, KvHeads, Groups, Head, SCALE>| {
                    let scales = history.scales(device);
                    attend::<Rows, KvHeads, Groups, Head, SCALE>(
                        device,
                        &projected,
                        (Source::widened(&landing.cells, &restored), values),
                        (&landing.bias, scales.as_ref()),
                    )
                };
            if const { Self::overlaps::<Rows>(Rows::SIZE) } {
                let mut landings = (
                    Landing::new(),
                    Landing::new(),
                    Transposed::<Rows, KvHeads, Groups, Head, SCALE>::new(),
                );
                lands(device, 0, &mut landings.0);
                wraps(device, &mut landings.0);
                return <Partition<Rows, KvHeads, Groups, Head, SCALE> as shard::Partition>::overlapped::<
                    _,
                    _,
                    _,
                    Rest<Windows>,
                    false,
                >(
                    device,
                    rest,
                    (&mut landings.0, &mut landings.1, &mut landings.2),
                    (&mut lands, prepares, attends),
                    (|device| once(device, 1), |_, _, _| ()),
                );
            }
            let mut landed = Landing::new();
            lands(device, 0, &mut landed);
            wraps(device, &mut landed);
            return <Partition<Rows, KvHeads, Groups, Head, SCALE> as shard::Partition>::windows::<
                _,
                Rest<Windows>,
            >(
                device,
                rest,
                landed,
                (
                    |device: &mut Device, at| {
                        let mut landing = Landing::new();
                        lands(device, at, &mut landing);
                        landing
                    },
                    |device: &mut Device, landing| {
                        let mut values = Transposed::<Rows, KvHeads, Groups, Head, SCALE>::new();
                        let restored = prepares(device, &landing, &mut values);
                        attends(device, (&landing, restored), &values)
                    },
                ),
                |device| once(device, 1),
            );
        }
        let rest = rest.extent::<m![{ Symbol<Rest<Windows>> }]>();
        let mut attended = Attended::<Rows, KvHeads, Groups, Head>::new();
        let inputs = (query, history, bias, rest);
        // Passes over the widest tiles a pass holds, then at most one over each narrower tile: most
        // of a pass's cost is paid once whatever its groups, so the live groups take the fewest.
        let (groups, wide, half) = (
            live.count::<Lanes<Rows>>(),
            <TileGroups<Rows, 32> as AxisName>::SIZE,
            <TileGroups<Rows, 16> as AxisName>::SIZE,
        );
        if const { Self::ROWS == 32 } {
            let (wides, halves) = (groups / wide, groups % wide / half);
            self.passes::<_, _, _, _, 32>(
                device,
                inputs,
                (0, wides, live),
                (&mut once, 1),
                &mut attended,
            );
            let first = wides * (wide / half);
            let times = 1 - wides.min(1);
            self.passes::<_, _, _, _, 16>(
                device,
                inputs,
                (first, halves, live),
                (&mut once, times),
                &mut attended,
            );
            let first = groups / half * half;
            let times = (1 - wides.min(1)) * (1 - halves);
            let pass = (first, groups % half, live);
            self.passes::<_, _, _, _, 8>(device, inputs, pass, (&mut once, times), &mut attended);
        } else if const { Self::ROWS == 16 } {
            let halves = groups / half;
            self.passes::<_, _, _, _, 16>(
                device,
                inputs,
                (0, halves, live),
                (&mut once, 1),
                &mut attended,
            );
            let pass = (halves * half, groups % half, live);
            let times = 1 - halves.min(1);
            self.passes::<_, _, _, _, 8>(device, inputs, pass, (&mut once, times), &mut attended);
        } else {
            self.passes::<_, _, _, _, 8>(
                device,
                inputs,
                (0, groups, live),
                (&mut once, 1),
                &mut attended,
            );
        }
        // Without a live row no pass runs, so the boundary runs alone.
        once(device, 1 - groups.min(1));
        attended
    }
}

impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, const SCALE: usize>
    Windowed<KvHeads, Groups, Head, SCALE>
{
    /// Rows a pass attends at most: 32, 16 or 8, the most whose keys and score masks a slice's
    /// registers hold and whose attention its data memory holds.
    const ROWS: usize = attention::pass(Head::SIZE, Groups::SIZE, Keys::<KvHeads>::SIZE);

    /// Whether a pass of `rows` rows over the envelope `Rows` lands each window while the one
    /// before it is attended.
    const fn overlaps<Rows: AxisName>(rows: usize) -> bool {
        attention::overlaps(
            Head::SIZE,
            Groups::SIZE,
            Keys::<KvHeads>::SIZE,
            (rows, Rows::SIZE),
        )
    }

    /// Whether a pass of `N` rows over the envelope `Rows` lands the next pass's first window: only
    /// the widest passes repeat, and only in an envelope that holds two of them.
    const fn follows<Rows: AxisName, const N: usize>() -> bool {
        N == Self::ROWS && <RowGroups<Rows> as M>::SIZE > <TileGroups<Rows, N> as AxisName>::SIZE
    }

    /// Whether a pass of `N` rows streams each window's keys and values from where they landed:
    /// where it lands each window while the one before is attended, and what follows its last
    /// window, if anything, only into a place already attended: a history of two windows at most.
    const fn streams<Rows: AxisName, Windows: AxisName, const N: usize>() -> bool {
        attention::streams(Head::SIZE, Groups::SIZE)
            && Self::overlaps::<Rows>(N)
            && (!Self::follows::<Rows, N>() || Windows::SIZE <= 2)
    }

    /// Attends the row groups of `count` passes over the `N`-row tiles from tile `first` and writes
    /// their rows in `attended`; `boundary` runs `times` times in the first pass, where
    /// [`Windowed`]'s runs. Overlapping passes share two places: each lands the next one's first
    /// window.
    fn passes<Rows: AxisName, Windows: AxisName, Source, Q, const N: usize>(
        &self,
        device: &mut Device,
        (query, history, bias, rest): (
            &Q,
            &Source,
            &HbmTensor<f32, Chip, m![Rows, Windows, Cache]>,
            usize,
        ),
        (first, count, live): (usize, usize, Live<(), Rows>),
        (boundary, times): (&mut impl FnMut(&mut Device, usize), usize),
        attended: &mut Attended<Rows, KvHeads, Groups, Head>,
    ) where
        Source: Landed<Rows, KvHeads, Head>
            + for<'c> crate::Module<
                (usize, Pass<N>),
                (),
                &'c mut Cells<Source, Rows, KvHeads, Head, TileRows<Rows, N>>,
            > + for<'c> crate::Module<
                (Wraps, Pass<N>),
                (),
                &'c mut Cells<Source, Rows, KvHeads, Head, TileRows<Rows, N>>,
            >,
        Q: Queries<Rows, KvHeads, Groups, Head>,
    {
        type Place<Rows, KvHeads, Groups, Head, Source, const N: usize> = Landing<
            Rows,
            KvHeads,
            Groups,
            Head,
            TileRows<Rows, N>,
            <Source as Landed<Rows, KvHeads, Head>>::Element,
        >;
        type Transposed<Rows, KvHeads, Groups, Head, const SCALE: usize, const N: usize> =
            shard::Transposed<TileShards<Rows, KvHeads, Groups, Head, SCALE, N>, TileRows<Rows, N>>;
        type Registers<Rows, KvHeads, Groups, Head, const SCALE: usize, const N: usize> =
            shard::Registers<TileShards<Rows, KvHeads, Groups, Head, SCALE, N>, TileRows<Rows, N>>;
        // The bias goes first: a transfer issued after the gather may wait for it to finish.
        let lands =
            |device: &mut Device,
             (at, pass): (usize, Pass<N>),
             landing: &mut Place<Rows, KvHeads, Groups, Head, Source, N>| {
                bias.view()
                    .tile::<m![Windows], 1, m![Rows, Windows = 1 # Windows, Cache]>(at)
                    .tile::<Tiles<Rows, N>, 1, m![
                        { Tiles<Rows, N> } = 1 # ({ Tiles<Rows, N> }), { TileRows<Rows, N> },
                        { Layout<Rows>::Lanes }, Windows = 1 # Windows, Cache
                    ]>(pass.tile)
                    .to_dm_view(&mut device.tdma, landing.bias.view_mut());
                crate::Module::forward(history, &mut *device, (), &mut landing.cells, (at, pass));
            };
        // The tokens past the rows' last window, into the head of their first window's place.
        let wraps =
            |device: &mut Device,
             pass: Pass<N>,
             landing: &mut Place<Rows, KvHeads, Groups, Head, Source, N>| {
                crate::Module::forward(
                    history,
                    &mut *device,
                    (),
                    &mut landing.cells,
                    (Wraps, pass),
                );
            };
        let projects = |device: &mut Device, pass: Pass<N>| query.pass(device, pass);
        // Everything a window's attention reads of its place, so the place is free once prepared;
        // the registers fill first, so the key fill runs beside the values' transpose.
        let prepares = |device: &mut Device,
                        landing: &Place<Rows, KvHeads, Groups, Head, Source, N>,
                        values: &mut Transposed<Rows, KvHeads, Groups, Head, SCALE, N>|
         -> Registers<Rows, KvHeads, Groups, Head, SCALE, N> {
            let mut restored = Source::place::<TileRows<Rows, N>>();
            let keys = history
                .keys(device, &landing.cells, &mut restored)
                .tile::<m![KeyValue], 1, Half<TileRows<Rows, N>, KvHeads, Head>>(0);
            let registers =
                <TileShards<Rows, KvHeads, Groups, Head, SCALE, N> as shard::Partition>::fill::<
                    TileRows<Rows, N>,
                    _,
                    _,
                >(device, keys, landing.bias.view());
            let values_half = chunk(&landing.cells)
                .tile::<m![KeyValue], 1, Half<TileRows<Rows, N>, KvHeads, Head>>(1);
            <TileShards<Rows, KvHeads, Groups, Head, SCALE, N> as shard::Partition>::transpose::<
                TileRows<Rows, N>,
                _,
                _,
                _,
            >(device, values_half, values.view_mut());
            registers
        };
        let attends =
            |device: &mut Device,
             projected: &Query<Rows, KvHeads, Groups, Head, TileRows<Rows, N>>,
             registers: Registers<Rows, KvHeads, Groups, Head, SCALE, N>,
             values: &Transposed<Rows, KvHeads, Groups, Head, SCALE, N>| {
                let mut partial =
                    shard::Partial::<TileShards<Rows, KvHeads, Groups, Head, SCALE, N>>::new();
                let mut lse =
                    shard::Local::<TileShards<Rows, KvHeads, Groups, Head, SCALE, N>>::new();
                let scales = history.scales(device);
                <TileShards<Rows, KvHeads, Groups, Head, SCALE, N> as shard::Partition>::attend::<
                    TileRows<Rows, N>,
                    _,
                    _,
                    _,
                    _,
                >(
                    device,
                    projected.view(),
                    (&registers, values.view(), scales.as_ref()),
                    (partial.view_mut(), lse.view_mut()),
                );
                (partial, lse)
            };
        let mut stores =
            |device: &mut Device,
             merged: shard::Attended<TileShards<Rows, KvHeads, Groups, Head, SCALE, N>>,
             pass: Pass<N>| {
                merged.view().to_dm_view(
                    &mut device.tdma,
                    attended.view_mut().tile::<Tiles<Rows, N>, 1, m![
                        { Tiles<Rows, N> } = 1 #{!} ({ Tiles<Rows, N> }), { TileRows<Rows, N> },
                        Groups, Head
                    ]>(pass.tile),
                );
            };
        if const { Self::streams::<Rows, Windows, N>() } {
            type Masks<Rows, KvHeads, Groups, Head, const SCALE: usize, const N: usize> =
                shard::Masks<TileShards<Rows, KvHeads, Groups, Head, SCALE, N>, TileRows<Rows, N>>;
            type Restored<Source, Rows, KvHeads, Head, const N: usize> =
                <Source as Landed<Rows, KvHeads, Head>>::Restored<TileRows<Rows, N>>;
            // Codes widen where they landed: the keys last here, as the next window's landing is
            // issued, and the values once scored, as Sub fills the weights.
            let prepares = |device: &mut Device,
                            landing: &Place<Rows, KvHeads, Groups, Head, Source, N>,
                            _: &mut ()|
             -> (
                Masks<Rows, KvHeads, Groups, Head, SCALE, N>,
                Restored<Source, Rows, KvHeads, Head, N>,
            ) {
                let masks =
                    <TileShards<Rows, KvHeads, Groups, Head, SCALE, N> as shard::Partition>::mask::<
                        TileRows<Rows, N>,
                        _,
                    >(device, landing.bias.view());
                let mut restored = Source::place::<TileRows<Rows, N>>();
                history.keys(device, &landing.cells, &mut restored);
                (masks, restored)
            };
            let (mut even, mut odd, mut nothing) = (
                Place::<Rows, KvHeads, Groups, Head, Source, N>::new(),
                Landing::new(),
                (),
            );
            for tile in 0..count {
                let pass = Pass::<N>::of(first + tile, live);
                // The query rows go ahead of the first window, which would hold them back in the
                // transfer queue, so they are spread while the window's landing is issued.
                let projected = projects(device, pass);
                // A pass lands its own first window unless the pass before it did, under its last.
                if const { Self::follows::<Rows, N>() } {
                    for _ in 0..(1 - tile.min(1) + (1 - rest.min(1))).min(1) {
                        lands(device, (0, pass), &mut even);
                    }
                } else {
                    lands(device, (0, pass), &mut even);
                }
                wraps(device, pass, &mut even);
                let query =
                    <TileShards<Rows, KvHeads, Groups, Head, SCALE, N> as shard::Partition>::station::<
                        TileRows<Rows, N>,
                        _,
                    >(device, projected.view());
                let mut land =
                    |device: &mut Device,
                     at,
                     landing: &mut Place<Rows, KvHeads, Groups, Head, Source, N>| {
                        lands(device, (at, pass), landing)
                    };
                let mut attend = |device: &mut Device,
                                  (landing, (masks, mut restored)): (
                    &Place<Rows, KvHeads, Groups, Head, Source, N>,
                    (
                        Masks<Rows, KvHeads, Groups, Head, SCALE, N>,
                        Restored<Source, Rows, KvHeads, Head, N>,
                    ),
                ),
                                  _: &()| {
                    let mut partial =
                        shard::Partial::<TileShards<Rows, KvHeads, Groups, Head, SCALE, N>>::new();
                    let mut lse =
                        shard::Local::<TileShards<Rows, KvHeads, Groups, Head, SCALE, N>>::new();
                    let scales =
                        history.scales::<TileShards<Rows, KvHeads, Groups, Head, SCALE, N>>(device);
                    let key = Source::widened(&landing.cells, &restored)
                        .tile::<m![KeyValue], 1, Half<TileRows<Rows, N>, KvHeads, Head>>(0);
                    let (weight, total) =
                        <TileShards<Rows, KvHeads, Groups, Head, SCALE, N> as shard::Partition>::score::<
                            TileRows<Rows, N>,
                            _,
                            _,
                        >(
                            device,
                            &query,
                            (key, &masks, scales.as_ref().map(|shard::Scales(key, _)| key)),
                            lse.view_mut(),
                        );
                    let value = history
                        .values(device, &landing.cells, &mut restored)
                        .tile::<m![KeyValue], 1, Half<TileRows<Rows, N>, KvHeads, Head>>(1);
                    <TileShards<Rows, KvHeads, Groups, Head, SCALE, N> as shard::Partition>::weigh::<
                        TileRows<Rows, N>,
                        _,
                        _,
                    >(
                        device,
                        (
                            &weight,
                            &total,
                            scales.as_ref().map(|shard::Scales(_, value)| value),
                        ),
                        value,
                        partial.view_mut(),
                    );
                    (partial, lse)
                };
                let boundary = |device: &mut Device| {
                    for _ in 0..1 - tile.min(1) {
                        boundary(device, times);
                    }
                };
                // The next pass's first window, into the place of this pass's first, which its
                // second is attended past.
                let mut follow =
                    |device: &mut Device,
                     at: usize,
                     landing: &mut Place<Rows, KvHeads, Groups, Head, Source, N>| {
                        let within = (rest + 1 - at).min(1);
                        for _ in 0..within + (1 - within) * ((count - tile) / 2).min(1) {
                            let pass = Pass::of(first + tile + 1 - within, live);
                            lands(device, (within * at, pass), landing);
                        }
                    };
                let places = (&mut even, &mut odd, &mut nothing);
                let steps = (&mut land, &prepares, &mut attend);
                let merged = if const { Self::follows::<Rows, N>() } {
                    <TileShards<Rows, KvHeads, Groups, Head, SCALE, N> as shard::Partition>::overlapped::<
                        _,
                        _,
                        _,
                        Rest<Windows>,
                        true,
                    >(device, rest, places, steps, (boundary, &mut follow))
                } else {
                    <TileShards<Rows, KvHeads, Groups, Head, SCALE, N> as shard::Partition>::overlapped::<
                        _,
                        _,
                        _,
                        Rest<Windows>,
                        false,
                    >(device, rest, places, steps, (boundary, &mut follow))
                };
                stores(device, merged, pass);
            }
        } else if const { Self::overlaps::<Rows>(N) } {
            let (mut even, mut odd) = (
                Place::<Rows, KvHeads, Groups, Head, Source, N>::new(),
                Landing::new(),
            );
            let mut values = Transposed::<Rows, KvHeads, Groups, Head, SCALE, N>::new();
            for tile in 0..count {
                let pass = Pass::<N>::of(first + tile, live);
                let projected = projects(device, pass);
                // A pass lands its own first window unless the pass before it did, under its last.
                if const { Self::follows::<Rows, N>() } {
                    for _ in 0..(1 - tile.min(1) + (1 - rest.min(1))).min(1) {
                        lands(device, (0, pass), &mut even);
                    }
                } else {
                    lands(device, (0, pass), &mut even);
                }
                // Behind the first window, whichever pass landed it, and ahead of every attention
                // of the pass, so none waits for it.
                wraps(device, pass, &mut even);
                let mut land =
                    |device: &mut Device,
                     at,
                     landing: &mut Place<Rows, KvHeads, Groups, Head, Source, N>| {
                        lands(device, (at, pass), landing)
                    };
                // Window `at` of this pass while its history holds it, past that the next pass's first.
                let mut follow =
                    |device: &mut Device,
                     at: usize,
                     landing: &mut Place<Rows, KvHeads, Groups, Head, Source, N>| {
                        let within = (rest + 1 - at).min(1);
                        for _ in 0..within + (1 - within) * ((count - tile) / 2).min(1) {
                            let pass = Pass::of(first + tile + 1 - within, live);
                            lands(device, (within * at, pass), landing);
                        }
                    };
                let mut attend = |device: &mut Device,
                                  (_, registers): (
                    &Place<Rows, KvHeads, Groups, Head, Source, N>,
                    Registers<Rows, KvHeads, Groups, Head, SCALE, N>,
                ),
                                  values: &Transposed<Rows, KvHeads, Groups, Head, SCALE, N>| {
                    attends(device, &projected, registers, values)
                };
                let boundary = |device: &mut Device| {
                    for _ in 0..1 - tile.min(1) {
                        boundary(device, times);
                    }
                };
                let places = (&mut even, &mut odd, &mut values);
                let steps = (&mut land, &prepares, &mut attend);
                let merged = if const { Self::follows::<Rows, N>() } {
                    <TileShards<Rows, KvHeads, Groups, Head, SCALE, N> as shard::Partition>::overlapped::<
                        _,
                        _,
                        _,
                        Rest<Windows>,
                        true,
                    >(device, rest, places, steps, (boundary, &mut follow))
                } else {
                    <TileShards<Rows, KvHeads, Groups, Head, SCALE, N> as shard::Partition>::overlapped::<
                        _,
                        _,
                        _,
                        Rest<Windows>,
                        false,
                    >(device, rest, places, steps, (boundary, &mut follow))
                };
                stores(device, merged, pass);
            }
        } else {
            for tile in 0..count {
                let pass = Pass::<N>::of(first + tile, live);
                let projected = projects(device, pass);
                let mut landed = Landing::new();
                lands(device, (0, pass), &mut landed);
                wraps(device, pass, &mut landed);
                let merged = <TileShards<Rows, KvHeads, Groups, Head, SCALE, N> as shard::Partition>::windows::<
                    _,
                    Rest<Windows>,
                >(
                    device,
                    rest,
                    landed,
                    (
                        |device: &mut Device, at| {
                            let mut landing = Landing::new();
                            lands(device, (at, pass), &mut landing);
                            landing
                        },
                        |device: &mut Device, landing| {
                            let mut values = Transposed::<Rows, KvHeads, Groups, Head, SCALE, N>::new();
                            let registers = prepares(device, &landing, &mut values);
                            attends(device, &projected, registers, &values)
                        },
                    ),
                    |device: &mut Device| {
                        for _ in 0..1 - tile.min(1) {
                            boundary(device, times);
                        }
                    },
                );
                stores(device, merged, pass);
            }
        }
    }
}

/// The codes of `cells`' half `half` (0 the keys, 1 the values) widened to BF16 into the same half
/// of `place`, exactly, in one pass without the vector stage: the cells as attention reads them.
fn widen<'c, Rows: AxisName, KvHeads: AxisName, Head: AxisName, Count: M>(
    device: &mut Device,
    cells: &'c InDm<Rows, KvHeads, Head, Count, f8e4m3>,
    place: &'c mut InDm<Rows, KvHeads, Head, Count>,
    half: usize,
) -> ChunkView<'c, Rows, KvHeads, Head, Count> {
    type Time<Count, KvHeads, Head, const N: usize> =
        m![{ Count }, { ShardKeys<KvHeads> }, Head / { N }];
    // SAFETY: as in `chunk`.
    let widened: ChunkViewMut<'_, Rows, KvHeads, Head, Count> =
        unsafe { place.view_mut().reshape() };
    device
        .main
        .begin(chunk(cells).tile::<m![KeyValue], 1, Half<Count, KvHeads, Head>>(half))
        .fetch::<Time<Count, KvHeads, Head, 16>, m![Head % 16]>()
        .fetch_cast::<f32>()
        .collect::<Time<Count, KvHeads, Head, 8>, m![Head % 8]>()
        .cast::<bf16, m![Head % 8 # 16]>()
        .commit_trim::<m![Head % 8]>()
        .commit_view(widened.tile::<m![KeyValue], 1, m![
            { Count }, KeyValue = 1 #{!} 2, { ShardKeys<KvHeads> }, Head
        ]>(half));
    chunk(place)
}

/// The values of every row group of `chunk`, each a `D`, transposed into `values`, as
/// [`attend`] reads them.
fn transposes<
    Rows: AxisName,
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    const SCALE: usize,
    D: shard::Value,
>(
    device: &mut Device,
    chunk: ChunkView<'_, Rows, KvHeads, Head, RowGroups<Rows>, D>,
    values: &mut shard::Transposed<Partition<Rows, KvHeads, Groups, Head, SCALE>, RowGroups<Rows>>,
) {
    #[unroll]
    for tile in 0..<Tiles<Rows> as M>::SIZE {
        <Partition<Rows, KvHeads, Groups, Head, SCALE> as shard::Partition>::transpose::<
            TileRows<Rows>,
            _,
            _,
            _,
        >(
            device,
            chunk
                .tile::<m![KeyValue], 1, Half<RowGroups<Rows>, KvHeads, Head>>(1)
                .tile::<Tiles<Rows>, 1, m![
                    { Tiles<Rows> } = 1 # ({ Tiles<Rows> }), { TileRows<Rows> }, KeyValue = 1 # 2,
                    { ShardKeys<KvHeads> }, Head
                ]>(tile),
            values.view_mut().tile::<Tiles<Rows>, 1, m![
                { Tiles<Rows> } = 1 #{!} ({ Tiles<Rows> }), { TileRows<Rows> }, Head,
                { ShardKeys<KvHeads> }
            ]>(tile),
        );
    }
}

/// Attention in the partition the chunk was gathered in: each shard scores its own slots and
/// normalises them by its own sum, so the partial it returns is already the shard's attention.
fn attend<
    Rows: AxisName,
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    const SCALE: usize,
>(
    device: &mut Device,
    query: &Query<Rows, KvHeads, Groups, Head>,
    (chunk, values): (
        ChunkView<'_, Rows, KvHeads, Head, RowGroups<Rows>>,
        &shard::Transposed<Partition<Rows, KvHeads, Groups, Head, SCALE>, RowGroups<Rows>>,
    ),
    (bias, scales): (
        &Bias<Rows, KvHeads, Groups>,
        Option<&shard::Scales<Partition<Rows, KvHeads, Groups, Head, SCALE>>>,
    ),
) -> (
    Partial<Rows, KvHeads, Groups, Head, SCALE>,
    Local<Rows, KvHeads, Groups, Head, SCALE>,
) {
    let mut partial = Partial::<Rows, KvHeads, Groups, Head, SCALE>::new();
    let mut lse = Local::<Rows, KvHeads, Groups, Head, SCALE>::new();
    #[unroll]
    for tile in 0..<Tiles<Rows> as M>::SIZE {
        let registers = <Partition<Rows, KvHeads, Groups, Head, SCALE> as shard::Partition>::fill::<
            TileRows<Rows>,
            _,
            _,
        >(
            device,
            chunk
                .tile::<m![KeyValue], 1, Half<RowGroups<Rows>, KvHeads, Head>>(0)
                .tile::<Tiles<Rows>, 1, m![
                    { Tiles<Rows> } = 1 # ({ Tiles<Rows> }), { TileRows<Rows> }, KeyValue = 1 # 2,
                    { ShardKeys<KvHeads> }, Head
                ]>(tile),
            bias.view().tile::<Tiles<Rows>, 1, m![
                { Tiles<Rows> } = 1 # ({ Tiles<Rows> }), { TileRows<Rows> }, Groups,
                { ShardKeys<KvHeads> }
            ]>(tile),
        );
        <Partition<Rows, KvHeads, Groups, Head, SCALE> as shard::Partition>::attend::<TileRows<Rows>, _, _, _, _>(
            device,
            query
                .view()
                .tile::<Tiles<Rows>, 1, m![{ Tiles<Rows> } = 1 # ({ Tiles<Rows> }), { TileRows<Rows> }, Groups, Head]>(tile),
            (
                &registers,
                values.view().tile::<Tiles<Rows>, 1, m![
                    { Tiles<Rows> } = 1 # ({ Tiles<Rows> }), { TileRows<Rows> }, Head,
                    { ShardKeys<KvHeads> }
                ]>(tile),
                scales,
            ),
            (
                partial
                    .view_mut()
                    .tile::<Tiles<Rows>, 1, m![{ Tiles<Rows> } = 1 #{!} ({ Tiles<Rows> }), { TileRows<Rows> }, Groups, Head]>(tile),
                lse.view_mut().tile::<Tiles<Rows>, 1, m![
                    { Tiles<Rows> } = 1 #{!} ({ Tiles<Rows> }), { TileRows<Rows> }, Groups, 1 # 8
                ]>(tile),
            ),
        );
    }
    (partial, lse)
}

#[cfg(test)]
mod tests {
    use crate::Op;
    use furiosa_opt_std::prelude::*;
    use furiosa_opt_std::runtime::launch;

    use super::shard::Partition;
    use super::{Cache, Chip, KeyValue, Live, Paged, Windowed};
    use crate::ops::attention::geometry::QueryGroups;
    use crate::ops::attention::tests::{Blocks, Chunks, Head, KvHeads};
    use crate::ops::index::table::BlockCounts;
    use crate::ops::index::table::block_counts;
    use crate::ops::index::table::{BlockIndex, PADDING_BIAS, TEST_TOKENS};
    use crate::ops::normalization::tests::random;
    use crate::tensor::axis::Rows;
    use crate::tensor::axis::{Eight, Single, Sixteen};
    use crate::tensor::cache::Block;
    use crate::tensor::live::tests::count;
    use crate::tensor::scalar::Value;
    use crate::tests::DEVICE;
    use std::f32::consts::FRAC_1_SQRT_2;
    use std::iter::repeat_n;

    axes![Groups = 2, Windows = 3];

    type PrefixGroups = QueryGroups<4>;

    /// The score multiplier of 128-wide heads, `1 / sqrt(128)`.
    const INVERSE_ROOT: usize = (FRAC_1_SQRT_2 / 8.0).to_bits() as usize;

    type Query<Count> = HbmTensor<bf16, Chip, m![Count, KvHeads, Groups, Head]>;

    type Source<Count, Depth = Chunks> =
        HbmTensor<bf16, Chip, m![Count, Depth, Cache, KeyValue, KvHeads, Head]>;

    type Bias<Count, Depth = Chunks> = HbmTensor<f32, Chip, m![Count, Depth, Cache]>;
    type Output<Count> = HbmTensor<bf16, Chip, m![KvHeads, Count, Groups, Head]>;

    type PrefixQuery<Count> = HbmTensor<bf16, Chip, m![Count, KvHeads, PrefixGroups, Head]>;
    type PrefixOutput<Count> = HbmTensor<bf16, Chip, m![KvHeads, Count, PrefixGroups, Head]>;

    #[device(chip = 1)]
    fn attend_rows<const SCALE: usize>(
        device: &mut Device,
        attention: &Windowed<KvHeads, Groups, Head, SCALE>,
        query: &Query<Eight>,
        source: &Source<Eight>,
        bias: &Bias<Eight>,
        rest: &Value,
        output: &mut Output<Eight>,
    ) {
        let query: super::Heads<Eight, KvHeads, Groups, Head> = query.to_dm(&mut device.tdma);
        attention
            .forward(&mut *device, (&query, source, bias, rest, Live::all()))
            .view()
            .to_hbm_view(&mut device.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    fn attend_row(
        device: &mut Device,
        query: &Query<Single>,
        source: &Source<Single>,
        bias: &Bias<Single>,
        rest: &Value,
        output: &mut Output<Single>,
    ) {
        let query: super::Heads<Single, KvHeads, Groups, Head> = query.to_dm(&mut device.tdma);
        Windowed::<KvHeads, Groups, Head, INVERSE_ROOT>::default()
            .forward(&mut *device, (&query, source, bias, rest, Live::all()))
            .view()
            .to_hbm_view(&mut device.tdma, output.view_mut());
    }

    #[device(chip = 1)]
    fn gather_rows<Count: AxisName>(
        device: &mut Device,
        blocks: &Blocks,
        table: &HbmTensor<i32, Chip, m![Chunks, Count, BlockIndex]>,
        lengths: &BlockCounts<Count, Chunks>,
        output: &mut HbmTensor<bf16, Chip, m![Chunks, Count, Cache, KeyValue, KvHeads, Head]>,
    ) {
        let history = Paged {
            blocks,
            table,
            lengths,
            scales: &(),
            wrap: (),
        };
        for w in 0..Chunks::SIZE {
            let mut landing = super::Landing::<Count, KvHeads, Groups, Head, _>::new();
            crate::Module::forward(
                &history,
                &mut *device,
                (),
                &mut landing.cells,
                (w, Live::all()),
            );
            super::chunk(&landing.cells).to_hbm_view(
                &mut device.tdma,
                output.view_mut().tile::<m![Chunks], 1, m![
                    Chunks = 1 #{!} Chunks, Count, Cache, KeyValue, KvHeads, Head
                ]>(w),
            );
        }
    }

    /// Every table entry writes its block's rows, in order, on the `Block::SIZE` chunk slots at the
    /// entry's position, for `Count` rows whose tables name the pool's blocks in different orders.
    async fn gathers_each_block_onto_its_slots_for<Count: AxisName>() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(gather_rows.topology()).unwrap();
        let row_len = KeyValue::SIZE * KvHeads::SIZE * Head::SIZE;
        let rows = random(TEST_TOKENS * row_len, 0x5851_f42d_4c95_7f2d);
        let blocks = Blocks::from_rows(&mut device, rows.clone()).await;
        let entries = BlockIndex::SIZE;
        let block = |row: usize, at: usize| (3 + row * 37 + at * 11) % (TEST_TOKENS / Block::SIZE);
        let table = HostTensor::<i32, m![Chunks, Count, BlockIndex]>::from_vec(
            (0..Chunks::SIZE * Count::SIZE * entries)
                .map(|at| {
                    let (window, row) = (at / (Count::SIZE * entries), at / entries % Count::SIZE);
                    block(row, window * entries + at % entries) as i32 * <Blocks>::BLOCK_STRIDE
                })
                .collect::<Vec<_>>(),
        )
        .to_hbm::<Chip, m![Chunks, Count, BlockIndex]>(&mut device.pdma)
        .await
        .unwrap();
        let mut host =
            HostTensor::<bf16, m![Chunks, Count, Cache, KeyValue, KvHeads, Head]>::zero();
        let mut output = host
            .to_hbm::<Chip, m![Chunks, Count, Cache, KeyValue, KvHeads, Head]>(&mut device.pdma)
            .await
            .unwrap();
        let lengths = block_counts::<Count, Chunks>(vec![
            i32::try_from(Chunks::SIZE * Cache::SIZE)
                .unwrap();
            Count::SIZE
        ])
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        launch(
            gather_rows,
            (&mut device, &blocks, &table, &lengths, &mut output),
        )
        .await
        .unwrap();
        output
            .to_host::<m![Chunks, Count, Cache, KeyValue, KvHeads, Head]>(&mut device.pdma)
            .output(&mut host)
            .await
            .unwrap();
        let chunk = host.into_vec();

        for row in 0..Count::SIZE {
            for slot in 0..Chunks::SIZE * Cache::SIZE {
                let pool_row = block(row, slot / Block::SIZE) * Block::SIZE + slot % Block::SIZE;
                let at =
                    (slot / Cache::SIZE * Count::SIZE + row) * Cache::SIZE + slot % Cache::SIZE;
                let got = &chunk[at * row_len..][..row_len];
                let want = &rows[pool_row * row_len..][..row_len];
                assert!(
                    got.iter()
                        .zip(want)
                        .all(|(got, want)| got.to_bits() == want.to_bits()),
                    "row {row} slot {slot} does not hold pool row {pool_row}"
                );
            }
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn gathers_each_block_onto_its_slots() {
        gathers_each_block_onto_its_slots_for::<Eight>().await;
    }

    /// The two valid slots of a row, distinct across rows and in different blocks of eight.
    fn valid(row: usize) -> [usize; 2] {
        let first = 37 + row * 61;
        [first, (first + 259) % Cache::SIZE]
    }

    /// A slot's value as the chunk encodes it.
    fn encoded(slot: usize) -> f32 {
        bf16::from_f32(slot as f32 / Cache::SIZE as f32).to_f32()
    }

    /// Every value row encodes its slot and each row's bias admits two slots in different blocks,
    /// so a row attends to their mean, for eight rows at once and for a row alone.
    #[tokio::test(flavor = "current_thread")]
    async fn attends_only_to_the_valid_slots_of_each_row() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(attend_rows.topology()).unwrap();
        let rest = count(0).to_hbm(&mut device.pdma).await.unwrap();
        let slot = |at: usize| (at / (KeyValue::SIZE * KvHeads::SIZE * Head::SIZE)) % Cache::SIZE;
        let chunk: Vec<bf16> = (0..Eight::SIZE
            * Chunks::SIZE
            * Cache::SIZE
            * KeyValue::SIZE
            * KvHeads::SIZE
            * Head::SIZE)
            .map(|at| {
                let key = at / (KvHeads::SIZE * Head::SIZE) % KeyValue::SIZE == 0;
                bf16::from_f32(if key { 0.0 } else { encoded(slot(at)) })
            })
            .collect();
        let bias: Vec<f32> = (0..Eight::SIZE)
            .flat_map(|row| {
                (0..Chunks::SIZE * Cache::SIZE).map(move |at| {
                    if valid(row).contains(&at) {
                        0.0
                    } else {
                        PADDING_BIAS
                    }
                })
            })
            .collect();
        let query = HostTensor::<bf16, m![Eight, KvHeads, Groups, Head]>::zero()
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let window_hbm =
            HostTensor::<bf16, m![Eight, Chunks, Cache, KeyValue, KvHeads, Head]>::from_vec(
                chunk.clone(),
            )
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let bias_hbm: Bias<Eight> =
            HostTensor::<f32, m![Eight, Chunks, Cache]>::from_vec(bias.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        let mut output = HostTensor::<bf16, m![KvHeads, Eight, Groups, Head]>::zero()
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        launch(
            attend_rows,
            (
                &mut device,
                &Windowed::<KvHeads, Groups, Head, INVERSE_ROOT>::default(),
                &query,
                &window_hbm,
                &bias_hbm,
                &rest,
                &mut output,
            ),
        )
        .await
        .unwrap();
        let batched = output
            .to_host::<m![KvHeads, Eight, Groups, Head]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (at, value) in batched.iter().enumerate() {
            let row = at / (Groups::SIZE * Head::SIZE) % Eight::SIZE;
            let want = valid(row).map(encoded).iter().sum::<f32>() / 2.0;
            assert!(
                (value.to_f32() - want).abs() <= 1.0 / 512.0,
                "index {at} of row {row}: got {}, want {want}",
                value.to_f32()
            );
        }

        let per_row = Chunks::SIZE * Cache::SIZE * KeyValue::SIZE * KvHeads::SIZE * Head::SIZE;
        let query = HostTensor::<bf16, m![Single, KvHeads, Groups, Head]>::zero()
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let window_hbm =
            HostTensor::<bf16, m![Single, Chunks, Cache, KeyValue, KvHeads, Head]>::from_vec(
                chunk[..per_row].to_vec(),
            )
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let bias_hbm = HostTensor::<f32, m![Single, Chunks, Cache]>::from_vec(
            bias[..Chunks::SIZE * Cache::SIZE].to_vec(),
        )
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        let mut output = HostTensor::<bf16, m![KvHeads, Single, Groups, Head]>::zero()
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        launch(
            attend_row,
            (
                &mut device,
                &query,
                &window_hbm,
                &bias_hbm,
                &rest,
                &mut output,
            ),
        )
        .await
        .unwrap();
        let alone = output
            .to_host::<m![KvHeads, Single, Groups, Head]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let batched_row: Vec<bf16> = batched
            .chunks_exact(Eight::SIZE * Groups::SIZE * Head::SIZE)
            .flat_map(|head| head[..Groups::SIZE * Head::SIZE].to_vec())
            .collect();
        assert_eq!(alone, batched_row, "one row alone differs from row 0");
    }

    /// A history whose chunks hold the given constants, admitting the slots of the ones named.
    /// Chunks past the lists hold nothing and admit nothing, so a caller need only describe the
    /// chunks it cares about however wide the envelope grows.
    async fn constant_history(values: &[f32], admitted: &[bool]) -> Vec<bf16> {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(attend_rows.topology()).unwrap();
        let rest = count(admitted.len() - 1)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let query = HostTensor::<bf16, m![Eight, KvHeads, Groups, Head]>::zero()
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let chunk =
            HostTensor::<bf16, m![Eight, Chunks, Cache, KeyValue, KvHeads, Head]>::from_vec(
                (0..Eight::SIZE
                    * Chunks::SIZE
                    * Cache::SIZE
                    * KeyValue::SIZE
                    * KvHeads::SIZE
                    * Head::SIZE)
                    .map(|at| {
                        let key = at / (KvHeads::SIZE * Head::SIZE) % KeyValue::SIZE == 0;
                        let w = at / (Cache::SIZE * KeyValue::SIZE * KvHeads::SIZE * Head::SIZE)
                            % Chunks::SIZE;
                        bf16::from_f32(if key {
                            0.0
                        } else {
                            *values.get(w).unwrap_or(&0.0)
                        })
                    }),
            )
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let bias: Bias<Eight> = HostTensor::<f32, m![Eight, Chunks, Cache]>::from_vec(
            (0..Eight::SIZE).flat_map(|row| {
                (0..Chunks::SIZE * Cache::SIZE).map(move |at| {
                    let slot = at % Cache::SIZE;
                    if *admitted.get(at / Cache::SIZE).unwrap_or(&false)
                        && slot % 8 == 0
                        && slot / 8 <= row
                    {
                        0.0
                    } else {
                        PADDING_BIAS
                    }
                })
            }),
        )
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        let mut output = HostTensor::<bf16, m![KvHeads, Eight, Groups, Head]>::zero()
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        launch(
            attend_rows,
            (
                &mut device,
                &Windowed::<KvHeads, Groups, Head, INVERSE_ROOT>::default(),
                &query,
                &chunk,
                &bias,
                &rest,
                &mut output,
            ),
        )
        .await
        .unwrap();
        output
            .to_host::<m![KvHeads, Eight, Groups, Head]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec()
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_row_attends_to_valid_slots_in_its_second_window() {
        let attended = constant_history(&[0.125, 0.625], &[false, true]).await;

        for value in attended {
            assert!(
                (value.to_f32() - 0.625).abs() <= 1.0 / 512.0,
                "got {}, want 0.625",
                value.to_f32()
            );
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn equally_weighted_windows_attend_to_the_mean_of_their_constants() {
        let attended = constant_history(&[0.125, 0.625], &[true, true]).await;

        for value in attended {
            assert!(
                (value.to_f32() - 0.375).abs() <= 1.0 / 512.0,
                "got {}, want 0.375",
                value.to_f32()
            );
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn attends_full_history() {
        let attended = constant_history(&[0.125, 0.25, 0.5, 0.625], &[true; 4]).await;
        assert!(
            attended
                .iter()
                .all(|value| (value.to_f32() - 0.375).abs() <= 1.0 / 512.0)
        );
    }

    #[device(chip = 1)]
    fn attend_prefix<Count: AxisName, Depth: AxisName>(
        device: &mut Device,
        query: &PrefixQuery<Count>,
        blocks: &Blocks,
        table: &HbmTensor<i32, Chip, m![Depth, Count, BlockIndex]>,
        lengths: &BlockCounts<Count, Depth>,
        bias: &Bias<Count, Depth>,
        (rest, live): (&Value, &Value),
        output: &mut PrefixOutput<Count>,
    ) {
        let query: super::Heads<Count, KvHeads, PrefixGroups, Head> = query.to_dm(&mut device.tdma);
        let history = Paged {
            blocks,
            table,
            lengths,
            scales: &(),
            wrap: (),
        };
        Windowed::<KvHeads, PrefixGroups, Head, INVERSE_ROOT>::default()
            .forward(
                &mut *device,
                (&query, &history, bias, rest, Live::<(), Count>::read(live)),
            )
            .view()
            .to_hbm_view(&mut device.tdma, output.view_mut());
    }

    /// Rows `0..live` of `Count` attend over the prefix of the bank each names in a history of
    /// `Depth` windows, through a scratch poisoned past every prefix; later rows are not live and
    /// write nothing checked.
    async fn prefix<Count: AxisName, Depth: AxisName>(rows: &[usize], live: usize) {
        const LENGTHS: [usize; 8] = [0, 1, 63, 64, 65, 511, 512, 513];
        const SHORTER: [usize; 8] = [0, 0, 1, 63, 64, 65, 511, 512];
        assert_eq!(rows.len(), Count::SIZE);
        let query = |bank: usize, group: usize| {
            bf16::from_f32(0.5 + bank as f32 / 8.0 + group as f32 / 4.0)
        };
        let key = |token: usize, head: usize| {
            bf16::from_f32(((token * 3 + head) % 17) as f32 / 32.0 - 0.25)
        };
        let value = |bank: usize, token: usize, head: usize, channel: usize| {
            bf16::from_f32(
                (((token * 3 + head * 5 + bank * 7) % 29) as f32 - 14.0)
                    / 16.0
                    / (1 << (channel % 4)) as f32,
            )
        };
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(attend_prefix.topology()).unwrap();
        let rest = count(1).to_hbm(&mut device.pdma).await.unwrap();
        let live_rows = count(live).to_hbm(&mut device.pdma).await.unwrap();
        let blocks = Blocks::from_rows(
            &mut device,
            (0..TEST_TOKENS * KeyValue::SIZE * KvHeads::SIZE * Head::SIZE)
                .map(|at| {
                    let channel = at % Head::SIZE;
                    let head = at / Head::SIZE % KvHeads::SIZE;
                    let kind = at / (Head::SIZE * KvHeads::SIZE) % KeyValue::SIZE;
                    let position = at / (Head::SIZE * KvHeads::SIZE * KeyValue::SIZE);
                    let bank = position / 1024;
                    let token = position % 1024;
                    if token >= LENGTHS[bank] {
                        bf16::from_f32(match (token + channel + kind) % 4 {
                            0 => f32::NAN,
                            1 => -f32::NAN,
                            2 => f32::INFINITY,
                            _ => f32::NEG_INFINITY,
                        })
                    } else if kind == 0 {
                        key(token, head)
                    } else {
                        value(bank, token, head, channel)
                    }
                })
                .collect(),
        )
        .await;
        let table = HostTensor::<i32, m![Depth, Count, BlockIndex]>::from_vec(
            (0..Depth::SIZE).flat_map(|window| {
                rows.iter().flat_map(move |&row| {
                    (0..BlockIndex::SIZE).map(move |entry| {
                        let at = window * BlockIndex::SIZE + entry;
                        (((row % 8) * 1024 / Block::SIZE + at % (1024 / Block::SIZE)) as i32)
                            * <Blocks>::BLOCK_STRIDE
                    })
                })
            }),
        )
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        let query_hbm = HostTensor::<bf16, m![Count, KvHeads, PrefixGroups, Head]>::from_vec(
            rows.iter().flat_map(|&row| {
                (0..KvHeads::SIZE * PrefixGroups::SIZE * Head::SIZE)
                    .map(move |at| query(row % 8, at / Head::SIZE % PrefixGroups::SIZE))
            }),
        )
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        let mut output =
            HostTensor::<bf16, m![KvHeads, Count, PrefixGroups, Head]>::from_vec(repeat_n(
                bf16::from_f32(f32::NAN),
                KvHeads::SIZE * Count::SIZE * PrefixGroups::SIZE * Head::SIZE,
            ))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        for (phase, admitted) in [LENGTHS, SHORTER, LENGTHS].into_iter().enumerate() {
            let lengths = rows
                .iter()
                .map(|&row| if row < 8 { admitted[row] } else { 0 })
                .collect::<Vec<_>>();
            // The first launch fills scratch with poisoned tails; later launches reuse it
            // while the gather writes only the shorter readable prefix.
            let lengths_hbm: BlockCounts<Count, Depth> =
                block_counts::<Count, Depth>(lengths.iter().map(|&length| {
                    if phase == 0 {
                        TEST_TOKENS as i32
                    } else {
                        length as i32
                    }
                }))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let bias = HostTensor::<f32, m![Count, Depth, Cache]>::from_vec(
                lengths.iter().flat_map(|&length| {
                    (0..Depth::SIZE * Cache::SIZE)
                        .map(move |token| if token < length { 0.0 } else { PADDING_BIAS })
                }),
            )
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
            launch(
                attend_prefix,
                (
                    &mut device,
                    &query_hbm,
                    &blocks,
                    &table,
                    &lengths_hbm,
                    &bias,
                    (&rest, &live_rows),
                    &mut output,
                ),
            )
            .await
            .unwrap();
            let got = output
                .to_host::<m![KvHeads, Count, PrefixGroups, Head]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            for (row, (&bank, &length)) in rows.iter().zip(&lengths).enumerate().take(live) {
                for head in 0..KvHeads::SIZE {
                    for group in 0..PrefixGroups::SIZE {
                        let scores = (0..length)
                            .map(|token| {
                                Head::SIZE as f64
                                    * query(bank % 8, group).to_f32() as f64
                                    * key(token, head).to_f32() as f64
                                    * f32::from_bits(INVERSE_ROOT as u32) as f64
                            })
                            .collect::<Vec<_>>();
                        let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        let weights = scores
                            .iter()
                            .map(|&score| (score - top).exp())
                            .collect::<Vec<_>>();
                        let total = weights.iter().sum::<f64>();
                        for channel in 0..Head::SIZE {
                            let want = if length == 0 {
                                0.0
                            } else {
                                weights
                                    .iter()
                                    .enumerate()
                                    .map(|(token, &weight)| {
                                        weight
                                            * value(bank % 8, token, head, channel).to_f32() as f64
                                    })
                                    .sum::<f64>()
                                    / total
                            } as f32;
                            let at = ((head * Count::SIZE + row) * PrefixGroups::SIZE + group)
                                * Head::SIZE
                                + channel;
                            let found = got[at].to_f32();
                            assert!(
                                found.is_finite(),
                                "phase {phase} row {row} length {length} index {at}: {found}"
                            );
                            let error = (found - want).abs();
                            assert!(
                                error <= 1.0 / 128.0,
                                "phase {phase} row {row} length {length} index {at}: {found} != {want}"
                            );
                            if length == 0 {
                                assert_eq!(found, 0.0, "empty row {row}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn wide_rows_preserve_prefix() {
        prefix::<Rows<64>, Chunks>(
            &(0..64)
                .map(|row| (row * 3 + row / 8) % 8)
                .collect::<Vec<_>>(),
            64,
        )
        .await;
    }

    /// Fewer live rows than the envelope attend each of their groups alone: five whole groups and
    /// a partial sixth, every prefix length among them, attend as when every row is live.
    #[tokio::test(flavor = "current_thread")]
    async fn live_groups_attend_alone() {
        prefix::<Rows<64>, Chunks>(
            &(0..64)
                .map(|row| (row * 3 + row / 8) % 8)
                .collect::<Vec<_>>(),
            41,
        )
        .await;
    }

    /// One live row of a wide envelope attends alone.
    #[tokio::test(flavor = "current_thread")]
    async fn one_live_row_attends_alone() {
        prefix::<Rows<64>, Chunks>(&(0..64).map(|row| (row + 7) % 8).collect::<Vec<_>>(), 1).await;
    }

    /// A history of a few windows holds each window's terms until one merge: five whole groups and
    /// a partial sixth, every prefix length among them, attend as the host does.
    #[tokio::test(flavor = "current_thread")]
    async fn live_groups_hold_their_windows() {
        prefix::<Rows<64>, Windows>(
            &(0..64)
                .map(|row| (row * 3 + row / 8) % 8)
                .collect::<Vec<_>>(),
            41,
        )
        .await;
    }

    /// One live row holds its windows' terms alone.
    #[tokio::test(flavor = "current_thread")]
    async fn one_live_row_holds_its_windows() {
        prefix::<Rows<64>, Windows>(&(0..64).map(|row| (row + 7) % 8).collect::<Vec<_>>(), 1).await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn wide_partial_rows_zero() {
        prefix::<Rows<64>, Chunks>(
            &(0..64)
                .map(|row| if row < 47 { (row * 5 + row / 8) % 8 } else { 8 })
                .collect::<Vec<_>>(),
            64,
        )
        .await;
    }

    /// Four row groups fill one pass: the batch attends in a single trip.
    #[tokio::test(flavor = "current_thread")]
    async fn full_pass_rows_preserve_prefix() {
        prefix::<Rows<32>, Chunks>(
            &(0..32)
                .map(|row| (row * 3 + row / 8) % 8)
                .collect::<Vec<_>>(),
            32,
        )
        .await;
    }

    /// A pass whose rows all read empty histories writes zeros beside a pass that attends.
    #[tokio::test(flavor = "current_thread")]
    async fn empty_pass_rows_zero() {
        prefix::<Rows<64>, Chunks>(
            &(0..64)
                .map(|row| if row < 32 { (row * 5 + row / 8) % 8 } else { 8 })
                .collect::<Vec<_>>(),
            64,
        )
        .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn tail_preserves_prefix() {
        prefix::<Eight, Chunks>(&[0, 1, 2, 3, 4, 5, 6, 7], 8).await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn rows_remain_independent() {
        prefix::<Sixteen, Chunks>(&[7, 6, 5, 4, 3, 2, 1, 0, 8, 9, 10, 11, 12, 13, 14, 15], 16)
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn empty_history_zero() {
        prefix::<Single, Chunks>(&[0], 1).await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn preserves_short_history() {
        let lease = DEVICE.lock().await;
        let mut device = Device::new(attend_one.topology()).unwrap();
        let query = HostTensor::<bf16, m![Eight, KvHeads, Groups, Head]>::zero()
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let chunk =
            HostTensor::<bf16, m![Eight, Single, Cache, KeyValue, KvHeads, Head]>::from_vec(
                (0..Eight::SIZE * Cache::SIZE * KeyValue::SIZE * KvHeads::SIZE * Head::SIZE).map(
                    |at| {
                        bf16::from_f32(if at / (KvHeads::SIZE * Head::SIZE) % KeyValue::SIZE == 0 {
                            0.0
                        } else {
                            0.375
                        })
                    },
                ),
            )
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let bias = HostTensor::<f32, m![Eight, Single, Cache]>::from_vec(
            (0..Eight::SIZE).flat_map(|row| {
                (0..Cache::SIZE).map(move |slot| {
                    if slot % 8 == 0 && slot / 8 <= row {
                        0.0
                    } else {
                        PADDING_BIAS
                    }
                })
            }),
        )
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        let mut output = HostTensor::<bf16, m![KvHeads, Eight, Groups, Head]>::zero()
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        launch(
            attend_one,
            (&mut device, &query, &chunk, &bias, &mut output),
        )
        .await
        .unwrap();
        let one = output
            .to_host::<m![KvHeads, Eight, Groups, Head]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        drop(device);
        drop(lease);
        let four = constant_history(&[0.375, f32::NAN, f32::NAN, f32::NAN], &[true]).await;
        assert!(
            one.iter().zip(&four).all(|(one, four)| one == four),
            "the envelope changed attention"
        );
        assert!(
            four.iter()
                .all(|value| (value.to_f32() - 0.375).abs() <= 1.0 / 512.0)
        );
    }

    #[device(chip = 1)]
    fn attend_one(
        device: &mut Device,
        query: &Query<Eight>,
        source: &Source<Eight, Single>,
        bias: &Bias<Eight, Single>,
        output: &mut Output<Eight>,
    ) {
        let chunk: DmTensor<
            bf16,
            Chip,
            super::Cluster<Eight>,
            super::CacheSlices<Eight, KvHeads>,
            m![{ super::RowGroups<Eight> }, KeyValue, { super::ShardKeys<KvHeads> }, Head],
        > = source.to_dm(&mut device.tdma);
        let query: super::Query<Eight, KvHeads, Groups, Head> = query.to_dm(&mut device.tdma);
        let bias: super::Bias<Eight, KvHeads, Groups> = bias.to_dm(&mut device.tdma);
        <super::Partition<Eight, KvHeads, Groups, Head, INVERSE_ROOT> as Partition>::windows::<
            _,
            super::Rest<Single>,
        >(
            device,
            0,
            (),
            (
                |_: &mut Device, _| (),
                |device: &mut Device, ()| {
                    let mut values = DmTensor::new();
                    super::transposes::<Eight, KvHeads, Groups, Head, INVERSE_ROOT, _>(
                        device,
                        chunk.view(),
                        &mut values,
                    );
                    super::attend::<Eight, KvHeads, Groups, Head, INVERSE_ROOT>(
                        device,
                        &query,
                        (chunk.view(), &values),
                        (&bias, None),
                    )
                },
            ),
            |_| (),
        )
        .view()
        .to_hbm_view(&mut device.tdma, output.view_mut());
    }

    /// The weights of a row sum to one over the slots its bias admits, so a chunk holding one
    /// constant everywhere attends to that constant no matter how many slots a row admits. Each
    /// row here admits a different number, one slot in each of the shards up to its own index, so
    /// a normalisation counting shards rather than admitted slots reads a different wrong value in
    /// every row, and every row leaves some shard admitting nothing at all. The constant makes it
    /// independent of which slots those are, which is what this pins that
    /// [`attends_only_to_the_valid_slots_of_each_row`] does not.
    #[tokio::test(flavor = "current_thread")]
    async fn a_constant_window_attends_to_its_constant_however_many_slots_a_row_admits() {
        const VALUE: f32 = 0.375;
        // A shard holds 64 consecutive slots, so slot `at` sits in shard `at / 64`; row `r`
        // admits the first slot of each of the shards up to its own index, and none beyond.
        let admits = |row: usize, at: usize| at % 64 == 0 && at / 64 <= row;
        let admitted = |row: usize| row + 1;

        let _lease = DEVICE.lock().await;
        let mut device = Device::new(attend_rows.topology()).unwrap();
        let rest = count(0).to_hbm(&mut device.pdma).await.unwrap();
        let chunk: Vec<bf16> = (0..Eight::SIZE
            * Chunks::SIZE
            * Cache::SIZE
            * KeyValue::SIZE
            * KvHeads::SIZE
            * Head::SIZE)
            .map(|at| {
                let key = at / (KvHeads::SIZE * Head::SIZE) % KeyValue::SIZE == 0;
                bf16::from_f32(if key { 0.0 } else { VALUE })
            })
            .collect();
        let bias: Vec<f32> = (0..Eight::SIZE)
            .flat_map(|row| {
                (0..Chunks::SIZE * Cache::SIZE)
                    .map(move |at| if admits(row, at) { 0.0 } else { PADDING_BIAS })
            })
            .collect();
        let query = HostTensor::<bf16, m![Eight, KvHeads, Groups, Head]>::zero()
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let window_hbm =
            HostTensor::<bf16, m![Eight, Chunks, Cache, KeyValue, KvHeads, Head]>::from_vec(chunk)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        let bias_hbm: Bias<Eight> = HostTensor::<f32, m![Eight, Chunks, Cache]>::from_vec(bias)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let mut output = HostTensor::<bf16, m![KvHeads, Eight, Groups, Head]>::zero()
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        launch(
            attend_rows,
            (
                &mut device,
                &Windowed::<KvHeads, Groups, Head, INVERSE_ROOT>::default(),
                &query,
                &window_hbm,
                &bias_hbm,
                &rest,
                &mut output,
            ),
        )
        .await
        .unwrap();
        let attended = output
            .to_host::<m![KvHeads, Eight, Groups, Head]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (at, value) in attended.iter().enumerate() {
            let row = at / (Groups::SIZE * Head::SIZE) % Eight::SIZE;
            assert!(
                (value.to_f32() - VALUE).abs() <= 1.0 / 512.0,
                "index {at} of row {row} admitting {} slots: got {}, want {VALUE}",
                admitted(row),
                value.to_f32()
            );
        }
    }

    /// A history whose slots carry chosen scores and values, run against a one-pass stable
    /// softmax over exactly the slots each row admits.
    ///
    /// [`constant_history`] pins the query to zero, so every slot scores the same and every
    /// window's log-sum-exp is equal: nothing there can tell a carry that weighs its two terms
    /// from one that averages them. Scoring the slots is what makes the running maximum and the
    /// two shares observable, and the oracle recomputes the whole attention in one pass so it
    /// shares no structure with the shard-local, folded, then merged form under test.
    ///
    /// The query is one everywhere, so a slot's logit is the sum of its key over `Head` times
    /// the leaf's `SCALE`; the oracle reads the keys back as `bf16` so the two agree on the input
    /// rather than on the arithmetic.
    ///
    /// Only the first `windows` are scored, the rest of the envelope holding nothing, so a caller
    /// need only describe the windows it asks for however wide the envelope grows.
    async fn scored_history<const SCALE: usize>(
        windows: usize,
        score: impl Fn(usize, usize) -> f32 + Copy,
        admits: impl Fn(usize, usize) -> bool + Copy,
    ) -> (Vec<bf16>, Vec<f32>) {
        const ROOT: f32 = 11.313_708;
        // Multiples of a sixteenth, and a per-dimension factor that is a power of two, so every
        // value the device sees is exact in `bf16` and a head-dimension mix-up cannot hide.
        let value = |w: usize, slot: usize, d: usize| {
            ((w * 7 + slot * 3) % 16) as f32 / 16.0 / (1 << (d % 4)) as f32
        };
        let key = |w: usize, slot: usize| bf16::from_f32(score(w, slot) / ROOT);
        let logit = |w: usize, slot: usize| {
            Head::SIZE as f32 * key(w, slot).to_f32() * f32::from_bits(SCALE as u32)
        };

        let _lease = DEVICE.lock().await;
        let mut device = Device::new(attend_rows.topology()).unwrap();
        let rest = count(windows - 1).to_hbm(&mut device.pdma).await.unwrap();
        let query = HostTensor::<bf16, m![Eight, KvHeads, Groups, Head]>::from_vec(
            repeat_n(
                bf16::from_f32(1.0),
                Eight::SIZE * KvHeads::SIZE * Groups::SIZE * Head::SIZE,
            )
            .collect::<Vec<_>>(),
        )
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        let chunk =
            HostTensor::<bf16, m![Eight, Chunks, Cache, KeyValue, KvHeads, Head]>::from_vec(
                (0..Eight::SIZE
                    * Chunks::SIZE
                    * Cache::SIZE
                    * KeyValue::SIZE
                    * KvHeads::SIZE
                    * Head::SIZE)
                    .map(|at| {
                        let d = at % Head::SIZE;
                        let is_key = at / (KvHeads::SIZE * Head::SIZE) % KeyValue::SIZE == 0;
                        let slot = at / (KeyValue::SIZE * KvHeads::SIZE * Head::SIZE) % Cache::SIZE;
                        let w = at / (Cache::SIZE * KeyValue::SIZE * KvHeads::SIZE * Head::SIZE)
                            % Chunks::SIZE;
                        if w >= windows {
                            bf16::from_f32(0.0)
                        } else if is_key {
                            key(w, slot)
                        } else {
                            bf16::from_f32(value(w, slot, d))
                        }
                    }),
            )
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let bias: Bias<Eight> = HostTensor::<f32, m![Eight, Chunks, Cache]>::from_vec(
            (0..Eight::SIZE).flat_map(|row| {
                (0..Chunks::SIZE * Cache::SIZE)
                    .map(move |at| if admits(row, at) { 0.0 } else { PADDING_BIAS })
            }),
        )
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        let mut output = HostTensor::<bf16, m![KvHeads, Eight, Groups, Head]>::zero()
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        launch(
            attend_rows,
            (
                &mut device,
                &Windowed::<KvHeads, Groups, Head, SCALE>::default(),
                &query,
                &chunk,
                &bias,
                &rest,
                &mut output,
            ),
        )
        .await
        .unwrap();
        let attended = output
            .to_host::<m![KvHeads, Eight, Groups, Head]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();

        let mut want = vec![0.0f32; KvHeads::SIZE * Eight::SIZE * Groups::SIZE * Head::SIZE];
        for row in 0..Eight::SIZE {
            let live: Vec<(usize, usize)> = (0..windows)
                .flat_map(|w| (0..Cache::SIZE).map(move |slot| (w, slot)))
                .filter(|(w, slot)| admits(row, w * Cache::SIZE + slot))
                .collect();
            let top = live
                .iter()
                .map(|&(w, slot)| logit(w, slot))
                .fold(f32::NEG_INFINITY, f32::max);
            let shares: Vec<f32> = live
                .iter()
                .map(|&(w, slot)| (logit(w, slot) - top).exp())
                .collect();
            let total: f32 = shares.iter().sum();
            for d in 0..Head::SIZE {
                let attended: f32 = live
                    .iter()
                    .zip(&shares)
                    .map(|(&(w, slot), share)| share * value(w, slot, d))
                    .sum::<f32>()
                    / total;
                for head in 0..KvHeads::SIZE {
                    for group in 0..Groups::SIZE {
                        want[((head * Eight::SIZE + row) * Groups::SIZE + group) * Head::SIZE
                            + d] = attended;
                    }
                }
            }
        }
        (attended, want)
    }

    /// The device's attention against the oracle's, to the precision `bf16` values and an `f32`
    /// accumulation leave.
    fn agrees(what: &str, got: &[bf16], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        let mut worst = 0.0f32;
        let mut at = 0;
        for (i, (got, want)) in got.iter().zip(want).enumerate() {
            assert!(got.to_f32().is_finite(), "{what}: index {i} is {got:?}");
            let error = (got.to_f32() - want).abs();
            if error > worst {
                worst = error;
                at = i;
            }
        }
        assert!(
            worst <= 1.0 / 128.0,
            "{what}: index {at} differs by {worst}, got {} want {}",
            got[at].to_f32(),
            want[at]
        );
    }

    /// Windows whose slots score differently: a carry that weighed both terms equally, or that
    /// dropped one, reads a different number here where an equally weighted history cannot tell.
    #[tokio::test(flavor = "current_thread")]
    async fn unequal_window_scores_weigh_each_window_by_its_own_statistic() {
        let scores = [-1.5f32, 2.25, 0.5, -0.75];
        let (got, want) = scored_history::<INVERSE_ROOT>(
            4,
            |w, slot| scores[w] + (slot % 5) as f32 / 4.0,
            |_, _| true,
        )
        .await;
        agrees("unequal window scores", &got, &want);
    }

    /// A maximum that rises with every window, so every fold has to advance it. A carry that kept
    /// its first maximum would exponentiate against a stale one and under-weight the later
    /// windows.
    #[tokio::test(flavor = "current_thread")]
    async fn a_rising_maximum_advances_the_carry() {
        let (got, want) = scored_history::<INVERSE_ROOT>(
            4,
            |w, slot| 2.0 * w as f32 + (slot % 3) as f32 / 8.0,
            |_, _| true,
        )
        .await;
        agrees("rising maximum", &got, &want);
    }

    /// The same history in reverse, so every fold has to keep the maximum it already holds. A
    /// carry that took the new window's maximum unconditionally would rescale the whole history
    /// by the wrong factor.
    #[tokio::test(flavor = "current_thread")]
    async fn a_falling_maximum_keeps_the_carry() {
        let (got, want) = scored_history::<INVERSE_ROOT>(
            4,
            |w, slot| 6.0 - 2.0 * w as f32 + (slot % 3) as f32 / 8.0,
            |_, _| true,
        )
        .await;
        agrees("falling maximum", &got, &want);
    }

    /// A window in the middle of the history admitting nothing at all: its shards score only
    /// padding, so it must take no share, and the windows around it must keep theirs.
    #[tokio::test(flavor = "current_thread")]
    async fn a_masked_middle_window_takes_no_share() {
        let (got, want) = scored_history::<INVERSE_ROOT>(
            4,
            |w, slot| w as f32 - 1.0 + (slot % 7) as f32 / 4.0,
            |_, at| at / Cache::SIZE != 1,
        )
        .await;
        agrees("masked middle window", &got, &want);
    }

    /// The same for the last window, which is the one a carry that folds after the loop rather
    /// than inside it would silently keep.
    #[tokio::test(flavor = "current_thread")]
    async fn a_masked_trailing_window_takes_no_share() {
        let (got, want) = scored_history::<INVERSE_ROOT>(
            4,
            |w, slot| 1.5 * w as f32 + (slot % 7) as f32 / 4.0,
            |_, at| at / Cache::SIZE != 3,
        )
        .await;
        agrees("masked trailing window", &got, &want);
    }

    /// Histories past the 2048 slots four windows hold, one length per row so no two rows span
    /// the same count of windows: the boundary itself, the slot after it, and a history twice as
    /// long, all in one launch. A per-row envelope taken from any one row, or a fold that stops
    /// at the fourth window, reads the wrong number here for every other row.
    #[tokio::test(flavor = "current_thread")]
    async fn histories_past_four_windows_attend_to_their_own_lengths() {
        const LENGTHS: [usize; 8] = [2048, 2049, 4096, 2560, 3000, 513, 512, 1];
        let windows = LENGTHS.iter().max().unwrap().div_ceil(Cache::SIZE);
        let (got, want) = scored_history::<INVERSE_ROOT>(
            windows,
            |w, slot| (slot % 13) as f32 / 4.0 - w as f32 / 2.0,
            |row, at| at < LENGTHS[row],
        )
        .await;
        agrees("histories past four windows", &got, &want);
    }

    /// Rows admitting different counts of scored slots, spread so that every row leaves some
    /// shard of some window admitting nothing. This is what a normalisation counting shards or
    /// windows rather than admitted slots cannot get right for every row at once.
    #[tokio::test(flavor = "current_thread")]
    async fn rows_admitting_different_counts_attend_to_their_own_slots() {
        let (got, want) = scored_history::<INVERSE_ROOT>(
            4,
            |w, slot| (slot % 11) as f32 / 3.0 - w as f32,
            |row, at| at % (row + 3) == 0 && (at / 64) % 3 != 2,
        )
        .await;
        agrees("unequal admitted counts", &got, &want);
    }

    /// Two leaves that differ only in their scale attend differently, each as an oracle that
    /// applies its own scale says.
    #[tokio::test(flavor = "current_thread")]
    async fn scores_scale_by_the_leaf_scale() {
        const DOUBLED: usize = (FRAC_1_SQRT_2 / 4.0).to_bits() as usize;
        // The score follows the value: a sharper softmax moves the average to larger values.
        let score = |w: usize, slot: usize| ((w * 7 + slot * 3) % 16) as f32 / 4.0 - w as f32 / 2.0;
        let (got, want) = scored_history::<INVERSE_ROOT>(2, score, |_, _| true).await;
        agrees("head scale", &got, &want);
        let (sharper, sharper_want) = scored_history::<DOUBLED>(2, score, |_, _| true).await;
        agrees("doubled scale", &sharper, &sharper_want);
        assert!(
            want.iter()
                .zip(&sharper_want)
                .any(|(one, other)| (one - other).abs() > 4.0 / 128.0),
            "the two scales must attend apart by more than the agreement tolerance"
        );
    }
}
