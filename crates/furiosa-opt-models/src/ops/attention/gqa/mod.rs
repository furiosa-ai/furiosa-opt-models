use crate::Op;
use crate::tensor::{axis::Lanes, live::Live, mapping::Rows as Layout};
use furiosa_opt_std::prelude::*;

use super::{Attention, Chip, KeyValue, TileGroups};
use crate::ops::index::table::{BlockCounts, BlockIndex};
use crate::specialize::attention::Strategy;
use crate::tensor::cache::{Block, Blocks};
use crate::tensor::scalar::{F32, Value};
use std::marker::PhantomData;

use crate::ops::linear;
pub use paged::{Pages, Read};

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
    m![Cache % 64],
    Cache,
    8,
    SCALE,
>;

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

/// A head's eight shards are consecutive slices so the inter-slice reducer can combine them.
/// The four lanes stay within each head's processing element, each occupying its own eight slices.
type CacheSlices<Rows, KvHeads> = m![KvHeads, { Layout<Rows>::Lanes } # 8 % 4, Cache / 64];

/// All eight shard slices hold a copy of the same reduced statistic.
type Agreed<Rows, KvHeads> = m![KvHeads, { Layout<Rows>::Lanes } # 8 % 4, 8];

/// One slice per row and head holds the merged attention.
type Merged<Rows, KvHeads> = m![KvHeads, { Layout<Rows>::Lanes } # 8 % 4, 1 # 8];

type RowGroups<Rows> = linear::Groups<Rows>;

/// The row groups of an `N`-row tile; at 32 rows the four independent row groups that fill the
/// 64 KiB register file, the most one pass attends.
type TileRows<Rows, const N: usize = 32> =
    m![{ RowGroups<Rows> } % ({ Symbol<TileGroups<Rows, N>> })];

/// The `N`-row tile a row group belongs to.
type Tiles<Rows, const N: usize = 32> = m![{ RowGroups<Rows> } / ({ Symbol<TileGroups<Rows, N>> })];

/// Pass `.0` over an envelope's `N`-row tiles: the row groups of that tile, attended together.
#[derive(Clone, Copy)]
pub struct Pass<const N: usize>(pub usize);

/// The cache rows a step reads, 64 whole rows of one head per slice, each row its token's key
/// and value; `Count` row groups of them, every group unless one is named.
type Chunk<Rows, KvHeads, Head, Count = RowGroups<Rows>> = DmTensor<
    bf16,
    Chip,
    Cluster<Rows>,
    CacheSlices<Rows, KvHeads>,
    m![{ Count }, KeyValue, Cache % 64, Head],
>;

/// Every window of every row's history, for rows that carry it outright; in DRAM, since a slice's
/// sixteen-chunk data memory would otherwise bound the windows a history spans.
pub(super) type Envelope<Rows, KvHeads, Head, Windows> =
    HbmTensor<bf16, Chip, m![Rows, Windows, Cache, KeyValue, KvHeads, Head]>;

/// One chunk as the gather writes it, needing no repartition: each table entry's token block on its
/// shard and its heads over their eight slices, the cells of [`Chunk`].
type InDm<Rows, KvHeads, Head, Count = RowGroups<Rows>> = DmTensor<
    bf16,
    Chip,
    Cluster<Rows>,
    m![KvHeads, { Layout<Rows>::Lanes } # 8 % 4, BlockIndex / 4],
    m![{ Count }, KeyValue, BlockIndex % 4, Block, Head],
>;

/// One chunk's entries of a step's table, the only shape a gather takes an index in.
type Entries<Rows> = HbmTensor<i32, Chip, m![Rows, BlockIndex]>;

/// One chunk's entries of the row groups of one `N`-row tile, their lanes in order.
type TileEntries<Rows, const N: usize> =
    HbmTensor<i32, Chip, m![{ TileRows<Rows, N> }, { Layout<Rows>::Lanes }, BlockIndex]>;

/// A layer's cache blocks and the table of blocks each row reads.
pub struct Paged<'l, Rows: AxisName, KvHeads: AxisName, Head: AxisName, Windows: AxisName> {
    pub blocks: &'l Blocks<KvHeads, Head>,
    pub table: &'l HbmTensor<i32, Chip, m![Rows, Windows, BlockIndex]>,
    pub lengths: &'l BlockCounts<Rows, Windows>,
}

impl<Rows: AxisName, KvHeads: AxisName, Head: AxisName, Windows: AxisName> Op<usize>
    for Envelope<Rows, KvHeads, Head, Windows>
{
    type Output = Chunk<Rows, KvHeads, Head>;

    fn forward(&self, device: &mut Device, at: usize) -> Self::Output {
        let mut chunk = Chunk::<Rows, KvHeads, Head>::new();
        self.view()
            .tile::<m![Windows], 1, m![Rows, Windows = 1 # Windows, Cache, KeyValue, KvHeads, Head]>(at)
            .to_dm_view(&mut device.tdma, chunk.view_mut());
        chunk
    }
}

impl<Rows: AxisName, KvHeads: AxisName, Head: AxisName, Windows: AxisName, const N: usize>
    Op<(usize, Pass<N>)> for Envelope<Rows, KvHeads, Head, Windows>
{
    type Output = Chunk<Rows, KvHeads, Head, TileRows<Rows, N>>;

    /// Window `at` of the row groups of `pass` alone.
    fn forward(&self, device: &mut Device, (at, pass): (usize, Pass<N>)) -> Self::Output {
        let mut chunk = Chunk::<Rows, KvHeads, Head, TileRows<Rows, N>>::new();
        self.view()
            .tile::<m![Windows], 1, m![Rows, Windows = 1 # Windows, Cache, KeyValue, KvHeads, Head]>(at)
            .tile::<Tiles<Rows, N>, 1, m![
                { Tiles<Rows, N> } = 1 # ({ Tiles<Rows, N> }), { TileRows<Rows, N> },
                { Layout<Rows>::Lanes }, Windows = 1 # Windows, Cache, KeyValue, KvHeads, Head
            ]>(pass.0)
            .to_dm_view(&mut device.tdma, chunk.view_mut());
        chunk
    }
}

impl<Rows: AxisName, KvHeads: AxisName, Head: AxisName, Windows: AxisName> Op<usize>
    for Paged<'_, Rows, KvHeads, Head, Windows>
{
    type Output = Chunk<Rows, KvHeads, Head>;

    /// A gather takes its index as a whole tensor, so one chunk's entries are copied out of the
    /// table before they address the pool.
    fn forward(&self, device: &mut Device, at: usize) -> Self::Output {
        let mut entries = Entries::<Rows>::new();
        self.table
            .view()
            .tile::<m![Windows], 1, m![Rows, Windows = 1 # Windows, BlockIndex]>(at)
            .to_hbm_view(&mut device.tdma, entries.view_mut());
        let mut lengths: HbmTensor<i32, Chip, m![Rows]> = HbmTensor::new();
        self.lengths
            .view()
            .tile::<m![Windows], 1, m![Windows = 1 # Windows, Rows]>(at)
            .to_hbm_view(&mut device.tdma, lengths.view_mut());
        let in_dm: InDm<Rows, KvHeads, Head> = self
            .blocks
            .tokens
            .gather::<m![1], m![Block, KeyValue, KvHeads, Head]>()
            .by_byte_offsets(&entries)
            .prefix::<m![BlockIndex], _>(&lengths, 0)
            // Every table entry is readable and the bias masks a row's tokens past its prefix.
            .overfetch()
            .to_dm(&mut device.tdma);
        // SAFETY: a shard's four blocks are its 64 consecutive slots, so both nests name the same
        // cells in the same order.
        unsafe { in_dm.reshape() }
    }
}

impl<Rows: AxisName, KvHeads: AxisName, Head: AxisName, Windows: AxisName, const N: usize>
    Op<(usize, Pass<N>)> for Paged<'_, Rows, KvHeads, Head, Windows>
{
    type Output = Chunk<Rows, KvHeads, Head, TileRows<Rows, N>>;

    /// Window `at` of the row groups of `pass` alone: only their entries address the pool.
    fn forward(&self, device: &mut Device, (at, pass): (usize, Pass<N>)) -> Self::Output {
        let mut entries = TileEntries::<Rows, N>::new();
        self.table
            .view()
            .tile::<m![Windows], 1, m![Rows, Windows = 1 # Windows, BlockIndex]>(at)
            .tile::<Tiles<Rows, N>, 1, m![
                { Tiles<Rows, N> } = 1 # ({ Tiles<Rows, N> }), { TileRows<Rows, N> },
                { Layout<Rows>::Lanes }, Windows = 1 # Windows, BlockIndex
            ]>(pass.0)
            .to_hbm_view(&mut device.tdma, entries.view_mut());
        let mut lengths: HbmTensor<i32, Chip, m![{ TileRows<Rows, N> }, { Layout<Rows>::Lanes }]> =
            HbmTensor::new();
        self.lengths
            .view()
            .tile::<m![Windows], 1, m![Windows = 1 # Windows, Rows]>(at)
            .tile::<Tiles<Rows, N>, 1, m![
                Windows = 1 # Windows, { Tiles<Rows, N> } = 1 # ({ Tiles<Rows, N> }),
                { TileRows<Rows, N> }, { Layout<Rows>::Lanes }
            ]>(pass.0)
            .to_hbm_view(&mut device.tdma, lengths.view_mut());
        let in_dm: InDm<Rows, KvHeads, Head, TileRows<Rows, N>> = self
            .blocks
            .tokens
            .gather::<m![1], m![Block, KeyValue, KvHeads, Head]>()
            .by_byte_offsets(&entries)
            .prefix::<m![BlockIndex], _>(&lengths, 0)
            .overfetch()
            .to_dm(&mut device.tdma);
        // SAFETY: a shard's four blocks are its 64 consecutive slots, so both nests name the same
        // cells in the same order.
        unsafe { in_dm.reshape() }
    }
}

/// One of the chunk's two halves, the keys or the values of its tokens.
type Half<Count, Head> = m![{ Count }, KeyValue = 1 # 2, Cache % 64, Head];

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

/// Each shard's bias of its 64 slots, held per query row: a register operand repeats a single flit,
/// so an operand eight flits wide names the `Groups` axis of the logits it multiplies.
type Bias<Rows, KvHeads, Groups, Count = RowGroups<Rows>> = DmTensor<
    f32,
    Chip,
    Cluster<Rows>,
    CacheSlices<Rows, KvHeads>,
    m![{ Count }, Groups, Cache % 64],
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
pub type History<'a, Rows, KvHeads, Groups, Head, Windows, Source> = (
    &'a Heads<Rows, KvHeads, Groups, Head>,
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
    const SCALE: usize,
> Op<History<'a, Rows, KvHeads, Groups, Head, Windows, Source>>
    for Windowed<KvHeads, Groups, Head, SCALE>
where
    Source: Op<usize, Output = Chunk<Rows, KvHeads, Head>>
        + Op<(usize, Pass<32>), Output = Chunk<Rows, KvHeads, Head, TileRows<Rows, 32>>>
        + Op<(usize, Pass<16>), Output = Chunk<Rows, KvHeads, Head, TileRows<Rows, 16>>>
        + Op<(usize, Pass<8>), Output = Chunk<Rows, KvHeads, Head, TileRows<Rows, 8>>>,
{
    type Output = Attended<Rows, KvHeads, Groups, Head>;

    /// Attends each row's history and merges its shard-local softmax terms, window zero seeding the
    /// carries; row groups run in shared passes when all hold a live row, else only the live ones.
    fn forward(
        &self,
        device: &mut Device,
        (query, history, bias, rest, live): History<
            'a,
            Rows,
            KvHeads,
            Groups,
            Head,
            Windows,
            Source,
        >,
    ) -> Self::Output {
        const {
            assert!(
                KvHeads::SIZE == 8,
                "one pass of the partition holds eight KV heads of four lanes and eight shards"
            )
        };
        if live.count::<Lanes<Rows>>() == <RowGroups<Rows> as M>::SIZE {
            // A slice holds the query rows of the head it owns: each head is copied once, then
            // broadcast to the eight shards inside its processing element.
            let projected: Query<Rows, KvHeads, Groups, Head> = device
                .sub
                .begin(query.view())
                .fetch::<m![{ RowGroups<Rows> }, Groups, Head / 16], m![Head % 16]>()
                .switch::<CacheSlices<Rows, KvHeads>, m![{ RowGroups<Rows> }, Groups, Head / 16]>(
                    SwitchConfig::CustomBroadcast { ring_size: 8 },
                )
                .collect::<m![{ RowGroups<Rows> }, Groups, Head / 16], m![Head % 16]>()
                .commit_trim::<m![Head % 16]>()
                .commit();
            return <Partition<Rows, KvHeads, Groups, Head, SCALE> as shard::Partition>::windows(
                device,
                rest.extent::<m![{ Symbol<Rest<Windows>> }]>(),
                |device, at| {
                    // Staged per window, not before the loop, a slice holds one window's bias
                    // however many a history spans, and a skipped window moves nothing.
                    let chunk = history.forward(&mut *device, at);
                    let mut admitted = Bias::<Rows, KvHeads, Groups>::new();
                    bias.view()
                        .tile::<m![Windows], 1, m![Rows, Windows = 1 # Windows, Cache]>(at)
                        .to_dm_view(&mut device.tdma, admitted.view_mut());
                    attend::<Rows, KvHeads, Groups, Head, SCALE>(
                        device, &projected, &chunk, &admitted,
                    )
                },
            );
        }
        let rest = rest.extent::<m![{ Symbol<Rest<Windows>> }]>();
        let mut attended = Attended::<Rows, KvHeads, Groups, Head>::new();
        let inputs = (query, history, bias, rest);
        // Passes over 32-row tiles, then at most one over 16 rows and one over 8: most of a pass's
        // cost is paid once whatever its groups, so the live groups take the fewest passes.
        let (groups, wide, half) = (
            live.count::<Lanes<Rows>>(),
            <TileGroups<Rows, 32> as AxisName>::SIZE,
            <TileGroups<Rows, 16> as AxisName>::SIZE,
        );
        for tile in 0..groups / wide {
            self.pass(device, inputs, Pass::<32>(tile), &mut attended);
        }
        for tile in 0..groups % wide / half {
            let pass = Pass::<16>(groups / wide * (wide / half) + tile);
            self.pass(device, inputs, pass, &mut attended);
        }
        for tile in 0..groups % half {
            self.pass(
                device,
                inputs,
                Pass::<8>(groups / half * half + tile),
                &mut attended,
            );
        }
        attended
    }
}

impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName, const SCALE: usize>
    Windowed<KvHeads, Groups, Head, SCALE>
{
    /// Attends the row groups of `pass` over their histories and writes their rows in `attended`.
    fn pass<Rows: AxisName, Windows: AxisName, Source, const N: usize>(
        &self,
        device: &mut Device,
        (query, history, bias, rest): (
            &Heads<Rows, KvHeads, Groups, Head>,
            &Source,
            &HbmTensor<f32, Chip, m![Rows, Windows, Cache]>,
            usize,
        ),
        pass: Pass<N>,
        attended: &mut Attended<Rows, KvHeads, Groups, Head>,
    ) where
        Source: Op<(usize, Pass<N>), Output = Chunk<Rows, KvHeads, Head, TileRows<Rows, N>>>,
    {
        let projected: Query<Rows, KvHeads, Groups, Head, TileRows<Rows, N>> = device
            .sub
            .begin(query.view().tile::<Tiles<Rows, N>, 1, m![
                { Tiles<Rows, N> } = 1 # ({ Tiles<Rows, N> }), { TileRows<Rows, N> }, Groups, Head
            ]>(pass.0))
            .fetch::<m![{ TileRows<Rows, N> }, Groups, Head / 16], m![Head % 16]>()
            .switch::<CacheSlices<Rows, KvHeads>, m![{ TileRows<Rows, N> }, Groups, Head / 16]>(
                SwitchConfig::CustomBroadcast { ring_size: 8 },
            )
            .collect::<m![{ TileRows<Rows, N> }, Groups, Head / 16], m![Head % 16]>()
            .commit_trim::<m![Head % 16]>()
            .commit();
        let merged =
            <TileShards<Rows, KvHeads, Groups, Head, SCALE, N> as shard::Partition>::windows(
                device,
                rest,
                |device, at| {
                    let chunk = history.forward(&mut *device, (at, pass));
                    let mut admitted = Bias::<Rows, KvHeads, Groups, TileRows<Rows, N>>::new();
                    bias.view()
                        .tile::<m![Windows], 1, m![Rows, Windows = 1 # Windows, Cache]>(at)
                        .tile::<Tiles<Rows, N>, 1, m![
                            { Tiles<Rows, N> } = 1 # ({ Tiles<Rows, N> }), { TileRows<Rows, N> },
                            { Layout<Rows>::Lanes }, Windows = 1 # Windows, Cache
                        ]>(pass.0)
                        .to_dm_view(&mut device.tdma, admitted.view_mut());
                    let mut partial =
                        shard::Partial::<TileShards<Rows, KvHeads, Groups, Head, SCALE, N>>::new();
                    let mut lse =
                        shard::Local::<TileShards<Rows, KvHeads, Groups, Head, SCALE, N>>::new();
                    <TileShards<Rows, KvHeads, Groups, Head, SCALE, N> as shard::Partition>::attend::<
                        TileRows<Rows, N>,
                        _,
                        _,
                        _,
                        _,
                        _,
                    >(
                        device,
                        projected.view(),
                        chunk
                            .view()
                            .tile::<m![KeyValue], 1, Half<TileRows<Rows, N>, Head>>(0),
                        chunk
                            .view()
                            .tile::<m![KeyValue], 1, Half<TileRows<Rows, N>, Head>>(1),
                        admitted.view(),
                        partial.view_mut(),
                        lse.view_mut(),
                    );
                    (partial, lse)
                },
            );
        merged.view().to_dm_view(
            &mut device.tdma,
            attended.view_mut().tile::<Tiles<Rows, N>, 1, m![
                { Tiles<Rows, N> } = 1 #{!} ({ Tiles<Rows, N> }), { TileRows<Rows, N> },
                Groups, Head
            ]>(pass.0),
        );
    }
}

/// Attention in the partition the chunk was gathered in: each shard scores its own 64 slots and
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
    chunk: &Chunk<Rows, KvHeads, Head>,
    bias: &Bias<Rows, KvHeads, Groups>,
) -> (
    Partial<Rows, KvHeads, Groups, Head, SCALE>,
    Local<Rows, KvHeads, Groups, Head, SCALE>,
) {
    let mut partial = Partial::<Rows, KvHeads, Groups, Head, SCALE>::new();
    let mut lse = Local::<Rows, KvHeads, Groups, Head, SCALE>::new();
    #[unroll]
    for tile in 0..<Tiles<Rows> as M>::SIZE {
        <Partition<Rows, KvHeads, Groups, Head, SCALE> as shard::Partition>::attend::<TileRows<Rows>, _, _, _, _, _>(
            device,
            query
                .view()
                .tile::<Tiles<Rows>, 1, m![{ Tiles<Rows> } = 1 # ({ Tiles<Rows> }), { TileRows<Rows> }, Groups, Head]>(tile),
            chunk
                .view()
                .tile::<m![KeyValue], 1, Half<RowGroups<Rows>, Head>>(0)
                .tile::<Tiles<Rows>, 1, m![
                    { Tiles<Rows> } = 1 # ({ Tiles<Rows> }), { TileRows<Rows> }, KeyValue = 1 # 2, Cache % 64, Head
                ]>(tile),
            chunk
                .view()
                .tile::<m![KeyValue], 1, Half<RowGroups<Rows>, Head>>(1)
                .tile::<Tiles<Rows>, 1, m![
                    { Tiles<Rows> } = 1 # ({ Tiles<Rows> }), { TileRows<Rows> }, KeyValue = 1 # 2, Cache % 64, Head
                ]>(tile),
            bias.view()
                .tile::<Tiles<Rows>, 1, m![{ Tiles<Rows> } = 1 # ({ Tiles<Rows> }), { TileRows<Rows> }, Groups, Cache % 64]>(tile),
            partial
                .view_mut()
                .tile::<Tiles<Rows>, 1, m![{ Tiles<Rows> } = 1 #{!} ({ Tiles<Rows> }), { TileRows<Rows> }, Groups, Head]>(tile),
            lse.view_mut().tile::<Tiles<Rows>, 1, m![
                { Tiles<Rows> } = 1 #{!} ({ Tiles<Rows> }), { TileRows<Rows> }, Groups, 1 # 8
            ]>(tile),
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

    axes![Groups = 2];

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
        table: &HbmTensor<i32, Chip, m![Count, Chunks, BlockIndex]>,
        lengths: &BlockCounts<Count, Chunks>,
        output: &mut HbmTensor<bf16, Chip, m![Chunks, Count, Cache, KeyValue, KvHeads, Head]>,
    ) {
        let history = Paged {
            blocks,
            table,
            lengths,
        };
        for w in 0..Chunks::SIZE {
            history.forward(&mut *device, w).view().to_hbm_view(
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
        let entries = Chunks::SIZE * Cache::SIZE / Block::SIZE;
        let block = |row: usize, at: usize| (3 + row * 37 + at * 11) % (TEST_TOKENS / Block::SIZE);
        let table = HostTensor::<i32, m![Count, Chunks, BlockIndex]>::from_vec(
            (0..Count::SIZE)
                .flat_map(|row| {
                    (0..entries).map(move |at| block(row, at) as i32 * <Blocks>::BLOCK_STRIDE)
                })
                .collect::<Vec<_>>(),
        )
        .to_hbm::<Chip, m![Count, Chunks, BlockIndex]>(&mut device.pdma)
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
    fn attend_prefix<Count: AxisName>(
        device: &mut Device,
        query: &PrefixQuery<Count>,
        blocks: &Blocks,
        table: &HbmTensor<i32, Chip, m![Count, Chunks, BlockIndex]>,
        lengths: &BlockCounts<Count, Chunks>,
        bias: &Bias<Count>,
        (rest, live): (&Value, &Value),
        output: &mut PrefixOutput<Count>,
    ) {
        let query: super::Heads<Count, KvHeads, PrefixGroups, Head> = query.to_dm(&mut device.tdma);
        let history = Paged {
            blocks,
            table,
            lengths,
        };
        Windowed::<KvHeads, PrefixGroups, Head, INVERSE_ROOT>::default()
            .forward(
                &mut *device,
                (&query, &history, bias, rest, Live::<(), Count>::read(live)),
            )
            .view()
            .to_hbm_view(&mut device.tdma, output.view_mut());
    }

    /// Rows `0..live` of `Count` attend over the prefix of the bank each names, through a
    /// scratch poisoned past every prefix; later rows are not live and write nothing checked.
    async fn prefix<Count: AxisName>(rows: &[usize], live: usize) {
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
        let table = HostTensor::<i32, m![Count, Chunks, BlockIndex]>::from_vec(
            rows.iter().flat_map(|&row| {
                (0..Chunks::SIZE * Cache::SIZE / Block::SIZE).map(move |at| {
                    (((row % 8) * 1024 / Block::SIZE + at % (1024 / Block::SIZE)) as i32)
                        * <Blocks>::BLOCK_STRIDE
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
            let lengths_hbm: BlockCounts<Count, Chunks> =
                block_counts::<Count, Chunks>(lengths.iter().map(|&length| {
                    if phase == 0 {
                        TEST_TOKENS as i32
                    } else {
                        length as i32
                    }
                }))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
            let bias = HostTensor::<f32, m![Count, Chunks, Cache]>::from_vec(
                lengths.iter().flat_map(|&length| {
                    (0..Chunks::SIZE * Cache::SIZE)
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
        prefix::<Rows<64>>(
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
        prefix::<Rows<64>>(
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
        prefix::<Rows<64>>(&(0..64).map(|row| (row + 7) % 8).collect::<Vec<_>>(), 1).await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn wide_partial_rows_zero() {
        prefix::<Rows<64>>(
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
        prefix::<Rows<32>>(
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
        prefix::<Rows<64>>(
            &(0..64)
                .map(|row| if row < 32 { (row * 5 + row / 8) % 8 } else { 8 })
                .collect::<Vec<_>>(),
            64,
        )
        .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn tail_preserves_prefix() {
        prefix::<Eight>(&[0, 1, 2, 3, 4, 5, 6, 7], 8).await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn rows_remain_independent() {
        prefix::<Sixteen>(&[7, 6, 5, 4, 3, 2, 1, 0, 8, 9, 10, 11, 12, 13, 14, 15], 16).await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn empty_history_zero() {
        prefix::<Single>(&[0], 1).await;
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
        let chunk: super::Chunk<Eight, KvHeads, Head> = source.to_dm(&mut device.tdma);
        let query: super::Query<Eight, KvHeads, Groups, Head> = query.to_dm(&mut device.tdma);
        let bias: super::Bias<Eight, KvHeads, Groups> = bias.to_dm(&mut device.tdma);
        <super::Partition<Eight, KvHeads, Groups, Head, INVERSE_ROOT> as Partition>::windows(
            device,
            0,
            |device, _| {
                super::attend::<Eight, KvHeads, Groups, Head, INVERSE_ROOT>(
                    device, &query, &chunk, &bias,
                )
            },
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
