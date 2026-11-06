//! Paged attention over a launch's host-planned work: the rows from row 0 attended one at a time
//! as work items over the pool, then the prompt chunks after them over the pool's committed history
//! and the chunks' own keys and values.
//!
//! - [`Work`] is what the host plans and stages for a launch; [`Work::read`] reads its counts once
//!   at the launch's start into the [`Context`] every layer borrows.
//! - Attention only reads: the step writes the item rows' keys and values before it, and the
//!   chunks' rows wherever its schedule puts the cache stage.
//! - The catalog's sequential layer has written every row's keys and values when it attends, so
//!   its host plans every row as an item; a step whose history retains the launch's own keys and
//!   values also attends prompt chunks.

use crate::Module;
use crate::ops::attention;
use crate::ops::attention::gqa::Attended;
use crate::tensor::{live::Live, parallel::Chip};
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::HbmBuffer;
use std::io;
use std::marker::PhantomData;

use crate::ops::index::gather::Gather;
use crate::ops::index::table::FIELDS;
use crate::tensor::cache::Blocks;
use gqa::items::SINKS;
use gqa::items::Shape;
use gqa::items::{GroupTile, Tiled};
pub use gqa::items::{Items, Options, Plan, Tiling, Waves};

mod append;
pub mod chunks;
mod gqa {
    use super::*;
    use crate::ops::index::gather::{Gather, Prefix, Read};
    use crate::tensor::cache::Block;
    use crate::tensor::cache::KeyValue;

    pub(crate) mod items;
}

type Attention<Heads, Groups, Head> = attention::Attention<PhantomData<(Heads, Groups, Head)>, f32>;

/// A launch's attention work as its host planned it: heads of `Head` channels, `G` to each of
/// `KvHeads`, items within `Waves` wide waves and at most `Sequences` prompt chunks, each over at
/// most `W` 512-key windows of history.
#[derive(DeviceSend)]
pub struct Work<
    Sequences: AxisName,
    KvHeads: AxisName,
    G: AxisName,
    Head: AxisName,
    Waves: AxisName,
    W: AxisName,
> {
    /// The rows from row 0, each attended over its own history in the pool.
    pub items: Items<KvHeads, G, Head, Waves>,
    /// The prompt chunks after them.
    pub chunks: chunks::Invocation<Sequences, W>,
}

/// A launch's [`Work`] as its layers borrow it, with the counts its loops run read once.
pub struct Context<
    'a,
    Sequences: AxisName,
    KvHeads: AxisName,
    G: AxisName,
    Head: AxisName,
    Waves: AxisName,
    W: AxisName,
    R,
> {
    work: &'a Work<Sequences, KvHeads, G, Head, Waves, W>,
    items: Live<(), R>,
    shape: Shape,
    chunks: usize,
}

impl<
    Sequences: AxisName,
    KvHeads: AxisName,
    G: AxisName,
    Head: AxisName,
    Waves: AxisName,
    W: AxisName,
> Work<Sequences, KvHeads, G, Head, Waves, W>
{
    /// The work whose arrays [`Plan::reads`], then [`chunks::Plan::reads`], list, as `next` names
    /// each in turn, with the chunks' `masks`.
    pub fn bind(
        mut next: impl FnMut() -> io::Result<HbmBuffer>,
        masks: &chunks::Masks<W>,
    ) -> io::Result<Self> {
        Ok(Self {
            items: Items::bind(&mut next)?,
            chunks: chunks::Invocation::bind(next, masks)?,
        })
    }

    /// The arrays, in [`Work::bind`] order, of a launch whose `R` rows (`segments`) attend as items
    /// planned as `options` says: its used leading words, then all; `ids` holds row IDs, padded
    /// to 16.
    pub fn sequential<R: AxisName>(
        segments: &[[i32; FIELDS]],
        ids: &[i32],
        options: impl Into<Options>,
    ) -> io::Result<Vec<(Vec<i32>, usize)>> {
        let items = Plan::new::<KvHeads, G, Head, Waves, Sinked<R>>(segments, ids, options);
        let items = items.reads().map(|(read, words)| (read.to_vec(), words));
        let chunks = chunks::Plan::<W>::new::<Sequences>().reads()?;
        Ok(items.chain(chunks).collect())
    }

    /// Bytes of each allocation the sequential layer's attention makes as it attends a layer of
    /// `R` rows: its query and attended rows, and its items' outputs.
    pub const fn scratch<R: AxisName>() -> [usize; 4] {
        let rows = <m![{ Symbol<Sinked<R>> }, KvHeads, G, Head]>::SIZE * size_of::<bf16>();
        let [outputs, scales] = Items::<KvHeads, G, Head, Waves>::SCRATCH;
        [rows, rows, outputs, scales]
    }

    /// The work as a launch's layers read it: a read of launch metadata holds the command issue
    /// until the DMA queue drains, so a launch reads these at its start, once.
    pub fn read<R: AxisName>(&self) -> Context<'_, Sequences, KvHeads, G, Head, Waves, W, R> {
        let chunks = self.chunks.live.extent::<m![Sequences]>();
        Context {
            work: self,
            items: Live::<(), R>::read(self.items.rows()),
            shape: self.items.shape(),
            chunks,
        }
    }
}

impl<
    Sequences: AxisName,
    KvHeads: AxisName,
    G: AxisName,
    Head: AxisName,
    Waves: AxisName,
    W: AxisName,
    R,
> Context<'_, Sequences, KvHeads, G, Head, Waves, W, R>
{
    /// The rows attended as items, from row 0.
    pub fn items(&self) -> Live<(), R> {
        self.items
    }
}

/// The history a scheduled step attends: the pool of committed and item rows' keys and values, and
/// the launch rows' as the projection retains them; `read` reads the pool's blocks.
pub struct Retained<'a, Reach: AxisName, Heads: AxisName, Head: AxisName> {
    pub pool: &'a Blocks<Heads, Head>,
    pub key: &'a HbmTensor<bf16, Chip, m![Reach, Heads, Head]>,
    pub value: &'a HbmTensor<bf16, Chip, m![Reach, Heads, Head]>,
    pub read: &'a Gather,
}

/// A launch's query rows of each (KV head, group) and the rows attention returns.
type Rows<Reach, Heads, Groups, Head> = HbmTensor<bf16, Chip, m![Reach, Heads, Groups, Head]>;

/// Every live row attends its visible keys into `attended`, items over the pool, chunks also over
/// their retained keys and values; `boundary` runs once the last history is in flight.
impl<
    Reach: AxisName,
    Sequences: AxisName,
    Heads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    Scale: Copy + Into<f32>,
    Waves: AxisName,
    W: AxisName,
    L: AxisName,
    H,
    G,
    Env: Copy,
>
    Module<
        (
            &Rows<Reach, Heads, Groups, Head>,
            &Retained<'_, Reach, Heads, Head>,
            &Context<'_, Sequences, Heads, Groups, Head, Waves, W, L>,
            (&H, &mut G),
            &mut Rows<Reach, Heads, Groups, Head>,
        ),
        Env,
        &mut (),
    > for attention::Attention<PhantomData<(Heads, Groups, Head)>, Scale>
where
    H: for<'d, 'g> Module<&'g mut G, Env, &'d mut (), Output = ()>,
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        env: Env,
        _: &mut (),
        (query, history, work, (boundary, state), attended): (
            &Rows<Reach, Heads, Groups, Head>,
            &Retained<'_, Reach, Heads, Head>,
            &Context<'_, Sequences, Heads, Groups, Head, Waves, W, L>,
            (&H, &mut G),
            &mut Rows<Reach, Heads, Groups, Head>,
        ),
    ) {
        let attention = attention::Attention {
            heads: self.heads,
            scale: self.scale.into(),
        };
        attention.forward(
            &mut *context,
            env,
            &mut (),
            (
                query,
                (&work.work.items, work.shape, (history.read, history.pool)),
                &mut *attended,
                (boundary, &mut *state),
            ),
        );
        // The chunks run `boundary` only when no item did.
        let idle = (Reach::SIZE - work.items.rows()) / Reach::SIZE;
        attention.forward(
            context,
            env,
            &mut (),
            (
                (query, history.key, history.value),
                (&work.work.chunks, work.chunks),
                (history.read, history.pool),
                attended,
                ((boundary, state), idle),
            ),
        );
    }
}

/// Every live row's query attends its visible keys, as the 5-tuple form does with no boundary,
/// into attended rows it returns.
impl<
    'a,
    Reach: AxisName,
    Sequences: AxisName,
    Heads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    Scale: Copy + Into<f32>,
    Waves: AxisName,
    W: AxisName,
    L: AxisName,
    Env: Copy,
>
    Module<
        (
            &Rows<Reach, Heads, Groups, Head>,
            Retained<'a, Reach, Heads, Head>,
            &Context<'_, Sequences, Heads, Groups, Head, Waves, W, L>,
        ),
        Env,
        &mut (),
    > for attention::Attention<PhantomData<(Heads, Groups, Head)>, Scale>
{
    type Output = Rows<Reach, Heads, Groups, Head>;

    fn forward(
        &self,
        context: &mut Device,
        env: Env,
        _: &mut (),
        (query, history, work): (
            &Rows<Reach, Heads, Groups, Head>,
            Retained<'a, Reach, Heads, Head>,
            &Context<'_, Sequences, Heads, Groups, Head, Waves, W, L>,
        ),
    ) -> Self::Output {
        let mut attended = Rows::new();
        self.forward(
            context,
            env,
            &mut (),
            (query, &history, work, (&(), &mut ()), &mut attended),
        );
        attended
    }
}

/// The rows of `R` with the items' sinks after them, as the items address them in HBM.
#[derive(Clone, Debug)]
struct Sinked<R>(PhantomData<R>);

/// The query or attended rows of `R` and the sinks after them, each row's group tiles in turn.
type TileMajor<R, Heads, Groups, Head> = HbmTensor<
    bf16,
    Chip,
    m![
        R # ({ Symbol<Sinked<R>> }),
        Groups / ({ Symbol<GroupTile<Groups>> }),
        Heads,
        Groups % ({ Symbol<GroupTile<Groups>> }),
        Head
    ],
>;

impl<R: AxisName> AxisName for Sinked<R> {
    const NAME: Ident = Ident::new("Rows");
    const SIZE: usize = R::SIZE + SINKS;
}

/// The heads of each live row of `R`, as the catalog's projection leaves them, attend as items over
/// the pool the layer's cache stage wrote them to.
impl<
    R: AxisName,
    Sequences: AxisName,
    Heads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    Scale: Copy + Into<f32>,
    Waves: AxisName,
    W: AxisName,
    Env: Copy,
>
    Module<
        (
            &Live<attention::Heads<R, Heads, Groups, Head>, R>,
            &Blocks<Heads, Head>,
            &Context<'_, Sequences, Heads, Groups, Head, Waves, W, R>,
        ),
        Env,
        &mut (),
    > for attention::Attention<PhantomData<(Heads, Groups, Head)>, Scale>
{
    type Output = Live<Attended<R, Heads, Groups, Head>, R>;

    fn forward(
        &self,
        context: &mut Device,
        env: Env,
        _: &mut (),
        (query, pool, work): (
            &Live<attention::Heads<R, Heads, Groups, Head>, R>,
            &Blocks<Heads, Head>,
            &Context<'_, Sequences, Heads, Groups, Head, Waves, W, R>,
        ),
    ) -> Self::Output {
        let attention = attention::Attention {
            heads: self.heads,
            scale: self.scale.into(),
        };
        let mut rows = TileMajor::<R, Heads, Groups, Head>::new();
        query
            .value()
            .view()
            .to_hbm_view(&mut context.tdma, rows.view_mut());
        // SAFETY: each row's group tiles in turn are item rows; the rows past `R` are the sinks,
        // which the items only write.
        let rows: Rows<Tiled<Sinked<R>, Groups>, Heads, GroupTile<Groups>, Head> =
            unsafe { rows.reshape() };
        let mut attended = Rows::<Tiled<Sinked<R>, Groups>, Heads, GroupTile<Groups>, Head>::new();
        attention.forward(
            &mut *context,
            env,
            &mut (),
            (
                &rows,
                (&work.work.items, work.shape, (&Gather, pool)),
                &mut attended,
                (&(), &mut ()),
            ),
        );
        // SAFETY: the same rows, the sinks after them as padding.
        let attended: TileMajor<R, Heads, Groups, Head> = unsafe { attended.reshape() };
        query.with(attended.to_dm(&mut context.tdma))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Op;
    use crate::load::Checkpoint;
    use crate::modules::attention::key_value::{History, Paging, Preparation};
    use crate::ops::attention::Attention;
    use crate::ops::attention::Heads;
    use crate::ops::attention::gqa::{Cache, Pages, Wrap};
    use crate::ops::index::table::{BlockIndex, Fields, PADDING_BIAS};
    use crate::specialize::attention::{HISTORY_WAVES, Strategy, WAVES};
    use crate::tensor::axis::Rows;
    use crate::tensor::cache::{Block, Blocks};
    use crate::tensor::live::tests::count;
    use crate::tensor::scalar::{F32, Value};
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;
    use gqa::items::tests::{
        self as items, Heads as KvHeads, KEYS, LONG, QueryEights as Eights, QueryGroups as Groups,
        QueryPairs as Pairs, QuerySingle as Single, QuerySixteens as Sixteens, SCALE, UNIT, WINDOW,
        Wide, Width as Head, stride,
    };
    use std::ops::Range;

    axes![Sequences = 1, Windows = 80];
    // A call's flat table of block IDs: 32 rows, each of a whole history's blocks.
    axes![Table = 81920];

    /// How the catalog's layers of `R` rows page a history of `H` KV heads of `D` channels.
    type Paged<R, H, D> = Preparation<R, Paging<H, D, R, Windows>>;

    type Layout<R, H, G, D> = m![R, H, G, D];
    type Queries<R, H, G, D> = HbmTensor<bf16, Chip, Layout<R, H, G, D>>;
    /// The items' pool of `H` KV heads of `D` channels, and each history's first block.
    type Histories<H, D> = (Blocks<H, D>, Vec<usize>);

    #[device(chip = 1)]
    fn attend_rows<
        R: AxisName,
        H: AxisName,
        G: AxisName,
        D: AxisName,
        const SCALE: usize,
        const WAVES: usize,
    >(
        device: &mut Device,
        scale: &F32<SCALE>,
        query: &Queries<R, H, G, D>,
        work: &Work<Sequences, H, G, D, Waves<WAVES>, Windows>,
        pool: &mut Blocks<H, D>,
        attended: &mut Queries<R, H, G, D>,
    ) {
        let context = work.read::<R>();
        let attention = Attention {
            heads: PhantomData::<(H, G, D)>,
            scale: f32::from(*scale),
        };
        let heads: Heads<R, H, G, D> = query.to_dm(&mut device.tdma);
        let query = context.items().with(heads);
        let output = Module::forward(
            &attention,
            &mut *device,
            (),
            &mut (),
            (&query, &*pool, &context),
        );
        output
            .value()
            .view()
            .to_hbm_view(&mut device.tdma, attended.view_mut());
    }

    #[device(chip = 1)]
    fn attend_windows<R: AxisName, H: AxisName, G: AxisName, D: AxisName, const SCALE: usize>(
        device: &mut Device,
        scale: &F32<SCALE>,
        query: &Queries<R, H, G, D>,
        pages: &Pages<R, Windows>,
        live: &Value,
        pool: &mut Blocks<H, D>,
        attended: &mut Queries<R, H, G, D>,
    ) {
        let attention = Attention::<PhantomData<(H, G, D)>, F32<SCALE>, { Strategy::Windows }> {
            heads: PhantomData,
            scale: *scale,
        };
        let heads: Heads<R, H, G, D> = query.to_dm(&mut device.tdma);
        let query = Live::<(), R>::read(live).with(heads);
        let output = Op::forward(&attention, &mut *device, (&query, &*pool, pages));
        output
            .value()
            .view()
            .to_hbm_view(&mut device.tdma, attended.view_mut());
    }

    /// The windowed leaf over the pages `preparation` derives from the compact `history` of a
    /// call's `live` rows, as the catalog's layers page their history.
    #[device(chip = 1)]
    fn attend_paged<R: AxisName, H: AxisName, G: AxisName, D: AxisName, const SCALE: usize>(
        device: &mut Device,
        scale: &F32<SCALE>,
        query: &Queries<R, H, G, D>,
        paged: (&Paged<R, H, D>, &History<R, Table, R>),
        live: &Value,
        pool: &mut Blocks<H, D>,
        attended: &mut Queries<R, H, G, D>,
    ) {
        let (preparation, history) = paged;
        let tiles = Live::<(), R>::read(live);
        let blocks = (&history.segments, &history.blocks);
        let logical = Op::forward(&preparation.derive, &mut *device, blocks);
        let context = Module::forward(
            &preparation.geometry,
            &mut *device,
            (),
            &mut (),
            (tiles, (&logical, history)),
        );
        let attention = Attention::<PhantomData<(H, G, D)>, F32<SCALE>, { Strategy::Windows }> {
            heads: PhantomData,
            scale: *scale,
        };
        let heads: Heads<R, H, G, D> = query.to_dm(&mut device.tdma);
        let query = tiles.with(heads);
        let output = Op::forward(&attention, &mut *device, (&query, &*pool, &context.read));
        output
            .value()
            .view()
            .to_hbm_view(&mut device.tdma, attended.view_mut());
    }

    /// Rows of `lengths` keys of `firsts`' pool as the windowed leaf reads them, in `R`'s rows:
    /// each window's block offsets, block counts and mask, and the windows past the first.
    async fn pages<R: AxisName, H: AxisName, D: AxisName>(
        device: &mut Device,
        lengths: &[usize],
        firsts: &[usize],
    ) -> Pages<R, Windows> {
        let (rows, windows, slots) = (R::SIZE, Windows::SIZE, BlockIndex::SIZE);
        let length = |row: usize| lengths.get(row).copied().unwrap_or(0);
        // Every entry names a readable block: past a row's blocks, its last one.
        let block = |at: usize| {
            let (window, row, slot) = (at / (rows * slots), at / slots % rows, at % slots);
            let at = window * slots + slot;
            let history = items::history(row, length(row).max(1));
            let last = firsts[history] + items::keys(history).div_ceil(Block::SIZE) - 1;
            let used = length(row).div_ceil(Block::SIZE).max(1);
            (last - at.min(used - 1)) as i32 * Blocks::<H, D>::BLOCK_STRIDE
        };
        let table: Vec<i32> = (0..rows * windows * slots).map(block).collect();
        let counts: Vec<i32> = (0..windows * rows)
            .map(|at| {
                let (window, row) = (at / rows, at % rows);
                let seen = length(row).saturating_sub(window * Cache::SIZE);
                seen.min(Cache::SIZE).div_ceil(Block::SIZE) as i32
            })
            .collect();
        let bias: Vec<f32> = (0..rows * windows * Cache::SIZE)
            .map(|at| {
                let (row, key) = (at / (windows * Cache::SIZE), at % (windows * Cache::SIZE));
                if key < length(row) { 0.0 } else { PADDING_BIAS }
            })
            .collect();
        let rest = lengths.iter().map(|keys| (keys - 1) / Cache::SIZE).max();
        let table = HostTensor::<i32, m![Windows, R, BlockIndex]>::from_vec(table);
        let counts = HostTensor::<i32, m![Windows, R]>::from_vec(counts);
        let bias = HostTensor::<f32, m![R, Windows, Cache]>::from_vec(bias);
        Pages {
            table: table.to_hbm(&mut device.pdma).await.unwrap(),
            lengths: counts.to_hbm(&mut device.pdma).await.unwrap(),
            bias: bias.to_hbm(&mut device.pdma).await.unwrap(),
            rest: count(rest.unwrap_or(0))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            wrap: Wrap {
                offsets: HostTensor::<i32, m![R, BlockIndex % 2, Block]>::zero()
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
                counts: HostTensor::<i32, m![R]>::zero()
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
                any: count(0).to_hbm(&mut device.pdma).await.unwrap(),
            },
        }
    }

    /// The rows of `lengths` keys of a call of `R` rows, attended as work items within `WAVES` wide
    /// waves through the head layout the catalog's projection leaves.
    async fn as_items<
        R: AxisName,
        H: AxisName,
        G: AxisName,
        D: AxisName,
        const SCALE: usize,
        const WAVES: usize,
    >(
        device: &mut Device,
        (pool, firsts): &mut Histories<H, D>,
        masks: &chunks::Masks<Windows>,
        lengths: &[usize],
    ) -> Vec<bf16> {
        let queries = items::queries::<H, G, D>(R::SIZE);
        let queries = HostTensor::<bf16, Layout<R, H, G, D>>::from_vec(queries);
        let queries: Queries<R, H, G, D> = queries.to_hbm(&mut device.pdma).await.unwrap();
        let rows = lengths.iter().copied().enumerate();
        let plan = items::plan::<H, G, D, WAVES, Sinked<R>>(rows, firsts, Tiling::Adaptive);
        let chunks = chunks::Plan::<Windows>::new::<Sequences>().reads().unwrap();
        let reads = plan
            .reads()
            .chain(chunks.iter().map(|(read, words)| (&read[..], *words)));
        let mut buffers = items::place(device, reads).await;
        let work: Work<Sequences, H, G, D, Waves<WAVES>, Windows> =
            Work::bind(|| Ok(buffers.next().expect("an array a buffer")), masks).unwrap();
        let mut attended: Queries<R, H, G, D> = HostTensor::<bf16, Layout<R, H, G, D>>::zero()
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let arguments = (
            &mut *device,
            &F32::<SCALE>,
            &queries,
            &work,
            &mut *pool,
            &mut attended,
        );
        launch(attend_rows, arguments).await.unwrap();
        attended
            .to_host::<Layout<R, H, G, D>>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec()
    }

    /// The rows of `lengths` keys of a call of `R` rows, attended by the windowed leaf over the
    /// pages of their histories.
    async fn by_windows<R: AxisName, H: AxisName, G: AxisName, D: AxisName, const SCALE: usize>(
        device: &mut Device,
        (pool, firsts): &mut Histories<H, D>,
        lengths: &[usize],
    ) -> Vec<bf16> {
        let queries = items::queries::<H, G, D>(R::SIZE);
        let queries = HostTensor::<bf16, Layout<R, H, G, D>>::from_vec(queries);
        let queries: Queries<R, H, G, D> = queries.to_hbm(&mut device.pdma).await.unwrap();
        let pages = pages::<R, H, D>(device, lengths, firsts).await;
        let live = count(lengths.len()).to_hbm(&mut device.pdma).await.unwrap();
        let mut attended: Queries<R, H, G, D> = HostTensor::<bf16, Layout<R, H, G, D>>::zero()
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let arguments = (
            &mut *device,
            &F32::<SCALE>,
            &queries,
            &pages,
            &live,
            &mut *pool,
            &mut attended,
        );
        launch(attend_windows, arguments).await.unwrap();
        attended
            .to_host::<Layout<R, H, G, D>>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec()
    }

    /// The largest distance from the f64 softmax of the rows of `lengths` keys of a call of `R`
    /// rows of `G` query groups a KV head, in each of the catalog's two attention forms.
    async fn errors<R: AxisName, G: AxisName, const WAVES: usize>(
        device: &mut Device,
        histories: &mut Histories<KvHeads, Head>,
        masks: &chunks::Masks<Windows>,
        lengths: &[usize],
    ) -> [f64; 2] {
        let forms = [
            as_items::<R, KvHeads, G, Head, SCALE, WAVES>(device, histories, masks, lengths).await,
            by_windows::<R, KvHeads, G, Head, SCALE>(device, histories, lengths).await,
        ];
        let width = stride::<KvHeads, G, Head>();
        forms.map(|output| {
            let row = |(row, &length): (usize, &usize)| {
                items::error::<KvHeads, G, Head, SCALE>(
                    row,
                    length,
                    &output[row * width..][..width],
                )
            };
            lengths.iter().enumerate().map(row).fold(0.0, f64::max)
        })
    }

    /// Attended as work items, the rows of the catalog's calls lie at most twice as far from the
    /// f64 softmax as the windowed leaf's do, at 25, 1,024, 4,096, 16,384 and 40,960 keys and
    /// mixed lengths: 64-row calls of four query groups a KV head within [`HISTORY_WAVES`], 16-row
    /// calls of two within [`WAVES`].
    #[tokio::test(flavor = "current_thread")]
    async fn errs_at_most_twice_the_windowed_leaf() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(attend_windows.topology()).unwrap();
        let mut pool = items::pool::<KvHeads, Head>(&mut device).await;
        let masks = chunks::Masks::<Windows>::new(&mut device).await.unwrap();
        let device = &mut device;
        let cases = |rows: usize| {
            [
                vec![25; 1],
                vec![25; rows],
                vec![1024; 8],
                vec![4096; rows / 2],
                vec![16384; 1],
                vec![LONG; rows],
                (0..rows).map(|row| items::MIXED[row % 16]).collect(),
            ]
        };
        let bounded = |groups: usize, lengths: &[usize], [items, windows]: [f64; 2]| {
            let keys = |keys: Option<&usize>| keys.copied().unwrap_or(0);
            let (shortest, longest) = (keys(lengths.iter().min()), keys(lengths.iter().max()));
            eprintln!(
                "ERRORS {groups} groups, {} rows of {shortest}..{longest} keys: items \
                 {items:.3e} windows {windows:.3e}",
                lengths.len()
            );
            assert!(
                items <= 2.0 * windows,
                "{groups} groups, {} rows: items {items} > 2 x windows {windows}",
                lengths.len()
            );
        };
        for lengths in cases(64) {
            let errors =
                errors::<Rows<64>, Groups, HISTORY_WAVES>(device, &mut pool, &masks, &lengths);
            bounded(Groups::SIZE, &lengths, errors.await);
        }
        for lengths in cases(16) {
            let errors = errors::<Rows<16>, Pairs, WAVES>(device, &mut pool, &masks, &lengths);
            bounded(Pairs::SIZE, &lengths, errors.await);
        }
    }

    /// Each live row of a sequential layer's envelope `R` of `G` query groups a KV head, planned
    /// within `WAVES` wide waves, through the head layout the catalog's projection leaves, attends
    /// its own prefix of the pool as a host softmax does.
    async fn check<R: AxisName, G: AxisName, const WAVES: usize>(
        device: &mut Device,
        histories: &mut Histories<KvHeads, Head>,
        masks: &chunks::Masks<Windows>,
        lengths: &[usize],
    ) {
        let output =
            as_items::<R, KvHeads, G, Head, SCALE, WAVES>(device, histories, masks, lengths);
        check_rows::<KvHeads, G, Head, SCALE>((lengths, items::prefix), &output.await);
    }

    /// Each row `r` of `output` attends what `seen(r, lengths[r])` names of the pool's histories,
    /// as a host softmax does.
    fn check_rows<H: AxisName, G: AxisName, D: AxisName, const SCALE: usize>(
        (lengths, seen): (&[usize], impl Fn(usize, usize) -> (usize, Range<usize>)),
        output: &[bf16],
    ) {
        let rows = output.chunks_exact(stride::<H, G, D>());
        for ((row, &length), got) in lengths.iter().enumerate().zip(rows) {
            items::check::<H, G, D, SCALE>(row, seen(row, length), got, lengths.len());
        }
    }

    /// The 16- and 64-row envelopes of four query groups a KV head and the 16-row one of one, two,
    /// eight and sixteen, all rows live or some, of one history length or many; 64 rows of the
    /// whole history within [`HISTORY_WAVES`].
    #[tokio::test(flavor = "current_thread")]
    async fn attends_every_live_row_of_a_sequential_layer() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(attend_rows.topology()).unwrap();
        let mut pool = items::pool::<KvHeads, Head>(&mut device).await;
        let masks = chunks::Masks::<Windows>::new(&mut device).await.unwrap();
        let device = &mut device;
        let mixed: Vec<usize> = (0..64).map(|row| items::MIXED[row % 16]).collect();
        for lengths in [items::MIXED.to_vec(), vec![KEYS; 5], vec![LONG, 17, 1]] {
            check::<Rows<16>, Groups, WAVES>(device, &mut pool, &masks, &lengths).await;
            check::<Rows<16>, Pairs, WAVES>(device, &mut pool, &masks, &lengths).await;
            check::<Rows<16>, Single, WAVES>(device, &mut pool, &masks, &lengths).await;
            check::<Rows<16>, Eights, WAVES>(device, &mut pool, &masks, &lengths).await;
            check::<Rows<16>, Sixteens, WAVES>(device, &mut pool, &masks, &lengths).await;
        }
        for lengths in [mixed.clone(), vec![KEYS; 33]] {
            check::<Rows<64>, Groups, WAVES>(device, &mut pool, &masks, &lengths).await;
        }
        for lengths in [vec![LONG; 64], mixed] {
            check::<Rows<64>, Groups, HISTORY_WAVES>(device, &mut pool, &masks, &lengths).await;
        }
    }

    /// Rows of 256-wide heads, two query groups to each of eight KV heads, attend under a unit
    /// score multiplier as a host softmax does, as work items and through the windowed leaf: a
    /// 32-row envelope all live, a 17-row tail of mixed lengths, and the whole history beside two
    /// short rows.
    #[tokio::test(flavor = "current_thread")]
    async fn attends_every_live_row_of_wide_heads() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(attend_rows.topology()).unwrap();
        let mut pool = items::pool::<KvHeads, Wide>(&mut device).await;
        let masks = chunks::Masks::new(&mut device).await.unwrap();
        let device = &mut device;
        let tail: Vec<usize> = (0..17).map(|row| items::MIXED[row % 16]).collect();
        for lengths in [vec![KEYS; 32], tail, vec![LONG, 17, 1]] {
            let output = as_items::<Rows<32>, KvHeads, Pairs, Wide, UNIT, WAVES>;
            let output = output(device, &mut pool, &masks, &lengths).await;
            check_rows::<KvHeads, Pairs, Wide, UNIT>((&lengths, items::prefix), &output);
            let output = by_windows::<Rows<32>, KvHeads, Pairs, Wide, UNIT>;
            let output = output(device, &mut pool, &lengths).await;
            check_rows::<KvHeads, Pairs, Wide, UNIT>((&lengths, items::prefix), &output);
        }
    }

    /// The rows of `lengths` keys of a call of `R` rows, attended by the windowed leaf over the
    /// pages a device paging of a window of [`WINDOW`] keys derives from the call's compact history:
    /// a segment a row, each row's block IDs from position 0 as a framework names them.
    async fn by_paging<R: AxisName, H: AxisName, G: AxisName, D: AxisName, const SCALE: usize>(
        device: &mut Device,
        lengths: &[usize],
    ) -> Vec<bf16> {
        let (mut pool, tables) = items::windowed::<H, D>(device, lengths, WINDOW).await;
        let empty = tempfile::tempdir().unwrap();
        let config = serde_json::json!({"rope_theta": 1e4, "max_position_embeddings": WINDOW});
        let checkpoint = Checkpoint::new(empty.path(), Some(config)).unwrap();
        let preparation: Paged<R, H, D> = checkpoint.load(device).unwrap();
        // Each row's table from a whole table row, the last ID repeated to its end.
        let (mut segments, mut blocks) = (vec![0; R::SIZE * FIELDS], Vec::new());
        for (row, (&length, table)) in lengths.iter().zip(&tables).enumerate() {
            let segment = [row, 1, length - 1, blocks.len(), table.len()];
            segments[row * FIELDS..][..FIELDS].copy_from_slice(&segment.map(|word| word as i32));
            let padded = table.len().next_multiple_of(BlockIndex::SIZE);
            blocks.extend((0..padded).map(|at| table[at.min(table.len() - 1)]));
        }
        blocks.resize(Table::SIZE, 0);
        let history = History {
            segments: HostTensor::<i32, m![R, Fields]>::from_vec(segments)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            blocks: HostTensor::<i32, m![Table]>::from_vec(blocks)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            selected: HostTensor::<i32, m![R]>::zero()
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            extents: HostTensor::<i32, m![3]>::zero()
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
        };
        let queries = items::queries::<H, G, D>(R::SIZE);
        let queries = HostTensor::<bf16, Layout<R, H, G, D>>::from_vec(queries);
        let queries: Queries<R, H, G, D> = queries.to_hbm(&mut device.pdma).await.unwrap();
        let live = count(lengths.len()).to_hbm(&mut device.pdma).await.unwrap();
        let mut attended: Queries<R, H, G, D> = HostTensor::<bf16, Layout<R, H, G, D>>::zero()
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let arguments = (
            &mut *device,
            &F32::<SCALE>,
            &queries,
            (&preparation, &history),
            &live,
            &mut pool,
            &mut attended,
        );
        launch(attend_paged, arguments).await.unwrap();
        attended
            .to_host::<Layout<R, H, G, D>>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec()
    }

    /// Through a device paging of a window of [`WINDOW`] keys, as the catalog's layers page their
    /// history, the windowed leaf's rows of 128-wide heads in four query groups and of 256-wide
    /// heads in two attend only their window as a host softmax does, though poison fills their
    /// keys below it and the null block takes the place of every block wholly below it.
    #[tokio::test(flavor = "current_thread")]
    async fn attends_only_each_row_window_through_its_pages() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(attend_paged.topology()).unwrap();
        let device = &mut device;
        let windowed = items::WINDOWED;
        let seen = |row: usize, length: usize| (row, items::lowest(length, WINDOW)..length);
        let mixed: Vec<usize> = (0..16).map(|row| windowed[row % windowed.len()]).collect();
        for lengths in [vec![2 * WINDOW + 17], windowed.to_vec(), mixed] {
            let output = by_paging::<Rows<16>, KvHeads, Groups, Head, SCALE>;
            let output = output(device, &lengths).await;
            check_rows::<KvHeads, Groups, Head, SCALE>((&lengths, seen), &output);
            let output = by_paging::<Rows<32>, KvHeads, Pairs, Wide, UNIT>;
            let output = output(device, &lengths).await;
            check_rows::<KvHeads, Pairs, Wide, UNIT>((&lengths, seen), &output);
        }
    }
}
