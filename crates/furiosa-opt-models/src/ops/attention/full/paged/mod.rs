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
pub use gqa::items::{Items, Plan, Tiling, Waves};

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
    /// tiled as `tiling` says: its used leading words, then all; `ids` holds row IDs, padded to 16.
    pub fn sequential<R: AxisName>(
        segments: &[[i32; FIELDS]],
        ids: &[i32],
        tiling: Tiling,
    ) -> io::Result<Vec<(Vec<i32>, usize)>> {
        let items = Plan::new::<KvHeads, G, Head, Waves, Sinked<R>>(segments, ids, tiling);
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
    use crate::ops::attention::Attention;
    use crate::ops::attention::Heads;
    use crate::ops::attention::gqa::{Cache, Pages};
    use crate::ops::index::table::{BlockIndex, PADDING_BIAS};
    use crate::specialize::attention::{HISTORY_WAVES, Strategy, WAVES};
    use crate::tensor::axis::Rows;
    use crate::tensor::cache::{Block, Blocks};
    use crate::tensor::live::tests::count;
    use crate::tensor::scalar::{F32, Value};
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;
    use gqa::items::tests::{
        self as items, Heads as KvHeads, KEYS, LONG, QueryEights as Eights, QueryGroups as Groups,
        QueryPairs as Pairs, QuerySingle as Single, QuerySixteens as Sixteens, Width as Head,
        stride,
    };

    axes![Sequences = 1, Windows = 80];

    /// The score multiplier the items' tests compile, `1 / sqrt(128)`, as FP32 bits.
    const SCALE: usize = items::SCALE.to_bits() as usize;

    type Layout<R, H, G> = m![R, H, G, Head];
    type Queries<R, H, G> = HbmTensor<bf16, Chip, Layout<R, H, G>>;
    /// The items' pool of eight KV heads, and each history's first block.
    type Histories = (items::Pool, Vec<usize>);

    #[device(chip = 1)]
    fn attend_rows<R: AxisName, H: AxisName, G: AxisName, const WAVES: usize>(
        device: &mut Device,
        query: &Queries<R, H, G>,
        work: &Work<Sequences, H, G, Head, Waves<WAVES>, Windows>,
        pool: &mut items::Pool<H>,
        attended: &mut Queries<R, H, G>,
    ) {
        let context = work.read::<R>();
        let attention = Attention {
            heads: PhantomData::<(H, G, Head)>,
            scale: items::SCALE,
        };
        let heads: Heads<R, H, G, Head> = query.to_dm(&mut device.tdma);
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
    fn attend_windows<R: AxisName, G: AxisName>(
        device: &mut Device,
        query: &Queries<R, KvHeads, G>,
        pages: &Pages<R, Windows>,
        live: &Value,
        pool: &mut items::Pool,
        attended: &mut Queries<R, KvHeads, G>,
    ) {
        let attention = Attention::<
            PhantomData<(KvHeads, G, Head)>,
            F32<SCALE>,
            { Strategy::Windows },
        >::default();
        let heads: Heads<R, KvHeads, G, Head> = query.to_dm(&mut device.tdma);
        let query = Live::<(), R>::read(live).with(heads);
        let output = Op::forward(&attention, &mut *device, (&query, &*pool, pages));
        output
            .value()
            .view()
            .to_hbm_view(&mut device.tdma, attended.view_mut());
    }

    /// Rows of `lengths` keys of `firsts`' pool as the windowed leaf reads them, in `R`'s rows:
    /// each window's block offsets, block counts and mask, and the windows past the first.
    async fn pages<R: AxisName>(
        device: &mut Device,
        lengths: &[usize],
        firsts: &[usize],
    ) -> Pages<R, Windows> {
        let (rows, windows, slots) = (R::SIZE, Windows::SIZE, BlockIndex::SIZE);
        let length = |row: usize| lengths.get(row).copied().unwrap_or(0);
        // Every entry names a readable block: past a row's blocks, its last one.
        let block = |at: usize| {
            let (row, at) = (at / (windows * slots), at % (windows * slots));
            let history = items::history(row, length(row).max(1));
            let last = firsts[history] + items::keys(history).div_ceil(Block::SIZE) - 1;
            let used = length(row).div_ceil(Block::SIZE).max(1);
            (last - at.min(used - 1)) as i32 * Blocks::<KvHeads, Head>::BLOCK_STRIDE
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
        let table = HostTensor::<i32, m![R, Windows, BlockIndex]>::from_vec(table);
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
        }
    }

    /// The largest distance from the f64 softmax of the rows of `lengths` keys of a call of `R`
    /// rows of `G` query groups a KV head, attended as work items within `WAVES` wide waves and by
    /// the windowed leaf, the catalog's two attention forms.
    async fn errors<R: AxisName, G: AxisName, const WAVES: usize>(
        device: &mut Device,
        (pool, firsts): &mut Histories,
        masks: &chunks::Masks<Windows>,
        lengths: &[usize],
    ) -> [f64; 2] {
        let queries = items::queries::<KvHeads, G>(R::SIZE);
        let queries = HostTensor::<bf16, Layout<R, KvHeads, G>>::from_vec(queries);
        let queries: Queries<R, KvHeads, G> = queries.to_hbm(&mut device.pdma).await.unwrap();
        let rows = lengths.iter().copied().enumerate();
        let plan = items::plan::<KvHeads, G, WAVES, Sinked<R>>(rows, firsts, Tiling::Adaptive);
        let chunks = chunks::Plan::<Windows>::new::<Sequences>().reads().unwrap();
        let reads = plan
            .reads()
            .chain(chunks.iter().map(|(read, words)| (&read[..], *words)));
        let mut buffers = items::place(device, reads).await;
        let work: Work<Sequences, KvHeads, G, Head, Waves<WAVES>, Windows> =
            Work::bind(|| Ok(buffers.next().expect("an array a buffer")), masks).unwrap();
        let pages = pages::<R>(device, lengths, firsts).await;
        let live = count(lengths.len()).to_hbm(&mut device.pdma).await.unwrap();
        let mut errors = [0.0; 2];
        for (form, error) in errors.iter_mut().enumerate() {
            let mut attended: Queries<R, KvHeads, G> =
                HostTensor::<bf16, Layout<R, KvHeads, G>>::zero()
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap();
            match form {
                0 => {
                    let arguments = (&mut *device, &queries, &work, &mut *pool, &mut attended);
                    launch(attend_rows, arguments).await.unwrap();
                }
                _ => {
                    let arguments = (
                        &mut *device,
                        &queries,
                        &pages,
                        &live,
                        &mut *pool,
                        &mut attended,
                    );
                    launch(attend_windows, arguments).await.unwrap();
                }
            }
            let output = attended
                .to_host::<Layout<R, KvHeads, G>>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            let width = stride::<KvHeads, G>();
            let row = |(row, &length): (usize, &usize)| {
                items::error::<KvHeads, G>(row, length, &output[row * width..][..width])
            };
            *error = lengths.iter().enumerate().map(row).fold(0.0, f64::max);
        }
        errors
    }

    /// Attended as work items, the rows of the catalog's calls lie at most twice as far from the
    /// f64 softmax as the windowed leaf's do, at 25, 1,024, 4,096, 16,384 and 40,960 keys and
    /// mixed lengths: 64-row calls of four query groups a KV head within [`HISTORY_WAVES`], 16-row
    /// calls of two within [`WAVES`].
    #[tokio::test(flavor = "current_thread")]
    async fn errs_at_most_twice_the_windowed_leaf() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(attend_windows.topology()).unwrap();
        let mut pool = items::pool::<KvHeads>(&mut device).await;
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
        (pool, firsts): &mut Histories,
        masks: &chunks::Masks<Windows>,
        lengths: &[usize],
    ) {
        let queries = items::queries::<KvHeads, G>(R::SIZE);
        let queries = HostTensor::<bf16, Layout<R, KvHeads, G>>::from_vec(queries);
        let queries: Queries<R, KvHeads, G> = queries.to_hbm(&mut device.pdma).await.unwrap();
        let rows = lengths.iter().copied().enumerate();
        let plan = items::plan::<KvHeads, G, WAVES, Sinked<R>>(rows, firsts, Tiling::Adaptive);
        let chunks = chunks::Plan::<Windows>::new::<Sequences>().reads().unwrap();
        let reads = plan
            .reads()
            .chain(chunks.iter().map(|(read, words)| (&read[..], *words)));
        let mut buffers = items::place(device, reads).await;
        let work: Work<Sequences, KvHeads, G, Head, Waves<WAVES>, Windows> =
            Work::bind(|| Ok(buffers.next().expect("an array a buffer")), masks).unwrap();
        let mut attended: Queries<R, KvHeads, G> =
            HostTensor::<bf16, Layout<R, KvHeads, G>>::zero()
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        let arguments = (&mut *device, &queries, &work, &mut *pool, &mut attended);
        launch(attend_rows, arguments).await.unwrap();
        let output = attended
            .to_host::<Layout<R, KvHeads, G>>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let rows = output.chunks_exact(stride::<KvHeads, G>());
        for ((row, &length), got) in lengths.iter().enumerate().zip(rows) {
            items::check::<KvHeads, G>(row, length, got, lengths.len());
        }
    }

    /// The 16- and 64-row envelopes of four query groups a KV head and the 16-row one of one, two,
    /// eight and sixteen, all rows live or some, of one history length or many; 64 rows of the
    /// whole history within [`HISTORY_WAVES`].
    #[tokio::test(flavor = "current_thread")]
    async fn attends_every_live_row_of_a_sequential_layer() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(attend_rows.topology()).unwrap();
        let mut pool = items::pool::<KvHeads>(&mut device).await;
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
}
