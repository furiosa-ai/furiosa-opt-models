use super::Attention;
use super::append::{self, Keys512, Keys1024, Keys2048, SingleTile, TileGroup, TilePair};
use crate::Module;
use crate::ops::index::gather::Gather;
use crate::tensor::cache::Block;
use crate::tensor::cache::Blocks;
use crate::tensor::parallel::Chip;
use crate::tensor::scalar::Value;
pub(crate) use append::Causal;
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::HbmBuffer;
use std::io;
use std::marker::PhantomData;

axes![Query = 1024];
// A chunk attends in tiles of `Cache` rows.
axes![Cache = 128];

/// The tiles of history a chunk attends at most: those of `W` 512-key windows.
#[derive(Clone, Debug)]
pub struct Chunks<W>(PhantomData<fn() -> W>);

impl<W: AxisName> AxisName for Chunks<W> {
    const NAME: Ident = Ident::new("Chunks");
    const SIZE: usize = W::SIZE * Keys512::SIZE / Cache::SIZE;
}

/// The identity slot table of chunks over `W` windows: every key's position in a full history.
pub type SlotTable<W> = HbmTensor<i32, Chip, m![{ Symbol<Chunks<W>> }, Cache]>;

impl<W: AxisName> Chunks<W> {
    /// Block table entries of one chunk: every block of a full history.
    const TABLE: usize = Self::SIZE * Cache::SIZE / Block::SIZE;
    const KEYS: usize = Self::SIZE * Cache::SIZE;
    /// A chunk's shape word: each query domain's record count (pairs, four-tile groups, single
    /// tiles), then each history run's block count (double, wide, early, rest, half, narrow,
    /// partial).
    const SHAPE: [usize; 10] = [
        DOMAINS[0],
        DOMAINS[1],
        DOMAINS[2],
        Self::KEYS / Keys2048::SIZE + 1,
        2,
        2,
        Self::KEYS / Keys1024::SIZE + 1,
        2,
        4,
        2,
    ];
    /// A chunk's begins words: the first block of each run that does not start at block 0 (wide,
    /// rest, half, narrow, partial).
    const BEGINS: [usize; 5] = [
        Self::KEYS / Keys1024::SIZE + 1,
        2,
        Self::KEYS / Keys512::SIZE + 1,
        Self::SIZE + 1,
        Self::SIZE + 1,
    ];
    /// Words of a chunk's work-table row: shape, begins, then one a record slot, small counts
    /// packed low to high under their radix so the table fits the scalar tables a kernel stages
    /// once.
    const ROW: usize = 1 + Field::words(&Self::BEGINS) + SLOTS;
}

/// The records a chunk holds at most: one pair, two four-tile groups, two single tiles.
const SLOTS: usize = 5;
/// A record's word: its first tile, its preceding whole fresh blocks, the first and the count of
/// its fresh panels before it within its own block, its guest count, a bit for each of its tiles
/// set where the tile holds rows past the chunk's last, and that last row within the tile.
const RECORD: [usize; 7] = [8, 2, 5, 4, 2, 1 << TilePair::SIZE, Cache::SIZE];
const _: () = assert!(Field::words(&RECORD) == 1);

/// A field of words packing fields below their radices in turn, a field in the next word where
/// the current one would pass I32: its word, divisor and radix.
#[derive(Clone, Copy)]
struct Field {
    word: usize,
    divisor: usize,
    radix: usize,
}

impl Field {
    /// Field `k` of the words packing fields below `radices`.
    const fn new(radices: &[usize], k: usize) -> Self {
        let mut field = Self {
            word: 0,
            divisor: 1,
            radix: radices[0],
        };
        let mut at = 0;
        while at < k {
            at += 1;
            let divisor = field.divisor * field.radix;
            field = match divisor * radices[at] <= i32::MAX as usize + 1 {
                true => Self {
                    word: field.word,
                    divisor,
                    radix: radices[at],
                },
                false => Self {
                    word: field.word + 1,
                    divisor: 1,
                    radix: radices[at],
                },
            };
        }
        field
    }

    /// The words packing fields below `radices`.
    const fn words(radices: &[usize]) -> usize {
        Self::new(radices, radices.len() - 1).word + 1
    }

    /// The largest word `word` of the words packing fields below `radices`.
    const fn largest(radices: &[usize], word: usize) -> usize {
        let (mut at, mut product) = (0, 1);
        while at < radices.len() {
            if Self::new(radices, at).word == word {
                product *= radices[at];
            }
            at += 1;
        }
        product - 1
    }

    /// The field of `word`.
    fn unpack(self, word: usize) -> usize {
        word / self.divisor % self.radix
    }
}

/// The largest shape word of a chunk over `W` windows: the bound the epoch reads it under.
#[derive(Clone, Debug)]
struct Shape<W>(PhantomData<W>);

impl<W: AxisName> AxisName for Shape<W> {
    const NAME: Ident = Ident::new("Shape");
    const SIZE: usize = Field::largest(&Chunks::<W>::SHAPE, 0);
}

/// The largest begins word `N` of a chunk over `W` windows.
#[derive(Clone, Debug)]
struct Begins<W, const N: usize>(PhantomData<W>);

impl<W: AxisName, const N: usize> AxisName for Begins<W, N> {
    const NAME: Ident = Ident::new("Begins");
    const SIZE: usize = Field::largest(&Chunks::<W>::BEGINS, N);
}

/// A query domain: the records of a chunk whose tiles one `Self`-tile body attends, its count a
/// field of the chunk's shape word and its records from a slot of the chunk's row.
pub(crate) trait Domain: AxisName {
    const COUNT: usize;
    const SLOT: usize;
}
impl Domain for TilePair {
    const COUNT: usize = 0;
    const SLOT: usize = 0;
}
impl Domain for TileGroup {
    const COUNT: usize = 1;
    const SLOT: usize = TilePair::SLOT + DOMAINS[TilePair::COUNT] - 1;
}
impl Domain for SingleTile {
    const COUNT: usize = 2;
    const SLOT: usize = TileGroup::SLOT + DOMAINS[TileGroup::COUNT] - 1;
}
/// The radices of the shape word's record counts, the same for every history.
const DOMAINS: [usize; 3] = [2, 3, 3];
const _: () = assert!(SingleTile::SLOT + DOMAINS[SingleTile::COUNT] - 1 == SLOTS);

/// The words of the work table of `S` chunks over `W` windows, one row each.
#[derive(Clone, Debug)]
pub struct Words<S, W>(PhantomData<(S, W)>);

impl<S: AxisName, W: AxisName> AxisName for Words<S, W> {
    const NAME: Ident = Ident::new("Words");
    const SIZE: usize = S::SIZE * Chunks::<W>::ROW;
}

/// The metadata of the prompt chunks of one launch, one entry per slot of `Sequences`, each over
/// at most `W` windows of history.
#[derive(DeviceSend)]
pub struct Invocation<Sequences: AxisName, W: AxisName> {
    /// The number of live sequences, which come first.
    pub live: Value,
    pub h: HbmTensor<i32, Chip, m![Sequences]>,
    pub lanes: HbmTensor<i32, Chip, m![Sequences, Cache]>,
    pub start: HbmTensor<i32, Chip, m![Sequences]>,
    pub tables: HbmTensor<i32, Chip, m![Sequences, { Symbol<Chunks<W>> }, Cache / Block]>,
    /// Each chunk's query records and the runs of its history, as [`Plan::reads`] plans them.
    pub work: HbmTensor<i32, Chip, m![{ Symbol<Words<Sequences, W>> }]>,
    pub slots: SlotTable<W>,
    pub causal: append::Causal,
}

/// A launch's prompt chunks as the arrays the chunk body reads, at the capacities of `Sequences`
/// chunks over `W` windows; the masks and the slot table are every launch's, placed once.
#[derive(Clone, Debug)]
pub struct Plan<W> {
    /// Chunks of the launch.
    pub count: usize,
    /// Per chunk: its rows, the history before them and its first row.
    pub q: Vec<i32>,
    pub h: Vec<i32>,
    pub start: Vec<i32>,
    /// Per chunk: its history length repeated across a tile.
    pub lanes: Vec<i32>,
    /// Per chunk: the pool byte offset of each block of its history and rows.
    pub tables: Vec<i32>,
    history: PhantomData<W>,
}

impl<W: AxisName> Plan<W> {
    /// No chunk yet, at the capacities of `Sequences` chunks.
    pub fn new<Sequences: AxisName>() -> Self {
        Self {
            count: 0,
            q: vec![0; Sequences::SIZE],
            h: vec![0; Sequences::SIZE],
            start: vec![0; Sequences::SIZE],
            lanes: vec![0; Sequences::SIZE * Cache::SIZE],
            tables: vec![0; Sequences::SIZE * Chunks::<W>::TABLE],
            history: PhantomData,
        }
    }

    /// Each array in [`Invocation::bind`] order: the leading words a launch uses, and all its
    /// words. A launch uses its chunks' rows of the tables only.
    pub fn reads(&self) -> io::Result<[(Vec<i32>, usize); 6]> {
        let all = |values: &[i32]| (values.to_vec(), values.len());
        let used = |values: &[i32], row: usize| (values[..self.count * row].to_vec(), values.len());
        // The work table: a row per chunk of its shape, its runs' first blocks and its records. It
        // fails before anything is staged when a chunk holds more rows or history than a chunk can.
        let work = || -> io::Result<Vec<i32>> {
            let pack = |radices: &[usize], values: &[usize]| {
                let mut words = vec![0; Field::words(radices)];
                for (k, &value) in values.iter().enumerate() {
                    let field = Field::new(radices, k);
                    assert!(value < field.radix, "a field fits its radix");
                    words[field.word] += value * field.divisor;
                }
                (words.into_iter())
                    .map(|word| i32::try_from(word).expect("a packed word fits I32"))
                    .collect::<Vec<_>>()
            };
            let (row_words, begins) = (Chunks::<W>::ROW, Field::words(&Chunks::<W>::BEGINS));
            let mut table = vec![0; self.q.len() * row_words];
            for (chunk, row) in table
                .chunks_exact_mut(row_words)
                .enumerate()
                .take(self.count)
            {
                let fact =
                    |values: &[i32]| usize::try_from(values[chunk]).map_err(io::Error::other);
                let (q, h) = (fact(&self.q)?, fact(&self.h)?);
                if q > Query::SIZE || h + q > Chunks::<W>::KEYS {
                    return Err(io::Error::other(format!(
                        "a chunk of {q} rows over {h} keys exceeds a chunk's capacity"
                    )));
                }
                let tiles = q.div_ceil(Cache::SIZE);
                let paired = tiles / TilePair::SIZE * TilePair::SIZE;
                let left = tiles - paired;
                // Three remainder tiles run as a group with a dead fourth, since single tiles read
                // each history block thrice; the fourth's rows are the next chunk's or a sink's.
                let domains: [Vec<usize>; 3] = [
                    (0..paired).step_by(TilePair::SIZE).collect(),
                    (0..(left + 1) / TileGroup::SIZE)
                        .map(|g| paired + g * TileGroup::SIZE)
                        .collect(),
                    (0..tiles % TileGroup::SIZE % 3)
                        .map(|r| tiles / TileGroup::SIZE * TileGroup::SIZE + r)
                        .collect(),
                ];
                // A record attends after the whole fresh blocks and the panels of its own block
                // before its first tile; the first record of the first chunk writes the guest.
                let sizes = [TilePair::SIZE, TileGroup::SIZE, SingleTile::SIZE];
                let records = domains.iter().zip(sizes).flat_map(|(firsts, size)| {
                    firsts.iter().map(move |&first| {
                        let block = first / TileGroup::SIZE;
                        let within = block * TileGroup::SIZE;
                        let guest = usize::from(chunk == 0 && first == 0);
                        let (tile, rows) = (q / Cache::SIZE, q % Cache::SIZE);
                        let (cut, last) = match rows != 0 && (first..first + size).contains(&tile) {
                            true => (1 << (tile - first), rows - 1),
                            false => (0, 0),
                        };
                        pack(
                            &RECORD,
                            &[first, block, within, first - within, guest, cut, last],
                        )[0]
                    })
                });
                let slots = [TilePair::SLOT, TileGroup::SLOT, SingleTile::SLOT];
                let slots = domains
                    .iter()
                    .zip(slots)
                    .flat_map(|(firsts, slot)| slot..slot + firsts.len());
                for (slot, record) in slots.zip(records) {
                    row[1 + begins + slot] = record;
                }
                // A width's blocks lie between the next wider width's whole blocks and `h`; a
                // pair's are 1024 keys wide, as its owners hold no 2048-key block.
                let below = |keys: usize, wider: usize| {
                    let begin = h / wider * wider / keys;
                    (begin, h / keys - begin)
                };
                let first = (h / Keys1024::SIZE).min(1);
                let panels = h / Cache::SIZE;
                let (wide, rest) = (
                    below(Keys1024::SIZE, Keys2048::SIZE),
                    (first, h / Keys1024::SIZE - first),
                );
                let (half, narrow) = (
                    below(Keys512::SIZE, Keys1024::SIZE),
                    below(Cache::SIZE, Keys512::SIZE),
                );
                let partial = (panels, h.div_ceil(Cache::SIZE) - panels);
                let runs = [
                    h / Keys2048::SIZE,
                    wide.1,
                    first,
                    rest.1,
                    half.1,
                    narrow.1,
                    partial.1,
                ];
                let counts = domains.iter().map(Vec::len).chain(runs).collect::<Vec<_>>();
                row[0] = pack(&Chunks::<W>::SHAPE, &counts)[0];
                let starts = [wide.0, rest.0, half.0, narrow.0, partial.0];
                row[1..1 + begins].copy_from_slice(&pack(&Chunks::<W>::BEGINS, &starts));
            }
            Ok(table)
        };
        Ok([
            (
                vec![i32::try_from(self.count).map_err(io::Error::other)?],
                1,
            ),
            all(&self.h),
            used(&self.lanes, Cache::SIZE),
            all(&self.start),
            used(&self.tables, Chunks::<W>::TABLE),
            all(&work()?),
        ])
    }
}

/// The causal masks and the identity slot table every launch's chunks over `W` windows read.
pub struct Masks<W> {
    causal: HbmBuffer,
    slots: HbmBuffer,
    history: PhantomData<fn() -> W>,
}

impl<W: AxisName> Masks<W> {
    /// The masks, placed once.
    pub async fn new(device: &mut Device) -> io::Result<Self> {
        let tile = Cache::SIZE;
        let causal: Causal = HostTensor::<i8, m![Cache, Cache]>::from_vec(
            (0..tile * tile).map(|at| i8::from(at % tile > at / tile)),
        )
        .to_hbm(&mut device.pdma)
        .await
        .map_err(io::Error::other)?;
        let slots: SlotTable<W> = HostTensor::<i32, m![{ Symbol<Chunks<W>> }, Cache]>::from_vec(
            (0..Chunks::<W>::KEYS).map(|at| i32::try_from(at).expect("a position fits I32")),
        )
        .to_hbm(&mut device.pdma)
        .await
        .map_err(io::Error::other)?;
        Ok(Self {
            causal: causal.into_buffer(),
            slots: slots.into_buffer(),
            history: PhantomData,
        })
    }
}

impl<Sequences: AxisName, W: AxisName> Invocation<Sequences, W> {
    /// The chunks whose arrays [`Plan::reads`] lists, as `next` names each in turn, with `masks`.
    pub fn bind(
        mut next: impl FnMut() -> io::Result<HbmBuffer>,
        masks: &Masks<W>,
    ) -> io::Result<Self> {
        use crate::tensor::memory::alias;
        Ok(Self {
            live: alias(&next()?, ..)?,
            h: alias(&next()?, ..)?,
            lanes: alias(&next()?, ..)?,
            start: alias(&next()?, ..)?,
            tables: alias(&next()?, ..)?,
            work: alias(&next()?, ..)?,
            slots: alias(&masks.slots, ..)?,
            causal: alias(&masks.causal, ..)?,
        })
    }
}

/// A chunk's work as its epoch reads it from the launch's work table, each word a staged scalar:
/// its shape and its runs' first blocks once, a record's word where the record is attended.
pub(crate) struct Epoch<'a, Sequences: AxisName, W: AxisName> {
    work: &'a HbmTensor<i32, Chip, m![{ Symbol<Words<Sequences, W>> }]>,
    row: usize,
    runs: Runs<W>,
}

impl<'a, Sequences: AxisName, W: AxisName> Epoch<'a, Sequences, W> {
    /// Chunk `chunk`'s row of `work`.
    pub(crate) fn new(
        work: &'a HbmTensor<i32, Chip, m![{ Symbol<Words<Sequences, W>> }]>,
        chunk: usize,
    ) -> Self {
        const { assert!(Field::words(&Chunks::<W>::SHAPE) == 1) };
        let row = chunk * Chunks::<W>::ROW;
        let first = Self::word::<m![{ Symbol<Begins<W, 0>> }]>(work, row + 1);
        // A history whose begins pass one word reads its second word too.
        let second = match const { Field::words(&Chunks::<W>::BEGINS) } {
            1 => first,
            _ => Self::word::<m![{ Symbol<Begins<W, 1>> }]>(work, row + 2),
        };
        let runs = Runs {
            shape: Self::word::<m![{ Symbol<Shape<W>> }]>(work, row),
            begins: [first, second],
            history: PhantomData,
        };
        Self { work, row, runs }
    }

    /// The chunk's record `index` of query domain `D`.
    pub(crate) fn record<D: Domain>(&self, index: usize) -> Record {
        let at = self.row + 1 + const { Field::words(&Chunks::<W>::BEGINS) } + D::SLOT + index;
        Record(Self::word::<Broadcast<{ Field::largest(&RECORD, 0) }>>(
            self.work, at,
        ))
    }

    /// The runs of the chunk's history.
    pub(crate) fn runs(&self) -> Runs<W> {
        self.runs
    }

    /// Word `at` of `work`, which the host bounds by `Max::SIZE`.
    fn word<Max: M>(
        work: &HbmTensor<i32, Chip, m![{ Symbol<Words<Sequences, W>> }]>,
        at: usize,
    ) -> usize {
        // SAFETY: one word of a table nothing writes.
        let word: Value = unsafe {
            work.view()
                .tile::<m![{ Symbol<Words<Sequences, W>> }], 1,
                    m![{ Symbol<Words<Sequences, W>> } = 1 # ({ Symbol<Words<Sequences, W>> })]>(at)
                .alias()
        };
        word.extent::<Max>()
    }
}

/// A chunk's history over at most `W` windows as runs of whole blocks of each width, in the order
/// a query attends them, and its partial last 128-key panel: its shape and begins words.
pub(crate) struct Runs<W> {
    shape: usize,
    begins: [usize; 2],
    history: PhantomData<W>,
}

impl<W> Clone for Runs<W> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<W> Copy for Runs<W> {}

impl<W: AxisName> Runs<W> {
    /// The chunk's records in query domain `D`.
    pub(crate) fn records<D: Domain>(&self) -> usize {
        const { Field::new(&Chunks::<W>::SHAPE, D::COUNT) }.unpack(self.shape)
    }

    /// A four-tile group's or single tile's 2048-key blocks.
    pub(crate) fn double(&self) -> Run {
        Run::first(const { Field::new(&Chunks::<W>::SHAPE, 3) }.unpack(self.shape))
    }

    /// A four-tile group's or single tile's 1024-key block after its 2048-key blocks.
    pub(crate) fn wide(&self) -> Run {
        self.after(
            const { Field::new(&Chunks::<W>::SHAPE, 4) }.unpack(self.shape),
            const { Field::new(&Chunks::<W>::BEGINS, 0) },
        )
    }

    /// A pair's first 1024-key block, which its early gather writes.
    pub(crate) fn early(&self) -> Run {
        Run::first(const { Field::new(&Chunks::<W>::SHAPE, 5) }.unpack(self.shape))
    }

    /// A pair's 1024-key blocks after its first.
    pub(crate) fn rest(&self) -> Run {
        self.after(
            const { Field::new(&Chunks::<W>::SHAPE, 6) }.unpack(self.shape),
            const { Field::new(&Chunks::<W>::BEGINS, 1) },
        )
    }

    /// The 512-key block below the last 1024-key block.
    pub(crate) fn half(&self) -> Run {
        self.after(
            const { Field::new(&Chunks::<W>::SHAPE, 7) }.unpack(self.shape),
            const { Field::new(&Chunks::<W>::BEGINS, 2) },
        )
    }

    /// The whole 128-key panels below the last 512-key block.
    pub(crate) fn narrow(&self) -> Run {
        self.after(
            const { Field::new(&Chunks::<W>::SHAPE, 8) }.unpack(self.shape),
            const { Field::new(&Chunks::<W>::BEGINS, 3) },
        )
    }

    /// The partial last 128-key panel, when the history ends inside one.
    pub(crate) fn partial(&self) -> Run {
        self.after(
            const { Field::new(&Chunks::<W>::SHAPE, 9) }.unpack(self.shape),
            const { Field::new(&Chunks::<W>::BEGINS, 4) },
        )
    }

    /// The `count` blocks from the begins words' field at `begin`.
    fn after(&self, count: usize, begin: Field) -> Run {
        Run {
            count,
            word: self.begins[begin.word],
            begin,
        }
    }
}

/// A query record of a chunk: its word.
#[derive(Clone, Copy)]
pub(crate) struct Record(usize);

impl Record {
    /// The first tile the record attends; a chunk's pair `r` attends from tile `r` times eight.
    pub(crate) fn first(&self) -> usize {
        const { Field::new(&RECORD, 0) }.unpack(self.0)
    }

    /// The whole fresh blocks before its tiles: at most one, the previous group's own panels.
    pub(crate) fn preceding(&self) -> usize {
        const { Field::new(&RECORD, 1) }.unpack(self.0)
    }

    /// The fresh panels before its tile within its own block.
    pub(crate) fn within(&self) -> Run {
        Run {
            count: const { Field::new(&RECORD, 3) }.unpack(self.0),
            word: self.0,
            begin: const { Field::new(&RECORD, 2) },
        }
    }

    /// The times it writes the guest: once for the first record of the first chunk.
    pub(crate) fn guest(&self) -> usize {
        const { Field::new(&RECORD, 4) }.unpack(self.0)
    }

    /// 1 when its tile `t` holds rows past the chunk's last row, else 0.
    pub(crate) fn cut(&self, t: usize) -> usize {
        let bits = const { Field::new(&RECORD, 5) };
        Field {
            word: 0,
            divisor: bits.divisor << t,
            radix: 2,
        }
        .unpack(self.0)
    }

    /// The chunk's last row within its cut tile.
    pub(crate) fn last(&self) -> usize {
        const { Field::new(&RECORD, 6) }.unpack(self.0)
    }
}

/// A run of a chunk's blocks or panels: how many, and the first, unpacked where it is used.
#[derive(Clone, Copy)]
pub(crate) struct Run {
    count: usize,
    word: usize,
    begin: Field,
}

impl Run {
    /// The `count` blocks from block 0.
    fn first(count: usize) -> Self {
        Self {
            count,
            word: 0,
            begin: Field {
                word: 0,
                divisor: 1,
                radix: 1,
            },
        }
    }

    pub(crate) fn count(&self) -> usize {
        self.count
    }

    pub(crate) fn begin(&self) -> usize {
        self.begin.unpack(self.word)
    }
}

/// Attends the `live` sequences of `invocation`, each from its start row of the batch `All`; with
/// `keeps` 1 the first record of the first runs `boundary` on its state under its causal block.
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
                &HbmTensor<bf16, Chip, m![All, KvHeads, Groups, Head]>,
                &HbmTensor<bf16, Chip, m![All, KvHeads, Head]>,
                &HbmTensor<bf16, Chip, m![All, KvHeads, Head]>,
            ),
            (&Invocation<Sequences, W>, usize),
            (&Gather, &Blocks<KvHeads, Head>),
            &mut append::Output<KvHeads, Groups, Head, All>,
            ((&H, &mut S), usize),
        ),
        Env,
        &mut (),
    > for Attention<KvHeads, Groups, Head>
where
    H: for<'d, 's> Module<&'s mut S, Env, &'d mut (), Output = ()>,
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        env: Env,
        _: &mut (),
        ((query, key, value), (invocation, live), pool, attended, ((boundary, state), keeps)): (
            (
                &HbmTensor<bf16, Chip, m![All, KvHeads, Groups, Head]>,
                &HbmTensor<bf16, Chip, m![All, KvHeads, Head]>,
                &HbmTensor<bf16, Chip, m![All, KvHeads, Head]>,
            ),
            (&Invocation<Sequences, W>, usize),
            (&Gather, &Blocks<KvHeads, Head>),
            &mut append::Output<KvHeads, Groups, Head, All>,
            ((&H, &mut S), usize),
        ),
    ) {
        for sequence in 0..live {
            // SAFETY: each alias reads one immutable metadata element; the host bounds all counts.
            let (h, start): (Value, Value) = unsafe {
                (
                    invocation
                        .h
                        .view()
                        .tile::<m![Sequences], 1, m![Sequences = 1 # Sequences]>(sequence)
                        .alias(),
                    invocation
                        .start
                        .view()
                        .tile::<m![Sequences], 1, m![Sequences = 1 # Sequences]>(sequence)
                        .alias(),
                )
            };
            // SAFETY: each immutable sequence window contains one complete table allocation.
            let (table, lanes): (append::Table<W>, HbmTensor<i32, Chip, m![Cache]>) = unsafe {
                (invocation.tables.view().tile::<m![Sequences], 1,
                    m![Sequences = 1 # Sequences, { Symbol<Chunks<W>> }, Cache / Block]>(sequence)
                    .alias(),
                  invocation.lanes.view().tile::<m![Sequences], 1,
                    m![Sequences = 1 # Sequences, Cache]>(sequence).alias())
            };
            // SAFETY: the producer query's head product is contiguous and matches the consumer.
            let query = unsafe {
                query
                    .view()
                    .reshape::<Chip, m![All, KvHeads, Groups, Head]>()
                    .alias()
            };
            append::Attend.forward(
                &mut *context,
                env,
                &mut (),
                (
                    (
                        append::Input {
                            attention: self,
                            query: &query,
                            key,
                            value,
                            start: start.extent::<m![All]>(),
                            causal: &invocation.causal,
                            table: &table,
                            history: &h,
                            slots: &invocation.slots,
                            lanes: &lanes,
                        },
                        &Epoch::new(&invocation.work, sequence),
                    ),
                    pool,
                    &mut *attended,
                    ((boundary, &mut *state), keeps),
                ),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::axis::Rows;

    // The history of a 40960-position context, and of a 131072-position one.
    axes![Shared = 80, Long = 256];

    /// A query record of the work table: its domain's tiles, and its fields.
    type Decoded = (usize, [usize; RECORD.len()]);

    /// Each chunk's records in the order the domains' bodies attend them, and its history runs, as
    /// the bodies unpack them from `plan`'s work table.
    fn decode<W: AxisName>(plan: &Plan<W>) -> Vec<(Vec<Decoded>, [[usize; 2]; 7])> {
        let [.., (work, words)] = plan.reads().unwrap();
        let row = Chunks::<W>::ROW;
        assert_eq!(
            (work.len(), words),
            (plan.q.len() * row, plan.q.len() * row)
        );
        let word = |chunk: usize, at: usize| usize::try_from(work[chunk * row + at]).unwrap();
        let begins = Field::words(&Chunks::<W>::BEGINS);
        let mut chunks = Vec::new();
        for chunk in 0..plan.count {
            let shape = word(chunk, 0);
            let starts: Vec<usize> = (0..begins).map(|at| word(chunk, 1 + at)).collect();
            assert!(shape <= Field::largest(&Chunks::<W>::SHAPE, 0));
            for (at, &start) in starts.iter().enumerate() {
                assert!(start <= Field::largest(&Chunks::<W>::BEGINS, at));
            }
            let runs = Runs::<W> {
                shape,
                begins: [starts[0], starts[begins - 1]],
                history: PhantomData,
            };
            let domains = [
                (runs.records::<TilePair>(), TilePair::SIZE, TilePair::SLOT),
                (
                    runs.records::<TileGroup>(),
                    TileGroup::SIZE,
                    TileGroup::SLOT,
                ),
                (
                    runs.records::<SingleTile>(),
                    SingleTile::SIZE,
                    SingleTile::SLOT,
                ),
            ];
            let mut records = Vec::new();
            for (count, tiles, slot) in domains {
                for r in 0..count {
                    let record = Record(word(chunk, 1 + begins + slot + r));
                    let within = record.within();
                    let fields = [record.first(), record.preceding(), within.begin()];
                    let [first, preceding, begin] = fields;
                    let cut = (0..TilePair::SIZE).map(|t| record.cut(t) << t).sum();
                    records.push((
                        tiles,
                        [
                            first,
                            preceding,
                            begin,
                            within.count(),
                            record.guest(),
                            cut,
                            record.last(),
                        ],
                    ));
                }
            }
            let runs = [
                runs.double(),
                runs.wide(),
                runs.early(),
                runs.rest(),
                runs.half(),
                runs.narrow(),
                runs.partial(),
            ];
            chunks.push((records, runs.map(|run| [run.begin(), run.count()])));
        }
        chunks
    }

    /// A launch of chunks of `(rows, history)` packed from row `start`.
    fn launch<W: AxisName>(chunks: &[(usize, usize)], start: usize) -> Plan<W> {
        let mut plan = Plan::new::<Rows<8>>();
        let mut row = start;
        for (index, &(q, h)) in chunks.iter().enumerate() {
            (plan.q[index], plan.h[index], plan.start[index]) = (q as i32, h as i32, row as i32);
            row += q;
        }
        plan.count = chunks.len();
        plan
    }

    /// The keys, as half-open intervals of a chunk's history keys then its rows' keys, that live
    /// row `j` of record `record` attends over a history of `h` keys with `runs`.
    fn keys(
        (tiles, words): Decoded,
        runs: [[usize; 2]; 7],
        (h, j): (usize, usize),
    ) -> Vec<(usize, usize)> {
        let [first, preceding, within, panels, ..] = words;
        let [double, wide, early, rest, half, narrow, partial] = runs;
        let blocks =
            |[begin, count]: [usize; 2], keys: usize| (begin * keys, (begin + count) * keys);
        let tail = (
            partial[0] * Cache::SIZE,
            ((partial[0] + partial[1]) * Cache::SIZE).min(h),
        );
        let mut keys = match tiles {
            8 => vec![(0, early[1] * Keys1024::SIZE), blocks(rest, Keys1024::SIZE)],
            _ => vec![blocks(double, Keys2048::SIZE), blocks(wide, Keys1024::SIZE)],
        };
        keys.extend([
            blocks(half, Keys512::SIZE),
            blocks(narrow, Cache::SIZE),
            tail,
        ]);
        let tile = j / Cache::SIZE;
        // A pair attends as two four-tile causal blocks, its second after its first.
        let block = first + (tile - first) / TileGroup::SIZE * TileGroup::SIZE;
        let fresh = |from: usize, to: usize| (h + from, h + to);
        keys.extend([
            fresh(block * Cache::SIZE, j + 1),
            fresh(first * Cache::SIZE, block * Cache::SIZE),
            fresh(0, preceding * Keys512::SIZE),
            fresh(within * Cache::SIZE, (within + panels) * Cache::SIZE),
        ]);
        keys.retain(|(from, to)| from < to);
        keys.sort_unstable();
        keys
    }

    /// Every live row of every chunk of `plan` attends each key of its history and of its chunk's
    /// rows up to its own exactly once, through one record; a record reads no word its body
    /// skips, and the run counts and starts fit the bounds its body reads them with.
    fn check_keys<W: AxisName>(plan: &Plan<W>) {
        let keys_of = |width: usize| Chunks::<W>::KEYS / width;
        for (chunk, (records, runs)) in decode(plan).into_iter().enumerate() {
            let (q, h) = (plan.q[chunk] as usize, plan.h[chunk] as usize);
            // Each run's blocks of its width in a full history, and the count its body reads.
            let (doubles, wides) = (keys_of(Keys2048::SIZE), keys_of(Keys1024::SIZE));
            let (halves, panels) = (keys_of(Keys512::SIZE), Chunks::<W>::SIZE);
            let bounds = [
                (doubles, doubles),
                (wides, 1),
                (wides, 1),
                (wides, wides),
                (halves, 1),
                (panels, 3),
                (panels, 1),
            ];
            for ([begin, count], (blocks, most)) in runs.iter().zip(bounds) {
                assert!(
                    begin + count <= blocks && *count <= most,
                    "chunk {chunk}: {runs:?}"
                );
            }
            let mut covered = vec![0; q];
            for (at, &(tiles, words @ [first, preceding, _, panels, ..])) in
                records.iter().enumerate()
            {
                match tiles {
                    8 => assert_eq!(words[..4], [8 * at, 0, 0, 0], "pair `at` from tile 8 at"),
                    4 => assert_eq!(panels, 0, "a group skips panels within its block"),
                    _ => {}
                }
                // The preceding block is the previous group's own panels.
                assert!(preceding == 0 || records[..at].iter().any(|&(t, w)| t == 4 && w[0] == 0));
                for j in (first * Cache::SIZE..(first + tiles) * Cache::SIZE).filter(|&j| j < q) {
                    covered[j] += 1;
                    let end =
                        keys((tiles, words), runs, (h, j))
                            .into_iter()
                            .fold(0, |at, (from, to)| {
                                assert_eq!(from, at, "chunk {chunk} ({q} rows over {h}): row {j}");
                                to
                            });
                    assert_eq!(end, h + j + 1, "chunk {chunk} ({q} rows over {h}): row {j}");
                }
            }
            assert!(
                covered.iter().all(|&rows| rows == 1),
                "chunk {chunk}: {q} rows over {h}"
            );
            // Only the tile holding the chunk's last row and rows past it is cut, at that row.
            let cuts = records.iter().flat_map(|&(_, [first, .., cut, last])| {
                (0..TilePair::SIZE)
                    .filter(move |t| cut >> t & 1 == 1)
                    .map(move |t| (first + t, last))
            });
            let tail = (q % Cache::SIZE != 0).then(|| (q / Cache::SIZE, q % Cache::SIZE - 1));
            assert!(cuts.eq(tail), "chunk {chunk}: {q} rows");
        }
    }

    /// A chunk of any row count over a history at, below and above every block width's edge, and
    /// up to its whole history, of a 40960-position context and a 131072-position one.
    #[test]
    fn attends_each_key_once() {
        fn attends<W: AxisName>() {
            let edges = [
                0, 1, 127, 128, 129, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 3000,
            ];
            let rows = [
                1, 2, 127, 128, 129, 255, 256, 257, 383, 384, 385, 511, 512, 513, 639, 640,
            ];
            let rows = rows
                .into_iter()
                .chain([641, 767, 768, 769, 895, 896, 897, 1023, 1024]);
            for q in rows {
                for h in edges.into_iter().chain([Chunks::<W>::KEYS - q]) {
                    check_keys(&launch::<W>(&[(q, h)], 0));
                }
            }
            for q in 0..=Query::SIZE {
                check_keys(&launch::<W>(&[(q, 0), (q, 1000)], 0));
            }
        }
        attends::<Shared>();
        attends::<Long>();
    }

    /// Adjacent chunks with unaligned starts, after item rows or none: every live row's output is
    /// last written by its own record, so a chunk's padding tiles may hold the next chunk's rows,
    /// and only the first record of the first chunk writes the guest.
    #[test]
    fn writes_each_live_row_last() {
        let launches = [
            vec![(2, 0), (384, 100)],
            vec![(384, 0), (128, 40)],
            vec![(640, 0), (128, 3000)],
            vec![(768, 0), (128, 0)],
            vec![(128, 0); 8],
            vec![(28, 1000)],
            vec![(992, 0)],
        ];
        for chunks in launches.iter().flat_map(|chunks| {
            let reversed = chunks.iter().rev().copied().collect::<Vec<_>>();
            [chunks.clone(), reversed]
        }) {
            for start in [0, 30] {
                let plan = launch::<Shared>(&chunks, start);
                check_keys(&plan);
                let mut writer = vec![None; 2 * Query::SIZE];
                let mut guests = Vec::new();
                for (chunk, (records, _)) in decode(&plan).into_iter().enumerate() {
                    let row = plan.start[chunk] as usize;
                    for (tiles, [first, _, _, _, guest, ..]) in records {
                        guests.extend((0..guest).map(|_| (chunk, first)));
                        for j in first * Cache::SIZE..(first + tiles) * Cache::SIZE {
                            writer[row + j] = Some((chunk, j));
                        }
                    }
                }
                for (chunk, &(q, _)) in chunks.iter().enumerate() {
                    let row = plan.start[chunk] as usize;
                    assert!(
                        (0..q).all(|j| writer[row + j] == Some((chunk, j))),
                        "{chunks:?}"
                    );
                }
                assert_eq!(guests, [(0, 0)], "{chunks:?} from row {start}");
            }
        }
    }

    /// A chunk of more rows or history than a chunk holds is refused before any word is staged.
    #[test]
    fn refuses_a_chunk_past_its_capacity() {
        fn refuses<W: AxisName>() {
            let keys = Chunks::<W>::KEYS;
            assert!(launch::<W>(&[(Query::SIZE + 1, 0)], 0).reads().is_err());
            assert!(launch::<W>(&[(1, keys)], 0).reads().is_err());
            let whole = launch::<W>(&[(Query::SIZE, keys - Query::SIZE)], 0);
            assert!(whole.reads().is_ok());
        }
        refuses::<Shared>();
        refuses::<Long>();
    }
}
