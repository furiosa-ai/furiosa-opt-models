use furiosa_opt_std::prelude::*;

use crate::models::nemo::serving::{mlp, mlp::Capacity, qkv};

mod block;
mod feweight;
mod fewfour;
mod fewone;
mod fewwide;
mod merge;
mod online;
pub mod prompt;
mod rowwise;
mod runs;
pub mod short;
mod single;

axes![
    Jobs = 64,
    Tokens = 512,
    Heads = 4,
    Groups = 4,
    Head = 128,
    Rows = 64,
    Blocks = 80,
    Entries = 32,
    Runs = 8,
    Folds = 2,
    Page = 16,
    Kv = 2,
    Lane = 8,
    Stat = 2,
    Burst = 8,
    Parity = 2,
    Word = 262144
];
pub type Chip = crate::models::nemo::serving::Chips;
type Cluster = m![Jobs / 32];
/// A PE's slices hold one head and one page token each, heads of either parity on the two
/// data-memory networks: a page gather lands its 64 packets on the PE's 64 slices in one visit,
/// the networks alternating every packet.
type Slices = m![Jobs % 32 / 8, Heads % 2, Heads / 2, Page];
/// One token of each of a job's 32 pages for the PE's eight jobs, its key then its value.
type Raw = DmTensor<f8e4m3, Chip, Cluster, Slices, m![Jobs % 8, Tokens / 16, Kv, Head]>;
type Decoded = DmTensor<bf16, Chip, Cluster, Slices, m![Jobs % 8, Tokens / 16, Head]>;
/// One job's query head in each wave of a burst, every group: a 1-KiB run of the row's query, on
/// half of a head ring's slices. The TRF fill broadcasts the eight to the ring's 16 slices. A
/// batch's tables sit in the half of its parity, so the next batch's land while it reads its own.
type Staged = DmTensor<
    bf16,
    Chip,
    Cluster,
    m![Jobs % 32 / 8, Heads % 2, Heads / 2, Jobs % 8, 1 # 2],
    m![Parity, Burst, Groups, Head],
>;
type Lanes = TrfTensor<bf16, Chip, Cluster, Slices, m![Groups], m![Jobs % 8, Head]>;
/// A wave's present pages of each job for the slice's token, landed in all four head rings of a
/// PE with its burst.
type Counts<'l> = DmTensorView<
    'l,
    f32,
    Chip,
    Cluster,
    Slices,
    m![Parity = 1 # 2, Burst = 1 # 8, Jobs % 8, 1 # 8],
>;
/// Each page's index within its job, in every slice.
pub type Positions = DmTensor<f32, Chip, Cluster, Slices, m![Tokens / 16]>;
/// The pages a one-row batch's job may hold: its wave computes only these.
pub const SINGLE_PAGES: usize = 8;
/// The rows of a few-row batch and the keys each may hold: one wave with no merge, a row's pages
/// over every PE ([`fewone`], up to 4 rows [`fewfour`]), one row a PE ([`feweight`]) or whole
/// rows in parts of eight ([`fewwide`]). Its four parts (25-32 rows) cost more in the step than
/// a wave of paired slots, so the host sends it at most three.
pub const FEW_ROWS: usize = 24;
pub const FEW_KEYS: usize = 2048;
/// The jobs a row holds in a wave of whole rows ([`rowwise`]).
pub const ROW_JOBS: usize = 4;
type Scores = DmTensor<f32, Chip, Cluster, Slices, m![Jobs % 8, Groups, Tokens / 16]>;
type Weights = DmTensor<bf16, Chip, Cluster, Slices, m![Jobs % 8, Groups, Tokens / 16]>;
type Local = DmTensor<f32, Chip, Cluster, Slices, m![Jobs % 8, Groups, 1 # 8]>;
/// A burst's peak and mass for every job of the slice's head: a job's pair is one 256-B run.
type Records = DmTensor<f32, Chip, Cluster, Slices, m![Burst, Jobs % 8, Stat, Groups, 1 # 8]>;
type Scalar = VrfTensor<f32, Chip, Cluster, Slices, m![Jobs % 8, Groups, 1 # 8]>;
type Partial = DmTensor<f32, Chip, Cluster, Slices, m![Jobs % 8, Groups, Head]>;
/// A burst's rounded outputs, stored as one record.
type Rounded = DmTensor<bf16, Chip, Cluster, Slices, m![Burst, Jobs % 8, Groups, Head]>;
/// The slices of one token position: every job's peak, mass and output reach all 16 after the
/// span reductions.
type Stored = m![Jobs % 32 / 8, Heads % 2, Heads / 2, Page = 1 # 16];
/// A burst's counts as landed, in all four head rings of a PE, by batch parity as `Staged`.
type Landed =
    DmTensor<f32, Chip, Cluster, m![Jobs % 32 / 8, 4, Page], m![Parity, Burst, Jobs % 8, Lane]>;

/// A layer's pool pages: each token's 1-KiB record holds its heads in turn, a head's key then its
/// value, so a 256-byte packet is one head's key and value.
pub type Pool = Slice<HbmTensor<f8e4m3, Chip, m![Page, Heads, Kv, Head]>>;
pub type Count = HbmTensor<i32, Chip, m![1]>;
/// The byte offset of each of a job's 32 pool pages within its layer's pool pages.
pub type Pages = HbmTensor<i32, Chip, m![Jobs, Tokens / 16]>;
/// Each job's written tokens: its gather reads only the pages that hold them, an idle job none.
pub type Lengths = HbmTensor<i32, Chip, m![Jobs]>;
/// A burst's present pages for each token of a page: entry `[t][job]` counts the pages of the
/// job that hold their token `t`, as eight vector lanes, so a slice's burst lands as 256-B runs.
pub type Presence = HbmTensor<f32, Chip, m![Page, Burst, Jobs, Lane]>;
/// Each page's index within its job, `0..32`.
pub type Indices = HbmTensor<f32, Chip, m![Tokens / 16]>;
/// The byte offset of each burst job's row query among the step's queries.
pub type Slots = HbmTensor<i32, Chip, m![Burst, Jobs]>;
/// A burst's peak and mass of every head and group, one vector lane record each: a head's pair is
/// one 256-B run from its slice, and a job's 1-KiB record is one merge entry.
pub type Statistics = HbmTensor<f32, Chip, m![Burst, Jobs, Heads, Stat, Groups, Lane]>;
/// A burst's unnormalized block outputs, rounded to bf16: they are the merge's largest gather.
pub type Outputs = HbmTensor<bf16, Chip, m![Burst, Jobs, Heads, Groups, Head]>;
pub type Restored = HbmTensor<f32, Chip, m![Rows, Heads, Groups, Head]>;
/// Byte offsets of the partial records a merge pass reads for every row.
pub type Merge = HbmTensor<i32, Chip, m![Rows, Entries]>;
pub type Visible = HbmTensor<f32, Chip, m![Rows, Entries]>;
/// A final merge pass of a batch whose rows read at most `Runs` entries each, and each entry's
/// tag in eight lanes: a slice holds one entry, so its tag is one 32-B run.
pub type RunMerge = HbmTensor<i32, Chip, m![Rows, Runs]>;
pub type RunVisible = HbmTensor<f32, Chip, m![Rows, Runs, Lane]>;
/// The present pages a job slot's counts carry where its job continues the run of its job in the
/// wave before: a whole job's, which the mask reads as all present.
pub const CARRIED: f32 = 64.0;

/// Merge passes, one per entry: each row's partial records and which of them count.
#[derive(DeviceSend)]
pub struct Passes {
    pub statistics: Slice<Merge>,
    pub outputs: Slice<Merge>,
    pub visible: Slice<Visible>,
}

/// The final passes of [`Passes`] for batches whose rows read at most `Runs` entries.
#[derive(DeviceSend)]
pub struct RunPasses {
    pub statistics: Slice<RunMerge>,
    pub outputs: Slice<RunMerge>,
    pub visible: Slice<RunVisible>,
}

/// The attention work of every 64-row batch of a step.
#[derive(DeviceSend)]
pub struct Plan {
    /// Each batch's [`word`].
    pub batches: Slice<Count>,
    /// Batch `b`'s job wave `w` at entry `b * Blocks + w`.
    pub pages: Slice<Pages>,
    pub lengths: Slice<Lengths>,
    /// Batch `b`'s burst `u` of waves at entry `b * Blocks / Burst + u`.
    pub counts: Slice<Presence>,
    pub positions: Indices,
    pub queries: Slice<Slots>,
    /// Batch `b`'s final merge pass at entry `b`.
    pub merge: Passes,
    /// The same pass at entry `b` over its first `Runs` entries of each row, read instead when
    /// the batch's rows read no more.
    pub runs: RunPasses,
    /// Batch `b`'s fold pass `f` at entry `b * Folds + f`; passes past its fold passes stay unread.
    pub fold: Passes,
}

const WAVE_SPAN: usize = 128;
const KIND_SPAN: usize = 4;
const DIRECT_SPAN: usize = 8;
const _: () = assert!(
    Blocks::SIZE < WAVE_SPAN
        && Folds::SIZE < KIND_SPAN
        && word(Blocks::SIZE, Folds::SIZE, 7, Entries::SIZE) < Word::SIZE
);

/// A batch's plan word, of `waves` job waves, `folds` fold passes, slot kind `direct` and the
/// `entries` its final merge reads of each row: `direct` 1 if
/// every row keeps one job slot through the waves, row `r` slot `8 * (r % 8) + r / 8`; 2 if each
/// of its at most 32 rows keeps two, that one and the one four slot times on, each over half the
/// row's blocks; 3 if each wave holds 16 whole rows of at most four jobs ([`rowwise`]); 4 if
/// each job slot carries one row's softmax over a run of its jobs and leaves a partial a wave;
/// 5 for a first batch of at most `FEW_ROWS` rows of at most `FEW_KEYS` keys in one wave, its
/// `entries` its rows; else 0, a partial a job. The first two carry each row's softmax in its
/// slots, the third merges each wave's rows on chip: none leaves partials. One word a batch, as each staged word
/// costs a transfer and a core sync.
pub const fn word(waves: usize, folds: usize, direct: usize, entries: usize) -> usize {
    waves + WAVE_SPAN * (folds + KIND_SPAN * (direct + DIRECT_SPAN * entries))
}

/// A batch's slot kind, from its word's `direct`, each flag 0 or 1: `paired` slots are `fixed`.
#[derive(Clone, Copy)]
struct Kind {
    fixed: usize,
    paired: usize,
    rowwise: usize,
    runs: usize,
    /// 1 for a few-row batch, which attends its rows in place.
    small: usize,
}

impl Kind {
    fn of(direct: usize) -> Self {
        let few = (direct / 5).min(1);
        let slots = direct % 4 % 3 * (1 - few);
        Self {
            fixed: slots.min(1),
            paired: slots / 2,
            rowwise: direct % 4 / 3 * (1 - few),
            runs: direct / 4 * (1 - few),
            small: few,
        }
    }
}

/// A few-row batch's chain by its rows, each 0 or 1.
struct Chain {
    one: usize,
    four: usize,
    eight: usize,
    /// Parts of eight rows, `0` or `2 ..= 4`.
    parts: usize,
}

impl Chain {
    /// The batch's rows into `output`, `under` issued behind its first page gather (the last
    /// one of [`fewwide`], whose parts use the record's idle `odd` page buffer and `rounded` outputs).
    fn attend(
        &self,
        context: &mut Device,
        (history, queries): (&History<'_>, &qkv::Queries),
        (odd, rounded): (&mut Raw, &mut Rounded),
        output: &mut Restored,
        under: &mut impl FnMut(&mut Device),
    ) {
        for _ in 0..self.one {
            fewone::attend(context, history, queries, output, &mut *under);
        }
        for _ in 0..self.four {
            fewfour::attend(context, history, queries, output, &mut *under);
        }
        for _ in 0..self.eight {
            feweight::attend(context, history, queries, output, &mut *under);
        }
        for _ in 0..self.parts.min(1) {
            let buffers = (&mut *odd, &mut *rounded);
            fewwide::attend(
                context,
                (history, queries),
                buffers,
                (&mut *output, self.parts),
                &mut *under,
            );
        }
    }

    fn of(rows: usize) -> Self {
        let (past_one, past_four) = ((rows / 2).min(1), (rows / 5).min(1));
        let past_eight = (rows / 9).min(1);
        Self {
            one: 1 - past_one,
            four: past_one * (1 - past_four),
            eight: past_four * (1 - past_eight),
            parts: past_eight * (rows + 7) / 8,
        }
    }
}

/// The job waves, fold passes, slot kind and merge entries of a batch's [`word`].
pub fn shape(word: usize) -> (usize, usize, usize, usize) {
    let kinds = word / WAVE_SPAN / KIND_SPAN;
    (
        (word % WAVE_SPAN).min(Blocks::SIZE),
        (word / WAVE_SPAN % KIND_SPAN).min(Folds::SIZE),
        kinds % DIRECT_SPAN,
        (kinds / DIRECT_SPAN).min(Entries::SIZE),
    )
}

/// One entry whose first `Folds` waves hold a fold record a row, then one entry of job records a
/// burst of waves.
#[derive(DeviceSend)]
pub struct Partials {
    pub stats: Slice<Statistics>,
    pub output: Slice<Outputs>,
}

pub struct History<'a> {
    pub pool: &'a Pool,
    /// The layer's first pool page.
    pub entry: usize,
    pub plan: &'a Plan,
    pub batch: usize,
    /// The batch's leading rows that are live: the final merge reads only their entries.
    pub rows: usize,
    /// [`positions`] of the plan, landed once a launch.
    pub positions: &'a Positions,
    /// [`few`] of the plan, chosen once a launch.
    pub few: &'a Few<'a>,
    /// 1 if batch `batch + 1` attends next in the layer, else 0: its first wave and tables land
    /// under this batch's last wave, so the page stream runs on through the batch's merge and tail.
    pub next: usize,
}

/// Length tables `1 ..= 4`, which the few-row chains read: the host lays out at least five.
pub type Few<'a> = [&'a Lengths; 4];

/// The plan's [`Few`] tables, chosen once a launch so that no chain's length transfer waits in a
/// guard behind the transfers its layer queues first.
pub fn few(plan: &Plan) -> Few<'_> {
    [
        &plan.lengths[1],
        &plan.lengths[2],
        &plan.lengths[3],
        &plan.lengths[4],
    ]
}

/// Each page's index within its job in every slice, for every batch's masks.
pub fn positions(context: &mut Device, plan: &Plan) -> Positions {
    plan.positions.to_dm(&mut context.tdma)
}

/// The layer's first gate half and where it lands, and 1 if this batch lands it, else 0: it lands
/// under the batch's last wave, which issues no transfer until its partials leave. `lead`'s
/// transfers, when its count is 1, land in an even last wave with a next batch, else behind the
/// batch's merge transfers.
pub struct Prefetch<'a, L: Lead> {
    pub gate: (mlp::weight::Weight<'a>, &'a mut mlp::weight::First, usize),
    pub lead: (&'a mut L, usize),
}

/// Transfers a batch's last wave issues for the layer's later work, where the wave's own gathers
/// are done and the transfer queue idles until the next batch's tables.
pub trait Lead {
    /// False for a batch known to follow the layer's first, which lands neither its own first wave
    /// nor these transfers: their guards would be device loops of a constant zero trips.
    const FIRST: bool = true;
    /// True for a batch whose second wave's pages the caller gathered ahead ([`gather_ahead`]).
    const AHEAD: bool = false;

    fn land(&mut self, context: &mut Device);
}

/// The lead of a batch that follows the layer's first, its second wave gathered ahead.
pub struct Later;

impl Lead for Later {
    const FIRST: bool = false;
    const AHEAD: bool = true;

    fn land(&mut self, _: &mut Device) {}
}

/// Batch `history.batch`'s second wave's pages into the odd buffer, ahead of its attention (`L::AHEAD`):
/// issued where the transfer queue idles, the batch's first two waves find their pages landed.
pub fn gather_ahead(
    context: &mut Device,
    (history, word): (&History<'_>, usize),
    buffers: &mut Buffers,
) {
    let (count, ..) = shape(word);
    for _ in 0..(count - 1).min(1) {
        gather(context, history, &mut buffers.odd, 1);
    }
}

/// No transfers for the layer's later work.
impl Lead for () {
    fn land(&mut self, _: &mut Device) {}
}

/// Plan entry `entry`'s wave, only each slot's written pages (stale bytes past them, masked): a
/// batch's first wave holds its partial jobs and idle slots, a wave of whole rows its rows' runs.
fn written(context: &mut Device, history: &History<'_>, entry: usize, raw: &mut Raw) {
    // SAFETY: as `gather`.
    unsafe {
        history
            .pool
            .gather::<m![1], m![Page, Heads, Kv, Head]>()
            .entry(history.entry)
            .by_byte_offsets(&history.plan.pages[entry])
            .prefix::<m![Tokens], _>(&history.plan.lengths[entry], 0)
            .unchecked()
    }
    .to_dm_view(&mut context.tdma, raw.view_mut());
}

/// Burst `index`'s tables into the half of batch parity `parity`: each job's row query, every
/// group, and the counts of its pages.
fn stage(
    context: &mut Device,
    (plan, queries): (&Plan, &qkv::Queries),
    (index, parity): (usize, usize),
    tables: &mut Tables,
) {
    // SAFETY: a capacity row's query heads are its `Heads × Groups` heads in head order.
    unsafe {
        queries
            .view()
            .reshape::<Chip, m![Capacity, Heads, Groups, Head]>()
    }
    .gather::<m![Capacity], m![Heads, Groups, Head]>()
    .by_byte_offsets(&plan.queries[index])
    .to_dm_view(
        &mut context.tdma,
        tables
            .staged
            .view_mut()
            .tile::<m![Parity], 1, m![Parity = 1 #{!} 2, Burst, Groups, Head]>(parity),
    );
    plan.counts[index].view().to_dm_view(
        &mut context.tdma,
        tables
            .landed
            .view_mut()
            .tile::<m![Parity], 1, m![Parity = 1 #{!} 2, Burst, Jobs % 8, Lane]>(parity),
    );
}

/// Wave `wave`'s pages, a whole job's in every slot.
fn gather(context: &mut Device, history: &History<'_>, raw: &mut Raw, wave: usize) {
    // SAFETY: the host plan bounds every page by the layer's pages, and each offset starts one.
    unsafe {
        history
            .pool
            .gather::<m![1], m![Page, Heads, Kv, Head]>()
            .entry(history.entry)
            .by_byte_offsets(&history.plan.pages[history.batch * Blocks::SIZE + wave])
            .unchecked()
    }
    .to_dm_view(&mut context.tdma, raw.view_mut());
}

/// The pages' tokens of one of the two records.
fn tokens(
    raw: &Raw,
    kv: usize,
) -> DmTensorView<'_, f8e4m3, Chip, Cluster, Slices, m![Jobs % 8, Tokens / 16, Kv = 1 # 2, Head]> {
    raw.view()
        .tile::<m![Kv], 1, m![Jobs % 8, Tokens / 16, Kv = 1 # 2, Head]>(kv)
}

/// Widened through f32: the A0 fetch converts an 8-bit float only to f32.
fn keys(context: &mut Device, raw: &Raw) -> Decoded {
    context
        .main
        .begin(tokens(raw, 0))
        .fetch::<m![Jobs % 8, Tokens / 16, Head / 8], m![Head % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Jobs % 8, Tokens / 16, Head / 8], m![Head % 8]>()
        .cast::<bf16, m![Head % 8 # 16]>()
        .commit_trim::<m![Head % 8]>()
        .commit()
}

const CLEARED: TagGuard =
    TagGuard::matches([BitReq::Zero, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]);
const SECOND_CLEARED: TagGuard =
    TagGuard::matches([BitReq::Ignore, BitReq::Zero, BitReq::Ignore, BitReq::Ignore]);

/// Decoded values with every non-finite cell cleared: a page's unwritten tail may hold NaN, and a
/// zero probability times NaN is NaN.
fn values(context: &mut Device, raw: &Raw) -> Decoded {
    context
        .main
        .begin(tokens(raw, 1))
        .fetch::<m![Jobs % 8, Tokens / 16, Head / 8], m![Head % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![Jobs % 8, Tokens / 16, Head / 8], m![Head % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Comparison([
            Cmp::Less(f32::INFINITY),
            Cmp::Greater(f32::NEG_INFINITY),
            Cmp::False,
            Cmp::False,
        ]))
        .vector_logic(
            LogicBinaryOpF32::BitAnd,
            Branched::imm(CLEARED, 0.0f32).imm(SECOND_CLEARED, 0.0f32),
        )
        .vector_final()
        .cast::<bf16, m![Head % 8 # 16]>()
        .commit_trim::<m![Head % 8]>()
        .commit()
}

/// One batch's tables and partials, which every wave of the batch reads or writes.
struct Batch<'a, 'b, L: Lead> {
    history: &'a History<'b>,
    queries: &'a qkv::Queries,
    partials: &'a mut Partials,
    prefetch: Prefetch<'a, L>,
}

/// The data-memory buffers a burst's waves share: its staged tables and its partials.
struct Tables {
    staged: Staged,
    landed: Landed,
    records: Records,
    rounded: Rounded,
}

/// The data-memory buffers every batch of a layer's decode attention uses in turn: the two page
/// buffers its waves alternate, even waves reading `even`, a burst's tables and partials, and
/// the softmax a batch of fixed slots carries.
pub struct Buffers {
    even: Raw,
    odd: Raw,
    tables: Tables,
    online: online::State,
}

impl Buffers {
    pub fn new() -> Self {
        Self {
            even: Raw::new(),
            odd: Raw::new(),
            tables: Tables {
                staged: Staged::new(),
                landed: Landed::new(),
                records: Records::new(),
                rounded: Rounded::new(),
            },
            online: online::State::new(),
        }
    }
}

/// Wave `at` of a burst's staged query heads in the half of `parity`, broadcast to every slice
/// of their head rings.
fn lanes(context: &mut Device, staged: &Staged, (parity, at): (usize, usize)) -> Lanes {
    context
        .sub
        .begin(
            staged
                .view()
                .tile::<m![Parity], 1, m![Parity = 1 # 2, Burst, Groups, Head]>(parity)
                .tile::<m![Burst], 1, m![Parity = 1 # 2, Burst = 1 # 8, Groups, Head]>(at),
        )
        .fetch::<m![Groups], m![Head]>()
        .switch::<Slices, m![Groups, Jobs % 8]>(SwitchConfig::CustomBroadcast { ring_size: 16 })
        .collect::<m![Groups, Jobs % 8, Head / 16], m![Head % 16]>()
        .to_trf()
}

/// Wave `at` of a burst's counts as landed in the half of `parity`.
fn counts(landed: &Landed, (parity, at): (usize, usize)) -> Counts<'_> {
    // SAFETY: the four head rings of a PE hold the same counts, and `Lane` lays out the eight
    // cells of `1 # 8`.
    unsafe {
        landed
            .view()
            .reshape::<Chip, Cluster, Slices, m![Parity, Burst, Jobs % 8, 1 # 8]>()
    }
    .tile::<m![Parity], 1, m![Parity = 1 # 2, Burst, Jobs % 8, 1 # 8]>(parity)
    .tile::<m![Burst], 1, m![Parity = 1 # 2, Burst = 1 # 8, Jobs % 8, 1 # 8]>(at)
}

impl<L: Lead> Batch<'_, '_, L> {
    /// Wave `wave` of `count`, an even wave over `even`'s pages if `EVEN`, gathering the next into
    /// the other buffer. A burst's tables land at its first wave and its partials leave at its
    /// last, so its waves queue only gathers; a batch's last wave lands the next batch's first
    /// wave into `even` and its first tables, each once this wave has read what it replaces.
    /// A batch of fixed slots (`fixed` 1) folds the wave into each slot's softmax instead, and
    /// its last wave stores the attended rows into `output`, a paired batch's (`paired` 1) once
    /// it has merged each row's two slots; a `rowwise` batch's every wave stores its rows.
    fn attend_wave<const EVEN: bool>(
        &mut self,
        context: &mut Device,
        buffers: &mut Buffers,
        (wave, count): (usize, usize),
        (
            Kind {
                fixed,
                paired,
                rowwise,
                runs,
                ..
            },
            output,
        ): (Kind, &mut Restored),
    ) {
        let plan = self.history.plan;
        let batch = self.history.batch;
        let (burst, at) = (wave / Burst::SIZE, wave % Burst::SIZE);
        let index = batch * (Blocks::SIZE / Burst::SIZE) + burst;
        let lead = (1 - (count - wave - 1).min(1)) * self.history.next;
        let parity = batch % 2;
        let Buffers {
            even,
            odd,
            tables: buffers,
            online,
        } = buffers;
        let (old, new, scale) = online.wave::<EVEN>();
        let (raw, next): (&Raw, &mut Raw) = if EVEN {
            (&*even, &mut *odd)
        } else {
            (&*odd, &mut *even)
        };
        // A batch past the first found its first burst's tables landed by the batch before; a batch
        // of whole rows that stages them gathers its second wave's written pages, with no wave guard.
        let staged = (1 - at.min(1)) * (1 - (1 - burst.min(1)) * batch.min(1));
        for _ in 0..staged {
            stage(context, (plan, self.queries), (index, parity), buffers);
            for _ in 0..rowwise * (count - wave - 1).min(1) {
                written(context, self.history, batch * Blocks::SIZE + wave + 1, next);
            }
        }
        // A batch gathered ahead has its second wave's pages issued already.
        let ahead = if EVEN && L::AHEAD { 1 - wave.min(1) } else { 0 };
        for _ in 0..(count - wave - 1).min(1) * (1 - ahead) * (1 - rowwise * staged) {
            gather(context, self.history, next, wave + 1);
        }
        // An odd last wave lands the next batch's tables into the other half at its start: behind
        // the next batch's first-wave gather their page remap would wait for that gather, and
        // behind the gate half the command processor would wait for their indirect query
        // gather's index list before the keys.
        if !EVEN {
            for _ in 0..lead {
                stage(
                    context,
                    (plan, self.queries),
                    ((batch + 1) * (Blocks::SIZE / Burst::SIZE), 1 - parity),
                    buffers,
                );
            }
        }
        for _ in 0..(1 - (count - wave - 1).min(1)) * self.prefetch.gate.2 {
            let (gate, landed, _) = &mut self.prefetch.gate;
            gate.land_first(context, landed);
        }
        // An even last wave with a next batch leaves the transfer queue idle until that batch's
        // tables: `lead`'s transfers land there.
        if EVEN && L::FIRST {
            for _ in 0..lead {
                for _ in 0..self.prefetch.lead.1 {
                    self.prefetch.lead.0.land(context);
                }
            }
        }
        let key = keys(context, raw);
        // An odd last wave reads `odd`: the next batch's first wave lands into `even` under it.
        if !EVEN {
            for _ in 0..lead {
                written(context, self.history, (batch + 1) * Blocks::SIZE, next);
            }
        }
        let lanes = lanes(context, &buffers.staged, (parity, at));
        let scores = block::scores(context, &key, &lanes);
        let present = counts(&buffers.landed, (parity, at));
        let scores = block::mask(context, &scores, (self.history.positions, present));
        let local = block::local_peak(context, &scores);
        let mut weights = Weights::new();
        // Run slots are job slots, so the run paths nest in the job blocks: a wave of another kind
        // skips one guard a block, as before runs existed.
        let jobs = (1 - fixed) * (1 - rowwise);
        for _ in 0..jobs {
            for _ in 0..1 - runs {
                block::job_peak(context, &local, block::stat(&mut buffers.records, at, 0));
                block::weights(context, &scores, &buffers.records, at, &mut weights);
            }
            // A run slot carries its run's softmax as a fixed slot does, its peak at a run's
            // start the lowest finite one: there it attends its job alone.
            for _ in 0..runs {
                let before = online::restart(context, old, counts(&buffers.landed, (parity, at)));
                let top = online::peak(context, &local, (&before, &mut *new), &mut *scale);
                block::weigh(context, &scores, &top, &mut weights);
            }
        }
        for _ in 0..rowwise {
            rowwise::weigh(context, (&scores, &local), &mut weights);
        }
        for _ in 0..fixed {
            let top = online::peak(context, &local, (old.peak(), &mut *new), &mut *scale);
            block::weigh(context, &scores, &top, &mut weights);
        }
        let masses = block::local_mass(context, &weights);
        let probabilities = block::fill(context, &weights);
        let value = values(context, raw);
        // An even last wave reads `even`: the next batch's first wave lands once it is read, its
        // tables just ahead. At the wave's start they would queue behind this batch's own tables
        // when the wave starts a burst, and their page remap would hold the keys.
        if EVEN {
            for _ in 0..lead {
                stage(
                    context,
                    (plan, self.queries),
                    ((batch + 1) * (Blocks::SIZE / Burst::SIZE), 1 - parity),
                    buffers,
                );
                written(context, self.history, (batch + 1) * Blocks::SIZE, even);
            }
        }
        let weighted = block::weighted(context, &value, &probabilities);
        // Each path's work past the values' contraction, its job masses among them (nothing reads
        // them earlier), sits in one guarded block: every block a wave skips costs its guard.
        let more = (count - wave - 1).min(1);
        for _ in 0..fixed {
            online::mass(context, &masses, (old, &mut *new), scale);
            for _ in 0..more {
                online::sum(context, &weighted, (old, &mut *new), scale);
            }
            for _ in 0..1 - more {
                for _ in 0..1 - paired {
                    online::finish(context, &weighted, (old, &mut *new), scale, &mut *output);
                }
                for _ in 0..paired {
                    online::pairs(context, &weighted, (old, &mut *new), scale, &mut *output);
                }
            }
        }
        for _ in 0..jobs {
            for _ in 0..1 - runs {
                block::job_mass(context, &masses, block::stat(&mut buffers.records, at, 1));
                block::settle(context, &weighted, &mut buffers.rounded, at);
            }
            for _ in 0..runs {
                online::mass(context, &masses, (old, &mut *new), scale);
                for _ in 0..more {
                    online::sum(context, &weighted, (old, &mut *new), scale);
                    let records = (&mut buffers.records, &mut buffers.rounded);
                    online::record(context, new, records, at);
                }
                for _ in 0..1 - more {
                    let records = (&mut buffers.records, &mut buffers.rounded);
                    online::close(context, &weighted, (old, &*new), scale, records, at);
                }
            }
            for _ in 0..1 - ((at + 1) % Burst::SIZE).min(1) * more {
                // SAFETY: `1 # 8` and `Lane` lay out the same eight cells per head and group.
                unsafe {
                    buffers
                        .records
                        .view()
                        .reshape::<Chip, Cluster, Slices, m![Burst, Jobs % 8, Stat, Groups, Lane]>()
                }
                .slice_tile::<m![Page], 1, Stored>(0)
                .to_hbm_view(&mut context.tdma, self.partials.stats[1 + burst].view_mut());
                buffers
                    .rounded
                    .view()
                    .slice_tile::<m![Page], 1, Stored>(0)
                    .to_hbm_view(
                        &mut context.tdma,
                        self.partials.output[1 + burst].view_mut(),
                    );
            }
        }
        for _ in 0..rowwise {
            // Whole-row waves store only their own rows: the batch's first clears every row, so
            // those past its waves read as zero, as every other kind leaves its dead rows. Inside
            // this block, the kinds that skip it keep their own blocks unchanged.
            for _ in 0..(1 - at.min(1)) * (1 - burst.min(1)) {
                rowwise::clear(context, &mut *output);
            }
            // A two-wave batch with a partial last block attends that block first: only the
            // first wave skips its slots' unwritten pages.
            let swap = (self.history.rows % rowwise::ROWS).min(1) * (count - 1).min(1);
            let block = wave + swap - 2 * wave * swap;
            rowwise::finish(context, (&masses, &weighted), block, &mut *output);
        }
    }
}

/// One batch's attended rows: its job waves, then its fold passes, then the final merge, or for
/// a batch of fixed slots its waves alone; a batch the host gives no job waves is one decode row
/// whose jobs fill one wave, merged on chip. The first batch lands its own first wave; a later one
/// finds it landed in `buffers`.
pub fn attend<L: Lead>(
    context: &mut Device,
    history: &History<'_>,
    (queries, partials): (&qkv::Queries, &mut Partials),
    prefetch: Prefetch<'_, L>,
    buffers: &mut Buffers,
) -> Merging {
    let word = batch_word(history.plan, history.batch);
    attend_word(
        context,
        (history, word),
        (queries, partials),
        prefetch,
        buffers,
    )
}

/// A few-row batch's attended rows (its word's slot kind 5), the chain alone: nothing lands under
/// it.
pub fn attend_few(context: &mut Device, history: &History<'_>, queries: &qkv::Queries) -> Restored {
    let (_, _, _, rows) = shape(batch_word(history.plan, history.batch));
    let mut output = Restored::new();
    let (mut odd, mut rounded) = (Raw::new(), Rounded::new());
    Chain::of(rows).attend(
        context,
        (history, queries),
        (&mut odd, &mut rounded),
        &mut output,
        &mut |_: &mut Device| {},
    );
    output
}

/// Batch `batch`'s plan [`word`], read ahead of its attention's first gather: at the fold loop,
/// the read would wait behind the waves' stores.
pub fn batch_word(plan: &Plan, batch: usize) -> usize {
    plan.batches[batch].extent::<m![Word]>()
}

/// `attend` of a batch whose plan word was read ahead.
pub fn attend_word<L: Lead>(
    context: &mut Device,
    (history, word): (&History<'_>, usize),
    (queries, partials): (&qkv::Queries, &mut Partials),
    prefetch: Prefetch<'_, L>,
    buffers: &mut Buffers,
) -> Merging {
    let plan = history.plan;
    // `Rows` rows of at most `Blocks` jobs fill at most `Blocks` waves of `Jobs`.
    let (count, folds, direct, entries) = shape(word);
    let kind = Kind::of(direct);
    let Kind {
        fixed,
        rowwise,
        runs,
        small,
        ..
    } = kind;
    // A skip's lengths load stalls the issue.
    if L::FIRST {
        for _ in 0..(1 - history.batch.min(1)) * (1 - small) {
            written(
                context,
                history,
                history.batch * Blocks::SIZE,
                &mut buffers.even,
            );
        }
    }
    for _ in 0..(fixed + runs) * count.min(1) {
        buffers.online.clear(context);
    }
    let waves = count.min(1);
    let mut output = Restored::new();
    let mut collected = single::Collected::new();
    let mut batch = Batch {
        history,
        queries,
        partials,
        prefetch,
    };
    for pair in 0..count.div_ceil(2) {
        batch.attend_wave::<true>(context, buffers, (2 * pair, count), (kind, &mut output));
        for _ in 0..(count - 2 * pair - 1).min(1) {
            batch.attend_wave::<false>(
                context,
                buffers,
                (2 * pair + 1, count),
                (kind, &mut output),
            );
        }
    }
    // A batch the host gives no job waves is one row whose jobs fill its first wave's first
    // `SINGLE_PAGES` pages: the wave's partials merge on chip, each PE's here and the PEs' at
    // `finish`.
    // A few-row batch attends inside the no-wave loop, so batches with waves see no new loop.
    for _ in 0..1 - waves {
        // The layer's first gate half lands under the rows, as under a single row's waves.
        let gate = &mut batch.prefetch.gate;
        let mut land = |context: &mut Device| {
            for _ in 0..gate.2 {
                gate.0.land_first(context, &mut *gate.1);
            }
        };
        // Only a step's first batch is a few-row batch, so later batches carry no chain.
        if L::FIRST {
            for _ in 0..small {
                let Buffers { odd, tables, .. } = &mut *buffers;
                Chain::of(entries).attend(
                    context,
                    (history, queries),
                    (odd, &mut tables.rounded),
                    &mut output,
                    &mut land,
                );
            }
        }
        for _ in 0..1 - small {
            let Buffers { even, tables, .. } = &mut *buffers;
            single::stage(context, (history, queries), tables);
            batch.attend_row(context, even, tables);
            single::collect(
                context,
                (&tables.records, &tables.rounded),
                &mut collected,
                &mut output,
            );
        }
    }
    for fold in 0..folds {
        merge::fold(
            context,
            (&plan.fold, history.batch * Folds::SIZE + fold),
            batch.partials,
            fold,
        );
    }
    // A batch of fixed slots or whole-row waves gathers no entries; one whose rows read at most
    // `Runs` entries gathers them from its run pass.
    let partial = waves * (1 - fixed) * (1 - rowwise);
    let few = 1 - (entries / (Runs::SIZE + 1)).min(1);
    let gathers = merge::rows(
        context,
        (&plan.merge, history.batch),
        batch.partials,
        history.rows * partial * (1 - few),
    );
    let gathered = runs::rows(
        context,
        (&plan.runs, history.batch),
        batch.partials,
        history.rows * partial * few,
    );
    // Elsewhere `lead`'s transfers land behind the batch's merge transfers.
    if L::FIRST {
        for _ in 0..1 - (count % 2) * history.next {
            for _ in 0..batch.prefetch.lead.1 {
                batch.prefetch.lead.0.land(context);
            }
        }
    }
    Merging {
        gathers,
        runs: gathered,
        collected,
        waves,
        small,
        partial: (1 - fixed) * (1 - rowwise),
        few,
        output,
    }
}

/// A batch's final merge with its entries gathered (a one-row batch's are its PEs' partials), so
/// work issued before `finish` lands under the gathers and ahead of the merge's own Main work.
pub struct Merging {
    gathers: merge::Gathers,
    runs: runs::Gathers,
    collected: single::Collected,
    waves: usize,
    /// 1 for a few-row batch, which attends its rows in place.
    small: usize,
    /// 1 if the batch's waves leave partials, else 0: fixed slots and whole-row waves attend the
    /// rows themselves.
    partial: usize,
    /// 1 if the batch's rows read their entries from its run pass.
    few: usize,
    output: Restored,
}

impl Merging {
    pub fn finish(mut self, context: &mut Device) -> Restored {
        for _ in 0..1 - self.waves {
            for _ in 0..1 - self.small {
                single::finish(context, &self.collected, &mut self.output);
            }
        }
        for _ in 0..self.waves {
            for _ in 0..self.partial * (1 - self.few) {
                merge::finish(context, &self.gathers, &mut self.output);
            }
            for _ in 0..self.partial * self.few {
                runs::finish(context, &self.runs, &mut self.output);
            }
        }
        self.output
    }
}
