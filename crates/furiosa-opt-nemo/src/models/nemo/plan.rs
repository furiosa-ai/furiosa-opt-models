//! Host tables for one step launch: per-row RoPE coefficients, cache-write offsets and the paged
//! attention plan. Each sequence owns a run of consecutive pages of the plan's page space starting
//! at its row's `base`; [`Pages`] maps them to a layer's pool pages.
//!
//! - Every row attends to its sequence up to its own position, in jobs of 512 keys (fewer in a
//!   batch whose 512-key jobs fit one wave, and in the jobs past a batch's full waves); a 64-row
//!   batch's jobs fill its job waves of 64, at plan entries `batch * Blocks + wave`, its partial
//!   wave first, and its bursts of eight waves stage their token counts and queries at entries
//!   `batch * Blocks / Burst + burst`.
//! - A row of up to 32 jobs merges in its batch's final pass; a longer row's final pass reads its
//!   first 30 jobs and one fold record for each further 32 jobs.
//! - A batch whose rows' 512-key blocks take no more waves when each row keeps one job slot
//!   (`Tables::fixed`; row `r` in slot `dealt(r)`, so a PE's slots hold rows eight apart) runs
//!   that way instead: each slot carries its row's softmax through the waves, and the batch writes
//!   no partials and runs no merge. Ahead of a prompt chunk, a batch of at most 32 rows that takes
//!   no more waves on two slots a row runs on such paired slots.
//! - Else a batch of rows of at most `ROW_KEYS` keys that takes no more waves of 16 whole rows,
//!   fills at least half a wave and leaves no partial wave but its first (`Tables::rowwise`; a
//!   two-wave batch attends its partial last block first) runs on them: each wave's rows merge on
//!   chip, so the batch too writes no partials and runs no merge.
use super::serving::{Chips, attention, mlp::Capacity, mlp::Waves, qkv::rotation, step};
use furiosa_opt_std::prelude::*;
use std::fmt;
use std::ops::Range;

/// Why rows leave the plan: the rule they break.
#[derive(Debug)]
pub struct Error(String);

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<String> for Error {
    fn from(reason: String) -> Self {
        Self(reason)
    }
}

impl From<&str> for Error {
    fn from(reason: &str) -> Self {
        Self(reason.into())
    }
}

impl From<std::num::TryFromIntError> for Error {
    fn from(error: std::num::TryFromIntError) -> Self {
        Self(error.to_string())
    }
}

impl From<furiosa_opt_std::Error> for Error {
    fn from(error: furiosa_opt_std::Error) -> Self {
        Self(error.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

pub const PAGE_TOKENS: usize = attention::Page::SIZE;
pub const HALF_RECORD: usize = attention::Heads::SIZE * attention::Head::SIZE;
pub const RECORD_BYTES: usize = attention::Kv::SIZE * HALF_RECORD;
pub const PAGE_BYTES: usize = PAGE_TOKENS * RECORD_BYTES;
/// The partial entries a batch writes: one of its fold records, then one per burst of job waves.
pub const PARTIALS: usize = 1 + attention::Blocks::SIZE / attention::Burst::SIZE;
const BLOCK: usize = attention::Tokens::SIZE;
const JOBS: usize = attention::Jobs::SIZE;
/// The jobs of one PE's slices (`Jobs % 8`); a wave spans as many PEs over both clusters.
const PE_JOBS: usize = 8;
const BATCH: usize = attention::Rows::SIZE;
const ENTRIES: usize = attention::Entries::SIZE;
const FOLDS: usize = attention::Folds::SIZE;
const BURST: usize = attention::Burst::SIZE;
const LANES: usize = attention::Lane::SIZE;
/// The jobs a folded row's final pass reads directly; its last entries are fold records.
const DIRECT: usize = ENTRIES - FOLDS;
const MAX_CONTEXT: usize = attention::Blocks::SIZE * BLOCK;
const JOB_PAGES: usize = BLOCK / PAGE_TOKENS;
/// The most keys a one-row batch merged on chip may hold: one wave of jobs of at most
/// `SINGLE_PAGES` pages, which its wave computes.
const SINGLE_KEYS: usize = JOBS * attention::SINGLE_PAGES * PAGE_TOKENS;
/// The rows of a wave of whole rows, and the most keys such a row holds.
const ROW_WAVE: usize = JOBS / attention::ROW_JOBS;
const ROW_KEYS: usize = attention::ROW_JOBS * BLOCK;
/// The clusters a wave's PEs span, the kernel's `Jobs / 32`.
const CLUSTERS: usize = 2;

/// The row of slot `slot` of block `block` in a batch of whole-row waves: a row takes one half of
/// a PE (slot times `4 h .. 4 h + 4`), a PE's two halves hold consecutive rows, and consecutive PE
/// pairs alternate clusters, so a wave of few rows reads no idle half's pages (wave 0 reads every
/// slot of a live PE to its longest job) and keeps both clusters' gathers busy.
const fn whole_row(block: usize, slot: usize) -> usize {
    let (pe, half) = (slot / PE_JOBS, slot % PE_JOBS / attention::ROW_JOBS);
    let (halves, cluster_pes) = (PE_JOBS / attention::ROW_JOBS, JOBS / PE_JOBS / CLUSTERS);
    ROW_WAVE * block + halves * (CLUSTERS * (pe % cluster_pes) + pe / cluster_pes) + half
}
const QUERY_BYTES: usize =
    attention::Heads::SIZE * attention::Groups::SIZE * attention::Head::SIZE * 2;
const STATISTIC_BYTES: usize = attention::Stat::SIZE
    * attention::Heads::SIZE
    * attention::Groups::SIZE
    * attention::Lane::SIZE
    * 4;
const OUTPUT_BYTES: usize =
    attention::Heads::SIZE * attention::Groups::SIZE * attention::Head::SIZE * 2;
const _: () = {
    assert!(DIRECT + FOLDS * ENTRIES >= MAX_CONTEXT.div_ceil(BLOCK));
    assert!(BATCH * MAX_CONTEXT.div_ceil(BLOCK) <= attention::Blocks::SIZE * JOBS);
    assert!(BATCH <= JOBS);
    assert!(FOLDS <= BURST && attention::Blocks::SIZE % BURST == 0);
    assert!(JOBS == PE_JOBS * PE_JOBS);
};

/// The last pages of every layer, one cache record a capacity row: a padding row writes only its
/// own, so no two rows of a step write the same bytes.
pub const SCRATCH_PAGES: usize = Capacity::SIZE / PAGE_TOKENS;

/// One capacity row: the token it embeds, its sequence's first page and its position there.
#[derive(Clone, Copy, PartialEq)]
pub struct Row {
    pub token: i32,
    pub base: usize,
    pub position: usize,
}

impl Row {
    /// Padding at capacity row `slot`, on the slot's scratch record.
    pub fn padding(pages: &Pages, slot: usize) -> Self {
        Self {
            token: 0,
            base: pages.scratch,
            position: slot,
        }
    }

    /// Byte offset of a position's record (every head's key and value) in this row's sequence.
    fn record(&self, pages: &Pages, position: usize) -> usize {
        pages.physical(self.base + position / PAGE_TOKENS) * PAGE_BYTES
            + position % PAGE_TOKENS * RECORD_BYTES
    }
}

/// A layer's pool pages and the plan's page space over them: each sequence's pages, as its block
/// table names them, then the scratch pages, the layer's last `SCRATCH_PAGES`.
pub struct Pages {
    /// Pool pages a layer.
    pub count: usize,
    /// The pool page of each page of the plan's space.
    map: Vec<usize>,
    /// The first scratch page of the plan's space.
    scratch: usize,
}

impl Pages {
    /// The space of `sequences`' pages, each a block table of pool pages, back to back, over a
    /// layer of `count` pool pages; refuses a layer without room for the scratch pages, or a
    /// sequence page among them.
    pub fn new(count: usize, sequences: impl IntoIterator<Item = usize>) -> Result<Self> {
        let first = count
            .checked_sub(SCRATCH_PAGES)
            .ok_or("a layer's pages leave no room for its scratch pages")?;
        let mut map: Vec<usize> = sequences.into_iter().collect();
        if map.iter().any(|&page| page >= first) {
            return Err("a sequence page lies on the scratch pages".into());
        }
        let scratch = map.len();
        map.extend(first..count);
        Ok(Self {
            count,
            map,
            scratch,
        })
    }

    /// The space of sequences that own consecutive pool pages from page 0, over a layer of `count`
    /// pool pages: the plan's space is the pool's.
    pub fn contiguous(count: usize) -> Self {
        Self {
            count,
            map: (0..count).collect(),
            scratch: count.saturating_sub(SCRATCH_PAGES),
        }
    }

    /// Pages of the plan's space.
    fn len(&self) -> usize {
        self.map.len()
    }

    fn physical(&self, page: usize) -> usize {
        self.map[page]
    }
}

/// The entries of each row a run pass holds.
const RUNS: usize = attention::Runs::SIZE;

pub fn offset(bytes: usize) -> Result<i32> {
    Ok(i32::try_from(bytes)?)
}

/// One merge pass's tables on the host, one inner vector per `Slice` entry.
#[derive(Default)]
pub struct Passes {
    pub statistics: Vec<Vec<i32>>,
    pub outputs: Vec<Vec<i32>>,
    pub visible: Vec<Vec<f32>>,
}

/// The plan's tensors on the host, one inner vector per `Slice` entry.
#[derive(Default)]
pub struct Tables {
    pub waves: Vec<Vec<i32>>,
    pub folds: Vec<Vec<i32>>,
    /// Each batch's slot kind, the `direct` of its `attention::word`.
    pub direct: Vec<Vec<i32>>,
    pub pages: Vec<Vec<i32>>,
    pub lengths: Vec<Vec<i32>>,
    pub counts: Vec<Vec<f32>>,
    pub queries: Vec<Vec<i32>>,
    /// Each batch's entries its final merge reads of every row.
    pub entries: Vec<Vec<i32>>,
    pub merge: Passes,
    pub fold: Passes,
}

/// One block of a row's history, at most 512 keys.
#[derive(Clone)]
struct Job {
    live: Row,
    query: i32,
    tokens: Range<usize>,
}

/// The job token at presence entry `at`, as the kernel lands a page's tokens: entry `[t][page]`
/// holds the page's token `t`.
#[cfg(test)]
const fn landed(at: usize) -> usize {
    PAGE_TOKENS * (at % JOB_PAGES) + at / JOB_PAGES
}

/// Job `at` of a full wave in the order a wave deals jobs to the PEs, each PE in turn (spread page
/// reads); the dealing is its own inverse.
const fn dealt(at: usize) -> usize {
    at - at % JOBS + at % PE_JOBS * PE_JOBS + at % JOBS / PE_JOBS
}

/// The slots before a batch's full waves: its partial wave goes first, so its Main work runs under
/// the next wave's gather instead of after the last one.
const fn lead(jobs: usize) -> usize {
    let full = jobs / JOBS * JOBS;
    if full > 0 && full < jobs { JOBS } else { 0 }
}

/// The job at slot `at` of a batch of `jobs`, `jobs` or more at an idle slot: the partial wave
/// packs its jobs so idle PEs skip theirs.
const fn job_at(at: usize, jobs: usize) -> usize {
    let (full, lead) = (jobs / JOBS * JOBS, lead(jobs));
    if at < lead {
        if at < jobs - full {
            full + at
        } else {
            jobs + at
        }
    } else if at - lead < full {
        dealt(at - lead)
    } else if full == 0 && at < jobs {
        at
    } else {
        jobs + at
    }
}

/// The slot of job `job` of a batch of `jobs`.
const fn slot_of(job: usize, jobs: usize) -> usize {
    let full = jobs / JOBS * JOBS;
    if job >= full {
        job - full
    } else {
        lead(jobs) + dealt(job)
    }
}

/// The jobs past a batch's full waves, split into near-equal pieces of the fewest pages that keep
/// them one wave and every row within its merge passes: the PEs of that wave then read short jobs,
/// and a piece rereads a page at most twice up to its PE's longest.
fn split(jobs: Vec<Job>, spans: &mut [Range<usize>], tail: Vec<bool>) -> (Vec<Job>, Vec<bool>) {
    let full = jobs.len() / JOBS * JOBS;
    if full == 0 || full == jobs.len() {
        return (jobs, tail);
    }
    let pieces = |job: &Job, pages: usize| job.tokens.len().div_ceil(pages * PAGE_TOKENS);
    let fits = |pages: usize| {
        (0..jobs.len())
            .filter(|&at| tail[at])
            .map(|at| pieces(&jobs[at], pages))
            .sum::<usize>()
            < JOBS
            && spans.iter().all(|span| {
                let added: usize = span
                    .clone()
                    .filter(|&at| tail[at])
                    .map(|at| pieces(&jobs[at], pages) - 1)
                    .sum();
                folds(span.len() + added) == folds(span.len())
            })
    };
    let pages = (1..JOB_PAGES)
        .find(|&pages| fits(pages))
        .unwrap_or(JOB_PAGES);
    let mut split = Vec::with_capacity(jobs.len() + JOBS);
    let mut tails = Vec::with_capacity(jobs.len() + JOBS);
    for span in spans.iter_mut() {
        let start = split.len();
        for (at, job) in jobs[span.clone()].iter().enumerate() {
            if !tail[span.start + at] {
                split.push(job.clone());
                tails.push(false);
                continue;
            }
            let (count, written) = (pieces(job, pages), job.tokens.len().div_ceil(PAGE_TOKENS));
            let token = |piece: usize| job.tokens.start + piece * written / count * PAGE_TOKENS;
            split.extend((0..count).map(|piece| Job {
                tokens: token(piece)..token(piece + 1).min(job.tokens.end),
                ..job.clone()
            }));
            tails.extend((0..count).map(|_| true));
        }
        *span = start..split.len();
    }
    (split, tails)
}

/// The keys of a batch's jobs. A batch's first wave, its partial one, reads every slot of a live PE
/// to the longest job there (one descriptor limit a PE), so a batch whose 512-key jobs fit one wave
/// takes the shortest page multiple that keeps it one wave and each row one merge pass.
fn block(rows: &[Row]) -> usize {
    let jobs = |block: usize| {
        rows.iter()
            .map(move |row| (row.position + 1).div_ceil(block))
    };
    if jobs(BLOCK).sum::<usize>() > JOBS {
        return BLOCK;
    }
    (PAGE_TOKENS..BLOCK)
        .step_by(PAGE_TOKENS)
        .find(|&block| {
            jobs(block).sum::<usize>() <= JOBS && jobs(block).all(|count| count <= ENTRIES)
        })
        .unwrap_or(BLOCK)
}

/// The fold passes a row of `jobs` jobs needs ahead of its final pass.
const fn folds(jobs: usize) -> usize {
    if jobs <= ENTRIES {
        0
    } else {
        (jobs - DIRECT).div_ceil(ENTRIES)
    }
}

impl Passes {
    /// Each pass's records of the first `RUNS` entries of every row.
    fn runs(&self, records: impl Fn(&Self) -> &Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        records(self)
            .iter()
            .map(|pass| {
                pass.chunks(ENTRIES)
                    .flat_map(|row| &row[..RUNS])
                    .copied()
                    .collect()
            })
            .collect()
    }

    fn push(&mut self, (records, visible): &(Vec<usize>, Vec<bool>)) -> Result<()> {
        let bytes = |record: usize| {
            records
                .iter()
                .map(|at| offset(at * record))
                .collect::<Result<Vec<_>>>()
        };
        self.statistics.push(bytes(STATISTIC_BYTES)?);
        self.outputs.push(bytes(OUTPUT_BYTES)?);
        self.visible
            .push(visible.iter().map(|&at| f32::from(u8::from(at))).collect());
        Ok(())
    }
}

/// Batch `batch`'s jobs over a layer of `pages` pool pages, with each row's run of them.
fn jobs(
    batch: usize,
    rows: &[Row],
    pages: &Pages,
) -> Result<(Vec<Job>, Vec<Range<usize>>, Vec<usize>)> {
    let block = block(rows);
    let mut jobs = Vec::new();
    let mut spans = Vec::with_capacity(BATCH);
    for (row, &live) in rows.iter().enumerate() {
        let length = live.position + 1;
        if length > MAX_CONTEXT || live.base + live.position / PAGE_TOKENS >= pages.len() {
            return Err(format!("row {row} of batch {batch} leaves the plan").into());
        }
        let first = jobs.len();
        let query = offset((batch * BATCH + row) * QUERY_BYTES)?;
        jobs.extend((0..length).step_by(block).map(|token| Job {
            live,
            query,
            tokens: token..(token + block).min(length),
        }));
        spans.push(first..jobs.len());
    }
    let tail = placement(&jobs);
    let (jobs, tail) = split(jobs, &mut spans, tail);
    // Full waves take the other jobs in order, the partial wave its jobs longest first, so a PE's
    // slots hold jobs of near-equal length.
    let mut lead: Vec<usize> = (0..jobs.len()).filter(|&at| tail[at]).collect();
    lead.sort_by_key(|&at| std::cmp::Reverse(jobs[at].tokens.len()));
    let order = (0..jobs.len())
        .filter(|&at| !tail[at])
        .chain(lead)
        .collect();
    Ok((jobs, spans, order))
}

/// Batch `batch`'s jobs with a partial wave of one job a row: each row's last job, the shortest
/// first, then each longest row's next one in turn, split as `split` does with `pieces`, longest
/// first in the wave.
fn spread(
    batch: usize,
    rows: &[Row],
    pieces: bool,
) -> Result<(Vec<Job>, Vec<Range<usize>>, Vec<usize>)> {
    let mut jobs = Vec::new();
    let mut spans = Vec::with_capacity(rows.len());
    for (row, &live) in rows.iter().enumerate() {
        let length = live.position + 1;
        let query = offset((batch * BATCH + row) * QUERY_BYTES)?;
        let first = jobs.len();
        jobs.extend((0..length).step_by(BLOCK).map(|token| Job {
            live,
            query,
            tokens: token..(token + BLOCK).min(length),
        }));
        spans.push(first..jobs.len());
    }
    let full = jobs.len() / JOBS * JOBS;
    let mut candidates: Vec<(usize, usize, std::cmp::Reverse<usize>, usize)> = spans
        .iter()
        .enumerate()
        .flat_map(|(row, span)| {
            (0..span.len()).map(move |back| {
                (
                    back,
                    row,
                    std::cmp::Reverse(span.len()),
                    span.end - 1 - back,
                )
            })
        })
        .map(|(back, _, longest, job)| (back, jobs[job].tokens.len(), longest, job))
        .collect();
    candidates.sort();
    let mut tail = vec![false; jobs.len()];
    for &(.., job) in &candidates[..jobs.len() - full] {
        tail[job] = true;
    }
    let (jobs, tail) = if pieces {
        split(jobs, &mut spans, tail)
    } else {
        (jobs, tail)
    };
    let mut lead: Vec<usize> = (0..jobs.len()).filter(|&at| tail[at]).collect();
    lead.sort_by_key(|&at| std::cmp::Reverse(jobs[at].tokens.len()));
    let order = (0..jobs.len())
        .filter(|&at| !tail[at])
        .chain(lead)
        .collect();
    Ok((jobs, spans, order))
}

/// Which jobs of a batch fill its partial wave: its shortest jobs, the latest among equals. Only
/// the partial wave reads a job up to its written tokens; a full wave reads a whole block a slot.
fn placement(jobs: &[Job]) -> Vec<bool> {
    let full = jobs.len() / JOBS * JOBS;
    let mut tail = vec![false; jobs.len()];
    if full == 0 || full == jobs.len() {
        return tail;
    }
    let mut at: Vec<usize> = (0..jobs.len()).collect();
    at.sort_by_key(|&at| (jobs[at].tokens.len(), std::cmp::Reverse(at)));
    for &at in &at[..jobs.len() - full] {
        tail[at] = true;
    }
    tail
}

/// The jobs of batch `batch`'s one row of at most `SINGLE_KEYS` keys: the shortest page multiple
/// that keeps them one wave.
fn single_jobs(batch: usize, row: &Row) -> Result<Vec<Job>> {
    let length = row.position + 1;
    let block = length.div_ceil(PAGE_TOKENS).div_ceil(JOBS) * PAGE_TOKENS;
    let query = offset(batch * BATCH * QUERY_BYTES)?;
    Ok((0..length)
        .step_by(block)
        .map(|token| Job {
            live: *row,
            query,
            tokens: token..(token + block).min(length),
        })
        .collect())
}

impl Tables {
    /// The plan of `rows` over a layer of `pages` pool pages, ahead of a prompt chunk of `chunk`
    /// rows in the same launch; every history must fit there.
    pub fn new(rows: &[Row], pages: &Pages, chunk: usize) -> Result<Self> {
        let mut tables = Self::default();
        for (batch, rows) in rows.chunks(BATCH).enumerate() {
            if batch == 0
                && rows.len() <= attention::FEW_ROWS
                && rows.iter().all(|row| row.position < attention::FEW_KEYS)
            {
                tables.few(rows, pages)?;
                continue;
            }
            if rows.len() == 1 && rows[0].position < SINGLE_KEYS {
                tables.single(batch, &rows[0], pages)?;
                tables.direct.push(vec![0]);
                continue;
            }
            let (jobs, spans, order) = jobs(batch, rows, pages)?;
            let blocks = rows.iter().map(|row| (row.position + 1).div_ceil(BLOCK));
            let (blocks, waves) = (blocks.max().unwrap_or(0), jobs.len().div_ceil(JOBS));
            if blocks <= waves {
                tables.fixed(batch, rows, pages, false)?;
                continue;
            }
            // Paired slots gather slower than the job plan's merge costs, unless that merge's
            // transfers would queue ahead of a prompt chunk's in the same launch.
            if chunk > 0 && rows.len() <= BATCH / 2 && blocks.div_ceil(2) <= waves {
                tables.fixed(batch, rows, pages, true)?;
                continue;
            }
            // ponytail: two shape bounds from the h1536 crossover (whole-row waves lose at b4,
            // win at b8 and b32): under half a wave the job plan's short jobs spread the gathers
            // over every PE and its merge stays cheap, and a whole-row wave past the first reads
            // all its slots' pages, so only the first may be partial. A host cost model of
            // gathers and merges replaces both bounds.
            let whole = rows.len().div_ceil(ROW_WAVE);
            if rows.iter().all(|row| row.position < ROW_KEYS)
                && whole <= waves
                && 2 * rows.len() >= ROW_WAVE
                && (whole <= 2 || rows.len() % ROW_WAVE == 0)
            {
                tables.rowwise(batch, rows, pages)?;
                continue;
            }
            if block(rows) == BLOCK && tables.segments(batch, rows, pages)? {
                continue;
            }
            tables.batch(&jobs, &order, pages)?;
            tables.merges(&jobs, &spans, &order)?;
            tables.direct.push(vec![0]);
        }
        Ok(tables)
    }

    /// A batch whose rows each keep one slot through waves of one block of up to 512 keys a row,
    /// row `r` in slot `dealt(r)`: the PEs read their slots' pages in turn, so a wave's PEs then
    /// read consecutive rows (b64 at 7,000 keys: page gathers 27.6 against 28.5 us with row `r` in
    /// slot `r`). Wave 0, whose gather reads only written pages, holds each row's last block, and
    /// the later waves its other blocks in turn from the slot's PE plus its slot time, so a slot
    /// time's PEs read different blocks of rows that may lie a power of two apart. A slot past its
    /// row's blocks reads layer pages that none of the batch's rows hold (while there are such
    /// pages) and the query of a distinct capacity row, masked. With `halves`, a batch of at most
    /// 32 rows gives row `r` slots `dealt(r)` and `dealt(r + 32)`, one PE's slot times `j` and
    /// `j + 4`: the first takes the row's first `ceil(blocks / 2)` blocks, the second the rest, its
    /// last block in wave 0, each in turn as above.
    fn fixed(&mut self, batch: usize, rows: &[Row], pages: &Pages, halves: bool) -> Result<()> {
        let blocks: Vec<usize> = rows
            .iter()
            .map(|row| (row.position + 1).div_ceil(BLOCK))
            .collect();
        let turn = |slot: usize| slot / PE_JOBS + slot % PE_JOBS;
        let job = |wave: usize, slot: usize| -> Option<Job> {
            let at = dealt(slot);
            let (at, half) = if halves {
                (at % (BATCH / 2), at / (BATCH / 2))
            } else {
                (at, 0)
            };
            let (row, blocks) = (rows.get(at)?, *blocks.get(at)?);
            let first = blocks.div_ceil(2);
            let rest = blocks - first;
            let block = match (halves, half, wave) {
                (false, _, 0) => blocks - 1,
                (false, _, _) if wave < blocks => (wave - 1 + turn(slot)) % (blocks - 1),
                (true, 0, _) if wave < first => (wave + turn(slot)) % first,
                (true, _, 0) if rest > 0 => blocks - 1,
                (true, _, _) if wave > 0 && wave < rest => {
                    first + (wave - 1 + turn(slot)) % (rest - 1)
                }
                _ => return None,
            };
            let length = row.position + 1;
            Some(Job {
                live: *row,
                query: offset((batch * BATCH + at) * QUERY_BYTES).ok()?,
                tokens: block * BLOCK..((block + 1) * BLOCK).min(length),
            })
        };
        let longest = blocks.iter().copied().max().unwrap_or(0);
        let waves = if halves { longest.div_ceil(2) } else { longest };
        self.slots(rows, pages, (waves, if halves { 2 } else { 1 }), job)
    }

    /// A batch whose waves each hold `ROW_WAVE` whole rows, merged on chip: each row takes the
    /// slots `whole_row` gives it, its history split into `ROW_JOBS` near-equal runs of whole
    /// pages.
    fn rowwise(&mut self, batch: usize, rows: &[Row], pages: &Pages) -> Result<()> {
        // The kernel's `rowwise` swap: a partial last block of two attends in wave 0.
        let swap = usize::from(rows.len() % ROW_WAVE != 0 && rows.len() > ROW_WAVE);
        let job = |wave: usize, slot: usize| -> Option<Job> {
            let at = whole_row(wave ^ swap, slot);
            let row = rows.get(at)?;
            let length = row.position + 1;
            let run = length.div_ceil(PAGE_TOKENS).div_ceil(attention::ROW_JOBS) * PAGE_TOKENS;
            let start = slot % attention::ROW_JOBS * run;
            if start >= length {
                return None;
            }
            Some(Job {
                live: *row,
                query: offset((batch * BATCH + at) * QUERY_BYTES).ok()?,
                tokens: start..(start + run).min(length),
            })
        };
        self.slots(rows, pages, (rows.len().div_ceil(ROW_WAVE), 3), job)
    }

    /// The tables of a batch of `waves` waves of slot kind `direct` whose slot `slot` of wave
    /// `wave` holds `job(wave, slot)`, and that runs no merge pass.
    fn slots(
        &mut self,
        rows: &[Row],
        pages: &Pages,
        (waves, direct): (usize, i32),
        job: impl Fn(usize, usize) -> Option<Job>,
    ) -> Result<()> {
        let held: std::collections::HashSet<usize> = rows
            .iter()
            .flat_map(|row| row.base..=row.base + row.position / PAGE_TOKENS)
            .map(|page| pages.physical(page))
            .collect();
        // Pages none of the rows hold, or any page when they hold them all.
        let mut spare: Vec<usize> = (0..pages.count)
            .filter(|page| !held.contains(page))
            .collect();
        if spare.is_empty() {
            spare = (0..pages.count).collect();
        }
        self.waves.push(vec![offset(waves)?]);
        self.folds.push(vec![0]);
        self.direct.push(vec![direct]);
        let mut lengths = Vec::with_capacity(attention::Blocks::SIZE * JOBS);
        for wave in 0..attention::Blocks::SIZE {
            let mut table = Vec::with_capacity(JOBS * JOB_PAGES);
            let mut idle = 0;
            for slot in 0..JOBS {
                match job(wave, slot) {
                    Some(job) => {
                        let written = job.tokens.len().div_ceil(PAGE_TOKENS);
                        for page in 0..JOB_PAGES {
                            let token = job.tokens.start + page % written * PAGE_TOKENS;
                            table.push(offset(job.live.record(pages, token))?);
                        }
                        lengths.push(job.tokens.len());
                    }
                    None => {
                        for page in 0..JOB_PAGES {
                            let at = spare[(idle * JOB_PAGES + page) % spare.len()];
                            table.push(offset(at * PAGE_BYTES)?);
                        }
                        idle += 1;
                        lengths.push(0);
                    }
                }
            }
            self.pages.push(table);
            self.lengths.push(
                lengths[wave * JOBS..]
                    .iter()
                    .map(|&length| offset(length))
                    .collect::<Result<_>>()?,
            );
        }
        for first in (0..attention::Blocks::SIZE * JOBS).step_by(BURST * JOBS) {
            let slots = first..first + BURST * JOBS;
            self.counts.push(
                (0..PAGE_TOKENS)
                    .flat_map(|token| {
                        lengths[slots.clone()].iter().flat_map(move |&length| {
                            [length.saturating_sub(token).div_ceil(PAGE_TOKENS) as f32; LANES]
                        })
                    })
                    .collect(),
            );
            self.queries.push(
                slots
                    .map(|slot| match job(slot / JOBS, slot % JOBS) {
                        Some(job) => Ok(job.query),
                        None => offset(slot % Capacity::SIZE * QUERY_BYTES),
                    })
                    .collect::<Result<_>>()?,
            );
        }
        self.entries.push(vec![0]);
        // No pass reads these: a batch of fixed slots or whole-row waves gathers no merge entries.
        let unread = || {
            (
                (0..BATCH * ENTRIES)
                    .map(|at| BURST * JOBS + at % JOBS)
                    .collect(),
                vec![false; BATCH * ENTRIES],
            )
        };
        self.merge.push(&unread())?;
        for _ in 0..FOLDS {
            self.fold.push(&unread())?;
        }
        Ok(())
    }

    /// A batch of one row of at most `SINGLE_KEYS` keys, merged on chip: no job waves or fold
    /// passes, and one first wave of the row's jobs over every PE. Its merge tables stay unread.
    fn single(&mut self, batch: usize, row: &Row, pages: &Pages) -> Result<()> {
        if row.base + row.position / PAGE_TOKENS >= pages.len() {
            return Err(format!("the row of batch {batch} leaves the plan").into());
        }
        let jobs = single_jobs(batch, row)?;
        let order: Vec<usize> = (0..jobs.len()).collect();
        self.batch(&jobs, &order, pages)?;
        self.merges(&jobs, &[0..jobs.len()], &order)?;
        *self.waves.last_mut().unwrap() = vec![0];
        *self.folds.last_mut().unwrap() = vec![0];
        *self.entries.last_mut().unwrap() = vec![0];
        Ok(())
    }

    /// The first batch of at most `FEW_ROWS` rows of at most `FEW_KEYS` keys in one wave with no
    /// merge, its word's entries its rows: `attention::fewone` for one row, `fewfour` up to 4,
    /// `feweight` up to 8. Row `r`'s pages from `128 r` of the wave-1 page table and its length
    /// at `r` of the wave-1 length table, and each slice's page times less its count of present
    /// pages (a key is absent where that reaches 0): one row's page groups `p % 8` as merge
    /// visibility 0, `[g][t][k]`; up to 4 rows' groups `p % 2` as the first burst's counts,
    /// `[r][g][t][k]`; up to 8 rows' pages there, `[r][t][p]`. Other wave-1 entries are 0, and
    /// the rest of the tables are a one-row batch's of row 0.
    fn few(&mut self, rows: &[Row], pages: &Pages) -> Result<()> {
        if rows
            .iter()
            .any(|row| row.base + row.position / PAGE_TOKENS >= pages.len())
        {
            return Err("a row of the first batch leaves the plan".into());
        }
        let count = rows.len();
        let jobs = single_jobs(0, &rows[0])?;
        let order: Vec<usize> = (0..jobs.len()).collect();
        self.batch(&jobs, &order, pages)?;
        self.merges(&jobs, &[0..jobs.len()], &order)?;
        self.direct.push(vec![0]);
        if count > PE_JOBS {
            self.parts(rows, pages)?;
        } else {
            self.eight(rows, pages)?;
        }
        *self.waves.last_mut().unwrap() = vec![0];
        *self.folds.last_mut().unwrap() = vec![0];
        *self.direct.last_mut().unwrap() = vec![5];
        *self.entries.last_mut().unwrap() = vec![offset(count)?];
        Ok(())
    }

    /// `few`'s tables of up to 8 rows.
    fn eight(&mut self, rows: &[Row], pages: &Pages) -> Result<()> {
        let count = rows.len();
        let row_pages = attention::FEW_KEYS / PAGE_TOKENS;
        let slots = if count == 1 {
            1
        } else if count <= 4 {
            4
        } else {
            8
        };
        let (groups, times) = match slots {
            1 => (8, 16),
            4 => (2, 64),
            _ => (1, 128),
        };
        let mut beyond = vec![0.0f32; slots * groups * PAGE_TOKENS * times];
        self.pages[1] = vec![0; JOBS * JOB_PAGES];
        self.lengths[1] = vec![0; JOBS];
        for r in 0..slots {
            let length = rows.get(r).map_or(0, |row| row.position + 1);
            if let Some(row) = rows.get(r) {
                let written = length.div_ceil(PAGE_TOKENS);
                for page in 0..row_pages {
                    self.pages[1][row_pages * r + page] =
                        offset(row.record(pages, page.min(written - 1) * PAGE_TOKENS))?;
                }
            }
            self.lengths[1][r] = offset(length)?;
            for group in 0..groups {
                for token in 0..PAGE_TOKENS {
                    let holding = length.saturating_sub(token).div_ceil(PAGE_TOKENS);
                    let present = holding.saturating_sub(group).div_ceil(groups);
                    for time in 0..times {
                        beyond[((r * groups + group) * PAGE_TOKENS + token) * times + time] =
                            time as f32 - present as f32;
                    }
                }
            }
        }
        if count == 1 {
            self.merge.visible[0] = beyond;
        } else {
            self.counts[0][..beyond.len()].copy_from_slice(&beyond);
        }
        Ok(())
    }

    /// `few`'s tables of 9 to 32 rows (`attention::fewwide`): each 16-row wave `w` of whole rows
    /// as a `rowwise` batch, its half `h`'s jobs `8 p + 4 h + j` as `[p][j]` of tables
    /// `1 + 2 w + h` and its counts as burst `w`. An odd last part packs wave 1's rows one a PE:
    /// PE `p = 4 c + q`'s jobs `0 .. 4` hold row `2 q + c`.
    fn parts(&mut self, rows: &[Row], pages: &Pages) -> Result<()> {
        let packed = rows.len().div_ceil(PE_JOBS) % 2 == 1;
        let burst = JOBS * LANES;
        for (wave, rows) in rows.chunks(ROW_WAVE).enumerate() {
            let mut own = Self::default();
            own.rowwise(0, rows, pages)?;
            let parts: Vec<(usize, Vec<(usize, usize)>)> = if packed && wave == 1 {
                let sources = (0..PE_JOBS)
                    .flat_map(|p| {
                        let row = 2 * (p % 4) + p / 4;
                        let (pe, half) = (4 * (row % 4 / 2) + row / 4, row % 2);
                        (0..4).map(move |j| (PE_JOBS * p + j, PE_JOBS * pe + 4 * half + j))
                    })
                    .collect();
                vec![(2, sources)]
            } else {
                (0..2)
                    .map(|half| {
                        let sources = (0..PE_JOBS)
                            .flat_map(|p| (0..4).map(move |j| PE_JOBS * p + 4 * half + j))
                            .map(|slot| (slot, slot))
                            .collect();
                        (2 * wave + half, sources)
                    })
                    .collect()
            };
            for token in 0..PAGE_TOKENS {
                let (at, from) = ((token * BURST + wave) * burst, token * BURST * burst);
                for (_, sources) in &parts {
                    for &(slot, source) in sources {
                        self.counts[0][at + slot * LANES..][..LANES]
                            .copy_from_slice(&own.counts[0][from + source * LANES..][..LANES]);
                    }
                }
            }
            for (part, sources) in parts {
                let (mut table, mut lengths) = (own.pages[0].clone(), own.lengths[0].clone());
                for (at, (_, source)) in sources.into_iter().enumerate() {
                    table[at * JOB_PAGES..][..JOB_PAGES]
                        .copy_from_slice(&own.pages[0][source * JOB_PAGES..][..JOB_PAGES]);
                    lengths[at] = own.lengths[0][source];
                    lengths[JOBS / 2 + at] = 0;
                }
                (self.pages[1 + part], self.lengths[1 + part]) = (table, lengths);
            }
        }
        Ok(())
    }

    /// One batch's job waves: every slot's pages, and each burst's token counts and queries. An idle
    /// slot reads the pages of a distinct job and the query of a distinct capacity row, masked:
    /// one page or query read by many slots serializes the gather.
    fn batch(&mut self, jobs: &[Job], order: &[usize], pages: &Pages) -> Result<()> {
        self.waves.push(vec![offset(jobs.len().div_ceil(JOBS))?]);
        // Job `order[v]` takes position `v` of `job_at`'s slots.
        let at = |slot: usize| order.get(job_at(slot, jobs.len())).map(|&job| &jobs[job]);
        let source = |slot: usize| &jobs[order[job_at(slot, jobs.len()) % jobs.len()]];
        for wave in 0..attention::Blocks::SIZE {
            let mut table = Vec::with_capacity(JOBS * JOB_PAGES);
            for slot in wave * JOBS..(wave + 1) * JOBS {
                let job = source(slot);
                let written = job.tokens.len().div_ceil(PAGE_TOKENS);
                // A page past the block rereads one of the job's written pages, masked.
                for page in 0..JOB_PAGES {
                    let token = job.tokens.start + page % written * PAGE_TOKENS;
                    table.push(offset(job.live.record(pages, token))?);
                }
            }
            self.pages.push(table);
            self.lengths.push(
                (wave * JOBS..(wave + 1) * JOBS)
                    .map(|slot| offset(at(slot).map_or(0, |job| job.tokens.len())))
                    .collect::<Result<_>>()?,
            );
        }
        for first in (0..attention::Blocks::SIZE * JOBS).step_by(BURST * JOBS) {
            let slots = first..first + BURST * JOBS;
            let lengths: Vec<usize> = slots
                .clone()
                .map(|slot| at(slot).map_or(0, |job| job.tokens.len()))
                .collect();
            self.counts.push(
                (0..PAGE_TOKENS)
                    .flat_map(|token| {
                        lengths.iter().flat_map(move |&length| {
                            [length.saturating_sub(token).div_ceil(PAGE_TOKENS) as f32; LANES]
                        })
                    })
                    .collect(),
            );
            self.queries.push(
                slots
                    .map(|slot| match at(slot) {
                        Some(job) => Ok(job.query),
                        None => offset(slot % Capacity::SIZE * QUERY_BYTES),
                    })
                    .collect::<Result<_>>()?,
            );
        }
        Ok(())
    }

    /// A batch of 512-key jobs on runs: its partial wave holds the job plan's pieces, each its
    /// own entry, and each slot of its full waves carries one row's softmax over a run of the
    /// row's other jobs, leaving one partial a run, so its merge reads a few entries a row. Each
    /// row's run in its own slot starts with its partial job (a continuing job is whole, its
    /// counts [`attention::CARRIED`]); longer rows overflow in whole runs into the slots shorter
    /// rows leave. False, and no tables, if a row would leave more partials than a run pass reads.
    fn segments(&mut self, batch: usize, rows: &[Row], pages: &Pages) -> Result<bool> {
        let total: usize = rows
            .iter()
            .map(|row| (row.position + 1).div_ceil(BLOCK))
            .sum();
        let full = total / JOBS * JOBS;
        // grid[wave][slot]: the job there and whether it starts a run; each row's runs.
        let lay = |(jobs, spans, order): (Vec<Job>, Vec<Range<usize>>, Vec<usize>)| {
            let lead = usize::from(full < jobs.len());
            let waves = lead + full / JOBS;
            let row_of = |job: usize| spans.iter().position(|span| span.contains(&job));
            let mut grid: Vec<Vec<Option<(usize, bool)>>> = vec![vec![None; JOBS]; waves];
            let mut runs = vec![0; spans.len()];
            for (slot, &job) in order[full..].iter().enumerate() {
                grid[0][slot] = Some((job, true));
                runs[row_of(job)?] += 1;
            }
            // Each row's full-wave jobs, its partial one first.
            let mut queue: Vec<Vec<usize>> = vec![Vec::new(); spans.len()];
            for &job in &order[..full] {
                queue[row_of(job)?].push(job);
            }
            for jobs_of in &mut queue {
                jobs_of.sort_by_key(|&job| jobs[job].tokens.len());
            }
            let mut next = vec![lead; JOBS];
            let mut by_count: Vec<usize> = (0..spans.len()).collect();
            by_count.sort_by_key(|&row| (std::cmp::Reverse(queue[row].len()), row));
            let mut place = |row: usize, slot: usize, taken: &mut usize, next: &mut [usize]| {
                let count = (queue[row].len() - *taken).min(waves - next[slot]);
                for at in 0..count {
                    grid[next[slot] + at][slot] = Some((queue[row][*taken + at], at == 0));
                }
                next[slot] += count;
                *taken += count;
                runs[row] += usize::from(count > 0);
                count
            };
            let mut taken = vec![0; spans.len()];
            for (slot, &row) in by_count.iter().enumerate() {
                place(row, slot, &mut taken[row], &mut next);
            }
            for &row in &by_count {
                while taken[row] < queue[row].len() {
                    let slot = (0..JOBS)
                        .max_by_key(|&slot| (waves - next[slot], std::cmp::Reverse(slot)))
                        .unwrap_or(0);
                    if place(row, slot, &mut taken[row], &mut next) == 0 {
                        return None;
                    }
                }
            }
            (runs.iter().all(|&count| count <= RUNS)).then_some((jobs, spans, grid, runs))
        };
        // The job plan's partial wave (its shortest jobs as pieces); else one job a row and the
        // longest rows' next ones, as pieces, then whole: each leaves a row fewer partials.
        let Some((jobs, spans, grid, runs)) = lay(jobs(batch, rows, pages)?)
            .or(lay(spread(batch, rows, true)?))
            .or(lay(spread(batch, rows, false)?))
        else {
            return Ok(false);
        };
        let waves = grid.len();
        let row_of = |job: usize| {
            spans
                .iter()
                .position(|span| span.contains(&job))
                .unwrap_or(0)
        };
        let job = |wave: usize, slot: usize| -> Option<(&Job, bool)> {
            let (job, starts) = (*grid.get(wave)?.get(slot)?)?;
            Some((&jobs[job], starts))
        };
        let held: std::collections::HashSet<usize> = rows
            .iter()
            .flat_map(|row| row.base..=row.base + row.position / PAGE_TOKENS)
            .map(|page| pages.physical(page))
            .collect();
        let mut spare: Vec<usize> = (0..pages.count)
            .filter(|page| !held.contains(page))
            .collect();
        if spare.is_empty() {
            spare = (0..pages.count).collect();
        }
        self.waves.push(vec![offset(waves)?]);
        self.folds.push(vec![0]);
        self.direct.push(vec![4]);
        let mut lengths = Vec::with_capacity(attention::Blocks::SIZE * JOBS);
        let mut unused = 0;
        for wave in 0..attention::Blocks::SIZE {
            let mut table = Vec::with_capacity(JOBS * JOB_PAGES);
            for slot in 0..JOBS {
                match job(wave, slot) {
                    Some((job, _)) => {
                        let written = job.tokens.len().div_ceil(PAGE_TOKENS);
                        for page in 0..JOB_PAGES {
                            let token = job.tokens.start + page % written * PAGE_TOKENS;
                            table.push(offset(job.live.record(pages, token))?);
                        }
                        lengths.push(job.tokens.len());
                    }
                    None => {
                        for page in 0..JOB_PAGES {
                            let at = spare[(unused * JOB_PAGES + page) % spare.len()];
                            table.push(offset(at * PAGE_BYTES)?);
                        }
                        unused += 1;
                        lengths.push(0);
                    }
                }
            }
            self.pages.push(table);
            self.lengths.push(
                lengths[wave * JOBS..]
                    .iter()
                    .map(|&length| offset(length))
                    .collect::<Result<_>>()?,
            );
        }
        for first in (0..attention::Blocks::SIZE * JOBS).step_by(BURST * JOBS) {
            let slots = first..first + BURST * JOBS;
            let present = |slot: usize, token: usize| match job(slot / JOBS, slot % JOBS) {
                Some((_, false)) => attention::CARRIED,
                _ => lengths[slot].saturating_sub(token).div_ceil(PAGE_TOKENS) as f32,
            };
            self.counts.push(
                (0..PAGE_TOKENS)
                    .flat_map(|token| {
                        slots
                            .clone()
                            .flat_map(move |slot| [present(slot, token); LANES])
                    })
                    .collect(),
            );
            self.queries.push(
                slots
                    .map(|slot| match job(slot / JOBS, slot % JOBS) {
                        Some((job, _)) => Ok(job.query),
                        None => offset(slot % Capacity::SIZE * QUERY_BYTES),
                    })
                    .collect::<Result<_>>()?,
            );
        }
        // Each row's partials: the record of every run's last job; idle entries read distinct
        // records the waves write.
        let record = |wave: usize, slot: usize| BURST * JOBS + wave * JOBS + slot;
        let mut records: Vec<usize> = (0..BATCH * ENTRIES)
            .map(|at| BURST * JOBS + at % (waves * JOBS))
            .collect();
        let mut visible = vec![false; BATCH * ENTRIES];
        let mut seen = vec![0; rows.len()];
        for slot in 0..JOBS {
            for wave in 0..waves {
                let Some((job, _)) = grid[wave][slot] else {
                    continue;
                };
                let ends = grid
                    .get(wave + 1)
                    .and_then(|next| next[slot])
                    .is_none_or(|(_, starts)| starts);
                if ends {
                    let row = row_of(job);
                    records[row * ENTRIES + seen[row]] = record(wave, slot);
                    visible[row * ENTRIES + seen[row]] = true;
                    seen[row] += 1;
                }
            }
        }
        self.entries
            .push(vec![offset(runs.iter().copied().max().unwrap_or(0))?]);
        let merge = (records, visible);
        self.merge.push(&merge)?;
        for _ in 0..FOLDS {
            self.fold.push(&merge)?;
        }
        Ok(true)
    }

    /// One batch's final merge pass, then its fold passes: a pass past the longest row's folds
    /// repeats the final pass and stays unread.
    fn merges(&mut self, jobs: &[Job], spans: &[Range<usize>], order: &[usize]) -> Result<()> {
        let mut rank = vec![0; order.len()];
        for (at, &job) in order.iter().enumerate() {
            rank[job] = at;
        }
        let passes = 1 + spans
            .iter()
            .map(|span| folds(span.len()))
            .max()
            .unwrap_or(0);
        // A job's record follows the first entry's `BURST` waves of records, its first `FOLDS`
        // waves the fold records.
        let record = |job: usize| BURST * JOBS + slot_of(rank[job], jobs.len());
        // Idle entries read distinct written records, an idle slot's among them: one shared record
        // serializes its HBM reads.
        let idle = |at: usize| BURST * JOBS + at % (jobs.len().div_ceil(JOBS) * JOBS);
        let mut merges: Vec<(Vec<usize>, Vec<bool>)> = (0..passes)
            .map(|pass| {
                (
                    (0..BATCH * ENTRIES)
                        .map(|entry| idle(pass * BATCH * ENTRIES + entry))
                        .collect(),
                    vec![false; BATCH * ENTRIES],
                )
            })
            .collect();
        let mut read = |pass: usize, row: usize, entry: usize, at: usize| {
            merges[pass].0[row * ENTRIES + entry] = at;
            merges[pass].1[row * ENTRIES + entry] = true;
        };
        for (row, span) in spans.iter().enumerate() {
            let direct = if folds(span.len()) == 0 {
                span.len()
            } else {
                DIRECT
            };
            for (entry, job) in (span.start..span.start + direct).enumerate() {
                read(0, row, entry, record(job));
            }
            for (fold, first) in (span.start + direct..span.end).step_by(ENTRIES).enumerate() {
                read(0, row, DIRECT + fold, fold * JOBS + row);
                for (entry, job) in (first..(first + ENTRIES).min(span.end)).enumerate() {
                    read(1 + fold, row, entry, record(job));
                }
            }
        }
        let entries = spans
            .iter()
            .map(|span| span.len().min(ENTRIES))
            .max()
            .unwrap_or(0);
        self.entries.push(vec![offset(entries)?]);
        self.folds.push(vec![offset(passes - 1)?]);
        self.merge.push(&merges[0])?;
        for fold in 1..=FOLDS {
            self.fold.push(merges.get(fold).unwrap_or(&merges[0]))?;
        }
        Ok(())
    }
}

async fn slice<D: MaterializableScalar + Copy, E: M>(
    device: &mut Device,
    tensors: &[Vec<D>],
) -> Result<Slice<HbmTensor<D, Chips, E>>> {
    let mut uploaded = Vec::with_capacity(tensors.len());
    for values in tensors {
        uploaded.push(copies(device, values).await?);
    }
    Ok(uploaded.into())
}

async fn passes(device: &mut Device, passes: &Passes) -> Result<attention::Passes> {
    Ok(attention::Passes {
        statistics: slice(device, &passes.statistics).await?,
        outputs: slice(device, &passes.outputs).await?,
        visible: slice(device, &passes.visible).await?,
    })
}

/// The attention plan of `rows` over a layer of `pages` pool pages ahead of a prompt chunk of
/// `chunk` rows, on both chips. A step whose rows are all a prompt chunk still carries one batch,
/// of one row on the last page, never read.
pub async fn plan(
    device: &mut Device,
    rows: &[Row],
    (pages, chunk): (&Pages, usize),
) -> Result<attention::Plan> {
    let idle = [Row {
        token: 0,
        base: pages.len() - 1,
        position: 0,
    }];
    Tables::new(if rows.is_empty() { &idle } else { rows }, pages, chunk)?
        .upload(device)
        .await
}

impl Tables {
    /// The tables on both chips, as the step reads them.
    pub async fn upload(&self, device: &mut Device) -> Result<attention::Plan> {
        Ok(attention::Plan {
            batches: slice(device, &self.words()?).await?,
            pages: slice(device, &self.pages).await?,
            lengths: {
                // The kernel chooses the few-row chains' tables 1 ..= 4 in every step.
                let mut lengths = self.lengths.clone();
                lengths.resize(lengths.len().max(5), vec![0; JOBS]);
                slice(device, &lengths).await?
            },
            counts: slice(device, &self.counts).await?,
            positions: copies(
                device,
                &(0..JOB_PAGES).map(|page| page as f32).collect::<Vec<_>>(),
            )
            .await?,
            queries: slice(device, &self.queries).await?,
            merge: passes(device, &self.merge).await?,
            runs: attention::RunPasses {
                statistics: slice(device, &self.merge.runs(|pass| &pass.statistics)).await?,
                outputs: slice(device, &self.merge.runs(|pass| &pass.outputs)).await?,
                visible: slice(
                    device,
                    &self
                        .merge
                        .visible
                        .iter()
                        .map(|seen| {
                            seen.chunks(ENTRIES)
                                .flat_map(|row| &row[..RUNS])
                                .flat_map(|&tag| [tag; LANES])
                                .collect()
                        })
                        .collect::<Vec<_>>(),
                )
                .await?,
            },
            fold: passes(device, &self.fold).await?,
        })
    }

    fn words(&self) -> Result<Vec<Vec<i32>>> {
        let count = |table: &[Vec<i32>], batch: usize| usize::try_from(table[batch][0]);
        (0..self.waves.len())
            .map(|batch| {
                let (waves, folds) = (count(&self.waves, batch)?, count(&self.folds, batch)?);
                let (direct, entries) = (count(&self.direct, batch)?, count(&self.entries, batch)?);
                Ok(vec![offset(attention::word(
                    waves, folds, direct, entries,
                ))?])
            })
            .collect()
    }
}

/// The prompt chunk's host tables: its 256-row blocks, its first row, its segments' bounds and
/// the spans of the geometry each segment attends in, the other geometries left empty.
pub struct Prompt {
    pub blocks: i32,
    pub start: i32,
    /// Each segment's first chunk row, then the chunk's rows.
    pub bounds: Vec<i32>,
    pub narrow: SpanTables,
    pub wide: SpanTables,
    pub small: SpanTables,
    /// Each segment's last wide span attended in the half geometry (1 or 0).
    pub half: Vec<i32>,
    /// The short path's 1,024-row passes; when nonzero the chunk attends no span.
    pub passes: i32,
    /// The short path's key pages and `[row][lane]` last visible keys.
    pub short_pages: Vec<i32>,
    pub short_limits: Vec<f32>,
    /// The chunk attends in spans and `ROUTE` models it faster D-sharded. The TP2 plan then reads
    /// no chunk, and the step's TP2 body only the rows before it.
    pub routed: bool,
    /// The job-plan rows before the chunk's padding.
    pub decode: usize,
}

/// One geometry's spans of the chunk's segments: how many each attends, how many each one's rows
/// see whole, the spans' key pages and each row's last visible key in every span.
pub struct SpanTables {
    /// Keys a span, query rows a block, spans of the plan's reach.
    pub geometry: (usize, usize, usize),
    /// Each segment's spans, 0 when it attends in another geometry.
    pub spans: Vec<i32>,
    pub full: Vec<i32>,
    /// Each segment's first position in query blocks, rounded up.
    pub ahead: Vec<i32>,
    /// One table of key pages a span of the plan's reach and segment: segment `s`'s span `t` at
    /// `s * reach + t`.
    pub pages: Vec<Vec<i32>>,
    /// `[span][row][lane]`: each lane holds the row's last visible key of its own sequence's span,
    /// -1 for none.
    pub limits: Vec<f32>,
}

const CHUNK: usize = attention::prompt::Prompt::SIZE;
const SEGMENTS: usize = attention::prompt::Segments::SIZE;
/// A segment's rows: its blocks keep their first span's peaks and product in data memory.
const SEGMENT_ROWS: usize =
    attention::prompt::narrow::QueryBlocks::SIZE * attention::prompt::narrow::Queries::SIZE;
const _: () = assert!(
    SEGMENT_ROWS
        == attention::prompt::wide::QueryBlocks::SIZE * attention::prompt::wide::Queries::SIZE
);
const NARROW: (usize, usize, usize) = (
    attention::prompt::narrow::Keys::SIZE,
    attention::prompt::narrow::Queries::SIZE,
    attention::prompt::narrow::Spans::SIZE,
);
const WIDE: (usize, usize, usize) = (
    attention::prompt::wide::Keys::SIZE,
    attention::prompt::wide::Queries::SIZE,
    attention::prompt::wide::Spans::SIZE,
);
const SMALL: (usize, usize, usize) = (
    attention::prompt::small::Keys::SIZE,
    attention::prompt::small::Queries::SIZE,
    attention::prompt::small::Spans::SIZE,
);
const _: () = assert!(
    SEGMENT_ROWS
        == attention::prompt::small::QueryBlocks::SIZE * attention::prompt::small::Queries::SIZE
);

/// Microseconds the D-sharded route adds to a launch over the TP2 body, per count of the launch's
/// shape (fitted on the step bench at 40 layers, both routes of one image).
pub struct Route {
    /// Each 256-row block of the chunk, padded: the D route's unit.
    pub block: i64,
    /// Each 64-row wave of the chunk: the TP2 body's unit.
    pub wave: i64,
    pub launch: i64,
    /// Beside decode rows the D route passes over the weights a second time, hidden more by
    /// each decode wave.
    pub beside: i64,
    pub decode_wave: i64,
    /// An odd block count: the D route's down contracts two blocks a panel, and one alone.
    pub odd: i64,
}

pub const ROUTE: Route = Route {
    block: 8_438,
    wave: -2_435,
    launch: -1_878,
    beside: 4_069,
    decode_wave: -236,
    odd: 3_227,
};

impl Route {
    /// Whether a chunk of `rows` after `start` decode rows (whole waves) runs faster D-sharded.
    pub fn sharded(&self, start: usize, rows: usize) -> bool {
        let count = |rows: usize, unit: usize| rows.div_ceil(unit) as i64;
        let beside = if start == 0 {
            0
        } else {
            self.beside + self.decode_wave * count(start, BATCH)
        };
        let blocks = count(rows, 4 * BATCH);
        let delta = self.block * blocks + self.odd * (blocks % 2) + self.wave * count(rows, BATCH);
        rows > 0 && delta + self.launch + beside < 0
    }
}
const KEYS: usize = NARROW.0;
const SPANS: usize = NARROW.2;
#[cfg(test)]
const SPAN_PAGES: usize = KEYS / PAGE_TOKENS;
const SHORT_KEYS: usize = attention::short::Keys::SIZE;
const SHORT_ROWS: usize = attention::short::Queries::SIZE;
const SHORT_PAGES: usize = SHORT_KEYS / PAGE_TOKENS;

/// A geometry's pair costs, tenths of a us a layer: a block's first span, each later span of a
/// block, and each span's landing and decode (narrow and wide: 125 / 100 / 12.5 us; small from the
/// row-parallel pieces of fresh 512- and 1,024-row prompts, 1 and 3 pairs over 1 and 2 spans, and
/// their Traces: 21.9 a block's first pair, 20.9 a later one, 2.5 a span's decode).
fn pair_costs(geometry: (usize, usize, usize)) -> (usize, usize, usize) {
    if geometry == SMALL {
        (219, 209, 25)
    } else {
        (1250, 1000, 125)
    }
}

impl SpanTables {
    fn new(geometry: (usize, usize, usize), segments: usize) -> Self {
        let (keys, _, spans) = geometry;
        Self {
            geometry,
            spans: vec![0; segments],
            full: vec![0; segments],
            ahead: vec![0; segments],
            pages: vec![vec![0; keys / PAGE_TOKENS]; spans * segments],
            limits: vec![-1.0; spans * CHUNK * attention::Lane::SIZE],
        }
    }

    /// Segment `segment`'s spans over `rows`, one sequence's consecutive positions, in a layer of
    /// `pages` pool pages.
    fn fill(&mut self, (segment, rows): (usize, &[Row]), pages: &Pages) -> Result<()> {
        let (keys, queries, reach) = self.geometry;
        let span_pages = keys / PAGE_TOKENS;
        // Entries past the segment read the layer's following pages, masked: on chip, a gather
        // whose entries repeat one page lands wrong data, so every entry reads its own page.
        if pages.count < span_pages {
            return Err(format!("a chunk's gather needs {span_pages} pages a layer").into());
        }
        let (first, last) = (rows[0], rows[rows.len() - 1].position);
        self.spans[segment] = offset(last / keys + 1)?;
        self.full[segment] = offset((first.position + 1) / keys)?;
        self.ahead[segment] = offset(first.position.div_ceil(queries))?;
        let tables = &mut self.pages[segment * reach..(segment + 1) * reach];
        for (span, table) in tables.iter_mut().enumerate() {
            for (page, at) in table.iter_mut().enumerate() {
                let page = (first.base + span * span_pages + page) % pages.len();
                *at = offset(pages.physical(page) * PAGE_BYTES)?;
            }
        }
        Ok(())
    }

    /// Every chunk row's last visible key of each span of its own sequence; a row past the chunk
    /// attends as its last row, so its softmax stays finite.
    fn limit(&mut self, chunk: &[Row]) -> Result<()> {
        let (keys, _, _) = self.geometry;
        let lanes = self.limits.chunks_exact_mut(attention::Lane::SIZE);
        for (at, lanes) in lanes.enumerate() {
            let (span, row) = (at / CHUNK, at % CHUNK);
            let sees = chunk[row.min(chunk.len() - 1)].position;
            let limit = (sees + 1).saturating_sub(span * keys).min(keys);
            lanes.fill(f32::from(u16::try_from(limit)?) - 1.0);
        }
        Ok(())
    }

    /// The positions of a span's keys, in key order.
    fn positions(&self) -> Result<Vec<f32>> {
        (0..self.geometry.0)
            .map(|key| Ok(f32::from(u16::try_from(key)?)))
            .collect()
    }

    /// Each segment's spans, whole spans and spans ahead, as tables of one entry a segment.
    fn counts(&self) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        let table = |values: &[i32]| {
            let mut table = values.to_vec();
            table.resize(SEGMENTS, 0);
            table
        };
        (table(&self.spans), table(&self.full), table(&self.ahead))
    }
}

impl Prompt {
    /// The chunk of rows `start..`, segments of consecutive positions of one sequence each that
    /// the job plan does not attend, or no chunk: `start` past every row's wave keeps them all on
    /// the job plan. Each segment attends in the geometry whose spans and blocks cost less; the
    /// step runs no short path (the small spans supersede it), so its passes stay 0.
    pub fn new(rows: &[Row], start: usize, pages: &Pages) -> Result<Self> {
        Self::with_short(rows, start, pages, false)
    }

    /// As `new`, with `short` allowing a one-segment chunk that sees at most a short tile's keys
    /// to attend in the short path's passes.
    pub fn with_short(rows: &[Row], start: usize, pages: &Pages, short: bool) -> Result<Self> {
        let chunk = rows.get(start..).unwrap_or_default();
        let blocks = chunk.len().div_ceil(4 * BATCH);
        // A segment starts where a row does not continue the row before it in one sequence.
        let continues = |at: usize| {
            chunk[at].base == chunk[at - 1].base && chunk[at].position == chunk[at - 1].position + 1
        };
        let bounds: Vec<usize> = (0..chunk.len())
            .filter(|&at| at == 0 || !continues(at))
            .chain([chunk.len()])
            .collect();
        let segments = bounds.len() - 1;
        let tables = segments.max(1);
        let mut prompt = Self {
            blocks: offset(blocks)?,
            start: offset(start)?,
            bounds: bounds
                .iter()
                .map(|&bound| offset(bound))
                .collect::<Result<_>>()?,
            narrow: SpanTables::new(NARROW, tables),
            wide: SpanTables::new(WIDE, tables),
            small: SpanTables::new(SMALL, tables),
            half: vec![0; tables],
            passes: 0,
            short_pages: vec![0; SHORT_PAGES],
            short_limits: vec![-1.0; CHUNK * attention::Lane::SIZE],
            routed: false,
            decode: rows[..start.min(rows.len())]
                .iter()
                .take_while(|row| row.base != pages.scratch)
                .count(),
        };
        let refused = || {
            format!(
                "chunk at row {start} of {} rows in {segments} segments leaves the plan",
                chunk.len()
            )
        };
        if start % BATCH != 0
            || chunk.len() > CHUNK
            || start / BATCH + blocks * 4 > Waves::SIZE
            || segments > SEGMENTS
        {
            return Err(refused().into());
        }
        if chunk.is_empty() {
            return Ok(prompt);
        }
        // A chunk whose one segment sees at most a short tile's keys attends in passes of 1,024
        // rows and no span. A pass reads 1,024 capacity rows of queries, so the last one must end
        // inside them.
        let (first, last) = (chunk[0], chunk[chunk.len() - 1].position);
        let short = short
            && segments == 1
            && last < SHORT_KEYS
            && start + chunk.len().div_ceil(SHORT_ROWS) * SHORT_ROWS <= Capacity::SIZE;
        if short {
            prompt.passes = offset(chunk.len().div_ceil(SHORT_ROWS))?;
            for (page, at) in prompt.short_pages.iter_mut().enumerate() {
                *at = offset(pages.physical((first.base + page) % pages.len()) * PAGE_BYTES)?;
            }
            let lanes = prompt.short_limits.chunks_exact_mut(attention::Lane::SIZE);
            for (row, lanes) in lanes.enumerate() {
                let sees = chunk[row.min(chunk.len() - 1)].position;
                lanes.fill(f32::from(u16::try_from(sees)?));
            }
            return Ok(prompt);
        }
        // The small geometry's segments attend after the other geometries', so a small segment's
        // last block must not pad into the rows of a later segment in another geometry.
        let mut large = CHUNK;
        for (segment, bound) in bounds.windows(2).enumerate().rev() {
            let rows = &chunk[bound[0]..bound[1]];
            let (first, last) = (rows[0], rows[rows.len() - 1].position);
            if rows.len() > SEGMENT_ROWS {
                return Err(refused().into());
            }
            if last >= SPANS * KEYS || first.base + last / PAGE_TOKENS >= pages.len() {
                return Err("chunk rows are not sequences' positions inside the plan".into());
            }
            // Each block attends the spans up to its last row's.
            let cost = |geometry: (usize, usize, usize)| {
                let (keys, queries, _) = geometry;
                let (first, later, span) = pair_costs(geometry);
                let pairs: usize = (0..rows.len().div_ceil(queries))
                    .map(|block| {
                        let end = (rows[0].position + (block + 1) * queries - 1).min(last);
                        first + later * (end / keys)
                    })
                    .sum();
                pairs + span * (last / keys + 1)
            };
            // Attention reads and writes whole query blocks, so the last block's padding rows must
            // fit the chunk's rows and the capacity rows too; a span must lie inside the reach.
            let fits = |(keys, queries, reach): (usize, usize, usize)| {
                let end = bound[0] + rows.len().div_ceil(queries) * queries;
                end <= CHUNK && start + end <= Capacity::SIZE && last < reach * keys
            };
            let pads = bound[0] + rows.len().div_ceil(SMALL.1) * SMALL.1 > large;
            let geometry = [NARROW, WIDE, SMALL]
                .into_iter()
                .filter(|&geometry| fits(geometry) && !(pads && geometry == SMALL))
                .min_by_key(|&geometry| cost(geometry))
                .ok_or_else(refused)?;
            if geometry != SMALL {
                large = bound[0];
            }
            // A wide segment whose rows see at most the first 1,024 keys of its last span after
            // the first attends that span in the half geometry.
            let half = attention::prompt::half::Keys::SIZE;
            prompt.half[segment] =
                i32::from(geometry == WIDE && last >= WIDE.0 && last % WIDE.0 < half);
            let spans = if geometry == WIDE {
                &mut prompt.wide
            } else if geometry == SMALL {
                &mut prompt.small
            } else {
                &mut prompt.narrow
            };
            spans.fill((segment, rows), pages)?;
        }
        prompt.narrow.limit(chunk)?;
        prompt.wide.limit(chunk)?;
        prompt.small.limit(chunk)?;
        prompt.routed = ROUTE.sharded(start, chunk.len());
        Ok(prompt)
    }

    /// The live rows the step's TP2 body runs: the job-plan rows before a routed chunk (not the
    /// padding up to its wave, which the TP2 body would attend and exchange), else all `rows`.
    pub fn live(&self, rows: usize) -> usize {
        if self.routed { self.decode } else { rows }
    }

    /// The D-sharded route's chunk counts, both 0 when the TP2 body runs the chunk.
    pub async fn upload_routed(&self, device: &mut Device) -> Result<step::Routed> {
        let (blocks, segments) = if self.routed {
            (self.blocks, offset(self.bounds.len() - 1)?)
        } else {
            (0, 0)
        };
        Ok(step::Routed {
            blocks: copies(device, &[blocks]).await?,
            segments: copies(device, &[segments]).await?,
        })
    }

    pub async fn upload_short(&self, device: &mut Device) -> Result<attention::short::Plan> {
        let positions: Vec<f32> = (0..SHORT_KEYS)
            .map(|key| Ok(f32::from(u16::try_from(key)?)))
            .collect::<Result<_>>()?;
        Ok(attention::short::Plan {
            passes: copies(device, &[self.passes]).await?,
            pages: copies(device, &self.short_pages).await?,
            limits: copies(device, &self.short_limits).await?,
            positions: copies(device, &positions).await?,
        })
    }

    pub async fn upload(&self, device: &mut Device) -> Result<attention::prompt::Plan> {
        let per_segment = |values: &[i32]| {
            let mut table = values.to_vec();
            table.resize(SEGMENTS, 0);
            table
        };
        let segments = self.bounds.len() - 1;
        let (narrow, wide, small) = (
            self.narrow.counts(),
            self.wide.counts(),
            self.small.counts(),
        );
        // A routed chunk is the D-sharded route's: the TP2 plan reads none.
        let (blocks, counted) = if self.routed {
            (0, 0)
        } else {
            (self.blocks, segments)
        };
        Ok(attention::prompt::Plan {
            blocks: copies(device, &[blocks]).await?,
            start: copies(device, &[self.start]).await?,
            segments: copies(device, &[offset(counted)?]).await?,
            first: copies(device, &per_segment(&self.bounds[..segments])).await?,
            end: copies(device, &per_segment(&self.bounds[1..])).await?,
            narrow: attention::prompt::narrow::Tables {
                spans: copies(device, &narrow.0).await?,
                full: copies(device, &narrow.1).await?,
                ahead: copies(device, &narrow.2).await?,
                pages: slice(device, &self.narrow.pages).await?,
                limits: copies(device, &self.narrow.limits).await?,
                positions: copies(device, &self.narrow.positions()?).await?,
            },
            wide: attention::prompt::wide::Tables {
                spans: copies(device, &wide.0).await?,
                full: copies(device, &wide.1).await?,
                ahead: copies(device, &wide.2).await?,
                pages: slice(device, &self.wide.pages).await?,
                limits: copies(device, &self.wide.limits).await?,
                positions: copies(device, &self.wide.positions()?).await?,
            },
            small: attention::prompt::small::Tables {
                spans: copies(device, &small.0).await?,
                full: copies(device, &small.1).await?,
                ahead: copies(device, &small.2).await?,
                pages: slice(device, &self.small.pages).await?,
                limits: copies(device, &self.small.limits).await?,
                positions: copies(device, &self.small.positions()?).await?,
            },
            half: copies(device, &per_segment(&self.half)).await?,
        })
    }
}

/// The first row of a chunk that follows `decode` rows: the next whole wave.
pub fn chunk_start(decode: usize) -> usize {
    decode.div_ceil(BATCH) * BATCH
}

/// Per-row RoPE coefficients in the rotation owner's `[Rows, OutputHalf, InputHalf, Half]` order,
/// one 64-row wave after another.
pub fn coefficients(rows: &[Row]) -> Result<Vec<bf16>> {
    let mut values = Vec::with_capacity(Capacity::SIZE * rotation::Stored::SIZE / 64);
    for row in 0..Capacity::SIZE {
        let position = rows.get(row).map_or(0, |live| live.position);
        let position = f32::from(u16::try_from(position)?);
        for (output, input) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            for feature in 0..64_u16 {
                let angle = position / rotation::THETA.powf(f32::from(2 * feature) / 128.0);
                values.push(bf16::from_f32(match (output, input) {
                    (0, 1) => -angle.sin(),
                    (1, 0) => angle.sin(),
                    _ => angle.cos(),
                }));
            }
        }
    }
    Ok(values)
}

/// K and V cache-write offsets per capacity row over a layer's `pages`; rows past `rows` are
/// padding. A live row must write below the scratch pages, which every step overwrites.
pub fn writes(rows: &[Row], pages: &Pages) -> Result<(Vec<i32>, Vec<i32>)> {
    let keys = (0..Capacity::SIZE)
        .map(|slot| {
            let padding = Row::padding(pages, slot);
            let row = rows.get(slot).copied().unwrap_or(padding);
            if row != padding && row.base + row.position / PAGE_TOKENS >= pages.scratch {
                return Err(format!("row {slot} writes into the scratch pages").into());
            }
            offset(row.record(pages, row.position))
        })
        .collect::<Result<Vec<_>>>()?;
    let half = offset(HALF_RECORD)?;
    let values = keys.iter().map(|key| key + half).collect();
    Ok((keys, values))
}

/// One draw: (inverse temperature, top-k, top-p, ban, uniform).
pub type Draw = (f32, i32, f32, f32, f32);
/// The greedy draw.
pub const GREEDY: Draw = (1.0, 1, f32::INFINITY, 0.0, 0.0);

/// A plan that draws the first `rows` selected rows, row `at` by `draws(at)`; every other row
/// greedy.
pub async fn sampling(
    device: &mut Device,
    (rows, draws): (usize, &dyn Fn(usize) -> Draw),
) -> Result<step::sample::Plan> {
    let table: Vec<Draw> = (0..step::Selected::SIZE)
        .map(|at| if at < rows { draws(at) } else { GREEDY })
        .collect();
    let field = |pick: &dyn Fn(&Draw) -> f32| {
        table
            .iter()
            .flat_map(|draw| [pick(draw), 0.0, 0.0, 0.0])
            .collect::<Vec<f32>>()
    };
    let top_k: Vec<i32> = table.iter().flat_map(|draw| [draw.1, 0, 0, 0]).collect();
    Ok(step::sample::Plan {
        rows: copies(device, &[i32::try_from(rows)?]).await?,
        inverse: copies(device, &field(&|draw| draw.0)).await?,
        top_k: copies(device, &top_k).await?,
        top_p: copies(device, &field(&|draw| draw.2)).await?,
        ban: copies(device, &field(&|draw| draw.3)).await?,
        uniform: copies(device, &field(&|draw| draw.4)).await?,
    })
}

/// A plan that samples no row: the readout's tokens are its greedy tokens.
pub async fn greedy(device: &mut Device) -> Result<step::sample::Plan> {
    sampling(device, (0, &|_| GREEDY)).await
}

/// `values`, chip-major, on both chips.
pub async fn upload<D: MaterializableScalar, E: M>(
    device: &mut Device,
    values: Vec<D>,
) -> Result<HbmTensor<D, Chips, E>> {
    Ok(HostTensor::<D, m![{ Chips }, { E }]>::from_vec(values)
        .to_hbm(&mut device.pdma)
        .await?)
}

/// `values`, repeated to fill `E`, on each chip.
pub async fn copies<D: MaterializableScalar + Copy, E: M>(
    device: &mut Device,
    values: &[D],
) -> Result<HbmTensor<D, Chips, E>> {
    upload(
        device,
        values.iter().copied().cycle().take(2 * E::SIZE).collect(),
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(lengths: &[usize]) -> Vec<Row> {
        let mut base = 0;
        lengths
            .iter()
            .map(|&length| {
                let row = Row {
                    token: 0,
                    base,
                    position: length - 1,
                };
                base += length.div_ceil(PAGE_TOKENS);
                row
            })
            .collect()
    }

    /// Slot `staged` of burst `burst`'s presence in landing order: entry `[t][page]` is 1 when the
    /// slot's counts reach the page for its token `t`.
    fn present(tables: &Tables, burst: usize, staged: usize) -> Vec<i32> {
        (0..BLOCK)
            .map(|at| {
                let count = tables.counts[burst][(at / JOB_PAGES * BURST * JOBS + staged) * LANES];
                i32::from(((at % JOB_PAGES) as f32) < count)
            })
            .collect()
    }

    /// A batch's job waves land every history token once, in the order the kernel deals a page's
    /// tokens over a job's slices, read only pages of the row's own history, stage a burst's
    /// tables at its entry, and count each slot's present tokens as its length.
    #[test]
    fn lands_every_history_token_once() {
        let long = [
            600, 17, 1, 513, 40_000, 40_000, 40_000, 40_000, 40_000, 40_000, 40_000,
        ];
        for (lengths, block, waves) in [
            (&long[..], BLOCK, 9),
            (&[7000; 55][..], BLOCK, 13),
            (&[7000, 1][..], 224, 1),
            (&[1024; 4][..], 64, 1),
            (&long[..4], 32, 1),
            (&[1024, 1024, 1024, 1024, 3000][..], 128, 1),
            (&[600, 17, 1, 513, 3000][..], 96, 1),
        ] {
            let rows = rows(lengths);
            assert_eq!(super::block(&rows), block);
            // A batch that runs on fixed slots or whole-row waves has no job waves to check here.
            let used = rows
                .iter()
                .map(|row| row.base + row.position / PAGE_TOKENS + 1)
                .max();
            if Tables::new(&rows, &Pages::contiguous(used.unwrap()), 0)
                .unwrap()
                .direct
                == vec![vec![0]]
            {
                lands(lengths, waves);
            }
        }
    }

    fn lands(lengths: &[usize], waves: i32) {
        let rows = rows(lengths);
        let pages = rows
            .iter()
            .map(|row| row.base + row.position / PAGE_TOKENS)
            .max()
            .unwrap()
            + 1;
        let layer = Pages::contiguous(pages);
        let tables = Tables::new(&rows, &layer, 0).unwrap();
        assert_eq!(tables.waves, vec![vec![waves]]);
        let mut reached = vec![Vec::new(); rows.len()];
        let (jobs, _, order) = jobs(0, &rows, &layer).unwrap();
        // Job `order[v]` takes position `v` of the batch's slots.
        let mut rank = vec![0; jobs.len()];
        for (at, &job) in order.iter().enumerate() {
            rank[job] = at;
        }
        let slot_rows: Vec<(usize, usize)> = jobs
            .iter()
            .map(|job| (job.query as usize / QUERY_BYTES, job.tokens.start))
            .collect();
        for (job, &(row, first)) in slot_rows.iter().enumerate() {
            let slot = slot_of(rank[job], slot_rows.len());
            let (wave, burst, staged) = (slot / JOBS, slot / (BURST * JOBS), slot % (BURST * JOBS));
            let pages = &tables.pages[wave][slot % JOBS * JOB_PAGES..][..JOB_PAGES];
            let present = present(&tables, burst, staged);
            assert_eq!(
                tables.queries[burst][staged],
                offset(row * QUERY_BYTES).unwrap()
            );
            for (at, &present) in present.iter().enumerate() {
                let token = landed(at);
                if present == 1 {
                    let position = first + token;
                    assert_eq!(
                        pages[token / PAGE_TOKENS],
                        offset(rows[row].record(&layer, position - position % PAGE_TOKENS))
                            .unwrap()
                    );
                    reached[row].push(position);
                }
            }
        }
        for (row, positions) in reached.iter_mut().enumerate() {
            positions.sort_unstable();
            assert_eq!(*positions, (0..=rows[row].position).collect::<Vec<_>>());
        }
        for slot in 0..tables.counts.len() * BURST * JOBS {
            let present = present(&tables, slot / (BURST * JOBS), slot % (BURST * JOBS));
            assert!(
                job_at(slot, slot_rows.len()) < slot_rows.len()
                    || present.iter().all(|&at| at == 0)
            );
            let length = tables.lengths[slot / JOBS][slot % JOBS];
            assert_eq!(length, present.iter().sum::<i32>(), "slot {slot}");
        }
    }

    /// A batch of one row of at most `SINGLE_KEYS` keys has no job waves and no fold passes, its
    /// jobs fit the pages its wave computes, and its first wave lands every history key once over
    /// all 64 slots; a longer row takes waves.
    #[test]
    fn single_rows_land_every_key_once() {
        for length in [attention::FEW_KEYS + 1, 7000, 8000, SINGLE_KEYS] {
            let rows = rows(&[length]);
            let tables = Tables::new(&rows, &Pages::contiguous(3000), 0).unwrap();
            assert_eq!(tables.waves, vec![vec![0]]);
            assert_eq!(tables.folds, vec![vec![0]]);
            let jobs = single_jobs(0, &rows[0]).unwrap();
            assert!(jobs.len() <= JOBS && (length <= 16 * JOBS || jobs.len() > JOBS / 2));
            assert!(
                jobs.iter()
                    .all(|job| job.tokens.len() <= attention::SINGLE_PAGES * PAGE_TOKENS)
            );
            let mut reached = Vec::new();
            for (job, at) in jobs.iter().enumerate() {
                let slot = slot_of(job, jobs.len());
                assert!(slot < JOBS);
                let pages = &tables.pages[0][slot * JOB_PAGES..][..JOB_PAGES];
                let present = present(&tables, 0, slot);
                assert_eq!(tables.queries[0][slot], 0);
                assert_eq!(tables.lengths[0][slot], present.iter().sum::<i32>());
                for (cell, &present) in present.iter().enumerate() {
                    let token = landed(cell);
                    if present == 1 {
                        let position = at.tokens.start + token;
                        assert_eq!(
                            pages[token / PAGE_TOKENS],
                            offset(rows[0].record(
                                &Pages::contiguous(3000),
                                position - position % PAGE_TOKENS
                            ))
                            .unwrap()
                        );
                        reached.push(position);
                    }
                }
            }
            reached.sort_unstable();
            assert_eq!(reached, (0..length).collect::<Vec<_>>(), "length {length}");
        }
        let tables = Tables::new(&rows(&[SINGLE_KEYS + 1]), &Pages::contiguous(3000), 0).unwrap();
        assert!(tables.waves[0][0] > 0);
    }

    /// No wave's gather reads one page more than twice, idle slots of a partial first wave
    /// included, and a one-row batch's burst reads no query more often than the row has jobs.
    #[test]
    fn spreads_gathers_over_distinct_sources() {
        let reads = |table: &[i32]| {
            table.iter().fold(
                std::collections::HashMap::<i32, usize>::new(),
                |mut reads, &at| {
                    *reads.entry(at).or_default() += 1;
                    reads
                },
            )
        };
        let rows = rows(&[7000; 55]);
        let pages = rows[54].base + rows[54].position / PAGE_TOKENS + 1;
        let tables = Tables::new(&rows, &Pages::contiguous(pages), 0).unwrap();
        assert_eq!(tables.waves, vec![vec![13]]);
        // The first wave's gather reads a PE's slots to its longest job, the others whole jobs.
        let read = |wave: usize| -> Vec<i32> {
            let pages = tables.pages[wave].chunks(PE_JOBS * JOB_PAGES);
            pages
                .zip(tables.lengths[wave].chunks(PE_JOBS))
                .flat_map(|(pages, lengths)| {
                    let longest = *lengths.iter().max().unwrap() as usize;
                    let taken = if wave == 0 {
                        longest.div_ceil(PAGE_TOKENS)
                    } else {
                        JOB_PAGES
                    };
                    pages
                        .chunks(JOB_PAGES)
                        .flat_map(move |slot| slot[..taken].to_vec())
                })
                .collect()
        };
        for wave in 0..13 {
            assert!(
                reads(&read(wave)).values().all(|&count| count <= 2),
                "wave {wave}"
            );
        }
        let tables = Tables::new(&rows[..1], &Pages::contiguous(pages), 0).unwrap();
        let jobs = single_jobs(0, &rows[0]).unwrap().len();
        assert!(
            reads(&tables.queries[0])
                .values()
                .all(|&count| count <= jobs)
        );
    }

    /// A batch whose rows' blocks take no more waves on fixed slots runs on them: row `r`'s slot
    /// `dealt(r)` (ahead of a prompt chunk, a batch of at most 32 rows that takes half the waves
    /// on two slots a row, also slot `dealt(r + 32)`) lands every token of its history once, its
    /// last block in wave 0, reads its row's query and counts its present tokens as its length; an
    /// idle cell reads only pages no row holds. Batches that would take more waves keep the job
    /// plan, and so do the paired cases without a chunk.
    #[test]
    fn fixed_slots_land_each_row_in_its_own_slot() {
        let mixed = |rows: usize| -> Vec<usize> {
            (0..rows)
                .map(|row| if row % 3 == 0 { 6600 } else { 7000 })
                .collect()
        };
        for (lengths, chunk, direct, waves) in [
            (vec![7000; 64], 0, 1, 14),
            (mixed(64), 0, 1, 14),
            (vec![200; 10], 0, 1, 1),
            (vec![7000; 55], 0, 0, 13),
            (vec![7000; 32], 1024, 2, 7),
            (mixed(32), 1024, 2, 7),
            (vec![1000; 5], 1024, 2, 1),
            (vec![7000; 32], 0, 0, 7),
            (vec![1000; 20], 0, 0, 1),
            (vec![1536; 5], 0, 0, 1),
            (vec![1536; 24], 0, 3, 2),
            (vec![7000; 33], 1024, 0, 8),
        ] {
            let rows = rows(&lengths);
            let used = rows
                .iter()
                .map(|row| row.base + row.position / PAGE_TOKENS + 1)
                .max()
                .unwrap();
            let tables = Tables::new(&rows, &Pages::contiguous(used + 64), chunk).unwrap();
            // The first batch's few-row path takes at most `FEW_ROWS` rows of `FEW_KEYS` keys.
            if lengths.len() <= attention::FEW_ROWS
                && lengths.iter().all(|&length| length <= attention::FEW_KEYS)
            {
                assert_eq!(tables.direct, vec![vec![5]], "{} rows", lengths.len());
                continue;
            }
            // A batch that would take more waves on fixed slots merges its jobs, on job waves or
            // on runs.
            if direct == 0 {
                let merged = tables.direct == vec![vec![0]] || tables.direct == vec![vec![4]];
                assert!(merged, "{} rows: {:?}", lengths.len(), tables.direct);
                continue;
            }
            assert_eq!(tables.direct, vec![vec![direct]], "{} rows", lengths.len());
            assert_eq!(tables.waves, vec![vec![waves]], "{} rows", lengths.len());
            // Kind 3 (whole-row waves) lands rows in `whole_row_waves_land_each_row_in_its_slots`.
            if direct == 0 || direct == 3 {
                continue;
            }
            assert_eq!(tables.folds, vec![vec![0]]);
            assert!(tables.merge.visible[0].iter().all(|&at| at == 0.0));
            let mut reached = vec![Vec::new(); rows.len()];
            for slot in 0..JOBS {
                let row = dealt(slot) % (JOBS / direct as usize);
                for wave in 0..waves as usize {
                    let staged = wave % BURST * JOBS + slot;
                    let pages = &tables.pages[wave][slot * JOB_PAGES..][..JOB_PAGES];
                    let present = present(&tables, wave / BURST, staged);
                    let length = tables.lengths[wave][slot];
                    assert_eq!(
                        length,
                        present.iter().sum::<i32>(),
                        "slot {slot} wave {wave}"
                    );
                    let Some(live) = rows.get(row).filter(|_| length > 0) else {
                        assert!(pages.iter().all(|&page| page as usize / PAGE_BYTES >= used));
                        continue;
                    };
                    assert_eq!(
                        tables.queries[wave / BURST][staged],
                        offset(row * QUERY_BYTES).unwrap()
                    );
                    for (at, &present) in present.iter().enumerate() {
                        if present == 1 {
                            let page = pages[landed(at) / PAGE_TOKENS] as usize / PAGE_BYTES;
                            reached[row]
                                .push((page - live.base) * PAGE_TOKENS + landed(at) % PAGE_TOKENS);
                        }
                    }
                }
            }
            for (row, positions) in reached.iter_mut().enumerate() {
                positions.sort_unstable();
                assert_eq!(
                    *positions,
                    (0..=rows[row].position).collect::<Vec<_>>(),
                    "row {row}"
                );
            }
        }
    }

    /// A batch of rows of at most `ROW_KEYS` keys that takes no more waves of whole rows than job
    /// waves, at least half of one, and no partial one but its first, runs on them: the slots
    /// `whole_row` gives a row land every token of its history once, read its query and count
    /// their present tokens as their lengths; an idle cell reads only pages no row holds.
    #[test]
    fn whole_row_waves_land_each_row_in_its_slots() {
        for (lengths, waves) in [
            (vec![1536; 16], 1),
            (vec![1536; 9], 1),
            (vec![1536; 8], 1),
            (vec![ROW_KEYS; 32], 2),
            (vec![1536; 24], 2),
            (vec![600, 17, 1, 513, 600, 17, 1, 513], 1),
        ] {
            let swap = usize::from(lengths.len() % ROW_WAVE != 0 && lengths.len() > ROW_WAVE);
            let rows = rows(&lengths);
            let used = rows
                .iter()
                .map(|row| row.base + row.position / PAGE_TOKENS + 1)
                .max()
                .unwrap();
            let tables = Tables::new(&rows, &Pages::contiguous(used + 64), 0).unwrap();
            // The first batch's few-row path takes at most `FEW_ROWS` rows of `FEW_KEYS` keys.
            if lengths.len() <= attention::FEW_ROWS
                && lengths.iter().all(|&length| length <= attention::FEW_KEYS)
            {
                assert_eq!(tables.direct, vec![vec![5]], "{} rows", lengths.len());
                continue;
            }
            assert_eq!(tables.direct, vec![vec![3]], "{} rows", lengths.len());
            assert_eq!(tables.waves, vec![vec![waves]], "{} rows", lengths.len());
            assert_eq!(tables.folds, vec![vec![0]]);
            assert!(tables.merge.visible[0].iter().all(|&at| at == 0.0));
            let mut reached = vec![Vec::new(); rows.len()];
            for wave in 0..waves as usize {
                for slot in 0..JOBS {
                    let row = whole_row(wave ^ swap, slot);
                    let staged = wave % BURST * JOBS + slot;
                    let pages = &tables.pages[wave][slot * JOB_PAGES..][..JOB_PAGES];
                    let present = present(&tables, wave / BURST, staged);
                    let length = tables.lengths[wave][slot];
                    assert_eq!(
                        length,
                        present.iter().sum::<i32>(),
                        "slot {slot} wave {wave}"
                    );
                    let Some(live) = rows.get(row).filter(|_| length > 0) else {
                        assert!(pages.iter().all(|&page| page as usize / PAGE_BYTES >= used));
                        continue;
                    };
                    assert_eq!(
                        tables.queries[wave / BURST][staged],
                        offset(row * QUERY_BYTES).unwrap()
                    );
                    for (at, &present) in present.iter().enumerate() {
                        if present == 1 {
                            let page = pages[landed(at) / PAGE_TOKENS] as usize / PAGE_BYTES;
                            reached[row]
                                .push((page - live.base) * PAGE_TOKENS + landed(at) % PAGE_TOKENS);
                        }
                    }
                }
            }
            for (row, positions) in reached.iter_mut().enumerate() {
                positions.sort_unstable();
                assert_eq!(
                    *positions,
                    (0..=rows[row].position).collect::<Vec<_>>(),
                    "row {row}"
                );
            }
        }
    }

    /// A batch's plan word gives the kernel back its job waves, fold passes and slot kind.
    #[test]
    fn batch_words_hold_waves_folds_and_kind() {
        for waves in 0..=attention::Blocks::SIZE {
            for (folds, direct, entries) in [
                (0, 0, 0),
                (1, 0, 0),
                (2, 0, 0),
                (0, 1, 0),
                (0, 2, 0),
                (0, 3, 0),
                (0, 5, 24),
            ] {
                let word = attention::word(waves, folds, direct, entries);
                assert_eq!(
                    attention::shape(word),
                    (waves, folds, direct, entries),
                    "word {word}"
                );
            }
        }
    }

    /// A fixed batch's waves after the first give a slot time's PEs different blocks of their rows
    /// (as many as a row's slot has), so rows a large power of two apart still spread the reads.
    #[test]
    fn fixed_waves_spread_a_slot_time_over_blocks() {
        for (length, chunk, halves) in [(8704, 0, 1), (1536, 0, 1), (7000, 1024, 2)] {
            let rows = rows(&vec![length; JOBS / halves]);
            let used = rows
                .last()
                .map(|row| row.base + row.position / PAGE_TOKENS + 1)
                .unwrap();
            let tables = Tables::new(&rows, &Pages::contiguous(used + 64), chunk).unwrap();
            assert_eq!(tables.direct, vec![vec![halves as i32]]);
            let blocks = length.div_ceil(BLOCK);
            let first = if halves == 2 { blocks.div_ceil(2) } else { 0 };
            for wave in 1..tables.waves[0][0] as usize {
                for time in 0..PE_JOBS {
                    let full = match (halves, time < PE_JOBS / 2) {
                        (2, true) => first,
                        (2, false) => blocks - first - 1,
                        _ => blocks - 1,
                    };
                    let read: std::collections::HashSet<usize> = (0..PE_JOBS)
                        .map(|pe| {
                            let slot = pe * PE_JOBS + time;
                            let row = &rows[dealt(slot) % (JOBS / halves)];
                            let page = tables.pages[wave][slot * JOB_PAGES] as usize / PAGE_BYTES;
                            (page - row.base) / JOB_PAGES
                        })
                        .collect();
                    assert_eq!(
                        read.len(),
                        full.min(PE_JOBS),
                        "{length} wave {wave} time {time}"
                    );
                }
            }
        }
    }

    /// Every job's partial reaches its row's final merge pass exactly once, directly or through
    /// one fold record, and idle entries read only records the batch's waves write.
    #[test]
    fn merges_every_job_once() {
        let rows = rows(&[MAX_CONTEXT, 16385, 515, 1]);
        let tables = Tables::new(&rows, &Pages::contiguous(4000), 0).unwrap();
        assert_eq!(tables.folds, vec![vec![2]]);
        let records = |table: &[i32]| -> Vec<usize> {
            table
                .iter()
                .map(|&at| at as usize / STATISTIC_BYTES)
                .collect()
        };
        let finals = records(&tables.merge.statistics[0]);
        let folds: Vec<Vec<usize>> = tables
            .fold
            .statistics
            .iter()
            .map(|table| records(table))
            .collect();
        let (jobs, spans, order) = jobs(0, &rows, &Pages::contiguous(4000)).unwrap();
        let jobs = jobs.len();
        let mut rank = vec![0; jobs];
        for (at, &job) in order.iter().enumerate() {
            rank[job] = at;
        }
        let record = |job: usize| BURST * JOBS + slot_of(rank[job], jobs);
        let stored: std::collections::HashSet<usize> = (0..jobs).map(record).collect();
        let written = BURST * JOBS..BURST * JOBS + jobs.div_ceil(JOBS) * JOBS;
        for (row, span) in spans.iter().enumerate() {
            let entries = row * ENTRIES..(row + 1) * ENTRIES;
            let reached: Vec<usize> = entries
                .clone()
                .filter(|&entry| tables.merge.visible[0][entry] > 0.0)
                .flat_map(|entry| match finals[entry] {
                    record if stored.contains(&record) => vec![record],
                    record => {
                        assert_eq!(record % JOBS, row);
                        let fold = record / JOBS;
                        entries
                            .clone()
                            .filter(|&at| tables.fold.visible[fold][at] > 0.0)
                            .map(|at| folds[fold][at])
                            .collect()
                    }
                })
                .collect();
            assert_eq!(
                reached,
                span.clone().map(record).collect::<Vec<_>>(),
                "row {row}"
            );
        }
        for (table, visible) in folds
            .iter()
            .chain([&finals])
            .zip(tables.fold.visible.iter().chain(&tables.merge.visible))
        {
            let mut idle = table
                .iter()
                .zip(visible)
                .filter(|&(_, &visible)| visible == 0.0);
            assert!(idle.all(|(record, _)| written.contains(record)));
        }
    }

    /// A chunk row sees exactly its own position and every earlier one of its sequence, span by
    /// span, a row past the chunk sees as the chunk's last row, each span's gather reads that
    /// sequence's written pages and then distinct pages, and no chunk attends nothing.
    #[test]
    fn chunk_rows_see_their_causal_prefix() {
        let first = Row {
            token: 0,
            base: 3,
            position: KEYS - 100,
        };
        let chunk: Vec<Row> = (0..300)
            .map(|at| Row {
                position: first.position + at,
                ..first
            })
            .collect();
        let padding = Row {
            token: 0,
            base: 63,
            position: 0,
        };
        let rows: Vec<Row> = rows(&[9, 40])
            .into_iter()
            .chain(std::iter::repeat_n(padding, 62))
            .chain(chunk)
            .collect();
        assert!(Prompt::new(&rows, 2, &Pages::contiguous(1200)).is_err());
        assert!(Prompt::new(&rows, chunk_start(2), &Pages::contiguous(64)).is_err());
        let tables = Prompt::new(&rows, chunk_start(2), &Pages::contiguous(1200)).unwrap();
        assert_eq!(
            (
                tables.blocks,
                tables.start,
                tables.narrow.spans[0],
                tables.narrow.full[0],
                tables.wide.spans[0]
            ),
            (2, 64, 2, 0, 0)
        );
        let last = KEYS + 199;
        for (span, table) in tables.narrow.pages.iter().enumerate() {
            let pages: Vec<usize> = table
                .iter()
                .map(|&page| page as usize / PAGE_BYTES)
                .collect();
            let written = (last + 1).saturating_sub(span * KEYS).div_ceil(PAGE_TOKENS);
            let first = 3 + span * SPAN_PAGES;
            assert_eq!(
                pages[..written.min(SPAN_PAGES)],
                (first..first + written.min(SPAN_PAGES)).collect::<Vec<_>>()[..],
                "span {span}"
            );
            assert_eq!(
                pages.iter().collect::<std::collections::HashSet<_>>().len(),
                pages.len()
            );
        }
        for (at, lanes) in tables
            .narrow
            .limits
            .chunks_exact(attention::Lane::SIZE)
            .enumerate()
        {
            let (span, row) = (at / CHUNK, at % CHUNK);
            let sees = KEYS - 100 + row.min(299);
            let expected = (sees as f32 - (span * KEYS) as f32).clamp(-1.0, (KEYS - 1) as f32);
            assert!(
                lanes.iter().all(|&lane| lane == expected),
                "span {span} row {row}"
            );
        }
        let none = Prompt::new(&rows[..2], chunk_start(2), &Pages::contiguous(64)).unwrap();
        assert_eq!(
            (
                none.blocks,
                none.narrow.spans[0],
                none.wide.spans[0],
                none.bounds.len()
            ),
            (0, 0, 0, 1)
        );
    }

    /// A chunk whose rows see at most a short tile's keys attends no span: each row's limit is its
    /// own position over the sequence's first pages, in passes of 1,024 rows.
    #[test]
    fn short_chunks_attend_their_position_in_passes() {
        let chunk: Vec<Row> = (0..1100)
            .map(|position| Row {
                token: 0,
                base: 5,
                position: 100 + position,
            })
            .collect();
        let tables = Prompt::with_short(&chunk, 0, &Pages::contiguous(1200), true).unwrap();
        assert_eq!(
            (
                tables.blocks,
                tables.narrow.spans[0],
                tables.wide.spans[0],
                tables.passes
            ),
            (5, 0, 0, 2)
        );
        let pages: Vec<usize> = tables
            .short_pages
            .iter()
            .map(|&page| page as usize / PAGE_BYTES)
            .collect();
        assert_eq!(pages, (5..5 + SHORT_PAGES).collect::<Vec<_>>());
        for (row, lanes) in tables
            .short_limits
            .chunks_exact(attention::Lane::SIZE)
            .enumerate()
        {
            let expected = (100 + row.min(1099)) as f32;
            assert!(lanes.iter().all(|&lane| lane == expected), "row {row}");
        }
        let long = Prompt::with_short(&chunk[..1000], 0, &Pages::contiguous(1200), true).unwrap();
        assert_eq!(long.passes, 1);
        let past: Vec<Row> = (0..300)
            .map(|position| Row {
                position: SHORT_KEYS - 100 + position,
                ..chunk[0]
            })
            .collect();
        let spanned = Prompt::with_short(&past, 0, &Pages::contiguous(1200), true).unwrap();
        assert_eq!(spanned.passes, 0);
        assert!(spanned.narrow.spans[0] + spanned.wide.spans[0] + spanned.small.spans[0] > 0);
        let spans = Prompt::with_short(&chunk, 0, &Pages::contiguous(1200), false).unwrap();
        assert_eq!(spans.passes, 0);
        assert!(spans.narrow.spans[0] + spans.wide.spans[0] + spans.small.spans[0] > 0);
    }

    /// A chunk of several sequences attends each as its own segment: a segment's rows see their
    /// own positions, its spans read its own sequence's pages, each segment takes its cheaper
    /// geometry among those whose blocks fit, a short path never takes such a chunk, and a
    /// segment past its blocks' state is refused.
    #[test]
    fn segments_attend_their_own_sequences() {
        let sequence = |base: usize, positions: std::ops::Range<usize>| {
            positions.map(move |position| Row {
                token: 0,
                base,
                position,
            })
        };
        // A long chunk ending past one span, then two cold prompts, the last one's wide and small
        // blocks ending past the capacity rows.
        let start = Capacity::SIZE - 2240;
        let chunk: Vec<Row> = sequence(3, KEYS - 100..KEYS + 156)
            .chain(sequence(700, 0..1024))
            .chain(sequence(800, 0..700))
            .collect();
        let rows: Vec<Row> = (0..start)
            .map(|slot| Row::padding(&Pages::contiguous(1200), slot))
            .chain(chunk.iter().copied())
            .collect();
        let tables = Prompt::with_short(&rows, start, &Pages::contiguous(1200), true).unwrap();
        assert_eq!(tables.bounds, vec![0, 256, 1280, 1980]);
        assert_eq!((tables.blocks, tables.passes), (8, 0));
        assert_eq!(tables.narrow.spans, vec![2, 0, 1]);
        assert_eq!(tables.wide.spans, vec![0, 0, 0]);
        assert_eq!(tables.small.spans, vec![0, 2, 0]);
        for (segment, base, table) in [
            (0, 3, &tables.narrow),
            (1, 700, &tables.small),
            (2, 800, &tables.narrow),
        ] {
            let first = table.pages[segment * table.geometry.2][0] as usize / PAGE_BYTES;
            assert_eq!(first, base, "segment {segment}");
        }
        for (at, lanes) in tables
            .small
            .limits
            .chunks_exact(attention::Lane::SIZE)
            .enumerate()
        {
            let (span, row) = (at / CHUNK, at % CHUNK);
            let sees = chunk[row.min(chunk.len() - 1)].position;
            let expected =
                (sees as f32 - (span * SMALL.0) as f32).clamp(-1.0, (SMALL.0 - 1) as f32);
            assert!(
                lanes.iter().all(|&lane| lane == expected),
                "span {span} row {row}"
            );
        }
        let four: Vec<Row> = (0..4).flat_map(|at| sequence(100 * at, 0..1024)).collect();
        let tables = Prompt::new(&four, 0, &Pages::contiguous(1200)).unwrap();
        assert_eq!(
            (tables.bounds, tables.small.spans),
            (vec![0, 1024, 2048, 3072, 4096], vec![2; 4])
        );
        let long: Vec<Row> = sequence(3, 0..SEGMENT_ROWS + 1).collect();
        assert!(Prompt::new(&long, 0, &Pages::contiguous(1200)).is_err());
    }

    /// A short history attends in 512-key spans, each block only up to its own rows' span: a fresh
    /// prompt's block `b` reaches span `b`, a chunk from position 300 one span further; a chunk
    /// whose rows reach past the small spans, or whose small pairs cost more, attends in a larger
    /// geometry.
    #[test]
    fn short_histories_attend_in_small_spans() {
        let chunk = |positions: std::ops::Range<usize>| -> Vec<Row> {
            positions
                .map(|position| Row {
                    token: 0,
                    base: 7,
                    position,
                })
                .collect()
        };
        for (rows, spans) in [(512, 1), (1024, 2)] {
            let tables = Prompt::new(&chunk(0..rows), 0, &Pages::contiguous(1200)).unwrap();
            let small = &tables.small;
            assert_eq!(
                (small.spans[0], small.full[0], small.ahead[0]),
                (spans, 0, 0),
                "{rows} rows"
            );
            assert_eq!(
                tables.narrow.spans[0] + tables.wide.spans[0],
                0,
                "{rows} rows"
            );
            for (span, table) in small.pages[..spans as usize].iter().enumerate() {
                let pages: Vec<usize> = table
                    .iter()
                    .map(|&page| page as usize / PAGE_BYTES)
                    .collect();
                let first = 7 + span * SMALL.0 / PAGE_TOKENS;
                assert_eq!(
                    pages,
                    (first..first + SMALL.0 / PAGE_TOKENS).collect::<Vec<_>>()
                );
            }
        }
        // A fresh 2,048-row prompt's ten small pairs cost less than its two wide ones.
        let fresh = Prompt::new(&chunk(0..2048), 0, &Pages::contiguous(1200)).unwrap();
        assert_eq!((fresh.small.spans[0], fresh.wide.spans[0]), (4, 0));
        let later = Prompt::new(&chunk(300..812), 0, &Pages::contiguous(1200)).unwrap();
        assert_eq!(
            (
                later.small.spans[0],
                later.small.full[0],
                later.small.ahead[0]
            ),
            (2, 0, 1)
        );
        let past = Prompt::new(&chunk(1800..2300), 0, &Pages::contiguous(1200)).unwrap();
        assert_eq!(past.small.spans[0], 0);
        assert!(past.narrow.spans[0] + past.wide.spans[0] > 0);
    }

    /// A short history whose last small block would pad into a later long history's rows attends
    /// in a larger geometry, since the small segments attend last; behind the long one, or ending
    /// on a block, it keeps the small spans.
    #[test]
    fn short_histories_padding_into_long_ones_leave_small_spans() {
        let segment = |base: usize, positions: std::ops::Range<usize>| {
            positions.map(move |position| Row {
                token: 0,
                base,
                position,
            })
        };
        let ahead: Vec<Row> = segment(7, 0..300).chain(segment(100, 6000..6600)).collect();
        let tables = Prompt::new(&ahead, 0, &Pages::contiguous(1200)).unwrap();
        assert_eq!(tables.small.spans[..2], [0, 0]);
        let behind: Vec<Row> = segment(100, 6000..6600).chain(segment(7, 0..300)).collect();
        let tables = Prompt::new(&behind, 0, &Pages::contiguous(1200)).unwrap();
        assert_eq!(tables.small.spans[..2], [0, 1]);
        let whole: Vec<Row> = segment(7, 0..512).chain(segment(100, 6000..6600)).collect();
        let tables = Prompt::new(&whole, 0, &Pages::contiguous(1200)).unwrap();
        assert_eq!(tables.small.spans[..2], [1, 0]);
    }

    /// A chunk whose last position lies past the plan's spans is refused.
    #[test]
    fn refuses_chunks_past_the_spans() {
        let chunk: Vec<Row> = (SPANS * KEYS - 10..SPANS * KEYS + 1)
            .map(|position| Row {
                token: 0,
                base: 0,
                position,
            })
            .collect();
        assert!(Prompt::new(&chunk, 0, &Pages::contiguous(4000)).is_err());
        assert!(Prompt::new(&chunk[..10], 0, &Pages::contiguous(4000)).is_ok());
    }

    /// A first batch of at most `FEW_ROWS` rows of at most `FEW_KEYS` keys takes no waves and
    /// carries its rows in its word; each row's present keys (page times below its count) number
    /// its length, and its pages are the row's.
    #[test]
    fn few_rows_count_every_key_once() {
        let cases: [&[usize]; 7] = [
            &[1],
            &[attention::FEW_KEYS],
            &[9, 1500],
            &[1, 2048, 600, 1536],
            &[17, 1, 1536, 2048, 300],
            &[1536; 7],
            &[1536; 8],
        ];
        for lengths in cases {
            let rows = rows(lengths);
            let tables = Tables::new(&rows, &Pages::contiguous(3000), 0).unwrap();
            assert_eq!(
                (tables.direct.clone(), tables.waves.clone()),
                (vec![vec![5]], vec![vec![0]])
            );
            assert_eq!(tables.entries, vec![vec![lengths.len() as i32]]);
            let (groups, times) = match rows.len() {
                1 => (8, 16),
                2..=4 => (2, 64),
                _ => (1, 128),
            };
            let beyond = if rows.len() == 1 {
                &tables.merge.visible[0]
            } else {
                &tables.counts[0]
            };
            for (r, row) in rows.iter().enumerate() {
                assert_eq!(tables.lengths[1][r], offset(lengths[r]).unwrap());
                for page in 0..lengths[r].div_ceil(PAGE_TOKENS) {
                    let at = tables.pages[1][attention::FEW_KEYS / PAGE_TOKENS * r + page];
                    assert_eq!(
                        at,
                        offset(row.record(&Pages::contiguous(3000), page * PAGE_TOKENS)).unwrap()
                    );
                }
                let cells = groups * PAGE_TOKENS * times;
                let present = beyond[r * cells..][..cells]
                    .iter()
                    .filter(|&&at| at < 0.0)
                    .count();
                assert_eq!(present, lengths[r], "{lengths:?}");
            }
        }
    }

    /// A history past the merge's reach, or past the layer's pages, is refused.
    #[test]
    fn refuses_histories_outside_the_plan() {
        assert!(Tables::new(&rows(&[MAX_CONTEXT + 1]), &Pages::contiguous(4000), 0).is_err());
        assert!(Tables::new(&rows(&[33]), &Pages::contiguous(2), 0).is_err());
    }

    /// Every capacity row writes its own cache record inside the layer, padding rows on the
    /// scratch pages only; a history reaching the scratch pages is refused.
    #[test]
    fn writes_each_record_once() {
        let pages = 60 + SCRATCH_PAGES;
        let chunk = (500..800).map(|position| Row {
            token: 0,
            base: 4,
            position,
        });
        let rows: Vec<Row> = rows(&[9, 40])
            .into_iter()
            .chain((2..chunk_start(2)).map(|slot| Row::padding(&Pages::contiguous(pages), slot)))
            .chain(chunk)
            .collect();
        let (keys, values) = writes(&rows, &Pages::contiguous(pages)).unwrap();
        let records: std::collections::HashSet<i32> = keys.iter().chain(&values).copied().collect();
        assert_eq!(records.len(), 2 * Capacity::SIZE);
        for (slot, &key) in keys.iter().enumerate() {
            let page = key as usize / PAGE_BYTES;
            let padding = !(0..2).contains(&slot) && !(64..364).contains(&slot);
            assert!(page < pages && (page >= 60) == padding, "row {slot}");
        }
        let late = Row {
            token: 0,
            base: 59,
            position: PAGE_TOKENS,
        };
        assert!(writes(&[late], &Pages::contiguous(pages)).is_err());
    }
}
