//! DMA queue, pinned host-slot leases and transfer completion tracking.

use std::collections::{BTreeMap, HashSet};
use std::io;
use std::ops::DerefMut;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak, mpsc};
use std::time::{Duration, Instant};

use furiosa_opt_std::prelude::{self, HostBuffer};

use crate::Shared;

use super::staging;

const COPY_THREADS: usize = 6;
const COPY_CHUNK_BLOCKS: usize = 2;

pub(super) struct Layer {
    pub(super) buffer: prelude::HbmBuffer,
    pub(super) page_bytes: usize,
    pub(super) host_offset: usize,
    pub(super) blocks: usize,
}

pub(super) struct Core {
    pub(super) context: Shared,
    pub(super) layers: Vec<Layer>,
    pub(super) groups: Vec<Vec<usize>>,
    pub(super) group_blocks: Vec<usize>,
    pub(super) chips: usize,
    pub(super) block_bytes: usize,
}

pub(super) struct Pool {
    pub(super) slots: Vec<Slot>,
    pub(super) block_bytes: usize,
    pub(super) core: Weak<Core>,
}

pub(super) struct Slot {
    pub(super) buffer: Mutex<HostBuffer>,
    pub(super) access: SlotAccess,
}

// Python buffer leases and DMA share one gate, so DMA cannot race a live host view.
#[derive(Default)]
pub(super) struct SlotAccess {
    state: Mutex<SlotAccessState>,
    changed: Condvar,
}

#[derive(Default)]
struct SlotAccessState {
    dma: bool,
    exported_views: usize,
    waiting_dma: usize,
}

struct DmaSlot<'a> {
    slot: &'a Slot,
    buffer: std::sync::MutexGuard<'a, HostBuffer>,
}

pub(super) struct Queue {
    core: Arc<Core>,
    sender: Option<mpsc::Sender<Chunk>>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

pub(super) struct Chunk {
    pages: Vec<TransferPage>,
    pool: Arc<Pool>,
    loading: bool,
    coalesced: bool,
    completion: Arc<ChunkCompletion>,
    job: Arc<Job>,
}

pub(super) struct TransferPage {
    pub(super) slot: usize,
    pub(super) routes: Vec<(usize, u32)>,
}

struct Job {
    left: AtomicUsize,
    submit_left: AtomicUsize,
    failure: Mutex<Option<io::Error>>,
    completion: Arc<Completion>,
}

pub(super) struct Completion {
    result: Mutex<Option<Result<(), String>>>,
    changed: Condvar,
    submission: Mutex<Option<Result<(), String>>>,
    submission_changed: Condvar,
}

pub(super) struct ChunkCompletion {
    result: Mutex<Option<Result<(), String>>>,
    changed: Condvar,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) enum TransferKey {
    Device(usize, u32),
    Host(usize),
}

#[derive(Default)]
// Each key retains only DMA chunks touching that page, not their whole framework job.
pub(super) struct Transfers(BTreeMap<TransferKey, Vec<Arc<ChunkCompletion>>>);

impl Slot {
    fn acquire_dma(&self) -> io::Result<DmaSlot<'_>> {
        self.access.acquire_dma()?;
        let buffer = match self.buffer.lock() {
            Ok(buffer) => buffer,
            Err(error) => {
                self.access.release_dma();
                return Err(io::Error::other(error.to_string()));
            }
        };
        Ok(DmaSlot { slot: self, buffer })
    }
}

impl SlotAccess {
    fn acquire_dma(&self) -> io::Result<()> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.waiting_dma = state
            .waiting_dma
            .checked_add(1)
            .ok_or_else(|| io::Error::other("too many pending host slot transfers"))?;
        while state.dma || state.exported_views != 0 {
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        state.waiting_dma -= 1;
        state.dma = true;
        Ok(())
    }

    fn release_dma(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.dma = false;
        self.changed.notify_all();
    }

    pub(super) fn acquire_view(&self) -> io::Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|error| io::Error::other(error.to_string()))?;
        while state.dma || state.waiting_dma != 0 {
            state = self
                .changed
                .wait(state)
                .map_err(|error| io::Error::other(error.to_string()))?;
        }
        state.exported_views = state
            .exported_views
            .checked_add(1)
            .ok_or_else(|| io::Error::other("too many active host slot views"))?;
        Ok(())
    }

    pub(super) fn try_acquire_view(&self) -> io::Result<bool> {
        let mut state = self
            .state
            .lock()
            .map_err(|error| io::Error::other(error.to_string()))?;
        if state.dma || state.waiting_dma != 0 {
            return Ok(false);
        }
        state.exported_views = state
            .exported_views
            .checked_add(1)
            .ok_or_else(|| io::Error::other("too many active host slot views"))?;
        Ok(true)
    }

    pub(super) fn release_view(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert!(state.exported_views > 0);
        if state.exported_views > 0 {
            state.exported_views -= 1;
        }
        self.changed.notify_all();
    }
}

impl std::ops::Deref for DmaSlot<'_> {
    type Target = HostBuffer;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl DerefMut for DmaSlot<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buffer
    }
}

impl Drop for DmaSlot<'_> {
    fn drop(&mut self) {
        // DMA is complete; the buffer mutex still serializes access until this guard drops.
        self.slot.access.release_dma();
    }
}

impl Queue {
    pub(super) fn start(core: Arc<Core>) -> io::Result<Self> {
        let (sender, receiver) = mpsc::channel::<Chunk>();
        let receiver = Arc::new(Mutex::new(receiver));
        let mut workers: Vec<std::thread::JoinHandle<()>> = Vec::with_capacity(COPY_THREADS);
        for _ in 0..COPY_THREADS {
            let executor = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(executor) => executor,
                Err(error) => {
                    drop(sender);
                    for worker in workers {
                        let _ = worker.join();
                    }
                    return Err(error);
                }
            };
            let (receiver, core) = (Arc::clone(&receiver), Arc::clone(&core));
            match std::thread::Builder::new()
                .name("kv-copy".into())
                .spawn(move || {
                    // A poisoned receiver means another copy thread panicked while holding it.
                    while let Some(chunk) = receiver.lock().ok().and_then(|next| next.recv().ok()) {
                        chunk.run(&executor, &core);
                    }
                }) {
                Ok(worker) => workers.push(worker),
                Err(error) => {
                    drop(sender);
                    for worker in workers {
                        let _ = worker.join();
                    }
                    return Err(error);
                }
            }
        }
        Ok(Self {
            core,
            sender: Some(sender),
            workers,
        })
    }

    pub(super) fn send(&self, chunk: Chunk) -> io::Result<()> {
        let Some(sender) = self.sender.as_ref() else {
            let error = io::Error::other("the cache copy queue is shut down");
            chunk.fail(io::Error::other(error.to_string()));
            return Err(error);
        };
        if chunk.coalesced {
            return self.send_coalesced(chunk);
        }
        sender.send(chunk).map_err(|send_error| {
            let error = io::Error::other("the cache copy workers stopped");
            send_error.0.fail(io::Error::other(error.to_string()));
            error
        })
    }

    fn send_coalesced(&self, chunk: Chunk) -> io::Result<()> {
        // Match the reference data path: coalesced copies stage and submit on their own thread,
        // outside the six-worker queue used for two-page copies.
        let core = Arc::clone(&self.core);
        let completion = Arc::clone(&chunk.completion);
        let job = Arc::clone(&chunk.job);
        let spawned = std::thread::Builder::new()
            .name("kv-copy-later".into())
            .spawn(
                move || match tokio::runtime::Builder::new_current_thread().build() {
                    Ok(executor) => chunk.run(&executor, &core),
                    Err(error) => chunk.fail(error),
                },
            );
        match spawned {
            Ok(_) => Ok(()),
            Err(error) => {
                let message = error.to_string();
                completion.finish(Err(io::Error::other(message.clone())));
                job.finish(Err(io::Error::other(message)));
                Err(error)
            }
        }
    }

    pub(super) fn shutdown(&mut self) {
        drop(self.sender.take());
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl Chunk {
    pub(super) fn split(
        pages: Vec<(usize, Vec<(usize, u32)>)>,
        pool: &Arc<Pool>,
        loading: bool,
        coalesced: bool,
        completion: &Arc<Completion>,
    ) -> (Vec<Self>, Vec<(TransferKey, Arc<ChunkCompletion>)>) {
        let chunks = Self::partition_pages(pages, coalesced);
        if chunks.is_empty() {
            completion.mark_submitted(Ok(()));
            completion.finish(Ok(()));
            return (Vec::new(), Vec::new());
        }
        let chunk_keys = chunks
            .iter()
            .map(|pages| {
                let mut keys = Vec::new();
                for page in pages {
                    keys.push(TransferKey::Host(page.slot));
                    keys.extend(
                        page.routes
                            .iter()
                            .map(|&(group, block)| TransferKey::Device(group, block)),
                    );
                }
                keys.sort_unstable();
                keys.dedup();
                keys
            })
            .collect::<Vec<_>>();
        let completions = (0..chunks.len())
            .map(|_| Arc::new(ChunkCompletion::default()))
            .collect::<Vec<_>>();
        let pending = chunk_keys
            .iter()
            .zip(&completions)
            .flat_map(|(keys, completion)| {
                keys.iter().map(move |&key| (key, Arc::clone(completion)))
            })
            .collect();
        let job = Arc::new(Job {
            left: AtomicUsize::new(chunks.len()),
            submit_left: AtomicUsize::new(chunks.len()),
            failure: Mutex::new(None),
            completion: Arc::clone(completion),
        });
        let chunks = chunks
            .into_iter()
            .zip(completions)
            .map(|(pages, completion)| Self {
                pages,
                pool: Arc::clone(pool),
                loading,
                coalesced,
                completion,
                job: Arc::clone(&job),
            })
            .collect();
        (chunks, pending)
    }

    fn partition_pages(
        pages: Vec<(usize, Vec<(usize, u32)>)>,
        coalesced: bool,
    ) -> Vec<Vec<TransferPage>> {
        if coalesced {
            let mut pages = pages
                .into_iter()
                .map(|(slot, routes)| TransferPage { slot, routes })
                .collect::<Vec<_>>();
            pages.sort_unstable_by_key(|page| page.slot);
            return if pages.is_empty() {
                Vec::new()
            } else {
                vec![pages]
            };
        }
        let mut chunks = Vec::<Vec<TransferPage>>::new();
        let mut seen_slots = HashSet::new();
        for (slot, routes) in pages {
            let page = TransferPage { slot, routes };
            let first_occurrence = seen_slots.insert(slot);
            match chunks.last_mut() {
                Some(last) if first_occurrence && last.len() < COPY_CHUNK_BLOCKS => {
                    last.push(page);
                }
                _ => chunks.push(vec![page]),
            }
        }
        // Keep input chunk boundaries; order each chunk's host locks to prevent lock cycles.
        for chunk in &mut chunks {
            chunk.sort_unstable_by_key(|page| page.slot);
        }
        chunks
    }

    fn run(self, executor: &tokio::runtime::Runtime, core: &Core) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.transfer(executor, core)
        }))
        .unwrap_or_else(|_| Err(io::Error::other("cache copy worker panicked")));
        self.completion.finish(clone_result(&result));
        self.job.finish(result);
    }

    pub(super) fn fail(&self, error: io::Error) {
        self.completion
            .finish(Err(io::Error::new(error.kind(), error.to_string())));
        self.job.finish(Err(error));
    }

    fn transfer(&self, executor: &tokio::runtime::Runtime, core: &Core) -> io::Result<()> {
        if self.coalesced {
            let mut hosts = Vec::new();
            let mut host_indices = BTreeMap::new();
            let mut page_hosts = Vec::with_capacity(self.pages.len());
            for page in &self.pages {
                let host_index = if let Some(&index) = host_indices.get(&page.slot) {
                    index
                } else {
                    let index = hosts.len();
                    hosts.push(self.pool.slots[page.slot].acquire_dma()?);
                    host_indices.insert(page.slot, index);
                    index
                };
                page_hosts.push(host_index);
            }
            return self.transfer_coalesced(executor, core, &mut hosts, &page_hosts);
        }
        let mut hosts = self
            .pages
            .iter()
            .map(|page| self.pool.slots[page.slot].acquire_dma())
            .collect::<io::Result<Vec<_>>>()?;
        let route_count = self
            .pages
            .iter()
            .flat_map(|page| &page.routes)
            .map(|(group, _)| core.groups[*group].len())
            .sum();
        let mut page_layers = Vec::with_capacity(route_count);
        for (page_index, page) in self.pages.iter().enumerate() {
            let first = page_layers.len();
            for &(group, block) in &page.routes {
                for &layer_id in &core.groups[group] {
                    page_layers.push((page_index, block, layer_id));
                }
            }
            page_layers[first..]
                .sort_unstable_by_key(|&(_, _, layer_id)| core.layers[layer_id].host_offset);
        }
        let mut aliases = Vec::with_capacity(route_count);
        for &(_, block, layer_id) in &page_layers {
            let layer = &core.layers[layer_id];
            let start = usize::try_from(block)
                .map_err(io::Error::other)?
                .checked_mul(layer.page_bytes)
                .ok_or_else(|| io::Error::other("device page offset overflows"))?;
            let end = start
                .checked_add(layer.page_bytes)
                .ok_or_else(|| io::Error::other("device page end overflows"))?;
            aliases.push(layer.buffer.alias(start..end).map_err(io::Error::other)?);
        }

        if self.loading {
            let mut pairs = Vec::with_capacity(route_count);
            for (&(page_index, _, layer_id), destination) in page_layers.iter().zip(&mut aliases) {
                let layer = &core.layers[layer_id];
                let length = layer
                    .page_bytes
                    .checked_mul(core.chips)
                    .ok_or_else(|| io::Error::other("host page size overflows"))?;
                let end = layer
                    .host_offset
                    .checked_add(length)
                    .ok_or_else(|| io::Error::other("host page end overflows"))?;
                let source = hosts[page_index]
                    .slice(layer.host_offset..end)
                    .map_err(io::Error::other)?;
                let destination = destination.slice_mut(.., ..).map_err(io::Error::other)?;
                pairs.push((source, destination));
            }
            let submitted = {
                let mut context = core
                    .context
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                context.device.pdma.to_hbm(pairs).submit()
            };
            self.job.submitted();
            executor.block_on(submitted).map_err(io::Error::other)
        } else {
            let mut pairs = Vec::with_capacity(route_count);
            let mut route_index = 0;
            for (page_index, host) in hosts.iter_mut().enumerate() {
                let mut rest = host.slice_mut(..).map_err(io::Error::other)?;
                let mut cursor = 0;
                while let Some(&(route_page, _, layer_id)) = page_layers.get(route_index) {
                    if route_page != page_index {
                        break;
                    }
                    let layer = &core.layers[layer_id];
                    let offset = layer.host_offset;
                    let length = layer
                        .page_bytes
                        .checked_mul(core.chips)
                        .ok_or_else(|| io::Error::other("host page size overflows"))?;
                    let gap = offset
                        .checked_sub(cursor)
                        .ok_or_else(|| io::Error::other("host layer ranges overlap"))?;
                    let (before, tail) = rest.split_at(gap).map_err(io::Error::other)?;
                    drop(before);
                    let (destination, tail) = tail.split_at(length).map_err(io::Error::other)?;
                    let source = aliases[route_index]
                        .slice(.., ..)
                        .map_err(io::Error::other)?;
                    pairs.push((source, destination));
                    rest = tail;
                    cursor = offset
                        .checked_add(length)
                        .ok_or_else(|| io::Error::other("host layer end overflows"))?;
                    route_index += 1;
                }
            }
            if route_index != page_layers.len() {
                return Err(io::Error::other("cache transfer has unrouted host layers"));
            }
            let submitted = {
                let mut context = core
                    .context
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                context.device.pdma.to_host(pairs).submit()
            };
            self.job.submitted();
            executor.block_on(submitted).map_err(io::Error::other)
        }
    }

    fn transfer_coalesced(
        &self,
        executor: &tokio::runtime::Runtime,
        core: &Core,
        hosts: &mut [DmaSlot<'_>],
        page_hosts: &[usize],
    ) -> io::Result<()> {
        let layers = core
            .layers
            .iter()
            .map(|layer| staging::LayerLayout {
                page_bytes: layer.page_bytes,
                host_offset: layer.host_offset,
            })
            .collect::<Vec<_>>();
        let layout = staging::Layout::new(&self.pages, &core.groups, &layers)?;
        if layout.bytes == 0 {
            self.job.submitted();
            return Ok(());
        }
        let mut stages = (0..core.chips)
            .map(|_| HostBuffer::zero(layout.bytes))
            .collect::<Vec<_>>();
        if stages.iter().any(|stage| !stage.is_pinned()) {
            return Err(io::Error::other(
                "coalesced cache copies require pinned staging buffers",
            ));
        }

        if self.loading {
            for item in &layout.items {
                for (chip, stage) in stages.iter_mut().enumerate() {
                    let source = hosts[page_hosts[item.page]]
                        .slice(..)
                        .map_err(io::Error::other)?;
                    let source = source.bytes().map_err(io::Error::other)?;
                    let mut destination = stage
                        .slice_mut(..)
                        .and_then(|slice| slice.into_bytes())
                        .map_err(io::Error::other)?;
                    item.place(chip, &source, &mut destination)?;
                }
            }
        }

        if self.loading {
            let mut aliases = staging_aliases(&layout, core)?;
            let mut pairs = Vec::with_capacity(layout.segments.len() * core.chips);
            for (chip, (stage, aliases)) in stages.iter().zip(&mut aliases).enumerate() {
                for (segment, alias) in layout.segments.iter().zip(aliases) {
                    let source = stage
                        .slice(segment.stage.clone())
                        .map_err(io::Error::other)?;
                    let destination = alias
                        .slice_mut(chip, segment.device.clone())
                        .map_err(io::Error::other)?;
                    pairs.push((source, destination));
                }
            }
            let submitted = {
                let mut context = core
                    .context
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                context.device.pdma.to_hbm(pairs).submit()
            };
            self.job.submitted();
            executor.block_on(submitted).map_err(io::Error::other)?;
        } else {
            let mut aliases = staging_aliases(&layout, core)?;
            let mut pairs = Vec::with_capacity(layout.segments.len() * core.chips);
            for (chip, (stage, aliases)) in stages.iter_mut().zip(&mut aliases).enumerate() {
                let mut rest = stage.slice_mut(..).map_err(io::Error::other)?;
                let mut cursor = 0;
                for (segment, alias) in layout.segments.iter().zip(aliases) {
                    let source = alias
                        .slice(chip, segment.device.clone())
                        .map_err(io::Error::other)?;
                    let gap = segment
                        .stage
                        .start
                        .checked_sub(cursor)
                        .ok_or_else(|| io::Error::other("cache staging segments overlap"))?;
                    let (before, tail) = rest.split_at(gap).map_err(io::Error::other)?;
                    drop(before);
                    let size = segment.stage.len();
                    let (destination, tail) = tail.split_at(size).map_err(io::Error::other)?;
                    rest = tail;
                    cursor = segment.stage.end;
                    pairs.push((source, destination));
                }
                if cursor != layout.bytes {
                    return Err(io::Error::other("cache staging has unrouted bytes"));
                }
            }
            let submitted = {
                let mut context = core
                    .context
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                context.device.pdma.to_host(pairs).submit()
            };
            self.job.submitted();
            executor.block_on(submitted).map_err(io::Error::other)?;

            for item in &layout.items {
                for (chip, stage) in stages.iter().enumerate() {
                    let source = stage.slice(..).map_err(io::Error::other)?;
                    let source = source.bytes().map_err(io::Error::other)?;
                    let mut destination = hosts[page_hosts[item.page]]
                        .slice_mut(..)
                        .and_then(|slice| slice.into_bytes())
                        .map_err(io::Error::other)?;
                    item.take(chip, &source, &mut destination)?;
                }
            }
        }
        Ok(())
    }
}

fn staging_aliases(
    layout: &staging::Layout,
    core: &Core,
) -> io::Result<Vec<Vec<prelude::HbmBuffer>>> {
    (0..core.chips)
        .map(|_| {
            layout
                .segments
                .iter()
                .map(|segment| core.layers[segment.layer].buffer.alias(..))
                .collect::<Result<Vec<_>, _>>()
                .map_err(io::Error::other)
        })
        .collect()
}

impl Job {
    fn submitted(&self) {
        if self.submit_left.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.completion.mark_submitted(Ok(()));
        }
    }

    fn finish(&self, result: io::Result<()>) {
        let mut failure = self
            .failure
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if let Err(error) = result {
            self.completion
                .mark_submitted(Err(io::Error::other(error.to_string())));
            failure.get_or_insert(error);
        }
        if self.left.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.completion.finish(failure.take().map_or(Ok(()), Err));
        }
    }
}

impl Transfers {
    pub(super) fn insert(&mut self, key: TransferKey, completion: Arc<ChunkCompletion>) {
        let entries = self.0.entry(key).or_default();
        if !entries.iter().any(|entry| Arc::ptr_eq(entry, &completion)) {
            entries.push(completion);
        }
    }

    pub(super) fn conflicts(&self, reservations: &[TransferKey]) -> Vec<Arc<ChunkCompletion>> {
        let mut seen = HashSet::new();
        reservations
            .iter()
            .filter_map(|reservation| self.0.get(reservation))
            .flatten()
            .filter(|copy| seen.insert(Arc::as_ptr(copy) as usize))
            .cloned()
            .collect()
    }

    pub(super) fn remove(&mut self, reservations: &[TransferKey]) {
        for reservation in reservations {
            self.0.remove(reservation);
        }
    }

    pub(super) fn drain(&mut self) -> io::Result<()> {
        let mut first_error = None;
        let mut seen = HashSet::new();
        for copy in self.0.values().flatten() {
            if !seen.insert(Arc::as_ptr(copy) as usize) {
                continue;
            }
            if let Err(error) = copy.wait() {
                first_error.get_or_insert(error);
            }
        }
        self.0.clear();
        first_error.map_or(Ok(()), Err)
    }
}

impl Completion {
    fn mark_submitted(&self, result: io::Result<()>) {
        let mut submission = self
            .submission
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if submission.is_none() {
            *submission = Some(result.map_err(|error| error.to_string()));
            self.submission_changed.notify_all();
        }
    }

    pub(super) fn wait_submitted(&self) -> io::Result<()> {
        let mut submission = self
            .submission
            .lock()
            .map_err(|error| io::Error::other(error.to_string()))?;
        loop {
            if let Some(result) = &*submission {
                return result
                    .as_ref()
                    .map(|()| ())
                    .map_err(|error| io::Error::other(error.clone()));
            }
            submission = self
                .submission_changed
                .wait(submission)
                .map_err(|error| io::Error::other(error.to_string()))?;
        }
    }

    fn finish(&self, result: io::Result<()>) {
        *self
            .result
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()) =
            Some(result.map_err(|error| error.to_string()));
        self.changed.notify_all();
    }

    pub(super) fn done(&self, timeout: Option<Duration>) -> io::Result<bool> {
        let started = Instant::now();
        let mut result = self
            .result
            .lock()
            .map_err(|error| io::Error::other(error.to_string()))?;
        loop {
            if let Some(result) = &*result {
                return result
                    .as_ref()
                    .map(|()| true)
                    .map_err(|error| io::Error::other(error.clone()));
            }
            match timeout {
                Some(timeout) => {
                    let left = timeout.saturating_sub(started.elapsed());
                    if left.is_zero() {
                        return Ok(false);
                    }
                    result = self
                        .changed
                        .wait_timeout(result, left)
                        .map_err(|error| io::Error::other(error.to_string()))?
                        .0;
                }
                None => {
                    result = self
                        .changed
                        .wait(result)
                        .map_err(|error| io::Error::other(error.to_string()))?;
                }
            }
        }
    }
}

impl Default for Completion {
    fn default() -> Self {
        Self {
            result: Mutex::new(None),
            changed: Condvar::new(),
            submission: Mutex::new(None),
            submission_changed: Condvar::new(),
        }
    }
}

impl ChunkCompletion {
    fn finish(&self, result: io::Result<()>) {
        *self
            .result
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) =
            Some(result.map_err(|error| error.to_string()));
        self.changed.notify_all();
    }

    pub(super) fn wait(&self) -> io::Result<()> {
        let mut result = self
            .result
            .lock()
            .map_err(|error| io::Error::other(error.to_string()))?;
        loop {
            if let Some(result) = &*result {
                return result
                    .as_ref()
                    .map(|()| ())
                    .map_err(|error| io::Error::other(error.clone()));
            }
            result = self
                .changed
                .wait(result)
                .map_err(|error| io::Error::other(error.to_string()))?;
        }
    }
}

impl Default for ChunkCompletion {
    fn default() -> Self {
        Self {
            result: Mutex::new(None),
            changed: Condvar::new(),
        }
    }
}

fn clone_result(result: &io::Result<()>) -> io::Result<()> {
    result
        .as_ref()
        .map(|()| ())
        .map_err(|error| io::Error::new(error.kind(), error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_reservations_track_only_chunks_that_touch_them() {
        let pool = Arc::new(Pool {
            slots: Vec::new(),
            block_bytes: 0,
            core: Weak::new(),
        });
        let job = Arc::new(Completion::default());
        let (chunks, reservations) = Chunk::split(
            vec![(0, vec![(0, 4)]), (1, vec![(0, 4)]), (2, vec![(0, 4)])],
            &pool,
            false,
            false,
            &job,
        );
        assert_eq!(chunks.len(), 2);

        let mut transfers = Transfers::default();
        for (key, completion) in reservations {
            transfers.0.entry(key).or_default().push(completion);
        }
        let page = transfers.conflicts(&[TransferKey::Device(0, 4)]);
        assert_eq!(page.len(), 2);
        let finished_chunk = Arc::clone(&chunks[0].completion);
        finished_chunk.finish(Ok(()));
        assert!(finished_chunk.wait().is_ok());
        assert_eq!(
            page.iter()
                .filter(|completion| completion.result.lock().unwrap().is_none())
                .count(),
            1
        );
    }

    #[test]
    fn coalesced_copies_keep_all_pages_in_one_stage_batch() {
        let pool = Arc::new(Pool {
            slots: Vec::new(),
            block_bytes: 0,
            core: Weak::new(),
        });
        let job = Arc::new(Completion::default());
        let (chunks, _) = Chunk::split(
            (0..7).map(|slot| (slot, vec![(0, slot as u32)])).collect(),
            &pool,
            true,
            true,
            &job,
        );
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].pages.len(), 7);
        assert!(chunks[0].coalesced);
    }

    #[test]
    fn page_batches_match_the_reference_queue_partition() {
        let pages = vec![
            (9, vec![(0, 9)]),
            (3, vec![(0, 3)]),
            (1, vec![(0, 1)]),
            (3, vec![(0, 30)]),
            (8, vec![(0, 8)]),
        ];

        let chunks = Chunk::partition_pages(pages, false);
        let slots: Vec<Vec<_>> = chunks
            .iter()
            .map(|chunk| chunk.iter().map(|page| page.slot).collect())
            .collect();

        assert_eq!(slots, [vec![3, 9], vec![1], vec![3, 8]]);
    }

    #[test]
    fn host_slot_views_and_dma_exclude_each_other() {
        let access = SlotAccess::default();
        access.acquire_dma().unwrap();
        assert!(!access.try_acquire_view().unwrap());
        access.release_dma();
        assert!(access.try_acquire_view().unwrap());
        access.release_view();
    }
}
