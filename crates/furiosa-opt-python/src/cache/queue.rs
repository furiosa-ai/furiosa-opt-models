//! DMA queue, pinned host-slot leases and transfer completion tracking.

use std::collections::{BTreeMap, HashSet};
use std::io;
use std::ops::DerefMut;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak, mpsc};
use std::time::{Duration, Instant};

use furiosa_opt_std::prelude::{self, HostBuffer};

use super::staging;

const COPY_THREADS: usize = 6;
const COPY_CHUNK_BLOCKS: usize = 2;
pub(super) type Pdma = prelude::DmaContext<{ prelude::Dma::Pcie }>;

pub(super) struct Layer {
    pub(super) buffer: prelude::HbmBuffer,
    pub(super) page_bytes: usize,
    pub(super) host_offset: usize,
    pub(super) blocks: usize,
}

pub(super) struct Core {
    pub(super) pdma: Pdma,
    pub(super) layers: Vec<Layer>,
    pub(super) groups: Vec<Vec<usize>>,
    pub(super) group_blocks: Vec<usize>,
    pub(super) chips: usize,
    pub(super) block_bytes: usize,
}

pub(super) struct Pool {
    pub(super) slots: Vec<Slot>,
    pub(super) block_bytes: usize,
    pub(super) layer_offsets: Vec<Option<usize>>,
    pub(super) included_groups: Vec<bool>,
    pub(super) core: Weak<Core>,
    // Backed slots are ranges in this single pool-owned allocation.
    pub(super) backing: Option<HostBacking>,
}

impl Pool {
    pub(super) fn includes_group(&self, group: usize) -> bool {
        self.included_groups.get(group).copied().unwrap_or(false)
    }

    pub(super) fn layer_offset(&self, layer: usize) -> io::Result<usize> {
        self.layer_offsets
            .get(layer)
            .copied()
            .flatten()
            .ok_or_else(|| io::Error::other("host cache does not include routed layer"))
    }

    pub(super) fn backing(&self) -> Option<&HostBacking> {
        self.backing.as_ref()
    }
}

/// Shared owner retained by every backed slot in one cache pool.
pub(super) struct HostBacking {
    pub(super) memory: Arc<Mutex<HostBuffer>>,
    pub(super) bytes: std::ops::Range<usize>,
}

impl HostBacking {
    pub(super) fn slot_range(
        &self,
        slot: usize,
        block_bytes: usize,
    ) -> io::Result<std::ops::Range<usize>> {
        let start = slot
            .checked_mul(block_bytes)
            .and_then(|offset| self.bytes.start.checked_add(offset))
            .ok_or_else(|| io::Error::other("host slot offset overflows"))?;
        let end = start
            .checked_add(block_bytes)
            .filter(|&end| end <= self.bytes.end)
            .ok_or_else(|| io::Error::other("host slot exceeds backing"))?;
        Ok(start..end)
    }
}

pub(super) struct Slot {
    pub(super) access: SlotAccess,
}

impl Slot {
    pub(super) fn owned(buffer: HostBuffer) -> Self {
        Self {
            access: SlotAccess::owned(buffer),
        }
    }

    pub(super) fn backed() -> Self {
        Self {
            access: SlotAccess::default(),
        }
    }
}

// One slot mutex protects its owned buffer and the lifetime state for both owned and backed slots.
#[derive(Default)]
pub(super) struct SlotAccess {
    state: Mutex<SlotAccessState>,
    changed: Condvar,
}

#[derive(Default)]
struct SlotAccessState {
    buffer: Option<HostBuffer>,
    dma: bool,
    leases: usize,
    waiting_dma: usize,
    waiting_leases: usize,
}

struct DmaSlot<'a> {
    access: &'a SlotAccess,
    state: std::sync::MutexGuard<'a, SlotAccessState>,
    release_on_drop: bool,
}

pub(super) struct SlotBuffer<'a> {
    state: std::sync::MutexGuard<'a, SlotAccessState>,
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
    job: Arc<Job>,
}

pub(super) struct TransferPage {
    pub(super) slot: usize,
    pub(super) routes: Vec<(usize, u32)>,
}

type PageLayer = (usize, u32, usize);

struct Job {
    left: AtomicUsize,
    failure: Mutex<Option<io::Error>>,
    completion: Arc<Completion>,
}

pub(super) struct Completion {
    result: Mutex<Option<Result<(), String>>>,
    changed: Condvar,
    status: AtomicU8,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TransferStatus {
    Pending,
    Succeeded,
    Failed,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct DevicePage(pub(super) usize, pub(super) u32);

#[derive(Default)]
// Device-page dependencies live for the whole Copy, across all its DMA chunks.
pub(super) struct Transfers(BTreeMap<DevicePage, Arc<Completion>>);

impl Slot {
    fn acquire_dma(&self) -> io::Result<DmaSlot<'_>> {
        let slot = self.access.acquire_dma()?;
        if slot.state.buffer.is_none() {
            drop(slot);
            return Err(io::Error::other(
                "a shared host-cache slot has no owned buffer",
            ));
        }
        Ok(slot)
    }

    fn acquire_dma_lease(&self) -> io::Result<DmaLease<'_>> {
        Ok(self.access.acquire_dma()?.into_lease())
    }
}

impl SlotAccess {
    fn owned(buffer: HostBuffer) -> Self {
        Self {
            state: Mutex::new(SlotAccessState {
                buffer: Some(buffer),
                ..SlotAccessState::default()
            }),
            changed: Condvar::new(),
        }
    }

    fn acquire_dma(&self) -> io::Result<DmaSlot<'_>> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while state.dma || state.leases != 0 {
            state.waiting_dma = state
                .waiting_dma
                .checked_add(1)
                .ok_or_else(|| io::Error::other("too many pending host slot transfers"))?;
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.waiting_dma -= 1;
        }
        state.dma = true;
        Ok(DmaSlot {
            access: self,
            state,
            release_on_drop: true,
        })
    }

    fn release_dma(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Self::release_dma_state(&mut state, &self.changed);
    }

    fn release_dma_state(state: &mut SlotAccessState, changed: &Condvar) {
        debug_assert!(state.dma);
        state.dma = false;
        if state.waiting_dma != 0 || state.waiting_leases != 0 {
            changed.notify_all();
        }
    }

    pub(super) fn acquire_lease(&self) -> io::Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|error| io::Error::other(error.to_string()))?;
        while state.dma || state.waiting_dma != 0 {
            state.waiting_leases = state
                .waiting_leases
                .checked_add(1)
                .ok_or_else(|| io::Error::other("too many pending host slot leases"))?;
            state = self
                .changed
                .wait(state)
                .map_err(|error| io::Error::other(error.to_string()))?;
            state.waiting_leases -= 1;
        }
        state.leases = state
            .leases
            .checked_add(1)
            .ok_or_else(|| io::Error::other("too many active host slot leases"))?;
        Ok(())
    }

    pub(super) fn lock_buffer(&self) -> io::Result<SlotBuffer<'_>> {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.buffer.is_none() {
            return Err(io::Error::other(
                "shared host-cache slot has no owned buffer",
            ));
        }
        Ok(SlotBuffer { state })
    }

    pub(super) fn release_lease(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert!(state.leases > 0);
        if state.leases > 0 {
            state.leases -= 1;
        }
        if state.leases == 0 && state.waiting_dma != 0 {
            self.changed.notify_all();
        }
    }
}

impl std::ops::Deref for DmaSlot<'_> {
    type Target = HostBuffer;

    fn deref(&self) -> &Self::Target {
        self.state
            .buffer
            .as_ref()
            .expect("owned host slot has no buffer")
    }
}

impl DerefMut for DmaSlot<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.state
            .buffer
            .as_mut()
            .expect("owned host slot has no buffer")
    }
}

impl<'a> DmaSlot<'a> {
    fn into_lease(mut self) -> DmaLease<'a> {
        self.release_on_drop = false;
        DmaLease {
            access: self.access,
        }
    }
}

impl Drop for DmaSlot<'_> {
    fn drop(&mut self) {
        if self.release_on_drop {
            SlotAccess::release_dma_state(&mut self.state, &self.access.changed);
        }
    }
}

struct DmaLease<'a> {
    access: &'a SlotAccess,
}

impl Drop for DmaLease<'_> {
    fn drop(&mut self) {
        self.access.release_dma();
    }
}

impl std::ops::Deref for SlotBuffer<'_> {
    type Target = HostBuffer;

    fn deref(&self) -> &Self::Target {
        self.state
            .buffer
            .as_ref()
            .expect("owned host slot has no buffer")
    }
}

impl DerefMut for SlotBuffer<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.state
            .buffer
            .as_mut()
            .expect("owned host slot has no buffer")
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
            let (receiver, core, mut pdma) =
                (Arc::clone(&receiver), Arc::clone(&core), core.pdma.clone());
            match std::thread::Builder::new()
                .name("kv-copy".into())
                .spawn(move || {
                    // A poisoned receiver means another copy thread panicked while holding it.
                    while let Some(chunk) = receiver.lock().ok().and_then(|next| next.recv().ok()) {
                        chunk.run(&executor, &core, &mut pdma);
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
        let mut pdma = self.core.pdma.clone();
        let job = Arc::clone(&chunk.job);
        let spawned = std::thread::Builder::new()
            .name("kv-copy-later".into())
            .spawn(
                move || match tokio::runtime::Builder::new_current_thread().build() {
                    Ok(executor) => chunk.run(&executor, &core, &mut pdma),
                    Err(error) => chunk.fail(error),
                },
            );
        match spawned {
            Ok(_) => Ok(()),
            Err(error) => {
                job.finish(Err(io::Error::other(error.to_string())));
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
        drop(self.sender.take());
        // Python may drop the queue while another thread needs the GIL to release a host-slot lease.
        // Closing the channel lets workers drain safely; dropping JoinHandles detaches them.
        self.workers.clear();
    }
}

impl Chunk {
    pub(super) fn split(
        pages: Vec<(usize, Vec<(usize, u32)>)>,
        pool: &Arc<Pool>,
        loading: bool,
        coalesced: bool,
        completion: &Arc<Completion>,
    ) -> Vec<Self> {
        let chunks = Self::partition_pages(pages, coalesced);
        if chunks.is_empty() {
            completion.finish(Ok(()));
            return Vec::new();
        }
        let job = Arc::new(Job {
            left: AtomicUsize::new(chunks.len()),
            failure: Mutex::new(None),
            completion: Arc::clone(completion),
        });
        chunks
            .into_iter()
            .map(|pages| Self {
                pages,
                pool: Arc::clone(pool),
                loading,
                coalesced,
                job: Arc::clone(&job),
            })
            .collect()
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
        for (slot, routes) in pages {
            let page = TransferPage { slot, routes };
            match chunks.last_mut() {
                Some(last)
                    if last.len() < COPY_CHUNK_BLOCKS
                        && last.iter().all(|page| page.slot != slot) =>
                {
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

    fn run(self, executor: &tokio::runtime::Runtime, core: &Core, pdma: &mut Pdma) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.transfer(executor, core, pdma)
        }))
        .unwrap_or_else(|_| Err(io::Error::other("cache copy worker panicked")));
        self.job.finish(result);
    }

    pub(super) fn fail(&self, error: io::Error) {
        self.job.finish(Err(error));
    }

    fn transfer(
        &self,
        executor: &tokio::runtime::Runtime,
        core: &Core,
        pdma: &mut Pdma,
    ) -> io::Result<()> {
        if self.coalesced {
            return self.transfer_coalesced(executor, core, pdma);
        }
        if self.pool.backing().is_some() {
            return self.transfer_backed(executor, core, pdma);
        }
        let mut hosts = self
            .pages
            .iter()
            .map(|page| self.pool.slots[page.slot].acquire_dma())
            .collect::<io::Result<Vec<_>>>()?;
        let (page_layers, mut aliases) = self.direct_routes(core)?;
        let route_count = page_layers.len();

        if self.loading {
            let mut pairs = Vec::with_capacity(route_count);
            for (&(page_index, _, layer_id), destination) in page_layers.iter().zip(&mut aliases) {
                let layer = &core.layers[layer_id];
                let length = layer
                    .page_bytes
                    .checked_mul(core.chips)
                    .ok_or_else(|| io::Error::other("host page size overflows"))?;
                let offset = self.pool.layer_offset(layer_id)?;
                let end = offset
                    .checked_add(length)
                    .ok_or_else(|| io::Error::other("host page end overflows"))?;
                let source = hosts[page_index]
                    .slice(offset..end)
                    .map_err(io::Error::other)?;
                let destination = destination.slice_mut(.., ..).map_err(io::Error::other)?;
                pairs.push((source, destination));
            }
            let submitted = pdma.to_hbm(pairs).submit();
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
                    let offset = self.pool.layer_offset(layer_id)?;
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
            let submitted = pdma.to_host(pairs).submit();
            executor.block_on(submitted).map_err(io::Error::other)
        }
    }

    fn transfer_backed(
        &self,
        executor: &tokio::runtime::Runtime,
        core: &Core,
        pdma: &mut Pdma,
    ) -> io::Result<()> {
        let backing = self
            .pool
            .backing()
            .ok_or_else(|| io::Error::other("shared host-cache backing is missing"))?;
        let _slots = self.acquire_dma_leases()?;
        let (page_layers, mut aliases) = self.direct_routes(core)?;
        let route_count = page_layers.len();

        // Submission retains the pinned DMA-buf in the runtime request. Keep the Pool Arc and
        // page leases through completion; release the backing mutex after submission.
        if self.loading {
            let submitted = {
                let memory = backing
                    .memory
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let mut pairs = Vec::with_capacity(route_count);
                for (&(page_index, _, layer_id), destination) in
                    page_layers.iter().zip(&mut aliases)
                {
                    let layer = &core.layers[layer_id];
                    let slot_range =
                        backing.slot_range(self.pages[page_index].slot, self.pool.block_bytes)?;
                    let start = slot_range
                        .start
                        .checked_add(self.pool.layer_offset(layer_id)?)
                        .ok_or_else(|| io::Error::other("host layer offset overflows"))?;
                    let length = layer
                        .page_bytes
                        .checked_mul(core.chips)
                        .ok_or_else(|| io::Error::other("host page size overflows"))?;
                    let end = start
                        .checked_add(length)
                        .filter(|&end| end <= slot_range.end)
                        .ok_or_else(|| io::Error::other("cache transfer exceeds host slot"))?;
                    let source = memory.slice(start..end).map_err(io::Error::other)?;
                    let destination = destination.slice_mut(.., ..).map_err(io::Error::other)?;
                    pairs.push((source, destination));
                }
                pdma.to_hbm(pairs).submit()
            };
            executor.block_on(submitted).map_err(io::Error::other)
        } else {
            let submitted = {
                let mut memory = backing
                    .memory
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let mut rest = memory
                    .slice_mut(backing.bytes.clone())
                    .map_err(io::Error::other)?;
                let mut cursor = backing.bytes.start;
                let mut pairs = Vec::with_capacity(route_count);
                for (&(page_index, _, layer_id), source) in page_layers.iter().zip(&mut aliases) {
                    let layer = &core.layers[layer_id];
                    let slot_range =
                        backing.slot_range(self.pages[page_index].slot, self.pool.block_bytes)?;
                    let start = slot_range
                        .start
                        .checked_add(self.pool.layer_offset(layer_id)?)
                        .ok_or_else(|| io::Error::other("host layer offset overflows"))?;
                    let length = layer
                        .page_bytes
                        .checked_mul(core.chips)
                        .ok_or_else(|| io::Error::other("host page size overflows"))?;
                    let end = start
                        .checked_add(length)
                        .filter(|&end| end <= slot_range.end)
                        .ok_or_else(|| io::Error::other("cache transfer exceeds host slot"))?;
                    let gap = start
                        .checked_sub(cursor)
                        .ok_or_else(|| io::Error::other("cache host ranges overlap"))?;
                    let (before, tail) = rest.split_at(gap).map_err(io::Error::other)?;
                    drop(before);
                    let (destination, tail) = tail.split_at(length).map_err(io::Error::other)?;
                    pairs.push((source.slice(.., ..).map_err(io::Error::other)?, destination));
                    rest = tail;
                    cursor = end;
                }
                pdma.to_host(pairs).submit()
            };
            executor.block_on(submitted).map_err(io::Error::other)
        }
    }

    fn acquire_dma_leases(&self) -> io::Result<Vec<DmaLease<'_>>> {
        let mut leases = Vec::with_capacity(self.pages.len());
        let mut previous_slot = None;
        for page in &self.pages {
            if previous_slot != Some(page.slot) {
                leases.push(self.pool.slots[page.slot].acquire_dma_lease()?);
                previous_slot = Some(page.slot);
            }
        }
        Ok(leases)
    }

    fn direct_routes(&self, core: &Core) -> io::Result<(Vec<PageLayer>, Vec<prelude::HbmBuffer>)> {
        let route_count = self
            .pages
            .iter()
            .flat_map(|page| &page.routes)
            .map(|(group, _)| core.groups[*group].len())
            .sum();
        let mut page_layers = Vec::with_capacity(route_count);
        for (page_index, page) in self.pages.iter().enumerate() {
            for &(group, block) in &page.routes {
                for &layer_id in &core.groups[group] {
                    page_layers.push((
                        page_index,
                        block,
                        layer_id,
                        self.pool.layer_offset(layer_id)?,
                    ));
                }
            }
        }
        page_layers.sort_unstable_by_key(|&(page_index, _, _, host_offset)| {
            (self.pages[page_index].slot, host_offset)
        });

        let mut aliases = Vec::with_capacity(route_count);
        for &(_, block, layer_id, _) in &page_layers {
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
        Ok((
            page_layers
                .into_iter()
                .map(|(page, block, layer, _)| (page, block, layer))
                .collect(),
            aliases,
        ))
    }

    fn transfer_coalesced(
        &self,
        executor: &tokio::runtime::Runtime,
        core: &Core,
        pdma: &mut Pdma,
    ) -> io::Result<()> {
        let layers = core
            .layers
            .iter()
            .enumerate()
            .map(|(index, layer)| staging::LayerLayout {
                page_bytes: layer.page_bytes,
                host_offset: self.pool.layer_offsets[index],
            })
            .collect::<Vec<_>>();
        let mut layout = staging::Layout::new(&self.pages, &core.groups, &layers, self.loading)?;
        if layout.bytes == 0 {
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
        let slot_items = self.group_coalesced_items(&mut layout.items);

        if self.loading {
            for group in &slot_items {
                let dma = self.pool.slots[group.slot].access.acquire_dma()?;
                self.place_coalesced_slot(
                    group.slot,
                    &layout.items[group.items.clone()],
                    &mut stages,
                    core.chips,
                    &dma,
                )?;
                // Staging now owns the load snapshot; do not hold the slot through DMA.
                drop(dma);
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
            let submitted = pdma.to_hbm(pairs).submit();
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
            let submitted = pdma.to_host(pairs).submit();
            executor.block_on(submitted).map_err(io::Error::other)?;

            for group in &slot_items {
                let mut dma = self.pool.slots[group.slot].access.acquire_dma()?;
                self.take_coalesced_slot(
                    group.slot,
                    &layout.items[group.items.clone()],
                    &stages,
                    core.chips,
                    &mut dma,
                )?;
            }
        }
        Ok(())
    }

    fn group_coalesced_items(&self, items: &mut [staging::Item]) -> Vec<SlotItems> {
        items.sort_unstable_by_key(|item| (self.pages[item.page].slot, item.stage.start));
        let mut groups = Vec::with_capacity(self.pages.len());
        let mut start = 0;
        while start < items.len() {
            let slot = self.pages[items[start].page].slot;
            let mut end = start + 1;
            while end < items.len() && self.pages[items[end].page].slot == slot {
                end += 1;
            }
            groups.push(SlotItems {
                slot,
                items: start..end,
            });
            start = end;
        }
        groups
    }

    fn place_coalesced_slot(
        &self,
        slot_index: usize,
        items: &[staging::Item],
        stages: &mut [HostBuffer],
        chips: usize,
        dma: &DmaSlot<'_>,
    ) -> io::Result<()> {
        if let Some(backing) = self.pool.backing() {
            let range = self.coalesced_backing_range(backing, slot_index, items, chips)?;
            let memory = backing
                .memory
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let source = memory.slice(range).map_err(io::Error::other)?;
            let source = source.bytes().map_err(io::Error::other)?;
            for item in items {
                item.place(stages, &source)?;
            }
        } else {
            let source = dma.slice(..).map_err(io::Error::other)?;
            let source = source.bytes().map_err(io::Error::other)?;
            for item in items {
                item.place(stages, &source)?;
            }
        }
        Ok(())
    }

    fn take_coalesced_slot(
        &self,
        slot_index: usize,
        items: &[staging::Item],
        stages: &[HostBuffer],
        chips: usize,
        dma: &mut DmaSlot<'_>,
    ) -> io::Result<()> {
        if let Some(backing) = self.pool.backing() {
            let range = self.coalesced_backing_range(backing, slot_index, items, chips)?;
            let mut memory = backing
                .memory
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let destination = memory.slice_mut(range).map_err(io::Error::other)?;
            let mut destination = destination.into_bytes().map_err(io::Error::other)?;
            for item in items {
                item.take(stages, &mut destination)?;
            }
        } else {
            let destination = dma.slice_mut(..).map_err(io::Error::other)?;
            let mut destination = destination.into_bytes().map_err(io::Error::other)?;
            for item in items {
                item.take(stages, &mut destination)?;
            }
        }
        Ok(())
    }

    fn coalesced_backing_range(
        &self,
        backing: &HostBacking,
        slot_index: usize,
        items: &[staging::Item],
        chips: usize,
    ) -> io::Result<std::ops::Range<usize>> {
        let slot = backing.slot_range(slot_index, self.pool.block_bytes)?;
        for item in items {
            let layer_end = item
                .page_bytes
                .checked_mul(chips)
                .and_then(|length| item.host_offset.checked_add(length))
                .ok_or_else(|| io::Error::other("host page size overflows"))?;
            if layer_end > slot.len() {
                return Err(io::Error::other("cache transfer exceeds host slot"));
            }
        }
        // Item::place/take applies its layer offset relative to the complete host slot.
        Ok(slot)
    }
}

struct SlotItems {
    slot: usize,
    items: std::ops::Range<usize>,
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
    fn finish(&self, result: io::Result<()>) {
        let mut failure = self
            .failure
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if let Err(error) = result {
            failure.get_or_insert(error);
        }
        if self.left.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.completion.finish(failure.take().map_or(Ok(()), Err));
        }
    }
}

impl Transfers {
    pub(super) fn insert(&mut self, pages: &[DevicePage], completion: Arc<Completion>) {
        for &page in pages {
            self.0.insert(page, Arc::clone(&completion));
        }
    }

    pub(super) fn conflicts(&mut self, pages: &[DevicePage]) -> io::Result<bool> {
        let mut conflict = false;
        let mut poisoned = false;
        for &page in pages {
            let Some(copy) = self.0.get(&page) else {
                continue;
            };
            match copy.status() {
                TransferStatus::Pending => conflict = true,
                TransferStatus::Succeeded => {
                    self.0.remove(&page);
                }
                TransferStatus::Failed => poisoned = true,
            }
        }
        if poisoned {
            return Err(io::Error::other(
                "a failed cache transfer left this device page unavailable",
            ));
        }
        Ok(conflict)
    }

    pub(super) fn drain(&mut self) -> io::Result<()> {
        let mut first_error = None;
        let mut seen = HashSet::new();
        for copy in self.0.values() {
            if !seen.insert(Arc::as_ptr(copy) as usize) {
                continue;
            }
            if let Err(error) = copy.done(None) {
                first_error.get_or_insert(error);
            }
        }
        self.0.clear();
        first_error.map_or(Ok(()), Err)
    }
}

impl Completion {
    fn finish(&self, result: io::Result<()>) {
        let status = if result.is_ok() {
            TransferStatus::Succeeded
        } else {
            TransferStatus::Failed
        };
        self.status.store(status as u8, Ordering::Release);
        *self
            .result
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()) =
            Some(result.map_err(|error| error.to_string()));
        self.changed.notify_all();
    }

    fn status(&self) -> TransferStatus {
        match self.status.load(Ordering::Acquire) {
            status if status == TransferStatus::Pending as u8 => TransferStatus::Pending,
            status if status == TransferStatus::Succeeded as u8 => TransferStatus::Succeeded,
            status if status == TransferStatus::Failed as u8 => TransferStatus::Failed,
            _ => unreachable!("invalid cache transfer status"),
        }
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
            status: AtomicU8::new(TransferStatus::Pending as u8),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_access_prioritizes_waiting_dma_over_new_leases() {
        let access = Arc::new(SlotAccess::default());
        access.acquire_lease().unwrap();

        let (started_tx, started_rx) = mpsc::channel();
        let (acquired_tx, acquired_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let waiting_access = Arc::clone(&access);
        let worker = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            let dma = waiting_access.acquire_dma().unwrap();
            acquired_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            drop(dma);
        });

        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let deadline = Instant::now() + Duration::from_secs(1);
        while access.state.lock().unwrap().waiting_dma == 0 && Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert_ne!(access.state.lock().unwrap().waiting_dma, 0);

        let (lease_started_tx, lease_started_rx) = mpsc::channel();
        let (lease_acquired_tx, lease_acquired_rx) = mpsc::channel();
        let lease_access = Arc::clone(&access);
        let lease_worker = std::thread::spawn(move || {
            lease_started_tx.send(()).unwrap();
            lease_access.acquire_lease().unwrap();
            lease_acquired_tx.send(()).unwrap();
            lease_access.release_lease();
        });
        lease_started_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(1);
        while access.state.lock().unwrap().waiting_leases == 0 && Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert_ne!(access.state.lock().unwrap().waiting_leases, 0);

        access.release_lease();
        acquired_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(
            lease_acquired_rx
                .recv_timeout(Duration::from_millis(20))
                .is_err()
        );
        release_tx.send(()).unwrap();
        worker.join().unwrap();
        lease_acquired_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        lease_worker.join().unwrap();
    }

    #[test]
    fn uncontended_dma_skips_waiter_bookkeeping() {
        let access = SlotAccess::default();

        let dma = access.acquire_dma().unwrap();

        assert_eq!(dma.state.waiting_dma, 0);
        assert_eq!(dma.state.waiting_leases, 0);
        drop(dma);

        let state = access.state.lock().unwrap();
        assert!(!state.dma);
        assert_eq!(state.waiting_dma, 0);
        assert_eq!(state.waiting_leases, 0);
    }

    #[test]
    fn contiguous_backing_maps_consecutive_slots_and_checks_bounds() {
        let backing = HostBacking {
            memory: Arc::new(Mutex::new(HostBuffer::zero(48))),
            bytes: 0..48,
        };

        assert_eq!(backing.slot_range(0, 16).unwrap(), 0..16);
        assert_eq!(backing.slot_range(1, 16).unwrap(), 16..32);
        assert_eq!(backing.slot_range(2, 16).unwrap(), 32..48);
        assert!(backing.slot_range(3, 16).is_err());
    }

    #[test]
    fn page_reservations_follow_the_whole_copy_completion() {
        let pool = Arc::new(Pool {
            slots: Vec::new(),
            block_bytes: 0,
            layer_offsets: Vec::new(),
            included_groups: Vec::new(),
            core: Weak::new(),
            backing: None,
        });
        let completion = Arc::new(Completion::default());
        let chunks = Chunk::split(
            vec![(0, vec![(0, 4)]), (1, vec![(0, 4)]), (2, vec![(0, 4)])],
            &pool,
            false,
            false,
            &completion,
        );
        assert_eq!(chunks.len(), 2);

        let page = DevicePage(0, 4);
        let mut transfers = Transfers::default();
        transfers.insert(&[page], Arc::clone(&completion));

        assert!(transfers.conflicts(&[page]).unwrap());
        chunks[0].job.finish(Ok(()));
        assert!(transfers.conflicts(&[page]).unwrap());
        chunks[1].job.finish(Ok(()));
        assert!(!transfers.conflicts(&[page]).unwrap());
    }

    #[test]
    fn completion_tracks_success_without_locking_result() {
        let completion = Completion::default();

        assert_eq!(completion.status(), TransferStatus::Pending);
        completion.finish(Ok(()));
        assert_eq!(completion.status(), TransferStatus::Succeeded);
    }

    #[test]
    fn failed_copy_keeps_all_device_page_reservations() {
        let pages = [DevicePage(0, 7), DevicePage(0, 8)];
        let completion = Arc::new(Completion::default());
        completion.finish(Err(io::Error::other("transfer failed")));
        let mut transfers = Transfers::default();
        transfers.insert(&pages, completion);

        let error = transfers.conflicts(&pages).unwrap_err();

        assert!(error.to_string().contains("device page unavailable"));
        assert!(pages.iter().all(|page| transfers.0.contains_key(page)));
    }

    #[test]
    fn coalesced_copies_keep_all_pages_in_one_stage_batch() {
        let pool = Arc::new(Pool {
            slots: Vec::new(),
            block_bytes: 0,
            layer_offsets: Vec::new(),
            included_groups: Vec::new(),
            core: Weak::new(),
            backing: None,
        });
        let job = Arc::new(Completion::default());
        let chunks = Chunk::split(
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
    fn page_batches_match_the_reference_for_reused_slots() {
        let pages = vec![
            (9, vec![(0, 9)]),
            (3, vec![(0, 3)]),
            (1, vec![(0, 1)]),
            (3, vec![(0, 30)]),
            (8, vec![(0, 8)]),
        ];

        let chunks = Chunk::partition_pages(pages, false);
        let pages: Vec<Vec<_>> = chunks
            .iter()
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|page| (page.slot, page.routes.clone()))
                    .collect()
            })
            .collect();

        assert_eq!(
            pages,
            [
                vec![(3, vec![(0, 3)]), (9, vec![(0, 9)])],
                vec![(1, vec![(0, 1)]), (3, vec![(0, 30)])],
                vec![(8, vec![(0, 8)])],
            ]
        );
    }
}
