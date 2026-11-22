//! The pinned host cache and its asynchronous HBM transfer queue.
//!
//! A slot holds every selected layer's bytes for one cache block. Six workers submit transfers in
//! chunks of two blocks, each chunk batching all its layer copies into one runtime read or write.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{c_int, c_void};
use std::io;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak, mpsc};
use std::time::{Duration, Instant};

use furiosa_opt_std::prelude::{self, HostBuffer};
use pyo3::exceptions::PyValueError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::tensor::HbmBuffer as PyHbmBuffer;
use crate::{Context, Shared};

mod huge;

const ALLOCATORS: usize = 8;
const ALLOCATION_GROUP_BYTES: usize = 256 << 20;
const COPY_THREADS: usize = 6;
const COPY_CHUNK_BLOCKS: usize = 2;

struct Layer {
    buffer: prelude::HbmBuffer,
    page_bytes: usize,
    host_offset: usize,
    blocks: usize,
}

struct Core {
    context: Shared,
    layers: Vec<Layer>,
    groups: Vec<Vec<usize>>,
    chips: usize,
    block_bytes: usize,
}

/// Python handle for the device cache's copy queue.
#[gen_stub_pyclass]
#[pyclass(name = "Copier", module = "engine", unsendable)]
pub(crate) struct PyCopier {
    core: Arc<Core>,
    queue: Queue,
    pending: Transfers,
}

/// A persistent pool of page-locked host blocks, one slot per offloaded cache block.
#[gen_stub_pyclass]
#[pyclass(name = "HostCache", module = "engine", frozen)]
pub(crate) struct PyHostCache {
    pool: Arc<Pool>,
}

/// A zero-copy view of one pinned host slot.
#[gen_stub_pyclass]
#[pyclass(name = "HostSlot", module = "engine", frozen)]
pub(crate) struct PyHostSlot {
    pool: Arc<Pool>,
    slot: usize,
    bytes: Range<usize>,
}

/// Completion of a queued cache copy.
#[gen_stub_pyclass]
#[pyclass(name = "Copy", module = "engine", frozen)]
#[derive(Clone)]
pub(crate) struct PyCopy(Arc<Completion>);

struct Pool {
    slots: Vec<Mutex<HostBuffer>>,
    block_bytes: usize,
    core: Weak<Core>,
}

struct Queue {
    sender: Option<mpsc::Sender<Chunk>>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

struct Chunk {
    pages: Vec<TransferPage>,
    pool: Arc<Pool>,
    loading: bool,
    job: Arc<Job>,
}

struct TransferPage {
    slot: usize,
    routes: Vec<(usize, u32)>,
}

struct Job {
    left: AtomicUsize,
    submit_left: AtomicUsize,
    failure: Mutex<Option<io::Error>>,
    completion: Arc<Completion>,
}

struct Completion {
    result: Mutex<Option<Result<(), String>>>,
    changed: Condvar,
    submission: Mutex<Option<Result<(), String>>>,
    submission_changed: Condvar,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum TransferKey {
    Device(usize, u32),
    Host(usize),
}

#[derive(Default)]
struct Transfers(BTreeMap<TransferKey, Arc<Completion>>);

#[gen_stub_pymethods]
#[pymethods]
impl PyCopier {
    /// Build one transfer queue over the layer buffers.
    ///
    /// `page_bytes`, `host_offsets`, and `groups` have one entry per layer. Page sizes are bytes
    /// per chip; offsets address the concatenated bytes in each host slot.
    #[new]
    fn new(
        py: Python<'_>,
        layers: Vec<PyRef<'_, PyHbmBuffer>>,
        page_bytes: Vec<usize>,
        host_offsets: Vec<usize>,
        groups: Vec<usize>,
        host_block_bytes: usize,
    ) -> PyResult<Self> {
        let first = layers
            .first()
            .ok_or_else(|| PyValueError::new_err("cache copies require nonempty layer buffers"))?;
        if page_bytes.len() != layers.len()
            || host_offsets.len() != layers.len()
            || groups.len() != layers.len()
        {
            return Err(PyValueError::new_err(
                "each cache layer needs a page size, host offset, and group",
            ));
        }
        if host_block_bytes == 0 {
            return Err(PyValueError::new_err(
                "host cache blocks must contain bytes",
            ));
        }
        let context = first.context.clone();
        let chips = usize::from(Context::lock(&context, py).device.topology().chips);
        let mut core_layers = Vec::with_capacity(layers.len());
        let mut core_groups = BTreeMap::<usize, Vec<usize>>::new();
        let mut ranges = Vec::with_capacity(layers.len());
        for (index, (((layer, &page), &offset), &group)) in layers
            .iter()
            .zip(&page_bytes)
            .zip(&host_offsets)
            .zip(&groups)
            .enumerate()
        {
            if !Arc::ptr_eq(&context, &layer.context) {
                return Err(PyValueError::new_err("cache layers must share a device"));
            }
            let bytes = layer.inner.size();
            if bytes == 0 || page == 0 || !page.is_multiple_of(8) || bytes % page != 0 {
                return Err(PyValueError::new_err(
                    "cache layers need nonempty, eight-byte-aligned whole pages",
                ));
            }
            let host_page = page
                .checked_mul(chips)
                .ok_or_else(|| PyValueError::new_err("cache page size overflows"))?;
            let end = offset
                .checked_add(host_page)
                .filter(|&end| end <= host_block_bytes)
                .ok_or_else(|| PyValueError::new_err("cache page exceeds its host block"))?;
            ranges.push((offset, end));
            let buffer = layer.inner.alias(..).map_err(crate::Error::from)?;
            core_layers.push(Layer {
                buffer,
                page_bytes: page,
                host_offset: offset,
                blocks: bytes / page,
            });
            core_groups.entry(group).or_default().push(index);
        }
        ranges.sort_unstable();
        if ranges.windows(2).any(|window| window[0].1 > window[1].0) {
            return Err(PyValueError::new_err("cache layer host ranges overlap"));
        }
        let groups = core_groups.into_values().collect::<Vec<_>>();
        let core = Arc::new(Core {
            context,
            layers: core_layers,
            groups,
            chips,
            block_bytes: host_block_bytes,
        });
        let queue = Queue::start(Arc::clone(&core)).map_err(crate::Error::from)?;
        Ok(Self {
            core,
            queue,
            pending: Transfers::default(),
        })
    }

    /// Submit device-to-host copies for host slots and their `(group, device_block)` routes.
    fn copy_out(
        &mut self,
        pages: Vec<(usize, Vec<(usize, u32)>)>,
        host: PyRef<'_, PyHostCache>,
    ) -> PyResult<PyCopy> {
        self.submit(pages, &host.pool, false)
    }

    /// Submit host-to-device copies for host slots and their `(group, device_block)` routes.
    fn copy_in(
        &mut self,
        pages: Vec<(usize, Vec<(usize, u32)>)>,
        host: PyRef<'_, PyHostCache>,
    ) -> PyResult<PyCopy> {
        self.submit(pages, &host.pool, true)
    }

    /// Wait for every submitted copy and stop the workers.
    fn shutdown(&mut self) -> PyResult<()> {
        let result = self.pending.drain();
        self.queue.shutdown();
        result.map_err(crate::Error::from)?;
        Ok(())
    }
}

impl PyCopier {
    fn submit(
        &mut self,
        pages: Vec<(usize, Vec<(usize, u32)>)>,
        pool: &Arc<Pool>,
        loading: bool,
    ) -> PyResult<PyCopy> {
        if pool.block_bytes != self.core.block_bytes
            || !pool
                .core
                .upgrade()
                .is_some_and(|core| Arc::ptr_eq(&core, &self.core))
        {
            return Err(PyValueError::new_err(
                "host cache was allocated for a different copier",
            ));
        }
        let mut destinations = BTreeSet::new();
        let mut host_slots = BTreeSet::new();
        for (slot, routes) in &pages {
            if *slot >= pool.slots.len() || routes.is_empty() || !host_slots.insert(*slot) {
                return Err(PyValueError::new_err(
                    "cache pages need distinct valid host slots and nonempty routes",
                ));
            }
            let mut page_groups = BTreeSet::new();
            for &(group, block) in routes {
                let layers =
                    self.core.groups.get(group).ok_or_else(|| {
                        PyValueError::new_err("cache copy names an unknown group")
                    })?;
                if !page_groups.insert(group)
                    || layers
                        .iter()
                        .any(|&index| block as usize >= self.core.layers[index].blocks)
                {
                    return Err(PyValueError::new_err(
                        "cache page has a duplicate group or an out-of-range device block",
                    ));
                }
                if loading && !destinations.insert(TransferKey::Device(group, block)) {
                    return Err(PyValueError::new_err(
                        "cache copy destinations must be distinct",
                    ));
                }
            }
        }
        let mut reservations = BTreeSet::new();
        for (slot, routes) in &pages {
            reservations.insert(TransferKey::Host(*slot));
            reservations.extend(
                routes
                    .iter()
                    .map(|&(group, block)| TransferKey::Device(group, block)),
            );
        }
        self.pending
            .check(&reservations, Some(Duration::ZERO))
            .map_err(crate::Error::from)?;

        let completion = Arc::new(Completion::default());
        let chunks = Chunk::split(pages, pool, loading, &completion);
        let mut chunks = chunks.into_iter();
        while let Some(chunk) = chunks.next() {
            let job = Arc::clone(&chunk.job);
            if self.queue.send(chunk).is_err() {
                let remaining = 1 + chunks.count();
                let error = io::Error::other("the cache copy workers stopped");
                for _ in 0..remaining {
                    job.finish(Err(io::Error::other(error.to_string())));
                }
                let _ = completion.done(None);
                return Err(crate::Error::from(error).into());
            }
        }
        self.pending.0.extend(
            reservations
                .into_iter()
                .map(|key| (key, Arc::clone(&completion))),
        );
        Ok(PyCopy(completion))
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyHostCache {
    #[new]
    fn new(py: Python<'_>, copier: PyRef<'_, PyCopier>, slots: usize) -> PyResult<Self> {
        let block_bytes = copier.core.block_bytes;
        if slots == 0 {
            return Err(PyValueError::new_err("host cache needs at least one slot"));
        }
        if block_bytes > 64 << 20 {
            return Err(PyValueError::new_err(
                "a host cache block cannot exceed 64 MiB",
            ));
        }
        let total = slots
            .checked_mul(block_bytes)
            .ok_or_else(|| PyValueError::new_err("host cache allocation size overflows"))?;
        let group_bytes = (ALLOCATION_GROUP_BYTES / block_bytes).max(1);
        let next = AtomicUsize::new(0);
        let allocate = || -> io::Result<Vec<(usize, Vec<Mutex<HostBuffer>>)>> {
            let mut groups = Vec::new();
            loop {
                let first = next.fetch_add(group_bytes, Ordering::Relaxed);
                if first >= slots {
                    break;
                }
                let count = group_bytes.min(slots - first);
                // Faulting and releasing THPs compacts memory before the dma-heap allocates slots.
                drop(huge::Pages::fault(count * block_bytes)?);
                let pinned = (0..count)
                    .map(|_| {
                        let buffer = HostBuffer::zero(block_bytes);
                        if !buffer.is_pinned() {
                            return Err(io::Error::other(
                                "the runtime did not provide page-locked host memory",
                            ));
                        }
                        Ok(Mutex::new(buffer))
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                groups.push((first, pinned));
            }
            Ok(groups)
        };
        let mut groups = py
            .detach(|| {
                std::thread::scope(|scope| {
                    let allocators = (0..ALLOCATORS)
                        .map(|_| std::thread::Builder::new().spawn_scoped(scope, allocate))
                        .collect::<io::Result<Vec<_>>>()?;
                    let mut groups = Vec::new();
                    for allocator in allocators {
                        groups.extend(
                            allocator.join().map_err(|_| {
                                io::Error::other("a host slot allocator panicked")
                            })??,
                        );
                    }
                    groups.sort_unstable_by_key(|(first, _)| *first);
                    Ok::<_, io::Error>(groups)
                })
            })
            .map_err(crate::Error::from)?;
        let allocated = groups.iter().map(|(_, slots)| slots.len()).sum::<usize>();
        if allocated != slots || total != allocated * block_bytes {
            return Err(PyValueError::new_err("host cache allocation is incomplete"));
        }
        Ok(Self {
            pool: Arc::new(Pool {
                slots: groups
                    .iter_mut()
                    .flat_map(|(_, slots)| std::mem::take(slots))
                    .collect(),
                block_bytes,
                core: Arc::downgrade(&copier.core),
            }),
        })
    }

    /// A writable, zero-copy view of one slot, for the filesystem and tensor adapters.
    fn slot(&self, index: usize) -> PyResult<PyHostSlot> {
        if index >= self.pool.slots.len() {
            return Err(PyValueError::new_err("host slot is outside the pool"));
        }
        Ok(PyHostSlot {
            pool: Arc::clone(&self.pool),
            slot: index,
            bytes: 0..self.pool.block_bytes,
        })
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyHostSlot {
    fn __len__(&self) -> usize {
        self.bytes.len()
    }

    /// Writable buffer access without copying.
    #[gen_stub(skip)]
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let (pointer, length) = {
            let slot = &slf.get().pool.slots[slf.get().slot];
            let mut buffer = slot
                .lock()
                .map_err(|error| io::Error::other(error.to_string()))?;
            let range = slf.get().bytes.clone();
            let mut bytes = buffer
                .slice_mut(range.clone())
                .and_then(|slice| slice.into_bytes())
                .map_err(crate::Error::from)?;
            (
                bytes.as_mut_ptr().cast::<c_void>(),
                ffi::Py_ssize_t::try_from(range.len())
                    .map_err(|_| PyValueError::new_err("host slot is too large for Python"))?,
            )
        };
        // SAFETY: the Python view holds `slf`, which keeps the pinned allocation alive.
        match unsafe { ffi::PyBuffer_FillInfo(view, slf.as_ptr(), pointer, length, 0, flags) } {
            0 => Ok(()),
            _ => Err(PyErr::fetch(slf.py())),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCopy {
    #[pyo3(signature = (timeout=Some(0.0)))]
    fn done(&self, py: Python<'_>, timeout: Option<f64>) -> PyResult<bool> {
        let timeout = timeout
            .map(Duration::try_from_secs_f64)
            .transpose()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(py
            .detach(|| self.0.done(timeout))
            .map_err(crate::Error::from)?)
    }

    fn wait(&self, py: Python<'_>) -> PyResult<()> {
        Ok(py
            .detach(|| self.0.done(None).map(drop))
            .map_err(crate::Error::from)?)
    }

    /// Wait until every runtime batch was submitted, without waiting for DMA completion.
    fn submitted(&self, py: Python<'_>) -> PyResult<()> {
        Ok(py
            .detach(|| {
                let submitted = self.0.wait_submitted();
                if let Err(error) = submitted {
                    // A later chunk can fail to enqueue after earlier chunks have started. Keep the
                    // pinned slots alive until every already-enqueued DMA has drained.
                    if let Err(drain_error) = self.0.done(None) {
                        return Err(io::Error::new(
                            error.kind(),
                            format!(
                                "{error}; draining earlier cache copies also failed: {drain_error}"
                            ),
                        ));
                    }
                    return Err(error);
                }
                Ok(())
            })
            .map_err(crate::Error::from)?)
    }
}

impl Queue {
    fn start(core: Arc<Core>) -> io::Result<Self> {
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
            sender: Some(sender),
            workers,
        })
    }

    fn send(&self, chunk: Chunk) -> io::Result<()> {
        self.sender
            .as_ref()
            .ok_or_else(|| io::Error::other("the cache copy queue is shut down"))?
            .send(chunk)
            .map_err(|_| io::Error::other("the cache copy workers stopped"))
    }

    fn shutdown(&mut self) {
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
    fn split(
        pages: Vec<(usize, Vec<(usize, u32)>)>,
        pool: &Arc<Pool>,
        loading: bool,
        completion: &Arc<Completion>,
    ) -> Vec<Self> {
        let chunks = Self::groups(pages);
        if chunks.is_empty() {
            completion.mark_submitted(Ok(()));
            completion.finish(Ok(()));
            return Vec::new();
        }
        let job = Arc::new(Job {
            left: AtomicUsize::new(chunks.len()),
            submit_left: AtomicUsize::new(chunks.len()),
            failure: Mutex::new(None),
            completion: Arc::clone(completion),
        });
        chunks
            .into_iter()
            .map(|pages| Self {
                pages,
                pool: Arc::clone(pool),
                loading,
                job: Arc::clone(&job),
            })
            .collect()
    }

    fn groups(pages: Vec<(usize, Vec<(usize, u32)>)>) -> Vec<Vec<TransferPage>> {
        let mut chunks =
            pages
                .into_iter()
                .fold(Vec::<Vec<_>>::new(), |mut chunks, (slot, mut routes)| {
                    routes.sort_unstable_by_key(|&(group, _)| group);
                    match chunks.last_mut() {
                        Some(last) if last.len() < COPY_CHUNK_BLOCKS => {
                            last.push(TransferPage { slot, routes });
                        }
                        _ => chunks.push(vec![TransferPage { slot, routes }]),
                    }
                    chunks
                });
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
        self.job.finish(result);
    }

    fn transfer(&self, executor: &tokio::runtime::Runtime, core: &Core) -> io::Result<()> {
        let mut hosts = self
            .pages
            .iter()
            .map(|page| {
                self.pool.slots[page.slot]
                    .lock()
                    .map_err(|error| io::Error::other(error.to_string()))
            })
            .collect::<io::Result<Vec<_>>>()?;
        let mut route_layers = Vec::new();
        let mut aliases = Vec::new();
        for (page_index, page) in self.pages.iter().enumerate() {
            for &(group, block) in &page.routes {
                for &layer_id in &core.groups[group] {
                    let layer = &core.layers[layer_id];
                    let start = usize::try_from(block)
                        .map_err(io::Error::other)?
                        .checked_mul(layer.page_bytes)
                        .ok_or_else(|| io::Error::other("device page offset overflows"))?;
                    let end = start
                        .checked_add(layer.page_bytes)
                        .ok_or_else(|| io::Error::other("device page end overflows"))?;
                    aliases.push(layer.buffer.alias(start..end).map_err(io::Error::other)?);
                    route_layers.push((page_index, group, layer_id));
                }
            }
        }

        if self.loading {
            let mut destinations = aliases
                .iter_mut()
                .map(|buffer| buffer.slice_mut(.., ..).map_err(io::Error::other))
                .collect::<io::Result<Vec<_>>>()?;
            let sources = route_layers
                .iter()
                .map(|&(page_index, _, layer_id)| {
                    let layer = &core.layers[layer_id];
                    let length = layer
                        .page_bytes
                        .checked_mul(core.chips)
                        .ok_or_else(|| io::Error::other("host page size overflows"))?;
                    let end = layer
                        .host_offset
                        .checked_add(length)
                        .ok_or_else(|| io::Error::other("host page end overflows"))?;
                    hosts[page_index]
                        .slice(layer.host_offset..end)
                        .map_err(io::Error::other)
                })
                .collect::<io::Result<Vec<_>>>()?;
            let pairs = sources
                .into_iter()
                .zip(destinations.drain(..))
                .collect::<Vec<_>>();
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
            let mut destinations_by_page = Vec::with_capacity(self.pages.len());
            for (page, host) in self.pages.iter().zip(&mut hosts) {
                // Split one host slot into disjoint layer ranges, including every routed group.
                let routes = page
                    .routes
                    .iter()
                    .flat_map(|&(group, _)| {
                        core.groups[group]
                            .iter()
                            .map(move |&layer_id| (group, layer_id))
                    })
                    .map(|(group, layer_id)| {
                        let layer = &core.layers[layer_id];
                        let length = layer
                            .page_bytes
                            .checked_mul(core.chips)
                            .ok_or_else(|| io::Error::other("host page size overflows"))?;
                        Ok((group, layer_id, layer.host_offset, length))
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                let mut routes = routes;
                routes.sort_unstable_by_key(|&(_, _, offset, _)| offset);
                let mut rest = host.slice_mut(..).map_err(io::Error::other)?;
                let mut cursor = 0;
                let mut by_layer = BTreeMap::new();
                for (group, layer_id, offset, length) in routes {
                    let (before, tail) =
                        rest.split_at(offset - cursor).map_err(io::Error::other)?;
                    drop(before);
                    let (page, tail) = tail.split_at(length).map_err(io::Error::other)?;
                    by_layer.insert((group, layer_id), page);
                    rest = tail;
                    cursor = offset + length;
                }
                destinations_by_page.push(by_layer);
            }
            let destinations = route_layers
                .iter()
                .map(|&(page_index, group, layer_id)| {
                    destinations_by_page[page_index]
                        .remove(&(group, layer_id))
                        .ok_or_else(|| io::Error::other("missing host layer range"))
                })
                .collect::<io::Result<Vec<_>>>()?;
            let sources = aliases
                .iter()
                .map(|buffer| buffer.slice(.., ..).map_err(io::Error::other))
                .collect::<io::Result<Vec<_>>>()?;
            let pairs = sources.into_iter().zip(destinations).collect::<Vec<_>>();
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
    fn check(
        &mut self,
        reservations: &BTreeSet<TransferKey>,
        timeout: Option<Duration>,
    ) -> io::Result<()> {
        for reservation in reservations {
            if let Some(copy) = self.0.get(reservation) {
                if !copy.done(timeout)? {
                    return Err(io::Error::new(
                        io::ErrorKind::WouldBlock,
                        "cache block has an unfinished transfer",
                    ));
                }
                self.0.remove(reservation);
            }
        }
        Ok(())
    }

    fn drain(&mut self) -> io::Result<()> {
        let mut first_error = None;
        for copy in self.0.values() {
            if let Err(error) = copy.done(None) {
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

    fn wait_submitted(&self) -> io::Result<()> {
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

    fn done(&self, timeout: Option<Duration>) -> io::Result<bool> {
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
