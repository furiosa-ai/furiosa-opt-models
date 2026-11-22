//! The pinned host cache and its asynchronous HBM transfer queue.
//!
//! A slot holds selected layer bytes for one cache block. Six workers submit transfers in chunks
//! of two blocks, each chunk batching all routed layer copies into one runtime read or write.

use std::collections::{BTreeMap, HashSet};
use std::ffi::c_int;
use std::io;
use std::ops::{DerefMut, Range};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak, mpsc};
use std::time::{Duration, Instant};

use furiosa_opt_std::prelude::{self, HostBuffer};
use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::tensor::HbmBuffer as PyHbmBuffer;
use crate::{Context, Shared};

mod huge;
mod staging;

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
    group_blocks: Vec<usize>,
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
#[derive(Clone)]
pub(crate) struct PyHostSlot {
    pool: Arc<Pool>,
    slot: usize,
    bytes: Range<usize>,
}

/// A host slot lease held for the lifetime of exported Python buffers.
#[gen_stub_pyclass]
#[pyclass(name = "HostSlotAccess", module = "engine", frozen)]
pub(crate) struct PyHostSlotAccess {
    slot: PyHostSlot,
}

/// Completion of a queued cache copy.
#[gen_stub_pyclass]
#[pyclass(name = "Copy", module = "engine", frozen)]
#[derive(Clone)]
pub(crate) struct PyCopy(Arc<Completion>);

struct Pool {
    slots: Vec<Slot>,
    block_bytes: usize,
    core: Weak<Core>,
}

struct Slot {
    buffer: Mutex<HostBuffer>,
    access: SlotAccess,
}

// Python buffer leases and DMA share one gate, so DMA cannot race a live host view.
#[derive(Default)]
struct SlotAccess {
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

struct Queue {
    sender: Option<mpsc::Sender<Chunk>>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

struct Chunk {
    pages: Vec<TransferPage>,
    pool: Arc<Pool>,
    loading: bool,
    coalesced: bool,
    completion: Arc<ChunkCompletion>,
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

struct ChunkCompletion {
    result: Mutex<Option<Result<(), String>>>,
    changed: Condvar,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum TransferKey {
    Device(usize, u32),
    Host(usize),
}

#[derive(Default)]
// Each key retains only DMA chunks touching that page, not their whole framework job.
struct Transfers(BTreeMap<TransferKey, Vec<Arc<ChunkCompletion>>>);

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

    fn acquire_view(&self) -> io::Result<()> {
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

    fn try_acquire_view(&self) -> io::Result<bool> {
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

    fn release_view(&self) {
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

impl Drop for PyHostSlotAccess {
    fn drop(&mut self) {
        self.slot.pool.slots[self.slot.slot].access.release_view();
    }
}

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
        let group_blocks = groups
            .iter()
            .map(|layers| {
                layers
                    .iter()
                    .map(|&index| core_layers[index].blocks)
                    .min()
                    .unwrap_or(0)
            })
            .collect();
        let core = Arc::new(Core {
            context,
            layers: core_layers,
            groups,
            group_blocks,
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
    /// Coalescing stages the whole call through per-chip pinned buffers and trades CPU copies and
    /// transient pinned memory for one batched runtime transfer.
    #[pyo3(signature = (pages, host, *, coalesce=false))]
    fn copy_out(
        &mut self,
        py: Python<'_>,
        pages: Vec<(usize, Vec<(usize, u32)>)>,
        host: PyRef<'_, PyHostCache>,
        coalesce: bool,
    ) -> PyResult<PyCopy> {
        self.submit(py, pages, &host.pool, false, coalesce)
    }

    /// Submit host-to-device copies for host slots and their `(group, device_block)` routes.
    /// Coalescing stages the whole call through per-chip pinned buffers and trades CPU copies and
    /// transient pinned memory for one batched runtime transfer.
    #[pyo3(signature = (pages, host, *, coalesce=false))]
    fn copy_in(
        &mut self,
        py: Python<'_>,
        pages: Vec<(usize, Vec<(usize, u32)>)>,
        host: PyRef<'_, PyHostCache>,
        coalesce: bool,
    ) -> PyResult<PyCopy> {
        self.submit(py, pages, &host.pool, true, coalesce)
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
        py: Python<'_>,
        mut pages: Vec<(usize, Vec<(usize, u32)>)>,
        pool: &Arc<Pool>,
        loading: bool,
        coalesce: bool,
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
        pages.sort_unstable_by_key(|(slot, _)| *slot);
        for (_, routes) in &mut pages {
            routes.sort_unstable_by_key(|&(group, _)| group);
        }
        if let Some(pair) = pages.windows(2).find(|pair| pair[0].0 == pair[1].0) {
            return Err(PyValueError::new_err(format!(
                "cache transfer repeats host slot {}",
                pair[0].0
            )));
        }

        let mut reservations = Vec::new();
        for (slot, routes) in &pages {
            if *slot >= pool.slots.len() {
                return Err(PyValueError::new_err(format!(
                    "cache transfer names host slot {slot}, but host cache has {} slots",
                    pool.slots.len()
                )));
            }
            if routes.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "cache transfer for host slot {slot} has no device routes"
                )));
            }
            reservations.push(TransferKey::Host(*slot));
            if let Some(pair) = routes.windows(2).find(|pair| pair[0].0 == pair[1].0) {
                return Err(PyValueError::new_err(format!(
                    "cache transfer for host slot {slot} repeats group {}",
                    pair[0].0
                )));
            }
            for &(group, block) in routes {
                let blocks = self.core.group_blocks.get(group).ok_or_else(|| {
                    PyValueError::new_err(format!("cache transfer names unknown group {group}"))
                })?;
                if block as usize >= *blocks {
                    return Err(PyValueError::new_err(format!(
                        "cache transfer for host slot {slot} names block {block} outside group {group}"
                    )));
                }
                reservations.push(TransferKey::Device(group, block));
            }
        }
        reservations.sort_unstable();
        for pair in reservations.windows(2) {
            if pair[0] != pair[1] || !loading {
                continue;
            }
            if let TransferKey::Device(group, block) = pair[0] {
                return Err(PyValueError::new_err(format!(
                    "cache load writes group {group} block {block} more than once"
                )));
            }
        }
        reservations.dedup();
        // Reuse waits for only prior DMA batches that share a host slot or device page.
        let previous = self.pending.conflicts(&reservations);
        if !previous.is_empty() {
            let result = py.detach(move || {
                let mut first_error = None;
                for copy in previous {
                    if let Err(error) = copy.wait() {
                        first_error.get_or_insert(error);
                    }
                }
                first_error.map_or(Ok(()), Err)
            });
            self.pending.remove(&reservations);
            result.map_err(crate::Error::from)?;
        }

        let completion = Arc::new(Completion::default());
        let (chunks, chunk_completions) = Chunk::split(pages, pool, loading, coalesce, &completion);
        let mut chunks = chunks.into_iter();
        while let Some(chunk) = chunks.next() {
            if let Err((error, failed_chunk)) = self.queue.send(chunk) {
                failed_chunk.fail(io::Error::other(error.to_string()));
                for chunk in chunks {
                    chunk.fail(io::Error::other(error.to_string()));
                }
                let _ = completion.done(None);
                self.pending.remove(&reservations);
                return Err(crate::Error::from(error).into());
            }
        }
        for (key, completion) in chunk_completions {
            let entries = self.pending.0.entry(key).or_default();
            if !entries.iter().any(|entry| Arc::ptr_eq(entry, &completion)) {
                entries.push(completion);
            }
        }
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
        let total = slots
            .checked_mul(block_bytes)
            .ok_or_else(|| PyValueError::new_err("host cache allocation size overflows"))?;
        let group_bytes = (ALLOCATION_GROUP_BYTES / block_bytes).max(1);
        let next = AtomicUsize::new(0);
        let allocate = || -> io::Result<Vec<(usize, Vec<Slot>)>> {
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
                        Ok(Slot {
                            buffer: Mutex::new(buffer),
                            access: SlotAccess::default(),
                        })
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

    /// Wait for DMA, then lease the slot while this object or its exported buffers remain alive.
    fn access(&self, py: Python<'_>) -> PyResult<PyHostSlotAccess> {
        let slot = &self.pool.slots[self.slot];
        py.detach(|| slot.access.acquire_view())
            .map_err(crate::Error::from)?;
        Ok(PyHostSlotAccess { slot: self.clone() })
    }

    /// Writable buffer access without copying.
    #[gen_stub(skip)]
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let host_slot = &slf.get().pool.slots[slf.get().slot];
        if !host_slot
            .access
            .try_acquire_view()
            .map_err(crate::Error::from)?
        {
            return Err(PyBufferError::new_err(
                "host slot has pending DMA; call access() to wait for it",
            ));
        }
        match unsafe {
            fill_slot_buffer(
                slf.py(),
                slf.as_ptr(),
                &host_slot.buffer,
                slf.get().bytes.clone(),
                view,
                flags,
            )
        } {
            Ok(()) => {
                // SAFETY: `view` remains live until `__releasebuffer__` frees this token.
                unsafe {
                    (*view).internal = Box::into_raw(Box::new(slf.get().slot)).cast();
                }
                Ok(())
            }
            Err(error) => {
                host_slot.access.release_view();
                Err(error)
            }
        }
    }

    #[gen_stub(skip)]
    unsafe fn __releasebuffer__(slf: Bound<'_, Self>, view: *mut ffi::Py_buffer) {
        // SAFETY: `__getbuffer__` stores one slot index in this view's internal field.
        let internal = unsafe { (*view).internal };
        if internal.is_null() {
            return;
        }
        // SAFETY: this token was allocated by `__getbuffer__` and is released exactly once here.
        let slot = unsafe { *Box::from_raw(internal.cast::<usize>()) };
        unsafe { (*view).internal = std::ptr::null_mut() };
        slf.get().pool.slots[slot].access.release_view();
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyHostSlotAccess {
    fn __len__(&self) -> usize {
        self.slot.bytes.len()
    }

    #[gen_stub(skip)]
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let slot = &slf.get().slot;
        let host_slot = &slot.pool.slots[slot.slot];
        unsafe {
            fill_slot_buffer(
                slf.py(),
                slf.as_ptr(),
                &host_slot.buffer,
                slot.bytes.clone(),
                view,
                flags,
            )
        }
    }
}

unsafe fn fill_slot_buffer(
    py: Python<'_>,
    object: *mut ffi::PyObject,
    buffer: &Mutex<HostBuffer>,
    range: Range<usize>,
    view: *mut ffi::Py_buffer,
    flags: c_int,
) -> PyResult<()> {
    let mut buffer = buffer
        .lock()
        .map_err(|error| io::Error::other(error.to_string()))?;
    let mut bytes = buffer
        .slice_mut(range.clone())
        .and_then(|slice| slice.into_bytes())
        .map_err(crate::Error::from)?;
    let length = ffi::Py_ssize_t::try_from(range.len())
        .map_err(|_| PyValueError::new_err("host slot is too large for Python"))?;
    // SAFETY: caller holds a slot lease and `object` keeps the pinned allocation alive.
    match unsafe {
        ffi::PyBuffer_FillInfo(view, object, bytes.as_mut_ptr().cast(), length, 0, flags)
    } {
        0 => Ok(()),
        _ => Err(PyErr::fetch(py)),
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

    fn send(&self, chunk: Chunk) -> Result<(), (io::Error, Chunk)> {
        let Some(sender) = self.sender.as_ref() else {
            return Err((io::Error::other("the cache copy queue is shut down"), chunk));
        };
        sender
            .send(chunk)
            .map_err(|error| (io::Error::other("the cache copy workers stopped"), error.0))
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
            let pages = pages
                .into_iter()
                .map(|(slot, routes)| TransferPage { slot, routes })
                .collect::<Vec<_>>();
            return if pages.is_empty() {
                Vec::new()
            } else {
                vec![pages]
            };
        }
        pages
            .into_iter()
            .fold(Vec::<Vec<_>>::new(), |mut chunks, (slot, routes)| {
                match chunks.last_mut() {
                    Some(last) if last.len() < COPY_CHUNK_BLOCKS => {
                        last.push(TransferPage { slot, routes });
                    }
                    _ => chunks.push(vec![TransferPage { slot, routes }]),
                }
                chunks
            })
    }

    fn run(self, executor: &tokio::runtime::Runtime, core: &Core) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.transfer(executor, core)
        }))
        .unwrap_or_else(|_| Err(io::Error::other("cache copy worker panicked")));
        self.completion.finish(clone_result(&result));
        self.job.finish(result);
    }

    fn fail(self, error: io::Error) {
        self.completion
            .finish(Err(io::Error::new(error.kind(), error.to_string())));
        self.job.finish(Err(error));
    }

    fn transfer(&self, executor: &tokio::runtime::Runtime, core: &Core) -> io::Result<()> {
        let mut hosts = self
            .pages
            .iter()
            .map(|page| self.pool.slots[page.slot].acquire_dma())
            .collect::<io::Result<Vec<_>>>()?;
        if self.coalesced {
            return self.transfer_coalesced(executor, core, &mut hosts);
        }
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
                    let source = hosts[item.page].slice(..).map_err(io::Error::other)?;
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
                    let mut destination = hosts[item.page]
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
    fn conflicts(&self, reservations: &[TransferKey]) -> Vec<Arc<ChunkCompletion>> {
        let mut seen = HashSet::new();
        reservations
            .iter()
            .filter_map(|reservation| self.0.get(reservation))
            .flatten()
            .filter(|copy| seen.insert(Arc::as_ptr(copy) as usize))
            .cloned()
            .collect()
    }

    fn remove(&mut self, reservations: &[TransferKey]) {
        for reservation in reservations {
            self.0.remove(reservation);
        }
    }

    fn drain(&mut self) -> io::Result<()> {
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

impl ChunkCompletion {
    fn finish(&self, result: io::Result<()>) {
        *self
            .result
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) =
            Some(result.map_err(|error| error.to_string()));
        self.changed.notify_all();
    }

    fn wait(&self) -> io::Result<()> {
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
    fn host_slot_views_and_dma_exclude_each_other() {
        let access = SlotAccess::default();
        access.acquire_dma().unwrap();
        assert!(!access.try_acquire_view().unwrap());
        access.release_dma();
        assert!(access.try_acquire_view().unwrap());
        access.release_view();
    }
}
