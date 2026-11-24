//! Python bindings for pinned host-cache slots and transfer submission.

use std::collections::{BTreeMap, HashSet};
use std::ffi::c_int;
use std::io;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use furiosa_opt_std::prelude::HostBuffer;
use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::Context;
use crate::tensor::HbmBuffer as PyHbmBuffer;

mod huge;
mod queue;
mod staging;

use queue::{
    Chunk, Completion, Core, Layer, Pool, Queue, Slot, SlotAccess, TransferKey, Transfers,
};

const ALLOCATORS: usize = 8;
const ALLOCATION_GROUP_BYTES: usize = 256 << 20;

/// Thread-safe Python handle for asynchronous device-cache copies.
///
/// `copy_in` and `copy_out` return a `Copy` handle without waiting for its DMA. Reusing a host slot
/// or device page may first wait for conflicting earlier copies. Python serializes queue mutation
/// under the GIL; completions can move to framework worker threads.
#[gen_stub_pyclass]
#[pyclass(name = "Copier", module = "engine")]
pub(crate) struct PyCopier {
    core: Arc<Core>,
    queue: Queue,
    pending: Transfers,
}

/// One layer's byte layout in the device cache and each pinned host-cache block.
#[gen_stub_pyclass]
#[pyclass(name = "CopierLayer", module = "engine", frozen)]
pub(crate) struct PyCopierLayer {
    buffer: Py<PyHbmBuffer>,
    page_bytes: usize,
    host_offset: usize,
    group: usize,
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

/// Completion handle for asynchronous cache copies.
#[gen_stub_pyclass]
#[pyclass(name = "Copy", module = "engine", frozen)]
#[derive(Clone)]
pub(crate) struct PyCopy(Arc<Completion>);

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
    /// Each layer names its own page size (bytes per chip), byte offset in a host slot, and cache
    /// group, so one layer's metadata cannot accidentally drift from another's buffer.
    #[new]
    fn new(
        py: Python<'_>,
        layers: Vec<PyRef<'_, PyCopierLayer>>,
        host_block_bytes: usize,
    ) -> PyResult<Self> {
        let first = layers
            .first()
            .ok_or_else(|| PyValueError::new_err("cache copies require nonempty layer buffers"))?;
        if host_block_bytes == 0 {
            return Err(PyValueError::new_err(
                "host cache blocks must contain bytes",
            ));
        }
        let context = first.buffer.bind(py).borrow().context.clone();
        let chips = usize::from(Context::lock(&context, py).device.topology().chips);
        let mut core_layers = Vec::with_capacity(layers.len());
        let mut core_groups = BTreeMap::<usize, Vec<usize>>::new();
        let mut ranges = Vec::with_capacity(layers.len());
        for (index, description) in layers.iter().enumerate() {
            let layer = description.buffer.bind(py).borrow();
            let page = description.page_bytes;
            let offset = description.host_offset;
            let group = description.group;
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

    /// Submit device-to-host copies without waiting for their DMA to finish.
    ///
    /// `pages` names host slots and their `(group, device_block)` routes. Reusing a host slot or
    /// device page may first wait for conflicting earlier copies.
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

    /// Submit host-to-device copies without waiting for their DMA to finish.
    ///
    /// `pages` names host slots and their `(group, device_block)` routes. Reusing a host slot or
    /// device page may first wait for conflicting earlier copies.
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
        for (_, routes) in &mut pages {
            routes.sort_unstable_by_key(|&(group, _)| group);
        }
        if coalesce {
            if !loading {
                let mut host_slots = HashSet::new();
                for (slot, _) in &pages {
                    if !host_slots.insert(*slot) {
                        return Err(PyValueError::new_err(format!(
                            "cache transfer repeats host slot {slot}"
                        )));
                    }
                }
            }
        } else if let Some(pair) = pages.windows(2).find(|pair| pair[0].0 == pair[1].0) {
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
            if let Err(error) = self.queue.send(chunk) {
                for chunk in chunks {
                    chunk.fail(io::Error::other(error.to_string()));
                }
                let error = match completion.done(None) {
                    Ok(_) => error,
                    Err(drain_error) => io::Error::new(
                        error.kind(),
                        format!(
                            "{error}; draining earlier cache copies also failed: {drain_error}"
                        ),
                    ),
                };
                self.pending.remove(&reservations);
                return Err(crate::Error::from(error).into());
            }
        }
        for (key, completion) in chunk_completions {
            self.pending.insert(key, completion);
        }
        Ok(PyCopy(completion))
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCopierLayer {
    /// Describe one cache layer's device page and its location in each host block.
    ///
    /// `page_bytes` is per chip; `host_offset` is a byte offset; `group` names the layer's cache
    /// group.
    #[pyo3(signature = (buffer, page_bytes, host_offset, group))]
    #[new]
    fn new(buffer: Py<PyHbmBuffer>, page_bytes: usize, host_offset: usize, group: usize) -> Self {
        Self {
            buffer,
            page_bytes,
            host_offset,
            group,
        }
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
    /// Return whether the copy finished before `timeout` elapsed.
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

    /// Wait until every copy in the request completes, raising its transfer error if any.
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
