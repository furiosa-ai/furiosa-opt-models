//! Python bindings for pinned host-cache slots and transfer submission.

use std::collections::{BTreeMap, HashSet};
use std::ffi::c_int;
use std::io;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use furiosa_opt_std::prelude::HostBuffer;
use pyo3::exceptions::{PyBlockingIOError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::Context;
use crate::tensor::{HbmBuffer as PyHbmBuffer, HostBuffer as PyHostBuffer};

mod huge;
mod queue;
mod staging;

use queue::{
    Chunk, Completion, Core, HostBacking, Layer, Pdma, Pool, Queue, Slot, TransferKey, Transfers,
};

const ALLOCATORS: usize = 8;
const ALLOCATION_GROUP_BYTES: usize = 256 << 20;
static NEXT_POOL_ID: AtomicU64 = AtomicU64::new(1);

/// Python handle for asynchronous device-cache copies. Submit calls from one caller at a time;
/// returned `Copy` handles can be waited or queried from worker threads.
///
/// `copy_in` and `copy_out` return a `Copy` handle without waiting for its DMA. The caller owns
/// host-slot and device-page lifetimes: before submitting work that reuses either, it must wait
/// for the earlier `Copy` to finish. An overlapping submission raises `BlockingIOError` and queues
/// no work. Python serializes queue mutation under the GIL; completions can move to framework
/// worker threads.
/// Layer groups are numbered contiguously from zero.
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

/// A pool of page-locked host pages for device transfers and external host I/O. Multiple pools can
/// share one copier when consumers need independent capacities or lifetimes. Without `backing`,
/// slots are separately allocated. A supplied backing aliases one contiguous page-major region.
#[gen_stub_pyclass]
#[pyclass(name = "HostCache", module = "engine", frozen)]
pub(crate) struct PyHostCache {
    pool: Arc<Pool>,
}

/// A handle to one pinned host page.
#[gen_stub_pyclass]
#[pyclass(name = "HostSlot", module = "engine", frozen)]
#[derive(Clone)]
pub(crate) struct PyHostSlot {
    pool: Arc<Pool>,
    slot: usize,
    bytes: Range<usize>,
}

/// A writable pinned-buffer lease. Keep it alive through CPU or external asynchronous I/O; the
/// copier cannot use this page until the lease drops. Multiple leases may coexist, so adapters
/// must serialize their own users of a slot.
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
        self.slot.pool.slots[self.slot.slot].access.release_lease();
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
        let (chips, pdma): (usize, Pdma) = {
            let context = Context::lock(&context, py);
            (
                usize::from(context.device.topology().chips),
                context.device.pdma.clone(),
            )
        };
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
        if core_groups
            .keys()
            .copied()
            .enumerate()
            .any(|(expected, actual)| expected != actual)
        {
            return Err(PyValueError::new_err(
                "cache group IDs must be contiguous starting at zero",
            ));
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
            pdma,
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
    /// `pages` names host slots and their `(group, device_block)` routes. The caller must wait for
    /// earlier copies that use any of these host slots or device pages. An in-flight overlap raises
    /// `BlockingIOError` and submits no work.
    /// Coalescing stages the whole call through per-chip pinned buffers and trades CPU copies and
    /// transient pinned memory for one batched runtime transfer.
    #[pyo3(signature = (pages, host, *, coalesce=false))]
    fn copy_out(
        &mut self,
        pages: Vec<(usize, Vec<(usize, u32)>)>,
        host: PyRef<'_, PyHostCache>,
        coalesce: bool,
    ) -> PyResult<PyCopy> {
        self.submit(pages, &host.pool, false, coalesce)
    }

    /// Submit host-to-device copies without waiting for their DMA to finish.
    ///
    /// `pages` names host slots and their `(group, device_block)` routes. The caller must wait for
    /// earlier copies that use any of these host slots or device pages. An in-flight overlap raises
    /// `BlockingIOError` and submits no work.
    /// Coalescing stages the whole call through per-chip pinned buffers and trades CPU copies and
    /// transient pinned memory for one batched runtime transfer.
    #[pyo3(signature = (pages, host, *, coalesce=false))]
    fn copy_in(
        &mut self,
        pages: Vec<(usize, Vec<(usize, u32)>)>,
        host: PyRef<'_, PyHostCache>,
        coalesce: bool,
    ) -> PyResult<PyCopy> {
        self.submit(pages, &host.pool, true, coalesce)
    }

    /// Wait for every submitted copy and stop the workers. Drop all host-slot access leases first.
    fn shutdown(&mut self, py: Python<'_>) -> PyResult<()> {
        let result = py.detach(|| {
            let result = self.pending.drain();
            self.queue.shutdown();
            result
        });
        result.map_err(crate::Error::from)?;
        Ok(())
    }
}

impl PyCopier {
    fn submit(
        &mut self,
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
        if !loading {
            let mut host_slots = HashSet::new();
            for (slot, _) in &pages {
                if !host_slots.insert(*slot) {
                    return Err(PyValueError::new_err(format!(
                        "cache store repeats host slot {slot}"
                    )));
                }
            }
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
            reservations.push(TransferKey::Host(pool.id, *slot));
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
        if self.pending.conflicts(&reservations) {
            return Err(PyBlockingIOError::new_err(
                "cache transfer overlaps an in-flight DMA; wait for the prior Copy before reusing its host slot or device page",
            ));
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
    /// `page_bytes` is per chip; `host_offset` is a byte offset; `group` is the zero-based cache
    /// group index.
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
    /// Allocate one pinned host-page pool for this copier. Other pools may share the copier; their
    /// slot indices and DMA reservations are independent. `backing`, when supplied, must be a
    /// pinned buffer of exactly `slots` times one host block's size. The pool retains that
    /// allocation and presents each host block as one slot. Before reusing this backing for a
    /// replacement pool, shut down its old copier and finish all copies and `HostSlot.access()`
    /// leases; distinct pools over one backing have independent slot-access gates.
    #[pyo3(signature = (copier, slots, *, backing=None))]
    fn new(
        py: Python<'_>,
        copier: PyRef<'_, PyCopier>,
        slots: usize,
        backing: Option<PyRef<'_, PyHostBuffer>>,
    ) -> PyResult<Self> {
        let block_bytes = copier.core.block_bytes;
        if slots == 0 {
            return Err(PyValueError::new_err("host cache needs at least one slot"));
        }
        let total = slots
            .checked_mul(block_bytes)
            .ok_or_else(|| PyValueError::new_err("host cache allocation size overflows"))?;
        let (pool_slots, backing_region) = if let Some(backing) = backing {
            if backing.bytes.len() != total {
                return Err(PyValueError::new_err(format!(
                    "host cache backing has {} bytes; expected {total}",
                    backing.bytes.len()
                )));
            }
            let region = HostBacking {
                memory: Arc::clone(&backing.memory),
                bytes: backing.bytes.clone(),
            };
            let memory = region
                .memory
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if !memory.is_pinned() {
                return Err(PyValueError::new_err(
                    "host cache backing must be page-locked",
                ));
            }
            drop(memory);
            ((0..slots).map(|_| Slot::backed()).collect(), Some(region))
        } else {
            let group_bytes = (ALLOCATION_GROUP_BYTES / block_bytes).max(1);
            let allocator_count = ((slots - 1) / group_bytes + 1).min(ALLOCATORS);
            let next = AtomicUsize::new(0);
            let allocate = || -> io::Result<Vec<(usize, Vec<HostBuffer>)>> {
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
                            Ok(buffer)
                        })
                        .collect::<io::Result<Vec<_>>>()?;
                    groups.push((first, pinned));
                }
                Ok(groups)
            };
            let mut groups = py
                .detach(|| {
                    std::thread::scope(|scope| {
                        let allocators = (0..allocator_count)
                            .map(|_| std::thread::Builder::new().spawn_scoped(scope, allocate))
                            .collect::<io::Result<Vec<_>>>()?;
                        let mut groups = Vec::new();
                        for allocator in allocators {
                            groups.extend(allocator.join().map_err(|_| {
                                io::Error::other("a host slot allocator panicked")
                            })??);
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
            let pool_slots = groups
                .iter_mut()
                .flat_map(|(_, buffers)| std::mem::take(buffers))
                .map(Slot::owned)
                .collect();
            (pool_slots, None)
        };
        let id = NEXT_POOL_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| PyValueError::new_err("host cache pool ID space exhausted"))?;
        Ok(Self {
            pool: Arc::new(Pool {
                id,
                slots: pool_slots,
                block_bytes,
                core: Arc::downgrade(&copier.core),
                backing: backing_region,
            }),
        })
    }

    /// Return an internal `HostBuffer` alias for retaining/registering the contiguous address range.
    ///
    /// Do not read or write through this view. Those accesses bypass per-slot exclusion; use
    /// `HostSlot.access()` for external I/O. This alias shares and retains the pool allocation and
    /// can be passed as `backing=` when creating its replacement after the Copier is shut down.
    ///
    /// Raises:
    ///     ValueError: This pool was allocated with separate slot buffers.
    fn _backing_buffer(&self) -> PyResult<PyHostBuffer> {
        self.pool
            .backing()
            .map(|backing| PyHostBuffer::shared(backing.memory.clone(), backing.bytes.clone()))
            .ok_or_else(|| PyValueError::new_err("host cache has no contiguous backing"))
    }

    /// Return a handle to one pinned host page. Call `access()` to lease its zero-copy buffer.
    ///
    /// The slot index is local to this pool.
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

    /// Lease the page for CPU or external I/O. The lease excludes Copier access to host-page bytes,
    /// but does not reserve a cache value for an entire transfer. Wait for relevant `Copy` handles
    /// before using or reusing that value when ordering or errors matter. This lease does not
    /// serialize other CPU or external users, and it does not report copy failures.
    fn access(&self, py: Python<'_>) -> PyResult<PyHostSlotAccess> {
        let slot = &self.pool.slots[self.slot];
        py.detach(|| slot.access.acquire_lease())
            .map_err(crate::Error::from)?;
        Ok(PyHostSlotAccess { slot: self.clone() })
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
        let host_slot = slot
            .pool
            .slots
            .get(slot.slot)
            .ok_or_else(|| PyValueError::new_err("host cache slot storage is unavailable"))?;
        if let Some(backing) = slot.pool.backing() {
            let slot_range = backing
                .slot_range(slot.slot, slot.pool.block_bytes)
                .map_err(crate::Error::from)?;
            let start = slot_range
                .start
                .checked_add(slot.bytes.start)
                .ok_or_else(|| PyValueError::new_err("host slot offset overflows"))?;
            let end = start
                .checked_add(slot.bytes.len())
                .filter(|&end| end <= slot_range.end)
                .ok_or_else(|| PyValueError::new_err("host slot exceeds its backing"))?;
            let mut memory = backing
                .memory
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            unsafe {
                fill_slot_buffer(slf.py(), slf.as_ptr(), &mut memory, start..end, view, flags)
            }
        } else {
            let mut memory = host_slot.access.lock_buffer().map_err(crate::Error::from)?;
            unsafe {
                fill_slot_buffer(
                    slf.py(),
                    slf.as_ptr(),
                    &mut memory,
                    slot.bytes.clone(),
                    view,
                    flags,
                )
            }
        }
    }
}

unsafe fn fill_slot_buffer(
    py: Python<'_>,
    object: *mut ffi::PyObject,
    buffer: &mut HostBuffer,
    range: Range<usize>,
    view: *mut ffi::Py_buffer,
    flags: c_int,
) -> PyResult<()> {
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

    /// Wait until every copy in the request completes, raising its transfer error if any. Waiting
    /// before resubmitting an overlapping host slot or device page satisfies the caller's reuse
    /// contract.
    fn wait(&self, py: Python<'_>) -> PyResult<()> {
        Ok(py
            .detach(|| self.0.done(None).map(drop))
            .map_err(crate::Error::from)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Weak;

    fn cache_from_backing(backing: &PyHostBuffer, slots: usize, block_bytes: usize) -> PyHostCache {
        let backing = HostBacking {
            memory: Arc::clone(&backing.memory),
            bytes: backing.bytes.clone(),
        };
        let host_slots = (0..slots).map(|_| Slot::backed()).collect();
        PyHostCache {
            pool: Arc::new(Pool {
                id: NEXT_POOL_ID.fetch_add(1, Ordering::Relaxed),
                slots: host_slots,
                block_bytes,
                core: Weak::new(),
                backing: Some(backing),
            }),
        }
    }

    fn buffer_address(buffer: &PyHostBuffer) -> usize {
        let mut memory = buffer
            .memory
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut bytes = memory
            .slice_mut(buffer.bytes.clone())
            .and_then(|slice| slice.into_bytes())
            .unwrap();
        bytes.as_mut_ptr() as usize
    }

    #[test]
    fn backing_view_retains_the_same_allocation_for_replacement_pools() {
        let source: PyHostBuffer = HostBuffer::zero(48).into();
        let address = buffer_address(&source);
        let first = cache_from_backing(&source, 3, 16);
        drop(source);

        let region = first._backing_buffer().unwrap();
        assert_eq!(region.bytes.len(), 48);
        assert_eq!(buffer_address(&region), address);
        drop(first);

        let replacement = cache_from_backing(&region, 3, 16);
        let replacement_region = replacement._backing_buffer().unwrap();
        assert_eq!(replacement_region.bytes.len(), 48);
        assert_eq!(buffer_address(&replacement_region), address);
    }

    #[test]
    fn separately_allocated_cache_has_no_whole_pool_backing() {
        let cache = PyHostCache {
            pool: Arc::new(Pool {
                id: 1,
                slots: vec![Slot::owned(HostBuffer::zero(16))],
                block_bytes: 16,
                core: Weak::new(),
                backing: None,
            }),
        };

        assert!(cache._backing_buffer().is_err());
    }

    #[test]
    fn python_backing_requires_page_lock_and_survives_copier_replacement() {
        crate::tensor::tests::run_on_device(
            c"
import ctypes

def address(buffer):
    return ctypes.addressof(ctypes.c_ubyte.from_buffer(memoryview(buffer)))

device_buffer = engine.HbmBuffer(device, 8)
def copier():
    layer = engine.CopierLayer(device_buffer, 8, 0, 0)
    return engine.Copier([layer], 8)

old_copier = copier()
source = engine.HostBuffer(16)

def check_pinned_backing(source_owner):
    source = source_owner.pop()
    old_cache = engine.HostCache(old_copier, 2, backing=source)
    del source
    region = old_cache._backing_buffer()
    base = address(region)
    assert len(region) == 16
    for slot in range(2):
        lease = old_cache.slot(slot).access()
        assert address(lease) == base + slot * 8
        del lease

    separate = engine.HostCache(old_copier, 1)
    assert len(separate.slot(0)) == 8
    try:
        separate._backing_buffer()
    except ValueError as error:
        assert str(error) == 'host cache has no contiguous backing', error
    else:
        raise AssertionError('separate slot allocations cannot provide one whole-pool region')

    payload = bytes(range(1, 9))
    lease = old_cache.slot(0).access()
    view = memoryview(lease)
    view[:] = payload
    view.release()
    copy_in = old_copier.copy_in([(0, [(0, 0)])], old_cache)
    assert not copy_in.done()
    del lease  # The pending DMA starts only after the slot-access lease is released.
    copy_in.wait()
    old_copier.copy_out([(1, [(0, 0)])], old_cache).wait()
    lease = old_cache.slot(1).access()
    assert bytes(memoryview(lease)) == payload
    del lease

    old_copier.copy_in([(0, [(0, 0)])], old_cache, coalesce=True).wait()
    old_copier.copy_out([(1, [(0, 0)])], old_cache, coalesce=True).wait()
    lease = old_cache.slot(1).access()
    assert bytes(memoryview(lease)) == payload
    del lease

    old_copier.shutdown()
    del old_cache, separate
    new_copier = copier()
    new_cache = engine.HostCache(new_copier, 2, backing=region)
    assert address(new_cache._backing_buffer()) == base
    new_copier.shutdown()

if source.is_pinned:
    source_owner = [source]
    source = None
    check_pinned_backing(source_owner)
else:
    try:
        engine.HostCache(old_copier, 2, backing=source)
    except ValueError as error:
        assert str(error) == 'host cache backing must be page-locked', error
    else:
        raise AssertionError('a pageable host buffer cannot back a DMA cache pool')
    old_copier.shutdown()
",
        );
    }
}
