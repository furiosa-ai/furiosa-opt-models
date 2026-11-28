//! Python bindings for pinned host-cache slots and transfer submission.

use std::collections::HashSet;
use std::ffi::c_int;
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use furiosa_opt_std::prelude::HostBuffer;
use pyo3::exceptions::{PyBlockingIOError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyMemoryView;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use crate::Context;
use crate::decoder::Cache;

mod huge;
mod layout;
mod queue;
mod staging;

use layout::SlotLayout;
use queue::{Chunk, Completion, Core, DevicePage, Layer, Pool, Queue, Slot, Transfers};

const ALLOCATORS: usize = 8;
const ALLOCATION_GROUP_BYTES: usize = 256 << 20;

/// Python handle for asynchronous device-cache copies. Submit calls from one caller at a time;
/// returned `Copy` handles can be waited or queried from worker threads.
///
/// `copy_in` and `copy_out` return a `Copy` handle without waiting for its DMA. The caller owns
/// host-slot and device-page lifetimes: before submitting work that reuses either, it must wait
/// for the earlier `Copy` to finish. An in-flight device-page overlap raises `BlockingIOError` and
/// queues no work; host-slot reuse is serialized by its per-slot gate. A failed copy leaves its
/// device pages unavailable for this copier's lifetime.
/// Coalescing gathers or scatters through temporary pinned buffers to reduce DMA commands.
/// Number groups from zero in the order supplied to the constructor.
#[gen_stub_pyclass]
#[pyclass(name = "Copier", module = "engine")]
pub(crate) struct PyCopier {
    core: Arc<Core>,
    layout: Arc<SlotLayout>,
    queue: Queue,
    pending: Transfers,
}

/// A pool that owns page-locked host slots for transfers and host I/O. Pools on the same copier
/// share its device-page reservations and hold independent host-slot gates.
#[gen_stub_pyclass]
#[pyclass(name = "HostCache", module = "engine", frozen)]
pub(crate) struct PyHostCache {
    pool: Arc<Pool>,
}

/// The buffer exporter retained by a slot's memoryviews until the last view is released.
#[pyclass(frozen)]
struct HostSlotAccess {
    pool: Arc<Pool>,
    index: usize,
}

/// Completion handle for asynchronous cache copies.
#[gen_stub_pyclass]
#[pyclass(name = "Copy", module = "engine", frozen)]
#[derive(Clone)]
pub(crate) struct PyCopy(Arc<Completion>);

impl Drop for HostSlotAccess {
    fn drop(&mut self) {
        self.pool.slots[self.index].release_lease();
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCopier {
    /// Build a transfer queue over an allocated decoder cache.
    ///
    /// Each group lists the stable layer IDs that move together for one logical block. Group and
    /// layer order determine the packed host slot layout. A layer can appear in at most one group.
    #[new]
    fn new(py: Python<'_>, cache: &Cache, groups: Vec<Vec<usize>>) -> PyResult<Self> {
        let chips = usize::from(Context::lock(&cache.context, py).device.topology().chips);
        let (layout, slot_bytes) = layout::pack(cache.inner.layer_page_bytes(), &groups, chips)
            .map_err(PyValueError::new_err)?;
        let mut layers = Vec::new();
        let mut layer_ranges = Vec::new();
        let mut packed_groups = Vec::with_capacity(layout.len());
        for group in layout {
            let first = layers.len();
            for entry in group {
                let buffer = cache.inner.layers()[entry.id]
                    .alias(..)
                    .map_err(crate::Error::from)?;
                layers.push(Layer {
                    buffer,
                    page_bytes: entry.page_bytes,
                });
                layer_ranges.push(entry.host_range);
            }
            packed_groups.push((first..layers.len()).collect());
        }
        let core = Arc::new(Core {
            context: Arc::clone(&cache.context),
            layers,
            groups: packed_groups,
            blocks: cache.inner.blocks(),
            chips,
        });
        let queue = Queue::start(Arc::clone(&core)).map_err(crate::Error::from)?;
        Ok(Self {
            core,
            layout: Arc::new(SlotLayout {
                slot_bytes,
                layer_ranges,
            }),
            queue,
            pending: Transfers::default(),
        })
    }

    /// Bytes in one host slot.
    #[getter]
    fn slot_bytes(&self) -> usize {
        self.layout.slot_bytes
    }

    /// Host-slot byte offsets and lengths in transfer group and layer order.
    #[getter]
    fn layer_ranges(&self) -> Vec<Vec<(usize, usize)>> {
        self.core
            .groups
            .iter()
            .map(|group| {
                group
                    .iter()
                    .map(|&layer| {
                        let range = &self.layout.layer_ranges[layer];
                        (range.start, range.len())
                    })
                    .collect()
            })
            .collect()
    }

    /// Queue device-to-host copies for host slots and their `(group, device_block)` routes.
    #[pyo3(signature = (pages, host, *, coalesce=false))]
    fn copy_out(
        &mut self,
        pages: Vec<(usize, Vec<(usize, u32)>)>,
        host: PyRef<'_, PyHostCache>,
        coalesce: bool,
    ) -> PyResult<PyCopy> {
        self.submit(pages, &host.pool, false, coalesce)
    }

    /// Queue host-to-device copies for host slots and their `(group, device_block)` routes.
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
        if !pool
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

        let mut device_pages = Vec::new();
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
            if let Some(pair) = routes.windows(2).find(|pair| pair[0].0 == pair[1].0) {
                return Err(PyValueError::new_err(format!(
                    "cache transfer for host slot {slot} repeats group {}",
                    pair[0].0
                )));
            }
            for &(group, block) in routes {
                if group >= self.core.groups.len() {
                    return Err(PyValueError::new_err(format!(
                        "cache transfer names unknown group {group}"
                    )));
                }
                if block as usize >= self.core.blocks {
                    return Err(PyValueError::new_err(format!(
                        "cache transfer for host slot {slot} names block {block} outside group {group}"
                    )));
                }
                device_pages.push(DevicePage(group, block));
            }
        }
        device_pages.sort_unstable();
        for pair in device_pages.windows(2) {
            if pair[0] != pair[1] || !loading {
                continue;
            }
            let DevicePage(group, block) = pair[0];
            return Err(PyValueError::new_err(format!(
                "cache load writes group {group} block {block} more than once"
            )));
        }
        device_pages.dedup();
        if self
            .pending
            .conflicts(&device_pages)
            .map_err(crate::Error::from)?
        {
            return Err(PyBlockingIOError::new_err(
                "cache transfer overlaps an in-flight device page; wait for the prior Copy before reusing it",
            ));
        }

        let completion = Arc::new(Completion::default());
        let chunks = Chunk::split(pages, pool, loading, coalesce, &completion);
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
                return Err(crate::Error::from(error).into());
            }
        }
        self.pending.insert(&device_pages, Arc::clone(&completion));
        Ok(PyCopy(completion))
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyHostCache {
    /// Allocate pinned host slots for the copier's page layout.
    #[new]
    fn new(py: Python<'_>, copier: &PyCopier, slots: usize) -> PyResult<Self> {
        let slot_bytes = copier.layout.slot_bytes;
        if slots == 0 {
            return Err(PyValueError::new_err("host cache needs at least one slot"));
        }
        let total = slots
            .checked_mul(slot_bytes)
            .ok_or_else(|| PyValueError::new_err("host cache allocation size overflows"))?;
        let slots_per_group = (ALLOCATION_GROUP_BYTES / slot_bytes).max(1);
        let allocator_count = ((slots - 1) / slots_per_group + 1).min(ALLOCATORS);
        let next = AtomicUsize::new(0);
        let allocate = || -> io::Result<Vec<(usize, Vec<HostBuffer>)>> {
            let mut groups = Vec::new();
            loop {
                let first = next.fetch_add(slots_per_group, Ordering::Relaxed);
                if first >= slots {
                    break;
                }
                let count = slots_per_group.min(slots - first);
                // Faulting and releasing THPs compacts memory before the dma-heap allocates slots.
                drop(huge::Pages::fault(count * slot_bytes)?);
                let pinned = (0..count)
                    .map(|_| {
                        let buffer = HostBuffer::zero(slot_bytes);
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
        if allocated != slots || total != allocated * slot_bytes {
            return Err(PyValueError::new_err("host cache allocation is incomplete"));
        }
        let slots = groups
            .iter_mut()
            .flat_map(|(_, buffers)| std::mem::take(buffers))
            .map(Slot::new)
            .collect();
        Ok(Self {
            pool: Arc::new(Pool {
                slots,
                layout: Arc::clone(&copier.layout),
                core: Arc::downgrade(&copier.core),
            }),
        })
    }

    /// Bytes in one slot.
    #[getter]
    fn slot_bytes(&self) -> usize {
        self.pool.layout.slot_bytes
    }

    /// Borrow one slot as a writable memoryview. DMA waits until every view of this borrow is
    /// released. Coordinate other host users yourself and wait for relevant copies before reuse.
    #[gen_stub(override_return_type(type_repr = "builtins.memoryview"))]
    fn access<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyMemoryView>> {
        let slot = self
            .pool
            .slots
            .get(index)
            .ok_or_else(|| PyValueError::new_err("host slot is outside the pool"))?;
        py.detach(|| slot.acquire_lease())
            .map_err(crate::Error::from)?;
        let lease = Bound::new(
            py,
            HostSlotAccess {
                pool: Arc::clone(&self.pool),
                index,
            },
        )?;
        PyMemoryView::from(lease.as_any())
    }
}

#[pymethods]
impl HostSlotAccess {
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let lease = slf.get();
        let length = ffi::Py_ssize_t::try_from(lease.pool.layout.slot_bytes)
            .map_err(|_| PyValueError::new_err("host slot is too large for Python"))?;
        lease.pool.slots[lease.index].with_buffer(|buffer| {
            let mut bytes = buffer
                .slice_mut(..)
                .and_then(|slice| slice.into_bytes())
                .map_err(crate::Error::from)?;
            // SAFETY: the exported view retains slf, which owns the pool and holds the slot lease.
            match unsafe {
                ffi::PyBuffer_FillInfo(
                    view,
                    slf.as_ptr(),
                    bytes.as_mut_ptr().cast(),
                    length,
                    0,
                    flags,
                )
            } {
                0 => Ok(()),
                _ => Err(PyErr::fetch(slf.py())),
            }
        })
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

    /// Wait for the whole request and raise its transfer error, if any.
    fn wait(&self, py: Python<'_>) -> PyResult<()> {
        Ok(py
            .detach(|| self.0.done(None).map(drop))
            .map_err(crate::Error::from)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use furiosa_opt_engine::decoder::{
        Cache as DeviceCache, CacheSpec, Group, Kind, PageFormat, Pool as Format,
    };
    use pyo3::types::PyDict;
    use std::sync::Weak;

    #[test]
    fn host_view_keeps_slot_alive() {
        let cache = PyHostCache {
            pool: Arc::new(Pool {
                slots: vec![Slot::new(HostBuffer::zero(8))],
                layout: Arc::new(SlotLayout {
                    slot_bytes: 8,
                    layer_ranges: Vec::new(),
                }),
                core: Weak::new(),
            }),
        };
        let pool = Arc::downgrade(&cache.pool);
        Python::initialize();
        Python::attach(|py| {
            assert!(cache.access(py, 1).is_err());
            let view = cache.access(py, 0).unwrap();
            let slice = view.call_method1("cast", ("B",)).unwrap();
            view.call_method0("release").unwrap();
            drop(cache);
            assert!(pool.upgrade().is_some());
            slice.call_method0("release").unwrap();
            assert!(pool.upgrade().is_none());
        });
    }

    #[test]
    fn python_host_cache_copies_after_access_ends() {
        Python::initialize();
        let context = crate::tests::context();
        Python::attach(|py| {
            let engine = PyModule::new(py, "engine").unwrap();
            crate::register(&engine).unwrap();
            let spec = CacheSpec {
                layer_count: 1,
                pools: vec![Format {
                    block: 1,
                    bytes: 8,
                    format: PageFormat::contiguous(vec![1, 8], 1).unwrap(),
                }],
                groups: vec![Group {
                    pool: 0,
                    layers: vec![0],
                    logical_group: 0,
                    kind: Kind::Paged { window: None },
                }],
            };
            let inner = DeviceCache::new(&Context::lock(&context, py).device, &spec, 2).unwrap();
            let scope = PyDict::new(py);
            scope.set_item("engine", engine).unwrap();
            scope
                .set_item("cache", Py::new(py, Cache { inner, context }).unwrap())
                .unwrap();
            let setup = py.run(
                c"
copier = engine.Copier(cache, [[0]])
assert copier.slot_bytes == 8
assert copier.layer_ranges == [[(0, 8)]]
",
                Some(&scope),
                None,
            );
            setup.unwrap();

            let copier = scope
                .get_item("copier")
                .unwrap()
                .unwrap()
                .extract::<PyRef<'_, PyCopier>>()
                .unwrap();
            let pinned = HostBuffer::zero(8).is_pinned();
            let (host, coalescing) = match PyHostCache::new(py, &copier, 2) {
                Ok(host) => {
                    assert!(pinned, "HostCache unexpectedly allocated pageable memory");
                    (host, true)
                }
                Err(error) => {
                    assert!(
                        !pinned,
                        "HostCache failed despite available pinned memory: {error}"
                    );
                    assert!(error.to_string().contains("page-locked"), "{error}");
                    let pool = Arc::new(Pool {
                        slots: (0..2)
                            .map(|_| Slot::new(HostBuffer::from_vec(vec![0; 8])))
                            .collect(),
                        layout: Arc::clone(&copier.layout),
                        core: Arc::downgrade(&copier.core),
                    });
                    (PyHostCache { pool }, false)
                }
            };
            scope.set_item("pool", Py::new(py, host).unwrap()).unwrap();
            scope.set_item("coalescing_supported", coalescing).unwrap();
            drop(copier);

            let result = py.run(
                c"
assert pool.slot_bytes == 8
payload = bytes(range(1, 9))
view = pool.access(0)
view[:] = payload
copy = copier.copy_in([(0, [(0, 0)])], pool)
pending = not copy.done()
view.release()
assert pending
copy.wait()

for coalesced in ((False, True) if coalescing_supported else (False,)):
    copier.copy_in([(0, [(0, 0)])], pool, coalesce=coalesced).wait()
    copier.copy_out([(1, [(0, 0)])], pool, coalesce=coalesced).wait()
    with pool.access(1) as restored:
        assert bytes(restored) == payload
copier.shutdown()
",
                Some(&scope),
                None,
            );
            scope.clear();
            result.unwrap();
        });
    }
}
