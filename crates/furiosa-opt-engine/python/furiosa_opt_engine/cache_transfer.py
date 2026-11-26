"""DMA completion and aligned file I/O shared by serving-framework adapters."""

from __future__ import annotations

import contextlib
import ctypes
import mmap
import os
import struct
import sys
import threading
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple, Protocol, TypeAlias, TypeVar, cast

from typing_extensions import Buffer

import furiosa_opt_engine as engine

from ._host_region import HostRegionLease, _byte_view, retain_until_exit

ByteBuffer: TypeAlias = Buffer | engine.HostSlot
DEFAULT_READ_THREADS = 8
DEFAULT_WRITE_THREADS = 4
_AT_FDCWD = -100
_STATX_DIOALIGN = 0x00002000
_STATX_SIZE = 256
_STATX_DIO_ALIGNMENT_OFFSET = 152


class PendingTransfer(Protocol):
    """The completion operations adapters need for a submitted host transfer."""

    def done(self) -> bool: ...

    def wait(self) -> None: ...


TransferT = TypeVar("TransferT", bound=PendingTransfer)


class LayerPageRoutes(NamedTuple):
    """Device pages and corresponding HostCache slots for one cache layer."""

    device_pages: Sequence[int]
    host_slots: Sequence[int]


class AlignedFileIO:
    """Keep framework-native page bytes; use O_DIRECT only for aligned pages."""

    def __init__(self, directory: str, *, use_direct: bool | None = None) -> None:
        self.offset_alignment = mmap.PAGESIZE
        self.memory_alignment = mmap.PAGESIZE
        self.use_direct = False
        if use_direct is not False:
            alignment = _probe_direct_io(directory)
            if alignment is not None:
                self.memory_alignment, self.offset_alignment = alignment
                self.use_direct = True
        self.local = threading.local()

    def write(self, path: str, source: ByteBuffer) -> None:
        source_view = _byte_view(source)
        try:
            self.prepare(path, len(source_view), 1)
            self._write_at(path, 0, source_view)
        finally:
            source_view.release()

    def read(self, path: str, destination: ByteBuffer) -> None:
        destination_view = _byte_view(destination)
        try:
            self._read_at(path, 0, destination_view)
        finally:
            destination_view.release()

    def prepare(
        self, path: str, page_bytes: int, pages: int, *, file_bytes: int | None = None
    ) -> None:
        """Create compact pages, optionally padded to the framework's chunk size."""
        if pages <= 0:
            raise ValueError("file transfers must contain pages")
        if page_bytes <= 0:
            raise ValueError("file transfers must contain bytes")
        page_data_bytes = page_bytes * pages
        size = page_data_bytes if file_bytes is None else file_bytes
        if size < page_data_bytes:
            raise ValueError("file size cannot be smaller than its cache pages")
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.ftruncate(descriptor, size)
        except Exception:
            os.close(descriptor)
            with contextlib.suppress(OSError):
                os.remove(path)
            raise
        os.close(descriptor)

    def writev_at(self, path: str, offset: int, sources: Iterable[ByteBuffer]) -> None:
        """Write vectors contiguously at an exact framework-native file offset."""
        self._write_or_readv(path, offset, sources, writing=True)

    def readv_at(self, path: str, offset: int, destinations: Iterable[ByteBuffer]) -> None:
        """Read vectors contiguously at an exact framework-native file offset."""
        self._write_or_readv(path, offset, destinations, writing=False)

    def writev_at_many(
        self,
        path: str,
        writes: Iterable[tuple[int, Iterable[ByteBuffer]]],
    ) -> None:
        """Write several vector ranges while reusing direct and buffered file handles."""
        self._write_or_readv_many(path, writes, writing=True)

    def readv_at_many(
        self,
        path: str,
        reads: Iterable[tuple[int, Iterable[ByteBuffer]]],
    ) -> None:
        """Read several vector ranges while reusing direct and buffered file handles."""
        self._write_or_readv_many(path, reads, writing=False)

    def _write_at(self, path: str, offset: int, source: memoryview) -> None:
        if offset < 0 or not source:
            raise ValueError("file writes require a nonempty buffer and nonnegative offset")
        direct = self._direct_compatible(offset, (source,))
        source_address = self._direct_address(source) if direct else None
        bounce = direct and source_address is None
        target = self._buffer(len(source)) if bounce else source
        try:
            if bounce:
                target[:] = source
            descriptor = os.open(path, os.O_WRONLY | self._direct_flag(direct))
            try:
                written = os.pwrite(descriptor, target, offset)
                if written != len(target):
                    raise OSError(f"short write of {path}: {written} of {len(target)} bytes")
            finally:
                os.close(descriptor)
        finally:
            if bounce:
                target.release()

    def _read_at(self, path: str, offset: int, destination: memoryview) -> None:
        if offset < 0 or not destination:
            raise ValueError("file reads require a nonempty buffer and nonnegative offset")
        direct = self._direct_compatible(offset, (destination,))
        destination_address = self._direct_address(destination) if direct else None
        bounce = direct and destination_address is None
        source = self._buffer(len(destination)) if bounce else destination
        try:
            descriptor = os.open(path, os.O_RDONLY | self._direct_flag(direct))
            try:
                read = os.preadv(descriptor, [source], offset)
                if read != len(source):
                    raise OSError(f"short read of {path}: {read} of {len(source)} bytes")
            finally:
                os.close(descriptor)
            if bounce:
                if destination_address is None:
                    destination_address = ctypes.addressof(ctypes.c_ubyte.from_buffer(destination))
                ctypes.memmove(destination_address, self.local.address, len(destination))
        finally:
            if bounce:
                source.release()

    def _write_or_readv(
        self,
        path: str,
        offset: int,
        buffers: Iterable[ByteBuffer],
        *,
        writing: bool,
    ) -> None:
        self._write_or_readv_many(path, ((offset, buffers),), writing=writing)

    def _write_or_readv_many(
        self,
        path: str,
        transfers: Iterable[tuple[int, Iterable[ByteBuffer]]],
        *,
        writing: bool,
    ) -> None:
        descriptors: dict[bool, int] = {}
        vector_limit = None
        processed = False
        transfer_iter = iter(transfers)
        cleanup = contextlib.ExitStack()
        close_transfers = getattr(transfer_iter, "close", None)
        if close_transfers is not None:
            cleanup.callback(close_transfers)

        def descriptor(direct: bool) -> int:
            if direct not in descriptors:
                flags = os.O_WRONLY if writing else os.O_RDONLY
                file_descriptor = os.open(path, flags | self._direct_flag(direct))
                descriptors[direct] = file_descriptor
                cleanup.callback(os.close, file_descriptor)
            return descriptors[direct]

        def iov_limit() -> int:
            nonlocal vector_limit
            if vector_limit is None:
                try:
                    vector_limit = int(os.sysconf("SC_IOV_MAX"))
                except (AttributeError, OSError, ValueError):
                    vector_limit = 1024
                vector_limit = max(vector_limit, 1)
            return vector_limit

        with cleanup:
            for offset, buffers in transfer_iter:
                processed = True
                if offset < 0:
                    raise ValueError("file offsets cannot be negative")
                views = []
                try:
                    for buffer in buffers:
                        views.append(_byte_view(buffer))
                    if not views or any(not view for view in views):
                        raise ValueError("vectored file transfers require nonempty buffers")

                    direct = self._direct_compatible(offset, views)
                    if len(views) == 1:
                        view = views[0]
                        address = self._direct_address(view) if direct else None
                        bounce = direct and address is None
                        target = self._buffer(len(view)) if bounce else view
                        try:
                            if writing:
                                if bounce:
                                    target[:] = view
                                actual = os.pwrite(descriptor(direct), target, offset)
                            else:
                                actual = os.preadv(descriptor(direct), [target], offset)
                            if actual != len(target):
                                operation = "write" if writing else "read"
                                raise OSError(
                                    f"short {operation} of {path}: {actual} of {len(target)} bytes"
                                )
                            if bounce and not writing:
                                if address is None:
                                    address = ctypes.addressof(ctypes.c_ubyte.from_buffer(view))
                                ctypes.memmove(address, self.local.address, len(view))
                        finally:
                            if bounce:
                                target.release()
                        continue

                    direct = direct and all(
                        self._direct_address(view) is not None for view in views
                    )
                    file_descriptor = descriptor(direct)
                    current_offset = offset
                    limit = iov_limit()
                    for start in range(0, len(views), limit):
                        batch = views[start : start + limit]
                        expected = sum(map(len, batch))
                        actual = (
                            os.pwritev(file_descriptor, batch, current_offset)
                            if writing
                            else os.preadv(file_descriptor, batch, current_offset)
                        )
                        if actual != expected:
                            operation = "write" if writing else "read"
                            raise OSError(
                                f"short {operation} of {path}: {actual} of {expected} bytes"
                            )
                        current_offset += expected
                finally:
                    for view in views:
                        view.release()
            if not processed:
                raise ValueError("vectored file transfers require at least one range")

    def _direct_flag(self, direct: bool) -> int:
        return getattr(os, "O_DIRECT", 0) if direct else 0

    def _direct_compatible(self, offset: int, views: Iterable[memoryview]) -> bool:
        return (
            self.use_direct
            and offset >= 0
            and offset % self.offset_alignment == 0
            and all(len(view) > 0 and len(view) % self.offset_alignment == 0 for view in views)
        )

    def _direct_address(self, buffer: memoryview) -> int | None:
        """Return the pinned address only when it satisfies the probed memory alignment."""
        if buffer.readonly:
            return None
        address = ctypes.addressof(ctypes.c_ubyte.from_buffer(buffer))
        return address if address % self.memory_alignment == 0 else None

    def _buffer(self, size: int) -> memoryview:
        if size <= 0:
            raise ValueError("file transfers must contain bytes")
        capacity = getattr(self.local, "capacity", 0)
        if capacity < size:
            aligned = _align_up(size, mmap.PAGESIZE)
            previous = getattr(self.local, "buffer", None)
            if previous is not None:
                previous.close()
            self.local.buffer = None
            self.local.capacity = 0
            buffer = mmap.mmap(-1, aligned + self.memory_alignment - 1)
            base_address = ctypes.addressof(ctypes.c_ubyte.from_buffer(buffer))
            offset = (-base_address) % self.memory_alignment
            self.local.buffer = buffer
            self.local.address = base_address + offset
            self.local.offset = offset
            self.local.capacity = aligned
        offset = self.local.offset
        return memoryview(self.local.buffer)[offset : offset + size]


def _probe_o_direct(directory: str) -> bool:
    """Check whether aligned direct I/O works for this filesystem."""
    return _probe_direct_io(directory) is not None


def _probe_direct_io(directory: str) -> tuple[int, int] | None:
    """Return filesystem memory/offset alignments after one aligned direct-I/O probe."""
    direct = getattr(os, "O_DIRECT", 0)
    if not direct or not sys.platform.startswith("linux"):
        return None
    path = os.path.join(directory, f".furiosa-o-direct-{uuid.uuid4().hex}")
    descriptor = None
    direct_descriptor = None
    mapping = None
    view = None
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        alignment = _statx_dio_alignment(path)
        if alignment is None:
            return None
        memory_alignment, offset_alignment = alignment
        size = _align_up(mmap.PAGESIZE, offset_alignment)
        mapping = mmap.mmap(-1, size + memory_alignment - 1)
        base_address = ctypes.addressof(ctypes.c_ubyte.from_buffer(mapping))
        offset = (-base_address) % memory_alignment
        view = memoryview(mapping)[offset : offset + size]
        direct_descriptor = os.open(path, os.O_RDWR | direct)
        if os.pwrite(direct_descriptor, view, 0) != size:
            return None
        if os.preadv(direct_descriptor, [view], 0) != size:
            return None
        return alignment
    except (AttributeError, OSError, ValueError):
        return None
    finally:
        if view is not None:
            view.release()
        if mapping is not None:
            mapping.close()
        if direct_descriptor is not None:
            os.close(direct_descriptor)
        if descriptor is not None:
            os.close(descriptor)
        with contextlib.suppress(OSError):
            os.remove(path)


def _statx_dio_alignment(path: str) -> tuple[int, int] | None:
    """Read direct-I/O alignment from statx without adding work to page transfers."""
    try:
        statx = ctypes.CDLL(None, use_errno=True).statx
    except (AttributeError, OSError):
        return None
    statx.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_void_p,
    ]
    statx.restype = ctypes.c_int
    result = ctypes.create_string_buffer(_STATX_SIZE)
    if statx(_AT_FDCWD, os.fsencode(path), 0, _STATX_DIOALIGN, ctypes.byref(result)) != 0:
        return None
    values = result.raw
    mask = struct.unpack_from("=I", values)[0]
    if mask & _STATX_DIOALIGN == 0:
        return None
    memory_alignment, offset_alignment = struct.unpack_from(
        "=II", values, _STATX_DIO_ALIGNMENT_OFFSET
    )
    if memory_alignment <= 0 or offset_alignment <= 0:
        return None
    return memory_alignment, offset_alignment


def _align_up(size: int, alignment: int) -> int:
    return ((size + alignment - 1) // alignment) * alignment


def submit(transfers: Iterable[TransferT]) -> tuple[TransferT, ...]:
    """Submit copies and drain earlier work if a later copy cannot be submitted."""
    submitted: list[TransferT] = []
    try:
        for transfer in transfers:
            submitted.append(transfer)
    except BaseException as submit_error:
        drain_error = settle(submitted)
        if drain_error is not None and drain_error is not submit_error:
            raise submit_error from drain_error
        raise
    return tuple(submitted)


def settle(transfers: Iterable[PendingTransfer]) -> BaseException | None:
    """Wait for every submitted copy and return the first failure after draining the rest."""
    failure = None
    for transfer in transfers:
        try:
            transfer.wait()
        except BaseException as error:
            if failure is None:
                failure = error
    return failure


def _cache_page_bytes(cache: engine.decoder.Cache) -> tuple[int, ...]:
    """Validate the cache spec and return each layer's page size in engine order."""
    if cache.blocks <= 0:
        raise ValueError("cache transfers require at least one block")
    device_layers: list[engine.HbmBuffer] = cache.layers
    if not device_layers:
        raise ValueError("cache transfers require at least one layer")

    page_bytes_by_layer: list[int | None] = [None] * len(device_layers)
    for group in cache.spec.groups:
        if not 0 <= group.pool < len(cache.spec.pools):
            raise ValueError(f"engine cache group names unknown pool {group.pool}")
        page_bytes = cache.spec.pools[group.pool].bytes
        if page_bytes <= 0:
            raise ValueError("cache pages must contain bytes")
        for index in group.layers:
            if not 0 <= index < len(page_bytes_by_layer):
                raise ValueError(f"engine cache group names unknown layer {index}")
            if page_bytes_by_layer[index] is not None:
                raise ValueError(f"engine cache layer {index} appears in multiple groups")
            if len(device_layers[index]) != cache.blocks * page_bytes:
                raise ValueError(
                    f"engine cache layer {index} allocation disagrees with its page size"
                )
            page_bytes_by_layer[index] = page_bytes

    if any(page_bytes is None for page_bytes in page_bytes_by_layer):
        raise ValueError("engine cache groups do not cover every layer exactly once")
    return tuple(page_bytes for page_bytes in page_bytes_by_layer if page_bytes is not None)


def cache_staging_page_bytes(cache: engine.decoder.Cache, chips: int) -> int:
    """Return pinned bytes needed for one page slot across every cache layer."""
    if chips <= 0:
        raise ValueError("cache staging requires at least one chip")
    return sum(cache_layer_page_bytes(cache, chips))


def bounded_staging_capacity(page_bytes: int, budget_bytes: int, *, max_pages: int) -> int:
    """Fit a bounded number of page slots into a pinned-memory budget."""
    if page_bytes <= 0:
        raise ValueError("cache staging page size must be positive")
    if budget_bytes <= 0:
        raise ValueError("cache staging budget must be positive")
    if max_pages <= 0:
        raise ValueError("cache staging maximum page count must be positive")

    capacity = min(max_pages, budget_bytes // page_bytes)
    if capacity <= 0:
        raise ValueError(
            f"cache staging budget ({budget_bytes} bytes) cannot hold one cache page "
            f"({page_bytes} bytes)"
        )
    return capacity


def cache_staging_capacity(
    cache: engine.decoder.Cache,
    chips: int,
    budget_bytes: int,
    *,
    max_pages: int | None = None,
) -> int:
    """Fit a bounded number of engine page slots into a pinned-memory budget."""
    page_bytes = cache_staging_page_bytes(cache, chips)
    if max_pages is not None and max_pages <= 0:
        raise ValueError("cache staging maximum page count must be positive")
    return bounded_staging_capacity(
        page_bytes,
        budget_bytes,
        max_pages=min(cache.blocks, max_pages) if max_pages is not None else cache.blocks,
    )


def cache_layer_offsets(cache: engine.decoder.Cache, chips: int) -> tuple[int, ...]:
    """Return each layer's byte offset inside one page-major host slot."""
    offsets = []
    offset = 0
    for page_bytes in cache_layer_page_bytes(cache, chips):
        offsets.append(offset)
        offset += page_bytes
    return tuple(offsets)


def cache_layer_page_bytes(cache: engine.decoder.Cache, chips: int) -> tuple[int, ...]:
    """Return each layer's host-page size, including every chip's shard."""
    if chips <= 0:
        raise ValueError("cache staging requires at least one chip")
    return tuple(page_bytes * chips for page_bytes in _cache_page_bytes(cache))


def cache_copier(cache: engine.decoder.Cache, chips: int) -> engine.Copier:
    """Build the shared engine copy queue with one independently routable group per layer."""
    page_bytes = _cache_page_bytes(cache)
    offsets = cache_layer_offsets(cache, chips)
    if len(cache.layers) != len(page_bytes):
        raise ValueError("engine cache layer count differs from its cache spec")
    layers = [
        engine.CopierLayer(
            buffer=buffer,
            page_bytes=size,
            host_offset=offset,
            group=index,
        )
        for index, (buffer, size, offset) in enumerate(
            zip(cache.layers, page_bytes, offsets, strict=True)
        )
    ]
    return engine.Copier(layers, sum(size * chips for size in page_bytes))


def _shutdown_copier_after_failure(copier: engine.Copier, operation_error: BaseException) -> None:
    try:
        copier.shutdown()
    except BaseException as cleanup_error:
        retain_until_exit(copier)
        operation_error.add_note(f"cache Copier cleanup failed: {cleanup_error!r}")


@dataclass(frozen=True, slots=True)
class CacheStagingLease:
    """Sorted reusable host page IDs; slots may span fragmented free runs."""

    identifier: int
    slots: tuple[int, ...]
    _owner: object = field(repr=False, compare=False)


class _FreeRun(NamedTuple):
    start: int
    count: int


class _BoundedPageSlots:
    """Allocate page IDs for ``BoundedCacheStaging`` without allocating host memory."""

    def __init__(self, capacity_pages: int) -> None:
        if capacity_pages <= 0:
            raise ValueError("cache staging requires at least one page slot")
        self.capacity_pages = capacity_pages
        self._owner = object()
        # Sorted, disjoint, nonadjacent free runs.
        self._free_runs: list[_FreeRun] = [_FreeRun(0, capacity_pages)]
        self._leases: dict[int, tuple[int, ...]] = {}
        self._next_identifier = 0
        self._lock = threading.Lock()

    def try_acquire(self, page_count: int) -> CacheStagingLease | None:
        """Reserve sorted page IDs, or return ``None`` when total free capacity is insufficient."""
        if page_count <= 0:
            raise ValueError("cache staging leases require at least one page")
        if page_count > self.capacity_pages:
            return None
        with self._lock:
            first_fit = None
            exact_fit = None
            for index, run in enumerate(self._free_runs):
                if run.count == page_count:
                    exact_fit = index
                    break
                if first_fit is None and run.count > page_count:
                    first_fit = index
            free_index = exact_fit if exact_fit is not None else first_fit
            if free_index is not None:
                start, count = self._free_runs[free_index]
                slots = tuple(range(start, start + page_count))
                if count == page_count:
                    del self._free_runs[free_index]
                else:
                    self._free_runs[free_index] = _FreeRun(start + page_count, count - page_count)
            else:
                if sum(run.count for run in self._free_runs) < page_count:
                    return None
                selected: list[int] = []
                remaining_runs: list[_FreeRun] = []
                for index, (start, count) in enumerate(self._free_runs):
                    take = min(count, page_count - len(selected))
                    selected.extend(range(start, start + take))
                    if take < count:
                        remaining_runs.append(_FreeRun(start + take, count - take))
                    if len(selected) == page_count:
                        remaining_runs.extend(self._free_runs[index + 1 :])
                        break
                slots = tuple(selected)
                self._free_runs = remaining_runs
            identifier = self._next_identifier
            self._next_identifier += 1
            self._leases[identifier] = slots
            return CacheStagingLease(identifier, slots, self._owner)

    def release(self, lease: CacheStagingLease) -> None:
        """Return a lease's slots exactly once."""
        with self._lock:
            slots = self._require_locked(lease)
            del self._leases[lease.identifier]
            self._return_slots(slots)

    def require(self, lease: CacheStagingLease) -> tuple[int, ...]:
        """Validate and return the slot IDs of an active lease."""
        with self._lock:
            return self._require_locked(lease)

    def has_active_leases(self) -> bool:
        """Whether any staging slots are still owned by a transfer caller."""
        with self._lock:
            return bool(self._leases)

    def _require_locked(self, lease: CacheStagingLease) -> tuple[int, ...]:
        if lease._owner is not self._owner:
            raise ValueError("cache staging lease belongs to another pool")
        slots = self._leases.get(lease.identifier)
        if slots is None or slots != lease.slots:
            raise ValueError("cache staging lease is no longer active")
        return slots

    def _return_slots(self, slots: tuple[int, ...]) -> None:
        returned_runs: list[_FreeRun] = []
        run_start = slots[0]
        run_end = run_start + 1
        for slot in slots[1:]:
            if slot == run_end:
                run_end += 1
                continue
            returned_runs.append(_FreeRun(run_start, run_end - run_start))
            run_start = slot
            run_end = slot + 1
        returned_runs.append(_FreeRun(run_start, run_end - run_start))

        existing_index = 0
        returned_index = 0
        merged: list[_FreeRun] = []
        while existing_index < len(self._free_runs) or returned_index < len(returned_runs):
            if returned_index == len(returned_runs) or (
                existing_index < len(self._free_runs)
                and self._free_runs[existing_index].start < returned_runs[returned_index].start
            ):
                run = self._free_runs[existing_index]
                existing_index += 1
            else:
                run = returned_runs[returned_index]
                returned_index += 1

            if merged and merged[-1].start + merged[-1].count > run.start:
                raise RuntimeError("cache staging free runs overlap")
            if merged and merged[-1].start + merged[-1].count == run.start:
                previous = merged[-1]
                merged[-1] = _FreeRun(previous.start, previous.count + run.count)
            else:
                merged.append(run)
        self._free_runs = merged


class BoundedCacheStaging:
    """A fixed-capacity, engine-pinned page pool shared by transfer adapters.

    ``HostCache`` retains the contiguous pinned backing and exposes slots to ``Copier``. A lease
    reserves slots shared by all layers. Adapters gather or scatter through the lease, then release
    it after network transfer completion. Capacity does not depend on cache size. The backing stays
    registered while the engine cache is released and binds to a replacement cache after wake-up.
    """

    def __init__(
        self,
        cache: engine.decoder.Cache,
        chips: int,
        capacity_pages: int,
    ) -> None:
        if chips <= 0:
            raise ValueError("cache staging requires at least one chip")
        if capacity_pages <= 0:
            raise ValueError("cache staging requires at least one page slot")
        self._cache: engine.decoder.Cache | None = cache
        self._blocks = cache.blocks
        self._chips = chips
        self.capacity_pages = capacity_pages
        self._page_bytes = _cache_page_bytes(cache)
        self.host_page_bytes = cache_layer_page_bytes(cache, chips)
        self.layer_offsets = cache_layer_offsets(cache, chips)
        self.host_block_bytes = sum(self.host_page_bytes)
        total_bytes = capacity_pages * self.host_block_bytes
        backing = engine.HostBuffer(total_bytes)
        active_copier = cache_copier(cache, chips)
        self._copier: engine.Copier | None = active_copier
        try:
            self._host_cache: engine.HostCache | None = engine.HostCache(
                active_copier, capacity_pages, backing=backing
            )
        except BaseException as operation_error:
            _shutdown_copier_after_failure(active_copier, operation_error)
            raise
        self._detached_backing: engine.HostBuffer | None = None

        self._slots = _BoundedPageSlots(capacity_pages)
        self._copying: set[int] = set()
        self._transport_access: dict[int, tuple[engine.HostSlotAccess, ...]] = {}
        self._lock = threading.Lock()
        self._cache_releasing = False

    def try_acquire(self, page_count: int) -> CacheStagingLease | None:
        """Reserve page slots, or return `None` while full or detached from the engine cache."""
        with self._lock:
            if self._cache is None or self._cache_releasing:
                return None
            return self._slots.try_acquire(page_count)

    def lease_host_region(self) -> HostRegionLease:
        """Lease complete pinned staging allocation for contiguous external I/O."""
        with self._lock:
            host_cache = self._host_cache
            if host_cache is None or self._cache_releasing:
                raise RuntimeError("cannot lease host memory while the engine cache is released")
            backing = host_cache._backing_buffer()
            view = memoryview(cast(Buffer, backing)).cast("B")
        try:
            return HostRegionLease(view, keepalive=backing)
        except BaseException:
            view.release()
            raise

    def release_engine_cache(self, cache: engine.decoder.Cache) -> None:
        """Drain and drop Copier's HBM aliases before vLLM releases its cache."""
        with self._lock:
            if self._cache is None:
                return
            self._require_cache(cache)
            if self._slots.has_active_leases() or self._copying or self._transport_access:
                raise RuntimeError(
                    "cannot release engine cache while staging leases, copies, or network "
                    "transfers remain active"
                )
            copier = self._copier
            if copier is None:
                raise RuntimeError("bounded staging cache has no Copier")
            host_cache = self._host_cache
            if host_cache is None:
                raise RuntimeError("bounded staging cache has no active HostCache")
            backing = host_cache._backing_buffer()
            self._cache_releasing = True

        try:
            copier.shutdown()
        except BaseException:
            with self._lock:
                self._cache_releasing = False
            raise
        with self._lock:
            if self._copier is copier:
                self._detached_backing = backing
                self._cache = None
                self._copier = None
                self._host_cache = None
            self._cache_releasing = False

    def restore_engine_cache(self, cache: engine.decoder.Cache) -> None:
        """Bind a replacement engine cache to the existing registered host buffer."""
        with self._lock:
            if self._cache is cache:
                return
            if self._cache is not None or self._cache_releasing:
                raise RuntimeError("bounded staging must release its old cache before rebinding")
            backing = self._detached_backing
            if backing is None:
                raise RuntimeError("bounded staging has no detached HostCache backing")
            copier, host_cache = self._new_cache_binding(cache, backing)
            self._copier = copier
            self._host_cache = host_cache
            self._cache = cache
            self._detached_backing = None

    def _new_cache_binding(
        self, cache: engine.decoder.Cache, backing: engine.HostBuffer
    ) -> tuple[engine.Copier, engine.HostCache]:
        if cache.blocks != self._blocks or _cache_page_bytes(cache) != self._page_bytes:
            raise ValueError("new engine cache does not match bounded staging geometry")
        if cache_layer_offsets(cache, self._chips) != self.layer_offsets:
            raise ValueError("new engine cache layer offsets differ from registered staging")
        copier = cache_copier(cache, self._chips)
        try:
            host_cache = engine.HostCache(copier, self.capacity_pages, backing=backing)
        except BaseException as operation_error:
            _shutdown_copier_after_failure(copier, operation_error)
            raise
        return copier, host_cache

    def _require_cache(self, cache: engine.decoder.Cache) -> None:
        if self._cache is None:
            raise RuntimeError("bounded staging has no engine cache; restore it before transfer")
        if cache is not self._cache:
            raise ValueError("current engine cache differs from bounded staging cache")
        if cache.blocks != self._blocks or _cache_page_bytes(cache) != self._page_bytes:
            raise ValueError("current engine cache does not match bounded staging geometry")

    def release(self, lease: CacheStagingLease) -> None:
        """Return a completed transfer's slots to the pool exactly once."""
        with self._lock:
            self._slots.require(lease)
            if lease.identifier in self._copying:
                raise RuntimeError("cannot release cache staging slots during an engine copy")
            if lease.identifier in self._transport_access:
                raise RuntimeError("cannot release cache staging slots during a network transfer")
            self._slots.release(lease)

    def begin_transport(self, lease: CacheStagingLease) -> None:
        """Mark the leased bytes in use by a transport until `complete_transport`."""
        with self._lock:
            if self._host_cache is None or self._cache_releasing:
                raise RuntimeError("cannot begin transport while the engine cache is released")
            slots = self._slots.require(lease)
            if lease.identifier in self._copying:
                raise RuntimeError("cannot start a network transfer during an engine copy")
            if lease.identifier in self._transport_access:
                raise RuntimeError("cache staging lease already has a network transfer")
            access = tuple(self._host_cache.slot(slot).access() for slot in slots)
            self._transport_access[lease.identifier] = access

    def complete_transport(self, lease: CacheStagingLease) -> None:
        """Mark a transport complete so its slots can be copied or released."""
        with self._lock:
            self._slots.require(lease)
            if lease.identifier not in self._transport_access:
                raise RuntimeError("cache staging lease has no network transfer in flight")
            self._transport_access.pop(lease.identifier)

    def copy_pages(
        self,
        cache: engine.decoder.Cache,
        pages_by_layer: Mapping[int, LayerPageRoutes],
        lease: CacheStagingLease,
        *,
        to_host: bool,
    ) -> CacheStagingCopy:
        """Submit a page copy; wait for its handle before reusing or releasing the lease."""
        with self._lock:
            if self._cache_releasing:
                raise RuntimeError("cannot copy while the engine cache is being released")
            self._require_cache(cache)
            leased_slots = set(self._slots.require(lease))
            if lease.identifier in self._copying:
                raise RuntimeError("cache staging lease already has an engine copy in flight")
            if lease.identifier in self._transport_access:
                raise RuntimeError("cannot copy cache staging slots during a network transfer")
            copier = self._copier
            host_cache = self._host_cache
            if copier is None or host_cache is None:
                raise RuntimeError("bounded staging has no active Copier")
            routes_by_slot: dict[int, list[tuple[int, int]]] = {}
            for index, layer_routes in pages_by_layer.items():
                pages = layer_routes.device_pages
                route_slots = layer_routes.host_slots
                if not 0 <= index < len(self._page_bytes):
                    raise ValueError("cache staging names an unknown layer")
                if len(pages) != len(route_slots):
                    raise ValueError("cache page and staging slot lists must have equal lengths")
                if any(slot not in leased_slots for slot in route_slots):
                    raise ValueError("cache staging routes use a slot outside their lease")
                if any(not 0 <= page < self._blocks for page in pages):
                    raise ValueError("engine cache page index is out of range")
                for page, slot in zip(pages, route_slots, strict=True):
                    routes_by_slot.setdefault(slot, []).append((index, page))
            routes = [(slot, routes_by_slot[slot]) for slot in sorted(routes_by_slot)]
            if not routes:
                return CacheStagingCopy.completed(self)
            self._copying.add(lease.identifier)
        try:
            transfer = (
                copier.copy_out(routes, host_cache)
                if to_host
                else copier.copy_in(routes, host_cache)
            )
        except BaseException:
            self._finish_copy(lease.identifier)
            raise
        return CacheStagingCopy(self, lease.identifier, transfer)

    def _finish_copy(self, identifier: int) -> None:
        with self._lock:
            self._copying.discard(identifier)


class CacheStagingCopy:
    """Completion handle that holds a staging lease through asynchronous DMA."""

    def __init__(
        self,
        staging: BoundedCacheStaging,
        identifier: int | None,
        transfer: PendingTransfer | None,
    ) -> None:
        self._staging = staging
        self._identifier = identifier
        self._transfer = transfer
        self._wait_lock = threading.Lock()
        self._finished = transfer is None

    @classmethod
    def completed(cls, staging: BoundedCacheStaging) -> CacheStagingCopy:
        return cls(staging, None, None)

    def done(self) -> bool:
        transfer = self._transfer
        return True if transfer is None else transfer.done()

    def wait(self) -> None:
        with self._wait_lock:
            if self._finished:
                return
            try:
                assert self._transfer is not None
                self._transfer.wait()
            finally:
                self._finished = True
                if self._identifier is not None:
                    self._staging._finish_copy(self._identifier)
