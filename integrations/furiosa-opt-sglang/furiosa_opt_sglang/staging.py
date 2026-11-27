"""Shared bounded host staging for SGLang's NIXL and Mooncake transports."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.staging_buffer import (
    StagingAllocator,
    staging_grid_tokens,
)
from sglang.srt.disaggregation.common.staging_handler import DecodeStagingHandler

import furiosa_opt_engine as engine
from furiosa_opt_engine.cache_transfer import (
    BoundedCacheStaging,
    CacheStagingCopy,
    CacheStagingLease,
    LayerPageRoutes,
    cache_copier,
    cache_staging_page_bytes,
    host_buffer_region,
)

CLEARED_CHUNK_STAGING_INFO = (-1, -1, 0, -1, 0)


class StagingBuffer:
    """Pinned host storage for one bounded SGLang transfer region."""

    def __init__(self, size_bytes: int) -> None:
        self.buffer = engine.HostBuffer(size_bytes)
        self.size_bytes = size_bytes
        self._region = host_buffer_region(self.buffer)
        self.data_ptr = self._region.region.address
        self.cache_staging: BoundedCacheStaging | None = None

    def get_ptr(self) -> int:
        return self.data_ptr

    def get_size(self) -> int:
        return self.size_bytes

    def fits(self, required_bytes: int) -> bool:
        return required_bytes <= self.size_bytes

    def attach_cache(
        self,
        cache: engine.decoder.Cache,
        chips: int,
        copier: engine.Copier,
    ) -> BoundedCacheStaging:
        """Alias this registered protocol buffer as page-major Copier slots."""
        existing = self.cache_staging
        if existing is not None and existing.matches_binding(cache, copier):
            return existing
        page_bytes = cache_staging_page_bytes(cache, chips)
        capacity_pages = self.size_bytes // page_bytes
        if capacity_pages <= 0:
            raise ValueError(
                f"SGLang staging buffer ({self.size_bytes} bytes) cannot hold one cache page "
                f"({page_bytes} bytes)"
            )
        backing = self.buffer.slice(0, capacity_pages * page_bytes)
        self.cache_staging = BoundedCacheStaging(
            cache,
            chips,
            capacity_pages,
            backing=backing,
            copier=copier,
        )
        return self.cache_staging


@dataclass(slots=True)
class StagingBinding:
    """Own Copier and HostCache aliases for one engine-cache storage generation."""

    storage: engine.decoder.Cache
    copier: engine.Copier
    buffers: list[StagingBuffer] = field(default_factory=list)

    def attach(self, buffer: StagingBuffer, chips: int) -> None:
        buffer.attach_cache(self.storage, chips, self.copier)
        if all(existing is not buffer for existing in self.buffers):
            self.buffers.append(buffer)

    def close(self) -> None:
        """Drain queue before dropping every HostCache alias bound to this storage."""
        try:
            self.copier.shutdown()
        finally:
            for buffer in self.buffers:
                staging = buffer.cache_staging
                if staging is not None and staging.matches_binding(self.storage, self.copier):
                    buffer.cache_staging = None
            self.buffers.clear()


def _manager_staging_buffers(manager: Any) -> tuple[StagingBuffer, ...]:
    context = getattr(manager, "_staging_ctx", None)
    buffers = list(getattr(context, "buffers", ()) or ())
    allocator = getattr(context, "allocator", None)
    if allocator is not None:
        buffers.append(allocator.buffer)
    unique = {id(buffer): buffer for buffer in buffers}
    return tuple(unique.values())


def attach_manager_staging(manager: Any, cache: engine.decoder.Cache, chips: int) -> engine.Copier:
    """Bind this manager's staging pools to one Copier for the current cache storage."""
    binding = getattr(manager, "_staging_binding", None)
    if binding is not None and binding.storage is not cache:
        release_manager_staging(manager)
        binding = None
    if binding is None:
        binding = StagingBinding(cache, cache_copier(cache, chips))
        try:
            for buffer in _manager_staging_buffers(manager):
                binding.attach(buffer, chips)
        except BaseException as attach_error:
            try:
                binding.close()
            except BaseException as close_error:
                raise BaseExceptionGroup(
                    "SGLang staging attach and cleanup failed", [attach_error, close_error]
                ) from None
            raise
        manager._staging_binding = binding
    else:
        for buffer in _manager_staging_buffers(manager):
            binding.attach(buffer, chips)
    return binding.copier


def manager_staging_copier(manager: Any) -> engine.Copier | None:
    """Return the queue owned by this manager's current storage binding."""
    binding = getattr(manager, "_staging_binding", None)
    return binding.copier if binding is not None else None


def release_manager_staging(manager: Any) -> None:
    """Drop the generation owner before SGLang releases its engine KV storage."""
    binding = getattr(manager, "_staging_binding", None)
    if binding is None:
        return
    manager._staging_binding = None
    binding.close()


class BoundedStagingAllocator(StagingAllocator):
    """SGLang's ring and watermark allocator backed by engine-pinned host memory."""

    def __init__(self, size_bytes: int) -> None:
        self.buffer = StagingBuffer(size_bytes)
        self.total_size = size_bytes
        self.base_ptr = self.buffer.data_ptr
        self.head = 0
        self.round = 0
        self.allocations: dict[int, tuple[int, int, int]] = {}
        self.alloc_order: list[int] = []
        self.next_alloc_id = 0
        self.watermark_round = 0
        self.watermark_tail = 0
        self.lock = threading.Lock()
        self._scatter_stream: _TransferDrain | None = None


def staging_chunk_pages(chunked_prefill_size: int, page_size: int, backend: str) -> int:
    """Return the page-aligned transfer grid used by SGLang's staging protocol."""
    if page_size <= 0:
        raise ValueError(f"SGLang {backend} staging requires a positive page size")
    pages = staging_grid_tokens(chunked_prefill_size, page_size) // page_size
    if pages <= 0:
        raise ValueError(f"SGLang {backend} staging chunks must contain at least one page")
    return pages


def init_prefill_staging_buffers(
    register_memory: Callable[[int, int], None],
    kv_args: Any,
    count: int,
    chunked_prefill_size: int,
    backend: str,
) -> list[StagingBuffer]:
    """Allocate and register one bounded engine-pinned buffer per transfer worker."""
    pages = staging_chunk_pages(chunked_prefill_size, kv_args.page_size, backend)
    size_bytes = pages * sum(kv_args.kv_item_lens)
    buffers = [StagingBuffer(size_bytes) for _ in range(count)]
    for buffer in buffers:
        register_memory(buffer.get_ptr(), buffer.get_size())
    return buffers


def init_decode_staging_allocator(
    register_memory: Callable[[int, int], None], size_bytes: int
) -> BoundedStagingAllocator:
    """Allocate and register SGLang's bounded decode ring."""
    if size_bytes <= 0:
        raise ValueError("SGLANG_DISAGG_STAGING_POOL_SIZE_MB must be positive")
    allocator = BoundedStagingAllocator(size_bytes)
    register_memory(allocator.get_base_ptr(), allocator.get_total_size())
    return allocator


def submit_packed_cache_pages(
    staging: BoundedCacheStaging,
    cache: engine.decoder.Cache,
    page_ids: Sequence[int],
    host_slots: Sequence[int],
    *,
    to_host: bool,
) -> _LeasedCopy:
    """Submit page routes and keep their host slots reserved until DMA finishes."""
    if len(page_ids) != len(host_slots):
        raise ValueError("SGLang page and staging slot lists must have equal lengths")
    lease = staging.reserve_slots(host_slots)
    pages_by_layer = {
        index: LayerPageRoutes(page_ids, host_slots) for index in range(len(staging.layout.layers))
    }
    try:
        transfer = staging.copy_pages(cache, pages_by_layer, lease, to_host=to_host)
    except BaseException:
        staging.release(lease)
        raise
    return _LeasedCopy(staging, lease, transfer)


class _LeasedCopy:
    """Release a ring's exact engine slots only after its asynchronous DMA completes."""

    def __init__(
        self,
        staging: BoundedCacheStaging,
        lease: CacheStagingLease,
        transfer: CacheStagingCopy,
    ) -> None:
        self._staging = staging
        self._lease = lease
        self._transfer = transfer
        self._lock = threading.Lock()
        self._finished = False

    def done(self) -> bool:
        return self._transfer.done()

    def wait(self) -> None:
        with self._lock:
            if self._finished:
                return
            failures = []
            try:
                self._transfer.wait()
            except BaseException as error:
                failures.append(error)
            try:
                self._staging.release(self._lease)
            except BaseException as error:
                failures.append(error)
            self._finished = True
            if len(failures) == 1:
                raise failures[0]
            if failures:
                raise BaseExceptionGroup("SGLang DMA and staging release failed", failures)


class _TransferBatch:
    """Track one asynchronous engine copy through SGLang's staging lifecycle."""

    def __init__(
        self,
        handler: EngineDecodeStagingHandler,
        decode_req: Any,
        transfers: _LeasedCopy,
    ) -> None:
        self.handler = handler
        self.decode_req = decode_req
        self._engine_batch = transfers
        self._lock = threading.Lock()
        self._finished = False

    def query(self) -> bool:
        if not self._engine_batch.done():
            return False
        self.synchronize()
        return True

    def synchronize(self, observed_error: Exception | None = None) -> None:
        with self._lock:
            if self._finished:
                return
            try:
                self._engine_batch.wait()
            except BaseException as error:
                failure = error
            else:
                failure = observed_error
            self._finished = True
        with self.handler._transfer_lock:
            self.handler._transfer_batches.discard(self)
        if failure is None:
            return
        self.decode_req._staging_failed = True
        room = self.decode_req.req.bootstrap_room
        reporting_errors = []
        try:
            self.handler.kv_manager.record_failure(room, str(failure))
        except BaseException as error:
            reporting_errors.append(error)
        try:
            self.handler.kv_manager.update_status(room, KVPoll.Failed)
        except BaseException as error:
            reporting_errors.append(error)
        if reporting_errors:
            raise BaseExceptionGroup(
                "SGLang could not report a staging transfer failure",
                [failure, *reporting_errors],
            )


class _TransferDrain:
    """Drain every submitted engine DMA before SGLang releases staging pages."""

    def __init__(self, handler: EngineDecodeStagingHandler) -> None:
        self.handler = handler

    def synchronize(self) -> None:
        with self.handler._transfer_lock:
            batches = tuple(self.handler._transfer_batches)
        errors = []
        for batch in batches:
            try:
                batch.synchronize()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise BaseExceptionGroup("SGLang staging transfer drain failed", errors)


class EngineDecodeStagingHandler(DecodeStagingHandler):
    """Keep SGLang's arrival/watermark lifecycle and DMA pages into engine HBM."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._transfer_batches: set[_TransferBatch] = set()
        self._transfer_lock = threading.Lock()
        self.staging_allocator._scatter_stream = _TransferDrain(self)

    def submit_chunk_scatter(
        self, room: int, chunk_idx: int, page_start: int, num_pages: int
    ) -> bool:
        with self._transfer_lock:
            decode_req = self._room_to_decode_req.get(room)
            if decode_req is None or decode_req._staging_failed:
                return False
            receiver = self._room_to_receiver.get(room)
            infos = receiver.chunk_staging_infos if receiver is not None else []
            if not 0 <= chunk_idx < len(infos):
                return False
            alloc_id, staging_offset, _, _, _ = infos[chunk_idx]
            if alloc_id < 0 or staging_offset < 0:
                return False
            try:
                transfers = self._scatter_region(staging_offset, page_start, num_pages, decode_req)
            except Exception as error:
                decode_req._staging_failed = True
                self.kv_manager.record_failure(room, str(error))
                self.kv_manager.update_status(room, KVPoll.Failed)
                return False
            batch = _TransferBatch(self, decode_req, transfers)
            self._transfer_batches.add(batch)
            decode_req._chunk_events.append((batch, alloc_id))
            infos[chunk_idx] = CLEARED_CHUNK_STAGING_INFO
            return True

    def _scatter_region(
        self,
        staging_offset: int,
        page_start: int,
        num_pages: int,
        decode_req: Any,
    ) -> _LeasedCopy:
        if page_start < 0:
            raise ValueError("SGLang staging page start cannot be negative")
        if num_pages <= 0:
            raise ValueError("SGLang staging scatter must contain pages")
        cache = self.kv_manager.engine_cache
        if cache is None:
            raise RuntimeError("SGLang staging receive has no attached engine cache")
        storage = cache.storage
        if storage is None:
            raise RuntimeError("engine KV storage was released during staging receive")
        page_size = cache.page_size
        copier = manager_staging_copier(self.kv_manager)
        if copier is None:
            raise RuntimeError("SGLang staging receive has no attached engine Copier")
        staging_buffer = self.staging_allocator.buffer
        pool = staging_buffer.attach_cache(storage, cache.pages.chips, copier)
        if staging_offset % pool.layout.slot_bytes:
            raise ValueError("SGLang staging offset is not engine-page aligned")

        req_pool_idx = decode_req.req.kv.req_pool_idx
        token_start = decode_req.req.kv.cache_protected_len + page_start * page_size
        token_end = token_start + num_pages * page_size
        token_ids = self.scheduler.req_to_token_pool.req_to_token[
            req_pool_idx, token_start:token_end
        ]
        if token_ids.numel() != num_pages * page_size:
            raise ValueError("SGLang request table does not cover the received KV pages")
        page_ids = [int(token) // page_size for token in token_ids[::page_size].tolist()]
        if len(page_ids) != num_pages:
            raise ValueError("SGLang request table page count differs from staging")

        first_host_page = staging_offset // pool.layout.slot_bytes
        host_slots = tuple(range(first_host_page, first_host_page + num_pages))
        if host_slots[-1] >= pool.layout.capacity_pages:
            raise ValueError("SGLang staging scatter exceeds its HostCache capacity")
        return submit_packed_cache_pages(
            pool,
            cache,
            page_ids,
            host_slots,
            to_host=False,
        )
