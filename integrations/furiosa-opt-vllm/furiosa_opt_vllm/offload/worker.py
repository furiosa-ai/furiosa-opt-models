"""The offloading connector's worker side on the engine cache, as vLLM's offloading worker runs it.

- `Worker` starts a step's stores in the next step, ends the stores a step flushes before its calls,
  and reports a load's request once the load landed.
- `Host` keeps one pinned engine slot per device block, including all selected cache-group page
  ranges.
  Six native workers submit two-slot DMA batches, each batching every routed layer into one runtime
  call.
- Copies return after submission so later device work can queue behind them. `Jobs` keeps each copy
  until DMA completes and vLLM can reuse its source or destination blocks.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING, TypeAlias, final

import torch
from typing_extensions import override
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.canonical_mapping import (
    derive_canonical_mappings,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
    TransferJob,
)
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

import furiosa_opt_engine as engine
from furiosa_opt_engine.cache_transfer import PendingTransfer
from furiosa_opt_engine.cache_transfer import settle as settle_transfers
from furiosa_opt_engine.cache_transfer import (
    submit as submit_transfers,
)
from furiosa_opt_vllm.cache import assigned, group_indices
from furiosa_opt_vllm.offload import pool as host_pool

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
        OffloadingConnectorMetadata,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.kv_offload.base import CanonicalPageMapping


# vLLM's runner-side block descriptor is GPU-named, while this adapter runs on the NPU.
DeviceLoadStoreSpec: TypeAlias = GPULoadStoreSpec


@final
class Worker:
    """The connector's worker side over `host`, as vLLM's offloading worker runs it: a step's
    stores start in the next step, after the step's tokens are read."""

    def __init__(self, host: Host, *, coalescing: bool = False) -> None:
        self.host = host
        self.coalescing = coalescing
        self.deferred: list[tuple[int, TransferJob]] = []
        # job_id -> the request whose blocks the load fills.
        self.loads: dict[int, str] = {}
        self.completed = OffloadingWorkerMetadata()

    def flush(self, metadata: OffloadingConnectorMetadata) -> None:
        """Submit the deferred stores and the step's stores that `metadata` flushes, then wait for
        the flushed jobs, whose blocks the step hands to other requests."""
        flushed = metadata.jobs_to_flush or set()
        self.deferred.extend(
            (job_id, job)
            for job_id in flushed
            if (job := metadata.store_jobs.pop(job_id, None)) is not None
        )
        self.submit()
        self.host.wait(flushed)

    def start(self, metadata: OffloadingConnectorMetadata) -> None:
        """Submit the deferred stores, then the loads of `metadata`."""
        self.submit()
        if self.coalescing and metadata.load_jobs:
            jobs: list[tuple[int, LoadStoreSpec, DeviceLoadStoreSpec]] = []
            for job_id, job in metadata.load_jobs.items():
                assert isinstance(job.dst_spec, DeviceLoadStoreSpec)
                jobs.append((job_id, job.src_spec, job.dst_spec))
            self.host.submit_loads(jobs)
            self.loads.update((job_id, job.req_id) for job_id, job in metadata.load_jobs.items())
            return
        for job_id, job in metadata.load_jobs.items():
            assert isinstance(job.dst_spec, DeviceLoadStoreSpec)
            self.loads[job_id] = job.req_id
            _ = self.host.submit_load(job_id, job.src_spec, job.dst_spec)

    def defer(self, metadata: OffloadingConnectorMetadata) -> None:
        """Keep the stores of `metadata` for the next step."""
        self.deferred.extend(metadata.store_jobs.items())

    def submit(self) -> None:
        for job_id, job in self.deferred:
            assert isinstance(job.src_spec, DeviceLoadStoreSpec)
            _ = self.host.submit_store(job_id, job.src_spec, job.dst_spec)
        self.deferred.clear()

    def finished(self) -> set[str]:
        """The requests whose loads ended since the last call, each job's end recorded for
        `report`."""
        received: set[str] = set()
        for result in self.host.get_finished():
            request = self.loads.pop(result.job_id, None)
            stats = self.completed.transfer_stats
            assert result.transfer_size is not None and result.transfer_time is not None
            (stats.store if request is None else stats.load).record(
                result.transfer_size, result.transfer_time
            )
            self.completed.mark_completed(result.job_id)
            if request is not None:
                received.add(request)
        return received

    def report(self) -> OffloadingWorkerMetadata | None:
        """The jobs that ended since the last report, once each."""
        if not self.completed.completed_jobs:
            return None
        report, self.completed = self.completed, OffloadingWorkerMetadata()
        return report

    def shutdown(self) -> None:
        """Drop the stores not yet submitted, then end the submitted jobs."""
        self.deferred.clear()
        self.host.shutdown()


@dataclass(frozen=True, slots=True)
class LayerRoute:
    layer_id: int
    device_page_bytes: int
    host_page_bytes: int
    host_offset: int


def _metadata_cache_views(
    cache: engine.decoder.Cache,
    config: KVCacheConfig,
    layer_owner: dict[int, tuple[int, int]],
) -> dict[str, torch.Tensor]:
    """Build metadata-only views for vLLM's canonical-layout validator."""
    views: dict[str, torch.Tensor] = {}
    for group_id, group in enumerate(config.kv_cache_groups):
        for name in group.layer_names:
            if not name.startswith("layers."):
                raise ValueError(f"vLLM layer name {name!r} is not an engine layer ID")
            try:
                layer_id = int(name.removeprefix("layers."))
            except ValueError as error:
                raise ValueError(f"vLLM layer name {name!r} is malformed") from error
            owner = layer_owner.get(layer_id)
            if owner is None or owner[0] != group_id:
                raise ValueError(f"engine cache metadata for {name} differs from vLLM's group")
            pool = cache.spec.pools[owner[1]]
            spec = assigned(group, name)
            element_bytes = pool.element_bytes
            shape = tuple(pool.shape)
            strides = tuple(pool.strides)
            expected_shape = (spec.block_size, 2, spec.num_kv_heads, spec.head_size)
            if shape != expected_shape or element_bytes != spec.dtype.itemsize:
                raise ValueError(
                    f"engine page format for {name} cannot describe vLLM's attention page"
                )
            if len(strides) != 4 or any(stride % element_bytes for stride in strides):
                raise ValueError(f"engine page strides for {name} do not align to its element size")
            if pool.bytes % element_bytes:
                raise ValueError(f"engine page size for {name} does not align to its element size")

            size = (config.num_blocks, 2, shape[0], shape[2], shape[3])
            stride = (
                pool.bytes // element_bytes,
                strides[1] // element_bytes,
                strides[0] // element_bytes,
                strides[2] // element_bytes,
                strides[3] // element_bytes,
            )
            views[name] = torch.empty_strided(size, stride, dtype=spec.dtype, device="meta")
    return views


def _identity_mapping(mapping: CanonicalPageMapping, page_bytes: int) -> bool:
    """Whether vLLM's canonical page is exactly the engine page byte-for-byte."""
    return (
        mapping.local_page_size_bytes == page_bytes
        and mapping.canonical_page_size_bytes == page_bytes
        and mapping.num_writers == 1
        and mapping.writer_index == 0
        and bool(mapping.runs)
        and all(
            run.local_offset == run.canonical_offset and run.local_stride == run.canonical_stride
            for run in mapping.runs
        )
    )


def _require_identity_mapping(
    mapping: CanonicalPageMapping | None, layer_name: str, page_bytes: int
) -> None:
    """Refuse canonical pages that would need a gather or scatter before DMA."""
    if mapping is None or not _identity_mapping(mapping, page_bytes):
        raise ValueError(
            f"vLLM canonical page for {layer_name} is not the engine page byte-for-byte"
        )


@final
class Host(OffloadingWorker):
    """Pinned slots for vLLM blocks and the engine queue that copies cache pages."""

    def __init__(
        self,
        cache: engine.decoder.Cache,
        chips: int,
        spec: CPUOffloadingSpec,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        selected_group_ids: tuple[int, ...],
    ) -> None:
        if chips <= 0 or cache.blocks <= 0:
            raise ValueError("cache transfer requires positive chip and block counts")
        self.layers = cache.layers
        self.blocks = spec.blocks_per_chunk
        self.chunks = spec.num_chunks
        self.row_stride = spec.kv_bytes_per_chunk
        engine_group_indices = group_indices(cache.spec, kv_cache_config.kv_cache_groups)
        layer_owner: dict[int, tuple[int, int]] = {}
        for group in cache.spec.groups:
            for layer_id in group.layers:
                if not 0 <= layer_id < len(cache.layers):
                    raise ValueError(f"engine cache group names unknown layer {layer_id}")
                layer_owner[layer_id] = (engine_group_indices[group.logical_group], group.pool)

        mappings: dict[str, CanonicalPageMapping] = {}
        if spec.config.canonical_layout:
            if chips != 1:
                raise ValueError("canonical_layout requires one engine chip per vLLM worker")
            mappings = derive_canonical_mappings(
                vllm_config,
                kv_cache_config,
                _metadata_cache_views(cache, kv_cache_config, layer_owner),
            )
            if spec.config.parallel.is_parallelism_agnostic and any(
                not mapping.parallelism_agnostic for mapping in mappings.values()
            ):
                raise ValueError(
                    "vLLM marked canonical pages portable but the engine mapping is "
                    "topology-specific"
                )

        self.routes: dict[int, tuple[LayerRoute, ...]] = {}
        self.group_indices: dict[int, int] = {}
        layer_order: list[tuple[int, int, int, int]] = []
        group_offset = 0
        for group_index, group_id in enumerate(selected_group_ids):
            if not 0 <= group_id < len(kv_cache_config.kv_cache_groups):
                raise ValueError(f"offloading selected unknown KV cache group {group_id}")
            kv_group = kv_cache_config.kv_cache_groups[group_id]
            names = tuple(kv_group.layer_names)
            try:
                expected = {
                    int(name.removeprefix("layers."))
                    for name in names
                    if name.startswith("layers.")
                }
            except ValueError as error:
                raise ValueError(f"vLLM group {group_id} has a malformed layer name") from error
            if len(expected) != len(names):
                raise ValueError(f"vLLM group {group_id} has a malformed layer name")
            actual = {layer_id for layer_id, (owner, _) in layer_owner.items() if owner == group_id}
            if expected != actual:
                raise ValueError(
                    f"engine cache layers for KV group {group_id} differ from vLLM's group"
                )

            self.group_indices[group_id] = group_index
            layer_routes: list[LayerRoute] = []
            layer_offset = 0
            for name in names:
                layer_id = int(name.removeprefix("layers."))
                layer = cache.layers[layer_id]
                if len(layer) % cache.blocks:
                    raise ValueError(f"engine cache layer {name} has a partial block")
                page_bytes = len(layer) // cache.blocks
                host_page_bytes = page_bytes * chips
                if page_bytes == 0 or page_bytes % 8:
                    raise ValueError(f"engine cache layer {name} has an invalid page size")
                owner, pool_index = layer_owner[layer_id]
                described_page_bytes = cache.spec.pools[pool_index].bytes
                if owner != group_id or described_page_bytes != page_bytes:
                    raise ValueError(
                        f"engine cache metadata for {name} disagrees with its allocation"
                    )
                if assigned(kv_group, name).page_size_bytes != page_bytes:
                    raise ValueError(
                        f"vLLM page size for {name} differs from the engine allocation"
                    )
                if spec.config.canonical_layout:
                    _require_identity_mapping(mappings.get(name), name, page_bytes)
                layer_routes.append(
                    LayerRoute(layer_id, page_bytes, host_page_bytes, group_offset + layer_offset)
                )
                layer_order.append((layer_id, page_bytes, group_index, group_offset + layer_offset))
                layer_offset += host_page_bytes
            self.routes[group_id] = tuple(layer_routes)
            group_offset += layer_offset

        if not selected_group_ids:
            raise ValueError("vLLM selected no cache groups for offloading")
        if self.chunks <= 0 or self.blocks <= 0 or self.row_stride <= 0:
            raise ValueError("offloading allocation cannot hold one cache chunk")
        if group_offset * self.blocks > self.row_stride:
            raise ValueError(
                "engine cache pages exceed vLLM's host chunk layout "
                f"({group_offset * self.blocks} > {self.row_stride} bytes)"
            )
        self.copier = engine.Copier(
            [cache.layers[layer_id] for layer_id, _, _, _ in layer_order],
            [page_bytes for _, page_bytes, _, _ in layer_order],
            [offset for _, _, _, offset in layer_order],
            [group for _, _, group, _ in layer_order],
            group_offset,
        )
        self.host_cache = engine.HostCache(self.copier, self.chunks * self.blocks)
        self.engine_id = spec.config.engine_id
        self.jobs = Jobs()
        try:
            host_pool.register(self.engine_id, self.host_cache)
        except Exception:
            self.copier.shutdown()
            raise

    @override
    def submit_store(
        self, job_id: int, src_spec: DeviceLoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        groups = self._groups(src_spec, dst_spec)
        self.jobs.submit(job_id, self.transfers(groups, store=True), self._size(groups))
        return True

    @override
    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: DeviceLoadStoreSpec
    ) -> bool:
        groups = self._groups(dst_spec, src_spec)
        self.jobs.submit(job_id, self.transfers(groups, store=False), self._size(groups))
        return True

    def submit_loads(
        self,
        jobs: Iterable[tuple[int, LoadStoreSpec, DeviceLoadStoreSpec]],
    ) -> None:
        """Coalesce every step load into one engine copy while retaining each vLLM job."""
        grouped = [
            (job_id, self._groups(dst_spec, src_spec)) for job_id, src_spec, dst_spec in jobs
        ]
        pages = [page for _, groups in grouped for page in self.routed_pages(groups)]
        transfer = self.copier.copy_in(pages, self.host_cache, coalesce=True)
        transfer.submitted()
        self.jobs.submit_batch(
            ((job_id, self._size(groups)) for job_id, groups in grouped),
            (transfer,),
        )

    @override
    def get_finished(self) -> list[TransferResult]:
        return self.jobs.finished()

    @override
    def wait(self, job_ids: set[int]) -> None:
        self.jobs.wait(job_ids)

    @override
    def shutdown(self) -> None:
        try:
            try:
                self.jobs.shutdown()
            finally:
                self.copier.shutdown()
        finally:
            host_pool.unregister(self.engine_id, self.host_cache)

    def transfers(self, groups: list[GroupWork], *, store: bool) -> Iterable[PendingTransfer]:
        """Route each pinned host slot across its cache groups in two-page DMA batches."""
        routed_pages = self.routed_pages(groups)
        transfer = (
            self.copier.copy_out(routed_pages, self.host_cache)
            if store
            else self.copier.copy_in(routed_pages, self.host_cache)
        )
        transfer.submitted()
        yield transfer

    def routed_pages(self, groups: list[GroupWork]) -> list[tuple[int, list[tuple[int, int]]]]:
        """Route each job's host slots across its selected cache groups."""
        pages: dict[int, list[tuple[int, int]]] = {}
        for work in groups:
            group_index = self.group_indices[work.group_id]
            for chunk in work.chunks:
                for block, slot in zip(chunk.device_blocks, chunk.host_slots, strict=True):
                    host_slot = chunk.chunk_id * self.blocks + slot
                    pages.setdefault(host_slot, []).append((group_index, block))
        # Preserve the framework's page-pair order: the engine forms two-page DMA chunks from
        # consecutive pairs, matching the cache transfer core's reference implementation.
        return list(pages.items())

    def _groups(self, device: DeviceLoadStoreSpec, host: LoadStoreSpec) -> list[GroupWork]:
        if not isinstance(host, CPULoadStoreSpec):
            raise TypeError(f"expected CPULoadStoreSpec, got {type(host).__name__}")
        group_sizes = device.group_sizes
        if group_sizes is None:
            group_sizes = (len(device.block_ids),)
        block_indices = device.block_indices
        if block_indices is None:
            block_indices = (0,)
        group_sizes = tuple(int(size) for size in group_sizes)
        block_indices = tuple(int(index) for index in block_indices)
        if len(group_sizes) != len(self.routes) or len(block_indices) != len(group_sizes):
            raise ValueError("vLLM transfer group ranges differ from selected cache groups")
        if sum(group_sizes) != len(device.block_ids):
            raise ValueError("vLLM transfer block IDs do not match its group ranges")
        if any(size < 0 for size in group_sizes) or any(index < 0 for index in block_indices):
            raise ValueError("vLLM transfer has a negative group size or block index")
        chunk_ids = tuple(int(chunk) for chunk in host.block_ids)
        if any(chunk < 0 or chunk >= self.chunks for chunk in chunk_ids):
            raise ValueError("vLLM transfer names a host chunk outside its allocation")

        groups: list[GroupWork] = []
        block_offset = 0
        chunk_offset = 0
        for group_id, count, start_index in zip(
            self.routes, group_sizes, block_indices, strict=True
        ):
            group_blocks = tuple(
                int(block) for block in device.block_ids[block_offset : block_offset + count]
            )
            block_offset += count
            if any(
                block < 0 or block * route.device_page_bytes >= len(self.layers[route.layer_id])
                for route in self.routes[group_id]
                for block in group_blocks
            ):
                raise ValueError(f"vLLM transfer names a block outside engine group {group_id}")
            if count == 0:
                groups.append(GroupWork(group_id, ()))
                continue

            first_slot = start_index % self.blocks
            chunks_needed = (first_slot + count + self.blocks - 1) // self.blocks
            if chunk_offset + chunks_needed > len(chunk_ids):
                raise ValueError("vLLM host chunks do not cover the requested block range")
            chunks: list[ChunkWork] = []
            group_cursor = 0
            for chunk_position in range(chunks_needed):
                slot_start = first_slot if chunk_position == 0 else 0
                length = min(count - group_cursor, self.blocks - slot_start)
                chunks.append(
                    ChunkWork(
                        chunk_id=chunk_ids[chunk_offset + chunk_position],
                        device_blocks=group_blocks[group_cursor : group_cursor + length],
                        host_slots=tuple(range(slot_start, slot_start + length)),
                    )
                )
                group_cursor += length
            if group_cursor != count:
                raise ValueError(f"vLLM transfer ranges do not cover KV group {group_id}")
            groups.append(GroupWork(group_id, tuple(chunks)))
            chunk_offset += chunks_needed
        if chunk_offset != len(chunk_ids):
            raise ValueError("vLLM host chunks do not match the requested group ranges")
        return groups

    def _size(self, groups: list[GroupWork]) -> int:
        return sum(
            len(chunk.device_blocks)
            * sum(route.host_page_bytes for route in self.routes[group.group_id])
            for group in groups
            for chunk in group.chunks
        )


@dataclass(frozen=True, slots=True)
class ChunkWork:
    chunk_id: int
    device_blocks: tuple[int, ...]
    host_slots: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GroupWork:
    group_id: int
    chunks: tuple[ChunkWork, ...]


@dataclass(frozen=True, slots=True)
class Job:
    """A submitted store or load: its copies, their bytes and when they were submitted."""

    transfers: tuple[PendingTransfer, ...]
    size: int
    started: float

    def done(self) -> bool:
        """Whether every copy ended; drains all copies before raising a failure."""
        for transfer in self.transfers:
            try:
                if not transfer.done():
                    return False
            except Exception as error:
                failure = self.settle()
                if failure is not None and failure is not error:
                    raise failure from error
                raise
        return True

    def settle(self) -> Exception | None:
        """Wait until every copy ended, then return the first failure among them."""
        return settle_transfers(self.transfers)


@final
class Jobs:
    """Submitted jobs, each reported once after its copies ended.

    A copy that fails raises from the call that finds it, once every copy that call waits for
    ended: it poisoned the memory it touched, and vLLM asserts that each job succeeded.
    """

    def __init__(self) -> None:
        self.running: dict[int, Job] = {}

    def submit(self, job_id: int, transfers: Iterable[PendingTransfer], size: int) -> None:
        """Run job `job_id`, whose copies move `size` bytes: `transfers` submits each copy as it
        is read, and the job keeps it at once. A copy that fails to submit raises once the copies
        submitted before it ended."""
        started = monotonic()
        self.running[job_id] = Job(submit_transfers(transfers), size, started)

    def submit_batch(
        self,
        jobs: Iterable[tuple[int, int]],
        transfers: Iterable[PendingTransfer],
    ) -> None:
        """Submit one shared copy completion for several independently reported jobs."""
        started = monotonic()
        copies = submit_transfers(transfers)
        for job_id, size in jobs:
            self.running[job_id] = Job(copies, size, started)

    def finished(self) -> list[TransferResult]:
        """The jobs whose copies ended since the last call, without waiting."""
        ended = [(job_id, job) for job_id, job in self.running.items() if job.done()]
        for job_id, _ in ended:
            del self.running[job_id]
        return [
            TransferResult(job_id, True, job.size, monotonic() - job.started)
            for job_id, job in ended
        ]

    def wait(self, job_ids: set[int]) -> None:
        """Wait until the running jobs of `job_ids` ended; `finished` still reports them."""
        self.drain(job_ids & self.running.keys())

    def shutdown(self) -> None:
        """Wait until every running job ended, then forget them."""
        try:
            self.drain(set(self.running))
        finally:
            self.running.clear()

    def drain(self, job_ids: set[int]) -> None:
        """Wait for each copy of the jobs of `job_ids`, then raise the first failure among them."""
        failures = [self.running[job_id].settle() for job_id in sorted(job_ids)]
        failure = next((error for error in failures if error is not None), None)
        if failure is not None:
            raise failure
