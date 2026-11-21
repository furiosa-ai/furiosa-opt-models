"""The offloading connector's worker side on the engine cache, as vLLM's offloading worker runs it.

- `Worker` starts a step's stores in the next step, ends the stores a step flushes before its calls,
  and reports a load's request once the load landed.
- Without tiering, `Host` keeps each layer's host rows in one engine host buffer. With tiering,
  vLLM's shared mmap is the primary host copy and a pinned engine buffer is temporary DMA staging.
  A chunk holds `blocks_per_chunk` rows of every selected layer; adjacent block rows move together.
- A copy returns once submitted: the device orders it after the calls and copies that wrote its
  bytes and before the ones that write or read them later. `Jobs` keeps each copy from its
  submission until the job reports it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from time import monotonic
from typing import TYPE_CHECKING, final

import torch
from typing_extensions import override
from vllm.config import VllmConfig
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
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

import furiosa_opt_engine as engine
from furiosa_opt_engine.cache_transfer import (
    CachePageRoute,
    PendingTransfer,
    copy_bytes,
    copy_cache_pages,
    page_runs,
    transfer_cache_pages,
)
from furiosa_opt_engine.cache_transfer import (
    settle as settle_transfers,
)
from furiosa_opt_engine.cache_transfer import (
    submit as submit_transfers,
)
from furiosa_opt_vllm.cache import assigned, group_indices

if TYPE_CHECKING:
    from collections.abc import Iterable

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
        OffloadingConnectorMetadata,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.kv_offload.base import CanonicalPageMapping


@final
class Worker:
    """The connector's worker side over `host`, as vLLM's offloading worker runs it: a step's
    stores start in the next step, after the step's tokens are read."""

    def __init__(self, host: Host) -> None:
        self.host = host
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
        for job_id, job in metadata.load_jobs.items():
            assert isinstance(job.dst_spec, GPULoadStoreSpec)
            self.loads[job_id] = job.req_id
            _ = self.host.submit_load(job_id, job.src_spec, job.dst_spec)

    def defer(self, metadata: OffloadingConnectorMetadata) -> None:
        """Keep the stores of `metadata` for the next step."""
        self.deferred.extend(metadata.store_jobs.items())

    def submit(self) -> None:
        for job_id, job in self.deferred:
            assert isinstance(job.src_spec, GPULoadStoreSpec)
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
    chunk_offset: int
    mapping: CanonicalPageMapping | None


def _metadata_cache_views(
    cache: engine.decoder.Cache,
    config: KVCacheConfig,
    layer_owner: dict[int, tuple[int, int]],
) -> dict[str, torch.Tensor]:
    """Build metadata-only views with the exact per-page strides held by the engine."""
    views: dict[str, torch.Tensor] = {}
    for group_id, group in enumerate(config.kv_cache_groups):
        for name in group.layer_names:
            if not name.startswith("layers."):
                raise ValueError(f"vLLM layer name {name!r} is not an engine layer ID")
            try:
                layer_id = int(name.removeprefix("layers."))
            except ValueError as error:
                raise ValueError(f"vLLM layer name {name!r} is malformed") from error
            engine_owner = layer_owner.get(layer_id)
            if engine_owner is None:
                raise ValueError(f"engine cache metadata has no page format for {name}")
            engine_group_id, pool_index = engine_owner
            if engine_group_id != group_id:
                raise ValueError(f"engine cache group for {name} differs from vLLM's group")
            pool = cache.spec.pools[pool_index]
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

            # PageFormat is logical [token, K/V, head, feature]. Add the block-table axis
            # and reorder those logical dimensions into the upstream split-K/V tensor view.
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


@final
class Host(OffloadingWorker):
    """The host tier of `cache`, whose layers span `chips` chips: chunks of selected group pages and
    the jobs that copy blocks between them."""

    def __init__(
        self,
        cache: engine.decoder.Cache,
        chips: int,
        spec: CPUOffloadingSpec,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        selected_group_ids: tuple[int, ...],
        mmap_region: SharedOffloadRegion | None = None,
    ) -> None:
        if chips <= 0 or cache.blocks <= 0:
            raise ValueError("cache transfer requires positive chip and block counts")
        self.layers = cache.layers
        self.chips = chips
        self.blocks = spec.blocks_per_chunk
        self.chunks = spec.num_chunks
        self.row_stride = spec.kv_bytes_per_chunk
        self.canonical_layout = spec.config.canonical_layout
        self.region = mmap_region
        self.mapped = (
            memoryview(mmap_region.create_kv_memoryview()).cast("B")
            if mmap_region is not None
            else None
        )
        self.rank_offset = (
            mmap_region.rank * spec.cpu_page_size_per_worker
            if mmap_region is not None and mmap_region.rank is not None
            else 0
        )

        engine_group_indices = group_indices(cache.spec, kv_cache_config.kv_cache_groups)
        layer_owner: dict[int, tuple[int, int]] = {}
        for group in cache.spec.groups:
            for layer_id in group.layers:
                if not 0 <= layer_id < len(cache.layers):
                    raise ValueError(f"engine cache group names unknown layer {layer_id}")
                layer_owner[layer_id] = (engine_group_indices[group.vllm_group], group.pool)

        mappings: dict[str, CanonicalPageMapping] = {}
        if self.canonical_layout:
            if chips != 1:
                raise ValueError("canonical cache offload requires one engine chip per vLLM worker")
            mappings = derive_canonical_mappings(
                vllm_config,
                kv_cache_config,
                _metadata_cache_views(cache, kv_cache_config, layer_owner),
            )
            if spec.config.parallel.is_parallelism_agnostic and any(
                not mapping.parallelism_agnostic for mapping in mappings.values()
            ):
                raise ValueError(
                    "vLLM marked canonical pages portable but an engine layer mapping "
                    "is topology-specific"
                )

        self.routes: dict[int, tuple[LayerRoute, ...]] = {}
        self.group_offsets: dict[int, int] = {}
        group_offset = 0
        primary_bytes_per_chunk = 0
        for group_id in selected_group_ids:
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

            self.group_offsets[group_id] = group_offset
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
                mapping = mappings.get(name) if self.canonical_layout else None
                if self.canonical_layout and mapping is None:
                    raise ValueError(f"vLLM cannot derive a canonical mapping for {name}")
                if mapping is not None and (
                    mapping.local_page_size_bytes > page_bytes
                    or mapping.canonical_page_size_bytes <= 0
                ):
                    raise ValueError(f"canonical mapping for {name} exceeds its cache page")
                primary_page_bytes = (
                    mapping.canonical_page_size_bytes if mapping is not None else host_page_bytes
                )
                layer_routes.append(
                    LayerRoute(layer_id, page_bytes, primary_page_bytes, layer_offset, mapping)
                )
                layer_offset += primary_page_bytes * self.blocks
            self.routes[group_id] = tuple(layer_routes)
            group_offset += layer_offset
            primary_bytes_per_chunk += layer_offset

        if not selected_group_ids:
            raise ValueError("vLLM selected no cache groups for offloading")
        if self.chunks <= 0 or self.blocks <= 0 or self.row_stride <= 0:
            raise ValueError("offloading allocation cannot hold one cache chunk")
        worker_capacity = (
            spec.cpu_page_size_per_worker if mmap_region is not None else self.row_stride
        )
        if primary_bytes_per_chunk > worker_capacity:
            raise ValueError(
                "engine cache pages exceed vLLM's host chunk layout "
                f"({primary_bytes_per_chunk} > {worker_capacity} bytes)"
            )
        if mmap_region is not None and self.rank_offset + primary_bytes_per_chunk > self.row_stride:
            raise ValueError("engine cache pages exceed this worker's shared host row")

        self._staging_pool: dict[int, list[engine.HostBuffer]] = {}
        self._staging_pool_bytes = 0
        self._staging_pool_limit = worker_capacity

        # A tiering spec's scheduler manager reads the shared mmap in a different process. The
        # engine HostBuffer cannot alias that mmap, so keep it as the sole primary copy and use a
        # page-locked transfer buffer only while a DMA is in flight.
        self.rows = (
            {}
            if mmap_region is not None
            else {
                route.layer_id: engine.HostBuffer(self.chunks * self.blocks * route.host_page_bytes)
                for routes in self.routes.values()
                for route in routes
            }
        )
        self.jobs = Jobs()

    @override
    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        groups = self._groups(src_spec, dst_spec)
        self.jobs.submit(job_id, self.transfers(groups, store=True), self._size(groups))
        return True

    @override
    def submit_load(self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec) -> bool:
        groups = self._groups(dst_spec, src_spec)
        self.jobs.submit(job_id, self.transfers(groups, store=False), self._size(groups))
        return True

    @override
    def get_finished(self) -> list[TransferResult]:
        return self.jobs.finished()

    @override
    def wait(self, job_ids: set[int]) -> None:
        self.jobs.wait(job_ids)

    @override
    def shutdown(self) -> None:
        try:
            self.jobs.shutdown()
        finally:
            self._staging_pool.clear()
            self._staging_pool_bytes = 0
            if self.mapped is not None:
                self.mapped.release()
            if self.region is not None:
                self.region.cleanup()

    def transfers(self, groups: list[GroupWork], *, store: bool) -> Iterable[PendingTransfer]:
        """Copy selected logical groups to/from their engine subgroups and host chunk regions."""
        for work in groups:
            group_id = work.group_id
            for chunk_id, device_blocks, host_slots in work.chunks:
                for route in self.routes[group_id]:
                    layer = self.layers[route.layer_id]
                    if route.mapping is None and self.region is None:
                        host_pages = tuple(chunk_id * self.blocks + slot for slot in host_slots)
                        yield from transfer_cache_pages(
                            layer,
                            self.rows[route.layer_id],
                            device_blocks,
                            host_pages,
                            route.device_page_bytes,
                            self.chips,
                            to_host=store,
                        )
                        continue
                    for block, slot, run_length in page_runs(device_blocks, host_slots, self.chips):
                        device_view = layer.slice(
                            block * route.device_page_bytes,
                            run_length * route.device_page_bytes,
                        )
                        host_row = self._host_slice(group_id, chunk_id, slot, run_length, route)
                        if route.mapping is not None:
                            if store:
                                yield self._staged_transfer(
                                    run_length * route.device_page_bytes,
                                    device_view.to_host,
                                    commit=partial(
                                        _copy_mapping,
                                        host_row,
                                        mapping=route.mapping,
                                        page_count=run_length,
                                        local_page_stride=route.device_page_bytes,
                                        store=True,
                                    ),
                                )
                            else:
                                yield self._staged_transfer(
                                    run_length * route.device_page_bytes,
                                    lambda stage, device=device_view: stage.to_hbm(device),
                                    prepare=partial(
                                        _copy_mapping,
                                        source=host_row,
                                        mapping=route.mapping,
                                        page_count=run_length,
                                        local_page_stride=route.device_page_bytes,
                                        store=False,
                                    ),
                                )
                        elif self.region is None:
                            yield (
                                device_view.to_host(host_row)
                                if store
                                else host_row.to_hbm(device_view)
                            )
                        else:
                            if store:
                                yield self._staged_transfer(
                                    run_length * route.host_page_bytes,
                                    device_view.to_host,
                                    commit=lambda stage, destination=host_row: copy_bytes(
                                        destination, stage
                                    ),
                                )
                            else:
                                yield self._staged_transfer(
                                    run_length * route.host_page_bytes,
                                    lambda stage, device=device_view: stage.to_hbm(device),
                                    prepare=lambda stage, source=host_row: copy_bytes(
                                        stage, source
                                    ),
                                )

    def _staged_transfer(
        self,
        size: int,
        submit: Callable[[engine.HostBuffer], PendingTransfer],
        *,
        prepare: Callable[[engine.HostBuffer], None] | None = None,
        commit: Callable[[engine.HostBuffer], None] | None = None,
    ) -> _StagedTransfer:
        stage = self._acquire_staging_buffer(size)
        try:
            if prepare is not None:
                prepare(stage)
            transfer = submit(stage)
        except Exception:
            self._release_staging_buffer(stage)
            raise
        finish = partial(commit, stage) if commit is not None else None
        return _StagedTransfer(
            transfer,
            stage,
            finish,
            release=lambda: self._release_staging_buffer(stage),
        )

    def _acquire_staging_buffer(self, size: int) -> engine.HostBuffer:
        buffers = self._staging_pool.get(size)
        if buffers:
            self._staging_pool_bytes -= size
            stage = buffers.pop()
            if not buffers:
                del self._staging_pool[size]
            return stage
        return engine.HostBuffer(size)

    def _release_staging_buffer(self, stage: engine.HostBuffer) -> None:
        size = len(stage)
        if (
            size > self._staging_pool_limit
            or self._staging_pool_bytes + size > self._staging_pool_limit
        ):
            return
        self._staging_pool.setdefault(size, []).append(stage)
        self._staging_pool_bytes += size

    def _groups(self, device: GPULoadStoreSpec, host: LoadStoreSpec) -> list[GroupWork]:
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
            chunks: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []
            group_cursor = 0
            for chunk_position in range(chunks_needed):
                slot_start = first_slot if chunk_position == 0 else 0
                length = min(count - group_cursor, self.blocks - slot_start)
                chunks.append(
                    (
                        chunk_ids[chunk_offset + chunk_position],
                        group_blocks[group_cursor : group_cursor + length],
                        tuple(range(slot_start, slot_start + length)),
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
            len(device_blocks) * sum(route.host_page_bytes for route in self.routes[group.group_id])
            for group in groups
            for _, device_blocks, _ in group.chunks
        )

    def _shared_slice(self, offset: int, size: int) -> memoryview:
        if self.mapped is None:
            raise RuntimeError("shared host region is not initialized")
        end = offset + size
        if offset < 0 or size <= 0 or end > len(self.mapped):
            raise ValueError("host transfer exceeds the shared cache region")
        return self.mapped[offset:end]

    def _host_slice(
        self, group_id: int, chunk_id: int, slot: int, count: int, route: LayerRoute
    ) -> memoryview | engine.HostBuffer:
        size = count * route.host_page_bytes
        if self.region is None:
            offset = (chunk_id * self.blocks + slot) * route.host_page_bytes
            return self.rows[route.layer_id].slice(offset, size)
        offset = (
            chunk_id * self.row_stride
            + self.rank_offset
            + self.group_offsets[group_id]
            + route.chunk_offset
            + slot * route.host_page_bytes
        )
        return self._shared_slice(offset, size)


@dataclass(frozen=True, slots=True)
class GroupWork:
    group_id: int
    chunks: tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...]


@final
class _StagedTransfer:
    """A pinned staging buffer held until its DMA ends, then copied into shared primary memory."""

    def __init__(
        self,
        transfer: PendingTransfer,
        staging: engine.HostBuffer,
        commit: Callable[[], None] | None = None,
        release: Callable[[], None] | None = None,
    ) -> None:
        self.transfer = transfer
        self.staging = staging
        self.commit = commit
        self.release = release
        self.settled = False
        self.failure: Exception | None = None

    def done(self) -> bool:
        if self.settled:
            return True
        if self.failure is not None:
            raise self.failure
        try:
            if not self.transfer.done():
                return False
            self.transfer.wait()
            self._finish()
            return True
        except Exception as error:
            self.failure = error
            raise

    def wait(self) -> None:
        if self.settled:
            return
        if self.failure is not None:
            raise self.failure
        try:
            self.transfer.wait()
            self._finish()
        except Exception as error:
            self.failure = error
            raise

    def _finish(self) -> None:
        if self.commit is not None:
            self.commit()
        if self.release is not None:
            self.release()
        self.settled = True


def _copy_mapping(
    destination: memoryview | engine.HostBuffer,
    source: memoryview | engine.HostBuffer,
    mapping: CanonicalPageMapping,
    page_count: int,
    *,
    local_page_stride: int,
    store: bool,
) -> None:
    """Apply vLLM's certified layout runs through the framework-neutral page router."""
    routes = tuple(
        CachePageRoute(
            cache_offset=run.local_offset,
            host_offset=run.canonical_offset,
            fragment_bytes=run.fragment_size,
            fragments=run.num_fragments,
            cache_stride=run.local_stride,
            host_stride=run.canonical_stride,
        )
        for run in mapping.runs
    )
    copy_cache_pages(
        cache=source if store else destination,
        host=destination if store else source,
        routes=routes,
        page_count=page_count,
        cache_page_bytes=mapping.local_page_size_bytes,
        cache_page_stride=local_page_stride,
        host_page_bytes=mapping.canonical_page_size_bytes,
        host_page_stride=mapping.canonical_page_size_bytes,
        to_host=store,
    )


@dataclass(frozen=True, slots=True)
class Job:
    """A submitted store or load: its copies, their bytes and when they were submitted."""

    transfers: tuple[PendingTransfer, ...]
    size: int
    started: float

    def done(self) -> bool:
        """Whether every copy ended, without waiting; a failed copy it reaches raises."""
        return all(transfer.done() for transfer in self.transfers)

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
