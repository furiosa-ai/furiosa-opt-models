"""vLLM's native Mooncake Store connector over bounded engine cache staging."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, fields, is_dataclass
from functools import partial
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import override
from vllm import envs
from vllm.distributed.kv_events import BlockStored
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector import (
    MooncakeStoreConnector,
    MooncakeStoreKVEvents,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    MooncakeStoreConnectorMetadata,
    MooncakeStoreWorkerMetadata,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import (
    MooncakeStoreScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (
    MooncakeStoreWorker,
    _estimate_disk_offload_staging_bytes,
    _get_replica_tiers_by_key,
    _get_usable_disk_offload_buffer_budget_bytes,
    _log_mooncake_load_tier_summary,
    _make_mooncake_group_id,
    _split_disk_offload_load_batches,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

import furiosa_opt_engine as engine
from furiosa_opt_engine._host_region import HostRegion, HostRegionLease, HostRegionRegistrar
from furiosa_opt_engine._mooncake import (
    ENGINE_PAGE_FORMAT_VERSION,
    MooncakeIOGuard,
    UnconfirmedMooncakeIO,
    require_confirmed_exists,
    require_confirmed_get,
    require_confirmed_put,
)
from furiosa_opt_engine.cache_transfer import (
    BoundedCacheStaging,
    CacheStagingLease,
    LayerPageRoutes,
    cache_staging_capacity,
)
from furiosa_opt_vllm.cache import assigned, group_indices

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = logging.getLogger(__name__)

_DEFAULT_STAGING_BUFFER_SIZE_MB = 1024
_DEFAULT_STORE_THREADS = 1
_LEASE_WAIT_SECONDS = 0.1


@dataclass(frozen=True, slots=True)
class _PageOperation:
    group: int
    block_id: int
    key: str
    start: int
    end: int
    block_hash: Any


def _cache_signature(vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> str:
    """Identify engine byte format and topology in upstream's cache-key namespace."""

    def spec_signature(spec: Any) -> object:
        if isinstance(spec, UniformTypeKVCacheSpecs):
            return {
                "type": type(spec).__qualname__,
                "layers": [
                    (name, spec_signature(layer_spec))
                    for name, layer_spec in sorted(spec.kv_cache_specs.items())
                ],
            }
        values = (
            {field.name: repr(getattr(spec, field.name)) for field in fields(spec)}
            if is_dataclass(spec)
            else {"repr": repr(spec)}
        )
        return {"type": type(spec).__qualname__, "fields": values}

    cache_config = vllm_config.cache_config
    resolved_layout = cache_config.get_resolved_kv_cache_layout()
    signature = {
        "page_format": ENGINE_PAGE_FORMAT_VERSION,
        "layout": getattr(resolved_layout, "name", str(resolved_layout)),
        "cache_dtype": str(cache_config.cache_dtype),
        "tensor_parallel_size": vllm_config.parallel_config.tensor_parallel_size,
        "pipeline_parallel_size": vllm_config.parallel_config.pipeline_parallel_size,
        "groups": [
            {
                "layers": sorted(group.layer_names),
                "spec": spec_signature(group.kv_cache_spec),
            }
            for group in kv_cache_config.kv_cache_groups
        ],
    }
    encoded = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _namespace_cache_prefix(vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> None:
    transfer = vllm_config.kv_transfer_config
    if transfer is None:
        raise ValueError("Furiosa Mooncake Store requires kv_transfer_config")
    extra = transfer.kv_connector_extra_config
    prefix = extra.get("cache_prefix", "")
    if not isinstance(prefix, str):
        raise TypeError("Mooncake cache_prefix must be a string")
    marker = f"{ENGINE_PAGE_FORMAT_VERSION}-{_cache_signature(vllm_config, kv_cache_config)}"
    if prefix == marker or prefix.startswith(f"{marker}:"):
        return
    extra["cache_prefix"] = f"{marker}:{prefix}" if prefix else marker


def _staging_budget_bytes(extra: dict[str, Any]) -> int:
    value = extra.get("staging_buffer_size_mb", _DEFAULT_STAGING_BUFFER_SIZE_MB)
    if type(value) is not int or value <= 0:
        raise ValueError("staging_buffer_size_mb must be a positive integer")
    return value * 1024 * 1024


def _oversized_disk_groups(
    groups: Sequence[Any],
    host_page_bytes: Sequence[int],
    disk_budget_bytes: int | None,
) -> frozenset[int]:
    """Identify cache groups whose single key exceeds Mooncake's raw disk budget."""
    if disk_budget_bytes is None:
        return frozenset()

    oversized_groups: set[int] = set()
    for group_index, group in enumerate(groups):
        page_sizes = [host_page_bytes[_layer_id(name)] for name in group.layer_names]
        if not page_sizes:
            raise ValueError(f"Mooncake cache group {group_index} has no layers")
        key_staging_bytes = _estimate_disk_offload_staging_bytes(page_sizes)
        if key_staging_bytes > disk_budget_bytes:
            oversized_groups.add(group_index)
    return frozenset(oversized_groups)


def _load_batches(
    keys: list[str],
    addresses: list[list[int]],
    sizes: list[list[int]],
    disk_budget_bytes: int | None,
) -> tuple[list[tuple[list[str], list[list[int]], list[list[int]], list[int]]], str | None]:
    """Apply upstream Mooncake's DirectIO budget splitter and retain page indices."""
    if disk_budget_bytes is None:
        return [(keys, addresses, sizes, list(range(len(keys))))], None

    split_batches, oversized_key = _split_disk_offload_load_batches(
        keys,
        addresses,
        sizes,
        _get_usable_disk_offload_buffer_budget_bytes(disk_budget_bytes),
        disk_budget_bytes,
    )
    if oversized_key is not None:
        return [], oversized_key

    index_by_addresses = {tuple(value): index for index, value in enumerate(addresses)}
    return [
        (
            batch_keys,
            batch_addresses,
            batch_sizes,
            [index_by_addresses[tuple(value)] for value in batch_addresses],
        )
        for batch_keys, batch_addresses, batch_sizes in split_batches
    ], None


def _attention_specs(spec: Any) -> tuple[Any, ...]:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return tuple(spec.kv_cache_specs.values())
    return (spec,)


def _layer_id(name: str) -> int:
    prefix = "layers."
    if not name.startswith(prefix):
        raise ValueError(f"Mooncake KV layer name {name!r} is not an engine layer")
    try:
        return int(name.removeprefix(prefix))
    except ValueError:
        raise ValueError(f"Mooncake KV layer name {name!r} is malformed") from None


class FuriosaMooncakeStoreWorker(MooncakeStoreWorker):
    """Keep upstream Store keys and lookup; move engine pages through one pinned staging slab."""

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> None:
        transfer = vllm_config.kv_transfer_config
        extra = transfer.kv_connector_extra_config if transfer is not None else {}
        if extra.get("custom_mem_pool"):
            raise ValueError("Furiosa Mooncake Store does not support Mooncake custom_mem_pool")
        if extra.get("load_async", True) is not True:
            raise ValueError("Furiosa Mooncake Store requires load_async=true")
        super().__init__(vllm_config, kv_cache_config)
        self._vllm_config = vllm_config
        self._cache: engine.decoder.Cache | None = None
        self._staging: BoundedCacheStaging | None = None
        self._staging_region: HostRegionLease | None = None
        self._staging_address: int | None = None
        self._unsafe_staging_leases: list[CacheStagingLease] = []
        self._region_registrar = HostRegionRegistrar(
            self._register_mooncake_region, self._unregister_mooncake_region
        )
        self._pointer_io = MooncakeIOGuard()
        self._load_workers = self.num_recv_threads
        self._store_workers = _DEFAULT_STORE_THREADS
        self._load_executor = ThreadPoolExecutor(
            max_workers=self._load_workers,
            thread_name_prefix="furiosa_vllm_mooncake_get",
        )
        self._store_executor = ThreadPoolExecutor(
            max_workers=self._store_workers,
            thread_name_prefix="furiosa_vllm_mooncake_put",
        )
        self._lease_slots = threading.BoundedSemaphore(1)
        self._load_futures: dict[str, Future[bool]] = {}
        self._store_futures: dict[int, tuple[str, Future[bool]]]
        self._store_futures = {}
        self._load_failures: set[str] = set()
        self._saved_offsets: dict[str, int] = {}
        self._saved_hashes: dict[str, tuple[Any, ...]] = {}
        self._pending_stores: dict[str, int] = {}
        self._retired_requests: set[str] = set()
        self._state_lock = threading.Lock()
        self._save_locks = tuple(threading.Lock() for _ in range(64))
        self._kv_events_lock = threading.Lock()
        self._kv_events: list[BlockStored] = []
        self._load_errors_lock = threading.Lock()
        self._load_error_block_ids: set[int] = set()
        self._max_batch_pages = 1
        self._disk_oversized_groups: set[int] = set()
        self._close_lock = threading.Lock()
        self._closing = False
        self._closed = False

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        """Ignore vLLM's empty host-tensor registration; engine cache is bound explicitly."""
        if kv_caches:
            raise ValueError(
                "Furiosa Mooncake Store registers the engine cache, not vLLM torch KV tensors"
            )

    def register_engine_cache(self, cache: engine.decoder.Cache, chips: int) -> None:
        if self._closed or self._closing:
            raise RuntimeError("Furiosa Mooncake Store worker is closed")
        if self._staging is not None:
            raise RuntimeError("Furiosa Mooncake staging was already registered")
        if self._capacity_only:
            return
        self._validate_engine_cache(cache, chips)
        if self.store_tp_size is not None:
            raise ValueError(
                "Furiosa Mooncake Store uses one object per engine cache group; "
                "store_tp_size is unsupported"
            )

        transfer = self._vllm_config.kv_transfer_config
        if transfer is None:
            raise RuntimeError("Mooncake KV transfer config disappeared after connector setup")
        extra = transfer.kv_connector_extra_config
        budget_bytes = _staging_budget_bytes(extra)
        capacity_pages = cache_staging_capacity(
            cache,
            chips,
            budget_bytes,
            max_pages=min(cache.blocks, self._kv_cache_config.num_blocks),
        )
        staging = BoundedCacheStaging(cache, chips, capacity_pages)
        region: HostRegionLease | None = None
        try:
            region = staging.lease_host_region()
            self._region_registrar.register((region,))
        except BaseException as error:
            if region is not None:
                region.release()
            try:
                staging.release_engine_cache(cache)
            except BaseException as cleanup_error:
                error.add_note(f"Mooncake staging cleanup failed: {cleanup_error!r}")
            raise

        assert region is not None
        self._cache = cache
        self._staging = staging
        self._staging_region = region
        self._staging_address = region.region.address
        active_workers = self._load_workers + self._store_workers
        self._lease_slots = threading.BoundedSemaphore(capacity_pages)
        self._max_batch_pages = max(1, capacity_pages // active_workers)
        self._disk_oversized_groups = _oversized_disk_groups(
            self._kv_cache_groups,
            staging.host_page_bytes,
            self.disk_offload_buffer_budget_bytes,
        )
        self.num_blocks = cache.blocks

    def _validate_engine_cache(self, cache: engine.decoder.Cache, chips: int) -> None:
        if chips <= 0 or cache.blocks <= 0:
            raise ValueError("Furiosa Mooncake requires positive chip and block counts")
        parallel = self._vllm_config.parallel_config
        if parallel.rank != 0 or parallel.world_size != parallel.tensor_parallel_size:
            raise ValueError(
                "Furiosa Mooncake requires one worker to own every tensor-parallel rank"
            )
        if chips != parallel.tensor_parallel_size:
            raise ValueError(
                f"engine cache spans {chips} chips, but vLLM configured tensor_parallel_size="
                f"{parallel.tensor_parallel_size}"
            )
        if cache.blocks != self._kv_cache_config.num_blocks:
            raise ValueError("vLLM and engine cache block counts differ")
        all_vllm_groups = self._kv_cache_config.kv_cache_groups
        store_group_ids = self._kv_cache_config.prefix_cacheable_group_ids
        store_groups = [all_vllm_groups[group_id] for group_id in store_group_ids]
        if len(self._kv_cache_groups) != len(store_groups):
            raise ValueError("vLLM Mooncake Store groups differ from prefix-cacheable groups")
        for group_id in self._kv_cache_config.prefix_cacheable_group_ids:
            group = all_vllm_groups[group_id]
            for spec in _attention_specs(group.kv_cache_spec):
                if not isinstance(spec, (FullAttentionSpec, SlidingWindowSpec)):
                    raise ValueError(
                        "Furiosa Mooncake Store supports Paged full and sliding attention; "
                        f"got {type(spec).__name__}"
                    )

        engine_to_vllm_group = group_indices(cache.spec, all_vllm_groups)
        vllm_to_store_group = {
            vllm_group_id: store_group_index
            for store_group_index, vllm_group_id in enumerate(store_group_ids)
        }
        layers_by_group: list[list[int]] = [[] for _ in self._kv_cache_groups]
        pool_by_layer: dict[int, int] = {}
        for engine_group in cache.spec.groups:
            try:
                vllm_group_id = engine_to_vllm_group[engine_group.logical_group]
            except KeyError:
                raise ValueError(
                    f"engine cache has unknown logical group {engine_group.logical_group}"
                ) from None
            store_group_index = vllm_to_store_group.get(vllm_group_id)
            if store_group_index is not None:
                layers_by_group[store_group_index].extend(engine_group.layers)
            if not 0 <= engine_group.pool < len(cache.spec.pools):
                raise ValueError(f"engine cache group has unknown pool {engine_group.pool}")
            for layer in engine_group.layers:
                if layer in pool_by_layer:
                    raise ValueError(f"engine cache layer {layer} appears in multiple groups")
                pool_by_layer[layer] = engine_group.pool

        for group_index, group in enumerate(self._kv_cache_groups):
            expected_layers = tuple(_layer_id(name) for name in group.layer_names)
            actual_layers = tuple(sorted(layers_by_group[group_index]))
            if len(set(actual_layers)) != len(actual_layers) or set(actual_layers) != set(
                expected_layers
            ):
                raise ValueError(
                    f"engine and vLLM layer membership differs for group {group_index}"
                )
            for name, layer in zip(group.layer_names, expected_layers, strict=True):
                spec = assigned(group, name)
                expected_page_bytes = spec.page_size_bytes * chips
                try:
                    pool_index = pool_by_layer[layer]
                except KeyError:
                    raise ValueError(f"engine cache has no storage pool for {name}") from None
                if cache.spec.pools[pool_index].bytes * chips != expected_page_bytes:
                    raise ValueError(f"vLLM and engine cache page sizes differ for {name}")

    def _register_mooncake_region(self, region: HostRegion) -> None:
        result = self.store.register_buffer(region.address, region.size)
        if result != 0:
            raise RuntimeError(
                f"Mooncake failed to register staging slab {region.address:#x} "
                f"({region.size} bytes): {result}"
            )

    def _unregister_mooncake_region(self, region: HostRegion) -> None:
        result = self.store.unregister_buffer(region.address)
        if result != 0:
            raise RuntimeError(
                f"Mooncake failed to unregister staging slab {region.address:#x}: {result}"
            )

    def _require_staging(self) -> tuple[engine.decoder.Cache, BoundedCacheStaging]:
        self._pointer_io.ensure_safe()
        if self._closed or self._closing or self._cache is None or self._staging is None:
            raise RuntimeError("Furiosa Mooncake engine cache has not been registered")
        return self._cache, self._staging

    def _operations(
        self,
        request: ReqMeta,
        *,
        load: bool,
        save_start: int = 0,
    ) -> list[_PageOperation]:
        if load:
            load_spec = request.load_spec
            if load_spec is None:
                return []
            token_len = load_spec.token_len
            mask_num = load_spec.vllm_cached_tokens // self.block_size * self.block_size
            masks = self.coord.load_mask(request.block_hashes, token_len)
            tail_boundaries = {
                boundary.group_id: boundary.num_tokens for boundary in load_spec.tail_key_boundaries
            }
        else:
            token_len = request.token_len_chunk
            masks = self.coord.store_mask(
                token_len,
                save_start,
                num_prompt_tokens=request.num_prompt_tokens,
            )
            mask_num = save_start
            tail_boundaries = {}

        if len(request.block_ids) != len(self.token_dbs):
            raise ValueError("Mooncake request block groups differ from cache groups")
        operations: list[_PageOperation] = []
        for group_index, database in enumerate(self.token_dbs):
            group = self._kv_cache_groups[group_index]
            if not group.kv_cache_spec.prefix_cacheable:
                continue
            shards = tuple(database.store_layout.local_shard_ids)
            if len(shards) != 1:
                raise ValueError("Furiosa Mooncake Store requires one local Store shard")
            group_blocks = request.block_ids[group_index]
            mask = masks[group_index]
            for start, end, block_hash in database.process_tokens(
                token_len,
                request.block_hashes,
                mask_num=mask_num,
                chunk_mask=mask if not load else None,
            ):
                chunk_index = start // database.block_size
                if load:
                    if chunk_index >= len(mask) or not mask[chunk_index]:
                        continue
                    boundary = tail_boundaries.get(group_index) if end == token_len else None
                    if boundary is not None:
                        hash_index = boundary // database.hash_block_size - 1
                        if not 0 <= hash_index < len(request.block_hashes):
                            raise ValueError("Mooncake load tail boundary has no block hash")
                        block_hash = request.block_hashes[hash_index]
                if chunk_index >= len(group_blocks):
                    raise ValueError("Mooncake request has fewer blocks than its token chunks")
                block_id = group_blocks[chunk_index]
                if block_id == NULL_BLOCK_ID:
                    continue
                shard = shards[0]
                key = database.store_layout.key_for(shard, block_hash)
                operations.append(
                    _PageOperation(
                        group=group_index,
                        block_id=block_id,
                        key=key,
                        start=start,
                        end=end,
                        block_hash=block_hash,
                    )
                )
        return operations

    def _take_lease(self, page_count: int) -> tuple[CacheStagingLease, int]:
        _, staging = self._require_staging()
        count = min(page_count, self._max_batch_pages, staging.capacity_pages)
        acquired = 0
        try:
            while acquired < count:
                if self._lease_slots.acquire(timeout=_LEASE_WAIT_SECONDS):
                    acquired += 1
                    continue
                self._pointer_io.ensure_safe()
                if self._closed or self._closing:
                    raise RuntimeError("Furiosa Mooncake worker is closing")
            if self._closed or self._closing:
                raise RuntimeError("Furiosa Mooncake worker is closing")
            lease = staging.try_acquire(count)
            if lease is None:
                raise RuntimeError("bounded Mooncake staging could not lease available pages")
            return lease, count
        except BaseException:
            if acquired:
                self._lease_slots.release(acquired)
            raise

    def _release_lease(
        self, staging: BoundedCacheStaging, lease: CacheStagingLease, count: int
    ) -> None:
        staging.release(lease)
        self._lease_slots.release(count)

    def _page_routes(
        self,
        operations: Sequence[_PageOperation],
        slots: Sequence[int],
    ) -> dict[int, LayerPageRoutes]:
        layer_pages: dict[int, list[int]] = {}
        layer_slots: dict[int, list[int]] = {}
        for operation, slot in zip(operations, slots, strict=True):
            names = self._kv_cache_groups[operation.group].layer_names
            for name in names:
                layer = _layer_id(name)
                layer_pages.setdefault(layer, []).append(operation.block_id)
                layer_slots.setdefault(layer, []).append(slot)
        return {
            layer: LayerPageRoutes(layer_pages[layer], layer_slots[layer]) for layer in layer_pages
        }

    def _buffer_values(
        self,
        staging: BoundedCacheStaging,
        operations: Sequence[_PageOperation],
        slots: Sequence[int],
    ) -> tuple[list[list[int]], list[list[int]]]:
        base = self._staging_address
        if base is None:
            raise RuntimeError("Mooncake staging slab has no registered address")
        addresses: list[list[int]] = []
        sizes: list[list[int]] = []
        for operation, slot in zip(operations, slots, strict=True):
            layer_ids = tuple(
                _layer_id(name) for name in self._kv_cache_groups[operation.group].layer_names
            )
            addresses.append(
                [
                    base + slot * staging.host_block_bytes + staging.layer_offsets[layer]
                    for layer in layer_ids
                ]
            )
            sizes.append([staging.host_page_bytes[layer] for layer in layer_ids])
        return addresses, sizes

    def _put_pages(self, operations: Sequence[_PageOperation]) -> bool:
        cache, staging = self._require_staging()
        all_succeeded = True
        for offset in range(0, len(operations), self._max_batch_pages):
            batch = operations[offset : offset + self._max_batch_pages]
            lease, permit_count = self._take_lease(len(batch))
            slots = lease.slots
            transport_started = False
            retain_lease = False
            try:
                staging.copy_pages(
                    cache, self._page_routes(batch, slots), lease, to_host=True
                ).wait()
                addresses, sizes = self._buffer_values(staging, batch, slots)
                staging.begin_transport(lease)
                transport_started = True
                put_started = perf_counter()
                batch_bytes = sum(sum(value_sizes) for value_sizes in sizes)
                if self.enable_group_semantics and self._supports_group_ids:
                    self.store_replicate_config.group_ids = [
                        _make_mooncake_group_id(
                            self.token_dbs[operation.group].metadata,
                            operation.key.rsplit("@", 1)[-1],
                        )
                        for operation in batch
                    ]
                try:
                    results = self._pointer_io.run_pointer_io(
                        self,
                        (cast(HostRegionLease, self._staging_region),),
                        partial(
                            self.store.batch_put_from_multi_buffers,
                            [operation.key for operation in batch],
                            addresses,
                            sizes,
                            self.store_replicate_config,
                        ),
                        partial(require_confirmed_put, count=len(batch)),
                        "PUT",
                    )
                except BaseException:
                    self._record_operation(
                        "save_put",
                        put_started,
                        len(batch),
                        num_bytes=batch_bytes,
                        status="error",
                        num_failed_keys=len(batch),
                    )
                    raise
                succeeded = [result == 0 for result in results]
                failed_keys = len(batch) - sum(succeeded)
                self._record_operation(
                    "save_put",
                    put_started,
                    len(batch),
                    num_bytes=batch_bytes,
                    status="partial_failure" if failed_keys else "ok",
                    num_failed_keys=failed_keys,
                )
                all_succeeded = all_succeeded and all(succeeded)
            except UnconfirmedMooncakeIO:
                retain_lease = True
                self._unsafe_staging_leases.append(lease)
                raise
            finally:
                if transport_started and not retain_lease:
                    staging.complete_transport(lease)
                if not retain_lease:
                    self._release_lease(staging, lease, permit_count)
        return all_succeeded

    def _get_pages(self, req_id: str, operations: Sequence[_PageOperation]) -> bool:
        cache, staging = self._require_staging()
        all_succeeded = True
        for offset in range(0, len(operations), self._max_batch_pages):
            batch = operations[offset : offset + self._max_batch_pages]
            lease, permit_count = self._take_lease(len(batch))
            slots = lease.slots
            transport_started = False
            retain_lease = False
            try:
                addresses, sizes = self._buffer_values(staging, batch, slots)
                keys = [operation.key for operation in batch]
                load_batches, oversized_key = _load_batches(
                    keys,
                    addresses,
                    sizes,
                    self.disk_offload_buffer_budget_bytes,
                )
                if oversized_key is not None:
                    key_index = keys.index(oversized_key)
                    required = _estimate_disk_offload_staging_bytes(sizes[key_index])
                    assert self.disk_offload_buffer_budget_bytes is not None
                    logger.warning(
                        "Skipping Mooncake load for request %s because key %s requires %d "
                        "staging bytes, exceeding budget %d",
                        req_id,
                        oversized_key,
                        required,
                        self.disk_offload_buffer_budget_bytes,
                    )
                    self._report_load_errors(operation.block_id for operation in batch)
                    all_succeeded = False
                    continue

                staging.begin_transport(lease)
                transport_started = True
                successful_operations: list[_PageOperation] = []
                successful_slots: list[int] = []
                batch_succeeded = True
                for batch_keys, batch_addresses, batch_sizes, indices in load_batches:
                    batch_operations = [batch[index] for index in indices]
                    batch_slots = [slots[index] for index in indices]
                    capacities = [sum(value_sizes) for value_sizes in batch_sizes]
                    tiers_by_key = (
                        _get_replica_tiers_by_key(self.store, batch_keys)
                        if envs.VLLM_MOONCAKE_STORE_TIER_LOG
                        else None
                    )
                    get_started = perf_counter()
                    batch_bytes = sum(capacities)
                    try:
                        results = self._pointer_io.run_pointer_io(
                            self,
                            (cast(HostRegionLease, self._staging_region),),
                            partial(
                                self.store.batch_get_into_multi_buffers,
                                batch_keys,
                                batch_addresses,
                                batch_sizes,
                            ),
                            partial(require_confirmed_get, capacities=capacities),
                            "GET",
                        )
                    except BaseException:
                        self._record_operation(
                            "load_get",
                            get_started,
                            len(batch_keys),
                            num_bytes=batch_bytes,
                            status="error",
                            num_failed_keys=len(batch_keys),
                        )
                        raise
                    if tiers_by_key is not None:
                        _log_mooncake_load_tier_summary(req_id, batch_keys, results, tiers_by_key)
                    succeeded = [
                        result == capacity
                        for result, capacity in zip(results, capacities, strict=True)
                    ]
                    failed_keys = len(batch_keys) - sum(succeeded)
                    self._record_operation(
                        "load_get",
                        get_started,
                        len(batch_keys),
                        num_bytes=batch_bytes,
                        status="partial_failure" if failed_keys else "ok",
                        num_failed_keys=failed_keys,
                    )
                    self._report_load_errors(
                        operation.block_id
                        for operation, success in zip(batch_operations, succeeded, strict=True)
                        if not success
                    )
                    successful_operations.extend(
                        operation
                        for operation, success in zip(batch_operations, succeeded, strict=True)
                        if success
                    )
                    successful_slots.extend(
                        slot
                        for slot, success in zip(batch_slots, succeeded, strict=True)
                        if success
                    )
                    batch_succeeded = batch_succeeded and all(succeeded)

                staging.complete_transport(lease)
                transport_started = False
                if successful_operations:
                    try:
                        staging.copy_pages(
                            cache,
                            self._page_routes(successful_operations, successful_slots),
                            lease,
                            to_host=False,
                        ).wait()
                    except BaseException:
                        self._report_load_errors(
                            operation.block_id for operation in successful_operations
                        )
                        raise
                all_succeeded = all_succeeded and batch_succeeded
            except UnconfirmedMooncakeIO:
                retain_lease = True
                self._unsafe_staging_leases.append(lease)
                raise
            finally:
                if transport_started and not retain_lease:
                    staging.complete_transport(lease)
                if not retain_lease:
                    self._release_lease(staging, lease, permit_count)
        return all_succeeded

    def _store_request(self, request: ReqMeta) -> bool:
        request_lock = self._save_locks[hash(request.req_id) % len(self._save_locks)]
        with request_lock:
            current_hashes = tuple(request.block_hashes)
            with self._state_lock:
                save_start = self._saved_offsets.get(request.req_id, 0)
                saved_hashes = self._saved_hashes.get(request.req_id)
                if saved_hashes is not None and current_hashes[: len(saved_hashes)] != saved_hashes:
                    save_start = 0
            token_len = request.token_len_chunk
            if token_len <= save_start:
                return True
            operations = self._operations(request, load=False, save_start=save_start)
            if not operations:
                with self._state_lock:
                    self._saved_offsets[request.req_id] = token_len
                    self._saved_hashes[request.req_id] = current_hashes
                return True

            exists_started = perf_counter()
            try:
                exists = require_confirmed_exists(
                    self.store.batch_is_exist([operation.key for operation in operations]),
                    len(operations),
                )
            except BaseException:
                self._record_operation(
                    "save_exists",
                    exists_started,
                    len(operations),
                    status="error",
                    num_failed_keys=len(operations),
                )
                raise
            self._record_operation("save_exists", exists_started, len(operations))
            missing = [
                operation for operation, found in zip(operations, exists, strict=True) if not found
            ]
            succeeded = not missing or self._put_pages(missing)
            if succeeded:
                self._record_store_events(request, missing)
                with self._state_lock:
                    self._saved_offsets[request.req_id] = token_len
                    self._saved_hashes[request.req_id] = current_hashes
            return succeeded

    def _load_request(self, request: ReqMeta) -> bool:
        operations = self._operations(request, load=True)
        if not operations:
            return True
        oversized = next(
            (
                operation
                for operation in operations
                if operation.group in self._disk_oversized_groups
            ),
            None,
        )
        if oversized is not None:
            assert self.disk_offload_buffer_budget_bytes is not None
            group = self._kv_cache_groups[oversized.group]
            _, staging = self._require_staging()
            key_size = sum(staging.host_page_bytes[_layer_id(name)] for name in group.layer_names)
            required = _estimate_disk_offload_staging_bytes([key_size])
            logger.warning(
                "Skipping Mooncake load for request %s because key %s requires %d "
                "staging bytes, exceeding budget %d",
                request.req_id,
                oversized.key,
                required,
                self.disk_offload_buffer_budget_bytes,
            )
            self._report_load_errors(operation.block_id for operation in operations)
            return False
        return self._get_pages(request.req_id, operations)

    def _record_operation(
        self,
        operation: str,
        start_time: float,
        num_keys: int,
        *,
        num_bytes: int = 0,
        status: str = "ok",
        num_failed_keys: int = 0,
    ) -> None:
        self._record_kv_connector_operation(
            operation=operation,
            duration_seconds=perf_counter() - start_time,
            num_keys=num_keys,
            num_bytes=num_bytes,
            status=status,
            num_failed_keys=num_failed_keys,
        )

    def _report_load_errors(self, block_ids: Iterable[int]) -> None:
        with self._load_errors_lock:
            self._load_error_block_ids.update(block_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        with self._load_errors_lock:
            block_ids, self._load_error_block_ids = self._load_error_block_ids, set()
        return block_ids

    def _record_store_events(self, request: ReqMeta, operations: Sequence[_PageOperation]) -> None:
        if not self.enable_kv_events or not operations:
            return
        events: list[BlockStored] = []
        for operation in operations:
            database = self.token_dbs[operation.group]
            token_ids = request.token_ids
            token_start = request.token_ids_start
            if token_ids is not None and token_start <= operation.start:
                left = operation.start - token_start
                right = operation.end - token_start
                token_ids = token_ids[left:right] if right <= len(token_ids) else []
            else:
                token_ids = []
            parent_hash = (
                maybe_convert_block_hash(
                    request.block_hashes[operation.start // database.hash_block_size - 1]
                )
                if operation.start > 0
                else None
            )
            events.append(
                BlockStored(
                    block_hashes=[maybe_convert_block_hash(operation.block_hash)],
                    parent_block_hash=parent_hash,
                    token_ids=token_ids,
                    block_size=database.block_size,
                    lora_id=None,
                    medium="cpu",
                    lora_name=None,
                    group_idx=operation.group,
                )
            )
        with self._kv_events_lock:
            self._kv_events.extend(events)

    @override
    def start_load_kv(self, metadata: MooncakeStoreConnectorMetadata) -> None:
        if self._capacity_only:
            return
        self._require_staging()
        with self._state_lock:
            for request in metadata.requests:
                load_spec = request.load_spec
                if load_spec is None or not load_spec.can_load:
                    continue
                if request.req_id in self._load_futures:
                    raise RuntimeError(f"Mooncake load already active for {request.req_id}")
                load_spec.token_len = load_spec.kvpool_cached_tokens
                self._load_futures[request.req_id] = self._load_executor.submit(
                    self._load_request, request
                )

    @override
    def wait_for_save(self, metadata: MooncakeStoreConnectorMetadata) -> None:
        if self._capacity_only or not self.can_put:
            return
        self._require_staging()
        with self._state_lock:
            for request in metadata.requests:
                if not request.can_save:
                    continue
                job_id = request.store_job_id
                if job_id is None:
                    raise ValueError("Mooncake save request has no store_job_id")
                if job_id in self._store_futures:
                    raise RuntimeError(f"Mooncake store job {job_id} was submitted twice")
                self._retired_requests.discard(request.req_id)
                future = self._store_executor.submit(self._store_request, request)
                self._store_futures[job_id] = (request.req_id, future)
                self._pending_stores[request.req_id] = (
                    self._pending_stores.get(request.req_id, 0) + 1
                )

    @override
    def get_finished(
        self,
        finished_req_ids: set[str],
        meta: MooncakeStoreConnectorMetadata,
    ) -> tuple[set[str], set[str]]:
        if self._capacity_only:
            return set(), set()
        completed: list[tuple[str, Future[bool]]] = []
        with self._state_lock:
            self._retired_requests.update(finished_req_ids)
            self._retired_requests.update(meta.preempted_req_ids)
            completed = [
                (req_id, future) for req_id, future in self._load_futures.items() if future.done()
            ]
            for req_id, _ in completed:
                del self._load_futures[req_id]

        done_recving: set[str] = set()
        for req_id, future in completed:
            try:
                succeeded = future.result()
            except UnconfirmedMooncakeIO:
                raise
            except Exception as error:
                logger.warning("Mooncake GET failed for request %s: %s", req_id, error)
                succeeded = False
            if succeeded:
                done_recving.add(req_id)
            else:
                with self._state_lock:
                    self._load_failures.add(req_id)
        self._cleanup_retired_requests()
        return set(), done_recving

    @override
    def get_transfer_results(
        self,
        finished_req_ids: set[str],
        meta: MooncakeStoreConnectorMetadata,
    ) -> KVConnectorTransferResults:
        done_sending, done_recving = self.get_finished(finished_req_ids, meta)
        with self._state_lock:
            failed_recving = set(self._load_failures)
            self._load_failures.clear()
        return KVConnectorTransferResults(
            finished_sending=done_sending,
            finished_recving=done_recving,
            failed_recving=failed_recving,
        )

    @override
    def build_connector_worker_meta(self) -> MooncakeStoreWorkerMetadata | None:
        with self._state_lock:
            completed = [
                (job_id, req_id, future)
                for job_id, (req_id, future) in self._store_futures.items()
                if future.done()
            ]
            for job_id, _, _ in completed:
                del self._store_futures[job_id]

        completed_saves: dict[int, int] = {}
        for job_id, req_id, future in completed:
            try:
                succeeded = future.result()
            except UnconfirmedMooncakeIO:
                raise
            except Exception as error:
                logger.warning("Mooncake PUT failed for store job %d: %s", job_id, error)
                succeeded = False
            if not succeeded:
                logger.warning("Mooncake PUT left store job %d uncached", job_id)
            completed_saves[job_id] = 1
            with self._state_lock:
                pending = self._pending_stores[req_id] - 1
                if pending:
                    self._pending_stores[req_id] = pending
                else:
                    del self._pending_stores[req_id]
                    if req_id in self._retired_requests:
                        self._saved_offsets.pop(req_id, None)
                        self._saved_hashes.pop(req_id, None)
                        self._retired_requests.discard(req_id)
        if not completed_saves:
            return None
        return MooncakeStoreWorkerMetadata(completed_saves=completed_saves)

    def _cleanup_retired_requests(self) -> None:
        with self._state_lock:
            for req_id in tuple(self._retired_requests):
                if req_id not in self._pending_stores and req_id not in self._load_futures:
                    self._saved_offsets.pop(req_id, None)
                    self._saved_hashes.pop(req_id, None)
                    self._retired_requests.discard(req_id)

    def get_kv_events(self) -> list[BlockStored]:
        with self._kv_events_lock:
            events, self._kv_events = self._kv_events, []
        return events

    @override
    def lookup(self, num_tokens: int, block_hashes: Sequence[Any]) -> Any:
        self._pointer_io.ensure_safe()
        return super().lookup(num_tokens, block_hashes)

    @override
    def get_mem_pool_context(self) -> None:
        return None

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closing = True
            self._load_executor.shutdown(wait=True, cancel_futures=False)
            self._store_executor.shutdown(wait=True, cancel_futures=False)
            self._pointer_io.ensure_safe()
            self._region_registrar.close()
            if self._staging is not None and self._cache is not None:
                self._staging.release_engine_cache(self._cache)
            if self._staging_region is not None:
                self._staging_region.release()
            super().close()
            self._staging_region = None
            self._staging = None
            self._cache = None
            self._closed = True
            self._closing = False


class FuriosaMooncakeStoreConnector(MooncakeStoreConnector):
    """Reuse upstream prefix lookup/scheduling; adapt worker I/O to the engine-owned cache."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None:
        if kv_cache_config is None:
            raise ValueError("Furiosa Mooncake Store requires KVCacheConfig")
        _namespace_cache_prefix(vllm_config, kv_cache_config)
        KVConnectorBase_V1.__init__(
            self,
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        assert vllm_config.kv_transfer_config is not None
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        self._capacity_only = (
            self.kv_role == "kv_consumer"
            and not extra.get("enable_lookup", True)
            and not extra.get("save_decode_cache", False)
        )
        if not self._capacity_only:
            self._validate_kv_cache_config(vllm_config, kv_cache_config)
        self._kv_cache_config = kv_cache_config
        self._kv_cache_events: MooncakeStoreKVEvents | None = None
        self.connector_scheduler: MooncakeStoreScheduler | None = None
        self.connector_worker: FuriosaMooncakeStoreWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = MooncakeStoreScheduler(vllm_config, kv_cache_config)
            # One Furiosa worker owns every TP chip and reports each store job once.
            self.connector_scheduler._num_workers = 1
        else:
            self.connector_worker = FuriosaMooncakeStoreWorker(vllm_config, kv_cache_config)

    def register_engine_cache(self, cache: engine.decoder.Cache, chips: int) -> None:
        if self.connector_worker is None:
            raise RuntimeError("engine cache registration requires the vLLM worker role")
        self.connector_worker.register_engine_cache(cache, chips)

    @override
    def get_kv_connector_kv_cache_events(self) -> MooncakeStoreKVEvents | None:
        worker = self.connector_worker
        if worker is None or not worker.enable_kv_events:
            return None
        events = worker.get_kv_events()
        aggregate = MooncakeStoreKVEvents(
            num_workers=1,
            group_tp_replication_factors=worker.group_tp_replication_factors,
        )
        aggregate.add_events(events)
        return aggregate
