"""vLLM's native NIXL protocol over bounded engine-owned staging slots."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, replace
from math import prod
from typing import TYPE_CHECKING, Any, cast

import torch
from typing_extensions import override
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlBaseConnector as _NixlBaseConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlPullConnector as _NixlPullConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlPushConnector as _NixlPushConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
    NixlPullConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_scheduler import (
    NixlPushConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import (
    NixlPushConnectorWorker,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from furiosa_opt_engine.cache_transfer import (
    BoundedCacheStaging,
    CacheStagingLease,
    CacheStagingSlots,
    LayerPageRoutes,
    bounded_staging_capacity,
    cache_staging_capacity,
)
from furiosa_opt_vllm import residency
from furiosa_opt_vllm.cache import assigned
from furiosa_opt_vllm.decoder import Decoder
from furiosa_opt_vllm.transfer import nixl_staging_budget_bytes

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
    from vllm.v1.request import Request

    import furiosa_opt_engine as engine
    from furiosa_opt_vllm.runner import Runner


@dataclass(frozen=True, slots=True)
class _Layer:
    index: int
    group: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    page_bytes: int
    raw: torch.Tensor
    nixl: torch.Tensor


class EngineCacheViews(dict[str, torch.Tensor]):
    """NIXL views alias the engine's bounded pinned staging pages."""

    def __init__(
        self,
        runner: Runner,
        config: KVCacheConfig,
        budget_bytes: int,
    ) -> None:
        cache = residency.cache(runner)
        model = runner.model
        if cache is None or not isinstance(model, Decoder) or model.model is None:
            raise RuntimeError("NIXL staging requires a loaded engine decoder cache")
        self.runner = runner
        # vLLM sets TP=1 for this adapter. Its NIXL pages are logical pages, as in Layers.copy.
        self.chips = 1
        self.capacity_pages = _staging_capacity(config, budget_bytes)
        engine_capacity = cache_staging_capacity(
            cache,
            chips=1,
            budget_bytes=budget_bytes,
            max_pages=min(cache.blocks, config.num_blocks),
        )
        if engine_capacity != self.capacity_pages:
            raise ValueError("vLLM and engine cache geometries disagree on bounded capacity")
        if cache.blocks != config.num_blocks:
            raise ValueError("vLLM and engine cache block counts differ for NIXL staging")
        self.staging = BoundedCacheStaging(cache, self.chips, self.capacity_pages)
        layout = self.staging.layout
        self._host_region = self.staging.lease_host_region()
        self._host_view = self._host_region.view
        self.layers: dict[str, _Layer] = {}
        self._transform_scratch = bytearray(max(layer.page_bytes for layer in layout.layers))
        self._transform_lock = threading.Lock()
        page_major = torch.frombuffer(self._host_view, dtype=torch.uint8)

        for name, group, spec in _layer_specs(config):
            index = extract_layer_index(name)
            if not 0 <= index < len(cache.layers):
                raise ValueError(f"engine cache has no storage for {name}")
            page_bytes = spec.page_size_bytes
            layer_layout = layout.layers[index]
            if page_bytes != layer_layout.page_bytes:
                raise ValueError(
                    f"NIXL page size for {name} is {page_bytes} bytes; "
                    f"engine page size is {layer_layout.page_bytes} bytes"
                )
            shape = (spec.block_size, 2, spec.num_kv_heads, spec.head_size)
            if prod(shape) * spec.dtype.itemsize != page_bytes:
                raise ValueError(f"NIXL page geometry differs from the engine cache for {name}")
            raw = page_major.as_strided(
                (layout.capacity_pages, page_bytes),
                (layout.slot_bytes, 1),
                storage_offset=layer_layout.byte_offset,
            )
            nixl_view = raw.view(spec.dtype).view(
                layout.capacity_pages,
                spec.num_kv_heads,
                spec.block_size,
                2 * spec.head_size,
            )
            meta_view = torch.empty(
                (
                    layout.capacity_pages,
                    spec.num_kv_heads,
                    spec.block_size,
                    2 * spec.head_size,
                ),
                dtype=spec.dtype,
                device="meta",
            )
            self.layers[name] = _Layer(
                index,
                group,
                shape,
                spec.dtype,
                page_bytes,
                raw,
                nixl_view,
            )
            self[name] = meta_view
        layer_indices = {layer.index for layer in self.layers.values()}
        if layer_indices != set(range(len(cache.layers))):
            raise ValueError("NIXL metadata does not cover every transferable engine layer")


def _layer_specs(config: KVCacheConfig) -> list[tuple[str, int, KVCacheSpec]]:
    if len(config.transfer_groups) != len(config.kv_cache_groups):
        raise ValueError("bounded NIXL staging requires every KV cache group to be transferable")
    layers = []
    names: set[str] = set()
    for group_index, group in enumerate(config.transfer_groups):
        for name in group.layer_names:
            if name in names:
                raise ValueError(f"NIXL cache layer {name} appears in multiple groups")
            names.add(name)
            spec = assigned(group, name)
            if not isinstance(spec, (FullAttentionSpec, SlidingWindowSpec)):
                raise ValueError(f"bounded NIXL staging does not support {type(spec).__name__}")
            layers.append((name, group_index, spec))
    return layers


def _staging_capacity(config: KVCacheConfig, budget_bytes: int) -> int:
    layers = _layer_specs(config)
    if not layers:
        raise ValueError("NIXL staging requires transferable attention layers")
    bytes_per_slot = sum(spec.page_size_bytes for _, _, spec in layers)
    return bounded_staging_capacity(bytes_per_slot, budget_bytes, max_pages=config.num_blocks)


def cache_views(runner: Runner, config: KVCacheConfig) -> EngineCacheViews:
    """Build NIXL registration metadata and bounded host staging for this engine cache."""
    return EngineCacheViews(runner, config, nixl_staging_budget_bytes(runner.vllm_config))


@dataclass(slots=True)
class _StagedRequest:
    lease: CacheStagingLease
    source: bool
    slots_by_group: list[dict[int, int]]
    _page_ids_by_group: list[list[int]]
    _free_slots_by_group: list[deque[int]]

    @classmethod
    def create(cls, lease: CacheStagingLease, groups: int, *, source: bool) -> _StagedRequest:
        return cls(
            lease,
            source,
            [{} for _ in range(groups)],
            [[] for _ in range(groups)],
            [deque(lease.slots) for _ in range(groups)],
        )

    def add(self, block_ids: BlockIds) -> None:
        if len(block_ids) != len(self.slots_by_group):
            raise ValueError("NIXL block groups differ from bounded staging groups")
        for group, ids in enumerate(block_ids):
            previous = self.slots_by_group[group]
            page_ids = list(dict.fromkeys(ids))
            if len(page_ids) > len(self.lease.slots):
                raise ValueError("NIXL request exceeds its bounded staging lease")
            previous_ids = self._page_ids_by_group[group]
            if page_ids == previous_ids:
                continue

            if len(page_ids) > len(previous_ids) and all(
                page == existing for page, existing in zip(page_ids, previous_ids, strict=False)
            ):
                free_slots = self._free_slots_by_group[group]
                new_pages = page_ids[len(previous_ids) :]
                if len(new_pages) > len(free_slots):
                    raise ValueError("NIXL request exceeds its bounded staging lease")
                for page in new_pages:
                    previous[page] = free_slots.popleft()
                previous_ids.extend(new_pages)
                continue

            current = {page: previous[page] for page in page_ids if page in previous}
            used = set(current.values())
            available = deque(slot for slot in self.lease.slots if slot not in used)
            for page in page_ids:
                if page not in current:
                    if not available:
                        raise ValueError("NIXL request exceeds its bounded staging lease")
                    current[page] = available.popleft()
            self.slots_by_group[group] = current
            self._page_ids_by_group[group] = page_ids
            self._free_slots_by_group[group] = available

    def map(self, block_ids: BlockIds) -> tuple[list[int], ...]:
        if len(block_ids) != len(self.slots_by_group):
            raise ValueError("NIXL block groups differ from bounded staging groups")
        try:
            return tuple(
                [self.slots_by_group[group][block_id] for block_id in ids]
                for group, ids in enumerate(block_ids)
            )
        except KeyError as error:
            raise ValueError(f"NIXL block {error.args[0]} has no staging slot") from None


class _BoundedScheduler:
    def __init__(self, config: VllmConfig, engine_id: str, kv_cache_config: KVCacheConfig) -> None:
        super().__init__(config, engine_id, kv_cache_config)
        budget_bytes = nixl_staging_budget_bytes(config)
        self._staging_capacity = _staging_capacity(kv_cache_config, budget_bytes)
        self._staging_slots = CacheStagingSlots(self._staging_capacity)
        self._staged_requests: dict[str, _StagedRequest] = {}
        self._pending_receive_leases: dict[str, CacheStagingLease] = {}
        self._cache_config = kv_cache_config

    @override
    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int, bool]:
        count, asynchronous = super().get_num_new_matched_tokens(request, num_computed_tokens)
        request_id = request.request_id
        pending_lease = self._pending_receive_leases.get(request_id)
        if (
            pending_lease is not None
            and count > 0
            and len(pending_lease.slots) < self._request_pages(count)
        ):
            raise RuntimeError(f"NIXL receive reservation grew for request {request_id}")
        if (
            count <= 0
            or not asynchronous
            or request_id in self._staged_requests
            or request_id in self._pending_receive_leases
        ):
            return count, asynchronous
        lease = self._staging_slots.try_acquire(self._request_pages(count))
        if lease is None:
            return 0, False
        self._pending_receive_leases[request_id] = lease
        return count, asynchronous

    @override
    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ) -> None:
        params = request.kv_transfer_params or {}
        request_id = request.request_id
        needs_push_registration = params.get("do_remote_prefill") and num_external_tokens > 0
        if params.get("do_remote_decode"):
            state = self._staged_requests.get(request_id)
            if state is None:
                state = self._reserve(
                    request_id, self._request_pages(request.num_tokens), source=True
                )
                if state is None:
                    params["do_remote_decode"] = False
            if state is not None:
                state.add(blocks.get_block_ids())
        elif params.get("do_remote_prefill") and num_external_tokens > 0:
            state = self._reserve(
                request_id, self._request_pages(num_external_tokens), source=False
            )
            if state is None:
                raise RuntimeError("NIXL staging reservation was lost after remote admission")
            state.add(tuple(tuple(group) for group in blocks.get_unhashed_block_ids_all_groups()))

        super().update_state_after_alloc(request, blocks, num_external_tokens)

        pending_registrations = getattr(self, "_push_pending_registrations", None)
        if needs_push_registration and pending_registrations is not None:
            registration = pending_registrations[request_id]
            registration["local_block_ids"] = self._mapped(
                request_id, cast("BlockIds", registration["local_block_ids"])
            )

    @override
    def request_finished(
        self, request: Request, block_ids: BlockIds
    ) -> tuple[bool, dict[str, Any] | None]:
        request_id = request.request_id
        state = self._staged_requests.get(request_id)
        if state is not None and state.source:
            state.add(block_ids)
        delay, transfer_params = super().request_finished(request, block_ids)
        if state is None:
            # A match can reserve a lease before allocation. If the request is aborted at that
            # point, native NIXL has no transfer to report that could release the pending slots.
            self._release(request_id)
            return delay, transfer_params
        if not state.source:
            return delay, transfer_params
        if transfer_params is None:
            self._release(request_id)
            return delay, None

        staged = self._mapped(request_id, cast("BlockIds", transfer_params["remote_block_ids"]))
        transfer_params["remote_block_ids"] = staged
        finished_blocks = getattr(self, "_finished_request_blocks", None)
        if finished_blocks is not None and request_id in finished_blocks:
            finished_blocks[request_id] = staged
        newly_finished = getattr(self, "_newly_finished_push_blocks", None)
        if newly_finished is not None and request_id in newly_finished:
            newly_finished[request_id] = staged
        # Save copies have completed before request_finished; NIXL now reads the host lease.
        return False, transfer_params

    @override
    def build_connector_meta(self, scheduler_output: Any) -> NixlConnectorMetadata:
        metadata = cast(NixlConnectorMetadata, super().build_connector_meta(scheduler_output))
        self._attach_maps(metadata)
        return metadata

    @override
    def update_connector_output(self, connector_output: Any) -> None:
        super().update_connector_output(connector_output)
        for request_id in connector_output.finished_sending or ():
            self._release(request_id)
        finished_receives = connector_output.finished_recving or set()
        for request_id in finished_receives | connector_output.failed_recving:
            self._release(request_id)

    def _request_pages(self, tokens: int) -> int:
        return max(
            (
                ceil_div(tokens, assigned(group, group.layer_names[0]).block_size)
                for group in self._cache_config.transfer_groups
            ),
            default=0,
        )

    def _reserve(self, request_id: str, pages: int, *, source: bool) -> _StagedRequest | None:
        current = self._staged_requests.get(request_id)
        if current is not None:
            return current
        lease = self._pending_receive_leases.pop(request_id, None)
        if lease is None:
            lease = self._staging_slots.try_acquire(pages)
        if lease is None:
            return None
        state = _StagedRequest.create(lease, len(self._cache_config.transfer_groups), source=source)
        self._staged_requests[request_id] = state
        return state

    def _mapped(self, request_id: str, block_ids: BlockIds) -> tuple[list[int], ...]:
        try:
            return self._staged_requests[request_id].map(block_ids)
        except KeyError:
            raise RuntimeError(f"NIXL request {request_id} has no staging lease") from None

    def _release(self, request_id: str) -> None:
        state = self._staged_requests.get(request_id)
        if state is not None:
            self._staging_slots.release(state.lease)
            del self._staged_requests[request_id]
        pending = self._pending_receive_leases.get(request_id)
        if pending is not None:
            self._staging_slots.release(pending)
            del self._pending_receive_leases[request_id]

    def _attach_maps(self, metadata: NixlConnectorMetadata) -> None:
        metadata.staging_slots_by_request = {
            request_id: tuple(dict(group) for group in state.slots_by_group)
            for request_id, state in self._staged_requests.items()
        }


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


class BoundedNixlPullScheduler(_BoundedScheduler, NixlPullConnectorScheduler):
    pass


class BoundedNixlPushScheduler(_BoundedScheduler, NixlPushConnectorScheduler):
    pass


class _BoundedNixlWorker:
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if not isinstance(kv_caches, EngineCacheViews):
            raise TypeError("vLLM NIXL requires engine-owned bounded cache views")
        self._engine_views = kv_caches
        if self._physical_blocks_per_logical_kv_block != 1:
            raise NotImplementedError("bounded NIXL staging requires one physical page per block")
        self.num_blocks = kv_caches.capacity_pages
        self._logical_num_blocks = kv_caches.capacity_pages
        self.kv_cache_config = replace(self.kv_cache_config, num_blocks=kv_caches.capacity_pages)
        # Registration geometry remains vLLM's logical NIXL layout; the host tensors are bounded.
        super().register_kv_caches(kv_caches)

    def initialize_host_xfer_buffer(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if not isinstance(kv_caches, EngineCacheViews):
            raise TypeError("vLLM NIXL requires engine-owned bounded cache views")
        self.host_xfer_buffers = {name: layer.nixl for name, layer in kv_caches.layers.items()}

    def save_kv_to_host(self, metadata: NixlConnectorMetadata) -> None:
        slots_by_request = metadata.staging_slots_by_request
        for request_id, request_meta in metadata.reqs_to_save.items():
            slots_by_group = slots_by_request.get(request_id)
            if slots_by_group is None:
                raise RuntimeError(f"NIXL save {request_id} has no bounded staging map")
            logical_pages = cast("BlockIds", request_meta.local_block_ids)
            stage_pages = _map_page_ids(slots_by_group, logical_pages)
            self._copy_pages(logical_pages, stage_pages, to_host=True)

    def sync_recved_kv_to_device(self, request_id: str, meta: Any) -> None:
        slots_by_group = getattr(self, "_staging_slots_by_request", {}).get(request_id)
        if slots_by_group is None:
            raise RuntimeError(f"NIXL receive {request_id} has no bounded staging map")
        logical_pages = cast("BlockIds", meta.local_block_ids)
        stage_pages = _map_page_ids(slots_by_group, logical_pages)
        self._copy_pages(logical_pages, stage_pages, to_host=False)

    def _copy_pages(self, logical_pages: BlockIds, stage_pages: BlockIds, *, to_host: bool) -> None:
        if not any(stage_pages):
            return
        views = self._engine_views
        cache = residency.cache(views.runner)
        if cache is None:
            raise RuntimeError("NIXL cannot copy while the engine cache is released")
        transfers_by_layer: dict[int, LayerPageRoutes] = {}
        for _name, layer in views.layers.items():
            pages = logical_pages[layer.group]
            slots = stage_pages[layer.group]
            if len(pages) != len(slots):
                raise ValueError(f"NIXL page map differs for cache group {layer.group}")
            transfers_by_layer[layer.index] = LayerPageRoutes(pages, slots)

        lease = views.staging.reserve_slots(
            sorted({slot for slots in stage_pages for slot in slots})
        )
        try:
            if to_host:
                views.staging.copy_pages(cache, transfers_by_layer, lease, to_host=True).wait()
                self._reorder_pages(stage_pages, to_nixl=True)
            else:
                self._reorder_pages(stage_pages, to_nixl=False)
                views.staging.copy_pages(cache, transfers_by_layer, lease, to_host=False).wait()
        finally:
            views.staging.release(lease)

    def _release_furiosa_engine_cache(self, cache: engine.decoder.Cache) -> None:
        self._engine_views.staging.release_engine_cache(cache)

    def _restore_furiosa_engine_cache(self, cache: engine.decoder.Cache) -> None:
        self._engine_views.staging.restore_engine_cache(cache)

    def _reorder_pages(self, slots_by_group: BlockIds, *, to_nixl: bool) -> None:
        views = self._engine_views
        scratch_bytes = torch.frombuffer(views._transform_scratch, dtype=torch.uint8)
        with views._transform_lock:
            for layer in views.layers.values():
                slots = slots_by_group[layer.group]
                if not slots:
                    continue
                raw_bytes = layer.raw
                engine_pages = raw_bytes.view(layer.dtype).view(views.capacity_pages, *layer.shape)
                nixl_pages = layer.nixl
                scratch = scratch_bytes[: layer.page_bytes].view(layer.dtype)
                for slot in dict.fromkeys(slots):
                    scratch.copy_(raw_bytes[slot].view(layer.dtype))
                    if to_nixl:
                        _engine_page_to_nixl(scratch.view(*layer.shape), nixl_pages[slot])
                    else:
                        source = scratch.view(*nixl_pages.shape[1:])
                        _nixl_page_to_engine(source, engine_pages[slot])

    def start_load_kv(self, metadata: NixlConnectorMetadata, *args: Any, **kwargs: Any) -> Any:
        self._staging_slots_by_request = dict(getattr(metadata, "staging_slots_by_request", {}))
        return super().start_load_kv(metadata, *args, **kwargs)

    def get_transfer_results(self) -> Any:
        results = super().get_transfer_results()
        mappings = getattr(self, "_staging_slots_by_request", {})
        completed = (
            *(results.finished_recving or ()),
            *(results.finished_sending or ()),
            *results.failed_recving,
        )
        for request_id in completed:
            mappings.pop(request_id, None)
        return results


class BoundedNixlPullWorker(_BoundedNixlWorker, NixlPullConnectorWorker):
    def _read_blocks_for_req(self, request_id: str, meta: Any) -> None:
        slots_by_group = self._staging_slots_by_request.get(request_id)
        if slots_by_group is None:
            return super()._read_blocks_for_req(request_id, meta)
        original_logical = meta.local_block_ids
        original_physical = meta.local_physical_block_ids
        meta.local_block_ids = _map_page_ids(slots_by_group, original_logical)
        meta.local_physical_block_ids = meta.local_block_ids
        try:
            super()._read_blocks_for_req(request_id, meta)
        finally:
            meta.local_block_ids = original_logical
            meta.local_physical_block_ids = original_physical


class BoundedNixlPushWorker(_BoundedNixlWorker, NixlPushConnectorWorker):
    pass


def _map_page_ids(
    slots_by_group: Sequence[dict[int, int]], block_ids: BlockIds
) -> tuple[list[int], ...]:
    if len(slots_by_group) != len(block_ids):
        raise ValueError("NIXL block groups differ from bounded staging groups")
    try:
        return tuple(
            [slots_by_group[group][block_id] for block_id in ids]
            for group, ids in enumerate(block_ids)
        )
    except KeyError as error:
        raise ValueError(f"NIXL block {error.args[0]} has no staging slot") from None


def _engine_page_to_nixl(source: torch.Tensor, target: torch.Tensor) -> None:
    """Convert `[tokens, K/V, heads, width]` into NIXL's `[heads, tokens, 2 * width]`."""
    width = source.shape[-1]
    target[..., :width].copy_(source[:, 0].permute(1, 0, 2))
    target[..., width:].copy_(source[:, 1].permute(1, 0, 2))


def _nixl_page_to_engine(source: torch.Tensor, target: torch.Tensor) -> None:
    """Convert NIXL's `[heads, tokens, 2 * width]` into `[tokens, K/V, heads, width]`."""
    width = target.shape[-1]
    target[:, 0].copy_(source[..., :width].permute(1, 0, 2))
    target[:, 1].copy_(source[..., width:].permute(1, 0, 2))


class NixlPullConnector(_NixlPullConnector):
    def __init__(self, config: VllmConfig, role: KVConnectorRole, cache: KVCacheConfig) -> None:
        _NixlBaseConnector.__init__(self, config, role, cache)
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = BoundedNixlPullScheduler(config, self.engine_id, cache)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = BoundedNixlPullWorker(config, self.engine_id, cache)
        else:
            raise ValueError(f"Unsupported KVConnectorRole: {role}")

    def wait_for_save(self) -> None:
        if self.connector_worker is None or not self.connector_worker.use_host_buffer:
            return
        assert isinstance(self._connector_metadata, NixlConnectorMetadata)
        self.connector_worker.save_kv_to_host(self._connector_metadata)

    @override
    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, NixlConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def _release_furiosa_engine_cache(self, cache: engine.decoder.Cache) -> None:
        if self.connector_worker is not None:
            self.connector_worker._release_furiosa_engine_cache(cache)

    def _restore_furiosa_engine_cache(self, cache: engine.decoder.Cache) -> None:
        if self.connector_worker is not None:
            self.connector_worker._restore_furiosa_engine_cache(cache)


class NixlPushConnector(_NixlPushConnector):
    def __init__(self, config: VllmConfig, role: KVConnectorRole, cache: KVCacheConfig) -> None:
        _NixlBaseConnector.__init__(self, config, role, cache)
        if config.parallel_config.decode_context_parallel_size > 1:
            raise ValueError("NixlPushConnector does not support decode_context_parallel_size > 1")
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = BoundedNixlPushScheduler(config, self.engine_id, cache)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = BoundedNixlPushWorker(config, self.engine_id, cache)
        else:
            raise ValueError(f"Unsupported KVConnectorRole: {role}")

    def wait_for_save(self) -> None:
        if self.connector_worker is None or not self.connector_worker.use_host_buffer:
            return
        assert isinstance(self._connector_metadata, NixlConnectorMetadata)
        self.connector_worker.save_kv_to_host(self._connector_metadata)

    @override
    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, NixlConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def _release_furiosa_engine_cache(self, cache: engine.decoder.Cache) -> None:
        if self.connector_worker is not None:
            self.connector_worker._release_furiosa_engine_cache(cache)

    def _restore_furiosa_engine_cache(self, cache: engine.decoder.Cache) -> None:
        if self.connector_worker is not None:
            self.connector_worker._restore_furiosa_engine_cache(cache)


NixlConnector = NixlPullConnector


__all__ = ["NixlConnector", "NixlPullConnector", "NixlPushConnector"]
