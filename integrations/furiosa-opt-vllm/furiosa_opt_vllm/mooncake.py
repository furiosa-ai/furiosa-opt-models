"""Adapt vLLM's Mooncake connector to the engine's unmapped cache memory.

Mooncake keeps its native prefill/decode control plane and registers a bounded pinned host pool.
The adapter stages only the request's pages through the engine's page-copy API.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnector as _MooncakeConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnectorMetadata,
)

from furiosa_opt_engine.cache_transfer import (
    BoundedCacheStaging,
    CacheStagingLease,
    CacheStagingSlots,
    bounded_staging_capacity,
    cache_staging_capacity,
)
from furiosa_opt_vllm.cache import assigned
from furiosa_opt_vllm.decoder import Decoder

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorOutput,
        KVConnectorRole,
        KVConnectorTransferResults,
    )
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

    import furiosa_opt_engine as engine
    from furiosa_opt_vllm.runner import Runner


@dataclass(frozen=True, slots=True)
class _Layer:
    """One bounded staging layer and its vLLM cache group."""

    index: int
    group: int
    view: torch.Tensor


@dataclass(frozen=True, slots=True)
class _StagedPages:
    lease: CacheStagingLease
    device_pages_by_layer: dict[int, list[int]]
    staging_pages_by_group: list[list[int]]
    device_pages_by_group: list[list[int]]


def _staging_budget_bytes(config: VllmConfig) -> int:
    transfer_config = config.kv_transfer_config
    extra_config = transfer_config.kv_connector_extra_config if transfer_config else {}
    capacity_mb = extra_config.get("staging_buffer_size_mb", 1024)
    if isinstance(capacity_mb, bool) or not isinstance(capacity_mb, int):
        raise ValueError("Mooncake staging_buffer_size_mb must be an integer")
    if capacity_mb <= 0:
        raise ValueError("Mooncake staging_buffer_size_mb must be positive")
    return capacity_mb * 1024 * 1024


def _page_bytes_by_layer(config: KVCacheConfig, chips: int) -> dict[str, int]:
    if chips <= 0:
        raise ValueError("Mooncake staging requires at least one chip")
    if len(config.transfer_groups) != len(config.kv_cache_groups):
        raise ValueError("Mooncake requires every KV cache group to be transferable")

    page_bytes: dict[str, int] = {}
    for group in config.transfer_groups:
        for name in group.layer_names:
            if name in page_bytes:
                raise ValueError(f"vLLM cache layer {name} appears in multiple groups")
            spec = assigned(group, name)
            if spec.page_size_bytes <= 0:
                raise ValueError(f"vLLM cache layer {name} has an invalid page size")
            page_bytes[name] = spec.page_size_bytes * chips
    if not page_bytes:
        raise ValueError("Mooncake requires transferable decoder attention layers")
    return page_bytes


def _staging_capacity_pages(config: KVCacheConfig, chips: int, budget_bytes: int) -> int:
    page_bytes = _page_bytes_by_layer(config, chips)
    return bounded_staging_capacity(
        sum(page_bytes.values()),
        budget_bytes,
        max_pages=config.num_blocks,
    )


class _CacheStaging:
    """A fixed-size pinned staging pool registered with Mooncake."""

    def __init__(self, runner: Runner, config: KVCacheConfig, budget_bytes: int) -> None:
        cache = runner.cache
        model = runner.model
        if cache is None or not isinstance(model, Decoder) or model.model is None:
            raise RuntimeError("Mooncake cache registration requires a loaded engine cache")
        self.runner = runner
        self.chips = model.model.parallel

        group_by_layer = config.transfer_group_index_by_layer
        groups_by_layer: dict[int, int] = {}
        for group in cache.spec.groups:
            for index in group.layers:
                if index in groups_by_layer:
                    raise ValueError(f"engine cache layer {index} appears in multiple groups")
                name = f"layers.{index}"
                try:
                    group_index = group_by_layer[name]
                except KeyError:
                    raise ValueError(f"vLLM has no transfer group for {name}") from None
                groups_by_layer[index] = group_index
        if set(groups_by_layer) != set(range(len(cache.layers))):
            raise ValueError("engine cache groups do not cover every layer exactly once")

        self.group_count = len(config.kv_cache_groups)
        if self.group_count <= 0:
            raise ValueError("Mooncake requires a decoder KV cache")
        page_bytes_by_layer = tuple(pool.bytes * self.chips for pool in cache.spec.pools)
        layer_page_bytes = []
        for group in cache.spec.groups:
            page_bytes = page_bytes_by_layer[group.pool]
            for index in group.layers:
                layer_page_bytes.append((index, page_bytes))
        layer_page_bytes.sort()
        if [index for index, _ in layer_page_bytes] != list(range(len(cache.layers))):
            raise ValueError("engine cache geometry does not cover every Mooncake layer")
        configured_page_bytes = _page_bytes_by_layer(config, self.chips)
        engine_page_bytes = {
            f"layers.{index}": page_bytes for index, page_bytes in layer_page_bytes
        }
        if configured_page_bytes != engine_page_bytes:
            raise ValueError("vLLM and engine cache geometries differ for Mooncake staging")
        if cache.blocks != config.num_blocks:
            raise ValueError("vLLM and engine cache block counts differ for Mooncake staging")
        capacity_pages = cache_staging_capacity(
            cache,
            self.chips,
            budget_bytes,
            max_pages=min(cache.blocks, config.num_blocks),
        )
        if capacity_pages != _staging_capacity_pages(config, self.chips, budget_bytes):
            raise ValueError("vLLM and engine cache geometries disagree on bounded capacity")
        self.pool = BoundedCacheStaging(cache, self.chips, capacity_pages)
        layout = self.pool.layout
        self._host_region = self.pool.lease_host_region()
        self._host_view = self._host_region.view
        self.capacity_pages = capacity_pages
        self.layers: dict[str, _Layer] = {}
        page_major = torch.frombuffer(self._host_view, dtype=torch.uint8)
        for index, page_bytes in layer_page_bytes:
            group = groups_by_layer[index]
            if not 0 <= group < self.group_count:
                raise ValueError(f"vLLM assigned layer {index} to unknown group {group}")
            view = page_major.as_strided(
                (capacity_pages, page_bytes),
                (layout.slot_bytes, 1),
                storage_offset=layout.layers[index].byte_offset,
            )
            self.layers[f"layers.{index}"] = _Layer(index, group, view)

    @property
    def views(self) -> dict[str, torch.Tensor]:
        """The bounded byte slots Mooncake registers, in engine layer order."""
        return {name: layer.view for name, layer in self.layers.items()}

    def try_stage(self, pages_by_group: Sequence[Sequence[int]]) -> _StagedPages | None:
        """Gather pages when a bounded host lease is available."""
        cache = self.runner.cache
        if cache is None:
            raise RuntimeError("Mooncake cannot copy while the engine cache is released")

        staged = self._try_reserve(pages_by_group)
        if staged is None:
            return None
        try:
            self.pool.copy_pages(
                cache, staged.device_pages_by_layer, staged.lease, to_host=True
            ).wait()
        except BaseException:
            self.pool.release(staged.lease)
            raise
        return staged

    def try_receive(self, pages_by_group: Sequence[Sequence[int]]) -> _StagedPages | None:
        """Reserve slots for incoming pages, or return `None` when pool is full."""
        staged = self._try_reserve(pages_by_group)
        if staged is None:
            return None
        try:
            self.pool.begin_transport(staged.lease)
        except BaseException:
            self.pool.release(staged.lease)
            raise
        return staged

    def _try_reserve(self, pages_by_group: Sequence[Sequence[int]]) -> _StagedPages | None:
        if len(pages_by_group) != self.group_count:
            raise ValueError(
                f"Mooncake supplied {len(pages_by_group)} block groups; expected {self.group_count}"
            )
        page_count = max(map(len, pages_by_group), default=0)
        if page_count == 0:
            raise ValueError("Mooncake cannot reserve staging for an empty request")
        lease = self.pool.try_acquire(page_count)
        if lease is None:
            return None
        try:
            normalized = [list(pages) for pages in pages_by_group]
            device_pages_by_layer = {
                layer.index: normalized[layer.group]
                for layer in self.layers.values()
                if normalized[layer.group]
            }
            return _StagedPages(
                lease,
                device_pages_by_layer,
                [list(lease.slots[: len(pages)]) for pages in normalized],
                normalized,
            )
        except BaseException:
            self.pool.release(lease)
            raise

    def begin_send(self, staged: _StagedPages) -> None:
        self.pool.begin_transport(staged.lease)

    def finish_send(self, staged: _StagedPages) -> None:
        """Release a send lease after Mooncake reports transfer completion."""
        self.pool.complete_transport(staged.lease)
        self.pool.release(staged.lease)

    def finish_receive(self, staged: _StagedPages, *, success: bool) -> BaseException | None:
        """Scatter a successful receive and release its lease after transfer completion.

        Return a copy error only after release succeeds, so callers can forget the lease safely.
        A completion or release error escapes with the lease still available for cleanup.
        """
        self.pool.complete_transport(staged.lease)
        copy_error: BaseException | None = None
        try:
            cache = self.runner.cache
            if success and staged.device_pages_by_layer:
                if cache is None:
                    raise RuntimeError("Mooncake cannot copy while the engine cache is released")
                self.pool.copy_pages(
                    cache, staged.device_pages_by_layer, staged.lease, to_host=False
                ).wait()
        except BaseException as error:
            copy_error = error
        finally:
            try:
                self.pool.release(staged.lease)
            except BaseException as release_error:
                if copy_error is None:
                    raise
                raise BaseExceptionGroup(
                    "Mooncake receive copy and lease release failed",
                    [copy_error, release_error],
                ) from None
        return copy_error


class FuriosaMooncakeConnector(_MooncakeConnector):
    """vLLM Mooncake connector whose registered regions are bounded engine-pinned staging."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        self._staging: _CacheStaging | None = None
        self._send_staging: dict[str, _StagedPages] = {}
        self._pending_sends: dict[str, tuple[str, list[list[int]]]] = {}
        self._receive_staging: dict[str, _StagedPages] = {}
        self._staging_failed_receives: set[str] = set()
        self._empty_receives: set[str] = set()
        self._receive_slots: CacheStagingSlots | None = None
        self._receive_leases: dict[str, CacheStagingLease] = {}
        self._active_receives: set[str] = set()
        super().__init__(vllm_config, role, kv_cache_config)
        if self.connector_scheduler is not None:
            if kv_cache_config is None:
                raise ValueError("Mooncake scheduler requires a KV cache config")
            budget_bytes = _staging_budget_bytes(vllm_config)
            capacity_pages = _staging_capacity_pages(
                kv_cache_config,
                vllm_config.parallel_config.tensor_parallel_size,
                budget_bytes,
            )
            self._receive_slots = CacheStagingSlots(capacity_pages)

    def register_engine_cache(self, runner: Runner, config: KVCacheConfig) -> None:
        """Register a fixed-size staging pool, never a cache-sized host image."""
        if self.connector_worker is None:
            raise RuntimeError("Mooncake cache registration requires a worker connector")
        if self._staging is not None:
            raise RuntimeError("Mooncake staging pool was already registered")
        staging = _CacheStaging(runner, config, _staging_budget_bytes(self.vllm_config))
        self._staging = staging
        super().register_kv_caches(staging.views)

        # Each registered base points at one layer inside a page-major HostCache slot. Keep the
        # whole-slot stride for block addressing, but transfer only that layer's bytes.
        worker = self.connector_worker
        if len(worker.block_len_per_layer) != len(staging.layers):
            raise RuntimeError("Mooncake registered a different number of engine cache layers")
        worker.kv_block_len_per_layer = [
            staging.pool.layout.layers[layer.index].page_bytes for layer in staging.layers.values()
        ]

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        # ActiveKVConnector then passes an empty mapping because engine owns the cache.
        if kv_caches:
            raise ValueError("Furiosa Mooncake registers engine staging, not torch caches")

    def _release_furiosa_engine_cache(self, cache: engine.decoder.Cache) -> None:
        if self._staging is not None:
            self._staging.pool.release_engine_cache(cache)

    def _restore_furiosa_engine_cache(self, cache: engine.decoder.Cache) -> None:
        if self._staging is not None:
            self._staging.pool.restore_engine_cache(cache)

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        count, asynchronous = super().get_num_new_matched_tokens(request, num_computed_tokens)
        slots = self._receive_slots
        if count <= 0 or not asynchronous or slots is None:
            return count, asynchronous

        request_id = request.request_id
        page_count = self._request_pages(count)
        if page_count > slots.capacity_pages:
            # Keep Mooncake's native receive failure/recompute path for requests that cannot fit.
            return count, asynchronous

        lease = self._receive_leases.get(request_id)
        if lease is not None:
            if len(lease.slots) < page_count:
                raise RuntimeError(f"Mooncake receive reservation shrank for request {request_id}")
            return count, asynchronous

        lease = slots.try_acquire(page_count)
        if lease is None:
            # vLLM skips this request and asks again after other requests make progress.
            return None, False
        self._receive_leases[request_id] = lease
        return count, asynchronous

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ) -> None:
        request_id = request.request_id
        lease = self._receive_leases.get(request_id)
        page_count = self._request_pages(num_external_tokens) if num_external_tokens > 0 else 0
        if lease is not None and len(lease.slots) < page_count:
            raise RuntimeError(
                f"Mooncake receive reservation is too small for request {request_id}"
            )
        if lease is None and 0 < page_count <= (
            self._receive_slots.capacity_pages if self._receive_slots else 0
        ):
            raise RuntimeError(
                f"Mooncake receive admission has no reservation for request {request_id}"
            )

        super().update_state_after_alloc(request, blocks, num_external_tokens)
        if page_count > 0:
            if lease is not None:
                self._active_receives.add(request_id)
        elif request_id not in self._active_receives:
            self._release_receive(request_id)

    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        result = super().request_finished(request, block_ids)
        if request.request_id not in self._active_receives:
            self._release_receive(request.request_id)
        return result

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        result = super().request_finished_all_groups(request, block_ids)
        if request.request_id not in self._active_receives:
            self._release_receive(request.request_id)
        return result

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        super().update_connector_output(connector_output)
        finished = set(connector_output.finished_recving or ())
        failed = set(connector_output.failed_recving or ())
        for request_id in finished | failed:
            self._release_receive(request_id)

    def _request_pages(self, tokens: int) -> int:
        scheduler = self.connector_scheduler
        if tokens <= 0 or scheduler is None:
            return 0
        pages_by_group = []
        for index, group in enumerate(scheduler.kv_cache_config.transfer_groups):
            block_size = assigned(group, group.layer_names[0]).block_size
            pages = (tokens + block_size - 1) // block_size
            if scheduler._is_hma_required and scheduler.blocks_per_sw[index] > 0:
                pages = min(pages, scheduler.blocks_per_sw[index])
            pages_by_group.append(pages)
        return max(pages_by_group, default=0)

    def _release_receive(self, request_id: str) -> None:
        lease = self._receive_leases.get(request_id)
        if lease is not None:
            slots = self._receive_slots
            if slots is None:
                raise RuntimeError("Mooncake receive lease exists without its scheduler pool")
            slots.release(lease)
            del self._receive_leases[request_id]
        self._active_receives.discard(request_id)

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        staging = self._require_staging()
        metadata = self._get_connector_metadata()
        staged_sends: dict[str, _StagedPages] = {}
        staged_receives: dict[str, _StagedPages] = {}
        original_sends = dict(getattr(metadata, "reqs_to_send", {}))
        original_pending_sends = dict(self._pending_sends)
        send_candidates = dict(original_pending_sends)
        native_sends = {
            request_id: transfer
            for request_id, transfer in original_sends.items()
            if not any(transfer[1])
        }
        for request_id, transfer in original_sends.items():
            if any(transfer[1]):
                if request_id in send_candidates:
                    raise RuntimeError(f"Mooncake has duplicate pending send {request_id}")
                send_candidates[request_id] = transfer
        deferred_sends: dict[str, tuple[str, list[list[int]]]] = {}
        original_receives: dict[str, list[list[int]]] = {}
        empty_receives: set[str] = set()
        failed_receives: set[str] = set()
        transport_started: set[int] = set()
        native_start_called = False
        native_start_submission_failed = False
        try:
            for request_id, (transfer_id, groups) in send_candidates.items():
                if request_id in self._send_staging:
                    raise RuntimeError(f"Mooncake already stages send request {request_id}")
                page_count = max(map(len, groups), default=0)
                if page_count > staging.capacity_pages:
                    raise RuntimeError(
                        f"Mooncake send {request_id} needs {page_count} staging pages; "
                        f"configured pool holds {staging.capacity_pages}"
                    )
                staged = staging.try_stage(groups)
                if staged is None:
                    deferred_sends[request_id] = (transfer_id, groups)
                    logger.info(
                        "Deferring Mooncake send %s until staging capacity is available", request_id
                    )
                    continue
                staged_sends[request_id] = staged
                staging.begin_send(staged)
                transport_started.add(staged.lease.identifier)
                native_sends[request_id] = (
                    transfer_id,
                    staged.staging_pages_by_group,
                )

            for requests in getattr(metadata, "reqs_to_recv", {}).values():
                for request_id, request in requests.items():
                    blocks = [list(group) for group in request.local_block_ids]
                    if not any(blocks):
                        empty_receives.add(request_id)
                        continue
                    previous = self._receive_staging.get(request_id)
                    if previous is not None:
                        if previous.device_pages_by_group != blocks:
                            raise RuntimeError(
                                f"Mooncake has overlapping receives for request {request_id}"
                            )
                        request.local_block_ids = previous.staging_pages_by_group
                        continue
                    staged = staging.try_receive(blocks)
                    if staged is None:
                        failed_receives.add(request_id)
                        continue
                    staged_receives[request_id] = staged
                    original_receives[request_id] = blocks
                    transport_started.add(staged.lease.identifier)
                    request.local_block_ids = staged.staging_pages_by_group

            self._send_staging.update(staged_sends)
            self._receive_staging.update(staged_receives)
            self._empty_receives.update(empty_receives)
            metadata.reqs_to_send.clear()
            metadata.reqs_to_send.update(native_sends)
            if failed_receives:
                logger.warning(
                    "Mooncake staging pool is full; reporting failed receives to vLLM: %s",
                    sorted(failed_receives),
                )
            original_reqs_to_recv = metadata.reqs_to_recv
            if failed_receives:
                metadata.reqs_to_recv = {
                    engine_id: {
                        request_id: request
                        for request_id, request in requests.items()
                        if request_id not in failed_receives
                    }
                    for engine_id, requests in original_reqs_to_recv.items()
                }
            # Native request_finished() retains source blocks until worker reports completion.
            self._pending_sends.update(deferred_sends)
            native_start_called = True
            try:
                try:
                    super().start_load_kv(forward_context, **kwargs)
                except Exception:
                    # In pinned vLLM, start_load_kv only submits receive and send coroutines.
                    # A synchronous exception means the send submission was not accepted.
                    native_start_submission_failed = True
                    raise
            finally:
                if failed_receives:
                    metadata.reqs_to_recv = original_reqs_to_recv
            for request_id in staged_sends:
                self._pending_sends.pop(request_id, None)
            self._staging_failed_receives.update(failed_receives)
        except BaseException as start_error:
            cleanup_errors: list[BaseException] = []
            if native_start_called:
                if native_start_submission_failed:
                    # vLLM 0.31.0 submits receives before sends. Keep receives that may have
                    # started, but return unsubmitted send leases and restore their source pages.
                    for request_id, staged in staged_sends.items():
                        try:
                            staging.finish_send(staged)
                        except BaseException as error:
                            cleanup_errors.append(error)
                        else:
                            self._send_staging.pop(request_id, None)
                    metadata.reqs_to_send.clear()
                    metadata.reqs_to_send.update(original_sends)
                    self._pending_sends = original_pending_sends
            else:
                self._send_staging.update(staged_sends)
                self._receive_staging.update(staged_receives)
                for request_id, staged in staged_sends.items():
                    try:
                        if staged.lease.identifier in transport_started:
                            staging.finish_send(staged)
                        else:
                            staging.pool.release(staged.lease)
                    except BaseException as error:
                        cleanup_errors.append(error)
                    else:
                        self._send_staging.pop(request_id, None)
                for request_id, staged in staged_receives.items():
                    try:
                        copy_error = staging.finish_receive(staged, success=False)
                        if copy_error is not None:
                            raise copy_error
                    except BaseException as error:
                        cleanup_errors.append(error)
                    else:
                        self._receive_staging.pop(request_id, None)
                self._empty_receives.difference_update(empty_receives)
                self._pending_sends = original_pending_sends
                metadata.reqs_to_send.clear()
                metadata.reqs_to_send.update(original_sends)
                for request_id, blocks in original_receives.items():
                    for requests in metadata.reqs_to_recv.values():
                        if request_id in requests:
                            requests[request_id].local_block_ids = blocks
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "Mooncake start and rollback failed",
                    [start_error, *cleanup_errors],
                ) from None
            raise

    def get_transfer_results(self, finished_req_ids: set[str]) -> KVConnectorTransferResults:
        result = super().get_transfer_results(finished_req_ids)
        staging_failed_receives = set(self._staging_failed_receives)
        if staging_failed_receives:
            # Match Mooncake's native async failure path. vLLM can recompute the
            # prefix locally when its KV load failure policy is `recompute`.
            result.failed_recving.update(staging_failed_receives)
            result.finished_recving.update(staging_failed_receives)
        finished = result.finished_recving
        failed = result.failed_recving
        staging = self._require_staging()
        for request_id in finished | failed:
            staged = self._receive_staging.get(request_id)
            if staged is None:
                if request_id in self._empty_receives:
                    self._empty_receives.discard(request_id)
                    continue
                if request_id in finished and request_id not in failed:
                    logger.error(
                        "Mooncake finished receiving %s without staging metadata", request_id
                    )
                    failed.add(request_id)
                continue
            try:
                copy_error = staging.finish_receive(
                    staged, success=request_id in finished and request_id not in failed
                )
            except Exception:
                failed.add(request_id)
                logger.exception("Could not scatter Mooncake KV into vLLM request %s", request_id)
            else:
                del self._receive_staging[request_id]
                if copy_error is not None:
                    if isinstance(copy_error, Exception):
                        failed.add(request_id)
                        logger.error(
                            "Could not scatter Mooncake KV into vLLM request %s",
                            request_id,
                            exc_info=(type(copy_error), copy_error, copy_error.__traceback__),
                        )
                    else:
                        raise copy_error
        for request_id in result.finished_sending:
            staged = self._send_staging.get(request_id)
            if staged is not None:
                staging.finish_send(staged)
                del self._send_staging[request_id]
        self._flush_pending_sends(staging)
        self._staging_failed_receives.difference_update(staging_failed_receives)
        return result

    def _flush_pending_sends(self, staging: _CacheStaging) -> None:
        if not self._pending_sends:
            return
        worker = self.connector_worker
        if worker is None:
            raise RuntimeError("Mooncake pending sends require a worker connector")

        remaining: dict[str, tuple[str, list[list[int]]]] = {}
        staged_sends: dict[str, _StagedPages] = {}
        transport_started: set[int] = set()
        metadata = MooncakeConnectorMetadata()
        try:
            for request_id, (transfer_id, groups) in self._pending_sends.items():
                page_count = max(map(len, groups), default=0)
                if page_count > staging.capacity_pages:
                    raise RuntimeError(
                        f"Mooncake send {request_id} needs {page_count} staging pages; "
                        f"configured pool holds {staging.capacity_pages}"
                    )
                staged = staging.try_stage(groups)
                if staged is None:
                    remaining[request_id] = (transfer_id, groups)
                    continue
                staged_sends[request_id] = staged
                staging.begin_send(staged)
                transport_started.add(staged.lease.identifier)
                metadata.reqs_to_send[request_id] = (
                    transfer_id,
                    staged.staging_pages_by_group,
                )
        except BaseException as submission_error:
            # Keep every acquired lease reachable until its release succeeds, including a lease
            # whose begin_transport failed before the worker could observe it.
            self._send_staging.update(staged_sends)
            cleanup_errors: list[BaseException] = []
            for request_id, staged in staged_sends.items():
                try:
                    if staged.lease.identifier in transport_started:
                        staging.finish_send(staged)
                    else:
                        staging.pool.release(staged.lease)
                except BaseException as error:
                    cleanup_errors.append(error)
                else:
                    self._send_staging.pop(request_id, None)
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "Mooncake send staging and cleanup failed",
                    [submission_error, *cleanup_errors],
                ) from None
            raise

        if not staged_sends:
            return
        self._send_staging.update(staged_sends)
        try:
            worker.start_load_kv(metadata)
        except Exception as submission_error:
            # The pinned vLLM worker only submits this send batch to its event loop. If that
            # synchronous submission fails, no request in the batch can complete; keep the
            # original queue and return every lease for a later retry.
            cleanup_errors: list[BaseException] = []
            for request_id, staged in staged_sends.items():
                try:
                    staging.finish_send(staged)
                except BaseException as error:
                    cleanup_errors.append(error)
                else:
                    self._send_staging.pop(request_id, None)
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "Mooncake send submission and cleanup failed",
                    [submission_error, *cleanup_errors],
                ) from None
            raise
        except BaseException:
            # An interrupt can arrive after vLLM has accepted the coroutine. Keep leases tracked
            # until upstream shutdown has drained its worker.
            raise
        self._pending_sends = remaining

    def shutdown(self) -> None:
        upstream_error: BaseException | None = None
        try:
            super().shutdown()
        except BaseException as error:
            upstream_error = error

        cleanup_errors: list[BaseException] = []
        for request_id in tuple(self._receive_leases):
            try:
                self._release_receive(request_id)
            except BaseException as error:
                cleanup_errors.append(error)
                logger.exception("Could not release Mooncake scheduler lease during shutdown")
        self._pending_sends.clear()

        staging = self._staging
        if staging is not None:
            for request_id, staged in tuple(self._send_staging.items()):
                try:
                    staging.finish_send(staged)
                except BaseException as error:
                    cleanup_errors.append(error)
                    logger.exception("Could not release Mooncake send lease during shutdown")
                else:
                    self._send_staging.pop(request_id, None)
            for request_id, staged in tuple(self._receive_staging.items()):
                try:
                    copy_error = staging.finish_receive(staged, success=False)
                    if copy_error is not None:
                        raise copy_error
                except BaseException as error:
                    cleanup_errors.append(error)
                    logger.exception("Could not release Mooncake receive lease during shutdown")
                else:
                    self._receive_staging.pop(request_id, None)

        if cleanup_errors:
            errors: list[BaseException] = [*cleanup_errors]
            if upstream_error is not None:
                errors.insert(0, upstream_error)
            raise BaseExceptionGroup("Mooncake connector shutdown failed", errors)
        self._staging = None
        self._empty_receives.clear()
        self._staging_failed_receives.clear()
        if upstream_error is not None:
            raise upstream_error

    def _require_staging(self) -> _CacheStaging:
        if self._staging is None:
            raise RuntimeError("Furiosa Mooncake staging pool was not registered")
        return self._staging
