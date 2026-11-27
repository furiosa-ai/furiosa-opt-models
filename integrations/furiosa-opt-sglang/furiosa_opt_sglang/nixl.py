"""Use SGLang's NIXL room and staging protocol with engine-owned KV pages."""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.base.conn import DisaggregationMode
from sglang.srt.disaggregation.nixl.conn import NixlKVManager, NixlKVReceiver
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_schedule

from furiosa_opt_engine.cache_transfer import BoundedCacheStaging, CacheStagingLease
from furiosa_opt_sglang.cache import Cache
from furiosa_opt_sglang.staging import (
    StagingBuffer,
    attach_manager_staging,
    init_decode_staging_allocator,
    init_prefill_staging_buffers,
    manager_staging_copier,
)

FORCE_STAGING_TP_SIZE = -1


@dataclass(slots=True)
class _StagingHandleLease:
    buffer: StagingBuffer
    pool: BoundedCacheStaging
    lease: CacheStagingLease
    transport_completed: bool = False


class FuriosaNixlKVManager(NixlKVManager):
    """SGLang 0.5.22 NIXL manager with bounded staging for engine-owned KV pages."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.engine_cache: Cache | None = None
        self._staging_handle_leases: dict[int, _StagingHandleLease] = {}
        self._staging_handle_lock = threading.Lock()
        super().__init__(*args, **kwargs)

    def _add_remote_peer(self, decode_kv_args: Any) -> None:
        """Select SGLang's native staging branch for engine-owned KV pages."""
        if (
            self.disaggregation_mode == DisaggregationMode.PREFILL
            and decode_kv_args.agent_name not in self.decode_kv_args_table
        ):
            if decode_kv_args.decode_tp_size != self.attn_tp_size:
                raise NotImplementedError(
                    "engine NIXL staging requires matching prefill/decode attention TP"
                )
            # NIXL's worker stages only when TP sizes differ. The engine needs
            # that same native path at equal TP, so retain the true size check above.
            decode_kv_args = replace(decode_kv_args, decode_tp_size=FORCE_STAGING_TP_SIZE)
        super()._add_remote_peer(decode_kv_args)

    def register_buffer_to_engine(self) -> None:
        """Register SGLang aux/state buffers, not protocol-only KV placeholders."""
        args = self.kv_args
        pointers, lengths, kinds = (
            args.kv_data_ptrs,
            args.kv_data_lens,
            args.kv_data_mem_kinds,
        )
        args.kv_data_ptrs, args.kv_data_lens, args.kv_data_mem_kinds = [], [], []
        try:
            super().register_buffer_to_engine()
        finally:
            args.kv_data_ptrs, args.kv_data_lens, args.kv_data_mem_kinds = (
                pointers,
                lengths,
                kinds,
            )

    def _register_staging_memory(self, ptr: int, size: int) -> None:
        descs = self.agent.register_memory([(ptr, size, 0, "")], "DRAM")
        if not descs:
            raise RuntimeError(
                f"NIXL could not register engine-pinned staging memory (ptr=0x{ptr:x}, size={size})"
            )

    def ensure_staging_quiescent(self) -> None:
        """Refuse cache release while a NIXL handle can still access a staging slot."""
        with self._staging_handle_lock:
            pending = len(self._staging_handle_leases)
        if pending:
            raise RuntimeError(
                f"cannot release the KV cache with {pending} NIXL staging transfers pending"
            )

    def _await_handles(self, handles: list[Any], *, failure_seen: bool) -> tuple[bool, bool]:
        settled, failed = super()._await_handles(handles, failure_seen=failure_seen)
        if not settled:
            return settled, failed
        cleanup_errors = []
        with self._staging_handle_lock:
            for handle in handles:
                handle_id = id(handle)
                staged = self._staging_handle_leases.get(handle_id)
                if staged is None:
                    continue
                try:
                    if not staged.transport_completed:
                        staged.pool.complete_transport(staged.lease)
                        staged.transport_completed = True
                    staged.pool.release(staged.lease)
                except BaseException as error:
                    cleanup_errors.append(error)
                else:
                    del self._staging_handle_leases[handle_id]
        if cleanup_errors:
            raise BaseExceptionGroup(
                "could not release completed NIXL staging slots", cleanup_errors
            )
        return settled, failed

    def _init_staging_buffers(self, count: int) -> None:
        self._staging_ctx.buffers = init_prefill_staging_buffers(
            self._register_staging_memory,
            self.kv_args,
            count,
            get_schedule().chunked_prefill_size,
            "NIXL",
        )
        if self.engine_cache is not None and self.engine_cache.storage is not None:
            attach_manager_staging(self, self.engine_cache.storage, self.engine_cache.pages.chips)

    def _init_staging_allocator(self) -> None:
        size = envs.SGLANG_DISAGG_STAGING_POOL_SIZE_MB.get() * 1024 * 1024
        self._staging_ctx.allocator = init_decode_staging_allocator(
            self._register_staging_memory, size
        )
        if self.engine_cache is not None and self.engine_cache.storage is not None:
            attach_manager_staging(self, self.engine_cache.storage, self.engine_cache.pages.chips)

    def _prepare_payload_xfer(self, peer_info: Any) -> None:
        if self.is_mla_backend or self.is_hybrid_mla_backend:
            raise NotImplementedError("engine NIXL staging requires ordinary attention KV")
        if self.kv_args.state_types:
            raise NotImplementedError("engine NIXL staging transfers attention KV only")
        if peer_info.requires_dcp_relayout:
            raise NotImplementedError("engine NIXL staging does not support DCP relayout")
        if peer_info.decode_tp_size not in (
            self.attn_tp_size,
            FORCE_STAGING_TP_SIZE,
        ):
            raise NotImplementedError(
                "engine NIXL staging requires matching prefill/decode attention TP"
            )
        if peer_info.dst_kv_item_lens != self.kv_args.kv_item_lens:
            raise ValueError("prefill and decode engine KV page geometry differs")
        if len(peer_info.dst_kv_ptrs) != len(self.kv_args.kv_data_ptrs):
            raise ValueError("prefill and decode engine KV layer counts differ")
        if not peer_info.staging_base_ptr or not peer_info.staging_total_size:
            raise ValueError("decode peer did not register its bounded NIXL staging ring")
        peer_info.dst_homogeneous_mem_kind = "DRAM"

    def send_kvcache_slice(self, *_args: Any, **_kwargs: Any) -> Any:
        """Never let native NIXL fall back to direct transfer of placeholder pointers."""
        raise RuntimeError("engine-owned NIXL KV pages require a registered bounded staging route")

    def attach_cache(self, cache: Cache) -> None:
        if cache.storage is None:
            raise RuntimeError("SGLang NIXL needs resident engine KV storage")
        page_bytes = cache.pages.pool.bytes * cache.pages.chips
        if page_bytes % 2:
            raise ValueError("engine KV page size cannot split into logical K and V items")
        expected = page_bytes // 2
        entries = 2 * len(cache.storage.layers)
        if not (
            len(self.kv_args.kv_data_ptrs)
            == len(self.kv_args.kv_data_lens)
            == len(self.kv_args.kv_item_lens)
            == entries
        ):
            raise ValueError("SGLang NIXL metadata count differs from engine KV storage")
        if any(item != expected for item in self.kv_args.kv_item_lens):
            raise ValueError("SGLang NIXL metadata differs from engine KV page geometry")
        expected_len = expected * cache.storage.blocks
        if any(length != expected_len for length in self.kv_args.kv_data_lens):
            raise ValueError("SGLang NIXL capacity differs from engine KV storage")
        if self.kv_args.page_size != cache.page_size:
            raise ValueError("SGLang and engine NIXL page sizes differ")
        self.engine_cache = cache
        attach_manager_staging(self, cache.storage, cache.pages.chips)
        self.kv_args.kv_head_num = cache.pages.heads
        self.kv_args.total_kv_head_num = cache.pages.heads
        layers = len(cache.storage.layers)
        self.kv_buffer_tensors = {
            "k_buffers": [None] * layers,
            "v_buffers": [None] * layers,
            "page_size": cache.page_size,
        }

    def set_kv_buffer_tensors(
        self,
        k_buffers: list,
        v_buffers: list,
        page_size: int,
        slot_layer_ids: list[int] | None = None,
    ) -> None:
        """Keep protocol placeholders; engine cache pages are not framework tensors."""
        _ = (k_buffers, v_buffers, slot_layer_ids)
        if page_size != self.kv_args.page_size:
            raise ValueError("SGLang staging and engine cache page sizes differ")

    def send_kvcache_staged(
        self,
        peer_name: str,
        prefill_kv_indices: np.ndarray,
        dst_staging_ptr: int,
        dst_staging_size: int,
        dst_gpu_id: int,
        dst_tp_rank: int,
        dst_attn_tp_size: int,
        dst_kv_item_len: int,
        notif: str,
        staging_buffer: StagingBuffer | None = None,
        **_kwargs: Any,
    ) -> Any:
        _ = (dst_gpu_id, dst_tp_rank, dst_attn_tp_size, dst_kv_item_len)
        cache = self.engine_cache
        storage = cache.storage if cache is not None else None
        if storage is None or staging_buffer is None:
            raise RuntimeError("engine NIXL staging is not initialized")
        copier = manager_staging_copier(self)
        if copier is None:
            raise RuntimeError("SGLang NIXL has no attached engine Copier")
        page_ids = [int(page) for page in prefill_kv_indices]
        page_count = len(page_ids)
        pool = staging_buffer.attach_cache(storage, cache.pages.chips, copier)
        host_page_bytes = pool.layout.slot_bytes
        transfer_bytes = page_count * host_page_bytes
        if transfer_bytes > staging_buffer.get_size():
            raise ValueError(
                f"prefill chunk needs {transfer_bytes} staging bytes, "
                f"buffer has {staging_buffer.get_size()}"
            )
        if transfer_bytes > dst_staging_size:
            raise ValueError(
                f"prefill chunk needs {transfer_bytes} decode staging bytes, "
                f"available {dst_staging_size}"
            )

        lease = pool.try_acquire(page_count)
        if lease is None:
            raise RuntimeError("SGLang NIXL staging buffer is already in use")
        transport_started = False
        try:
            pages_by_layer = {index: page_ids for index in range(len(storage.layers))}
            pool.copy_pages(cache, pages_by_layer, lease, to_host=True).wait()
            pool.begin_transport(lease)
            transport_started = True
            src_reqs = np.asarray(
                [
                    [staging_buffer.get_ptr() + slot * host_page_bytes, host_page_bytes, 0]
                    for slot in lease.slots
                ],
                dtype=np.int64,
            )
            dst_reqs = np.asarray(
                [
                    [dst_staging_ptr + index * host_page_bytes, host_page_bytes, 0]
                    for index in range(page_count)
                ],
                dtype=np.int64,
            )
            src_descs = self.agent.get_xfer_descs(src_reqs, "DRAM")
            dst_descs = self.agent.get_xfer_descs(dst_reqs, "DRAM")
            handle = self.agent.initialize_xfer(
                "WRITE", src_descs, dst_descs, peer_name, notif.encode("ascii")
            )
            if not handle:
                raise RuntimeError("NIXL could not create the engine staging transfer")
            if self.agent.transfer(handle) == "ERR":
                raise RuntimeError("NIXL could not post the engine staging transfer")
            with self._staging_handle_lock:
                self._staging_handle_leases[id(handle)] = _StagingHandleLease(
                    staging_buffer, pool, lease
                )
            return handle
        except BaseException as transfer_error:
            cleanup_errors = []
            if transport_started:
                try:
                    pool.complete_transport(lease)
                except BaseException as error:
                    cleanup_errors.append(error)
            if not cleanup_errors:
                try:
                    pool.release(lease)
                except BaseException as error:
                    cleanup_errors.append(error)
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "NIXL transfer and staging cleanup failed",
                    [transfer_error, *cleanup_errors],
                ) from None
            raise


class FuriosaNixlKVReceiver(NixlKVReceiver):
    """Require scatter/completion tracking for engine pages even when TP matches."""

    def init(self, *args: Any, **kwargs: Any) -> Any:
        result = super().init(*args, **kwargs)
        if (
            isinstance(self.kv_mgr, FuriosaNixlKVManager)
            and self.kv_mgr.enable_staging
            and self.conclude_state != KVPoll.Failed
        ):
            # SGLang gates this lifecycle on heterogeneous TP; engine pages still need scatter.
            self.require_staging = True
        return result
