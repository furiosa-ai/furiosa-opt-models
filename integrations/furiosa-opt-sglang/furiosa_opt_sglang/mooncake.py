"""Adapt SGLang Mooncake's chunked staging path to engine-owned KV pages."""

from __future__ import annotations

import threading
from dataclasses import replace
from typing import Any

import numpy as np
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager, MooncakeKVReceiver
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_schedule

from furiosa_opt_sglang.cache import Cache
from furiosa_opt_sglang.staging import (
    StagingBuffer,
    attach_manager_staging,
    init_decode_staging_allocator,
    init_prefill_staging_buffers,
    manager_staging_copier,
)

FORCE_STAGING_TP_SIZE = -1


class FuriosaMooncakeKVManager(MooncakeKVManager):
    """Keep Mooncake's room/chunk lifecycle and stage engine pages through its ring."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.engine_cache: Cache | None = None
        self._remote_tp_sizes: dict[str, int] = {}
        self._remote_tp_sizes_lock = threading.Lock()
        super().__init__(*args, **kwargs)

    def _register_staging_memory(self, ptr: int, size: int) -> None:
        result = self.engine.batch_register([ptr], [size])
        if result != 0:
            raise RuntimeError(
                f"Mooncake could not register bounded staging memory at 0x{ptr:x} ({size} bytes)"
            )

    def _init_staging_buffers(self, count: int) -> None:
        self._staging_ctx.buffers = init_prefill_staging_buffers(
            self._register_staging_memory,
            self.kv_args,
            count,
            get_schedule().chunked_prefill_size,
            "Mooncake",
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

    def attach_cache(self, cache: Cache) -> None:
        if self.is_mla_backend or self.is_hybrid_mla_backend:
            raise NotImplementedError(
                "SGLang Mooncake engine staging requires ordinary attention KV"
            )
        storage = cache.storage
        if storage is None:
            raise RuntimeError("SGLang Mooncake needs resident engine KV storage")
        page_bytes = cache.pages.pool.bytes * cache.pages.chips
        layers = len(storage.layers)
        if not (
            len(self.kv_args.kv_data_ptrs)
            == len(self.kv_args.kv_data_lens)
            == len(self.kv_args.kv_item_lens)
            == layers
        ):
            raise ValueError("SGLang Mooncake registration differs from engine KV layers")
        if any(item != page_bytes for item in self.kv_args.kv_item_lens):
            raise ValueError("SGLang Mooncake page geometry differs from engine KV storage")
        if any(length != page_bytes for length in self.kv_args.kv_data_lens):
            raise ValueError("SGLang Mooncake registered more than one bounded page per layer")
        if self.kv_args.page_size != cache.page_size:
            raise ValueError("SGLang and engine Mooncake page sizes differ")
        if page_bytes % 2:
            raise ValueError("engine KV page size cannot split into logical K and V items")
        # SGLang's native STAGING_REQ allocator computes one receive region from paired K/V item
        # lengths. Mooncake registers our packed K/V page as one region per layer, so keep the
        # pointer table intact and expose only the equivalent logical lengths to that allocator.
        self.kv_args.kv_item_lens = [page_bytes // 2] * (2 * layers)
        self.engine_cache = cache
        attach_manager_staging(self, storage, cache.pages.chips)
        self.kv_args.kv_head_num = cache.pages.heads
        self.kv_args.total_kv_head_num = cache.pages.heads
        self.kv_buffer_tensors = {
            "k_buffers": [None] * layers,
            "v_buffers": [None] * layers,
            "page_size": cache.page_size,
        }

    def _publish_peer_registration(self, mooncake_session_id: str, decode_kv_args: Any):
        """Make SGLang choose its native staging branch even for equal TP sizes."""
        tp_size = decode_kv_args.dst_attn_tp_size
        with self._remote_tp_sizes_lock:
            previous_tp_size = self._remote_tp_sizes.get(mooncake_session_id)
            self._remote_tp_sizes[mooncake_session_id] = tp_size

            def restore_tp_size() -> None:
                if previous_tp_size is None:
                    self._remote_tp_sizes.pop(mooncake_session_id, None)
                else:
                    self._remote_tp_sizes[mooncake_session_id] = previous_tp_size

            # Workers can observe a registration immediately, so do not mutate its input later.
            try:
                mismatch = super()._publish_peer_registration(
                    mooncake_session_id,
                    replace(decode_kv_args, dst_attn_tp_size=FORCE_STAGING_TP_SIZE),
                )
            except BaseException:
                restore_tp_size()
                raise
            if mismatch is not None:
                restore_tp_size()
            return mismatch

    def _remote_tp_size(self, mooncake_session_id: str) -> int:
        with self._remote_tp_sizes_lock:
            try:
                return self._remote_tp_sizes[mooncake_session_id]
            except KeyError:
                raise RuntimeError(
                    f"Mooncake has no staging TP size for session {mooncake_session_id}"
                ) from None

    def _do_staging_transfer(
        self,
        staging_strategy: Any,
        kv_chunk: Any,
        req: Any,
        target_info: Any,
        chunked_dst_kv_indice: Any,
        executor: Any,
        queue: Any,
        prefill_unique_rank: int,
    ):
        session_id = req.mooncake_session_id
        staging_target = replace(target_info, dst_attn_tp_size=self._remote_tp_size(session_id))
        return super()._do_staging_transfer(
            staging_strategy,
            kv_chunk,
            req,
            staging_target,
            chunked_dst_kv_indice,
            executor,
            queue,
            prefill_unique_rank,
        )

    def maybe_send_extra(
        self,
        req: Any,
        prefill_state_indices: Any,
        executor: Any,
        target_rank_registration_info: Any = None,
    ) -> int:
        if target_rank_registration_info is not None:
            target_rank_registration_info = replace(
                target_rank_registration_info,
                dst_attn_tp_size=self._remote_tp_size(req.mooncake_session_id),
            )
        return super().maybe_send_extra(
            req,
            prefill_state_indices,
            executor,
            target_rank_registration_info,
        )

    def send_kvcache_staged(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: np.ndarray,
        dst_staging_ptr: int,
        dst_staging_size: int,
        dst_tp_rank: int,
        dst_attn_tp_size: int,
        dst_kv_item_len: int,
        dst_layer_ids: list[int],
        staging_buffer: StagingBuffer | None = None,
        dst_slot_layer_ids: list[int] | None = None,
    ) -> int:
        _ = (dst_tp_rank, dst_attn_tp_size, dst_kv_item_len, dst_layer_ids, dst_slot_layer_ids)
        cache = self.engine_cache
        storage = cache.storage if cache is not None else None
        if cache is None or storage is None or staging_buffer is None:
            raise RuntimeError("SGLang Mooncake engine staging is not initialized")
        page_ids = [int(page) for page in prefill_kv_indices]
        if not page_ids:
            return 0
        copier = manager_staging_copier(self)
        if copier is None:
            raise RuntimeError("SGLang Mooncake has no attached engine Copier")
        pool = staging_buffer.attach_cache(storage, cache.pages.chips, copier)
        host_block_bytes = pool.layout.slot_bytes
        transfer_bytes = len(page_ids) * host_block_bytes
        if transfer_bytes > staging_buffer.get_size() or transfer_bytes > dst_staging_size:
            raise ValueError(
                f"Mooncake chunk needs {transfer_bytes} staging bytes, source has "
                f"{staging_buffer.get_size()} and destination has {dst_staging_size}"
            )
        lease = pool.try_acquire(len(page_ids))
        if lease is None:
            raise RuntimeError("SGLang Mooncake staging buffer is already in use")
        pages_by_layer = {index: page_ids for index in range(len(storage.layers))}
        transport_started = False
        try:
            pool.copy_pages(cache, pages_by_layer, lease, to_host=True).wait()
            pool.begin_transport(lease)
            transport_started = True
            transfers = [
                (
                    staging_buffer.get_ptr() + slot * host_block_bytes,
                    dst_staging_ptr + index * host_block_bytes,
                    host_block_bytes,
                )
                for index, slot in enumerate(lease.slots)
            ]
            result = self._transfer_data(mooncake_session_id, transfers)
            if result != 0:
                raise RuntimeError(f"Mooncake engine staging transfer failed with status {result}")
            return result
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
                    "Mooncake transfer and staging cleanup failed",
                    [transfer_error, *cleanup_errors],
                ) from None
            raise


class FuriosaMooncakeKVReceiver(MooncakeKVReceiver):
    """Wait for the engine scatter after SGLang reports a received staging chunk."""

    def init(self, *args: Any, **kwargs: Any) -> Any:
        result = super().init(*args, **kwargs)
        if self.kv_mgr.enable_staging and self.conclude_state != KVPoll.Failed:
            self.require_staging = True
        return result
