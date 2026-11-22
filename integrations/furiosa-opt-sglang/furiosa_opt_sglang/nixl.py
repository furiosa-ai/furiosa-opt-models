"""Use SGLang's NIXL room and staging protocol with engine-owned KV pages."""

from __future__ import annotations

import ctypes
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.staging_buffer import StagingAllocator
from sglang.srt.disaggregation.common.staging_handler import DecodeStagingHandler
from sglang.srt.disaggregation.nixl import conn as nixl_conn
from sglang.srt.disaggregation.nixl.conn import NixlKVManager, NixlKVReceiver
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_schedule

import furiosa_opt_engine as engine
from furiosa_opt_engine.cache_transfer import settle as settle_transfers
from furiosa_opt_engine.cache_transfer import transfer_cache_pages
from furiosa_opt_sglang.cache import Cache

FastQueue = nixl_conn.FastQueue
TransferKVChunk = nixl_conn.TransferKVChunk
build_dcp_token_transfer_plan = nixl_conn.build_dcp_token_transfer_plan
NIXL_ERR_SETTLE_TIMEOUT_S = nixl_conn.NIXL_ERR_SETTLE_TIMEOUT_S
_NIXL_TRANSPORT_ERRORS = nixl_conn._NIXL_TRANSPORT_ERRORS
logger = nixl_conn.logger

if TYPE_CHECKING:
    from furiosa_opt_engine import Transfer


class StagingBuffer:
    """Pinned host storage for one bounded SGLang transfer region."""

    def __init__(self, size_bytes: int) -> None:
        self.buffer = engine.HostBuffer(size_bytes)
        self.size_bytes = size_bytes
        view = memoryview(self.buffer)
        try:
            self.data_ptr = ctypes.addressof(ctypes.c_ubyte.from_buffer(view))
        finally:
            view.release()

    def get_ptr(self) -> int:
        return self.data_ptr

    def get_size(self) -> int:
        return self.size_bytes

    def fits(self, required_bytes: int) -> bool:
        return required_bytes <= self.size_bytes


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


class _TransferBatch:
    """A set of per-layer engine DMA operations tracked as one staging scatter."""

    def __init__(
        self,
        handler: EngineDecodeStagingHandler,
        decode_req: Any,
        transfers: Sequence[Transfer],
    ) -> None:
        self.handler = handler
        self.decode_req = decode_req
        self.transfers = tuple(transfers)
        self._lock = threading.Lock()
        self._finished = False

    def query(self) -> bool:
        try:
            if not all(transfer.done() for transfer in self.transfers):
                return False
        except Exception as error:
            self.synchronize(error)
            return True
        self.synchronize()
        return True

    def synchronize(self, observed_error: Exception | None = None) -> None:
        with self._lock:
            if self._finished:
                return
            failure = settle_transfers(self.transfers)
            if failure is None:
                failure = observed_error
            self._finished = True
        with self.handler._transfer_lock:
            self.handler._transfer_batches.discard(self)
        if failure is None:
            return
        self.decode_req._staging_failed = True
        room = self.decode_req.req.bootstrap_room
        self.handler.kv_manager.record_failure(room, str(failure))
        self.handler.kv_manager.update_status(room, KVPoll.Failed)


class _TransferDrain:
    """Drain every submitted engine DMA before SGLang releases staging pages."""

    def __init__(self, handler: EngineDecodeStagingHandler) -> None:
        self.handler = handler

    def synchronize(self) -> None:
        with self.handler._transfer_lock:
            batches = tuple(self.handler._transfer_batches)
        for batch in batches:
            batch.synchronize()


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
            if chunk_idx >= len(infos):
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
            infos[chunk_idx] = (-1, -1, 0, -1, 0)
            return True

    def _scatter_region(
        self,
        staging_offset: int,
        page_start: int,
        num_pages: int,
        decode_req: Any,
    ) -> tuple[Transfer, ...]:
        cache = self.kv_manager.engine_cache
        if cache.storage is None:
            raise RuntimeError("engine KV storage was released during NIXL receive")
        page_size = cache.page_size
        page_bytes = cache.pages.pool.bytes
        host_page_bytes = page_bytes * cache.pages.chips
        if staging_offset % host_page_bytes:
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
            raise ValueError("SGLang request table page count differs from NIXL staging")

        first_host_page = staging_offset // host_page_bytes
        transfers: list[Transfer] = []
        try:
            for layer_id, layer in enumerate(cache.storage.layers):
                start = first_host_page + layer_id * num_pages
                transfers.extend(
                    transfer_cache_pages(
                        layer,
                        self.staging_allocator.buffer.buffer,
                        page_ids,
                        range(start, start + num_pages),
                        page_bytes,
                        cache.pages.chips,
                        to_host=False,
                    )
                )
        except Exception:
            settle_transfers(transfers)
            raise
        return tuple(transfers)


class FuriosaNixlKVManager(NixlKVManager):
    """SGLang 0.5.22 NIXL manager with bounded staging for engine-owned KV pages."""

    force_staging = True

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

    def _init_staging_buffers(self, count: int) -> None:
        from sglang.srt.disaggregation.common.staging_buffer import staging_grid_tokens

        pages = (
            staging_grid_tokens(get_schedule().chunked_prefill_size, self.kv_args.page_size)
            // self.kv_args.page_size
        )
        size = pages * sum(self.kv_args.kv_item_lens)
        self._staging_ctx.buffers = [StagingBuffer(size) for _ in range(count)]
        for buffer in self._staging_ctx.buffers:
            self._register_staging_memory(buffer.get_ptr(), buffer.get_size())

    def _init_staging_allocator(self) -> None:
        size = envs.SGLANG_DISAGG_STAGING_POOL_SIZE_MB.get() * 1024 * 1024
        if size <= 0:
            raise ValueError("SGLANG_DISAGG_STAGING_POOL_SIZE_MB must be positive")
        self._staging_ctx.allocator = BoundedStagingAllocator(size)
        allocator = self._staging_ctx.allocator
        self._register_staging_memory(allocator.get_base_ptr(), allocator.get_total_size())

    def _prepare_payload_xfer(self, peer_info: Any) -> None:
        if self.is_mla_backend or self.is_hybrid_mla_backend:
            raise NotImplementedError("engine NIXL staging requires ordinary attention KV")
        if self.kv_args.state_types:
            raise NotImplementedError("engine NIXL staging transfers attention KV only")
        if peer_info.requires_dcp_relayout:
            raise NotImplementedError("engine NIXL staging does not support DCP relayout")
        if peer_info.decode_tp_size != self.attn_tp_size:
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

    def _prefetch_staging_reqs(self, room: int) -> None:
        if not self.enable_staging or self.kv_buffer_tensors is None:
            return
        if room in self._staging_ctx.prefetched_rooms:
            return
        from sglang.srt.disaggregation.common.staging_handler import (
            prefetch_staging_reqs,
        )

        prefetch_staging_reqs(
            room,
            self.transfer_infos,
            self.kv_buffer_tensors,
            get_schedule().chunked_prefill_size,
            self._staging_ctx.prefetch_requested,
            self._staging_ctx.prefetch_sockets,
        )
        self._staging_ctx.prefetched_rooms.add(room)

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
        if cache.storage is None or staging_buffer is None:
            raise RuntimeError("engine NIXL staging is not initialized")
        page_ids = [int(page) for page in prefill_kv_indices]
        page_count = len(page_ids)
        page_bytes = cache.pages.pool.bytes
        host_page_bytes = page_bytes * cache.pages.chips
        transfer_bytes = page_count * host_page_bytes * len(cache.storage.layers)
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

        transfers: list[Transfer] = []
        try:
            for layer_id, layer in enumerate(cache.storage.layers):
                host_start = layer_id * page_count
                transfers.extend(
                    transfer_cache_pages(
                        layer,
                        staging_buffer.buffer,
                        page_ids,
                        range(host_start, host_start + page_count),
                        page_bytes,
                        cache.pages.chips,
                        to_host=True,
                    )
                )
        except Exception as error:
            settle_transfers(transfers)
            raise error
        if (failure := settle_transfers(transfers)) is not None:
            raise failure

        src_reqs = np.asarray([[staging_buffer.get_ptr(), transfer_bytes, 0]], dtype=np.int64)
        dst_reqs = np.asarray([[dst_staging_ptr, transfer_bytes, 0]], dtype=np.int64)
        src_descs = self.agent.get_xfer_descs(src_reqs, "DRAM")
        dst_descs = self.agent.get_xfer_descs(dst_reqs, "DRAM")
        handle = self.agent.initialize_xfer(
            "WRITE", src_descs, dst_descs, peer_name, notif.encode("ascii")
        )
        if not handle:
            raise RuntimeError("NIXL could not create the engine staging transfer")
        if self.agent.transfer(handle) == "ERR":
            raise RuntimeError("NIXL could not post the engine staging transfer")
        return handle

    def transfer_worker(self, queue: FastQueue, staging_buffer=None, worker_index=0):
        # Pinned SGLang 0.5.22 worker; staging selection accounts for engine-owned pages.
        # Per-worker staging strategy: lazy-created on first chunk so we
        # see kv_buffer_tensors (set by ModelRunner after engine init).
        # Never cache on self -- multiple workers would race the ring.
        staging_strategy = None

        while True:
            kv_chunk: TransferKVChunk = queue.get()
            room = kv_chunk.room
            handles: list[Any] = []
            settle_timed_out = False
            room_transfer_infos = None
            try:
                if room not in self.request_status:
                    logger.debug(
                        "Skipping chunk for room %s because it has been cleared",
                        room,
                    )
                    self._staging_outstanding.pop(room, None)
                    if self.enable_deferred_decode_kv_release:
                        # clear() keeps the target while a chunk is counted.
                        self._maybe_ack_drained_abort(room)
                    continue

                # Counted at dequeue, before the status check, so
                # `outstanding == 0` means nothing is dequeued or in flight --
                # the predicate the abort ack relies on. The flag survives
                # re-enqueue on defer.
                if not kv_chunk.staging_counted:
                    self._staging_outstanding[room] += 1
                    kv_chunk.staging_counted = True

                if self.check_status(room) == KVPoll.Failed:
                    self._staging_outstanding.pop(room, None)
                    if self.enable_deferred_decode_kv_release:
                        # Skipped => nothing written for this aborted room; ack.
                        self._maybe_ack_drained_abort(room)
                    continue

                room_transfer_infos = self.transfer_infos.get(room)
                if room_transfer_infos is None:
                    logger.debug(
                        "Skipping chunk for room %s because its transfer metadata has been cleared",
                        room,
                    )
                    self._staging_outstanding.pop(room, None)
                    if self.enable_deferred_decode_kv_release:
                        self._maybe_ack_drained_abort(room)
                    continue

                # Lazily build a per-worker staging strategy bound to this
                # worker's private staging buffer (matches mooncake).
                if self.enable_staging and staging_strategy is None and staging_buffer is not None:
                    staging_strategy = self._try_create_staging_strategy(staging_buffer)

                self.update_status(room, KVPoll.Transferring)

                reqs_to_be_processed = list(room_transfer_infos.values())
                # Note(kpham-sgl): Pack each DCP rank once into its fixed region.
                # NIXL reads regions asynchronously; the chunk barrier prevents
                # reuse until every transfer completes.
                packed_source_by_dcp_rank = {}

                # Set when staging allocation/watermark is not yet ready and
                # the chunk has been re-enqueued. We then break out of the
                # per-req loop and `continue` the worker main loop without
                # touching room status -- the next pop will retry.
                staging_deferred = False

                for req in reqs_to_be_processed:
                    assert room == req.room
                    if req.is_dummy:
                        continue

                    assert req.agent_name in self.decode_kv_args_table
                    dst_info = self.decode_kv_args_table[req.agent_name]
                    decode_tp_size = dst_info.decode_tp_size

                    # Skip KV RDMA transfer when there are no pages to send
                    # (e.g., decode-side radix cache matched the entire prefix).
                    # Aux data is still sent below when is_last_chunk=True.
                    if len(kv_chunk.prefill_kv_indices) > 0 and self.kv_args.kv_data_ptrs:
                        is_dcp_transfer = dst_info.requires_dcp_relayout
                        if is_dcp_transfer:
                            chunked_dst_kv_indice = req.dst_kv_indices
                        else:
                            chunked_dst_kv_indice = req.dst_kv_indices[kv_chunk.index_slice]

                            # This upstream workaround handles a shorter destination page list.
                            if len(chunked_dst_kv_indice) < len(kv_chunk.prefill_kv_indices):
                                logger.warning(
                                    "destination page count %s is shorter than "
                                    "source page count %s",
                                    len(chunked_dst_kv_indice),
                                    len(kv_chunk.prefill_kv_indices),
                                )
                                kv_chunk.prefill_kv_indices = kv_chunk.prefill_kv_indices[
                                    : len(chunked_dst_kv_indice)
                                ]

                        src_prefill_kv_indices = kv_chunk.prefill_kv_indices

                        notif = (
                            f"{req.room}_kv_{kv_chunk.chunk_id}"
                            f"_{int(kv_chunk.is_last_chunk)}_{self.transfer_source_rank}"
                        )

                        # Decide which kv send path to use:
                        #   1. Staging (native heterogeneous TP or engine-forced same-TP)
                        #   2. send_kvcache (MLA or homogeneous TP)
                        #   3. send_kvcache_slice (heterogeneous TP fallback,
                        #      or staging hard-failed for this chunk)
                        use_staging = (
                            self.enable_staging
                            and staging_strategy is not None
                            and not self.is_mla_backend
                            and not self.is_hybrid_mla_backend
                            and (self.force_staging or decode_tp_size != self.attn_tp_size)
                            and (dst_info.staging_base_ptr != 0 or dst_info.staging_total_size != 0)
                        )
                        if self.force_staging and not use_staging:
                            raise RuntimeError(
                                "engine KV transfer requires a registered bounded staging route"
                            )

                        kv_xfer_handle = None
                        if use_staging:
                            kv_xfer_handle, deferred = self._do_staging_transfer(
                                staging_strategy,
                                kv_chunk,
                                src_prefill_kv_indices,
                                req,
                                dst_info,
                                queue,
                            )
                            if deferred:
                                # Chunk re-enqueued; stop processing remaining
                                # reqs for this chunk and let the worker loop
                                # pick it up again on the next pop.
                                staging_deferred = True
                                break

                        if kv_xfer_handle is None:
                            if is_dcp_transfer:
                                pack_buffer = (
                                    self._dcp_pack_buffers[worker_index]
                                    if self._dcp_pack_buffers
                                    else None
                                )
                                if kv_chunk.num_kv_tokens is None:
                                    raise ValueError("PD DCP transfer requires num_kv_tokens")
                                plan = build_dcp_token_transfer_plan(
                                    src_prefill_kv_indices,
                                    chunked_dst_kv_indice,
                                    physical_page_size=self.kv_args.page_size,
                                    dcp_size=dst_info.dst_dcp_size,
                                    dcp_rank=dst_info.dst_dcp_rank,
                                    src_page_offset=(kv_chunk.index_slice.start or 0),
                                    decode_prefix_len=req.decode_prefix_len or 0,
                                    num_kv_tokens=kv_chunk.num_kv_tokens,
                                )
                                packed_src = self._pack_dcp_rank_once(
                                    pack_buffer,
                                    dst_info,
                                    plan.target_src_token_indices,
                                    packed_source_by_dcp_rank,
                                )
                                handles.extend(
                                    self.send_kvcache_dcp(
                                        req.agent_name,
                                        dst_info,
                                        plan,
                                        notif,
                                        packed_src,
                                    )
                                )
                            elif (
                                self.is_mla_backend
                                or self.is_hybrid_mla_backend
                                or decode_tp_size == self.attn_tp_size
                            ):
                                if dst_info.kv_xfer_segments is None:
                                    if dst_info.dst_homogeneous_mem_kind is None:
                                        raise RuntimeError(
                                            "Missing NIXL destination KV memory kind"
                                        )
                                    kv_xfer_handle = self.send_kvcache(
                                        req.agent_name,
                                        src_prefill_kv_indices,
                                        dst_info.dst_kv_ptrs,
                                        chunked_dst_kv_indice,
                                        dst_info.gpu_id,
                                        notif,
                                        dst_mem_kind=(dst_info.dst_homogeneous_mem_kind),
                                    )
                                else:
                                    handles.extend(
                                        self.send_kvcache_mixed(
                                            req.agent_name,
                                            src_prefill_kv_indices,
                                            chunked_dst_kv_indice,
                                            notif,
                                        )
                                    )
                            else:
                                kv_xfer_handle = self.send_kvcache_slice(
                                    req.agent_name,
                                    src_prefill_kv_indices,
                                    chunked_dst_kv_indice,
                                    notif,
                                )

                        if kv_xfer_handle is not None:
                            handles.append(kv_xfer_handle)

                    if kv_chunk.is_last_chunk:
                        dst_info = self.decode_kv_args_table[req.agent_name]
                        if kv_chunk.state_indices:
                            state_xfer_handles = self.maybe_send_extra(
                                req.agent_name,
                                kv_chunk.state_indices,
                                dst_info.dst_state_data_ptrs,
                                req.dst_state_indices,
                                dst_info.gpu_id,
                                f"{req.room}_state_{self.transfer_source_rank}",
                                decode_tp_size,
                                decode_tp_rank=dst_info.decode_tp_rank,
                                dst_state_item_lens=dst_info.dst_state_item_lens,
                                dst_state_dim_per_tensor=dst_info.dst_state_dim_per_tensor,
                                dst_state_layer_ids=dst_info.dst_state_layer_ids,
                            )
                            handles.extend(h for h in state_xfer_handles if h is not None)

                        if kv_chunk.prefill_aux_index is None:
                            raise RuntimeError("Missing aux index for last chunk")
                        # A no-KV notification still identifies its PP source.
                        # Empty non-final chunks do not consume chunk IDs, so a
                        # final no-KV chunk_id equals the prior KV chunk count.
                        aux_notif = f"{req.room}_aux"
                        if len(kv_chunk.prefill_kv_indices) == 0 or not self.kv_args.kv_data_ptrs:
                            aux_notif += f"_nokv_{self.transfer_source_rank}_{kv_chunk.chunk_id}"
                        aux_xfer_handle = self.send_aux(
                            req.agent_name,
                            kv_chunk.prefill_aux_index,
                            dst_info.dst_aux_ptrs,
                            req.dst_aux_index,
                            aux_notif,
                        )
                        handles.append(aux_xfer_handle)

                if staging_deferred:
                    # Chunk has been re-enqueued; do not advance status.
                    continue

                # Raise only once every handle of this batch settled, not on the
                # first ERR: a sibling still in PROC keeps writing into the
                # decode's KV pages, and the failure path below tells the decode
                # those pages are free.
                settled, any_failed = self._await_handles(handles, failure_seen=False)
                if not settled:
                    settle_timed_out = True
                    raise RuntimeError(
                        f"NIXL transfer for room {room} left a handle running "
                        f"{NIXL_ERR_SETTLE_TIMEOUT_S}s after a peer handle failed"
                    )
                if any_failed:
                    raise RuntimeError(f"NIXL transfer encountered ERR room={room}")

                # Clear with the decrement: the failure path below reads this
                # flag, so a raise after this point must not uncount twice.
                kv_chunk.staging_counted = False
                self._staging_outstanding[room] -= 1
                if self.enable_deferred_decode_kv_release:
                    # Handles all DONE => this room's writes landed; ack if it
                    # was aborted and nothing else is outstanding.
                    self._maybe_ack_drained_abort(room)
                if kv_chunk.is_last_chunk:
                    self.update_status(room, KVPoll.Success)
                elif self.check_status(room) != KVPoll.Success:
                    # A deferred earlier chunk can complete after the last chunk
                    # already concluded Success; don't regress the status.
                    self.update_status(room, KVPoll.Transferring)

                # Drop per-room state only when no chunk is still outstanding and
                # the room has concluded: Success, or a Failed *last* chunk. A
                # non-last Failed chunk keeps the room (more chunks may follow); a
                # late chunk for an already-Failed room is skipped at loop top.
                if self._staging_outstanding.get(room, 0) <= 0 and (
                    self.check_status(room) == KVPoll.Success
                    or (kv_chunk.is_last_chunk and self.check_status(room) == KVPoll.Failed)
                ):
                    self._staging_outstanding.pop(room, None)
                    self.transfer_infos.pop(room, None)
                    self.req_to_decode_prefix_len.pop(room, None)
                    if self.enable_staging and self._staging_ctx is not None:
                        self._staging_ctx.prefetched_rooms.discard(room)
                        # Snapshot first: the scheduler thread adds concurrently.
                        for k in list(self._staging_ctx.prefetch_requested):
                            if k[0] == room:
                                self._staging_ctx.prefetch_requested.discard(k)
            except Exception as e:
                # Catch all exceptions to prevent silently killing this
                # worker thread, but still propagate via failure_exception().
                if isinstance(e, _NIXL_TRANSPORT_ERRORS):
                    logger.warning(f"NIXL transport error for room {room}: {e}")
                else:
                    logger.exception(f"Unexpected transfer worker error for room {room}")
                self.exceptions[room] = e
                # An exception raised while the batch was still being built
                # leaves the handles posted so far running, so settle here too
                # rather than only after the barrier.
                notify = False
                if not settle_timed_out:
                    notify, _ = self._await_handles(handles, failure_seen=True)
                if notify:
                    self.conclude_failure(bootstrap_room=room, failure_reason=str(e))
                    # Every handle settled => the writes are done, but the
                    # normal-path decrement was never reached, so without this
                    # the room's abort ack could never fire.
                    if kv_chunk.staging_counted:
                        kv_chunk.staging_counted = False
                        self._staging_outstanding[room] -= 1
                    if self.enable_deferred_decode_kv_release:
                        self._maybe_ack_drained_abort(room)
                else:
                    # A handle can still write into the decode's KV pages, so
                    # leave the room to the decode's waiting timeout rather
                    # than telling it those pages are free.
                    self.record_failure(room, str(e))
                    self.update_status(room, KVPoll.Failed)
                    # This chunk stays counted in _staging_outstanding, so no
                    # drain ACK can follow; discard the target and fall back to
                    # the timeout.
                    self.poison_deferred_ack_room(room)
                # Settle first; NIXL 1.3.0 undoes a reload when an old handle fails.
                # room_transfer_infos survives the sender's clear() of this room.
                self._reload_invalidated_peers(room_transfer_infos or {})


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


def attach_manager(manager: Any, cache: Cache) -> None:
    if not isinstance(manager, FuriosaNixlKVManager):
        raise TypeError(f"expected {FuriosaNixlKVManager.__name__}, got {type(manager).__name__}")
    manager.attach_cache(cache)


def attach_after_init(original: Any, owner: Any, *args: Any, **kwargs: Any) -> Any:
    manager = original(owner, *args, **kwargs)
    if isinstance(manager, FuriosaNixlKVManager):
        cache = getattr(owner, "token_to_kv_pool", None)
        if not isinstance(cache, Cache):
            raise TypeError("SGLang NIXL adapter requires the engine-owned token cache")
        attach_manager(manager, cache)
    return manager


def select_manager(original: Any, transfer_backend: Any, class_type: Any) -> Any:
    result = original(transfer_backend, class_type)
    if getattr(transfer_backend, "value", None) == "nixl":
        match getattr(class_type, "value", None):
            case "manager":
                return FuriosaNixlKVManager
            case "receiver":
                return FuriosaNixlKVReceiver
    return result
