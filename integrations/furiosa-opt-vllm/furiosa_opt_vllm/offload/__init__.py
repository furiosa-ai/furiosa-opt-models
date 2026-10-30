"""vLLM's KV cache offloading to host memory on RNGD: `RNGDOffloadingConnector`, the connector the
platform makes `--kv-offloading-size` select.

- Its scheduler side is vLLM's offloading scheduler over vLLM's CPU offloading spec: which blocks to
  store and load, their eviction, and the host tier's size.
- Its worker side, `worker.Worker`, follows vLLM's offloading worker on the engine cache that
  `register` gives it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, final

from typing_extensions import override
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import build_offloading_config
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
    OffloadPromMetrics,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

from furiosa_opt_vllm.offload.worker import Host, Worker

if TYPE_CHECKING:
    from collections.abc import Iterable

    import torch
    from vllm.config import VllmConfig
    from vllm.distributed.kv_events import KVCacheEvent
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
    from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
        KVConnectorPromMetrics,
        PromMetric,
        PromMetricT,
    )
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

    import furiosa_opt_engine as engine


@final
class RNGDOffloadingConnector(KVConnectorBase_V1, SupportsHMA):
    """vLLM's offloading of the engine cache to host memory: vLLM's offloading scheduler chooses the
    blocks to store and load, and the worker side moves them once `register` gives it the cache."""

    def __init__(
        self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        self._spec = CPUOffloadingSpec(build_offloading_config(vllm_config, kv_cache_config))
        self._scheduler = (
            OffloadingConnectorScheduler(self._spec, vllm_config, kv_cache_config)
            if role == KVConnectorRole.SCHEDULER
            else None
        )
        self._worker: Worker | None = None

    def register(self, cache: engine.decoder.Cache, chips: int) -> None:
        """Move the blocks of `cache`, whose layers span `chips` chips, from now on: vLLM registers
        caches as torch tensors, which the engine's cache layers are not."""
        chunks = self._spec.num_chunks
        if chunks == 0:
            raise ValueError("kv_offloading_size must hold at least one chunk of cache blocks")
        self._worker = Worker(Host(cache, chips, chunks, self._spec.blocks_per_chunk))

    @property
    def scheduler(self) -> OffloadingConnectorScheduler:
        assert self._scheduler is not None
        return self._scheduler

    @property
    def worker(self) -> Worker:
        assert self._worker is not None, "the runner registers the cache before the first step"
        return self._worker

    # The worker side.

    @override
    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        assert isinstance(kv_connector_metadata, OffloadingConnectorMetadata)
        self.worker.flush(kv_connector_metadata)

    @override
    def start_load_kv(self, forward_context: ForwardContext, **kwargs: object) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, OffloadingConnectorMetadata)
        self.worker.start(metadata)

    @override
    def wait_for_layer_load(self, layer_name: str) -> None:
        # The engine reads a block once the copies writing it ended.
        pass

    @override
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: object,
    ) -> None:
        # Stores move whole blocks of every layer, in `wait_for_save`.
        pass

    @override
    def wait_for_save(self) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, OffloadingConnectorMetadata)
        self.worker.defer(metadata)

    @override
    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        # Stores report through the worker metadata, which fences their blocks' reuse.
        return set(), self.worker.finished()

    @override
    def build_connector_worker_meta(self) -> OffloadingWorkerMetadata | None:
        return None if self._worker is None else self._worker.report()

    @override
    def shutdown(self) -> None:
        if self._scheduler is not None:
            self._scheduler.shutdown()
        if self._worker is not None:
            self._worker.shutdown()

    # The scheduler side, as vLLM's own offloading connector delegates it.

    @property
    @override
    def requires_kv_delivery(self) -> bool:
        # A store that never lands is a later cache miss.
        return False

    @override
    def on_new_request(self, request: Request) -> None:
        self.scheduler.on_new_request(request)

    @override
    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        # The engine cache is one group, so no other group bounds what a load may reach.
        return self.scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    @override
    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ) -> None:
        self.scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    @override
    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        return self.scheduler.build_connector_meta(scheduler_output)

    @override
    def has_pending_push_work(self) -> bool:
        return self.scheduler.has_pending_push_work()

    @override
    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        self.scheduler.update_connector_output(connector_output)

    @override
    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, object] | None]:
        return self.scheduler.request_finished(request)

    @override
    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, object] | None]:
        return self.scheduler.request_finished(request)

    @override
    def take_events(self) -> Iterable[KVCacheEvent]:
        return self.scheduler.take_events()

    @override
    def reset_cache(self) -> bool:
        self.scheduler.reset_cache()
        return True

    @override
    def get_kv_connector_stats(self) -> OffloadingConnectorStats | None:
        return None if self._scheduler is None else self._scheduler.get_stats()

    @classmethod
    @override
    def build_kv_connector_stats(
        cls, data: dict[str, object] | None = None
    ) -> OffloadingConnectorStats:
        return OffloadingConnectorStats() if data is None else OffloadingConnectorStats(data=data)

    @classmethod
    @override
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics:
        return OffloadPromMetrics(vllm_config, metric_types, labelnames, per_engine_labelvalues)
