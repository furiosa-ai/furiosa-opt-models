"""vLLM spec backed by the engine's pinned host slots and batched DMA queue."""

from __future__ import annotations

from typing import Any

from typing_extensions import override
from vllm.v1.kv_offload.base import OffloadingManager, OffloadingMetricMetadata
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec


class _LayoutView:
    """Primary-tier size metadata; cache bytes live in engine-owned pinned slots."""

    def __init__(self, block_size: int) -> None:
        self.strides = (block_size, 1)

    def release(self) -> None:
        pass


class _LayoutRegion:
    """The Tiering manager's layout contract without allocating a duplicate KV mmap."""

    def __init__(self, block_size: int) -> None:
        self.view = _LayoutView(block_size)

    def create_kv_memoryview(self) -> _LayoutView:
        return self.view

    def cleanup(self) -> None:
        pass


class _PinnedPrimaryTier(CPUPrimaryTierOffloadingManager):
    """vLLM CPU metadata and eviction policy over engine-owned pinned memory."""

    def __init__(self, spec: CPUOffloadingSpec) -> None:
        super().__init__(
            num_chunks=spec.num_chunks,
            mmap_region=_LayoutRegion(spec.kv_bytes_per_chunk),  # type: ignore[arg-type]
            cache_policy=spec.eviction_policy,
            cache_policy_module_path=spec.cache_policy_module_path,
            enable_events=spec.kv_events_config.enable_kv_cache_events,
        )


class FuriosaOffloadingSpec(CPUOffloadingSpec):
    """Keep vLLM's scheduler policy while the worker uses #322's transfer plane."""

    @classmethod
    @override
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        if extra_config.get("secondary_tiers"):
            return TieringOffloadingSpec.build_metric_definitions(extra_config)
        return CPUOffloadingSpec.build_metric_definitions(extra_config)

    @override
    def get_manager(self) -> OffloadingManager:
        if self._manager is not None:
            return self._manager
        secondary_configs = self.extra_config.get("secondary_tiers", [])
        if not secondary_configs:
            self._manager = super().get_manager()
            return self._manager

        if int(self.extra_config.get("store_threshold", 0)) >= 2:
            raise ValueError("store_threshold is not supported with filesystem offload")
        if not isinstance(secondary_configs, list):
            raise ValueError("secondary_tiers must be a list of tier configurations")

        primary = _PinnedPrimaryTier(self)
        secondary_tiers = []
        try:
            for tier_config in secondary_configs:
                if not isinstance(tier_config, dict) or tier_config.get("type") != "fs":
                    raise ValueError(
                        "Furiosa vLLM offload supports the filesystem tier in this release"
                    )
                options = {
                    **tier_config,
                    "type": "FuriosaFileSystemTierManager",
                    "module_path": "furiosa_opt_vllm.offload.storage",
                }
                secondary_tiers.append(
                    SecondaryTierFactory.create_secondary_tier(
                        options, primary.get_kv_memoryview(), self
                    )
                )
            self._manager = TieringOffloadingManager(primary, secondary_tiers)
        except Exception:
            for tier in reversed(secondary_tiers):
                tier.shutdown()
            primary.shutdown()
            raise
        return self._manager
