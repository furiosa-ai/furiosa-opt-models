"""vLLM spec backed by the engine's pinned host slots and batched DMA queue."""

from __future__ import annotations

from typing import Any, cast

from typing_extensions import override
from vllm.v1.kv_offload.base import OffloadingManager, OffloadingMetricMetadata
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec


def _secondary_tier_configs(extra_config: dict[str, Any]) -> list[dict[str, Any]]:
    secondary_configs = extra_config.get("secondary_tiers", [])
    if not isinstance(secondary_configs, list):
        raise ValueError("secondary_tiers must be a list of tier configurations")
    for index, config in enumerate(secondary_configs):
        if not isinstance(config, dict):
            raise ValueError(f"secondary_tiers[{index}] must be a tier configuration object")
    return cast(list[dict[str, Any]], secondary_configs)


def _tier_factory_config(tier_config: dict[str, Any]) -> dict[str, Any]:
    tier_type = tier_config.get("type")
    if tier_type == "fs":
        manager_type = "FuriosaFileSystemTierManager"
        module_path = "furiosa_opt_vllm.offload.storage"
    else:
        raise ValueError(
            f"unsupported secondary tier type {tier_type!r}; "
            "Furiosa vLLM offload supports only 'fs'; use the native Mooncake Store connector "
            "for shared Mooncake storage"
        )
    return {**tier_config, "type": manager_type, "module_path": module_path}


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
    """Keep vLLM's aggregate sizing and scheduler policy; engine slots hold all TP shards."""

    @property
    def coalescing(self) -> bool:
        """Whether one vLLM step's loads share an engine staging batch."""
        value = self.extra_config.get("coalescing", False)
        if type(value) is not bool:
            raise ValueError("coalescing must be a boolean")
        return value

    @classmethod
    @override
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        secondary_configs = _secondary_tier_configs(extra_config)
        if secondary_configs:
            metric_config = {
                **extra_config,
                "secondary_tiers": [
                    _tier_factory_config(tier_config) for tier_config in secondary_configs
                ],
            }
            return TieringOffloadingSpec.build_metric_definitions(metric_config)
        return CPUOffloadingSpec.build_metric_definitions(extra_config)

    @override
    def get_manager(self) -> OffloadingManager:
        if self._manager is not None:
            return self._manager
        secondary_configs = _secondary_tier_configs(self.extra_config)
        if not secondary_configs:
            self._manager = super().get_manager()
            return self._manager

        if int(self.extra_config.get("store_threshold", 0)) >= 2:
            raise ValueError("store_threshold is not supported with secondary storage offload")

        primary = _PinnedPrimaryTier(self)
        secondary_tiers = []
        try:
            for tier_config in secondary_configs:
                secondary_tiers.append(
                    SecondaryTierFactory.create_secondary_tier(
                        _tier_factory_config(tier_config), primary.get_kv_memoryview(), self
                    )
                )
            self._manager = TieringOffloadingManager(primary, secondary_tiers)
        except BaseException as error:
            cleanup_errors = []
            for resource in (*reversed(secondary_tiers), primary):
                try:
                    resource.shutdown()
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            if cleanup_errors:
                error.add_note(
                    "offloading setup cleanup failed: "
                    + "; ".join(repr(cleanup_error) for cleanup_error in cleanup_errors)
                )
            raise
        return self._manager
