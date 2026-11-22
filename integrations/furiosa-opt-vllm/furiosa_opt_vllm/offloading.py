"""Configuration and admission helpers for the vLLM host offloading adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from pydantic import TypeAdapter
from vllm import envs

from furiosa_opt_vllm.contracts import Contract

if TYPE_CHECKING:
    from vllm.config import KVTransferConfig, ModelConfig, VllmConfig


# The connector selected for vLLM's native host and filesystem offloading.
CONNECTOR: Final = "RNGDOffloadingConnector"
CONNECTOR_MODULE: Final = "furiosa_opt_vllm.offload"


@dataclass(frozen=True, slots=True)
class Offloading:
    """The supported spec, page layout, and load failure policy of one offload connector."""

    supported_spec: bool
    canonical_layout: bool
    load_failure_policy: Literal["recompute", "fail"]

    @classmethod
    def of(cls, transfer: KVTransferConfig) -> Offloading:
        extra: Mapping[str, object] = transfer.kv_connector_extra_config
        secondary_tiers = bool(extra.get("secondary_tiers"))
        spec_name = extra.get("spec_name")
        if spec_name is None:
            spec_name = "TieringOffloadingSpec" if secondary_tiers else "CPUOffloadingSpec"
        spec_module = extra.get("spec_module_path")
        supported_spec = spec_module is None and (
            (spec_name == "CPUOffloadingSpec" and not secondary_tiers)
            or spec_name == "TieringOffloadingSpec"
        )
        return cls(
            supported_spec=supported_spec,
            canonical_layout=bool(extra.get("canonical_layout")),
            load_failure_policy=transfer.kv_load_failure_policy,
        )


def offloading(config: VllmConfig) -> Offloading | None:
    """The settings for a config that uses the plugin's offloading connector."""
    transfer = config.kv_transfer_config
    if transfer is None or transfer.kv_connector != CONNECTOR:
        return None
    return Offloading.of(transfer)


def direct_offloadings(config: VllmConfig) -> tuple[Offloading, ...]:
    """The plugin's offloading settings when configured as vLLM's direct connector."""
    settings = offloading(config)
    return () if settings is None else (settings,)


def engine_cache(model: ModelConfig | None) -> bool:
    """Whether the engine holds a decoder's paged attention cache."""
    return model is None or (
        model.runner_type == "generate" and not (model.has_inner_state or model.is_encoder_decoder)
    )


def offloading_extra(extra: object, bytes_to_use: int | float) -> dict[str, object]:
    """Return vLLM offloading options with the byte budget and stock tier inference."""
    options = dict(TypeAdapter(dict[str, object]).validate_python(extra))
    options["cpu_bytes_to_use"] = bytes_to_use
    if options.get("spec_name") is None:
        if options.get("secondary_tiers"):
            options["spec_name"] = "TieringOffloadingSpec"
        else:
            options.pop("spec_name", None)
    return options


def offloading_budget(config: VllmConfig) -> int | float | None:
    """The native offloading byte budget selected by `--kv-offloading-size`, if enabled."""
    cache = config.cache_config
    if (
        cache.kv_offloading_size is None
        or cache.kv_offloading_backend != "native"
        or envs.VLLM_USE_SIMPLE_KV_OFFLOAD
    ):
        return None
    return cache.kv_offloading_size * (1 << 30)


def offload(config: VllmConfig) -> None:
    """Configure the offload connector from `--kv-offloading-size` and consume that option."""
    from vllm.config import KVTransferConfig

    cache = config.cache_config
    bytes_to_use = offloading_budget(config)
    if bytes_to_use is None:
        return
    transfer = config.kv_transfer_config
    if transfer is not None and (
        transfer.kv_connector not in (None, CONNECTOR)
        or transfer.kv_connector_module_path not in (None, CONNECTOR_MODULE)
    ):
        # Leave an explicit connector for admission to accept or refuse; do not erase its choice.
        return
    transfer = transfer or KVTransferConfig()
    transfer.kv_connector = CONNECTOR
    transfer.kv_connector_module_path = CONNECTOR_MODULE
    transfer.kv_role = "kv_both"
    transfer.kv_connector_extra_config = offloading_extra(
        transfer.kv_connector_extra_config, bytes_to_use
    )
    config.kv_transfer_config = transfer
    cache.kv_offloading_size = None


def offloading_contracts(
    settings_for: Callable[[VllmConfig], tuple[Offloading, ...]],
) -> tuple[Contract, ...]:
    """Requirements for the offloading connectors selected from a transfer config."""
    return (
        Contract(
            "kv_offloading_backend",
            lambda config: (
                config.cache_config.kv_offloading_size is None
                or config.cache_config.kv_offloading_backend == "native"
            ),
            "offloads the KV cache with kv_offloading_backend=native",
        ),
        Contract(
            "VLLM_USE_SIMPLE_KV_OFFLOAD",
            lambda config: (
                config.cache_config.kv_offloading_size is None
                or not envs.VLLM_USE_SIMPLE_KV_OFFLOAD
            ),
            f"offloads through {CONNECTOR}, so it serves no VLLM_USE_SIMPLE_KV_OFFLOAD=1",
        ),
        Contract(
            "kv_connector_extra_config",
            lambda config: all(settings.supported_spec for settings in settings_for(config)),
            "uses vLLM's CPUOffloadingSpec for host offload or TieringOffloadingSpec when "
            + "secondary_tiers are configured",
        ),
        Contract(
            "kv_load_failure_policy",
            lambda config: all(
                settings.load_failure_policy == "fail" for settings in settings_for(config)
            ),
            "stops the engine when an offloaded block fails to copy: kv_load_failure_policy=fail",
        ),
        Contract(
            "tensor_parallel_size",
            lambda config: (
                not settings_for(config) or config.parallel_config.tensor_parallel_size == 1
            ),
            "offloads at tensor_parallel_size=1: vLLM awaits each rank; one worker drives all chips",
        ),
        Contract(
            "pipeline_parallel_size",
            lambda config: (
                not settings_for(config) or config.parallel_config.pipeline_parallel_size == 1
            ),
            "offloads at pipeline_parallel_size=1: vLLM awaits each rank; one worker drives all chips",
        ),
        Contract(
            "data_parallel_size",
            lambda config: (
                not settings_for(config) or config.parallel_config.data_parallel_size == 1
            ),
            "offloads at data_parallel_size=1: vLLM ranks each replica's worker among all replicas",
        ),
        Contract(
            "kv_offloading_size",
            lambda config: (
                not settings_for(config) or engine_cache(config.model_config)
            ),
            "offloads the attention cache of a decoder, not recurrent state or encoder memory",
        ),
        Contract(
            "kv_offloading_size",
            lambda config: not settings_for(config) or not config.is_mm_encoder_only,
            "offloads the attention cache of a decoder, which an encoder-only instance does not run",
        ),
        Contract(
            "enable_sleep_mode",
            lambda config: (
                not settings_for(config)
                or not (config.model_config and config.model_config.enable_sleep_mode)
            ),
            "offloads the KV cache without sleep mode, which frees the cache its host tier copies",
        ),
    )


def direct_connector_contracts() -> tuple[Contract, ...]:
    """The connectors accepted by the filesystem offload adapter without composition."""
    def direct_offloading(config: VllmConfig) -> bool:
        transfer = config.kv_transfer_config
        return transfer is None or (
            transfer.kv_connector == CONNECTOR
            and transfer.kv_connector_module_path == CONNECTOR_MODULE
        )

    return (
        Contract(
            "kv_transfer_config",
            lambda config: (
                config.kv_transfer_config is None
                or config.kv_transfer_config.kv_connector != "OffloadingConnector"
            ),
            f"offloads through {CONNECTOR}, which --kv-offloading-size selects: vLLM's "
            + "OffloadingConnector copies through CUDA streams",
        ),
        Contract(
            "kv_transfer_config",
            direct_offloading,
            f"serves no KV connector but {CONNECTOR}, which --kv-offloading-size selects",
        ),
    )
