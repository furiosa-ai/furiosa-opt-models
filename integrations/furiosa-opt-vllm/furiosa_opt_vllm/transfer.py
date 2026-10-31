"""The vLLM KV transfer settings the plugin serves, each with the requirement the platform's
refusal names.

- `offload` makes `--kv-offloading-size` select the plugin's KV connector, `CONNECTOR`, before the
  platform checks `TRANSFER`.
- `TRANSFER` checks one option each, as `(option, accepts, requirement)`; `admission` makes them
  contracts in its own order.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from vllm import envs

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig


# The KV connector `--kv-offloading-size` selects, by the class and module vLLM's factory loads.
CONNECTOR: Final = "RNGDOffloadingConnector"
CONNECTOR_MODULE: Final = "furiosa_opt_vllm.offload"


def offload(config: VllmConfig) -> None:
    """Turn `--kv-offloading-size` into the transfer config vLLM derives from it for its own
    connector, naming `CONNECTOR` instead, and take the size: vLLM derives its config after the
    platform's check, from a size it still sees."""
    from vllm.config import KVTransferConfig

    cache = config.cache_config
    if (
        cache.kv_offloading_size is None
        or cache.kv_offloading_backend != "native"
        or envs.VLLM_USE_SIMPLE_KV_OFFLOAD
    ):
        return
    transfer = config.kv_transfer_config or KVTransferConfig()
    transfer.kv_connector = CONNECTOR
    transfer.kv_connector_module_path = CONNECTOR_MODULE
    transfer.kv_role = "kv_both"
    transfer.kv_connector_extra_config["cpu_bytes_to_use"] = cache.kv_offloading_size * (1 << 30)
    config.kv_transfer_config = transfer
    cache.kv_offloading_size = None


@dataclass(frozen=True, slots=True)
class Offloading:
    """What the contracts check of an offloading transfer config through `CONNECTOR`, with its
    `kv_connector_extra_config` keys read as vLLM's offloading reads them."""

    # No spec_name but vLLM's CPUOffloadingSpec, and no spec_module_path.
    cpu_spec: bool
    canonical_layout: bool
    load_failure_policy: Literal["recompute", "fail"]


def offloading(config: VllmConfig) -> Offloading | None:
    """The offloading settings of `config` when it runs through `CONNECTOR`, or none without it."""
    transfer = config.kv_transfer_config
    if transfer is None or transfer.kv_connector != CONNECTOR:
        return None
    extra: Mapping[str, object] = transfer.kv_connector_extra_config
    return Offloading(
        extra.get("spec_name", "CPUOffloadingSpec") == "CPUOffloadingSpec"
        and "spec_module_path" not in extra,
        bool(extra.get("canonical_layout")),
        transfer.kv_load_failure_policy,
    )


def engine_cache(model: ModelConfig | None) -> bool:
    """Whether the engine holds `model`'s cache: the paged attention cache of a generating decoder
    with no recurrent state or encoder memory."""
    return model is None or (
        model.runner_type == "generate" and not (model.has_inner_state or model.is_encoder_decoder)
    )


TRANSFER: Final[tuple[tuple[str, Callable[[VllmConfig], bool], str], ...]] = (
    (
        "kv_offloading_backend",
        lambda c: (
            c.cache_config.kv_offloading_size is None
            or c.cache_config.kv_offloading_backend == "native"
        ),
        "offloads the KV cache with kv_offloading_backend=native",
    ),
    (
        "VLLM_USE_SIMPLE_KV_OFFLOAD",
        lambda c: c.cache_config.kv_offloading_size is None or not envs.VLLM_USE_SIMPLE_KV_OFFLOAD,
        f"offloads through {CONNECTOR}, so it serves no VLLM_USE_SIMPLE_KV_OFFLOAD=1",
    ),
    (
        "kv_transfer_config",
        lambda c: (
            c.kv_transfer_config is None
            or c.kv_transfer_config.kv_connector != "OffloadingConnector"
        ),
        f"offloads through {CONNECTOR}, which --kv-offloading-size selects: vLLM's "
        + "OffloadingConnector copies through CUDA streams",
    ),
    (
        "kv_transfer_config",
        lambda c: (
            c.kv_transfer_config is None
            or (
                c.kv_transfer_config.kv_connector == CONNECTOR
                and c.kv_transfer_config.kv_connector_module_path == CONNECTOR_MODULE
            )
        ),
        f"serves no KV connector but {CONNECTOR}, which --kv-offloading-size selects",
    ),
    (
        "kv_connector_extra_config",
        lambda c: (settings := offloading(c)) is None or settings.cpu_spec,
        "keeps offloaded blocks in host memory by vLLM's CPUOffloadingSpec, so it takes no other "
        + "spec_name or spec_module_path",
    ),
    (
        "kv_connector_extra_config",
        lambda c: (settings := offloading(c)) is None or not settings.canonical_layout,
        "keeps offloaded blocks in the device's order, so it takes no canonical_layout",
    ),
    # A failed copy poisons the blocks and host rows it touched, so no request reruns on them.
    (
        "kv_load_failure_policy",
        lambda c: (settings := offloading(c)) is None or settings.load_failure_policy == "fail",
        "stops the engine when an offloaded block fails to copy: kv_load_failure_policy=fail",
    ),
    # vLLM's offloading scheduler ends a job once each of its ranks' workers reports it, and one
    # worker drives every chip of the replica.
    (
        "tensor_parallel_size",
        lambda c: offloading(c) is None or c.parallel_config.tensor_parallel_size == 1,
        "offloads at tensor_parallel_size=1: vLLM awaits each rank; one worker drives all chips",
    ),
    (
        "pipeline_parallel_size",
        lambda c: offloading(c) is None or c.parallel_config.pipeline_parallel_size == 1,
        "offloads at pipeline_parallel_size=1: vLLM awaits each rank; one worker drives all chips",
    ),
    (
        "kv_offloading_size",
        lambda c: offloading(c) is None or engine_cache(c.model_config),
        "offloads the attention cache of a decoder, not recurrent state or encoder memory",
    ),
    (
        "kv_offloading_size",
        lambda c: offloading(c) is None or not c.is_mm_encoder_only,
        "offloads the attention cache of a decoder, which an encoder-only instance does not run",
    ),
)
