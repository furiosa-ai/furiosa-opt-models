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
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

from pydantic import TypeAdapter
from vllm import envs

if TYPE_CHECKING:
    from vllm.config import KVTransferConfig, ModelConfig, VllmConfig


# The KV connector `--kv-offloading-size` selects, by the class and module vLLM's factory loads.
CONNECTOR: Final = "RNGDOffloadingConnector"
CONNECTOR_MODULE: Final = "furiosa_opt_vllm.offload"

# vLLM's own NIXL connectors, which stage blocks in host buffers of their own with
# `kv_buffer_device="cpu"`; `NixlConnector` names the pull one.
NIXL: Final = ("NixlConnector", "NixlPullConnector", "NixlPushConnector")

# The engine maps no device memory for another process or library to register.
DEVICE_MEMORY: Final = (
    "registers the cache's device memory, which the engine does not map for it, rather than "
    "staging blocks in host buffers"
)
TENSORS: Final = (
    "copies blocks of the cache's torch tensors itself, rather than staging blocks in host buffers"
)
# vLLM's connectors that reach the cache other than through host buffers of their own.
UNSTAGED: Final[Mapping[str, str]] = MappingProxyType(
    {
        "MooncakeConnector": f"serves no Mooncake connector: it {DEVICE_MEMORY}",
        "MooncakeStoreConnector": f"serves no Mooncake connector: it {DEVICE_MEMORY}",
        "MoRIIOConnector": f"serves no MoRIIOConnector: it {DEVICE_MEMORY}",
        "LMCacheConnectorV1": f"serves no LMCache connector: it {TENSORS}",
        "LMCacheMPConnector": f"serves no LMCache connector: it {TENSORS}",
    }
)


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

    @classmethod
    def of(cls, transfer: KVTransferConfig) -> Offloading:
        extra: Mapping[str, object] = transfer.kv_connector_extra_config
        return cls(
            extra.get("spec_name", "CPUOffloadingSpec") == "CPUOffloadingSpec"
            and "spec_module_path" not in extra,
            bool(extra.get("canonical_layout")),
            transfer.kv_load_failure_policy,
        )


def connectors(transfer: KVTransferConfig | None) -> tuple[KVTransferConfig, ...]:
    """The transfer configs of the connectors `transfer` runs: each one vLLM's MultiConnector
    runs, configured as MultiConnector configures it, or `transfer` itself."""
    from vllm.config import KVTransferConfig

    if transfer is None:
        return ()
    if transfer.kv_connector != "MultiConnector" or transfer.kv_connector_module_path is not None:
        return (transfer,)
    extra: Mapping[str, object] = transfer.kv_connector_extra_config
    entries = TypeAdapter(list[dict[str, object]]).validate_python(extra.get("connectors", []))
    parse = TypeAdapter(KVTransferConfig).validate_python
    return tuple(
        child
        for entry in entries
        for child in connectors(parse({"engine_id": transfer.engine_id, **entry}))
    )


def is_nixl(transfer: KVTransferConfig) -> bool:
    """Whether `transfer` names one of vLLM's own NIXL connectors."""
    return transfer.kv_connector in NIXL and transfer.kv_connector_module_path is None


def offloading(config: VllmConfig) -> tuple[Offloading, ...]:
    """The offloading settings of each connector of `config` that runs through `CONNECTOR`."""
    return tuple(
        Offloading.of(transfer)
        for transfer in connectors(config.kv_transfer_config)
        if transfer.kv_connector == CONNECTOR
    )


def nixl(config: VllmConfig) -> tuple[KVTransferConfig, ...]:
    """The transfer configs of `config`'s connectors that are vLLM's own NIXL connectors."""
    return tuple(filter(is_nixl, connectors(config.kv_transfer_config)))


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
        lambda c: all(
            t.kv_connector != "OffloadingConnector" for t in connectors(c.kv_transfer_config)
        ),
        f"offloads through {CONNECTOR}, which --kv-offloading-size selects: vLLM's "
        + "OffloadingConnector copies through CUDA streams",
    ),
    *(
        (
            "kv_transfer_config",
            lambda c, name=name: all(
                t.kv_connector != name for t in connectors(c.kv_transfer_config)
            ),
            requirement,
        )
        for name, requirement in UNSTAGED.items()
    ),
    (
        "kv_transfer_config",
        lambda c: all(
            is_nixl(t)
            or (t.kv_connector == CONNECTOR and t.kv_connector_module_path == CONNECTOR_MODULE)
            for t in connectors(c.kv_transfer_config)
        ),
        f"serves no KV connector but {CONNECTOR}, which --kv-offloading-size selects, and vLLM's "
        + f"own {', '.join(NIXL[:-1])} and {NIXL[-1]}, which stage blocks in host buffers, or "
        + "vLLM's MultiConnector of them",
    ),
    # The runner copies blocks between the engine cache and NIXL's host buffers; NIXL registers
    # only those.
    (
        "kv_transfer_config.kv_buffer_device",
        lambda c: all(transfer.kv_buffer_device == "cpu" for transfer in nixl(c)),
        "stages NIXL's blocks in host buffers, so it takes kv_buffer_device=cpu: NIXL cannot "
        + "register device memory, which the engine does not map for it",
    ),
    # vLLM's scheduler takes a request's blocks as received once each rank's worker reports them,
    # and one worker drives every chip of the replica.
    (
        "tensor_parallel_size",
        lambda c: not nixl(c) or c.parallel_config.tensor_parallel_size == 1,
        "connects through NIXL at tensor_parallel_size=1: vLLM awaits each rank; one worker "
        + "drives all chips",
    ),
    (
        "kv_transfer_config",
        lambda c: not nixl(c) or engine_cache(c.model_config),
        "connects the attention cache of a decoder, not recurrent state or encoder memory",
    ),
    (
        "kv_transfer_config",
        lambda c: not nixl(c) or not c.is_mm_encoder_only,
        "connects the attention cache of a decoder, which an encoder-only instance does not run",
    ),
    (
        "kv_connector_extra_config",
        lambda c: all(settings.cpu_spec for settings in offloading(c)),
        "keeps offloaded blocks in host memory by vLLM's CPUOffloadingSpec, so it takes no other "
        + "spec_name or spec_module_path",
    ),
    (
        "kv_connector_extra_config",
        lambda c: not any(settings.canonical_layout for settings in offloading(c)),
        "keeps offloaded blocks in the device's order, so it takes no canonical_layout",
    ),
    # A failed copy poisons the blocks and host rows it touched, so no request reruns on them.
    (
        "kv_load_failure_policy",
        lambda c: all(settings.load_failure_policy == "fail" for settings in offloading(c)),
        "stops the engine when an offloaded block fails to copy: kv_load_failure_policy=fail",
    ),
    # vLLM's offloading scheduler ends a job once each of its ranks' workers reports it, and one
    # worker drives every chip of the replica.
    (
        "tensor_parallel_size",
        lambda c: not offloading(c) or c.parallel_config.tensor_parallel_size == 1,
        "offloads at tensor_parallel_size=1: vLLM awaits each rank; one worker drives all chips",
    ),
    (
        "pipeline_parallel_size",
        lambda c: not offloading(c) or c.parallel_config.pipeline_parallel_size == 1,
        "offloads at pipeline_parallel_size=1: vLLM awaits each rank; one worker drives all chips",
    ),
    (
        "kv_offloading_size",
        lambda c: not offloading(c) or engine_cache(c.model_config),
        "offloads the attention cache of a decoder, not recurrent state or encoder memory",
    ),
    (
        "kv_offloading_size",
        lambda c: not offloading(c) or not c.is_mm_encoder_only,
        "offloads the attention cache of a decoder, which an encoder-only instance does not run",
    ),
)
