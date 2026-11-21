"""The vLLM KV transfer settings the plugin serves, each with the requirement the platform's
refusal names.

- `offload` makes `--kv-offloading-size` configure `CONNECTOR`, preserving a NIXL connector beside
  it in vLLM's `MultiConnector`, before the platform checks `TRANSFER`.
- `TRANSFER` names each option, predicate, and refusal requirement; `admission` makes them
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


@dataclass(frozen=True, slots=True)
class TransferContract:
    """One vLLM option's acceptance predicate and the refusal requirement it must meet."""

    option: str
    accepts: Callable[[VllmConfig], bool]
    requirement: str


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


@dataclass(frozen=True, slots=True)
class Offloading:
    """What the contracts check of an offloading transfer config through `CONNECTOR`, with its
    `kv_connector_extra_config` keys read as vLLM's offloading reads them."""

    # vLLM's stock CPU spec, or stock tiering spec when secondary tiers are configured.
    supported_spec: bool
    load_failure_policy: Literal["recompute", "fail"]

    @classmethod
    def of(cls, transfer: KVTransferConfig) -> Offloading:
        extra: Mapping[str, object] = transfer.kv_connector_extra_config
        secondary_tiers = bool(extra.get("secondary_tiers"))
        spec_name = extra.get("spec_name")
        if spec_name is None:
            spec_name = "TieringOffloadingSpec" if secondary_tiers else "CPUOffloadingSpec"
        spec_module = extra.get("spec_module_path")
        return cls(
            spec_module is None
            and (
                (spec_name == "CPUOffloadingSpec" and not secondary_tiers)
                or spec_name == "TieringOffloadingSpec"
            ),
            transfer.kv_load_failure_policy,
        )


def offload(config: VllmConfig) -> None:
    """Configure `CONNECTOR` from `--kv-offloading-size`, preserving NIXL beside it when set, and
    clear the size before vLLM derives its own transfer config after the platform's check."""
    from vllm.config import KVTransferConfig

    cache = config.cache_config
    if (
        cache.kv_offloading_size is None
        or cache.kv_offloading_backend != "native"
        or envs.VLLM_USE_SIMPLE_KV_OFFLOAD
    ):
        return
    bytes_to_use = cache.kv_offloading_size * (1 << 30)
    transfer = config.kv_transfer_config
    if (
        transfer is not None
        and transfer.kv_connector == "MultiConnector"
        and transfer.kv_connector_module_path is None
    ):
        extra = dict(transfer.kv_connector_extra_config)
        entries = TypeAdapter(list[dict[str, object]]).validate_python(extra.get("connectors", []))
        offload_entries = [
            index
            for index, entry in enumerate(entries)
            if entry.get("kv_connector") == CONNECTOR
            and entry.get("kv_connector_module_path") == CONNECTOR_MODULE
        ]
        if len(offload_entries) > 1:
            raise ValueError("--kv-offloading-size can configure only one RNGDOffloadingConnector")
        if offload_entries:
            index = offload_entries[0]
            entry = dict(entries[index])
            entry["kv_connector_extra_config"] = _offloading_extra(
                entry.get("kv_connector_extra_config", {}), bytes_to_use
            )
            entries[index] = entry
        else:
            entries.append(
                {
                    "kv_connector": CONNECTOR,
                    "kv_connector_module_path": CONNECTOR_MODULE,
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": _offloading_extra({}, bytes_to_use),
                }
            )
        extra["connectors"] = entries
        transfer.kv_connector_extra_config = extra
    elif transfer is not None and is_nixl(transfer):
        # Preserve a directly configured NIXL connector when offloading adds its own adapter.
        entry = TypeAdapter(KVTransferConfig).dump_python(
            transfer, mode="python", exclude_none=True
        )
        transfer = KVTransferConfig(
            engine_id=transfer.engine_id,
            kv_connector="MultiConnector",
            kv_role=transfer.kv_role,
            kv_connector_extra_config={
                "connectors": [
                    entry,
                    {
                        "kv_connector": CONNECTOR,
                        "kv_connector_module_path": CONNECTOR_MODULE,
                        "kv_role": "kv_both",
                        "kv_connector_extra_config": _offloading_extra({}, bytes_to_use),
                    },
                ]
            },
        )
        config.kv_transfer_config = transfer
    else:
        transfer = transfer or KVTransferConfig()
        transfer.kv_connector = CONNECTOR
        transfer.kv_connector_module_path = CONNECTOR_MODULE
        transfer.kv_role = "kv_both"
        transfer.kv_connector_extra_config = _offloading_extra(
            transfer.kv_connector_extra_config, bytes_to_use
        )
        config.kv_transfer_config = transfer
    cache.kv_offloading_size = None


def _offloading_extra(extra: object, bytes_to_use: int) -> dict[str, object]:
    """vLLM's offloading options with its byte budget and stock tier inference."""
    options = dict(TypeAdapter(dict[str, object]).validate_python(extra))
    options["cpu_bytes_to_use"] = bytes_to_use
    if options.get("spec_name") is None:
        if options.get("secondary_tiers"):
            options["spec_name"] = "TieringOffloadingSpec"
        else:
            options.pop("spec_name", None)
    return options


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


TRANSFER: Final[tuple[TransferContract, ...]] = (
    TransferContract(
        option="kv_offloading_backend",
        accepts=lambda c: (
            c.cache_config.kv_offloading_size is None
            or c.cache_config.kv_offloading_backend == "native"
        ),
        requirement="offloads the KV cache with kv_offloading_backend=native",
    ),
    TransferContract(
        option="VLLM_USE_SIMPLE_KV_OFFLOAD",
        accepts=lambda c: (
            c.cache_config.kv_offloading_size is None or not envs.VLLM_USE_SIMPLE_KV_OFFLOAD
        ),
        requirement=f"offloads through {CONNECTOR}, so it serves no VLLM_USE_SIMPLE_KV_OFFLOAD=1",
    ),
    TransferContract(
        option="kv_transfer_config",
        accepts=lambda c: all(
            t.kv_connector != "OffloadingConnector" for t in connectors(c.kv_transfer_config)
        ),
        requirement=f"offloads through {CONNECTOR}, which --kv-offloading-size selects: vLLM's "
        + "OffloadingConnector copies through CUDA streams",
    ),
    *(
        TransferContract(
            option="kv_transfer_config",
            accepts=lambda c, name=name: all(
                t.kv_connector != name for t in connectors(c.kv_transfer_config)
            ),
            requirement=requirement,
        )
        for name, requirement in UNSTAGED.items()
    ),
    TransferContract(
        option="kv_transfer_config",
        accepts=lambda c: all(
            is_nixl(t)
            or (t.kv_connector == CONNECTOR and t.kv_connector_module_path == CONNECTOR_MODULE)
            for t in connectors(c.kv_transfer_config)
        ),
        requirement=f"serves no KV connector but {CONNECTOR}, which --kv-offloading-size "
        + "selects, and vLLM's "
        + f"own {', '.join(NIXL[:-1])} and {NIXL[-1]}, which stage blocks in host buffers, or "
        + "vLLM's MultiConnector of them",
    ),
    # The runner copies blocks between the engine cache and NIXL's host buffers; NIXL registers
    # only those.
    TransferContract(
        option="kv_transfer_config.kv_buffer_device",
        accepts=lambda c: all(transfer.kv_buffer_device == "cpu" for transfer in nixl(c)),
        requirement="stages NIXL's blocks in host buffers, so it takes kv_buffer_device=cpu: "
        + "NIXL cannot register device memory, which the engine does not map for it",
    ),
    # vLLM's scheduler takes a request's blocks as received once each rank's worker reports them,
    # and one worker drives every chip of the replica.
    TransferContract(
        option="tensor_parallel_size",
        accepts=lambda c: not nixl(c) or c.parallel_config.tensor_parallel_size == 1,
        requirement="connects through NIXL at tensor_parallel_size=1: vLLM awaits each rank; "
        + "one worker drives all chips",
    ),
    TransferContract(
        option="kv_transfer_config",
        accepts=lambda c: not nixl(c) or engine_cache(c.model_config),
        requirement=(
            "connects the attention cache of a decoder, not recurrent state or encoder memory"
        ),
    ),
    TransferContract(
        option="kv_transfer_config",
        accepts=lambda c: not nixl(c) or not c.is_mm_encoder_only,
        requirement=(
            "connects the attention cache of a decoder, which an encoder-only instance does not run"
        ),
    ),
    TransferContract(
        option="kv_connector_extra_config",
        accepts=lambda c: all(settings.supported_spec for settings in offloading(c)),
        requirement="uses vLLM's CPUOffloadingSpec for host offload or stock "
        + "TieringOffloadingSpec when secondary tiers are configured",
    ),
    TransferContract(
        option="kv_transfer_config",
        accepts=lambda c: len(offloading(c)) <= 1,
        requirement=f"uses at most one {CONNECTOR} in a MultiConnector",
    ),
    # A failed copy poisons the blocks and host rows it touched, so no request reruns on them.
    TransferContract(
        option="kv_load_failure_policy",
        accepts=lambda c: all(settings.load_failure_policy == "fail" for settings in offloading(c)),
        requirement="stops the engine when an offloaded block fails to copy: "
        + "kv_load_failure_policy=fail",
    ),
    # vLLM's offloading scheduler ends a job once each of its ranks' workers reports it, and one
    # worker drives every chip of the replica.
    TransferContract(
        option="tensor_parallel_size",
        accepts=lambda c: not offloading(c) or c.parallel_config.tensor_parallel_size == 1,
        requirement="offloads at tensor_parallel_size=1: vLLM awaits each rank; "
        + "one worker drives all chips",
    ),
    TransferContract(
        option="pipeline_parallel_size",
        accepts=lambda c: not offloading(c) or c.parallel_config.pipeline_parallel_size == 1,
        requirement="offloads at pipeline_parallel_size=1: vLLM awaits each rank; "
        + "one worker drives all chips",
    ),
    # Each worker joins vLLM's process groups as rank 0 of 1, which data parallelism renumbers.
    TransferContract(
        option="data_parallel_size",
        accepts=lambda c: not offloading(c) or c.parallel_config.data_parallel_size == 1,
        requirement="offloads at data_parallel_size=1: vLLM ranks each replica's worker among "
        + "all replicas",
    ),
    TransferContract(
        option="kv_offloading_size",
        accepts=lambda c: not offloading(c) or engine_cache(c.model_config),
        requirement=(
            "offloads the attention cache of a decoder, not recurrent state or encoder memory"
        ),
    ),
    TransferContract(
        option="kv_offloading_size",
        accepts=lambda c: not offloading(c) or not c.is_mm_encoder_only,
        requirement=(
            "offloads the attention cache of a decoder, which an encoder-only instance does not run"
        ),
    ),
    # Sleep mode frees the cache storage and allocates new storage on waking, while the host tier
    # keeps copying the storage it was built over.
    TransferContract(
        option="enable_sleep_mode",
        accepts=lambda c: (
            not offloading(c) or not (c.model_config and c.model_config.enable_sleep_mode)
        ),
        requirement="offloads the KV cache without sleep mode, which frees the cache its host tier "
        + "copies",
    ),
)
