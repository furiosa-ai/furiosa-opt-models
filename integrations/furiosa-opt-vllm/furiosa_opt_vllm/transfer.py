"""The vLLM KV transfer settings the plugin serves, each with the requirement the platform's
refusal names.

- `offload` makes `--kv-offloading-size` configure `CONNECTOR`, preserving a NIXL connector beside
  it in vLLM's `MultiConnector`, before the platform checks `TRANSFER`.
- `TRANSFER` names NIXL connector support; shared filesystem offload contracts come from #57.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter

from furiosa_opt_vllm.contracts import Contract
from furiosa_opt_vllm.offloading import (
    CONNECTOR,
    CONNECTOR_MODULE,
    Offloading,
    engine_cache,
    offloading_budget,
    offloading_contracts,
    offloading_extra,
)
from furiosa_opt_vllm.offloading import (
    offload as configure_offload,
)

if TYPE_CHECKING:
    from vllm.config import KVTransferConfig, VllmConfig


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
    """Configure shared offload beside NIXL, or delegate direct offload setup to #57."""
    bytes_to_use = offloading_budget(config)
    if bytes_to_use is None:
        return
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
            entry["kv_connector_extra_config"] = offloading_extra(
                entry.get("kv_connector_extra_config", {}), bytes_to_use
            )
            entries[index] = entry
        else:
            entries.append(
                {
                    "kv_connector": CONNECTOR,
                    "kv_connector_module_path": CONNECTOR_MODULE,
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": offloading_extra({}, bytes_to_use),
                }
            )
        extra["connectors"] = entries
        transfer.kv_connector_extra_config = extra
    elif transfer is not None and is_nixl(transfer):
        from vllm.config import KVTransferConfig

        # Preserve directly configured NIXL settings as one child of MultiConnector.
        entry = TypeAdapter(KVTransferConfig).dump_python(
            transfer, mode="python", exclude_none=True
        )
        config.kv_transfer_config = KVTransferConfig(
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
                        "kv_connector_extra_config": offloading_extra({}, bytes_to_use),
                    },
                ]
            },
        )
    else:
        configure_offload(config)
        return
    config.cache_config.kv_offloading_size = None


def connectors(transfer: KVTransferConfig | None) -> tuple[KVTransferConfig, ...]:
    """The transfer configs of `transfer`'s leaves, following vLLM MultiConnector nesting."""
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


def offloadings(config: VllmConfig) -> tuple[Offloading, ...]:
    """The shared offload settings among `config`'s leaf connectors."""
    return tuple(
        Offloading.of(transfer)
        for transfer in connectors(config.kv_transfer_config)
        if transfer.kv_connector == CONNECTOR
    )


def nixl(config: VllmConfig) -> tuple[KVTransferConfig, ...]:
    """The configs of `config`'s connectors that are vLLM's own NIXL connectors."""
    return tuple(filter(is_nixl, connectors(config.kv_transfer_config)))


TRANSFER: Final[tuple[Contract, ...]] = (
    *offloading_contracts(offloadings),
    Contract(
        option="kv_transfer_config",
        accepts=lambda c: all(
            t.kv_connector != "OffloadingConnector" for t in connectors(c.kv_transfer_config)
        ),
        requirement=f"offloads through {CONNECTOR}, which --kv-offloading-size selects: vLLM's "
        + "OffloadingConnector copies through CUDA streams",
    ),
    *(
        Contract(
            option="kv_transfer_config",
            accepts=lambda c, name=name: all(
                t.kv_connector != name for t in connectors(c.kv_transfer_config)
            ),
            requirement=requirement,
        )
        for name, requirement in UNSTAGED.items()
    ),
    Contract(
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
    Contract(
        option="kv_transfer_config.kv_buffer_device",
        accepts=lambda c: all(transfer.kv_buffer_device == "cpu" for transfer in nixl(c)),
        requirement="stages NIXL's blocks in host buffers, so it takes kv_buffer_device=cpu: "
        + "NIXL cannot register device memory, which the engine does not map for it",
    ),
    # vLLM's scheduler takes a request's blocks as received once each rank's worker reports them,
    # and one worker drives every chip of the replica.
    Contract(
        option="tensor_parallel_size",
        accepts=lambda c: not nixl(c) or c.parallel_config.tensor_parallel_size == 1,
        requirement="connects through NIXL at tensor_parallel_size=1: vLLM awaits each rank; "
        + "one worker drives all chips",
    ),
    Contract(
        option="kv_transfer_config",
        accepts=lambda c: not nixl(c) or engine_cache(c.model_config),
        requirement=(
            "connects the attention cache of a decoder, not recurrent state or encoder memory"
        ),
    ),
    Contract(
        option="kv_transfer_config",
        accepts=lambda c: not nixl(c) or not c.is_mm_encoder_only,
        requirement=(
            "connects the attention cache of a decoder, which an encoder-only instance does not run"
        ),
    ),
    Contract(
        option="kv_transfer_config",
        accepts=lambda c: len(offloadings(c)) <= 1,
        requirement=f"uses at most one {CONNECTOR} in a MultiConnector",
    ),
)
