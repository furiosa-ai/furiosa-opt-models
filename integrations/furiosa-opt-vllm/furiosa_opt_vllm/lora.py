"""vLLM's LoRA adapters as engine adapters, kept by vLLM's own LRU caches as its LRU LoRA manager
keeps them.

- Loaded adapters, by LoRA ID: the worker's `add_lora`, or a step that schedules a request naming
  an adapter, builds the adapter's PEFT checkpoint on host, again when the request loads it in
  place; up to `max_cpu_loras` stay loaded, least recently used out first.
- Active adapters: the `max_loras` device adapters, allocated when the model loads, before the
  cache sizes, are the slots of the list each call takes. vLLM activates every adapter it adds; an
  inactive one loads its host revision into the device adapter of a free slot, or of the least
  recently used unpinned active adapter, and holds the slot once the load returned. Each engine
  request names its sequence's slot.
- A pinned adapter stays loaded and in its slot until it is removed. An add or a pin that needs a
  slot while pinned adapters hold every slot is refused before anything changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, final

from vllm.lora.model_manager import AdapterLRUCache

import furiosa_opt_engine as engine

if TYPE_CHECKING:
    from collections.abc import Iterable

    from vllm.config import LoRAConfig
    from vllm.lora.request import LoRARequest
    from vllm.v1.worker.gpu_input_batch import CachedRequestState


@final
class Adapters:
    """The adapters vLLM loaded, by LoRA ID, and the device adapters of the calls' slots, each
    holding the active adapter `lora_index_to_id` names."""

    def __init__(
        self, config: LoRAConfig, spec: engine.decoder.AdapterSpec, positions: int
    ) -> None:
        """The adapters of a decoder of `spec` under vLLM's LoRA `config`; the device adapters
        are allocated now."""
        self.config: LoRAConfig = config
        self.spec: engine.decoder.AdapterSpec = spec
        # The checkpoint's position limit, which vLLM's PEFT parser records.
        self.positions: int = positions
        self.devices: list[engine.decoder.Adapter] = [
            engine.decoder.Adapter(spec) for _ in range(config.max_loras)
        ]
        self.lora_index_to_id: list[int | None] = [None] * config.max_loras
        # vLLM's caches: an inactive adapter frees its slot, an unloaded one leaves it.
        self.active: AdapterLRUCache[None] = AdapterLRUCache(config.max_loras, self._free)
        self.registered: AdapterLRUCache[engine.decoder.HostAdapter] = AdapterLRUCache(
            config.max_cpu_loras or config.max_loras, lambda lora_id: self.active.pop(lora_id, None)
        )

    @property
    def slots(self) -> list[engine.decoder.Adapter | None]:
        """The list every call takes: the device adapter of each slot an adapter holds."""
        return [
            None if lora_id is None else device
            for device, lora_id in zip(self.devices, self.lora_index_to_id, strict=True)
        ]

    def add(self, request: LoRARequest) -> bool:
        """Build the adapter of `request` on host unless it is loaded, or again when `request`
        loads it in place, unloading the least recently used one past `max_cpu_loras`; then
        activate it."""
        lora_id = request.lora_int_id
        if lora_id not in self.registered or request.load_inplace:
            self._admit(lora_id)
            host = self._read(request)
            _ = self.remove(lora_id)
            if len(self.registered) + 1 > self.registered.capacity:
                self.registered.remove_oldest()
            self.registered[lora_id] = host
        else:
            # A lookup touches the adapter's place in the cache.
            _ = self.registered.get(lora_id)
        self._activate(lora_id)
        return True

    def remove(self, lora_id: int) -> bool:
        """Unload adapter `lora_id`, freeing its slot; `False` if it is not loaded. A request that
        names it loads it again."""
        return self.registered.pop(lora_id, None) is not None

    def remove_all(self) -> None:
        """Unload every adapter."""
        self.registered.clear()

    def pin(self, lora_id: int) -> bool:
        """Keep the loaded adapter `lora_id` loaded and in a slot until it is removed."""
        if lora_id not in self.registered:
            raise ValueError(f"Pinning failed. LoRA adapter {lora_id} is not loaded")
        self._admit(lora_id)
        self.registered.pin(lora_id)
        self._activate(lora_id)
        self.active.pin(lora_id)
        return True

    def ids(self) -> set[int]:
        """The LoRA IDs of the loaded adapters."""
        return set(self.registered)

    def activate(self, requests: Iterable[CachedRequestState]) -> dict[str, int]:
        """The slot of each of `requests` that names an adapter, once vLLM added each of their
        adapters in turn."""
        named = {
            request.req_id: lora
            for request in requests
            if (lora := request.lora_request) is not None and lora.lora_int_id
        }
        adapters = {lora.lora_int_id: lora for lora in named.values()}
        if len(adapters) > (slots := self.config.max_loras):
            requested = f"Number of requested LoRAs ({len(adapters)})"
            raise RuntimeError(
                f"{requested} is greater than the number of GPU LoRA slots ({slots})."
            )
        for lora in adapters.values():
            _ = self.add(lora)
        # An adapter a later one of the step evicted has no slot, which vLLM fails on as well.
        return {
            req_id: self.lora_index_to_id.index(lora.lora_int_id) for req_id, lora in named.items()
        }

    def _admit(self, lora_id: int) -> None:
        """Refuse an adapter `lora_id` needing a slot that only pinned adapters hold."""
        active = self.active
        full = len(active) >= active.capacity and set(active) <= active.pinned_items
        if full and lora_id not in active:
            held = "pinned adapters hold every slot"
            raise RuntimeError(f"LoRA adapter {lora_id}: {held}; raise --max-loras")

    def _activate(self, lora_id: int) -> None:
        """Put the loaded adapter `lora_id` in a slot unless it holds one, loading its host
        revision into the slot's device adapter; the slot holds it once the load returned."""
        if lora_id not in self.active:
            if len(self.active) >= self.active.capacity:
                self.active.remove_oldest()
            index = self.lora_index_to_id.index(None)
            self.devices[index].load(self.registered[lora_id])
            self.active[lora_id] = None
            self.lora_index_to_id[index] = lora_id
        self.active.touch(lora_id)

    def _free(self, lora_id: int) -> None:
        """Free the slot of the adapter `lora_id`, which left the active ones."""
        if lora_id in self.lora_index_to_id:
            self.lora_index_to_id[self.lora_index_to_id.index(lora_id)] = None

    def _read(self, request: LoRARequest) -> engine.decoder.HostAdapter:
        """The PEFT checkpoint of `request` on host, its `adapter_config.json` read once for
        vLLM's checks and the engine's."""
        from vllm.lora.peft_helper import PEFTHelper
        from vllm.lora.utils import get_adapter_absolute_path

        if request.tensorizer_config_dict is not None:
            raise ValueError(f"LoRA adapter {request.lora_name}: reads no tensorizer checkpoint")
        path = Path(get_adapter_absolute_path(request.lora_path))
        if not (path / "adapter_model.safetensors").is_file():
            raise ValueError(f"LoRA adapter {request.lora_name}: reads adapter_model.safetensors")
        config = json.loads((path / "adapter_config.json").read_text())
        # vLLM's own refusals, such as a rank above --max-lora-rank.
        peft = PEFTHelper.from_dict(config | {"vllm_max_position_embeddings": self.positions})
        peft.validate_legal(self.config)
        return engine.decoder.HostAdapter(self.spec, engine.Checkpoint(path, config))
