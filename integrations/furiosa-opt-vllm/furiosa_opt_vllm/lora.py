"""vLLM's LoRA adapters as engine adapters, kept as vLLM's LRU LoRA manager keeps them.

- Loaded adapters, by LoRA ID: the worker's `add_lora`, or a step that schedules a request naming
  an adapter, loads the adapter's PEFT checkpoint onto the device, again when the request asks to
  load in place; up to `max_cpu_loras` stay loaded, least recently used out first.
- Active adapters: the slots of the list each call takes, `max_loras` of them. A load puts its
  adapter into a slot, as vLLM activates every adapter it adds, and each engine request names its
  sequence's slot; a slot holds an adapter by reference, which the engine reads where it lies.
- A pinned adapter stays loaded and in its slot until it is removed. A load or a pin that would
  have to evict a pinned adapter, or an adapter of the step, from the loaded adapters or from the
  slots is refused before anything changes.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Final, final

import furiosa_opt_engine as engine

if TYPE_CHECKING:
    from collections.abc import Iterable, Set

    from vllm.config import LoRAConfig
    from vllm.lora.request import LoRARequest
    from vllm.v1.worker.gpu_input_batch import CachedRequestState

# The LoRA IDs a load keeps by default besides the pinned ones: none.
NOTHING: Final[frozenset[int]] = frozenset()


@final
class Adapters:
    """The adapters vLLM loaded, by LoRA ID, and the slot each active one holds in the calls'
    list."""

    def __init__(
        self, config: LoRAConfig, spec: engine.decoder.AdapterSpec, positions: int
    ) -> None:
        self.config: LoRAConfig = config
        self.spec: engine.decoder.AdapterSpec = spec
        # The checkpoint's position limit, which vLLM's PEFT parser records.
        self.positions: int = positions
        # LoRA ID -> its engine adapter, least recently used first.
        self.loaded: OrderedDict[int, engine.decoder.Adapter] = OrderedDict()
        self.pins: set[int] = set()
        # LoRA ID -> its slot, least recently used first.
        self.held: OrderedDict[int, int] = OrderedDict()
        # The list every call takes: the adapter of each slot, or none.
        self.slots: list[engine.decoder.Adapter | None] = [None] * config.max_loras

    def add(self, request: LoRARequest, keep: Set[int] = NOTHING) -> bool:
        """Load the adapter of `request` unless it is loaded, or again in place when `request`
        asks to, under its pin, and put it in a slot unless it holds one; refused, with nothing
        changed, when it needs room among the loaded adapters or the slots that only pinned
        adapters and those in `keep` hold."""
        lora_id = request.lora_int_id
        unloaded, displaced = self._admit(lora_id, keep)
        if lora_id not in self.loaded or request.load_inplace:
            adapter = self._read(request)
            if unloaded is not None:
                _ = self.remove(unloaded)
            self.loaded[lora_id] = adapter
        self.loaded.move_to_end(lora_id)
        self._hold(lora_id, displaced)
        return True

    def remove(self, lora_id: int) -> bool:
        """Unload adapter `lora_id`; `False` if it is not loaded. A request that names it loads it
        again."""
        self.pins.discard(lora_id)
        if (slot := self.held.pop(lora_id, None)) is not None:
            self.slots[slot] = None
        return self.loaded.pop(lora_id, None) is not None

    def pin(self, lora_id: int) -> bool:
        """Keep the loaded adapter `lora_id` loaded and in a slot until it is removed; refused,
        with nothing changed, when every slot holds a pinned adapter."""
        if lora_id not in self.loaded:
            raise ValueError(f"LoRA adapter {lora_id} is not loaded")
        _, displaced = self._admit(lora_id, NOTHING)
        self._hold(lora_id, displaced)
        self.pins.add(lora_id)
        return True

    def ids(self) -> set[int]:
        """The LoRA IDs of the loaded adapters."""
        return set(self.loaded)

    def activate(self, requests: Iterable[CachedRequestState]) -> dict[str, int]:
        """The slot of each of `requests` that names an adapter, once each of their adapters is
        loaded and holds a slot, as `add` puts it there."""
        named = {
            request.req_id: lora
            for request in requests
            if (lora := request.lora_request) is not None and lora.lora_int_id
        }
        adapters = {lora.lora_int_id: lora for lora in named.values()}
        if len(adapters) > self.config.max_loras:
            raise RuntimeError(f"a step runs {len(adapters)} adapters, above --max-loras")
        step = frozenset(adapters)
        for lora in adapters.values():
            _ = self.add(lora, step)
        return {req_id: self.held[lora.lora_int_id] for req_id, lora in named.items()}

    def _admit(self, lora_id: int, keep: Set[int]) -> tuple[int | None, int | None]:
        """The adapters that make room for adapter `lora_id`, each the least recently used one
        neither pinned nor in `keep`: one to unload when `lora_id` is not loaded and
        `max_cpu_loras` are, and one to leave its slot when `lora_id` holds none and no slot is
        free once the first is unloaded; refused when room is needed and there is no such one."""
        kept = self.pins | keep | {lora_id}
        unloaded = None
        capacity = self.config.max_cpu_loras or self.config.max_loras
        if lora_id not in self.loaded and len(self.loaded) >= capacity:
            unloaded = next((other for other in self.loaded if other not in kept), None)
            if unloaded is None:
                held = f"the {capacity} loaded adapters are pinned or in use"
                raise RuntimeError(f"LoRA adapter {lora_id}: {held}; raise --max-cpu-loras")
        holders = [other for other in self.held if other != unloaded]
        displaced = None
        if lora_id not in self.held and len(holders) >= self.config.max_loras:
            displaced = next((other for other in holders if other not in kept), None)
            if displaced is None:
                held = "pinned or running adapters hold every slot"
                raise RuntimeError(f"LoRA adapter {lora_id}: {held}; raise --max-loras")
        return unloaded, displaced

    def _hold(self, lora_id: int, displaced: int | None) -> None:
        """Put the loaded adapter `lora_id` in its slot, or in the slot `displaced` leaves, or in
        a free one, as `_admit` found room."""
        if displaced is not None:
            self.slots[self.held.pop(displaced)] = None
        slot = self.held.pop(lora_id, None)
        if slot is None:
            slot = min(set(range(self.config.max_loras)) - set(self.held.values()))
        self.held[lora_id] = slot
        self.slots[slot] = self.loaded[lora_id]

    def _read(self, request: LoRARequest) -> engine.decoder.Adapter:
        """The PEFT checkpoint of `request` onto the device, its `adapter_config.json` read once
        for vLLM's checks and the engine's."""
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
        return engine.decoder.Adapter(self.spec, engine.Checkpoint(path, config))
