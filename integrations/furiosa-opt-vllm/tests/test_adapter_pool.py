"""vLLM's own LRU caches decide which adapters stay loaded and which hold a device adapter: the
device adapters are allocated when the model loads, before the cache sizes, and a miss loads the
host revision built when vLLM added the adapter into one of them in place."""

from __future__ import annotations

import json
import operator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, cast, final

import pytest
import torch
from safetensors.torch import save_file
from vllm.config import LoRAConfig
from vllm.lora.request import LoRARequest

import furiosa_opt_engine as engine
from furiosa_opt_vllm.lora import Adapters

if TYPE_CHECKING:
    from collections.abc import Callable

    from vllm.v1.worker.gpu_input_batch import CachedRequestState

# The engine adapter type of a loaded decoder; the engine double below never reads it.
SPEC: Final = cast("engine.decoder.AdapterSpec", object())
# What a device adapter holds after a load into it failed partway, and a checkpoint the engine
# refuses to build.
TORN: Final = "torn"
BROKEN: Final = "broken"
FACTOR: Final = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"


@dataclass(frozen=True)
class Host:
    """A revision in host memory of the checkpoint directory `name`, which nothing changes."""

    name: str


@dataclass
class Device:
    """The engine the host tests stand in for: the revisions built on host, the device adapters
    allocated, and each load as the index of the device adapter and the revision it writes."""

    hosts: list[str] = field(default_factory=list)
    sources: list[tuple[engine.Checkpoint, str]] = field(default_factory=list)
    adapters: list[Adapter] = field(default_factory=list)
    loads: list[tuple[int, str]] = field(default_factory=list)
    # Whether the next load into a device adapter fails partway.
    failing: bool = False


@final
class Adapter:
    """A device adapter of the double: allocated once, its revision replaced by a load."""

    def __init__(self, device: Device) -> None:
        self.device: Device = device
        self.holds: str | None = None

    def load(self, host: Host) -> None:
        self.device.loads.append((self.device.adapters.index(self), host.name))
        if self.device.failing:
            self.holds = TORN
            raise RuntimeError("the device fails")
        self.holds = host.name


@pytest.fixture
def device(monkeypatch: pytest.MonkeyPatch) -> Device:
    device = Device()
    checkpoint = engine.Checkpoint

    def read(path: str | Path, config: dict[str, object] | None = None) -> engine.Checkpoint:
        device.sources.append((checkpoint(path, config), Path(path).name))
        return device.sources[-1][0]

    def host(spec: object, source: object) -> Host:
        assert spec is SPEC, "a revision is of its decoder's adapter type"
        name = next(name for opened, name in device.sources if opened is source)
        if name == BROKEN:
            raise ValueError("lora_A is of rank 32, not the configured 8")
        device.hosts.append(name)
        return Host(name)

    def adapter(spec: object) -> Adapter:
        assert spec is SPEC, "a device adapter is of its decoder's adapter type"
        device.adapters.append(Adapter(device))
        return device.adapters[-1]

    monkeypatch.setattr(engine, "Checkpoint", read)
    monkeypatch.setattr(engine.decoder, "HostAdapter", host)
    monkeypatch.setattr(engine.decoder, "Adapter", adapter)
    return device


@pytest.fixture
def request_of(tmp_path: Path) -> Callable[..., LoRARequest]:
    """vLLM's requests of PEFT adapter directories by name, each name under an ID of its own."""
    ids: dict[str, int] = {}

    def make(name: str) -> LoRARequest:
        directory = tmp_path / name
        if not directory.is_dir():
            directory.mkdir()
            config = {"r": 8, "lora_alpha": 16, "target_modules": ["q_proj"]}
            _ = (directory / "adapter_config.json").write_text(json.dumps(config))
            save_file({FACTOR: torch.zeros(8, 16)}, directory / "adapter_model.safetensors")
        return LoRARequest(name, ids.setdefault(name, len(ids) + 1), str(directory))

    return make


def manager(device: Device, slots: int = 2, loaded: int = 3) -> Adapters:
    """The adapters of a decoder under `slots` device adapters and `loaded` host revisions."""
    _ = device
    config = LoRAConfig(max_lora_rank=32, max_loras=slots, max_cpu_loras=loaded)
    return Adapters(config, SPEC, 2048)


def held(adapters: Adapters) -> list[str | None]:
    """The revision each slot of the calls' list holds; None for an empty slot."""
    listed = [cast("Adapter | None", cast("object", slot)) for slot in adapters.slots]
    return [None if slot is None else slot.holds for slot in listed]


def step(adapters: Adapters, *loras: LoRARequest) -> dict[str, int]:
    """The slots of a step whose requests each name one of `loras`, by its name."""
    states = [SimpleNamespace(req_id=lora.lora_name, lora_request=lora) for lora in loras]
    return adapters.activate(cast("list[CachedRequestState]", states))


def test_cycles_more_adapters_than_slots(
    device: Device, request_of: Callable[..., LoRARequest]
) -> None:
    """With more adapters than device adapters, a miss loads the host revision into the device
    adapter of the least recently used active adapter, and a hit loads nothing."""
    adapters = manager(device, loaded=4)
    a, b, c = (request_of(name) for name in "abc")
    for lora in [a, b, c]:
        assert adapters.add(lora)
    # vLLM activates every adapter it adds: c takes the device adapter of a.
    assert device.loads == [(0, "a"), (1, "b"), (0, "c")]
    assert step(adapters, a) == {"a": 1}
    assert step(adapters, c, a) == {"c": 0, "a": 1}
    assert step(adapters, b) == {"b": 0}
    assert device.loads[3:] == [(1, "a"), (0, "b")]
    assert held(adapters) == ["b", "a"]
    assert device.hosts == ["a", "b", "c"]
    assert all(map(operator.is_, adapters.slots, device.adapters))


def test_builds_a_revision_once_per_add(
    device: Device, request_of: Callable[..., LoRARequest]
) -> None:
    """An add builds its adapter's host revision once, and an add of a loaded adapter and its
    steps build and load none. A reload in place that the engine refuses keeps the old revision
    in its slot; one it accepts loads into the slot the old revision frees."""
    adapters = manager(device)
    a = request_of("a")
    assert adapters.add(a)
    _ = step(adapters, a)
    assert adapters.add(a)
    assert (device.hosts, device.loads) == (["a"], [(0, "a")])
    broken = LoRARequest("a", a.lora_int_id, request_of(BROKEN).lora_path, load_inplace=True)
    with pytest.raises(ValueError, match="not the configured 8"):
        _ = adapters.add(broken)
    assert (held(adapters), device.loads) == (["a", None], [(0, "a")])
    reload = LoRARequest("a", a.lora_int_id, request_of("a2").lora_path, load_inplace=True)
    assert adapters.add(reload)
    assert (device.hosts, device.loads) == (["a", "a2"], [(0, "a"), (0, "a2")])
    assert held(adapters) == ["a2", None]


def test_publishes_only_a_load_that_returned(
    device: Device, request_of: Callable[..., LoRARequest]
) -> None:
    """A load into a device adapter that fails raises out of the step, and no slot lists that
    device adapter until a later load into it returns."""
    adapters = manager(device)
    a, b, c = (request_of(name) for name in "abc")
    assert adapters.add(a) and adapters.add(b)
    device.failing = True
    with pytest.raises(RuntimeError, match="the device fails"):
        _ = step(adapters, c)
    assert held(adapters) == [None, "b"]
    device.failing = False
    assert step(adapters, c) == {"c": 0}
    assert held(adapters) == ["c", "b"]
    assert device.hosts == ["a", "b", "c"]


def test_unloads_the_least_recently_used_revision(
    device: Device, request_of: Callable[..., LoRARequest]
) -> None:
    """An add past `max_cpu_loras` unloads the least recently used revision, which frees its slot
    for the new one, so no other active adapter leaves its slot."""
    adapters = manager(device, loaded=2)
    a, b, c = (request_of(name) for name in "abc")
    for lora in [a, b, c]:
        assert adapters.add(lora)
    assert adapters.ids() == {b.lora_int_id, c.lora_int_id}
    assert held(adapters) == ["c", "b"]


def test_keeps_pinned_adapters_in_their_slots(
    device: Device, request_of: Callable[..., LoRARequest]
) -> None:
    """Pinning an inactive adapter loads it into a slot, a miss evicts an unpinned adapter around
    it, and with every slot pinned an add or a pin that needs a slot is refused before anything
    is built or unloaded; pinning an adapter that is not loaded is refused."""
    adapters = manager(device)
    spare, a, b, c = (request_of(name) for name in ["spare", "a", "b", "c"])
    for lora in [spare, a, b]:
        assert adapters.add(lora)
    assert adapters.pin(spare.lora_int_id)
    assert step(adapters, a) == {"a": 0}
    assert held(adapters) == ["a", "spare"]
    assert adapters.pin(a.lora_int_id)
    with pytest.raises(RuntimeError, match="hold every slot"):
        _ = adapters.add(c)
    with pytest.raises(RuntimeError, match="hold every slot"):
        _ = adapters.pin(b.lora_int_id)
    with pytest.raises(ValueError, match="not loaded"):
        _ = adapters.pin(c.lora_int_id)
    assert adapters.ids() == {spare.lora_int_id, a.lora_int_id, b.lora_int_id}
    assert device.hosts == ["spare", "a", "b"]
    assert adapters.remove(spare.lora_int_id)
    assert adapters.add(c) and adapters.pin(c.lora_int_id)
    assert held(adapters) == ["a", "c"]


def test_bounds_a_step_as_vllm_does(device: Device, request_of: Callable[..., LoRARequest]) -> None:
    """A step of more adapters than slots is refused with vLLM's reason; a step whose second
    adapter evicts its first from the only unpinned slot fails before any call, as in vLLM."""
    adapters = manager(device)
    pinned, a, b = (request_of(name) for name in ["pinned", "a", "b"])
    with pytest.raises(RuntimeError, match="greater than the number of GPU LoRA slots"):
        _ = step(adapters, pinned, a, b)
    assert adapters.add(pinned) and adapters.pin(pinned.lora_int_id)
    with pytest.raises(ValueError, match="is not in list"):
        _ = step(adapters, a, b)
