"""SGLang's own pool decides which adapter revisions the device holds: its entries are device
adapters taken once, before the cache sizes; a load builds a host revision only, and a batch's
bind writes each revision its entries miss in place, which its calls then read."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.lora.mem_pool import LoRAMemoryPool
from test_adapters import CONFIG, FACTOR, HEADROOM, SIZE, Device, held, peft
from test_adapters import device as device
from test_adapters import manager as manager

from furiosa_opt_sglang.lora import Adapters

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def pool(adapters: Adapters) -> LoRAMemoryPool:
    """SGLang's pool, which the manager holds as SGLang's own manager does."""
    assert isinstance(adapters.memory_pool, LoRAMemoryPool), "the manager holds SGLang's pool"
    return adapters.memory_pool


def load(adapters: Adapters, directory: Path, *, pinned: bool = False) -> LoRARef:
    """The record of a successful load of a fresh PEFT adapter at `directory`."""
    loaded = LoRARef(lora_name=directory.name, lora_path=str(peft(directory)), pinned=pinned)
    result = adapters.load_lora_adapter(loaded)
    assert result.success, result.error_message
    return loaded


def test_takes_its_entries_before_any_load(
    manager: Callable[..., Adapters], device: Device
) -> None:
    """The manager takes `--max-loras-per-batch` device adapters of no updates at once, also more
    than one engine call takes, and SGLang's pool of that many entries holds them."""
    adapters = manager(max_loras_per_batch=8)
    assert [device.name(adapter.revision) for adapter in device.adapters] == [None] * 8
    assert pool(adapters).max_loras_per_batch == 8


def test_loads_after_startup_under_default_sizing(
    manager: Callable[..., Adapters], device: Device, tmp_path: Path
) -> None:
    """Once the cache took every byte past the calls' headroom, loads still succeed: they build
    host revisions and take no device memory, more of them than the pool has entries."""
    adapters = manager(max_loras_per_batch=3)
    taken = len(device.adapters)
    device.available = HEADROOM + SIZE // 2
    for index in range(5):
        _ = load(adapters, tmp_path / f"adapter{index}")
    assert (len(device.adapters), device.writes) == (taken, [])


def test_writes_a_revision_on_a_miss_only(
    manager: Callable[..., Adapters], device: Device, tmp_path: Path
) -> None:
    """A bind writes each adapter its entries miss once, and none of a resident one."""
    adapters = manager(max_loras_per_batch=3)
    a, b = load(adapters, tmp_path / "a"), load(adapters, tmp_path / "b")
    first = adapters.bind([a.lora_id])
    assert len(device.writes) == 1
    assert held(device, adapters.bind([a.lora_id]).adapters) == held(device, first.adapters)
    assert held(device, adapters.bind([None, b.lora_id, a.lora_id]).adapters) == ["b", "a"]
    assert len(device.writes) == 2


def test_cycles_more_adapters_than_entries(
    manager: Callable[..., Adapters], device: Device, tmp_path: Path
) -> None:
    """Five adapters with the base model cycle through three entries, each bind's calls reading
    exactly the revisions their requests name."""
    adapters = manager(max_loras_per_batch=3)
    refs = [load(adapters, tmp_path / f"adapter{index}") for index in range(5)]
    assert len(device.adapters) == 3
    for step in range(10):
        pair = [refs[step % 5], refs[(step + 2) % 5]]
        binding = adapters.bind([None, *(each.lora_id for each in pair)])
        assert held(device, binding.adapters) == [each.lora_name for each in pair]
    assert len(device.adapters) == 3


def test_lists_the_entries_a_call_reads(
    manager: Callable[..., Adapters], device: Device, tmp_path: Path
) -> None:
    """A call's list holds the pool's entry of each adapter its requests name, in first-named
    order; the base model's entry and empty entries are never listed."""
    adapters = manager(max_loras_per_batch=8)
    a, b = load(adapters, tmp_path / "a"), load(adapters, tmp_path / "b")
    _ = adapters.bind([None, a.lora_id])
    binding = adapters.bind([None, b.lora_id, a.lora_id, b.lora_id])
    entries = pool(adapters).uid_to_buffer_id
    listed = (device.adapters[entries[b.lora_id]], device.adapters[entries[a.lora_id]])
    assert binding.adapters == listed
    assert dict(binding.slots) == {b.lora_id: 0, a.lora_id: 1}
    assert adapters.bind([None]).adapters == ()


def test_pins_leave_one_entry_to_the_rest(
    manager: Callable[..., Adapters], device: Device, tmp_path: Path
) -> None:
    """Of three entries two may be pinned, as SGLang allows; the base model and an unpinned
    adapter then share the third, and the pinned revisions are never written again."""
    adapters = manager(max_loras_per_batch=3)
    a, b = load(adapters, tmp_path / "a", pinned=True), load(adapters, tmp_path / "b", pinned=True)
    third = LoRARef(lora_name="p", lora_path=str(peft(tmp_path / "p")), pinned=True)
    assert not adapters.load_lora_adapter(third).success
    c = load(adapters, tmp_path / "c")
    batches = {(None, c.lora_id): False, (c.lora_id,): True, (a.lora_id, c.lora_id): True}
    batches |= {(a.lora_id, b.lora_id, c.lora_id): False, (None, a.lora_id): True}
    assert {ids: adapters.validate_lora_batch(set(ids)) for ids in batches} == batches
    pinned = adapters.bind([a.lora_id, b.lora_id]).adapters
    for ids in ([c.lora_id], [None], [c.lora_id], [a.lora_id, c.lora_id]):
        _ = adapters.bind(ids)
    assert held(device, pinned) == ["a", "b"]
    assert not any(adapter in pinned for adapter, _ in device.writes[2:])
    assert held(device, adapters.bind([c.lora_id]).adapters) == ["c"]


def test_publishes_only_a_whole_write(
    manager: Callable[..., Adapters], device: Device, tmp_path: Path
) -> None:
    """A load into an entry that fails raises out of the bind and leaves the adapter unpublished;
    the next bind writes it whole, and no call ever lists the torn entry as holding it."""
    adapters = manager(max_loras_per_batch=2)
    a, b, c = (load(adapters, tmp_path / name) for name in "abc")
    _ = adapters.bind([a.lora_id, b.lora_id])
    device.failing = True
    with pytest.raises(RuntimeError, match="the device fails"):
        _ = adapters.bind([c.lora_id])
    assert c.lora_id not in pool(adapters).uid_to_buffer_id
    device.failing = False
    assert held(device, adapters.bind([c.lora_id]).adapters) == ["c"]


def test_bind_alone_changes_residency(
    manager: Callable[..., Adapters], device: Device, tmp_path: Path
) -> None:
    """Loads, unloads and SGLang's batch hooks change no device adapter, so a bound step's calls
    read its revisions until the next bind; an unloaded adapter's entry is the next miss's."""
    adapters = manager(max_loras_per_batch=2)
    a = load(adapters, tmp_path / "a")
    bound = adapters.bind([a.lora_id])
    b, c = load(adapters, tmp_path / "b"), load(adapters, tmp_path / "c")
    adapters.fetch_new_loras({b.lora_id, c.lora_id})
    adapters.prepare_lora_batch(cast("ForwardBatch", cast("object", None)))
    assert adapters.unload_lora_adapter(a).success
    assert (held(device, bound.adapters), len(device.writes)) == (["a"], 1)
    entry = bound.adapters[0]
    assert entry in adapters.bind([b.lora_id, c.lora_id]).adapters
    with pytest.raises(KeyError):
        _ = adapters.bind([a.lora_id])


@pytest.mark.parametrize("malformed", ["config", "weights", "tensors"])
def test_refused_loads_keep_residency(
    manager: Callable[..., Adapters], device: Device, tmp_path: Path, malformed: str
) -> None:
    """A malformed load is refused before any device effect: resident revisions stay and serve."""
    adapters = manager(max_loras_per_batch=3)
    a = load(adapters, tmp_path / "a")
    _ = adapters.bind([a.lora_id])
    taken = len(device.adapters)
    if malformed == "tensors":
        other = LoRARef(lora_name="other", lora_path="__tensor__")
        result = adapters.load_lora_adapter_from_tensors(other, {FACTOR: [0.0]}, CONFIG)
    else:
        directory = peft(tmp_path / "other", r=None if malformed == "config" else 8)
        if malformed == "weights":
            (directory / "adapter_model.safetensors").unlink()
        result = adapters.load_lora_adapter(LoRARef(lora_name="other", lora_path=str(directory)))
    assert not result.success
    assert (len(device.adapters), len(device.writes)) == (taken, 1)
    assert (held(device, adapters.bind([a.lora_id]).adapters), len(device.writes)) == (["a"], 1)


def test_reloads_as_a_new_revision(
    manager: Callable[..., Adapters], device: Device, tmp_path: Path
) -> None:
    """A name loaded again after its unload is another host revision, which its next bind writes
    though the old one may still sit in an entry."""
    adapters = manager(max_loras_per_batch=3)
    first = load(adapters, tmp_path / "sky")
    _ = adapters.bind([first.lora_id])
    _ = adapters.unload_lora_adapter(first)
    again = LoRARef(lora_name="sky", lora_path=str(tmp_path / "sky"))
    assert adapters.load_lora_adapter(again).success
    _ = adapters.bind([again.lora_id])
    assert len(device.writes) == 2
    [(_, old), (entry, new)] = device.writes
    assert new is not old and entry.revision is new
