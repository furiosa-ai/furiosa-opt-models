"""A step's calls take the adapters their requests name, in slots the engine call reads, within the
engine's slots a batch; a step reads what its entries held when it was bound."""

from collections.abc import Callable
from pathlib import Path

import pytest
from test_adapters import Device, held, peft, ref
from test_adapters import adapter as adapter
from test_adapters import device as device
from test_adapters import manager as manager

from furiosa_opt_sglang.generation import Sampled
from furiosa_opt_sglang.lora import Adapters


def test_issued_step_keeps_unloaded_adapters(
    manager: Callable[..., Adapters], device: Device, adapter: Path
) -> None:
    """An adapter unloaded while a step that reads it runs stays in the entry the step reads, until
    a later bind loads another adapter into it."""
    adapters = manager()
    sky = ref("sky", adapter)
    _ = adapters.load_lora_adapter(sky)
    step = Sampled((), 1, adapters.bind([sky.lora_id]))
    assert adapters.unload_lora_adapter(sky).loaded_adapters == {}
    assert held(device, step.adapters.adapters) == ["sky"]


@pytest.mark.parametrize(
    ("bound", "ids", "admitted"),
    [
        (3, {None, "a", "b"}, True),
        (3, {"a", "b", "c"}, False),
        (2, {None, "a", "b"}, False),
    ],
    ids=["base-and-slots", "past-slots", "past-bound"],
)
def test_bounds_a_batch(
    manager: Callable[..., Adapters], bound: int, ids: set[str | None], admitted: bool
) -> None:
    """A batch runs at most the engine's slots of adapters, and the base model counts against
    `--max-loras-per-batch` as SGLang counts it."""
    adapters = manager(max_loras_per_batch=bound)
    assert adapters.validate_lora_batch(ids) is admitted


def test_binds_adapters_in_first_named_order(
    manager: Callable[..., Adapters], device: Device, tmp_path: Path
) -> None:
    """A step's calls take each named adapter once, in the order its requests first name them;
    the base model takes no slot."""
    adapters = manager()
    a, b = ref("a", peft(tmp_path / "a")), ref("b", peft(tmp_path / "b"))
    _ = adapters.load_lora_adapter(a), adapters.load_lora_adapter(b)
    binding = adapters.bind([None, b.lora_id, a.lora_id, b.lora_id])
    assert held(device, binding.adapters) == ["b", "a"]
    assert dict(binding.slots) == {b.lora_id: 0, a.lora_id: 1}
    assert adapters.bind([None]).adapters == ()


def test_finds_no_slot_for_an_unbound_id(manager: Callable[..., Adapters], adapter: Path) -> None:
    """A request's ID that its step did not bind is an error, never the base model; the base
    model, `None`, takes no slot."""
    adapters = manager()
    sky = ref("sky", adapter)
    _ = adapters.load_lora_adapter(sky)
    binding = adapters.bind([sky.lora_id])
    assert (binding.slot(sky.lora_id), binding.slot(None)) == (0, None)
    with pytest.raises(KeyError):
        _ = binding.slot("unbound")
