"""The device adapters of vLLM's adapter slots live as long as the loaded model: the worker's
`load_model`, which vLLM runs before it sizes the cache, allocates them, and a reload of the
weights unloads every adapter, as vLLM's own workers do, and keeps them."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from test_adapter_pool import SPEC, Device, held, manager, step
from test_adapter_pool import device as device
from test_adapter_pool import request_of as request_of
from vllm.engine.arg_utils import EngineArgs
from vllm.lora.request import LoRARequest

import furiosa_opt_engine as engine
from furiosa_opt_vllm import residency, selection
from furiosa_opt_vllm.decoder import Decoder
from furiosa_opt_vllm.worker import FuriosaWorker

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from furiosa_opt_vllm.checkpoint import Location
    from furiosa_opt_vllm.runner import Runner


@pytest.mark.usefixtures("platform_env")
def test_allocates_every_device_adapter_when_the_model_loads(
    served: Path,
    device: Device,
    request_of: Callable[..., LoRARequest],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker's `load_model`, which vLLM runs before it sizes the cache, allocates the
    `max_loras` device adapters, and adapters added and activated later allocate none."""
    model = SimpleNamespace(
        output=engine.decoder.Output.TOKENS, cache=SimpleNamespace(pools=()), adapters=SPEC
    )

    def loaded(config: VllmConfig) -> SimpleNamespace:
        _ = config
        return SimpleNamespace(model=model)

    def choose(*arguments: object) -> None:
        _ = arguments

    monkeypatch.setattr(Decoder, "load", loaded)
    monkeypatch.setattr(selection, "choose", choose)
    config = EngineArgs(
        model=str(served),
        dtype="bfloat16",
        skip_tokenizer_init=True,
        enable_lora=True,
        max_loras=2,
        max_cpu_loras=4,
    ).create_engine_config()
    worker = FuriosaWorker(config, local_rank=0, rank=0, distributed_init_method="")
    worker.init_device()
    worker.load_model()
    assert len(device.adapters) == 2
    for name in ["a", "b", "c", "d"]:
        assert worker.add_lora(request_of(name))
    runner = worker.runner
    assert runner is not None and runner.adapters is not None
    _ = step(runner.adapters, request_of("a"))
    assert len(device.adapters) == 2


def test_reload_unloads_every_adapter(
    device: Device, request_of: Callable[..., LoRARequest], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`reload_weights` ends with no adapter loaded, as vLLM's own workers end it, and keeps the
    device adapters for the next loads."""
    adapters = manager(device)
    a = request_of("a")
    assert adapters.add(a) and adapters.pin(a.lora_int_id)

    def loaded(self: Decoder, checkpoint: object) -> Decoder:
        _ = checkpoint
        return self

    monkeypatch.setattr(Decoder, "loaded", loaded)
    # The decoder fields a reload reads: its checkpoint's files and the device it waits for.
    checkpoint = cast("Location", cast("object", SimpleNamespace(mapped=lambda: None)))
    chips = cast("engine.Device", cast("object", SimpleNamespace(synchronize=lambda: None)))
    options = cast("engine.LoadOptions", cast("object", None))
    runner = SimpleNamespace(
        asleep={},
        model=Decoder(None, None, checkpoint, options, chips, 1),
        adapters=adapters,
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(enable_sleep_mode=False)),
    )
    residency.reload(cast("Runner", cast("object", runner)))
    assert (adapters.ids(), held(adapters)) == (set(), [None, None])
    assert len(device.adapters) == 2
