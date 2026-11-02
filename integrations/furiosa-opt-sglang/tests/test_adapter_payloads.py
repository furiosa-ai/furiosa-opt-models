"""The configuration, added tokens and factors a load brings arrive from outside the server: a
malformed one, or one the engine refuses, is a refused load that builds nothing, and well-formed
tensors load as a directory's."""

import json
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
import torch
from test_adapters import CONFIG, FACTOR, Device, Host, peft, ref
from test_adapters import adapter as adapter
from test_adapters import device as device
from test_adapters import manager as manager

import furiosa_opt_engine as engine
from furiosa_opt_sglang.lora import Adapters


def test_loads_tensors(manager: Callable[..., Adapters], device: Device) -> None:
    """`/load_lora_adapter_from_tensors` loads its factors and configuration as a directory's,
    under SGLang's tensor path."""
    adapters = manager()
    tensors = {FACTOR: torch.zeros(8, 16)}
    result = adapters.load_lora_adapter_from_tensors(ref("sky", "__tensor__"), tensors, CONFIG)
    assert (result.success, result.loaded_adapters) == (True, {"sky": "__tensor__"})
    assert len(device.hosts) == 1


@pytest.mark.parametrize(
    ("tensors", "config", "setting", "reason"),
    [
        ({FACTOR: torch.zeros(8, 16)}, CONFIG | {"r": 16}, {"max_lora_rank": 8}, "rank 16"),
        (
            {FACTOR: torch.zeros(8, 16)},
            {"r": 8, "lora_alpha": 16},
            {},
            "configuration's target_modules: Field required",
        ),
        ({FACTOR: [0.0]}, CONFIG, {}, f"'{FACTOR}' names no tensor"),
    ],
    ids=["rank", "config", "value"],
)
def test_refuses_tensors_before_building(
    manager: Callable[..., Adapters],
    device: Device,
    tensors: dict[str, object],
    config: dict[str, object],
    setting: dict[str, object],
    reason: str,
) -> None:
    """Tensors and configurations are refused as a directory's, before anything is built."""
    adapters = manager(**setting)
    result = adapters.load_lora_adapter_from_tensors(ref("sky", "__tensor__"), tensors, config)
    assert not result.success and reason in str(result.error_message), result.error_message
    assert device.hosts == []


@pytest.mark.parametrize(
    ("changes", "setting", "tokens"),
    [
        ({"r": "8"}, {"max_lora_rank": 8}, None),
        ({"target_modules": 5}, {}, None),
        ({"target_modules": [1]}, {"lora_target_modules": {"q_proj"}}, None),
        ({}, {}, {"<t>": "9"}),
    ],
    ids=["rank-string", "modules-scalar", "module-number", "token-string"],
)
def test_refuses_malformed_directories(
    manager: Callable[..., Adapters],
    device: Device,
    tmp_path: Path,
    changes: dict[str, object],
    setting: dict[str, object],
    tokens: dict[str, object] | None,
) -> None:
    """A malformed `adapter_config.json` or `added_tokens.json` is a refused load, which leaves the
    loaded adapters as they were and builds nothing, as SGLang's manager answers any load."""
    adapters = manager(**setting)
    other = peft(tmp_path / "other", **changes)
    if tokens is not None:
        _ = (other / "added_tokens.json").write_text(json.dumps(tokens))
    result = adapters.load_lora_adapter(ref("other", other))
    assert (result.success, result.loaded_adapters) == (False, {})
    assert "LoRA adapter other:" in str(result.error_message)
    assert device.hosts == []


@pytest.mark.parametrize(
    "error",
    [NotImplementedError("rank 64 is past 32"), ValueError("lora_B of q_proj has another shape")],
    ids=["unsupported", "malformed"],
)
def test_answers_engine_refusals(
    manager: Callable[..., Adapters],
    monkeypatch: pytest.MonkeyPatch,
    adapter: Path,
    error: Exception,
) -> None:
    """An adapter the engine refuses is answered as a failed load, with the engine's reason."""

    def refuse(spec: object, checkpoint: object) -> None:
        _ = spec, checkpoint
        raise error

    adapters = manager()
    monkeypatch.setattr(engine.decoder, "HostAdapter", refuse)
    result = adapters.load_lora_adapter(ref("sky", adapter))
    assert (result.success, result.loaded_adapters) == (False, {})
    assert str(error) in str(result.error_message)


def test_raises_engine_failures(
    manager: Callable[..., Adapters], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An engine failure that refuses no adapter raises out of the scheduler and loads nothing; a
    later load succeeds."""
    adapters = manager()
    build = cast("Callable[[object, object], Host]", engine.decoder.HostAdapter)

    def fail(spec: object, checkpoint: object) -> None:
        _ = spec, checkpoint
        raise RuntimeError("the engine fails")

    monkeypatch.setattr(engine.decoder, "HostAdapter", fail)
    with pytest.raises(RuntimeError, match="the engine fails"):
        _ = adapters.load_lora_adapter(ref("sky", peft(tmp_path / "sky")))
    monkeypatch.setattr(engine.decoder, "HostAdapter", build)
    other = peft(tmp_path / "other")
    assert adapters.load_lora_adapter(ref("other", other)).loaded_adapters == {"other": str(other)}


def quantized() -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.quantize_per_tensor(torch.zeros(8, 16), 0.1, 0, torch.qint8)


def nested() -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.nested.nested_tensor([torch.zeros(8), torch.zeros(16)])


@pytest.mark.parametrize(
    ("tensors", "config", "tokens"),
    [
        (lambda: {FACTOR: torch.zeros(8, 16)}, [CONFIG], None),
        (lambda: {1: torch.zeros(8, 16)}, CONFIG, None),
        (lambda: {FACTOR: torch.zeros(8, 16)}, CONFIG | {"r": "8"}, None),
        (lambda: {FACTOR: torch.zeros(8, 16)}, CONFIG, {"<t>": "9"}),
        (lambda: {FACTOR: quantized()}, CONFIG, None),
        (lambda: {FACTOR: nested()}, CONFIG, None),
        (lambda: [torch.zeros(8, 16)], CONFIG, None),
    ],
    ids=["config-list", "key-number", "rank-string", "token-string", "qint8", "nested", "no-dict"],
)
def test_refuses_malformed_tensor_loads(
    manager: Callable[..., Adapters],
    device: Device,
    tensors: Callable[[], object],
    config: object,
    tokens: object,
) -> None:
    """The factors, configuration and added tokens of `/load_lora_adapter_from_tensors` arrive from
    the request; malformed ones are a refused load that builds nothing."""
    adapters = manager(max_lora_rank=8)
    result = adapters.load_lora_adapter_from_tensors(
        ref("sky", "__tensor__"),
        cast("dict[str, torch.Tensor]", tensors()),
        cast("dict[str, object]", config),
        cast("dict[str, int] | None", tokens),
    )
    assert (result.success, result.loaded_adapters) == (False, {})
    assert "LoRA adapter sky:" in str(result.error_message)
    assert device.hosts == []
