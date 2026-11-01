"""A load the server refuses changes nothing it serves: requests of the base model and of loaded
adapters generate the tokens they did before."""

import base64
import json
import pickle
import shutil
from collections.abc import Callable
from http.client import HTTPResponse
from pathlib import Path

import pytest
import torch
from test_lora_lifecycle import CONFIG, generate, refused
from test_lora_lifecycle import adapters as adapters
from test_lora_lifecycle import served as served
from test_server import chips as chips
from test_server import plugin as plugin

pytestmark = pytest.mark.feature("LoRA")


def payload(factors: object) -> str:
    """`factors` as a `/load_lora_adapter_from_tensors` request carries them."""
    return base64.b64encode(pickle.dumps(factors)).decode()


def test_refuses_malformed_loads_and_keeps_serving(
    served: Callable[..., HTTPResponse], adapters: dict[str, Path], tmp_path: Path
) -> None:
    """Loads of malformed configurations or factors answer 400 with the loaded adapters as they
    were, and the server's answers stay the same."""
    from safetensors.torch import load_file

    names: list[str | None] = [None, "full"]
    before = [answer["output_ids"] for answer in generate(served, names)]
    source = adapters["partial"]
    config = CONFIG.validate_json((source / "adapter_config.json").read_bytes())
    factors = load_file(source / "adapter_model.safetensors")
    scalar = Path(shutil.copytree(source, tmp_path / "scalar"))
    _ = (scalar / "adapter_config.json").write_text(json.dumps(config | {"target_modules": 5}))
    loads: list[tuple[str, dict[str, object]]] = [
        ("/load_lora_adapter", {"lora_path": str(scalar)}),
        ("/load_lora_adapter_from_tensors", {"config_dict": config | {"r": "4"}}),
        (
            "/load_lora_adapter_from_tensors",
            {"serialized_named_tensors": [payload({1: torch.zeros(1)})]},
        ),
    ]
    tensors = {"config_dict": config, "serialized_named_tensors": [payload(factors)]}
    for path, body in loads:
        sent = (tensors if "tensors" in path else {}) | body | {"lora_name": "bad"}
        answer = refused(served, path, sent)
        assert "LoRA adapter bad:" in str(answer["error_message"]), answer
        assert "bad" not in answer["loaded_adapters"]
    assert [answer["output_ids"] for answer in generate(served, names)] == before
