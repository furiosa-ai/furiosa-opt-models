"""A rotary scaling serves past the trained length, or is refused before any weight is read."""

import json
from pathlib import Path
from typing import Final, Literal, NotRequired

import pytest
import torch
from huggingface_hub import snapshot_download
from pydantic import JsonValue, TypeAdapter
from test_server import Chips
from test_server import chips as chips
from test_server import plugin as plugin
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing_extensions import TypedDict

import furiosa_opt_engine as engine

CHECKPOINT: Final = "Qwen/Qwen3-0.6B"
"""Declares 40960 positions; its decoder is compiled for 131072."""
THETA: Final = 1000000.0


class Rope(TypedDict):
    """A rotary scaling, as `rope_parameters` or `rope_scaling` spell it."""

    rope_type: Literal["yarn", "linear", "dynamic"]
    factor: float
    original_max_position_embeddings: NotRequired[int]
    rope_theta: NotRequired[float]


class Scaling(TypedDict, total=False):
    """The config fields a case sets."""

    rope_parameters: Rope
    rope_scaling: Rope
    max_position_embeddings: int


class Generated(TypedDict):
    output_ids: list[int]


FIELDS: Final = TypeAdapter(dict[str, JsonValue])
TOKENS: Final = TypeAdapter(list[int])
GENERATED: Final = TypeAdapter(Generated)


@pytest.mark.feature("Context extension past the trained length")
def test_refuses_extension(tmp_path: Path) -> None:
    """The engine refuses a rotary scaling it does not compute before reading a weight."""
    config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B").to_dict()
    config["rope_parameters"] = {"rope_type": "dynamic", "factor": 2.0, "rope_theta": 1000000.0}
    with pytest.raises(NotImplementedError, match=r'rope_type "dynamic"'):
        engine.LoadOptions().parse(engine.Checkpoint(tmp_path, config=config))


@pytest.mark.feature("Context extension past the trained length")
def test_refuses_extension_at_startup(
    tmp_path_factory: pytest.TempPathFactory, chips: Chips
) -> None:
    """The server stops at startup with the engine's reason for a rope type it does not compute."""
    dynamic = Scaling(rope_parameters=Rope(rope_type="dynamic", factor=2.0, rope_theta=THETA))
    flags = ["--json-model-override-args", json.dumps(dynamic)]
    log = chips.refusal(tmp_path_factory, {"id": CHECKPOINT, "task": "text", "flags": flags})
    refusals = [line for line in log.splitlines() if "furiosa-opt-sglang " in line]
    assert any('rope_type "dynamic"' in line for line in refusals), refusals


@pytest.mark.feature("Context extension past the trained length")
@pytest.mark.parametrize(
    ("scaling", "source"),
    [
        # Qwen's YaRN recipe: four times the 32768 positions Qwen3 was trained on.
        pytest.param(
            Scaling(
                rope_parameters=Rope(
                    rope_type="yarn",
                    factor=4.0,
                    original_max_position_embeddings=32768,
                    rope_theta=THETA,
                ),
                max_position_embeddings=131072,
            ),
            "override",
            id="yarn",
        ),
        pytest.param(
            Scaling(rope_parameters=Rope(rope_type="linear", factor=3.2, rope_theta=THETA)),
            "override",
            id="linear",
        ),
        # The same YaRN in the checkpoint's config, as Qwen's model card adds it.
        pytest.param(
            Scaling(
                rope_scaling=Rope(
                    rope_type="yarn", factor=4.0, original_max_position_embeddings=32768
                )
            ),
            "config",
            id="declared",
        ),
    ],
)
def test_extended_context_matches_transformers(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    chips: Chips,
    scaling: Scaling,
    source: Literal["override", "config"],
) -> None:
    """Greedy tokens past the declared positions are within 5% of Transformers' top probability."""
    fields = FIELDS.validate_json(json.dumps(scaling))
    flags = ["--context-length", "131072", "--chunked-prefill-size", "2048"]
    if source == "override":
        served, overrides = CHECKPOINT, fields
        flags += ["--json-model-override-args", json.dumps(scaling)]
    else:
        snapshot = Path(snapshot_download(CHECKPOINT))
        copy = tmp_path_factory.mktemp("declared")
        for file in snapshot.iterdir():
            if file.name != "config.json":
                (copy / file.name).symlink_to(file)
        config = FIELDS.validate_json((snapshot / "config.json").read_bytes()) | fields
        _ = (copy / "config.json").write_text(json.dumps(config))
        served, overrides = str(copy), {}
        # SGLang takes the 40960 positions the config declares; the model card serves 131072.
        monkeypatch.setenv("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")
    fact = "The vault's password is 7381, and the gardener's name is Ottilie."
    lines = " ".join(
        f"Line {line}: the river ran slowly past mill {line % 97}." for line in range(3200)
    )
    question = "Recall the opening line. The gardener's name is"
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    ids = TOKENS.validate_python(tokenizer(f"{fact} {lines} {question}").input_ids)
    assert 40960 < len(ids) < 3 * 40960
    greedy = {"temperature": 0, "max_new_tokens": 4, "ignore_eos": True}
    row = {"id": served, "task": "text", "flags": flags}
    # The answer waits for the prefill of the whole prompt.
    with (
        chips.serve(tmp_path_factory, row) as send,
        send("/generate", {"input_ids": ids, "sampling_params": greedy}, 600) as response,
    ):
        generated = GENERATED.validate_json(response.read())["output_ids"]
    assert len(generated) == 4
    config = AutoConfig.from_pretrained(served, **overrides)
    reference = AutoModelForCausalLM.from_pretrained(served, config=config, dtype=torch.bfloat16)
    tokens, cache = torch.tensor([ids]), None
    with torch.inference_mode():
        # Each token extends the cache, as Transformers generates.
        for token in generated:
            step = reference(tokens, past_key_values=cache, use_cache=True, logits_to_keep=1)
            expected, cache = step.logits[0, -1].float().log_softmax(0), step.past_key_values
            # BF16 rounding can tie the two highest scores.
            assert (expected[token] - expected.max()).exp() >= 0.95
            tokens = torch.tensor([[token]])
