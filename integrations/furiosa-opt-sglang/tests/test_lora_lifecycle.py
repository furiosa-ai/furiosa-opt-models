"""SGLang serves LoRA adapters on the engine: each request runs with its own adapter, adapters and
the base model share a batch, and SGLang's adapter endpoints load and unload them."""

import base64
import json
import pickle
import shutil
from collections.abc import Callable, Generator
from contextlib import ExitStack
from functools import cache
from http.client import HTTPResponse
from pathlib import Path
from typing import TYPE_CHECKING, Final, NotRequired
from urllib.error import HTTPError

import pytest
import torch
from pydantic import JsonValue, TypeAdapter
from test_server import Chips
from test_server import chips as chips
from test_server import plugin as plugin
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from collections.abc import Mapping

    from peft import LoraConfig

MODEL: Final = "Qwen/Qwen3-0.6B"
# 33 tokens: two whole cache pages, which SGLang's radix cache keeps for a later request.
PROMPT: Final = (
    "Explain briefly why the sky is blue during the day and why sunsets turn red and orange in"
    " the evening, as a teacher would explain it to a child."
)
TOKENS: Final = 8
ATTENTION: Final = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP: Final = ["gate_proj", "up_proj", "down_proj"]
LORA: Final = ["--enable-lora", "--max-running-requests", "4"]


class Meta(TypedDict):
    cached_tokens: int
    spec_accept_length: NotRequired[float]


class Answer(TypedDict):
    """SGLang's answer to one prompt of a `/generate` request."""

    output_ids: list[int]
    text: str
    meta_info: Meta


class Update(TypedDict):
    """SGLang's answer to a load or an unload of an adapter: every loaded adapter's path by name."""

    error_message: str | None
    loaded_adapters: dict[str, str]


class Completion(TypedDict):
    text: str


class Completions(TypedDict):
    choices: list[Completion]


ANSWERS: Final = TypeAdapter(list[Answer])
UPDATE: Final = TypeAdapter(Update)
COMPLETIONS: Final = TypeAdapter(Completions)
CONFIG: Final = TypeAdapter(dict[str, JsonValue])


def save(
    load: Callable[[], torch.nn.Module],
    forms: "Mapping[str, LoraConfig]",
    directory: Path,
    *,
    scale: float = 0.02,
) -> dict[str, Path]:
    """PEFT checkpoints of the model `load` returns under each LoRA config of `forms`, each factor
    `scale * N(0, 1)` from the seed of its form's position: at 0.02 its update changes the greedy
    tokens of an open prompt, while a much larger one flattens the scores until BF16 rounding
    picks the token."""
    from peft import get_peft_model

    for seed, (name, config) in enumerate(forms.items()):
        model = get_peft_model(load(), config)
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for parameter, value in model.named_parameters():
                if ".lora_" in parameter:
                    _ = value.copy_(scale * torch.randn(value.shape, generator=generator))
        model.save_pretrained(str(directory / name))
    return {name: directory / name for name in forms}


@pytest.fixture(scope="module")
def adapters(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Of rank 32, the adapter terms' rank, on every attention and MLP projection, of rank 8 under
    rsLoRA on the attention projections, and of rank 4 on the query and output projections, whose
    larger scale keeps its update as large as the others' on fewer modules."""
    from peft import LoraConfig

    forms = {
        # An update grows as alpha / r times sqrt(r): alpha 45 at rank 32 keeps rank 16's size.
        "full": LoraConfig(r=32, lora_alpha=45, target_modules=ATTENTION + MLP),
        "rslora": LoraConfig(r=8, lora_alpha=16, use_rslora=True, target_modules=ATTENTION),
        "partial": LoraConfig(r=4, lora_alpha=32, target_modules=["q_proj", "o_proj"]),
    }
    return save(
        lambda: AutoModelForCausalLM.from_pretrained(MODEL),
        forms,
        tmp_path_factory.mktemp("adapters"),
    )


@pytest.fixture(scope="module")
def served(
    tmp_path_factory: pytest.TempPathFactory, chips: Chips, adapters: dict[str, Path]
) -> Generator[Callable[..., HTTPResponse]]:
    """The model with every adapter loaded at startup, four requests running at once and the cache
    sized by default, so adapters loaded later find it took all the memory calls leave; a test
    that serves another model stops it, and it starts again for the next request."""
    flags = [*LORA, "--lora-paths", *(f"{name}={path}" for name, path in adapters.items())]
    row = {"id": MODEL, "task": "text", "flags": flags}
    with ExitStack() as stack:
        live = stack.enter_context(chips.serve(tmp_path_factory, row))

        def send(path: str, body: dict[str, object] | None = None) -> HTTPResponse:
            nonlocal live
            if not chips.holds(live):
                live = stack.enter_context(chips.serve(tmp_path_factory, row))
            return live(path, body)

        yield send


@cache
def prompt() -> list[int]:
    return AutoTokenizer.from_pretrained(MODEL)(PROMPT).input_ids


def generate(
    send: Callable[..., HTTPResponse], names: list[str | None], tokens: int = TOKENS
) -> list[Answer]:
    """SGLang's answers to `PROMPT` under each adapter of `names`, `None` the base model's, in one
    request."""
    params = {"temperature": 0, "max_new_tokens": tokens, "ignore_eos": True}
    body = {"input_ids": [prompt()] * len(names), "lora_path": names, "sampling_params": params}
    with send("/generate", body) as response:
        return ANSWERS.validate_json(response.read())


def tokens(send: Callable[..., HTTPResponse], name: str) -> list[int]:
    """The greedy tokens of `PROMPT` under the adapter `name`, alone in its batch."""
    (answer,) = generate(send, [name])
    return answer["output_ids"]


def update(send: Callable[..., HTTPResponse], path: str, body: "Mapping[str, object]") -> Update:
    """SGLang's answer to a load or an unload of an adapter, which succeeded."""
    with send(path, body) as response:
        return UPDATE.validate_json(response.read())


def refused(send: Callable[..., HTTPResponse], path: str, body: "Mapping[str, object]") -> Update:
    """SGLang's answer to a load of an adapter, which failed."""
    with pytest.raises(HTTPError) as error:
        _ = send(path, body)
    assert error.value.code == 400
    return UPDATE.validate_json(error.value.read())


def assert_peft_greedy(tokens: list[int], adapter: Path | None) -> None:
    """Each of `tokens` is a greedy token of `PROMPT` and the tokens before it under Transformers
    with `adapter` merged by PEFT, in FP32: its probability is at least 95% of the highest one's."""
    from peft import PeftModel

    reference = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    if adapter is not None:
        reference = PeftModel.from_pretrained(reference, str(adapter)).merge_and_unload()
    context = torch.tensor([prompt()])
    with torch.inference_mode():
        for token in tokens:
            scores = reference(context).logits[0, -1]
            # BF16 rounding can tie the two highest scores; compare the chosen probability.
            assert (scores[token] - scores.max()).exp() >= 0.95, (adapter, tokens)
            context = torch.cat([context, torch.tensor([[token]])], dim=-1)


@pytest.mark.feature("LoRA")
def test_adapted_requests_match_peft(
    served: Callable[..., HTTPResponse], adapters: dict[str, Path]
) -> None:
    """In one request of the base model and three adapters, more than a batch's slots hold, each
    generates the greedy tokens of Transformers with its adapter merged by PEFT, though the base
    model's run of the same prompt left its pages in SGLang's radix cache."""
    _ = generate(served, [None])
    names: list[str | None] = [None, *adapters]
    answers = generate(served, names)
    # The base model reads its prompt's pages back, which no adapter's request may.
    assert answers[0]["meta_info"]["cached_tokens"] >= 16
    assert [answer["meta_info"]["cached_tokens"] for answer in answers[1:]] == [0] * len(adapters)
    for name, answer in zip(names, answers, strict=True):
        assert_peft_greedy(answer["output_ids"], None if name is None else adapters[name])
    assert len({tuple(answer["output_ids"]) for answer in answers}) == len(answers)


@pytest.mark.feature("LoRA")
def test_loads_and_unloads(
    served: Callable[..., HTTPResponse], adapters: dict[str, Path], tmp_path: Path
) -> None:
    """`/load_lora_adapter` adds an adapter and `/unload_lora_adapter` removes it: a copy of an
    adapter under another name generates its tokens, also named by the OpenAI API's `model`, and
    again once a request names it after its unload, which SGLang loads again. A name already loaded,
    and checkpoints the engine refuses, leave the loaded adapters as they were."""
    copy = shutil.copytree(adapters["full"], tmp_path / "copy")
    _ = update(served, "/load_lora_adapter", {"lora_name": "copy", "lora_path": str(copy)})
    first = tokens(served, "full")
    (copied,) = generate(served, ["copy"])
    body = {"model": f"{MODEL}:copy", "prompt": prompt(), "max_tokens": TOKENS, "temperature": 0}
    with served("/v1/completions", body | {"ignore_eos": True}) as response:
        (completed,) = COMPLETIONS.validate_json(response.read())["choices"]
    unloaded = update(served, "/unload_lora_adapter", {"lora_name": "copy"})
    assert "copy" not in unloaded["loaded_adapters"]
    again = tokens(served, "copy")
    assert copied["output_ids"] == first
    assert completed["text"] == copied["text"]
    assert again == first
    # A configuration of a rank its factors do not have, and factors of another model's width:
    # SGLang accepts both, the engine neither.
    broken = Path(shutil.copytree(adapters["full"], tmp_path / "broken"))
    config = CONFIG.validate_json((broken / "adapter_config.json").read_bytes())
    _ = (broken / "adapter_config.json").write_text(json.dumps(config | {"r": 8}))
    wide = Path(shutil.copytree(adapters["partial"], tmp_path / "wide"))
    from safetensors.torch import load_file, save_file

    factors = load_file(wide / "adapter_model.safetensors")
    save_file(
        {name: torch.cat([factor, factor], dim=-1) for name, factor in factors.items()},
        wide / "adapter_model.safetensors",
    )
    loaded = {*adapters, "copy"}
    for name, path, reason in [
        ("full", adapters["full"], "already loaded"),
        ("broken", broken, "not the configured 8"),
        ("wide", wide, "shape"),
    ]:
        answer = refused(served, "/load_lora_adapter", {"lora_name": name, "lora_path": str(path)})
        assert reason in str(answer["error_message"]), answer
        assert set(answer["loaded_adapters"]) == loaded
    assert tokens(served, "full") == first


@pytest.mark.feature("LoRA")
def test_reload_runs_the_new_revision_alone(
    served: Callable[..., HTTPResponse], adapters: dict[str, Path], tmp_path: Path
) -> None:
    """A name unloaded and loaded again from changed files is a new revision: its requests run the
    new weights and read none of the pages the old revision's requests left cached."""
    path = shutil.copytree(adapters["full"], tmp_path / "swap")
    body = {"lora_name": "swap", "lora_path": str(path)}
    _ = update(served, "/load_lora_adapter", body)
    _ = generate(served, ["swap"])
    (cached,) = generate(served, ["swap"])
    assert cached["meta_info"]["cached_tokens"] >= 16
    _ = update(served, "/unload_lora_adapter", {"lora_name": "swap"})
    shutil.rmtree(path)
    _ = shutil.copytree(adapters["rslora"], path)
    _ = update(served, "/load_lora_adapter", body)
    (answer,) = generate(served, ["swap"])
    assert answer["meta_info"]["cached_tokens"] == 0
    assert_peft_greedy(answer["output_ids"], adapters["rslora"])


@pytest.mark.feature("LoRA")
def test_loads_adapters_from_tensors(
    served: Callable[..., HTTPResponse], adapters: dict[str, Path]
) -> None:
    """`/load_lora_adapter_from_tensors` loads an adapter from its factors and configuration,
    whose requests generate the greedy tokens of Transformers with the adapter merged by PEFT;
    a configuration SGLang refuses leaves the loaded adapters as they were."""
    from safetensors.torch import load_file

    source = adapters["partial"]
    factors = load_file(source / "adapter_model.safetensors")
    config = CONFIG.validate_json((source / "adapter_config.json").read_bytes())
    payload = base64.b64encode(pickle.dumps(factors)).decode()
    body: dict[str, object] = {"config_dict": config, "serialized_named_tensors": [payload]}
    loaded = update(served, "/load_lora_adapter_from_tensors", body | {"lora_name": "tensors"})
    assert loaded["loaded_adapters"]["tensors"] == "__tensor__"
    assert_peft_greedy(tokens(served, "tensors"), source)
    answer = refused(
        served,
        "/load_lora_adapter_from_tensors",
        body | {"lora_name": "bare", "config_dict": {"r": 4, "lora_alpha": 32}},
    )
    assert "target_modules" in str(answer["error_message"]), answer
    assert set(answer["loaded_adapters"]) == set(loaded["loaded_adapters"])
