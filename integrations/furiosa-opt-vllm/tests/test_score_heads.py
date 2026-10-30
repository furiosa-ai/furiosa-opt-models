"""A score head of any label count the engine compiles takes vLLM's activation for its problem type
and vLLM's calibration; a classification route without a score head it can serve refuses to start.

Each head is a seeded synthetic one over a cached decoder, so it proves the head's arithmetic and
vLLM's semantics, not a model's quality.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple, assert_never

import pytest
from conftest import Reference
from pydantic import JsonValue, TypeAdapter

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel
    from vllm import LLM
    from vllm.config import PoolerConfig

FRAME = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"
"""A converted reranker: its configuration and tokenizer frame each synthetic classifier."""
DECODER = "Qwen/Qwen3-Reranker-0.6B"
"""The same decoder in BF16 without a score head, whose weights each classifier reads in place."""
PROMPTS = ["Paris is the capital of France.", "What is a compiler? " * 6]
MEAN, SIGMA = 0.5, 2.0
"""The calibration of the three-label head: activation((logit - MEAN) / SIGMA)."""


class Head(NamedTuple):
    """A synthetic score head of `labels` FP32 rows, with or without a bias, for `problem`; none
    leaves vLLM to activate by the label count."""

    labels: int
    problem: Literal["multi_label_classification", "regression"] | None
    bias: bool
    calibrated: bool = False
    task: Literal["classify", "token_classify"] = "classify"


HEADS = [
    Head(3, None, bias=True, calibrated=True),
    Head(256, "multi_label_classification", bias=False),
    Head(2, "regression", bias=True),
]
# The tests of one head share the engine `served` starts for it.
heads = pytest.mark.parametrize(
    "head", HEADS, ids=[f"{head.labels}-{head.problem}" for head in HEADS], scope="module"
)


class Served(NamedTuple):
    llm: LLM
    head: Head
    weight: torch.Tensor
    bias: torch.Tensor | None


def checkpoint(directory: Path, head: Head) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Write a classifier of `head` in `directory`: the frame's configuration with the head's
    labels and problem, its tokenizer, the decoder's weights linked in place and a seeded head."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    frame = Path(hf_hub_download(FRAME, "config.json")).read_bytes()
    config = TypeAdapter(dict[str, JsonValue]).validate_json(frame)
    config["id2label"] = {str(label): f"LABEL_{label}" for label in range(head.labels)}
    config["label2id"] = {f"LABEL_{label}": label for label in range(head.labels)}
    config["torch_dtype"] = "bfloat16"
    if head.problem is not None:
        config["problem_type"] = head.problem
    _ = (directory / "config.json").write_text(json.dumps(config))
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        (directory / name).symlink_to(hf_hub_download(FRAME, name))
    hidden = TypeAdapter(int).validate_python(config["hidden_size"])
    generator = torch.Generator().manual_seed(head.labels)
    weight = torch.randn(head.labels, hidden, generator=generator) * 0.02
    bias = torch.randn(head.labels, generator=generator) if head.bias else None
    tensors = {"score.weight": weight} | ({} if bias is None else {"score.bias": bias})
    save_file(tensors, directory / "head.safetensors")
    decoder = hf_hub_download(DECODER, "model.safetensors")
    (directory / "decoder.safetensors").symlink_to(decoder)
    with safe_open(decoder, "pt") as weights:
        names = list(weights.keys())
    index = dict.fromkeys(names, "decoder.safetensors") | dict.fromkeys(tensors, "head.safetensors")
    _ = (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    return weight, bias


@pytest.fixture(scope="module")
def decoder() -> PreTrainedModel:
    """The decoder in FP32, the reference precision for every head."""
    import torch
    from transformers import AutoModel

    return AutoModel.from_pretrained(DECODER, dtype=torch.float32).eval()


@pytest.fixture(scope="module")
def served(head: Head, tmp_path_factory: pytest.TempPathFactory) -> Generator[Served]:
    from vllm import LLM
    from vllm.config import PoolerConfig

    directory = tmp_path_factory.mktemp(f"head-{head.labels}")
    weight, bias = checkpoint(directory, head)
    pooler: PoolerConfig = (
        PoolerConfig(task=head.task, use_activation=False, logit_mean=MEAN, logit_sigma=SIGMA)
        if head.calibrated
        else PoolerConfig(task=head.task)
    )
    llm = LLM(
        model=str(directory),
        runner="pooling",
        convert="classify",
        dtype="bfloat16",
        max_num_seqs=2,
        max_num_batched_tokens=16,
        enable_chunked_prefill=True,
        pooler_config=pooler,
    )
    yield Served(llm, head, weight, bias)
    llm.llm_engine.engine_core.shutdown()


def scores(served: Served, use_activation: bool | None) -> tuple[torch.Tensor, list[list[int]]]:
    """Each prompt's sequence scores at `use_activation` (`None` keeps the configured default)."""
    import torch
    from vllm import PoolingParams

    outputs = served.llm.classify(
        PROMPTS, pooling_params=PoolingParams(use_activation=use_activation), use_tqdm=False
    )
    return torch.tensor([output.outputs.probs for output in outputs]), [
        output.prompt_token_ids for output in outputs
    ]


@heads
def test_scores_last(served: Served, decoder: PreTrainedModel) -> None:
    import torch

    actual, ids = scores(served, use_activation=False)
    with torch.inference_mode():
        rows = [
            decoder(input_ids=torch.tensor([tokens])).last_hidden_state[0, -1] for tokens in ids
        ]
    if served.head.calibrated:
        # vLLM calibrates before it activates: (logit - MEAN) / SIGMA.
        actual = actual * SIGMA + MEAN
    Reference(decoder, served.weight, served.bias).check(actual, torch.stack(rows))


@heads
def test_activates(served: Served) -> None:
    import torch

    raw, _ = scores(served, use_activation=False)
    # A request's activation overrides the configured default.
    activated, _ = scores(served, use_activation=True)
    match served.head.problem:
        case "multi_label_classification":
            expected = torch.sigmoid(raw)
        case "regression":
            expected = raw
        case None:
            expected = torch.softmax(raw, dim=-1)
        case _:
            assert_never(served.head.problem)
    torch.testing.assert_close(activated, expected)
    default, _ = scores(served, use_activation=None)
    torch.testing.assert_close(default, raw if served.head.calibrated else activated)


@pytest.mark.parametrize(
    ("model", "overrides", "refusal"),
    [
        (FRAME, {"head_dtype": "model"}, "computes in float32"),
        ("Qwen/Qwen3-0.6B", {}, "has no score head"),
    ],
    ids=["head-dtype", "vocabulary-head"],
)
def test_refuses_head(
    model: str, overrides: dict[str, str], refusal: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm import LLM

    # The engine core runs in this process, so the load's refusal reaches the test.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    with pytest.raises(ValueError, match=refusal):
        _ = LLM(
            model=model,
            runner="pooling",
            convert="classify",
            dtype="bfloat16",
            hf_overrides=overrides,
        )


@pytest.mark.parametrize(
    "head", [HEADS[0]._replace(task="token_classify")], ids=["tokens"], scope="module"
)
def test_scores_tokens(served: Served, decoder: PreTrainedModel) -> None:
    import torch
    from vllm import PoolingParams

    prompts = PROMPTS[1:]
    raw = served.llm.encode(
        prompts, PoolingParams(use_activation=False), pooling_task="token_classify", use_tqdm=False
    )
    activated = served.llm.encode(
        prompts, PoolingParams(use_activation=True), pooling_task="token_classify", use_tqdm=False
    )
    for unactivated, scored in zip(raw, activated, strict=True):
        ids = torch.tensor([unactivated.prompt_token_ids])
        with torch.inference_mode():
            rows = decoder(input_ids=ids).last_hidden_state[0]
        Reference(decoder, served.weight, served.bias).check(
            unactivated.outputs.data * SIGMA + MEAN, rows
        )
        torch.testing.assert_close(scored.outputs.data, unactivated.outputs.data.softmax(-1))
