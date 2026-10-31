"""SGLang's `--quantization fp8` serves a BF16 checkpoint as SGLang quantizes it at load."""

from collections.abc import Callable
from http.client import HTTPResponse
from typing import Final

import pytest
import torch
from pydantic import TypeAdapter
from test_server import NGRAM, STANDALONE, Chips
from test_server import chips as chips
from test_server import plugin as plugin
from transformers import AutoModelForCausalLM
from typing_extensions import TypedDict

FP8: Final = ("--quantization", "fp8")
DRAFT: Final = (*STANDALONE, "Qwen/Qwen3-0.6B")
# Prompts on which each size's BF16 weights, and its weights under one scale a tensor, answer apart
# from SGLang's FP8 ones, so the tests fail when the load ignores the quantization or its scales.
PROMPTS: Final = (
    "The first rule of programming is",
    "The quick brown fox",
    "In a distant galaxy, a small robot",
)
TOKENS: Final = 16


class Tokenized(TypedDict):
    tokens: list[list[int]]


class Generated(TypedDict):
    output_ids: list[int]


TOKENIZED: Final = TypeAdapter(Tokenized)
ANSWERS: Final = TypeAdapter(list[Generated])


def quantized(weight: torch.Tensor) -> torch.Tensor:
    """`weight` restored to BF16 from the E4M3 codes SGLang's `fp8` gives a BF16 projection: one
    FP32 scale per output channel, its largest magnitude over 448, each code its value times the
    scale's reciprocal, saturated, as SGLang's per-token FP8 kernel computes them."""
    largest = torch.finfo(torch.float8_e4m3fn).max
    scale = weight.float().abs().amax(dim=1, keepdim=True) / largest
    inverse = torch.where(scale == 0, 0.0, 1.0 / scale)
    codes = (weight.float() * inverse).clamp(-largest, largest).to(torch.float8_e4m3fn)
    return (codes.float() * scale).to(torch.bfloat16)


def generate(send: Callable[..., HTTPResponse]) -> tuple[list[list[int]], list[list[int]]]:
    """The tokens of `PROMPTS` and their greedy answers of `TOKENS` tokens, in one batch."""
    greedy = {"temperature": 0, "max_new_tokens": TOKENS, "ignore_eos": True}
    with send("/v1/tokenize", {"prompt": list(PROMPTS)}) as response:
        prompts = TOKENIZED.validate_json(response.read())["tokens"]
    with send("/generate", {"input_ids": prompts, "sampling_params": greedy}) as response:
        answers = [answer["output_ids"] for answer in ANSWERS.validate_json(response.read())]
    return prompts, answers


def assert_quantized(checkpoint: str, prompts: list[list[int]], answers: list[list[int]]) -> None:
    """Each greedy token is within 5% of the top probability of Transformers on `checkpoint` as
    SGLang quantizes it."""
    reference = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32)
    head = reference.get_output_embeddings()
    with torch.inference_mode():
        # SGLang quantizes each linear layer but the vocabulary head row by row, so its fused q/k/v
        # and gate/up quantize as their parts do.
        for module in reference.modules():
            if isinstance(module, torch.nn.Linear) and module is not head:
                _ = module.weight.copy_(quantized(module.weight))
        for prompt, answer in zip(prompts, answers, strict=True):
            assert len(answer) == TOKENS
            logits: torch.Tensor = reference(torch.tensor([prompt + answer])).logits
            expected = logits[0, len(prompt) - 1 : -1].log_softmax(-1)
            chosen = expected[torch.arange(TOKENS), torch.tensor(answer)]
            # BF16 rounding can tie the two highest scores.
            assert ((chosen - expected.max(-1).values).exp() >= 0.95).all(), (prompt, answer)


@pytest.mark.feature("Online quantization")
@pytest.mark.parametrize(
    ("checkpoint", "flags"),
    [
        pytest.param("Qwen/Qwen3-0.6B", (), id="0.6B"),
        pytest.param("Qwen/Qwen3-1.7B", (), id="1.7B"),
        pytest.param("Qwen/Qwen3-4B", (), id="4B"),
        pytest.param("Qwen/Qwen3-1.7B", NGRAM, id="1.7B-ngram"),
        pytest.param("Qwen/Qwen3-1.7B", DRAFT, id="1.7B-inherited-draft"),
        pytest.param(
            "Qwen/Qwen3-1.7B",
            (*DRAFT, "--speculative-draft-model-quantization", "unquant"),
            id="1.7B-unquantized-draft",
        ),
    ],
)
def test_generates_as_the_checkpoint_quantized_by_sglang(
    tmp_path_factory: pytest.TempPathFactory,
    chips: Chips,
    checkpoint: str,
    flags: tuple[str, ...],
) -> None:
    """Greedy tokens of a BF16 checkpoint served with `--quantization fp8`, alone or verifying the
    drafts of a speculative algorithm, are those of the checkpoint as SGLang quantizes it."""
    row = {"id": checkpoint, "task": "text", "flags": [*FP8, *flags]}
    with chips.serve(tmp_path_factory, row) as send:
        prompts, answers = generate(send)
    assert_quantized(checkpoint, prompts, answers)


@pytest.mark.feature("Online quantization")
def test_refuses_a_draft_inheriting_an_unserved_quantization(
    tmp_path_factory: pytest.TempPathFactory, chips: Chips
) -> None:
    """A STANDALONE draft model takes the target's `fp8`, so a draft size with no FP8 form stops
    the server with the engine's reason."""
    row = {"id": "Qwen/Qwen3-1.7B", "task": "text", "flags": [*FP8, *STANDALONE, "Qwen/Qwen3-8B"]}
    log = chips.refusal(tmp_path_factory, row)
    assert "a load quantizes the checkpoint, the line holds BF16 projections" in log


@pytest.mark.feature("Online quantization")
@pytest.mark.feature("Sleep mode")
@pytest.mark.parametrize("backup", [False, True], ids=["files", "host-copy"])
def test_wakes_to_the_quantized_answers(
    tmp_path_factory: pytest.TempPathFactory, chips: Chips, backup: bool
) -> None:
    """Weights released by the memory saver resume quantized: the answers after a release and a
    resume are those of before, which are the checkpoint's as SGLang quantizes it."""
    checkpoint = "Qwen/Qwen3-0.6B"
    saver = ("--enable-memory-saver", *(("--enable-weights-cpu-backup",) if backup else ()))
    row = {"id": checkpoint, "task": "text", "flags": [*FP8, *saver]}
    with chips.serve(tmp_path_factory, row) as send:
        prompts, before = generate(send)
        for path in ("/release_memory_occupation", "/resume_memory_occupation"):
            with send(path, {"tags": None}) as response:
                assert response.status == 200
        _, after = generate(send)
    assert after == before
    assert_quantized(checkpoint, prompts, before)
