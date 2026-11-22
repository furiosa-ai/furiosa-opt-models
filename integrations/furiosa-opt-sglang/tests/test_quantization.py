"""SGLang's `--quantization fp8` uses the non-CUDA per-tensor weight fallback."""

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
# Prompts exercise the tensor-wise FP8 reference and shared scales for fused projections.
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


def quantized(weight: torch.Tensor, maximum: torch.Tensor | None = None) -> torch.Tensor:
    """`weight` restored to BF16 from E4M3 using one FP32 scale, its maximum over 448."""
    largest = torch.finfo(torch.float8_e4m3fn).max
    maximum = weight.float().abs().amax() if maximum is None else maximum
    scale = maximum / largest
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
    """Each greedy token is within 5% of Transformers with tensor-wise FP8 weights and BF16 acts."""
    reference = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16)
    head = reference.get_output_embeddings()
    with torch.inference_mode():
        linears = {
            name: module
            for name, module in reference.named_modules()
            if isinstance(module, torch.nn.Linear) and module is not head
        }
        fused = (("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj"))
        maxima: dict[str, torch.Tensor] = {}
        for name, module in linears.items():
            parent, projection = name.rsplit(".", 1)
            group = next((group for group in fused if projection in group), None)
            if group is None:
                maxima[name] = module.weight.float().abs().amax()
                continue
            key = f"{parent}:{','.join(group)}"
            if key not in maxima:
                siblings = [linears[f"{parent}.{part}"] for part in group]
                maxima[key] = torch.stack(
                    [sibling.weight.float().abs().amax() for sibling in siblings]
                ).amax()
            maxima[name] = maxima[key]
        # SGLang fuses Q/K/V and gate/up into projection tensors before its non-CUDA fallback.
        for name, module in linears.items():
            _ = module.weight.copy_(quantized(module.weight, maxima[name]))
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
def test_generates_with_sglang_non_cuda_fp8_weights(
    tmp_path_factory: pytest.TempPathFactory,
    chips: Chips,
    checkpoint: str,
    flags: tuple[str, ...],
) -> None:
    """Greedy tokens match the BF16 model with SGLang's non-CUDA tensor-wise FP8 weight fallback."""
    row = {"id": checkpoint, "task": "text", "flags": [*FP8, *flags]}
    with chips.serve(tmp_path_factory, row) as send:
        prompts, answers = generate(send)
    assert_quantized(checkpoint, prompts, answers)


@pytest.mark.feature("Online quantization")
@pytest.mark.feature("Speculative decoding (ngram, suffix, draft model)")
def test_draft_owned_fp8_keeps_bf16_target_answers(
    tmp_path_factory: pytest.TempPathFactory, chips: Chips
) -> None:
    """A draft's online FP8 leaves its BF16 target's greedy answers unchanged."""
    checkpoint = "Qwen/Qwen3-1.7B"
    with chips.serve(tmp_path_factory, {"id": checkpoint, "task": "text"}) as send:
        expected = generate(send)
    flags = (*DRAFT, "--speculative-draft-model-quantization", "fp8")
    row = {"id": checkpoint, "task": "text", "flags": flags}
    with chips.serve(tmp_path_factory, row) as send:
        actual = generate(send)
    assert actual == expected


@pytest.mark.feature("Online quantization")
@pytest.mark.parametrize(
    ("target_flags", "draft_flags"),
    [
        pytest.param(FP8, (), id="inherited"),
        pytest.param((), ("--speculative-draft-model-quantization", "fp8"), id="draft-owned-fp8"),
    ],
)
def test_refuses_a_draft_with_unserved_fp8(
    tmp_path_factory: pytest.TempPathFactory,
    chips: Chips,
    target_flags: tuple[str, ...],
    draft_flags: tuple[str, ...],
) -> None:
    """An inherited or draft-owned `fp8` refuses a draft size with no FP8 form."""
    row = {
        "id": "Qwen/Qwen3-1.7B",
        "task": "text",
        "flags": [*target_flags, *STANDALONE, "Qwen/Qwen3-8B", *draft_flags],
    }
    log = chips.refusal(tmp_path_factory, row)
    assert (
        "furiosa-opt-sglang a load quantizes the checkpoint, the line holds BF16 projections" in log
    )


@pytest.mark.feature("Online quantization")
@pytest.mark.feature("Sleep mode")
@pytest.mark.parametrize("backup", [False, True], ids=["files", "host-copy"])
def test_wakes_to_the_quantized_answers(
    tmp_path_factory: pytest.TempPathFactory, chips: Chips, backup: bool
) -> None:
    """Weights released by the memory saver resume quantized and preserve the answers."""
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
