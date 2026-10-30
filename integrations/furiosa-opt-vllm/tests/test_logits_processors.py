"""A custom logits processor of Model Runner V1's interface steers the requests it applies to and
no other, under Model Runner V1, which serves custom processors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest
import torch
from transformers import PreTrainedTokenizerBase
from typing_extensions import override
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor
from vllm.v1.sample.logits_processor.builtin import process_dict_updates

if TYPE_CHECKING:
    from collections.abc import Generator

    from vllm.config import VllmConfig

CHECKPOINT: Final = "Qwen/Qwen3-0.6B"
QUESTION: Final = "What is the capital of France? Answer with only the city name."
LENGTH: Final = 8


def target(params: SamplingParams) -> int | None:
    """The token a request's extra arguments name as its only one, if any."""
    match params.extra_args:
        case {"target_token": int() as token}:
            return token
        case _:
            return None


class TargetToken(LogitsProcessor):
    """vLLM's example processor: a request whose extra arguments name a `target_token` generates
    only that token."""

    @override
    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool) -> None:
        # Batch row -> the token its request generates.
        self.targets: dict[int, int] = {}

    @override
    def is_argmax_invariant(self) -> bool:
        return False

    @override
    def update_state(self, batch_update: BatchUpdate | None) -> None:
        _ = process_dict_updates(self.targets, batch_update, lambda params, _, __: target(params))

    @override
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for row, token in self.targets.items():
            kept = logits[row, token].clone()
            _ = logits[row].fill_(-torch.inf)
            logits[row, token] = kept
        return logits


@pytest.fixture(scope="module")
def prompt(tokenizer: PreTrainedTokenizerBase) -> TokensPrompt:
    """The question as one user message of the checkpoint's chat template."""
    match tokenizer.apply_chat_template(
        [{"role": "user", "content": QUESTION}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ):
        case str() as text:
            return TokensPrompt(prompt_token_ids=tokenizer.encode(text, add_special_tokens=False))
        case _:
            raise TypeError("untokenized chat rendering must return text")


@pytest.fixture(scope="module")
def banana(tokenizer: PreTrainedTokenizerBase) -> int:
    """A token that no answer to the question starts with."""
    return tokenizer.encode(" banana", add_special_tokens=False)[0]


@pytest.fixture(scope="module")
def llm() -> Generator[LLM]:
    """An engine under Model Runner V1 with the processor, which vLLM loads at startup."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
        llm = LLM(
            CHECKPOINT,
            dtype="bfloat16",
            generation_config="vllm",
            max_model_len=1024,
            logits_processors=[TargetToken],
        )
        try:
            yield llm
        finally:
            llm.llm_engine.engine_core.shutdown()


@pytest.fixture(scope="module")
def batched(llm: LLM, prompt: TokensPrompt, banana: int) -> tuple[list[int], list[int]]:
    """The tokens of a request that targets `banana`, and of a plain one in the same batch."""
    targeted, plain = llm.generate(
        [prompt, prompt],
        [
            SamplingParams(temperature=0, max_tokens=LENGTH, extra_args={"target_token": banana}),
            SamplingParams(temperature=0, max_tokens=LENGTH),
        ],
    )
    return list(targeted.outputs[0].token_ids), list(plain.outputs[0].token_ids)


@pytest.mark.feature("Custom logits processors")
def test_generates_target(batched: tuple[list[int], list[int]], banana: int) -> None:
    """A custom processor forces its requested token for every generated position."""
    targeted, _ = batched
    assert targeted == [banana] * LENGTH


def test_keeps_plain(batched: tuple[list[int], list[int]], llm: LLM, prompt: TokensPrompt) -> None:
    """A custom processor leaves an untargeted request's answer unchanged."""
    _, plain = batched
    alone = llm.generate(prompt, SamplingParams(temperature=0, max_tokens=LENGTH))[0]
    assert plain == list(alone.outputs[0].token_ids)
