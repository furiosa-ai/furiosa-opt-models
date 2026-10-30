"""vLLM ends a reasoning checkpoint's thinking once a request's thinking token budget is spent."""

from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    from vllm import LLM

TASK = "text"
START, END = "<think>", "</think>"
BUDGET = 4


@pytest.fixture(scope="module")
def tokenizer(row: dict[str, Any]) -> "PreTrainedTokenizerBase":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(row["id"])
    if START not in tokenizer.get_vocab():
        pytest.skip("the checkpoint has no thinking markers")
    return tokenizer


@pytest.fixture(scope="module")
def llm(options: dict[str, Any], tokenizer: "PreTrainedTokenizerBase") -> Generator["LLM"]:
    """The engine with the reasoning parser; `tokenizer` skips a checkpoint that cannot think."""
    from vllm import LLM

    llm = LLM(**options | {"max_num_seqs": 2, "reasoning_parser": "qwen3"})
    yield llm
    llm.llm_engine.engine_core.shutdown()


def thinking(tokenizer: "PreTrainedTokenizerBase", tokens: list[int]) -> int | None:
    """Tokens between the last thinking start and the end after it; None while thinking goes on."""
    start = tokenizer.encode(START, add_special_tokens=False)
    end = tokenizer.encode(END, add_special_tokens=False)
    begun = max(
        (at + len(start) for at in range(len(tokens)) if tokens[at : at + len(start)] == start),
        default=None,
    )
    if begun is None:
        return None
    return next(
        (at - begun for at in range(begun, len(tokens)) if tokens[at : at + len(end)] == end),
        None,
    )


@pytest.mark.feature("Tool calling and reasoning outputs")
def test_thinking_ends_within_the_budget(llm: "LLM", tokenizer: "PreTrainedTokenizerBase") -> None:
    from vllm import SamplingParams

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "How many prime numbers are below 30?"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    free, budgeted = (
        llm.generate(prompt, SamplingParams(temperature=0, max_tokens=64, **budget))[0]
        for budget in ({}, {"thinking_token_budget": BUDGET})
    )
    prompted = list(free.prompt_token_ids or [])
    unbounded = prompted + list(free.outputs[0].token_ids)
    if START not in tokenizer.decode(unbounded):
        pytest.skip("the checkpoint does not think on this prompt")
    if (spent := thinking(tokenizer, unbounded)) is not None and spent <= BUDGET:
        pytest.skip("the checkpoint thinks within the budget unprompted")
    spent = thinking(tokenizer, prompted + list(budgeted.outputs[0].token_ids))
    assert spent is not None and spent <= BUDGET
