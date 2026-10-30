"""vLLM's replay options on the plugin's host sampler: trace replay and sampling masks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import pytest
import torch
from transformers import PreTrainedTokenizerBase
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

if TYPE_CHECKING:
    from vllm.logprobs import SampleLogprobs

CHECKPOINT: Final = "Qwen/Qwen3-0.6B"
QUESTION: Final = "What is the capital of France? Answer with only the city name."


@pytest.fixture(scope="module")
def prompt(tokenizer: PreTrainedTokenizerBase) -> list[int]:
    """The question as one user message of the checkpoint's chat template."""
    match tokenizer.apply_chat_template(
        [{"role": "user", "content": QUESTION}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ):
        case str() as text:
            return tokenizer.encode(text, add_special_tokens=False)
        case _:
            raise TypeError("untokenized chat rendering must return text")


def steps(logprobs: SampleLogprobs | None, vocabulary: int) -> list[torch.Tensor]:
    """Each generated token's logprobs over the whole vocabulary."""
    assert logprobs is not None
    return [torch.tensor([step[index].logprob for index in range(vocabulary)]) for step in logprobs]


def test_scores_trace(tokenizer: PreTrainedTokenizerBase, prompt: list[int]) -> None:
    """Trace replay returns forced tokens with the checkpoint's scores at each trace prefix."""
    # Words that no answer to the question starts with.
    trace = tokenizer.encode(" banana kiwi mango", add_special_tokens=False)
    llm = LLM(
        CHECKPOINT,
        dtype="bfloat16",
        generation_config="vllm",
        max_model_len=1024,
        max_logprobs=-1,
        enable_trace_replay=True,
        # Attention rounds a row by the other rows of its call, so each step schedules one row:
        # the replay and every prefix then compute each position in the same one-row call.
        max_num_batched_tokens=1,
        max_num_seqs=1,
    )
    try:
        traced = llm.generate(
            TokensPrompt(prompt_token_ids=prompt),
            SamplingParams(temperature=0, logprobs=-1, trace_decode_token_ids=trace),
        )[0].outputs[0]
        # The distribution after each of the trace's prefixes, with the prefix as a prompt.
        prefixed = [
            llm.generate(
                TokensPrompt(prompt_token_ids=prompt + trace[:length]),
                SamplingParams(temperature=0, logprobs=-1, max_tokens=1),
            )[0].outputs[0]
            for length in range(len(trace))
        ]
        vocabulary = llm.llm_engine.model_config.get_vocab_size()
    finally:
        llm.llm_engine.engine_core.shutdown()
    assert list(traced.token_ids) == trace
    expected = [steps(output.logprobs, vocabulary)[0] for output in prefixed]
    measured = steps(traced.logprobs, vocabulary)
    assert all(torch.allclose(*pair, atol=1e-3) for pair in zip(measured, expected, strict=True))


def test_reports_support(prompt: list[int]) -> None:
    """Each sampling mask equals processed support and includes the generated token."""
    llm = LLM(
        CHECKPOINT,
        dtype="bfloat16",
        generation_config="vllm",
        max_model_len=1024,
        max_logprobs=-1,
        return_sampling_mask=True,
        logprobs_mode="processed_logprobs",
    )
    try:
        params = SamplingParams(temperature=1, top_k=5, seed=3, max_tokens=6, logprobs=-1)
        output = llm.generate(TokensPrompt(prompt_token_ids=prompt), params)[0].outputs[0]
        vocabulary = llm.llm_engine.model_config.get_vocab_size()
    finally:
        llm.llm_engine.engine_core.shutdown()
    assert output.sampling_mask is not None
    supports = [
        [index for index, logprob in enumerate(step.tolist()) if logprob != -math.inf]
        for step in steps(output.logprobs, vocabulary)
    ]
    assert output.sampling_mask.token_ids == supports
    assert all(token in support for token, support in zip(output.token_ids, supports, strict=True))
