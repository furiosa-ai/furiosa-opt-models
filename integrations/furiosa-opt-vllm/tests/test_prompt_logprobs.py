"""The logprobs a request asks for of fixed token IDs at its prompt rows are its prompt logprobs."""

from __future__ import annotations

from typing import Final

import numpy as np
import pytest
from transformers import PreTrainedTokenizerBase
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

CHECKPOINT: Final = "Qwen/Qwen3-0.6B"
QUESTION: Final = "Explain briefly. " * 8 + "What is the capital of France?"
# Tokens a step schedules, fewer than the prompt holds, so the prompt is computed in chunks.
BUDGET: Final = 32
START: Final = 2


@pytest.mark.parametrize("cached", [False, True], ids=["scores", "prefix-hit"])
def test_scores_prompt(tokenizer: PreTrainedTokenizerBase, cached: bool) -> None:
    """Requested prompt-token scores match prompt logprobs, or are omitted after a prefix hit."""
    match tokenizer.apply_chat_template(
        [{"role": "user", "content": QUESTION}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ):
        case str() as text:
            prompt = TokensPrompt(prompt_token_ids=tokenizer.encode(text, add_special_tokens=False))
        case _:
            raise TypeError("untokenized chat rendering must return text")
    asked = [tokenizer.encode(word, add_special_tokens=False)[0] for word in (" Paris", " Tokyo")]
    llm = LLM(
        CHECKPOINT,
        dtype="bfloat16",
        generation_config="vllm",
        max_model_len=1024,
        max_num_batched_tokens=BUDGET,
        max_logprobs=-1,
    )
    try:
        full, scored = (
            llm.generate(prompt, params)[0]
            for params in (
                SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=-1),
                SamplingParams(
                    temperature=0,
                    max_tokens=1,
                    prompt_logprob_token_ids=asked,
                    prompt_logprob_start=START,
                    skip_reading_prefix_cache=not cached,
                ),
            )
        )
    finally:
        llm.llm_engine.engine_core.shutdown()
    length = len(prompt["prompt_token_ids"])
    assert length > BUDGET and full.prompt_logprobs is not None
    if cached:
        assert scored.num_cached_tokens > START
        assert scored.prompt_token_id_logprobs is None
        return
    assert scored.prompt_token_id_logprobs is not None
    # Row p predicts prompt token p + 1, whose prompt logprobs vLLM reports at p + 1.
    expected = [
        [step[token].logprob for token in asked]
        for step in full.prompt_logprobs[START + 1 :]
        if step is not None
    ]
    assert scored.prompt_token_id_logprobs.shape == (length - 1 - START, len(asked))
    assert np.allclose(scored.prompt_token_id_logprobs, expected, atol=1e-3)
