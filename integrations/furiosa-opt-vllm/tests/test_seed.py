"""vLLM's global seed, `--seed`, makes the samples of requests without a seed of their own repeat
from one server start to the next."""

from __future__ import annotations

from typing import Final

import pytest
from vllm import LLM, SamplingParams

CHECKPOINT: Final = "Qwen/Qwen3-0.6B"
PROMPTS: Final = ["Name a color.", "Write one sentence about rivers.", "Count to five."]


def sample(seed: int) -> list[list[int]]:
    """Each prompt's tokens, sampled without a request seed by a server started with `seed`."""
    llm = LLM(CHECKPOINT, dtype="bfloat16", generation_config="vllm", max_model_len=256, seed=seed)
    try:
        # One request a call, so arrival cannot change the calls or the order of draws.
        params = SamplingParams(temperature=1, max_tokens=16)
        outputs = [llm.generate(prompt, params, use_tqdm=False)[0] for prompt in PROMPTS]
    finally:
        llm.llm_engine.engine_core.shutdown()
    return [list(output.outputs[0].token_ids) for output in outputs]


@pytest.fixture(scope="module")
def first() -> list[list[int]]:
    return sample(1)


def test_repeats_seed(first: list[list[int]]) -> None:
    """The same server seed repeats tokens under the same sampling order and call composition."""
    assert sample(1) == first


def test_distinguishes_seed(first: list[list[int]]) -> None:
    """A different server seed changes the unseeded requests' sampled tokens."""
    assert sample(2) != first
