"""Under a logits mode of `--logprobs-mode`, vLLM reports each prompt row's logits where it reports
prompt logprobs; prompt rows skip the sampler's processing, so both logits modes report the same."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest
import torch
from vllm import LLM, SamplingParams

if TYPE_CHECKING:
    from vllm.config.model import LogprobsMode

CHECKPOINT: Final = "Qwen/Qwen3-0.6B"
PROMPT: Final = "The capital of France is"


def prompt_scores(mode: LogprobsMode) -> list[torch.Tensor]:
    """Each prompt row's scores over the whole vocabulary, from a server started in `mode`."""
    llm = LLM(
        CHECKPOINT,
        dtype="bfloat16",
        generation_config="vllm",
        max_model_len=256,
        max_logprobs=-1,
        logprobs_mode=mode,
    )
    try:
        params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=-1, detokenize=False)
        output = llm.generate(PROMPT, params)[0]
        vocabulary = llm.llm_engine.model_config.get_vocab_size()
    finally:
        llm.llm_engine.engine_core.shutdown()
    assert output.prompt_logprobs is not None
    return [
        torch.tensor([row[token].logprob for token in range(vocabulary)])
        for row in output.prompt_logprobs
        if row is not None
    ]


@pytest.fixture(scope="module")
def logprobs() -> list[torch.Tensor]:
    return prompt_scores("raw_logprobs")


@pytest.mark.parametrize("mode", ["raw_logits", "processed_logits"])
def test_reports_logits(mode: LogprobsMode, logprobs: list[torch.Tensor]) -> None:
    """Each prompt row's scores normalize to its logprobs without being normalized themselves."""
    logits = prompt_scores(mode)
    assert all(abs(float(torch.logsumexp(row, 0))) > 1 for row in logits)
    assert all(
        torch.allclose(row.log_softmax(0), reference, atol=1e-3)
        for row, reference in zip(logits, logprobs, strict=True)
    )
