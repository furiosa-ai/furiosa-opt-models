"""A Qwen3-0.6B server with a watermark generates text vLLM's watermark detector scores as
watermarked, unless the request opts out, and repeats it on a seeded rerun."""

from collections.abc import Generator
from typing import Final

import pytest
from vllm import LLM, SamplingParams
from vllm.v1.watermarking import GumbelWatermarkDetector

KEY: Final = 20261023
# Far below vLLM's default of 0.01, so an unwatermarked text passes for a watermarked one with odds
# of one in a million rather than one in a hundred.
DETECTOR: Final = GumbelWatermarkDetector(KEY, p_value_threshold=1e-6)
PROMPTS: Final = [
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Describe a walk through a forest in autumn, in a few paragraphs.",
]


@pytest.fixture(scope="module")
def llm() -> Generator[LLM]:
    # Without prefix caching, a rerun computes its prompts' logits as the first run did.
    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        max_model_len=2048,
        max_num_seqs=2,
        enable_prefix_caching=False,
        watermark_config={"key": KEY},
    )
    yield llm
    llm.llm_engine.engine_core.shutdown()


def generated(llm: LLM, prompts: list[str], *, watermarking: bool, seed: int) -> list[list[int]]:
    """The 128 tokens a seeded request samples for each of `prompts`, run together, keeping or
    opting out of the server's watermark."""
    params = SamplingParams(
        temperature=1.0, seed=seed, max_tokens=128, ignore_eos=True, watermarking=watermarking
    )
    return [list(output.outputs[0].token_ids) for output in llm.generate(prompts, params)]


@pytest.mark.feature("Watermarking")
def test_detects_watermark(llm: LLM) -> None:
    """vLLM's detector finds the watermark in every text."""
    texts = generated(llm, PROMPTS, watermarking=True, seed=0)
    assert all(DETECTOR.detect(tokens).is_watermarked for tokens in texts)


def test_honors_optout(llm: LLM) -> None:
    """vLLM's detector finds no watermark in the text of requests that opt out."""
    texts = generated(llm, PROMPTS, watermarking=False, seed=0)
    assert not any(DETECTOR.detect(tokens).is_watermarked for tokens in texts)


@pytest.mark.parametrize("watermarking", [True, False], ids=["watermarked", "opted-out"])
def test_repeats_seeded(llm: LLM, watermarking: bool) -> None:
    """A rerun with the same seed samples the same tokens."""
    # The engine starts a step once a request arrives, so a batch's later request joins its first
    # step or the next; attention rounds a row by its call's other rows, so each prompt runs alone.
    first, rerun = (
        [generated(llm, [prompt], watermarking=watermarking, seed=7) for prompt in PROMPTS]
        for _ in range(2)
    )
    assert rerun == first
