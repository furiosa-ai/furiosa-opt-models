"""Under `VLLM_BATCH_INVARIANT=1` a request's tokens and hidden rows keep their bits whatever other
requests share its steps."""

from __future__ import annotations

from array import array
from typing import TYPE_CHECKING, Final

import pytest
from vllm import SamplingParams
from vllm.inputs import TokensPrompt

if TYPE_CHECKING:
    from collections.abc import Generator

    from vllm import LLM
    from vllm.config.model import ConvertOption, RunnerOption

FEATURE: Final = "Batch invariance"
# A step's token budget: behind another request's 76-token prompt, this 68-token prompt prefills
# over two steps, alone in one.
BUDGET: Final = 128
REQUEST: Final = [100, 101, 102, 103] * 17
GREEDY: Final = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
# Prompts beside it: long ones that push its prefill across steps, short ones that decode beside
# its prefill.
OTHERS: Final = {
    "long": [[200, 201, 202, 203] * 19],
    "short": [[300 + row, 7, 9] for row in range(5)],
}
COMPOSITIONS: Final = pytest.mark.parametrize(
    ("before", "after"),
    [("long", "short"), ("short", "long"), ("long", "long"), ("short", "short")],
)


@pytest.fixture(scope="module")
def llm(request: pytest.FixtureRequest) -> Generator[LLM]:
    """Qwen3-0.6B served batch invariant, generating, or pooling its last hidden row."""
    runner: RunnerOption = request.param
    convert: ConvertOption = "embed" if runner == "pooling" else "auto"
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("VLLM_BATCH_INVARIANT", "1")
        from vllm import LLM

        model = LLM(
            model="Qwen/Qwen3-0.6B",
            runner=runner,
            convert=convert,
            dtype="bfloat16",
            max_model_len=256,
            max_num_seqs=16,
            max_num_batched_tokens=BUDGET,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
        )
        yield model
        model.llm_engine.engine_core.shutdown()


def prompts(before: str, after: str) -> tuple[list[TokensPrompt], int]:
    """The request between three copies of the `before` prompts and the `after` ones, and its
    index."""
    ahead = OTHERS[before] * 3
    tokens = [*ahead, REQUEST, *OTHERS[after]]
    return [TokensPrompt(prompt_token_ids=prompt) for prompt in tokens], len(ahead)


def bits(embedding: list[float]) -> bytes:
    """The bytes of `embedding`'s values as float64, which tell -0.0 from 0.0 as `==` does not."""
    return array("d", embedding).tobytes()


@pytest.mark.feature(FEATURE)
@pytest.mark.parametrize("llm", ["generate"], indirect=True)
@COMPOSITIONS
def test_tokens_keep_bits(llm: LLM, before: str, after: str) -> None:
    """The request alone, its prompt prefilled in one step, decodes the tokens it decodes between
    other requests: its prompt then prefills over several steps, its rows sit at other offsets of
    its calls, and the calls hold other requests' prefill and decode rows."""
    alone = llm.generate([TokensPrompt(prompt_token_ids=REQUEST)], GREEDY, use_tqdm=False)
    shared, at = prompts(before, after)
    outputs = llm.generate(shared, GREEDY, use_tqdm=False)
    assert outputs[at].outputs[0].token_ids == alone[0].outputs[0].token_ids


@pytest.mark.feature(FEATURE)
@pytest.mark.parametrize("llm", ["pooling"], indirect=True)
@COMPOSITIONS
def test_embedding_keeps_bits(llm: LLM, before: str, after: str) -> None:
    """The request's embedding, its normalized last hidden row, has the same bits alone as between
    other requests whose prompts share its steps."""
    alone = llm.embed([TokensPrompt(prompt_token_ids=REQUEST)], use_tqdm=False)
    shared, at = prompts(before, after)
    outputs = llm.embed(shared, use_tqdm=False)
    assert bits(outputs[at].outputs.embedding) == bits(alone[0].outputs.embedding)
