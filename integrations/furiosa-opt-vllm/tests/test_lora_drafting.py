"""A drafter beside a LoRA target drafts with the base model while the target verifies each draft
with its request's adapter, so every adapted request keeps its adapter's greedy tokens."""

import gc
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest
from test_draft_model import ACCEPTED, DRAFTED, counter
from test_lora import MODEL, assert_peft_greedy, generate

if TYPE_CHECKING:
    from vllm import LLM

PARALLEL: Final = "amd/PARD-Qwen3-0.6B"


@pytest.fixture(
    scope="module",
    params=[(MODEL, False), (PARALLEL, True)],
    ids=["draft-model", "parallel-draft-model"],
)
def llm(request: pytest.FixtureRequest) -> Generator["LLM"]:
    """Two adapter slots and three loaded adapters, drafted for by a draft model of the base."""
    from vllm import LLM

    # The previous case's engine and a failed test's models sit in reference cycles until a
    # collection, and they hold the chip the engine core opens.
    _ = gc.collect()
    draft, parallel = request.param
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=2048,
        generation_config="vllm",
        enable_lora=True,
        max_loras=2,
        max_cpu_loras=3,
        max_lora_rank=32,
        max_num_seqs=4,
        disable_log_stats=False,
        speculative_config={
            "method": "draft_model",
            "model": draft,
            "num_speculative_tokens": 3,
            "parallel_drafting": parallel,
        },
    )
    yield llm
    llm.llm_engine.engine_core.shutdown()


@pytest.mark.feature("LoRA")
def test_drafted_requests_match_peft(llm: "LLM", adapters: dict[str, Path]) -> None:
    """In one call of the base model and three adapters, the drafter proposes and the target
    accepts drafts, and each request generates the greedy tokens of Transformers with its adapter
    merged by PEFT, whatever the drafts were."""
    from vllm.lora.request import LoRARequest

    named = [
        LoRARequest(name, lora_id, str(path))
        for lora_id, (name, path) in enumerate(adapters.items(), start=1)
    ]
    drafted, accepted = counter(llm, DRAFTED), counter(llm, ACCEPTED)
    generated = generate(llm, [None, *named])
    drafted, accepted = counter(llm, DRAFTED) - drafted, counter(llm, ACCEPTED) - accepted
    print(f"ACCEPTANCE adapters: {accepted:.0f}/{drafted:.0f} drafts")
    # A drafter that proposed nothing, or that verification never agreed with, would leave the
    # tokens greedy too: only the counters show that drafting ran.
    assert drafted > 0 and accepted > 0
    for request, tokens in zip([None, *named], generated, strict=True):
        assert_peft_greedy(tokens, None if request is None else Path(request.lora_path))
