"""A parallel draft model drafts a step's positions as one forward of its known tokens and a mask
token at each later position computes them, and keeps the target's greedy tokens."""

from __future__ import annotations

import gc
from collections.abc import Generator
from typing import TYPE_CHECKING, Final

import pytest
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams
from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.metrics.reader import Counter
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

import furiosa_opt_engine as engine
from furiosa_opt_vllm.additional import Section
from furiosa_opt_vllm.admission import Draft, speculative
from furiosa_opt_vllm.decoder import Decoder
from furiosa_opt_vllm.drafting import Drafter, Drafting
from furiosa_opt_vllm.runner import Runner

if TYPE_CHECKING:
    from conftest import Greedy
    from vllm import RequestOutput

TARGET: Final = "Qwen/Qwen3-1.7B"
DRAFT: Final = "amd/PARD-Qwen3-0.6B"
VOCABULARY: Final = 151936
TEXT = "The quick brown fox jumps over the lazy dog. " * 8
PROMPTS: Final = [
    f"Copy the following text exactly, without any explanation:\n{TEXT}",
    "Describe the water cycle.",
]
DRAFTED = "vllm:spec_decode_num_draft_tokens"
ACCEPTED = "vllm:spec_decode_num_accepted_tokens"
# The tokens each prompt generates at most.
LENGTH: Final = 64


@pytest.fixture(scope="module")
def config() -> VllmConfig:
    """The target drafted by the parallel draft model on the target's chip."""
    return EngineArgs(
        model=TARGET,
        dtype="bfloat16",
        max_model_len=2048,
        speculative_config={
            "method": "draft_model",
            "model": DRAFT,
            "num_speculative_tokens": 4,
            "parallel_drafting": True,
        },
    ).create_engine_config()


# Four drafts' masks fit one of Qwen3-0.6B's 16-row calls; eighteen drafts' masks take two.
@pytest.mark.parametrize("count", [4, 18], ids=["one-call", "two-calls"])
def test_proposes_parallel_drafts(config: VllmConfig, greedy: Greedy, count: int) -> None:
    """Each request's drafts are the draft model's greedy tokens at its last known position and
    at `count - 1` mask tokens after it, as one Transformers forward of them all computes them:
    proposed together, after accepted and rejected drafts whose positions the next step writes
    again, and for known tokens that take more than one call's rows."""
    draft = speculative(config, Section.parse(config))
    assert isinstance(draft, Draft)
    assert draft.mask is not None
    # A failed earlier case's models sit in its traceback's reference cycle until a collection,
    # and they hold the chip this load opens.
    _ = gc.collect()
    target = Decoder.load(config)
    assert isinstance(target.model, engine.qwen3.ForCausalLM)
    block = target.model.cache.pools[0].block
    # The runner's requests and batch, which a proposal reads and its sampled tokens extend.
    runner = Runner(config)
    runner.input_batch = InputBatch(
        max_num_reqs=2,
        max_model_len=2048,
        max_num_batched_tokens=32,
        device=torch.device("cpu"),
        vocab_size=VOCABULARY,
        block_sizes=[block],
        kernel_block_sizes=[block],
        max_num_blocks_per_req=[128],
    )
    drafter = Drafter.load(draft, target)
    drafting = Drafting(drafter, 256, runner.requests)
    match AutoTokenizer.from_pretrained(DRAFT):
        case PreTrainedTokenizerBase() as tokenizer:
            text = tokenizer.encode(
                "Repeat this exactly: The quick brown fox jumps over the lazy dog.\n"
            )
        case _:
            raise TypeError("tokenizer loading must return a pretrained tokenizer")
    prompts = {
        "short": text,
        "long": (text * (drafter.model.rows // len(text) + 2))[: drafter.model.rows + 3],
    }
    tables = {"short": list(range(128)), "long": list(range(128, 256))}
    for req_id, prompt in prompts.items():
        runner.requests[req_id] = CachedRequestState(
            req_id=req_id,
            prompt_token_ids=prompt,
            mm_features=[],
            sampling_params=SamplingParams(temperature=0),
            generator=None,
            block_ids=(tables[req_id],),
            num_computed_tokens=0,
            output_token_ids=[],
        )
        _ = runner.input_batch.add_request(runner.requests[req_id])

    proposed: dict[str, list[int]] = {req_id: [] for req_id in prompts}
    # Later drafts a drafter that ignored the mask would propose and this oracle refuses.
    sequential: list[tuple[str, int, int]] = []
    # A step's tokens reject a request's first draft, or accept all of its drafts and add one.
    for rejected in [(), ("short",), ("long",)]:
        sampled = {
            req_id: [(proposed[req_id][0] + 1) % VOCABULARY]
            if req_id in rejected
            else [*proposed[req_id], 13]
            for req_id in prompts
        }
        for req_id, tokens in sampled.items():
            runner.extend(runner.requests[req_id], tokens)
        drafts = drafting.propose(runner.input_batch, list(sampled.values()), count)
        proposed = dict(zip(prompts, drafts, strict=True))
        for req_id, prompt in prompts.items():
            known = [*prompt, *runner.requests[req_id].output_token_ids]
            assert len(proposed[req_id]) == count, (req_id, proposed[req_id])
            scores = greedy.scores(DRAFT, known + [draft.mask] * (count - 1))[len(known) - 1 :]
            greedy.check(scores, proposed[req_id])
            continued = list(known)
            for _ in range(count):
                continued.append(int(greedy.scores(DRAFT, continued)[-1].argmax()))
            sequential.extend(
                (req_id, at, token)
                for at, token, _ in greedy.far(scores, continued[len(known) :])
                if at > 0
            )
    print(f"SEQUENTIAL refused at {len(sequential)} of {3 * 2 * (count - 1)} later drafts")
    # The check passes only for drafts read after mask tokens, not for drafts read after drafts.
    assert sequential


@pytest.fixture(scope="module", params=[4, 18], ids=["one-call", "two-calls"])
def llm(request: pytest.FixtureRequest) -> Generator[LLM]:
    # The models this module's earlier tests load sit in reference cycles until a collection, and
    # they hold the chip the engine core opens.
    _ = gc.collect()
    llm = LLM(
        model=TARGET,
        dtype="bfloat16",
        max_model_len=2048,
        generation_config="vllm",
        disable_log_stats=False,
        speculative_config={
            "method": "draft_model",
            "model": DRAFT,
            "num_speculative_tokens": request.param,
            "parallel_drafting": True,
        },
    )
    yield llm
    llm.llm_engine.engine_core.shutdown()


def counter(llm: LLM, name: str) -> float:
    """vLLM's counter `name`, summed over its labels."""
    return sum(
        metric.value
        for metric in llm.get_metrics()
        if isinstance(metric, Counter) and metric.name == name
    )


@pytest.mark.feature("Speculative decoding (EAGLE, MTP, Medusa, MLP, parallel draft models)")
def test_keeps_greedy(llm: LLM, greedy: Greedy) -> None:
    """Requests drafted alone and together keep the target's greedy tokens, and on copied text the
    target accepts a share of the drafts no draft model that lost its prefix reaches."""

    def check(results: list[RequestOutput]) -> None:
        for result in results:
            assert result.prompt_token_ids is not None
            tokens = list(result.outputs[0].token_ids)
            greedy.continuation(TARGET, list(result.prompt_token_ids), tokens, LENGTH)

    drafted, accepted = counter(llm, DRAFTED), counter(llm, ACCEPTED)
    check(llm.generate(PROMPTS[:1], SamplingParams(temperature=0, max_tokens=LENGTH)))
    copied = counter(llm, ACCEPTED) - accepted, counter(llm, DRAFTED) - drafted
    print(f"ACCEPTANCE copy: {copied[0]:.0f}/{copied[1]:.0f} drafts")
    assert copied[1] > 0 and copied[0] >= copied[1] / 2
    check(llm.generate(PROMPTS, SamplingParams(temperature=0, max_tokens=LENGTH)))
