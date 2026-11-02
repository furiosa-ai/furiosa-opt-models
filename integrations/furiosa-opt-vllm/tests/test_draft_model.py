"""A model drafter loaded beside its target preserves target decoding."""

from __future__ import annotations

import gc
from collections.abc import Callable, Generator
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

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
from furiosa_opt_vllm.admission import Draft, Refused, speculative
from furiosa_opt_vllm.decoder import Decoder
from furiosa_opt_vllm.drafting import Drafter, Drafting
from furiosa_opt_vllm.runner import Runner

if TYPE_CHECKING:
    from conftest import Greedy

TARGET = "Qwen/Qwen3-1.7B"
TEXT = "The quick brown fox jumps over the lazy dog. " * 8
PROMPT = f"Copy the following text exactly, without any explanation:\n{TEXT}"
DRAFTED = "vllm:spec_decode_num_draft_tokens"
ACCEPTED = "vllm:spec_decode_num_accepted_tokens"


def refusal(load: Callable[[], Drafter]) -> str:
    """The option `load` refuses; the `except` clause drops the refusal, and with it any model
    loaded before it, before another load takes the chips."""
    try:
        _ = load()
    except Refused as error:
        return error.option
    raise AssertionError("the draft model loaded")


@pytest.mark.usefixtures("platform_env")
def test_refuses_tokenless(served: Path) -> None:
    """A draft model drafts only beside a decoder that takes token IDs, which an encoder-only
    instance does not load."""
    config = EngineArgs(
        model=str(served),
        dtype="bfloat16",
        skip_tokenizer_init=True,
        speculative_config={
            "method": "draft_model",
            "model": str(served),
            "num_speculative_tokens": 3,
        },
    ).create_engine_config()
    draft = speculative(config, Section.parse(config))
    assert isinstance(draft, Draft)
    assert refusal(lambda: Drafter.load(draft, None)) == "speculative_config.model"


@pytest.fixture(scope="module")
def config() -> VllmConfig:
    """A Qwen3-0.6B target that Qwen3-0.6B drafts for on the target's chip."""
    return EngineArgs(
        model="Qwen/Qwen3-0.6B",
        dtype="bfloat16",
        max_model_len=2048,
        speculative_config={
            "method": "draft_model",
            "model": "Qwen/Qwen3-0.6B",
            "num_speculative_tokens": 3,
        },
    ).create_engine_config()


@pytest.fixture(scope="module")
def draft(config: VllmConfig) -> Draft:
    draft = speculative(config, Section.parse(config))
    assert isinstance(draft, Draft)
    return draft


@pytest.fixture
def target(config: VllmConfig) -> Decoder:
    """The target loaded as the runner loads it, released after each test before another load
    takes the chip."""
    # A failed earlier test's models sit in its traceback's reference cycle until a collection,
    # and they hold the chip this load opens.
    _ = gc.collect()
    return Decoder.load(config)


def test_proposes_greedy_drafts(
    config: VllmConfig, draft: Draft, target: Decoder, greedy: Greedy
) -> None:
    """Drafts proposed together are each request's greedy drafts of its known tokens after rejected
    and accepted suffixes, also when its known tokens take more than one call's rows."""
    model = target.model
    assert isinstance(model, engine.qwen3.ForCausalLM)
    block = model.cache.pools[0].block
    # The runner's requests and batch, which a proposal reads and its sampled tokens extend.
    runner = Runner(config)
    runner.input_batch = InputBatch(
        max_num_reqs=2,
        max_model_len=2048,
        max_num_batched_tokens=32,
        device=torch.device("cpu"),
        vocab_size=151936,
        block_sizes=[block],
        kernel_block_sizes=[block],
        max_num_blocks_per_req=[128],
    )
    drafter = Drafter.load(draft, target)
    drafting = Drafting(drafter, 256, runner.requests)
    match AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B"):
        case PreTrainedTokenizerBase() as tokenizer:
            text = tokenizer.encode(
                "Repeat this exactly: The quick brown fox jumps over the lazy dog.\n"
            )
        case _:
            raise TypeError("tokenizer loading must return a pretrained tokenizer")
    rows = drafter.model.rows
    prompts = {"short": text, "long": (text * (rows // len(text) + 2))[: rows + 3]}
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
    # A step's tokens reject a request's first draft, or accept all of its drafts and add one.
    for rejected in [(), ("short",), ("long",)]:
        sampled = {
            req_id: [(proposed[req_id][0] + 1) % 151936]
            if req_id in rejected
            else [*proposed[req_id], 13]
            for req_id in prompts
        }
        for req_id, tokens in sampled.items():
            runner.extend(runner.requests[req_id], tokens)
        drafts = drafting.propose(runner.input_batch, list(sampled.values()), 3)
        proposed = dict(zip(prompts, drafts, strict=True))
        for req_id, prompt in prompts.items():
            known = [*prompt, *runner.requests[req_id].output_token_ids]
            assert len(proposed[req_id]) == 3, (req_id, proposed[req_id])
            greedy.drafts("Qwen/Qwen3-0.6B", known, proposed[req_id])


def test_refuses_shorter_window(draft: Draft, target: Decoder) -> None:
    """A draft model must hold as long a context as the target serves."""
    assert target.model is not None
    shorter = replace(draft, max_model_len=target.model.window + 1)
    assert refusal(lambda: Drafter.load(shorter, target)) == "speculative_config.model"


def test_loads_its_cache_dtype(draft: Draft, target: Decoder) -> None:
    """A draft model loads with its admitted cache dtype: the engine caches Qwen3-0.6B in BF16
    only and refuses an FP8 cache before it reads any weights."""
    fp8 = replace(draft, cache_dtype="float8_e4m3fn")
    with pytest.raises(NotImplementedError, match="kv_cache"):
        _ = Drafter.load(fp8, target)


def test_refuses_another_vocabulary(draft: Draft, target: Decoder) -> None:
    """A draft model must take the target's token IDs: Qwen3-Embedding-0.6B's 151669-token
    vocabulary is not Qwen3-0.6B's."""
    assert refusal(lambda: Drafter.load(draft, replace(target, vocabulary=151669))) == (
        "speculative_config.model"
    )


@pytest.fixture(scope="module")
def llm() -> Generator[LLM]:
    # The models this module's earlier tests load sit in reference cycles until a collection, and
    # they hold the chip the engine core opens.
    _ = gc.collect()
    llm = LLM(
        model=TARGET,
        dtype="bfloat16",
        max_model_len=2048,
        max_num_seqs=2,
        max_num_batched_tokens=32,
        generation_config="vllm",
        disable_log_stats=False,
        speculative_config={
            "method": "draft_model",
            "model": "Qwen/Qwen3-0.6B",
            "num_speculative_tokens": 3,
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


@pytest.mark.feature("Speculative decoding (ngram, suffix, draft model)")
def test_keeps_greedy(llm: LLM, greedy: Greedy) -> None:
    """Drafts keep the target's greedy tokens, and on copied text the target accepts most."""
    drafted, accepted = counter(llm, DRAFTED), counter(llm, ACCEPTED)
    result = llm.generate(PROMPT, SamplingParams(temperature=0, max_tokens=64))[0]
    assert result.prompt_token_ids is not None
    tokens = list(result.outputs[0].token_ids)
    greedy.continuation(TARGET, list(result.prompt_token_ids), tokens, 64)
    drafted, accepted = counter(llm, DRAFTED) - drafted, counter(llm, ACCEPTED) - accepted
    # Both models copy the same text with one tokenizer, so their greedy tokens mostly agree; a
    # drafter that lost the accepted prefix agrees with the target almost never.
    assert drafted > 0 and accepted >= drafted / 2


def test_sampling_finishes(llm: LLM) -> None:
    """Rejection sampling finishes seeded requests drafted together."""
    _ = llm.generate(PROMPT, SamplingParams(temperature=0, max_tokens=1))
    results = llm.generate(
        [PROMPT, "Describe the water cycle."],
        SamplingParams(temperature=0.7, seed=17, max_tokens=16),
    )
    assert all(result.finished and result.outputs[0].token_ids for result in results)
