"""A separately placed model drafter preserves target decoding."""

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams
from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.metrics.reader import Counter
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

import furiosa_opt_engine as engine
from furiosa_opt_vllm.additional import Section
from furiosa_opt_vllm.admission import Draft, Refused, speculative
from furiosa_opt_vllm.drafting import Drafter, Drafting
from furiosa_opt_vllm.runner import Runner

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
    """A draft model drafts only for a target whose attention cache takes token IDs."""
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
    """A Qwen3-0.6B target that Qwen3-0.6B drafts for from the chips the target leaves free."""
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
def standalone() -> engine.relation.CausalLM[engine.Sequence]:
    """The draft model loaded alone, on the first free chips, as a target loads."""
    parser: type[engine.Model[engine.relation.CausalLM[engine.Sequence]]] = engine.Model
    return engine.LoadOptions().load(parser.from_pretrained(snapshot_download("Qwen/Qwen3-0.6B")))


def test_matches_standalone(
    config: VllmConfig, draft: Draft, standalone: engine.relation.CausalLM[engine.Sequence]
) -> None:
    """Drafts proposed together match each request's standalone continuation after rejected and
    accepted suffixes, also when its known tokens take more than one call's rows."""
    body = standalone.scores.body
    # The runner's requests and batch, which a proposal reads and its sampled tokens extend.
    runner = Runner(config)
    runner.input_batch = InputBatch(
        max_num_reqs=2,
        max_model_len=2048,
        max_num_batched_tokens=32,
        device=torch.device("cpu"),
        vocab_size=151936,
        block_sizes=[body.cache.block.tokens],
        kernel_block_sizes=[body.cache.block.tokens],
        max_num_blocks_per_req=[128],
    )
    drafting = Drafting(Drafter.load(draft, body.cache), 256, runner.requests)
    match AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B"):
        case PreTrainedTokenizerBase() as tokenizer:
            text = tokenizer.encode(
                "Repeat this exactly: The quick brown fox jumps over the lazy dog.\n"
            )
        case _:
            raise TypeError("tokenizer loading must return a pretrained tokenizer")
    prompts = {
        "short": text,
        "long": (text * (body.cache.rows // len(text) + 2))[: body.cache.rows + 3],
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

    def continuation(prefix: list[int]) -> list[int]:
        cache = engine.Cache(body.cache, 128)
        blocks = list(range(128))
        output: engine.HbmTensor | None = None
        for position, token in enumerate(prefix):
            output = standalone.forward(
                [engine.Request(blocks, position, engine.HostTensor((token,)).to_hbm(body.device))],
                cache,
            )
        assert output is not None, "a reference prefix holds a token"
        tokens: list[int] = []
        for offset in range(3):
            tokens.append(int(output.to_host().numpy()[0]))
            if offset < 2:
                output = standalone.forward(
                    [engine.Request(blocks, len(prefix) + offset, output)], cache
                )
        return tokens

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
            assert proposed[req_id] == continuation(known), (rejected, req_id)


def test_refuses_shorter_window(
    draft: Draft, standalone: engine.relation.CausalLM[engine.Sequence]
) -> None:
    """A draft model must hold as long a context as the target serves."""
    cache = standalone.scores.body.cache
    shorter = replace(draft, max_model_len=cache.window + 1)
    assert refusal(lambda: Drafter.load(shorter, cache)) == "speculative_config.model"


def test_loads_its_cache_dtype(
    draft: Draft, standalone: engine.relation.CausalLM[engine.Sequence]
) -> None:
    """A draft model loads with its admitted cache dtype: the engine compiles Qwen3-0.6B without an
    FP8 cache and refuses one before it reads any weights."""
    fp8 = replace(draft, cache_dtype="float8_e4m3fn")
    with pytest.raises(NotImplementedError, match="cache dtype"):
        _ = Drafter.load(fp8, standalone.scores.body.cache)


def test_refuses_another_vocabulary(draft: Draft) -> None:
    """A draft model must take the target's token IDs: Qwen3-Embedding-0.6B's 151669-token
    vocabulary is not Qwen3-0.6B's."""
    parser: type[engine.Model[engine.Sequence]] = engine.Model
    target = (
        engine.LoadOptions()
        .drop("head")
        .load(parser.from_pretrained(snapshot_download("Qwen/Qwen3-Embedding-0.6B")))
    )
    assert refusal(lambda: Drafter.load(draft, target.cache)) == "speculative_config.model"


@pytest.fixture(scope="module", name="_baseline")
def baseline() -> list[int]:
    """The target's greedy tokens without drafts, from an engine that shuts down before `llm`'s
    starts on the same chips."""
    llm = LLM(
        model="Qwen/Qwen3-1.7B",
        dtype="bfloat16",
        max_model_len=2048,
        max_num_seqs=2,
        max_num_batched_tokens=32,
        generation_config="vllm",
    )
    try:
        return list(
            llm.generate(PROMPT, SamplingParams(temperature=0, max_tokens=64))[0]
            .outputs[0]
            .token_ids
        )
    finally:
        llm.llm_engine.engine_core.shutdown()


@pytest.fixture(scope="module")
def llm(_baseline: list[int]) -> Generator[LLM]:
    llm = LLM(
        model="Qwen/Qwen3-1.7B",
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
def test_keeps_greedy(llm: LLM, _baseline: list[int]) -> None:
    """Drafts keep the target's greedy tokens, and on copied text the target accepts most."""
    drafted, accepted = counter(llm, DRAFTED), counter(llm, ACCEPTED)
    result = llm.generate(PROMPT, SamplingParams(temperature=0, max_tokens=64))[0]
    assert list(result.outputs[0].token_ids) == _baseline
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
