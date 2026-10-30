"""Resumed requests preserve tokens; call-dependent rounding permits top-five agreement."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final, Literal

import pytest
from transformers import PreTrainedTokenizerBase
from vllm import LLM, PoolingRequestOutput, RequestOutput, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.sampling_params import RequestOutputKind
from vllm.v1.metrics.reader import Counter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vllm.logprobs import SampleLogprobs

CHECKPOINT: Final = "Qwen/Qwen3-0.6B"
CONTEXT: Final = 512
TOPICS: Final = ("the ocean", "volcanoes", "the history of Rome", "how compilers work")
# Each prompt and its answer span most of a context.
ANSWER: Final = 160
PREEMPTED: Final = "vllm:num_preemptions"
# vLLM selects V1 for ngram and suffix decoding and on a host without Triton, V2 otherwise.
RUNNERS: Final = pytest.mark.parametrize("runner", ["0", "1"], ids=["v1", "v2"])
TOP: Final = 5


def start(*, blocks: int | None = None, budget: int = 64) -> LLM:
    """An engine that runs the four requests at once and schedules `budget` tokens a step, with
    `blocks` cache blocks if set."""
    return LLM(
        CHECKPOINT,
        dtype="bfloat16",
        generation_config="vllm",
        max_model_len=CONTEXT,
        max_num_seqs=len(TOPICS),
        max_num_batched_tokens=budget,
        num_gpu_blocks_override=blocks,
        disable_log_stats=False,
    )


def greedy(kind: RequestOutputKind = RequestOutputKind.FINAL_ONLY) -> SamplingParams:
    """Greedy decoding of a full answer, reported as `kind` says."""
    return SamplingParams(temperature=0, max_tokens=ANSWER, ignore_eos=True, output_kind=kind)


def counter(llm: LLM, name: str) -> float:
    """vLLM's counter `name`, summed over its labels."""
    return sum(
        metric.value
        for metric in llm.get_metrics()
        if isinstance(metric, Counter) and metric.name == name
    )


def answers(reports: Sequence[RequestOutput | PoolingRequestOutput]) -> dict[str, list[int]]:
    """The tokens of each request that `reports` finish."""
    finished: dict[str, list[int]] = {}
    for report in reports:
        match report:
            case RequestOutput(finished=True, outputs=[answer]):
                finished[report.request_id] = list(answer.token_ids)
            case RequestOutput():
                pass
            case PoolingRequestOutput():
                raise TypeError("a generating engine reports no pooled output")
    return finished


@pytest.fixture(scope="module")
def prompts(tokenizer: PreTrainedTokenizerBase) -> list[list[int]]:
    context = "Answer in detail and do not stop early. " * 12
    prompts: list[list[int]] = []
    for topic in TOPICS:
        match tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{context}Write an essay about {topic}."}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ):
            case str() as text:
                prompts.append(tokenizer.encode(text, add_special_tokens=False))
            case _:
                raise TypeError("untokenized chat rendering must return text")
    return prompts


def generate(llm: LLM, prompts: list[list[int]]) -> list[list[int]]:
    """Each prompt's greedy answer from `llm`."""
    requests = [TokensPrompt(prompt_token_ids=prompt) for prompt in prompts]
    return [list(output.outputs[0].token_ids) for output in llm.generate(requests, greedy())]


def assert_close(
    llm: LLM, prompts: list[list[int]], answers: list[list[int]], expected: list[list[int]]
) -> None:
    """As vLLM's logprobs-close check: at the first difference, and at every later token on its own
    prefix, the token lies among the `TOP` most likely."""
    for prompt, answer, reference in zip(prompts, answers, expected, strict=True):
        assert len(answer) == len(reference)
        differing = (
            index
            for index, (token, other) in enumerate(zip(answer, reference, strict=True))
            if token != other
        )
        if (index := next(differing, None)) is None:
            continue
        shared = TokensPrompt(prompt_token_ids=prompt + reference[:index])
        scored = llm.generate(shared, SamplingParams(temperature=0, max_tokens=1, logprobs=TOP))
        logprobs = scored[0].outputs[0].logprobs
        assert logprobs is not None
        assert {answer[index], reference[index]} <= logprobs[0].keys()
        scored = llm.generate(
            TokensPrompt(prompt_token_ids=prompt + answer),
            SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=TOP),
        )[0]
        assert scored.prompt_logprobs is not None
        assert_greedy(answer[index:], scored.prompt_logprobs[len(prompt) + index :])


def assert_greedy(tokens: Sequence[int], logprobs: SampleLogprobs) -> None:
    for token, scores in zip(tokens, logprobs, strict=True):
        assert scores is not None and scores[token].rank <= TOP


@pytest.mark.parametrize("late", [False, True], ids=["large-gap", "after-tie"])
def test_rejects_errors(late: bool) -> None:
    """The top-five check rejects out-of-support tokens before or after a near tie."""
    from vllm.logprobs import Logprob

    near = {1: Logprob(math.log(0.5), rank=1), 2: Logprob(math.log(0.48), rank=2)}
    wrong = {1: Logprob(math.log(0.9), rank=1), 2: Logprob(math.log(0.001), rank=TOP + 1)}
    with pytest.raises(AssertionError):
        assert_greedy([2, 2] if late else [2], [near, wrong] if late else [wrong])


@RUNNERS
def test_keeps_preempted(
    runner: Literal["0", "1"], prompts: list[list[int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requests preempted for cache blocks resume with their tokens and keep their greedy answer."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", runner)
    llm = start()
    try:
        expected = generate(llm, prompts)
        block = llm.llm_engine.vllm_config.cache_config.block_size
        assert counter(llm, PREEMPTED) == 0
    finally:
        llm.llm_engine.engine_core.shutdown()
    # One full context and a quarter more holds about two of the four requests.
    llm = start(blocks=CONTEXT * 5 // 4 // block)
    try:
        resumed = generate(llm, prompts)
        assert counter(llm, PREEMPTED) > 0
        assert_close(llm, prompts, resumed, expected)
    finally:
        llm.llm_engine.engine_core.shutdown()


@RUNNERS
def test_keeps_reset(
    runner: Literal["0", "1"], prompts: list[list[int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prefix-cache reset resumes in-flight requests with tokens satisfying the top-five check."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", runner)
    # A step holds every request's whole context, so the resumed requests recompute in one.
    llm = start(budget=CONTEXT * len(TOPICS))
    try:
        expected = generate(llm, prompts)
        engine = llm.llm_engine
        for index, prompt in enumerate(prompts):
            # Reported at every step, so that the reset follows a request's first token.
            stepwise = greedy(RequestOutputKind.CUMULATIVE)
            _ = engine.add_request(str(index), TokensPrompt(prompt_token_ids=prompt), stepwise)
        reports: list[RequestOutput | PoolingRequestOutput] = []
        while not reports:
            reports = engine.step()
        assert llm.reset_prefix_cache(reset_running_requests=True)
        while engine.has_unfinished_requests():
            reports.extend(engine.step())
        assert counter(llm, PREEMPTED) > 0
        finished = answers(reports)
        resumed = [finished[str(index)] for index in range(len(prompts))]
        assert_close(llm, prompts, resumed, expected)
    finally:
        llm.llm_engine.engine_core.shutdown()
