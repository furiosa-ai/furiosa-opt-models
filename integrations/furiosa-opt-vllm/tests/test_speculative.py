"""Ngram and suffix drafts keep greedy tokens, and suffix decoding drafts repeats from responses."""

from collections.abc import Generator
from typing import TYPE_CHECKING, Final, Literal, TypedDict

import pytest
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.sampling_params import RequestOutputKind
from vllm.v1.metrics.reader import Counter

if TYPE_CHECKING:
    from conftest import Options, Row
    from transformers import PreTrainedTokenizerBase

TASK = "text"
TEXT = "The quick brown fox jumps over the lazy dog. " * 8
DRAFTED = "vllm:spec_decode_num_draft_tokens"
ACCEPTED = "vllm:spec_decode_num_accepted_tokens"


class Ngram(TypedDict):
    method: Literal["ngram"]
    num_speculative_tokens: int
    prompt_lookup_min: int
    prompt_lookup_max: int


class Suffix(TypedDict):
    method: Literal["suffix"]
    num_speculative_tokens: int


NGRAM: Final = Ngram(
    method="ngram", num_speculative_tokens=3, prompt_lookup_min=1, prompt_lookup_max=3
)
SUFFIX: Final = Suffix(method="suffix", num_speculative_tokens=3)

# The tests of one drafter share the engine `llm` starts for it.
drafters = pytest.mark.parametrize(
    "drafter", [NGRAM, SUFFIX], ids=["ngram", "suffix"], scope="module"
)


@pytest.fixture(scope="module")
def tokenizer(row: "Row") -> "PreTrainedTokenizerBase":
    from transformers import AutoTokenizer, PreTrainedTokenizerBase

    match AutoTokenizer.from_pretrained(row["id"]):
        case PreTrainedTokenizerBase() as tokenizer:
            return tokenizer
        case _:
            raise TypeError("tokenizer loading must return a pretrained tokenizer")


def render(tokenizer: "PreTrainedTokenizerBase", text: str, cue: str) -> str:
    """The chat prompt for one user message; without a chat template, `text` and then `cue`."""
    if not tokenizer.chat_template:
        return f"{text}\n{cue}\n"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    match rendered:
        case str():
            return rendered
        case _:
            raise TypeError("untokenized chat rendering must return text")


@pytest.fixture(scope="module")
def prompt(tokenizer: "PreTrainedTokenizerBase") -> str:
    return render(
        tokenizer, f"Copy the following text exactly, without any explanation:\n{TEXT}", "Copy:"
    )


@pytest.fixture(scope="module", name="_baseline")
def baseline(options: "Options", prompt: str) -> list[int]:
    """The prompt's greedy tokens without drafts, from an engine that shuts down before `llm`'s
    starts on the same chip."""
    llm = LLM(**options, max_num_seqs=2, max_num_batched_tokens=32)
    try:
        return list(
            llm.generate(prompt, SamplingParams(temperature=0, max_tokens=64))[0]
            .outputs[0]
            .token_ids
        )
    finally:
        llm.llm_engine.engine_core.shutdown()


@pytest.fixture(scope="module")
def llm(options: "Options", drafter: Ngram | Suffix, _baseline: list[int]) -> Generator[LLM]:
    llm = LLM(
        **options,
        max_num_seqs=2,
        max_num_batched_tokens=32,
        disable_log_stats=False,
        speculative_config=drafter,
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


@drafters
@pytest.mark.feature("Speculative decoding (ngram, suffix, draft model)")
def test_keeps_greedy(llm: LLM, prompt: str, _baseline: list[int]) -> None:
    """Accepted drafts leave the greedy output unchanged."""
    before = counter(llm, ACCEPTED)
    result = llm.generate(prompt, SamplingParams(temperature=0, max_tokens=64))[0]
    assert list(result.outputs[0].token_ids) == _baseline
    assert counter(llm, ACCEPTED) > before


@drafters
def test_sampling_finishes(llm: LLM, prompt: str) -> None:
    """Random rejection sampling completes a seeded request with a cached prompt."""
    _ = llm.generate(prompt, SamplingParams(temperature=0, max_tokens=1))
    result = llm.generate(prompt, SamplingParams(temperature=0.7, seed=17, max_tokens=16))[0]
    assert result.finished and result.outputs[0].token_ids


@pytest.mark.parametrize("drafter", [SUFFIX], ids=["suffix"], scope="module")
def test_suffix_reuses_responses(llm: LLM, tokenizer: "PreTrainedTokenizerBase") -> None:
    """A repeated request drafts from the response cached for the first while it follows it."""
    # Its answer is not in the question, so the prompt cannot draft the repeat.
    question = render(tokenizer, "List ten facts about the planet Mars, one per line.", "Facts:")
    first = llm.generate(question, SamplingParams(temperature=0, max_tokens=64))[0]
    cached = list(first.outputs[0].token_ids)
    engine = llm.llm_engine
    _ = engine.add_request(
        "repeat",
        TokensPrompt(prompt_token_ids=first.prompt_token_ids or []),
        SamplingParams(temperature=0, max_tokens=64, output_kind=RequestOutputKind.CUMULATIVE),
    )
    drafted = accepted = 0
    while engine.has_unfinished_requests():
        counts = counter(llm, DRAFTED), counter(llm, ACCEPTED)
        # The repeat's verifying calls hold other rows than the first's, which can resolve a near
        # tie otherwise; a step counts while every token it emits follows the cached response.
        if all(
            list(report.outputs[0].token_ids) == cached[: len(report.outputs[0].token_ids)]
            for report in engine.step()
        ):
            drafted += counter(llm, DRAFTED) - counts[0]
            accepted += counter(llm, ACCEPTED) - counts[1]
    # A draft misses only after a recurring phrase.
    assert 2 * accepted > drafted
