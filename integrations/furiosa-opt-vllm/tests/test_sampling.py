import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, assert_never

import numpy as np
import pytest
import torch
from vllm import SamplingParams
from vllm.config.model import LogprobsMode
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.sample.ops.topk_topp_sampler import random_sample
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.watermarking import create_watermarker
from vllm.v1.watermarking.watermarker import RandomSampler
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from furiosa_opt_vllm.additional import Section
from furiosa_opt_vllm.admission import speculative
from furiosa_opt_vllm.generation import greedy, report, sample
from furiosa_opt_vllm.runner import Logits, Runner
from furiosa_opt_vllm.selection import Sampling, Speculation, accepted, choose


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        (SamplingParams(temperature=0), True),
        (SamplingParams(temperature=1), False),
        (SamplingParams(temperature=0, logprobs=1), False),
        (SamplingParams(temperature=0, prompt_logprobs=1), False),
        (SamplingParams(temperature=0, repetition_penalty=1.1), False),
        (SamplingParams(temperature=0, allowed_token_ids=[1, 2]), False),
        (SamplingParams(temperature=0, logit_bias={1: 1}), False),
        (SamplingParams(temperature=0, trace_decode_token_ids=[1]), False),
        (SamplingParams(temperature=0, prompt_logprob_token_ids=[1]), False),
    ],
)
def test_requires_unmodified(params: SamplingParams, expected: bool) -> None:
    """Only unmodified greedy requests may bypass the host sampler."""
    assert greedy(params) is expected


def admitted(
    req_id: str, prompt: list[int], params: SamplingParams, *, generated: Sequence[int] = ()
) -> CachedRequestState:
    """The runner's state of a request vLLM just scheduled."""
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=prompt,
        mm_features=[],
        sampling_params=params,
        generator=None if params.seed is None else torch.Generator().manual_seed(params.seed),
        block_ids=([0],),
        num_computed_tokens=len(prompt) + len(generated) if generated else 0,
        output_token_ids=list(generated),
    )


def batch(
    requests: Sequence[CachedRequestState],
    drafts: Mapping[str, list[int]] | None = None,
    *,
    vocabulary: int = 8,
) -> InputBatch:
    """vLLM's batch of `requests` in order, each with the `drafts` vLLM schedules for it."""
    batch = InputBatch(
        max_num_reqs=max(len(requests), 2),
        max_model_len=16,
        max_num_batched_tokens=8,
        device=torch.device("cpu"),
        vocab_size=vocabulary,
        block_sizes=[4],
        kernel_block_sizes=[4],
        max_num_blocks_per_req=[4],
        num_spec_tokens=2,
        logitsprocs_need_output_token_ids=True,
    )
    for request in requests:
        _ = batch.add_request(request)
        batch.update_req_spec_token_ids(request, dict(drafts or {}))
    batch.refresh_metadata()
    return batch


def certain(tokens: Sequence[int]) -> torch.Tensor:
    """Logits rows of an 8-token vocabulary, each giving all its probability to its token."""
    logits = torch.full((len(tokens), 8), float("-inf"))
    logits[torch.arange(len(tokens)), torch.tensor(tokens)] = 0
    return logits


@pytest.fixture
def speculation(served: Path) -> Speculation:
    """vLLM's rejection sampler, beside an ngram proposer that these tests leave idle."""
    config = EngineArgs(
        model=str(served),
        dtype="bfloat16",
        skip_tokenizer_init=True,
        max_model_len=16,
        speculative_config={"method": "ngram", "num_speculative_tokens": 2},
    ).create_engine_config()
    return Speculation(RejectionSampler(Sampler()), NgramProposer(config))


@pytest.mark.usefixtures("platform_env")
def test_keeps_drafts(speculation: Speculation) -> None:
    """Rejection keeps matching drafts and replaces the first mismatch with its sampled token."""
    params = SamplingParams(temperature=0.7)
    drafted = batch([admitted("a", [1, 2], params), admitted("b", [3], params)], {"a": [1, 7]})
    # Request a's rows for its two drafts and the token after them, then request b's row.
    drawn, _ = speculation.sample(certain([1, 2, 3, 4]), [3, 1], drafted)
    assert drawn == [[1, 2], [4]]


@pytest.mark.usefixtures("platform_env")
def test_scores_tokens(speculation: Speculation) -> None:
    """Each kept token's logprob is its own, from the row that sampled it."""
    params = SamplingParams(temperature=0.7, logprobs=0)
    drafted = batch([admitted("a", [1, 2], params), admitted("b", [3], params)], {"a": [1, 7]})
    _, logprobs = speculation.sample(certain([1, 2, 3, 4]), [3, 1], drafted)
    assert logprobs is not None
    assert np.array_equal(logprobs.logprob_token_ids, [[1], [2], [4]])
    assert np.array_equal(logprobs.logprobs, [[0.0], [0.0], [0.0]])
    assert logprobs.cu_num_generated_tokens == [0, 2, 3]


@pytest.mark.usefixtures("platform_env")
def test_reports_order(served: Path) -> None:
    """Step output preserves request order and extends only live requests."""
    runner = Runner(
        EngineArgs(
            model=str(served), dtype="bfloat16", skip_tokenizer_init=True
        ).create_engine_config()
    )
    live = admitted("live", [1, 2], SamplingParams(temperature=0))
    live.output_token_ids.append(3)
    runner.input_batch = batch([live], {})
    runner.requests["live"] = live
    output = report(runner, {"live": [7], "gone": [9], "idle": []}, None, None)
    assert output.sampled_token_ids == [[7], [9], []]
    assert live.output_token_ids == [3, 7]


@pytest.mark.usefixtures("platform_env")
@pytest.mark.parametrize("variant", ["plain", "rejection", "dual-key"])
def test_matches_fp64(variant: Literal["plain", "rejection", "dual-key"], served: Path) -> None:
    """Seeded draws match vLLM's float64 exponential-race reference."""
    config = EngineArgs(
        model=str(served),
        dtype="bfloat16",
        skip_tokenizer_init=True,
        use_fp64_gumbel=True,
        speculative_config={"method": "ngram", "num_speculative_tokens": 2}
        if variant == "rejection"
        else None,
        watermark_config={"key": 7, "algorithm": "dual_key_gumbel", "alpha": 0.4419158997424826}
        if variant == "dual-key"
        else None,
    ).create_engine_config()
    from furiosa_opt_vllm.replay import Replay

    params = SamplingParams(temperature=1, seed=0 if variant == "plain" else 1)
    requests = [admitted("a", [1], params, generated=[0])]
    drafted = batch(requests, {})
    selection = choose(config, speculative(config, Section.parse(config)), None, {"a": requests[0]})
    logits = (
        torch.empty(8, dtype=torch.float64)
        .exponential_(generator=torch.Generator().manual_seed(0))
        .log()
        .float()[None]
    )
    if variant == "dual-key":
        logits = torch.arange(8, dtype=torch.float32)[None] / 8
    generator = torch.Generator().manual_seed(params.seed)
    reference = random_sample(logits.softmax(-1), {0: generator}, True)
    if config.watermark_config is not None:

        class Reference(RandomSampler):
            def __call__(self, logits: torch.Tensor) -> torch.Tensor:
                return random_sample(logits.softmax(-1), {0: generator}, True)

        reference = (
            create_watermarker(config.watermark_config)
            .sample(
                logits,
                torch.tensor([[-1, -1, -1, 0]]),
                Reference(torch.tensor([0]), torch.ones(1), torch.zeros(1), torch.zeros(1), True),
            )
            .token_ids
        )
    match selection:
        case Speculation():
            # A zero-probability draft must recover a token, never emit the bonus token.
            logits = torch.tensor(
                [
                    -0.37614334,
                    -0.33671010,
                    -0.91682160,
                    0.23631993,
                    -0.72290897,
                    -2.36353111,
                    -2.07986927,
                    -math.inf,
                ]
            ).repeat(2, 1)
            drafted = batch(requests, {"a": [7]})
            reference = (
                RejectionSampler(Sampler(use_fp64_gumbel=True))
                .forward(
                    SpecDecodeMetadata.make_dummy([[7]], torch.device("cpu")),
                    None,
                    logits.clone(),
                    batch([admitted("a", [1], params)], {"a": [7]}).sampling_metadata,
                )
                .sampled_token_ids
            )
            assert reference[0, 0] != 7 and reference[0, 1] == -1
            reference = reference[:, 0]
            tokens, _ = selection.sample(logits.clone(), [2], drafted)
        case Sampling():
            tokens, _, _ = selection.sample(logits.clone(), drafted, Replay({}, None))
        case unreachable:
            assert_never(unreachable)
    assert tokens == reference[:, None].tolist(), (tokens, reference.tolist())


def test_tokens_match_logits() -> None:
    """Greedy rejection keeps each draft's token while it matches, then the first that differs."""
    # No draft, all drafts accepted, a mismatch, and a placeholder that rejection refuses.
    targets = [[5], [1, 2, 3, 4], [1, 2, 3, 4], [6, 7, 8]]
    drafts = [[], [1, 2, 3], [1, 9, 3], [-1, 7]]
    assert [accepted(t, d) for t, d in zip(targets, drafts, strict=True)] == [
        [5],
        [1, 2, 3, 4],
        [1, 2],
        [6],
    ]


def step(
    checkpoint: Path, requests: Sequence[CachedRequestState], mode: LogprobsMode, masks: bool
) -> Runner:
    """A pending step scores only its first request, with optional V2 sampling masks."""
    runner = Runner(
        EngineArgs(
            model=str(checkpoint),
            dtype="bfloat16",
            skip_tokenizer_init=True,
            logprobs_mode=mode,
            return_sampling_mask=masks,
        ).create_engine_config()
    )
    runner.selection = Sampling(Sampler(logprobs_mode=mode))
    runner.input_batch = batch(requests, {})
    runner.requests = {request.req_id: request for request in requests}
    scores = torch.full((runner.vllm_config.model_config.get_vocab_size(),), -math.inf)
    scores[:8] = torch.arange(8)
    runner.pending = (SchedulerOutput.make_empty(), Logits({requests[0].req_id: [scores]}))
    return runner


@pytest.mark.usefixtures("platform_env")
def test_scores_trace(served: Path) -> None:
    """Trace replay reports the forced token's logprob and rank at its generated position."""
    traced = admitted(
        "traced", [1], SamplingParams(temperature=0, logprobs=1, trace_decode_token_ids=[5, 6])
    )
    traced.output_token_ids.append(5)
    runner = step(served, [traced], "raw_logprobs", masks=False)

    output = sample(runner, None)

    assert isinstance(output, ModelRunnerOutput) and output.logprobs is not None
    distribution = torch.arange(8, dtype=torch.float32).log_softmax(0)
    assert output.sampled_token_ids == [[6]]
    assert output.logprobs.logprob_token_ids[0].tolist() == [6, 7]
    assert output.logprobs.logprobs[0][0] == pytest.approx(float(distribution[6]))
    assert output.logprobs.sampled_token_ranks[0] == int((distribution >= distribution[6]).sum())


@pytest.mark.usefixtures("platform_env")
def test_reports_support(served: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sampling masks report top-k support for sampled requests and nothing for idle requests."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    params = SamplingParams(temperature=1, top_k=3, seed=1)
    drawing, waiting = admitted("drawing", [1], params), admitted("waiting", [1], params)
    runner = step(served, [drawing, waiting], "processed_logprobs", masks=True)

    output = sample(runner, None)

    assert isinstance(output, ModelRunnerOutput) and output.sampling_masks is not None
    assert output.sampling_masks.to_nested_list() == [[5, 6, 7], []]
