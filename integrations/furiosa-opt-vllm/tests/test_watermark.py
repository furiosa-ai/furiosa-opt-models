"""Host sampling draws a watermarked request's tokens with vLLM's watermarker, beside trace replay
and sampling masks."""

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal, Required, TypedDict

import numpy as np
import pytest
import torch
from test_sampling import admitted, batch
from vllm import SamplingParams
from vllm.config import VllmConfig
from vllm.config.model import LogprobsMode
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.outputs import LogprobsLists
from vllm.v1.sample.sampler import Sampler
from vllm.v1.watermarking import DualKeyGumbelWatermarker, create_watermarker
from vllm.v1.worker.gpu_input_batch import CachedRequestState

from furiosa_opt_vllm.replay import Replay
from furiosa_opt_vllm.selection import Sampling, choose

VOCAB: Final = 32
WIDTH: Final = 2


class Watermark(TypedDict, total=False):
    """vLLM's `--watermark-config` settings a case sets."""

    key: Required[int]
    algorithm: Literal["gumbel", "dual_key_gumbel"]
    alpha: float
    context_width: int
    deduplicate_contexts: Literal["none", "single_turn", "all"]


GUMBEL: Final = Watermark(key=7, context_width=WIDTH)


def configured(
    checkpoint: Path, watermark: Watermark, mode: LogprobsMode = "raw_logprobs"
) -> VllmConfig:
    """vLLM's configuration of the checkpoint in `checkpoint`, served with `watermark`."""
    return EngineArgs(
        model=str(checkpoint),
        dtype="bfloat16",
        skip_tokenizer_init=True,
        logprobs_mode=mode,
        watermark_config={**watermark},
    ).create_engine_config()


def scores(rows: int) -> torch.Tensor:
    """Fixed logits of `rows` rows over the vocabulary."""
    return torch.randn(rows, VOCAB, generator=torch.Generator().manual_seed(0))


def selected(config: VllmConfig, requests: Sequence[CachedRequestState]) -> Sampling:
    """The runner's selection for `config`, which serves `requests` without speculation."""
    sampling = choose(config, None, None, {request.req_id: request for request in requests})
    assert isinstance(sampling, Sampling)
    return sampling


def drawn(
    config: VllmConfig, requests: Sequence[CachedRequestState], logits: torch.Tensor
) -> tuple[list[int], LogprobsLists | None]:
    """The token and logprobs the runner's selection for `config` draws for each of `requests` from
    its row of `logits`."""
    replay = Replay.of(requests, set(range(len(requests))), masks=False)
    tokens, logprobs, _ = selected(config, requests).sample(
        logits.clone(), batch(requests, vocabulary=VOCAB), replay
    )
    return [token for (token,) in tokens], logprobs


def unwatermarked(
    requests: Sequence[CachedRequestState], logits: torch.Tensor, mode: LogprobsMode
) -> tuple[list[int], LogprobsLists | None]:
    """The token and logprobs vLLM's sampler draws for each of `requests` without a watermark."""
    replay = Replay.of(requests, set(range(len(requests))), masks=False)
    tokens, logprobs, _ = Sampling(Sampler(logprobs_mode=mode)).sample(
        logits.clone(), batch(requests, vocabulary=VOCAB), replay
    )
    return [token for (token,) in tokens], logprobs


@pytest.mark.usefixtures("platform_env")
@pytest.mark.parametrize(
    "watermark",
    [
        GUMBEL,
        Watermark(key=7, context_width=WIDTH, algorithm="dual_key_gumbel", alpha=0.0),
        Watermark(key=7, context_width=WIDTH, algorithm="dual_key_gumbel", alpha=1.0),
    ],
    ids=["gumbel", "key-a", "key-b"],
)
def test_matches_watermark(watermark: Watermark, served: Path) -> None:
    """Watermarked tokens match vLLM for temperature-scaled logits and padded output contexts."""
    config = configured(served, watermark)
    temperatures = [0.3, 2.0, 0.8, 1.5] * 2
    generated = [[5, 9, 11], [], [4], [6, 2, 8], [1], [3, 3], [], [7, 0, 2, 5]]
    requests = [
        admitted(
            f"r{index}",
            [1, 2],
            SamplingParams(temperature=temperature, seed=index),
            generated=tokens,
        )
        for index, (tokens, temperature) in enumerate(zip(generated, temperatures, strict=True))
    ]
    logits = scores(len(requests))
    contexts = [[9, 11], [-1, -1], [-1, 4], [2, 8], [-1, 1], [3, 3], [-1, -1], [2, 5]]
    assert config.watermark_config is not None
    expected = create_watermarker(config.watermark_config).sample(
        logits / torch.tensor(temperatures).unsqueeze(-1), torch.tensor(contexts)
    )
    assert torch.equal(torch.tensor(drawn(config, requests, logits)[0]), expected.token_ids)


@pytest.mark.usefixtures("platform_env")
@pytest.mark.parametrize("mode", ["raw_logprobs", "processed_logprobs", "processed_logits"])
def test_honors_optout(mode: LogprobsMode, served: Path) -> None:
    """Opted-out requests match unwatermarked vLLM tokens and scores in the chosen mode."""

    def requests() -> list[CachedRequestState]:
        return [
            admitted(
                "a",
                [1, 2],
                SamplingParams(temperature=0.8, seed=3, logprobs=2, watermarking=False),
                generated=[5, 9],
            ),
            admitted(
                "b",
                [1, 2],
                SamplingParams(temperature=1.0, seed=4, logprobs=2, watermarking=False),
                generated=[4],
            ),
        ]

    logits = scores(2)
    tokens, logprobs = drawn(configured(served, GUMBEL, mode), requests(), logits)
    expected, reference = unwatermarked(requests(), logits, mode)
    assert tokens == expected
    assert logprobs is not None and reference is not None
    assert np.array_equal(logprobs.logprob_token_ids, reference.logprob_token_ids)
    assert np.array_equal(logprobs.logprobs, reference.logprobs)


@pytest.mark.usefixtures("platform_env")
def test_watermarks_eligible(served: Path) -> None:
    """Mixed batches watermark eligible rows while preserving greedy, opt-out and trace tokens."""
    config = configured(served, GUMBEL)

    def requests() -> list[CachedRequestState]:
        traced = SamplingParams(temperature=0.8, seed=5, logprobs=0, trace_decode_token_ids=[3, 30])
        return [
            admitted(
                "marked",
                [1, 2],
                SamplingParams(temperature=0.8, seed=3, logprobs=0),
                generated=[5, 9],
            ),
            admitted(
                "out",
                [1, 2],
                SamplingParams(temperature=0.8, seed=4, watermarking=False),
                generated=[6],
            ),
            admitted("greedy", [1, 2], SamplingParams(temperature=0), generated=[7]),
            admitted("traced", [1, 2], traced, generated=[3]),
        ]

    logits = scores(4)
    tokens, logprobs = drawn(config, requests(), logits)
    plain, _ = unwatermarked(requests(), logits, "raw_logprobs")
    assert config.watermark_config is not None
    marked = create_watermarker(config.watermark_config).sample(
        logits[:1] / 0.8, torch.tensor([[5, 9]])
    )
    assert tokens == [marked.token_ids.item(), plain[1], logits[2].argmax().item(), 30]
    assert logprobs is not None
    assert logprobs.logprob_token_ids[[0, 3], 0].tolist() == [tokens[0], 30]


@pytest.mark.usefixtures("platform_env")
def test_keeps_support(served: Path) -> None:
    """Watermarking changes token selection without changing the processed sampling support."""
    config = configured(served, GUMBEL, "processed_logprobs")
    requests = [
        admitted(
            f"r{index}",
            [1, 2],
            SamplingParams(temperature=1.0, top_k=3, seed=index),
            generated=[index, 9],
        )
        for index in range(4)
    ]
    logits = scores(4)
    replay = Replay.of(requests, set(range(4)), masks=True)
    tokens, _, masks = selected(config, requests).sample(
        logits.clone(), batch(requests, vocabulary=VOCAB), replay
    )
    kept = logits.topk(3).indices.sort().values
    assert masks is not None and masks.to_nested_list() == kept.numpy().tolist()
    assert config.watermark_config is not None
    masked = torch.full_like(logits, -math.inf).scatter(1, kept, logits.gather(1, kept))
    contexts = torch.tensor([[index, 9] for index in range(4)])
    marked = create_watermarker(config.watermark_config).sample(masked, contexts)
    assert tokens == [[int(token)] for token in marked.token_ids]


@pytest.mark.usefixtures("platform_env")
@pytest.mark.parametrize(
    ("scope", "watermarked"),
    [
        ("none", [True, True, True, True]),
        ("single_turn", [False, True, True, True]),
        ("all", [False, False, False, True]),
    ],
    ids=["none", "single_turn", "all"],
)
def test_skips_context(
    scope: Literal["none", "single_turn", "all"], watermarked: list[bool], served: Path
) -> None:
    """Repeated or incomplete contexts fall back to vLLM sampling under the chosen dedup scope."""
    config = configured(served, Watermark(key=7, context_width=WIDTH, deduplicate_contexts=scope))

    def requests() -> list[CachedRequestState]:
        return [
            # Its context [3, 4] preceded its third token.
            admitted(
                "output", [1, 2], SamplingParams(temperature=1.0, seed=0), generated=[3, 4, 5, 3, 4]
            ),
            # Its context [1, 2] is its prompt.
            admitted(
                "prompt", [1, 2], SamplingParams(temperature=1.0, seed=1), generated=[3, 1, 2]
            ),
            admitted("short", [1, 2], SamplingParams(temperature=1.0, seed=2), generated=[5]),
            admitted("fresh", [1, 2], SamplingParams(temperature=1.0, seed=3), generated=[7, 8, 9]),
        ]

    logits = scores(4)
    tokens, _ = drawn(config, requests(), logits)
    plain, _ = unwatermarked(requests(), logits, "raw_logprobs")
    assert config.watermark_config is not None
    marks = create_watermarker(config.watermark_config).sample(
        logits, torch.tensor([[3, 4], [1, 2], [-1, 5], [8, 9]])
    )
    assert tokens == [
        int(mark) if keep else ordinary
        for keep, mark, ordinary in zip(watermarked, marks.token_ids, plain, strict=True)
    ]


@pytest.mark.usefixtures("platform_env")
def test_draws_keys(served: Path) -> None:
    """Dual-key draws select both upstream watermark keys and repeat under the same seed."""
    watermark = Watermark(key=7, context_width=WIDTH, algorithm="dual_key_gumbel", alpha=0.5)
    config = configured(served, watermark)

    def requests() -> list[CachedRequestState]:
        return [
            admitted(
                f"r{index}", [1, 2], SamplingParams(temperature=1.0, seed=index), generated=[index]
            )
            for index in range(8)
        ]

    logits = scores(8)
    tokens = torch.tensor(drawn(config, requests(), logits)[0])
    keys = DualKeyGumbelWatermarker(7, WIDTH, alpha=0.5)
    contexts = torch.tensor([[-1, index] for index in range(8)])
    first = keys.draft_watermarker.sample(logits, contexts).token_ids
    second = keys.target_watermarker.sample(logits, contexts).token_ids
    assert ((tokens == first) | (tokens == second)).all()
    assert ((tokens == first) & (first != second)).any()
    assert ((tokens == second) & (first != second)).any()
    assert torch.equal(torch.tensor(drawn(config, requests(), logits)[0]), tokens)
