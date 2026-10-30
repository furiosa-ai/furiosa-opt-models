"""vLLM classifications, scores and rewards equal the checkpoint's score head on its last hidden
row, unchanged by an unrelated request, a chunk boundary or a prefix-cache hit."""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING, Final

import pytest
from conftest import Reference, post

if TYPE_CHECKING:
    import torch
    from conftest import Options, Row
    from vllm import LLM
    from vllm.entrypoints.pooling.scoring.typing import ScoreInput

TASK = "classification"
PROMPTS: Final = [
    "Paris is the capital of France, and its largest city. " * 3,
    "What is a compiler?",
]
"""The first prompt is longer than a scheduler step, so it is prefilled in chunks."""
QUERY: Final = "What is the capital of France?"
DOCUMENTS: Final[list[ScoreInput]] = [
    "Tokyo is the capital of Japan.",
    "Paris is the capital of France.",
]


@pytest.fixture(scope="module")
def reference(row: Row) -> Reference:
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(row["id"]).float().eval()
    with safe_open(hf_hub_download(row["id"], "model.safetensors"), "pt") as tensors:
        weight = tensors.get_tensor("score.weight").float()
    return Reference(model, weight)


@pytest.fixture(scope="module")
def llm(options: Options) -> Generator[LLM]:
    from vllm import LLM

    llm = LLM(
        **options,
        max_num_seqs=2,
        # Smaller than the first prompt, so it is prefilled in chunks.
        max_num_batched_tokens=16,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
    )
    yield llm
    llm.llm_engine.engine_core.shutdown()


def raw(llm: LLM, prompts: list[str]) -> tuple[torch.Tensor, list[list[int]]]:
    """Each prompt's unactivated sequence scores and the token IDs vLLM scored."""
    import torch
    from vllm import PoolingParams

    outputs = llm.encode(
        prompts, PoolingParams(use_activation=False), pooling_task="classify", use_tqdm=False
    )
    return torch.stack([output.outputs.data for output in outputs]), [
        output.prompt_token_ids for output in outputs
    ]


@pytest.mark.feature("Classification and rewards from a score head")
def test_matches_transformers(llm: LLM, reference: Reference) -> None:
    import torch

    actual, ids = raw(llm, PROMPTS)
    for side in ("right", "left"):
        rows = torch.stack([rows[-1] for rows in reference.rows(ids, side)])
        reference.check(actual, rows)


def test_activates_scores(llm: LLM) -> None:
    import torch

    scores, _ = raw(llm, PROMPTS)
    probs = torch.tensor([output.outputs.probs for output in llm.classify(PROMPTS)])
    # One label: vLLM's classifier activation is a sigmoid of the label's own score.
    torch.testing.assert_close(probs, torch.sigmoid(scores))


def test_ignores_batching(llm: LLM) -> None:
    import torch
    from vllm import PoolingParams

    together, _ = raw(llm, PROMPTS)
    alone = torch.cat([raw(llm, [prompt])[0] for prompt in PROMPTS])
    torch.testing.assert_close(alone, together, atol=0, rtol=0)
    # A prompt no other test sends, so the first request misses the cache.
    prompt = "A compiler translates source code into machine code. " * 3
    first, _ = raw(llm, [prompt])
    repeated = llm.encode(
        [prompt], PoolingParams(use_activation=False), pooling_task="classify", use_tqdm=False
    )
    assert repeated[0].num_cached_tokens > 0
    torch.testing.assert_close(repeated[0].outputs.data[None], first, atol=0, rtol=0)


def test_scores_pairs(llm: LLM, reference: Reference) -> None:
    import torch
    from vllm import PoolingParams

    scored = llm.score(QUERY, DOCUMENTS, use_tqdm=False)
    unactivated = llm.score(
        QUERY, DOCUMENTS, pooling_params=PoolingParams(use_activation=False), use_tqdm=False
    )
    actual = torch.tensor([[result.outputs.score] for result in unactivated])
    pairs = reference.rows([result.prompt_token_ids for result in unactivated])
    reference.check(actual, torch.stack([rows[-1] for rows in pairs]))
    activated = torch.tensor([[result.outputs.score] for result in scored])
    torch.testing.assert_close(activated, torch.sigmoid(actual))


def test_matches_offline(llm: LLM, server: str, row: Row) -> None:
    import torch
    from vllm.entrypoints.pooling.classify.protocol import (
        ClassificationCompletionRequest,
        ClassificationResponse,
    )
    from vllm.entrypoints.pooling.scoring.protocol import (
        RerankRequest,
        RerankResponse,
        ScoreResponse,
        ScoreTextRequest,
    )

    model = row["id"]
    scores, _ = raw(llm, PROMPTS)
    classify = ClassificationCompletionRequest(model=model, input=PROMPTS)
    classified = post(f"{server}/classify", classify, ClassificationResponse).data
    assert [(item.label, item.num_classes) for item in classified] == [("LABEL_0", 1)] * 2
    probs = torch.tensor([item.probs for item in classified])
    torch.testing.assert_close(probs, torch.sigmoid(scores))
    raw_request = ClassificationCompletionRequest(model=model, input=PROMPTS, use_activation=False)
    unactivated = post(f"{server}/classify", raw_request, ClassificationResponse).data
    torch.testing.assert_close(torch.tensor([item.probs for item in unactivated]), scores)
    pairs = [result.outputs.score for result in llm.score(QUERY, DOCUMENTS, use_tqdm=False)]
    score = ScoreTextRequest(model=model, text_1=QUERY, text_2=DOCUMENTS)
    scored = post(f"{server}/score", score, ScoreResponse).data
    torch.testing.assert_close(torch.tensor([item.score for item in scored]), torch.tensor(pairs))
    rerank = RerankRequest(model=model, query=QUERY, documents=DOCUMENTS, top_n=1)
    (best,) = post(f"{server}/rerank", rerank, RerankResponse).results
    assert best.index == max(range(len(pairs)), key=pairs.__getitem__)
    torch.testing.assert_close(best.relevance_score, max(pairs))
