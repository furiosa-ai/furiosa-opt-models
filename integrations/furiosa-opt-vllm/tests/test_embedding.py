"""vLLM embeddings equal the checkpoint's normalized last-token hidden state."""

from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    import torch
    from vllm import LLM

TASK = "embedding"
PROMPTS = ["The capital of France is Paris. " * 5, "What is a compiler?"]


@pytest.fixture(scope="module")
def expected(row: dict[str, Any]) -> "torch.Tensor":
    """Transformers' normalized last-token hidden state per prompt."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(row["id"], padding_side="left")
    tokens = tokenizer(PROMPTS, padding=True, return_tensors="pt")
    reference = AutoModel.from_pretrained(row["id"], dtype=torch.bfloat16).eval()
    with torch.inference_mode():
        hidden = reference(**tokens).last_hidden_state[:, -1].float()
    return torch.nn.functional.normalize(hidden, dim=-1)


@pytest.fixture(scope="module")
def llm(options: dict[str, Any]) -> Generator["LLM"]:
    from vllm import LLM

    llm = LLM(
        **options
        | {
            "max_num_seqs": 2,
            # Smaller than the first prompt, so it is prefilled in chunks.
            "max_num_batched_tokens": 8,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
        }
    )
    yield llm
    llm.llm_engine.engine_core.shutdown()


@pytest.mark.feature("Embeddings")
def test_embeddings_match_transformers(llm: "LLM", expected: "torch.Tensor") -> None:
    import torch

    actual = torch.tensor([result.outputs.embedding for result in llm.embed(PROMPTS)])
    torch.testing.assert_close(actual.norm(dim=-1), torch.ones(len(PROMPTS)))
    similarity = torch.nn.functional.cosine_similarity(actual, expected)
    assert torch.all(similarity > 0.999), similarity


def test_prefix_cache(llm: "LLM") -> None:
    import torch

    # A prompt no other test sends, so the first request misses the cache.
    prompts = ["A compiler translates source code. " * 5]
    first = torch.tensor([result.outputs.embedding for result in llm.embed(prompts)])
    repeated = llm.embed(prompts)
    assert repeated[0].num_cached_tokens > 0
    cached = torch.tensor([result.outputs.embedding for result in repeated])
    torch.testing.assert_close(first, cached, atol=0, rtol=0)


def test_dimensions_truncate_and_normalize(llm: "LLM", expected: "torch.Tensor") -> None:
    import torch
    from vllm import PoolingParams

    reduced = torch.tensor(
        [
            result.outputs.embedding
            for result in llm.embed(PROMPTS, pooling_params=PoolingParams(dimensions=128))
        ]
    )
    assert reduced.shape == (len(PROMPTS), 128)
    torch.testing.assert_close(reduced.norm(dim=-1), torch.ones(len(PROMPTS)))
    similarity = torch.nn.functional.cosine_similarity(reduced, expected[:, :128])
    assert torch.all(similarity > 0.999), similarity
