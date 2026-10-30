"""vLLM rerank scores equal the checkpoint's yes probability after the yes and no tokens."""

from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from vllm import LLM

TASK = "reranking"
QUERY = "What is the capital of France?"
DOCUMENTS = ["Paris is the capital of France.", "Tokyo is the capital of Japan."]


@pytest.fixture(scope="module")
def llm(options: dict[str, Any]) -> Generator["LLM"]:
    from vllm import LLM

    llm = LLM(
        **options
        | {
            "max_num_seqs": 2,
            # Smaller than each query and document, so both are prefilled in chunks.
            "max_num_batched_tokens": 16,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
        }
    )
    yield llm
    llm.llm_engine.engine_core.shutdown()


def score(llm: "LLM", query: str, documents: list[str]) -> list[float]:
    """Scores as `vllm serve` returns them; the server applies `--chat-template` to each pair."""
    results = llm.score(query, documents, chat_template=llm.chat_template)
    return [result.outputs.score for result in results]


@pytest.mark.feature("Reranking")
def test_reranking_matches_transformers(row: dict[str, Any], llm: "LLM") -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(row["id"])
    reference = AutoModelForCausalLM.from_pretrained(row["id"], dtype=torch.bfloat16).eval()
    expected = []
    with torch.inference_mode():
        for document in DOCUMENTS:
            tokens = tokenizer.apply_chat_template(
                [{"role": "query", "content": QUERY}, {"role": "document", "content": document}],
                return_tensors="pt",
                return_dict=False,
            )
            logits = reference(tokens).logits[0, -1].float()
            expected.append(logits[tokenizer.convert_tokens_to_ids(["no", "yes"])].softmax(-1)[1])
    actual = torch.tensor(score(llm, QUERY, DOCUMENTS))
    torch.testing.assert_close(actual, torch.stack(expected), atol=0.01, rtol=0.02)


def test_prefix_cache(llm: "LLM") -> None:
    # A query no other test sends, so the first request misses the cache.
    query = "Which city is the capital of France?"
    first = score(llm, query, DOCUMENTS)
    repeated = llm.score(query, DOCUMENTS, chat_template=llm.chat_template)
    assert repeated[0].num_cached_tokens > 0
    assert [result.outputs.score for result in repeated] == first
