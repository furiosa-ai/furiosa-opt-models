"""LoRA adapters compose with the features SGLang serves them with: NGRAM speculation verifies
with each request's adapter, and an embedding model's adapter changes its embeddings."""

from pathlib import Path
from typing import Final

import pytest
import torch
from pydantic import TypeAdapter
from test_lora_lifecycle import ANSWERS, ATTENTION, LORA, MLP, MODEL, PROMPT, Answer, save
from test_server import Chips
from test_server import chips as chips
from test_server import plugin as plugin
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing_extensions import TypedDict

EMBEDDING: Final = "Qwen/Qwen3-Embedding-0.6B"
REPEATED: Final = "Repeat: red blue red blue red blue"
# The base model copies `REPEATED` by a wide margin at every token, which adapters of the fixture's
# 0.02 factors never move; factors 3.5 times larger take the continuation over, each repeating a
# token of its own that NGRAM then drafts.
LOUD: Final = 0.07
NGRAM: Final = (
    "--speculative-algorithm",
    "NGRAM",
    "--speculative-ngram-max-bfs-breadth",
    "1",
    "--speculative-num-draft-tokens",
    "4",
)


class Encoded(TypedDict):
    embedding: list[float]


ENCODED: Final = TypeAdapter(list[Encoded])


@pytest.mark.feature("LoRA")
def test_ngram_drafts_verify_with_adapters(
    tmp_path_factory: pytest.TempPathFactory, chips: Chips, tmp_path: Path
) -> None:
    """Under NGRAM speculation, a batch of the base model and two adapters verifies each request's
    drafts with its own adapter: with both servers batch invariant, the greedy tokens are those of
    ordinary generation, and an adapter's request accepts drafts."""
    from peft import LoraConfig

    forms = {
        "partial": LoraConfig(r=4, lora_alpha=32, target_modules=["q_proj", "o_proj"]),
        "rslora": LoraConfig(r=8, lora_alpha=16, use_rslora=True, target_modules=ATTENTION),
    }
    loud = save(lambda: AutoModelForCausalLM.from_pretrained(MODEL), forms, tmp_path, scale=LOUD)
    names: list[str | None] = [None, *loud]
    paths = ["--lora-paths", *(f"{name}={path}" for name, path in loud.items())]

    def repeat(speculation: tuple[str, ...]) -> list[Answer]:
        # Batch invariance makes a verification call's rows the bits of one-row decode calls.
        flags = [*LORA, "--enable-deterministic-inference", *speculation, *paths]
        params = {"temperature": 0, "max_new_tokens": 24, "ignore_eos": True}
        body = {"text": [REPEATED] * len(names), "lora_path": names, "sampling_params": params}
        with (
            chips.serve(tmp_path_factory, {"id": MODEL, "task": "text", "flags": flags}) as send,
            send("/generate", body) as response,
        ):
            return ANSWERS.validate_json(response.read())

    expected = [answer["output_ids"] for answer in repeat(())]
    # Each adapter changes its request's tokens, so a verification that read another request's
    # adapter, or none, would change them.
    assert len({tuple(tokens) for tokens in expected}) == len(names), expected
    answers = repeat(NGRAM)
    assert [answer["output_ids"] for answer in answers] == expected
    assert max(answer["meta_info"].get("spec_accept_length", 0) for answer in answers[1:]) > 1


@pytest.mark.feature("LoRA")
def test_embeds_with_adapters(
    tmp_path_factory: pytest.TempPathFactory, chips: Chips, tmp_path: Path
) -> None:
    """An embedding model's adapter, named as PEFT names the causal model's modules, changes its
    embeddings to those of Transformers with the adapter merged by PEFT, beside the base model's
    in one request."""
    from peft import LoraConfig, PeftModel

    forms = {"sky": LoraConfig(r=32, lora_alpha=45, target_modules=ATTENTION + MLP)}

    def causal() -> torch.nn.Module:
        return AutoModelForCausalLM.from_pretrained(EMBEDDING)

    (adapter,) = save(causal, forms, tmp_path).values()
    inputs = AutoTokenizer.from_pretrained(EMBEDDING)([PROMPT], return_tensors="pt")

    def reference(merged: bool) -> torch.Tensor:
        model = AutoModelForCausalLM.from_pretrained(EMBEDDING, dtype=torch.bfloat16).eval()
        if merged:
            model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
        with torch.inference_mode():
            last = model.model(**inputs).last_hidden_state[0, -1].float()
        return torch.nn.functional.normalize(last, dim=-1)

    flags = ["--enable-lora", "--lora-paths", f"sky={adapter}"]
    row = {"id": EMBEDDING, "task": "embedding", "flags": flags}
    with chips.serve(tmp_path_factory, row) as send:
        body = {"text": [PROMPT, PROMPT], "lora_path": [None, "sky"]}
        with send("/encode", body) as response:
            plain, adapted = (
                torch.tensor(item["embedding"]) for item in ENCODED.validate_json(response.read())
            )
    similarity = torch.nn.functional.cosine_similarity
    base, merged = reference(merged=False), reference(merged=True)
    # The oracle tells the adapter's embedding from the base model's.
    assert similarity(base, merged, dim=0) < 0.99
    assert similarity(plain, base, dim=0) > 0.999
    assert similarity(adapted, merged, dim=0) > 0.999
