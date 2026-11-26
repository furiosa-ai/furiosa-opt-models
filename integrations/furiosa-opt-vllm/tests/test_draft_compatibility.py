"""Draft tokenizer compatibility checks without loading engine models or opening devices."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from vllm import tokenizers

import furiosa_opt_engine as engine
from furiosa_opt_vllm import checkpoint
from furiosa_opt_vllm.admission import Draft, Refused
from furiosa_opt_vllm.decoder import Decoder
from furiosa_opt_vllm.drafting import Drafter

if TYPE_CHECKING:
    from vllm.config import ModelConfig


def model(tokenizer: str) -> SimpleNamespace:
    """Resolved tokenizer settings consumed by vLLM's tokenizer factory."""
    return SimpleNamespace(
        tokenizer=tokenizer,
        tokenizer_revision=None,
        tokenizer_mode="auto",
        trust_remote_code=False,
        get_vocab_size=lambda: 2,
    )


def test_refuses_different_token_to_id_mappings_before_opening_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vocabularies = {
        "target": {"a": 0, "b": 1},
        "draft": {"b": 0, "a": 1},
    }

    def get_tokenizer(path: str, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(get_vocab=lambda: vocabularies[path])

    monkeypatch.setattr(tokenizers, "get_tokenizer", get_tokenizer)

    def fail_if_loaded(*_args: object, **_kwargs: object) -> None:
        pytest.fail("tokenizer mismatch must be refused before opening or loading the draft")

    monkeypatch.setattr(engine, "Device", fail_if_loaded)
    monkeypatch.setattr(engine, "LoadOptions", fail_if_loaded)
    monkeypatch.setattr(checkpoint, "located", fail_if_loaded)
    placement = engine.Placement([0], range(8))
    device = SimpleNamespace(placement=placement)
    target = cast(
        Decoder,
        cast(
            object,
            SimpleNamespace(
                model=object(),
                vocabulary=2,
                device=device,
            ),
        ),
    )
    draft_model = model("draft")
    draft = cast(
        Draft,
        cast(
            object,
            SimpleNamespace(
                model=draft_model,
                placement=SimpleNamespace(chips=(1,), pes=(0, 8)),
                parallel=SimpleNamespace(tensor_parallel_size=1),
            ),
        ),
    )

    with pytest.raises(Refused, match="matching target and draft token-to-ID mappings"):
        _ = Drafter.load(draft, target, cast("ModelConfig", cast(object, model("target"))))
