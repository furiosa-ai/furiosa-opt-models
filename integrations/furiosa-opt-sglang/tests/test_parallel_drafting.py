"""SGLang's parallel draft cache replaces accepted and rejected mask placeholders."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

TARGET = "Qwen/Qwen3-1.7B"
DRAFT = "amd/PARD-Qwen3-0.6B"
NO_SECOND_NPU = 77


@pytest.mark.feature("Speculative decoding (EAGLE, MTP, Medusa, MLP, parallel drafting)")
@pytest.mark.parametrize("disjoint", [False, True], ids=["colocated", "separate-draft-chip"])
def test_replaces_mask_rows_after_acceptance_and_rejection(disjoint: bool) -> None:
    """Replay accepted and rejected drafts on NPU, including on a separate draft chip."""
    if disjoint:
        probe = subprocess.run(
            [sys.executable, __file__, "--probe-disjoint"],
            capture_output=True,
            check=False,
            text=True,
            timeout=60,
        )
        if probe.returncode == NO_SECOND_NPU:
            pytest.skip(probe.stderr.strip() or "a second visible NPU chip is unavailable")
        probe.check_returncode()
    args = [sys.executable, __file__]
    if disjoint:
        args.append("--disjoint")
    _ = subprocess.run(args, check=True, timeout=900)


def exercise_cache_transition(*, disjoint: bool = False) -> None:
    """Every draft call runs on NPU; committed target IDs force the cache transition."""
    from types import SimpleNamespace
    from unittest.mock import patch

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    import furiosa_opt_engine as engine
    from furiosa_opt_engine.placement import open_placed_device
    from furiosa_opt_sglang.checkpoint import checkout
    from furiosa_opt_sglang.decoder import PES, Decoder, Location, chips
    from furiosa_opt_sglang.platform import DraftPlacement
    from furiosa_opt_sglang.speculative import Decoding, standalone
    from furiosa_opt_sglang.speculative.standalone import Draft

    target_directory = checkout(TARGET, None)
    target_config = AutoConfig.from_pretrained(target_directory)
    target_tokenizer = AutoTokenizer.from_pretrained(target_directory)
    placement = engine.Placement(chips(0, 1), PES)
    selected = open_placed_device(placement)
    target = Decoder.load(
        Location(target_directory, None),
        engine.LoadOptions(),
        selected,
        target_config.vocab_size,
    )
    assert target.model is not None
    serving = SimpleNamespace(
        tokenizer_mode="auto",
        tokenizer_path=target_directory,
        tokenizer_backend="auto",
    )
    model_config = SimpleNamespace(trust_remote_code=False, revision=None, kv_cache_dtype="auto")
    options = SimpleNamespace(
        speculative_draft_model_path=DRAFT,
        speculative_draft_model_revision=None,
        speculative_draft_kv_cache_dtype=None,
        draft_placement=(
            DraftPlacement(chips=tuple(chips(1, 1)), pes=(0, 8)) if disjoint else None
        ),
    )
    with (
        patch.object(standalone, "get_serving", return_value=serving),
        patch.object(standalone, "get_model", return_value=model_config),
        patch.object(
            standalone,
            "get_tokenizer",
            side_effect=lambda name, **_kwargs: AutoTokenizer.from_pretrained(name),
        ),
    ):
        draft = Draft(options, target, context=2048)
    mask = draft._mask
    assert mask is not None
    draft.allocate(256)
    rows = draft._decoder.rows
    reference = AutoModelForCausalLM.from_pretrained(DRAFT, dtype=torch.float32).eval()

    repeated = target_tokenizer.encode(
        "Repeat this exactly: the quick brown fox jumps over the lazy dog.\n",
        add_special_tokens=False,
    )
    prompts = (
        tuple(repeated),
        tuple((repeated * (rows // len(repeated) + 2))[: rows + 3]),
    )
    requests = (
        Decoding("short", prompts[0], 4, tuple(range(128))),
        Decoding("long", prompts[1], rows + 2, tuple(range(128, 256))),
    )

    def check(requests: tuple[Decoding, ...], proposals: tuple[tuple[int, ...], ...]) -> None:
        for request, tokens in zip(requests, proposals, strict=True):
            assert len(tokens) == request.drafts, (request.rid, tokens)
            inputs = [*request.tokens, *([mask] * (request.drafts - 1))]
            with torch.inference_mode():
                scores = reference(torch.tensor([inputs], dtype=torch.long)).logits[0]
            scores = scores[len(request.tokens) - 1 :]
            chosen = scores.gather(1, torch.tensor(tokens, dtype=torch.long)[:, None])[:, 0]
            ratios = (chosen - scores.max(dim=-1).values).exp()
            assert torch.isfinite(ratios).all() and (ratios >= 0.95).all(), (
                request.rid,
                tokens,
                ratios,
            )

    proposals = draft.propose(requests)
    check(requests, proposals)

    # The short request accepts its first draft and rejects its second; the long one accepts all.
    committed = (
        (proposals[0][0], (proposals[0][1] + 1) % draft._decoder.vocabulary),
        (*proposals[1], 13),
    )
    requests = tuple(
        Decoding(request.rid, (*request.tokens, *added), request.drafts, request.blocks)
        for request, added in zip(requests, committed, strict=True)
    )
    check(requests, draft.propose(requests))

    # A request ID can return with different blocks; forget must force a full cache rebuild.
    short = Decoding("short", (*requests[0].tokens, 23), requests[0].drafts, requests[1].blocks)
    draft.forget({short.rid})
    check((short,), draft.propose((short,)))

    # Reset before a reused ID changes both tokens and block tables.
    reset = Decoding("short", (17, 19, 23), short.drafts, requests[0].blocks)
    draft.reset()
    check((reset,), draft.propose((reset,)))

    # Near a request's last block, the shared session clips drafts to the slots that remain.
    block_capacity = 2 * draft._decoder.pool.block
    edge_tokens = tuple((repeated * (block_capacity // len(repeated) + 1))[: block_capacity - 1])
    edge_blocks = tuple(range(2))
    edge = Decoding("edge", edge_tokens, 3, edge_blocks)
    proposals = draft.propose((edge,))
    assert len(proposals[0]) == 2, proposals
    check((Decoding(edge.rid, edge.tokens, 2, edge.blocks),), proposals)

    draft.reset()
    full = Decoding("full", (*edge_tokens, repeated[0]), 3, edge_blocks)
    proposals = draft.propose((full,))
    assert len(proposals[0]) == 1, proposals
    check((Decoding(full.rid, full.tokens, 1, full.blocks),), proposals)


def probe_disjoint_placement() -> None:
    """Exit 77 only when the configured visible-chip list excludes a second chip."""
    import furiosa_opt_engine as engine
    from furiosa_opt_sglang.decoder import PES, chips

    visible = os.environ.get("FURIOSA_VISIBLE_CHIPS", "")
    listed = tuple(chip.strip() for chip in visible.split(","))
    if visible.strip() and all(chip.isdecimal() for chip in listed) and len(listed) < 2:
        print("separate draft placement needs two visible NPU chips", file=sys.stderr)
        raise SystemExit(NO_SECOND_NPU)
    target_chip = chips(0, 1)[0]
    draft_chip = chips(1, 1)[0]
    if draft_chip == target_chip:
        raise ValueError("the target and draft resolve to the same physical chip")
    _ = engine.Device(engine.Placement([draft_chip], PES))


if __name__ == "__main__":
    if sys.argv[1:] == ["--probe-disjoint"]:
        probe_disjoint_placement()
    elif sys.argv[1:] == ["--disjoint"]:
        exercise_cache_transition(disjoint=True)
    elif not sys.argv[1:]:
        exercise_cache_transition()
    else:
        raise SystemExit(f"unknown arguments: {sys.argv[1:]}")
