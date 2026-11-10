"""Speculative decoding, role switching and data-parallel replicas beside prefill-decode
disaggregation keep a request's tokens those of a server without disaggregation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from disaggregated import (
    LONG,
    STANDALONE,
    SWITCHING,
    TOKENS,
    Prefill,
    decoding,
    generate,
    greedy,
    prompt,
    read,
    tokens,
)
from disaggregated import chips as chips
from disaggregated import oracle as oracle
from disaggregated import prefill as prefill
from test_server import plugin as plugin

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.feature(
    "Disaggregated prefill and KV connectors (NIXL, Mooncake, MoRIIO, LMCache)"
)


def test_decodes_on_data_parallel_replicas(
    oracle: str, prefill: Prefill, chips: list[str], tmp_path: Path
) -> None:
    """Requests that a decode server of two data-parallel replicas receives decode as on a server
    without disaggregation."""
    prompts = [tokens(oracle, prompt(topic)) for topic in ("a valley", "a desert")]
    with decoding(chips, tmp_path, ["--dp-size", "2"]) as decode:
        answers = [prefill.decode(decode, ids)[1] for ids in prompts]
    expected = [generate(oracle, ids, greedy(TOKENS))["output_ids"] for ids in prompts]
    assert [answer["output_ids"] for answer in answers] == expected


@pytest.mark.parametrize("prefill", ["standalone"], indirect=True)
def test_decodes_speculatively(
    oracle: str, prefill: Prefill, chips: list[str], tmp_path: Path
) -> None:
    """A decode server whose draft model drafts over a remote prefill's cache accepts some drafts
    and decodes as a server without disaggregation."""
    ids = tokens(oracle, prompt("a cup"))
    with decoding(chips, tmp_path, STANDALONE) as decode:
        _, answer = prefill.decode(decode, ids, LONG)
    assert answer["meta_info"].get("spec_accept_length", 0) > 1, answer["meta_info"]
    assert answer["output_ids"] == generate(oracle, ids, greedy(LONG))["output_ids"]


@pytest.mark.parametrize("prefill", ["role-switch"], indirect=True)
def test_decodes_after_switching_roles(
    oracle: str, prefill: Prefill, chips: list[str], tmp_path: Path
) -> None:
    """A pair whose prefill and decode servers switch roles sends a prompt, and then one whose
    prefix the new prefill server caches, as a server without disaggregation decodes them."""
    from sglang.srt.utils.network import get_free_port

    bootstrap = get_free_port()
    # A radix cache it keeps through the switch gives the new prefill server a prefix to send early.
    flags = [*SWITCHING, "--disaggregation-bootstrap-port", str(bootstrap)]
    flags += ["--disaggregation-decode-enable-radix-cache"]
    prefix = tokens(oracle, prompt("a bell"))
    prompts = [prefix + tokens(oracle, f" Name bell {n}.") for n in (1, 2)]
    with decoding(chips, tmp_path, flags) as decode:
        _ = read(f"{decode}/pd_role_switch", {"new_role": "prefill"})
        # SGLang captures no decode graphs here, so it asks the size of a peer's.
        switched = {"new_role": "decode", "decode_cuda_graph_memory_gb": 0}
        _ = read(f"{prefill.url}/pd_role_switch", switched)
        swapped = Prefill(decode, bootstrap)
        answers = [swapped.decode(prefill.url, ids) for ids in prompts]
    sent, _ = answers[1]
    assert sent["meta_info"]["cached_tokens"] > 0, sent["meta_info"]
    expected = [generate(oracle, ids, greedy(TOKENS))["output_ids"] for ids in prompts]
    assert [decoded["output_ids"] for _, decoded in answers] == expected
