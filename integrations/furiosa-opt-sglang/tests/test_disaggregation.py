"""Requests that a prefill server prefills and a decode server decodes, their cache moved by the
transfer backend through the token pools' host images, continue as a server without
disaggregation continues them, to the bit."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import pytest
from disaggregated import (
    CHUNK,
    LONG,
    PAGE,
    PAUSED,
    SAVING,
    TOKENS,
    Answer,
    Prefill,
    decoding,
    ended,
    generate,
    greedy,
    prompt,
    read,
    stream,
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


def test_decodes_a_remote_prefill(
    oracle: str, prefill: Prefill, chips: list[str], tmp_path: Path
) -> None:
    """The decode server's greedy tokens of a prompt whose cache the prefill server sent equal a
    server's own."""
    ids = tokens(oracle, prompt("the color of the sky"))
    with decoding(chips, tmp_path) as decode:
        _, answer = prefill.decode(decode, ids)
    assert answer["output_ids"] == generate(oracle, ids, greedy(TOKENS))["output_ids"]


def test_decodes_a_cached_prefix(
    oracle: str, prefill: Prefill, chips: list[str], tmp_path: Path
) -> None:
    """A prompt whose prefix the prefill server holds in its prefix cache, which SGLang sends
    before it prefills the rest, decodes as on a server without disaggregation."""
    prefix = tokens(oracle, prompt("a fruit"))
    prompts = [prefix + tokens(oracle, f" Name fruit {n}.") for n in (1, 2)]
    with decoding(chips, tmp_path) as decode:
        answers = [prefill.decode(decode, ids) for ids in prompts]
    sent, _ = answers[1]
    assert sent["meta_info"]["cached_tokens"] > 0
    expected = [generate(oracle, ids, greedy(TOKENS))["output_ids"] for ids in prompts]
    assert [decoded["output_ids"] for _, decoded in answers] == expected


def test_decodes_after_a_retracting_pause(
    oracle: str, prefill: Prefill, chips: list[str], tmp_path: Path
) -> None:
    """Requests that `/pause_generation` retracts mid-decode, whose cache the decode server
    receives again from the prefill server on `/continue_generation`, decode as on a server
    without disaggregation."""
    prompts = [tokens(oracle, prompt(topic)) for topic in ("a planet", "a river")]
    started = [threading.Event() for _ in prompts]

    # One request on both servers, which reports once the decode server streamed 8 tokens.
    def run(decode: str, ids: list[int], event: threading.Event) -> Answer:
        def seen(count: int) -> None:
            if count >= 8:
                event.set()

        room = prefill.room()
        with ThreadPoolExecutor(1) as pool:
            sent = pool.submit(generate, prefill.url, ids, greedy(PAUSED), room)
            answer = stream(decode, ids, greedy(PAUSED), room, seen)
            _ = sent.result()
        return answer

    with decoding(chips, tmp_path) as decode, ThreadPoolExecutor(len(prompts)) as pool:
        running = [pool.submit(run, decode, *pair) for pair in zip(prompts, started, strict=True)]
        assert all(event.wait(120) for event in started)
        _ = read(f"{decode}/pause_generation", {"mode": "retract"})
        _ = read(f"{decode}/continue_generation", {})
        answers = [future.result() for future in running]
    for ids, answer in zip(prompts, answers, strict=True):
        assert answer["meta_info"]["num_retractions"] > 0, answer["meta_info"]
        assert ended(answer), answer["meta_info"]
        assert answer["output_ids"] == generate(oracle, ids, greedy(PAUSED))["output_ids"]


def test_decodes_after_a_restore(
    oracle: str, prefill: Prefill, chips: list[str], tmp_path: Path
) -> None:
    """Requests that a decode server retracts to free cache memory, backing their cache up on the
    host and restoring it into new pages, decode as on a server without disaggregation."""
    prompts = [tokens(oracle, prompt(topic)) for topic in ("a bird", "a tree", "a stone")]
    # SGLang's switch that retracts a running request every few decode steps.
    with (
        decoding(chips, tmp_path, env={"SGLANG_TEST_RETRACT": "1"}) as decode,
        ThreadPoolExecutor(len(prompts)) as pool,
    ):
        running = [pool.submit(prefill.decode, decode, ids, LONG) for ids in prompts]
        answers = [future.result()[1] for future in running]
    assert any(answer["meta_info"]["num_retractions"] for answer in answers)
    for ids, answer in zip(prompts, answers, strict=True):
        assert ended(answer), answer["meta_info"]
        assert answer["output_ids"] == generate(oracle, ids, greedy(LONG))["output_ids"]


def test_decodes_a_prefix_the_decode_server_caches(
    oracle: str, prefill: Prefill, chips: list[str], tmp_path: Path
) -> None:
    """A prompt that continues an earlier answer, whose pages the decode server's own radix cache
    holds, receives only the rest from the prefill server and decodes as on a server without
    disaggregation."""
    first = tokens(oracle, prompt("a mountain"))
    with decoding(chips, tmp_path, ["--disaggregation-decode-enable-radix-cache"]) as decode:
        _, answer = prefill.decode(decode, first, LONG)
        # The answer's pages the decode server computed itself, which no transfer wrote.
        later = first + answer["output_ids"] + tokens(oracle, " Name one more.")
        sent, continued = prefill.decode(decode, later)
    assert continued["meta_info"]["cached_tokens"] > sent["meta_info"]["cached_tokens"], (
        sent["meta_info"],
        continued["meta_info"],
    )
    assert continued["output_ids"] == generate(oracle, later, greedy(TOKENS))["output_ids"]


def test_decodes_a_prefix_only_the_decode_server_caches(
    oracle: str, prefill: Prefill, chips: list[str], tmp_path: Path
) -> None:
    """A prompt whose prefix only the decode server's radix cache holds, from a request that the
    prefill server's handoff token ended, decodes as on a server without disaggregation."""
    first = tokens(oracle, prompt("a harbor"))
    later = first + tokens(oracle, " Name one more.")
    with decoding(chips, tmp_path, ["--disaggregation-decode-enable-radix-cache"]) as decode:
        _ = prefill.decode(decode, first, 1)
        # The prefill server then sends every page of the next prompt but those the decode server
        # holds.
        _ = read(f"{prefill.url}/flush_cache?timeout=10", {})
        _, continued = prefill.decode(decode, later)
    assert continued["meta_info"]["cached_tokens"] > 0, continued["meta_info"]
    assert continued["output_ids"] == generate(oracle, later, greedy(TOKENS))["output_ids"]


def test_decodes_a_prefix_the_decode_server_reloads_from_host_memory(
    oracle: str, prefill: Prefill, chips: list[str], tmp_path: Path
) -> None:
    """A prompt whose prefix the decode server's hierarchical cache backed up to host memory, from a
    request that the prefill server's handoff token ended, and loads back decodes as on a server
    without disaggregation."""
    first, *others = (tokens(oracle, prompt(topic, items=5)) for topic in ("tin", "silk", "jade"))
    later = first + tokens(oracle, " Name one more.")
    # A device pool of 12 pages: the three 5-page prompts evict the first to the host tier, and the
    # last prompt's 6 pages fit once it evicts another, beside a host tier twice its size.
    flags = ["--disaggregation-decode-enable-radix-cache", "--max-total-tokens", "192"]
    flags += ["--enable-hierarchical-cache", "--hicache-ratio", "2"]
    # SGLang admits a request only with this many decode tokens of headroom beyond its pages, 512
    # by default, which the small pool never has.
    flags += ["--num-reserved-decode-tokens", str(TOKENS)]
    with decoding(chips, tmp_path, flags) as decode:
        for ids in (first, *others):
            _ = prefill.decode(decode, ids, 1)
        _ = read(f"{prefill.url}/flush_cache?timeout=10", {})
        _, continued = prefill.decode(decode, later)
    assert continued["meta_info"]["cached_tokens"] > 0, continued["meta_info"]
    assert ended(continued), continued["meta_info"]
    assert continued["output_ids"] == generate(oracle, later, greedy(TOKENS))["output_ids"]


@pytest.mark.parametrize("prefill", ["chunked"], indirect=True)
def test_decodes_a_chunked_prefill(
    oracle: str, prefill: Prefill, chips: list[str], tmp_path: Path
) -> None:
    """A prompt the prefill server prefills and sends in several chunks, its last page partial,
    decodes as on a server that prefills it at once."""
    # Two whole chunks, then a last one ending in a partial page.
    ids = tokens(oracle, prompt("the color of the sea"))[: 2 * CHUNK + PAGE + PAGE // 2]
    assert divmod(len(ids), CHUNK) == (2, PAGE + PAGE // 2), len(ids)
    with decoding(chips, tmp_path) as decode:
        _, answer = prefill.decode(decode, ids)
    assert answer["output_ids"] == generate(oracle, ids, greedy(TOKENS))["output_ids"]


@pytest.mark.parametrize("prefill", ["hierarchical"], indirect=True)
def test_decodes_a_prefix_reloaded_from_host_memory(
    oracle: str, prefill: Prefill, chips: list[str], tmp_path: Path
) -> None:
    """A prompt whose cached prefix other prompts evicted from the prefill server's device, which
    its hierarchical cache loads back before SGLang sends it, decodes as on a server without
    disaggregation."""
    first, *others = (tokens(oracle, prompt(topic, items=5)) for topic in ("ink", "salt", "wool"))
    with decoding(chips, tmp_path) as decode:
        for ids in (first, *others):
            _ = prefill.decode(decode, ids)
        sent, answer = prefill.decode(decode, first)
    details = sent["meta_info"].get("cached_tokens_details")
    assert details is not None and details["host"] > 0, sent["meta_info"]
    assert answer["output_ids"] == generate(oracle, first, greedy(TOKENS))["output_ids"]


@pytest.mark.parametrize("prefill", ["memory-saver"], indirect=True)
def test_decodes_after_releasing_memory(
    oracle: str, prefill: Prefill, chips: list[str], tmp_path: Path
) -> None:
    """Servers that release and resume their memory between two requests decode both as a server
    without disaggregation."""
    prompts = [tokens(oracle, prompt(topic)) for topic in ("a lake", "a cloud")]
    with decoding(chips, tmp_path, SAVING) as decode:
        _, before = prefill.decode(decode, prompts[0])
        for path in ("/release_memory_occupation", "/resume_memory_occupation"):
            for url in (prefill.url, decode):
                _ = read(f"{url}{path}", {"tags": None})
        _, after = prefill.decode(decode, prompts[1])
    expected = [generate(oracle, ids, greedy(TOKENS))["output_ids"] for ids in prompts]
    assert [before["output_ids"], after["output_ids"]] == expected
