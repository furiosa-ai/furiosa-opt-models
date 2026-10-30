"""Submit a planned call to a paged decoder, whose cache vLLM's block tables address.

A call's rows take their tokens from one operand on the device, except that the last row of a
decoding run may take the token an uncollected call selects there; a sequence's rows run with the
adapter of its slot in the runner's list of adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import furiosa_opt_engine as engine
from furiosa_opt_vllm.batch import operand

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from furiosa_opt_vllm.batch import Call
    from furiosa_opt_vllm.decoder import Decoder
    from furiosa_opt_vllm.runner import Runner

    # A decoder's call: its hidden rows, or a causal model's greedy tokens, each request running
    # with the adapter of its slot in the list.
    Entry: TypeAlias = Callable[
        [
            list[engine.decoder.Request],
            engine.decoder.Cache,
            list[engine.decoder.Adapter | None] | None,
        ],
        engine.HbmTensor,
    ]


@dataclass(frozen=True, slots=True)
class Feed:
    """What a call's rows take: the device `operand` of the rows the host knows, and each other
    row's token, in an uncollected call's output at its row there; `every` keeps the rows before
    such a token. A sequence's rows run with the adapter of its slot in `slots`, or none."""

    operand: engine.HbmTensor | None
    fed: Mapping[int, tuple[engine.HbmTensor, int]]
    every: bool
    slots: Mapping[str, int]


def upload(
    decoder: Decoder, ids: list[int], fed: Mapping[int, tuple[engine.HbmTensor, int]]
) -> engine.HbmTensor | None:
    """A call's token IDs on the decoder's device in one copy, none when every row takes a `fed`
    token instead."""
    if len(fed) >= len(ids):
        return None
    return engine.HostTensor.from_dlpack(operand(ids, decoder.vocabulary)).to_hbm(decoder.device)


def forward(runner: Runner, entry: Entry, call: Call[str], feed: Feed) -> engine.HbmTensor:
    """`entry`'s output of `call`'s rows, appended to the runner's cache with one engine request
    per run of a sequence's rows and one more for a run's device token, each with its sequence's
    adapter slot."""
    cache = runner.cache
    if cache is None:
        raise RuntimeError("initialize_kv_cache must precede paged execution")
    # --8<-- [start:execute]
    kept = set(call.kept)
    requests: list[engine.decoder.Request] = []
    for request in call.requests:
        run = list(request.rows)
        blocks = runner.requests[request.sequence].block_ids
        last = engine.decoder.Keep.LAST if request.complete else engine.decoder.Keep.NONE
        slot = feed.slots.get(request.sequence)
        # Kept rows end a run, so a run kept from its first row returns every row.
        keep = engine.decoder.Keep.ALL if run[0] in kept else last
        # The last row of a decoding request may take its token from the device.
        tail = feed.fed.get(run[-1])
        head = run[:-1] if tail is not None else run
        if head:
            if feed.operand is None:
                raise RuntimeError("rows with known tokens need the uploaded IDs")
            head_keep = keep if tail is None or feed.every else engine.decoder.Keep.NONE
            tokens = feed.operand[head[0] : head[-1] + 1]
            requests.append(
                engine.decoder.Request(blocks, request.cached, tokens, head_keep, adapter=slot)
            )
        if tail is not None:
            selector, index = tail
            token = selector[index : index + 1]
            cached = request.cached + len(head)
            requests.append(engine.decoder.Request(blocks, cached, token, keep, adapter=slot))
    adapters = runner.adapters.slots if runner.adapters is not None else None
    return entry(requests, cache, adapters)
    # --8<-- [end:execute]
