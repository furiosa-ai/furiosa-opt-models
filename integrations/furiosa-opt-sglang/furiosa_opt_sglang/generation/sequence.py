from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, cast

import torch

import furiosa_opt_engine as engine
from furiosa_opt_sglang.batch import Batch, Call, Row, operand
from furiosa_opt_sglang.decoder import RELEASED, Decoder
from furiosa_opt_sglang.generation.model import Model

if TYPE_CHECKING:
    from collections.abc import Callable

    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

    from furiosa_opt_sglang.runner import Runner

    # Each call's hidden rows or greedy tokens, with the (request, row) pairs they hold.
    Rows = Iterator[tuple[list[tuple[int, int]], engine.HbmTensor]]


# PyTorch erases element types from integer tensor list conversion.
values = cast("Callable[[torch.Tensor], list[int]]", torch.Tensor.tolist)


def forward(
    runner: Runner,
    batch: ScheduleBatch,
    snapshot: Batch,
    *,
    every: bool,
    feed: Mapping[int, engine.HbmTensor],
) -> Rows:
    """Each call's rows with the request rows they hold: hidden rows, or a causal decoder's greedy
    token of each kept row; a request in `feed` takes its one row's token on the device."""
    model = runner.model.loaded
    pool = runner.req_to_token_pool
    if pool is None:
        raise RuntimeError("cache initialization must precede scheduled generation")
    yield from paged(model, model.core, pool, batch, snapshot, every=every, feed=feed)


def paged(
    model: Model,
    decoder: Decoder,
    pool: ReqToTokenPool,
    batch: ScheduleBatch,
    snapshot: Batch,
    *,
    every: bool,
    feed: Mapping[int, engine.HbmTensor],
) -> Rows:
    """Causal rows over the paged cache, as many a call as the decoder runs."""
    inputs = snapshot.tokens
    positions = snapshot.cached
    lengths = [len(tokens) for tokens in inputs]
    if model.cache is None:
        raise RuntimeError("cache initialization must precede paged generation")
    storage, loaded = model.cache.storage, decoder.model
    if storage is None or loaded is None:
        raise RuntimeError(RELEASED)
    block = decoder.pool.block
    queries = [(i, offset) for i, length in enumerate(lengths) for offset in range(length)]
    for start in range(0, len(queries), decoder.rows):
        rows = queries[start : start + decoder.rows]
        query_positions = [positions[i] + offset for i, offset in rows]
        # Rows that take a selected token read none of the uploaded IDs.
        ids = [0 if i in feed else inputs[i][offset] for i, offset in rows]
        uploaded = (
            engine.HostTensor.from_dlpack(operand(ids, decoder.vocabulary)).to_hbm(decoder.device)
            if any(i not in feed for i, _ in rows)
            else None
        )
        # --8<-- [start:forward]
        call = Call[int].plan(
            [Row(i, positions[i] + offset, offset + 1 == lengths[i]) for i, offset in rows],
            every=every,
        )
        requests: list[engine.decoder.Request] = []
        for request in call.requests:
            i = request.sequence
            run = list(request.rows)
            blocks = (
                pool.req_to_token[
                    batch.reqs[i].kv.req_pool_idx, : query_positions[run[-1]] + 1 : block
                ]
                // block
            )
            last = engine.decoder.Keep.LAST if request.complete else engine.decoder.Keep.NONE
            tokens = feed.get(i)
            if tokens is None:
                assert uploaded is not None, "a row whose token is not on the device is uploaded"
                tokens = uploaded[run[0] : run[-1] + 1]
            requests.append(
                engine.decoder.Request(
                    [values(blocks)],
                    request.cached,
                    tokens,
                    engine.decoder.Keep.ALL if every else last,
                )
            )
        output = loaded.forward(requests, storage)
        # --8<-- [end:forward]
        yield [rows[row] for row in call.kept], output
