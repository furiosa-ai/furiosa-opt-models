from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

from furiosa_opt_engine.cache_transfer import LayerPageRoutes


def test_sglang_packed_staging_builds_one_engine_route_per_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from furiosa_opt_sglang import staging

    cache = SimpleNamespace()
    pool = SimpleNamespace(layout=SimpleNamespace(layers=(object(), object())))
    captured: dict[str, object] = {}
    released = []
    lease = object()
    result = SimpleNamespace(done=lambda: True, wait=lambda: None)

    def reserve_slots(slots):
        captured["slots"] = slots
        return lease

    def submit(cache_arg, routes, lease_arg, *, to_host):
        captured.update(cache=cache_arg, routes=routes, lease=lease_arg, to_host=to_host)
        return result

    pool.reserve_slots = reserve_slots
    pool.copy_pages = submit
    pool.release = released.append

    actual = staging.submit_packed_cache_pages(pool, cache, [7, 9], [3, 4], to_host=False)

    assert captured["slots"] == [3, 4]
    assert captured["cache"] is cache
    assert captured["lease"] is lease
    assert captured["to_host"] is False
    assert captured["routes"] == {
        0: LayerPageRoutes([7, 9], [3, 4]),
        1: LayerPageRoutes([7, 9], [3, 4]),
    }
    actual.wait()
    assert released == [lease]


def test_scatter_region_maps_suffix_pages_past_protected_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from furiosa_opt_sglang import staging

    storage = object()
    cache = SimpleNamespace(
        storage=storage,
        page_size=2,
        pages=SimpleNamespace(pool=SimpleNamespace(bytes=16), chips=2),
    )
    pool = SimpleNamespace(layout=SimpleNamespace(slot_bytes=32, capacity_pages=8))
    staging_buffer = SimpleNamespace(
        buffer=SimpleNamespace(attach_cache=lambda cache_arg, chips, copier: pool)
    )
    handler = object.__new__(staging.EngineDecodeStagingHandler)
    handler.kv_manager = SimpleNamespace(
        engine_cache=cache,
        _staging_binding=SimpleNamespace(copier=object()),
    )
    handler.scheduler = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=torch.tensor([[0] * 8, list(range(10, 18))]))
    )
    handler.staging_allocator = SimpleNamespace(buffer=staging_buffer.buffer)
    decode_req = SimpleNamespace(
        req=SimpleNamespace(kv=SimpleNamespace(req_pool_idx=1, cache_protected_len=2))
    )
    calls: list[tuple[object, ...]] = []
    result = object()

    def submit(cache_arg, pool_arg, page_ids, host_slots, *, to_host):
        calls.append((cache_arg, pool_arg, page_ids, host_slots, to_host))
        return result

    monkeypatch.setattr(staging, "submit_packed_cache_pages", submit)

    actual = handler._scatter_region(96, 1, 2, decode_req)

    assert actual is result
    assert calls == [(cache, pool, [7, 8], (3, 4), False)]


@pytest.mark.parametrize(
    ("page_start", "num_pages", "message"),
    [(-1, 1, "page start"), (0, 0, "must contain pages"), (0, -1, "must contain pages")],
)
def test_scatter_region_rejects_invalid_page_ranges(
    page_start: int, num_pages: int, message: str
) -> None:
    from furiosa_opt_sglang.staging import EngineDecodeStagingHandler

    handler = object.__new__(EngineDecodeStagingHandler)

    with pytest.raises(ValueError, match=message):
        handler._scatter_region(0, page_start, num_pages, object())


def test_scatter_rejects_negative_chunk_index() -> None:
    from furiosa_opt_sglang.staging import EngineDecodeStagingHandler

    handler = object.__new__(EngineDecodeStagingHandler)
    handler._transfer_lock = threading.Lock()
    handler._room_to_decode_req = {7: SimpleNamespace(_staging_failed=False)}
    handler._room_to_receiver = {7: SimpleNamespace(chunk_staging_infos=[(1, 0, 0, 0, 1)])}

    assert handler.submit_chunk_scatter(7, -1, 0, 1) is False


def test_scatter_dma_failure_marks_only_the_room_failed() -> None:
    from sglang.srt.disaggregation.base import KVPoll

    from furiosa_opt_sglang import staging

    failure = RuntimeError("engine DMA failed")

    class _FailedBatch:
        def done(self) -> bool:
            return True

        def wait(self) -> None:
            raise failure

    events: list[tuple[object, ...]] = []
    decode_req = SimpleNamespace(
        _staging_failed=False,
        _chunk_events=[],
        req=SimpleNamespace(bootstrap_room=7),
    )
    handler = object.__new__(staging.EngineDecodeStagingHandler)
    handler._transfer_lock = threading.Lock()
    handler._room_to_decode_req = {7: decode_req}
    handler._room_to_receiver = {7: SimpleNamespace(chunk_staging_infos=[(3, 0, 0, 0, 1)])}
    handler._transfer_batches = set()
    handler.kv_manager = SimpleNamespace(
        record_failure=lambda room, error: events.append(("failure", room, error)),
        update_status=lambda room, status: events.append(("status", room, status)),
    )
    handler._scatter_region = lambda *_args: _FailedBatch()

    assert handler.submit_chunk_scatter(7, 0, 0, 1) is True
    batch, alloc_id = decode_req._chunk_events[0]
    assert alloc_id == 3
    assert batch.query() is True
    assert decode_req._staging_failed is True
    assert events == [
        ("failure", 7, str(failure)),
        ("status", 7, KVPoll.Failed),
    ]
