from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("to_host", [True, False], ids=["gather", "scatter"])
def test_nixl_worker_routes_group_pages_to_engine_layers(
    monkeypatch: pytest.MonkeyPatch, to_host: bool
) -> None:
    import furiosa_opt_vllm.nixl as nixl

    cache = object()
    events = []

    lease = object()
    staging = SimpleNamespace(layer_buffers=["host-0", "host-1"], chips=2)

    def reserve_slots(slots):
        events.append(("reserve", slots))
        return lease

    def copy_pages(cache_arg, layers_arg, lease_arg, *, to_host):
        events.append(("copy", cache_arg, layers_arg, lease_arg, to_host))
        return SimpleNamespace(wait=lambda: events.append(("wait",)))

    def release(lease_arg):
        events.append(("release", lease_arg))

    staging.reserve_slots = reserve_slots
    staging.copy_pages = copy_pages
    staging.release = release

    views = SimpleNamespace(
        runner=object(),
        layers={
            "layers.0": SimpleNamespace(index=0, group=1),
            "layers.1": SimpleNamespace(index=1, group=0),
        },
        staging=staging,
    )
    worker = object.__new__(nixl._BoundedNixlWorker)
    worker._engine_views = views
    worker._reorder_pages = lambda pages, *, to_nixl: events.append(("reorder", pages, to_nixl))
    monkeypatch.setattr(nixl.residency, "cache", lambda _runner: cache)

    logical_pages = ([10, 11], [20])
    staging_pages = ([3, 4], [8])
    worker._copy_pages(logical_pages, staging_pages, to_host=to_host)

    copy_event = next(event for event in events if event[0] == "copy")
    assert copy_event[1] is cache
    assert copy_event[3] is lease
    assert copy_event[4] is to_host
    assert events[0] == ("reserve", [3, 4, 8])
    assert events[-1] == ("release", lease)
    routes = copy_event[2]
    assert routes[0].device_pages == [20]
    assert routes[0].host_slots == [8]
    assert routes[1].device_pages == [10, 11]
    assert routes[1].host_slots == [3, 4]
    assert [event[0] for event in events] == (
        ["reserve", "copy", "wait", "reorder", "release"]
        if to_host
        else ["reserve", "reorder", "copy", "wait", "release"]
    )
    reorder_event = next(event for event in events if event[0] == "reorder")
    assert reorder_event[1] == staging_pages
    assert reorder_event[2] is to_host


def test_nixl_page_layout_conversion_round_trips() -> None:
    import torch

    import furiosa_opt_vllm.nixl as nixl

    engine_page = torch.arange(3 * 2 * 2 * 4, dtype=torch.float32).view(3, 2, 2, 4)
    nixl_page = torch.empty((2, 3, 8), dtype=engine_page.dtype)

    nixl._engine_page_to_nixl(engine_page, nixl_page)

    expected = torch.cat(
        (engine_page[:, 0].permute(1, 0, 2), engine_page[:, 1].permute(1, 0, 2)),
        dim=-1,
    )
    assert torch.equal(nixl_page, expected)

    restored = torch.empty_like(engine_page)
    nixl._nixl_page_to_engine(nixl_page, restored)

    assert torch.equal(restored, engine_page)


def test_nixl_staged_request_reuses_unchanged_slots_and_tracks_reordered_pages() -> None:
    import furiosa_opt_vllm.nixl as nixl

    state = nixl._StagedRequest.create(SimpleNamespace(slots=(4, 5)), groups=1, source=True)
    state.add(([7, 8],))
    mapping = state.slots_by_group[0]

    state.add(([7, 8],))
    assert state.slots_by_group[0] is mapping

    state.add(([8, 7],))
    assert state.map(([8, 7],)) == ([5, 4],)
