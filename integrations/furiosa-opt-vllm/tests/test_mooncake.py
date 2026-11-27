from types import SimpleNamespace

import pytest


class _Transfer:
    def __init__(self, complete) -> None:
        self._complete = complete

    def wait(self) -> None:
        self._complete()


class _CopierLayer:
    def __init__(self, buffer, page_bytes: int, host_offset: int, group: int) -> None:
        self.buffer = buffer
        self.page_bytes = page_bytes
        self.host_offset = host_offset
        self.group = group


class _Copier:
    def __init__(self, layers, host_block_bytes: int) -> None:
        self.layers = layers
        self.host_block_bytes = host_block_bytes

    def _copy(self, routes, *, to_host: bool):
        for slot, layer_routes in routes:
            for group, page in layer_routes:
                layer = self.layers[group]
                host_offset = slot * self.host_block_bytes + layer.host_offset
                device_offset = page * layer.page_bytes
                if to_host:
                    layer.buffer.events.append(
                        ("hbm-to-host", layer.buffer.layer, device_offset, host_offset)
                    )
                else:
                    layer.buffer.events.append(
                        ("host-to-hbm", layer.buffer.layer, host_offset, device_offset)
                    )
        return _Transfer(lambda: None)

    def copy_out(self, routes, _host):
        return self._copy(routes, to_host=True)

    def copy_in(self, routes, _host):
        return self._copy(routes, to_host=False)


class _HostCache:
    def __init__(self, _copier, slots: int, *, backing) -> None:
        self.slots = [object() for _ in range(slots)]
        self.backing = backing

    def slot(self, index: int):
        slot = self.slots[index]
        return SimpleNamespace(access=lambda: slot)

    def _backing_buffer(self):
        return self.backing


class _HostSlice:
    def __init__(self, buffer, offset: int, size: int) -> None:
        self.buffer = buffer
        self.offset = offset
        self.size = size

    def __len__(self) -> int:
        return self.size

    def slice(self, offset: int, size: int) -> "_HostSlice":
        assert 0 <= offset and 0 < size and offset + size <= self.size
        return _HostSlice(self.buffer, self.offset + offset, size)

    def to_hbm(self, device):
        assert self.size == device.size * device.buffer.chips
        return _Transfer(
            lambda: device.buffer.events.append(
                ("host-to-hbm", device.buffer.layer, self.offset, device.offset)
            )
        )


class _HostBuffer(bytearray):
    def __new__(cls, size: int):
        return super().__new__(cls, size)

    def __init__(self, size: int) -> None:
        super().__init__(size)

    def slice(self, offset: int, size: int) -> _HostSlice:
        return _HostSlice(self, offset, size)


class _DeviceSlice:
    def __init__(self, buffer, offset: int, size: int) -> None:
        self.buffer = buffer
        self.offset = offset
        self.size = size

    def to_host(self, host: _HostSlice):
        assert host.size == self.size * self.buffer.chips
        return _Transfer(
            lambda: self.buffer.events.append(
                ("hbm-to-host", self.buffer.layer, self.offset, host.offset)
            )
        )


class _DeviceBuffer:
    def __init__(self, layer: int, size: int, chips: int, events) -> None:
        self.layer = layer
        self.size = size
        self.chips = chips
        self.events = events

    def __len__(self) -> int:
        return self.size

    def slice(self, offset: int, size: int) -> _DeviceSlice:
        return _DeviceSlice(self, offset, size)


class _TensorView:
    def as_strided(self, shape, _strides, *, storage_offset=0):
        result = _TensorView()
        result.shape = shape
        result.storage_offset = storage_offset
        return result

    def view(self, *shape):
        self.shape = shape
        return self


def _runner(chips: int = 2):
    page_bytes, blocks = 24, 100_000
    events: list[tuple[object, ...]] = []
    cache = SimpleNamespace(
        blocks=blocks,
        layers=[_DeviceBuffer(index, blocks * page_bytes, chips, events) for index in range(2)],
        spec=SimpleNamespace(
            groups=[SimpleNamespace(pool=0, layers=[0, 1])],
            pools=[SimpleNamespace(bytes=page_bytes)],
        ),
    )

    class Decoder:
        model = SimpleNamespace(parallel=chips)

    return SimpleNamespace(cache=cache, model=Decoder()), events, page_bytes, blocks


def test_mooncake_uses_bounded_staging_for_selected_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    import furiosa_opt_engine as engine
    import furiosa_opt_vllm.mooncake as mooncake

    runner, events, page_bytes, blocks = _runner()
    monkeypatch.setattr(engine, "HostBuffer", _HostBuffer)
    monkeypatch.setattr(engine, "CopierLayer", _CopierLayer)
    monkeypatch.setattr(engine, "Copier", _Copier)
    monkeypatch.setattr(engine, "HostCache", _HostCache)
    monkeypatch.setattr(torch, "frombuffer", lambda *_args, **_kwargs: _TensorView())
    monkeypatch.setattr(mooncake, "Decoder", runner.model.__class__)
    page_spec = SimpleNamespace(page_size_bytes=page_bytes, block_size=16)
    group = SimpleNamespace(
        layer_names=["layers.0", "layers.1"],
        kv_cache_spec=page_spec,
    )
    config = SimpleNamespace(
        transfer_group_index_by_layer={"layers.0": 0, "layers.1": 0},
        transfer_groups=[group],
        kv_cache_groups=[group],
        num_blocks=blocks,
    )
    staging = mooncake._CacheStaging(runner, config, budget_bytes=1024 * 1024)
    layout = staging.pool.layout

    assert staging.capacity_pages < blocks
    assert [tuple(layer.view.shape) for layer in staging.layers.values()] == [
        (staging.capacity_pages, page_bytes * 2)
    ] * 2

    send = staging.try_stage([[1, 2]])
    assert send is not None
    staging.begin_send(send)
    assert events == [
        (
            "hbm-to-host",
            layer,
            page * page_bytes,
            layout.layers[layer].byte_offset + stage * layout.slot_bytes,
        )
        for page, stage in ((1, 0), (2, 1))
        for layer in range(2)
    ]
    staging.finish_send(send)

    events.clear()
    receive = staging.try_receive([[3]])
    assert receive is not None
    assert staging.finish_receive(receive, success=True) is None
    assert events == [
        (
            "host-to-hbm",
            layer,
            layout.layers[layer].byte_offset,
            3 * page_bytes,
        )
        for layer in range(2)
    ]


def test_connector_rewrites_native_send_and_receive_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import furiosa_opt_vllm.mooncake as mooncake

    send = SimpleNamespace(lease=SimpleNamespace(identifier=1), staging_pages_by_group=[[31, 32]])
    receive = SimpleNamespace(
        lease=SimpleNamespace(identifier=2),
        device_pages_by_group=[[3]],
        staging_pages_by_group=[[41]],
    )

    class _Staging:
        capacity_pages = 4

        def try_stage(self, groups):
            assert groups == [[1, 2]]
            return send

        def begin_send(self, staged):
            assert staged is send

        def try_receive(self, groups):
            assert groups == [[3]]
            return receive

    request = SimpleNamespace(local_block_ids=[[3]])
    metadata = SimpleNamespace(
        reqs_to_send={"send": ("transfer", [[1, 2]])},
        reqs_to_recv={"prefill": {"receive": request}},
    )
    connector = object.__new__(mooncake.FuriosaMooncakeConnector)
    connector._staging = _Staging()
    connector._send_staging = {}
    connector._pending_sends = {}
    connector._receive_staging = {}
    connector._staging_failed_receives = set()
    connector._empty_receives = set()
    connector._get_connector_metadata = lambda: metadata
    observed: dict[str, object] = {}

    def start_load(self, _forward_context, **_kwargs):
        observed["sends"] = dict(metadata.reqs_to_send)
        observed["receive_ids"] = request.local_block_ids

    monkeypatch.setattr(mooncake._MooncakeConnector, "start_load_kv", start_load)

    connector.start_load_kv(object())

    assert observed["sends"] == {"send": ("transfer", [[31, 32]])}
    assert observed["receive_ids"] == [[41]]
    assert connector._send_staging == {"send": send}
    assert connector._receive_staging == {"receive": receive}


def test_connector_restores_send_metadata_if_staging_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import furiosa_opt_vllm.mooncake as mooncake

    send = SimpleNamespace(lease=SimpleNamespace(identifier=1), staging_pages_by_group=[[31]])
    released = []

    class _Staging:
        capacity_pages = 4
        pool = SimpleNamespace(release=released.append)

        def try_stage(self, _groups):
            return send

        def begin_send(self, _staged):
            raise RuntimeError("staging setup failed")

    metadata = SimpleNamespace(reqs_to_send={"send": ("transfer", [[1]])}, reqs_to_recv={})
    original_sends = dict(metadata.reqs_to_send)
    connector = object.__new__(mooncake.FuriosaMooncakeConnector)
    connector._staging = _Staging()
    connector._send_staging = {}
    connector._pending_sends = {}
    connector._receive_staging = {}
    connector._staging_failed_receives = set()
    connector._empty_receives = set()
    connector._get_connector_metadata = lambda: metadata

    def unexpected_start_load(self, *_args, **_kwargs):
        pytest.fail("native Mooncake must not start after staging setup failed")

    monkeypatch.setattr(mooncake._MooncakeConnector, "start_load_kv", unexpected_start_load)

    with pytest.raises(RuntimeError, match="staging setup failed"):
        connector.start_load_kv(object())

    assert metadata.reqs_to_send == original_sends
    assert connector._send_staging == {}
    assert released == [send.lease]


def test_connector_completes_staging_from_native_transfer_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import furiosa_opt_vllm.mooncake as mooncake

    receive_ok = SimpleNamespace(lease=SimpleNamespace(identifier=1))
    receive_failed = SimpleNamespace(lease=SimpleNamespace(identifier=2))
    send_done = SimpleNamespace(lease=SimpleNamespace(identifier=3))
    events = []

    class _Staging:
        def finish_receive(self, staged, *, success):
            events.append(("receive", staged, success))

        def finish_send(self, staged):
            events.append(("send", staged))

    result = SimpleNamespace(
        finished_recving={"ok", "failed"},
        failed_recving={"failed"},
        finished_sending={"sent"},
    )
    connector = object.__new__(mooncake.FuriosaMooncakeConnector)
    connector._staging = _Staging()
    connector._send_staging = {"sent": send_done}
    connector._pending_sends = {}
    connector._receive_staging = {"ok": receive_ok, "failed": receive_failed}
    connector._staging_failed_receives = set()
    connector._empty_receives = set()
    monkeypatch.setattr(
        mooncake._MooncakeConnector,
        "get_transfer_results",
        lambda self, _finished: result,
    )

    returned = connector.get_transfer_results(set())

    assert returned is result
    assert {(event[1].lease.identifier, event[2]) for event in events if event[0] == "receive"} == {
        (1, True),
        (2, False),
    }
    assert events[-1] == ("send", send_done)
    assert connector._receive_staging == {}
    assert connector._send_staging == {}


def test_connector_reports_full_receive_pool_as_native_async_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import furiosa_opt_vllm.mooncake as mooncake

    full_request = SimpleNamespace(local_block_ids=[[3, 4]])
    staged_request = SimpleNamespace(local_block_ids=[[5, 6]])
    staged_receive = SimpleNamespace(
        lease=SimpleNamespace(identifier=1),
        device_pages_by_layer={0: [5, 6]},
        staging_pages_by_group=[[20, 21]],
    )
    metadata = SimpleNamespace(
        reqs_to_send={},
        reqs_to_recv={"prefill": {"full": full_request, "staged": staged_request}},
    )
    original_reqs_to_recv = metadata.reqs_to_recv
    observed: dict[str, object] = {}
    result = SimpleNamespace(
        finished_recving={"staged"},
        failed_recving=set(),
        finished_sending=set(),
    )

    class _Staging:
        def try_receive(self, blocks):
            if blocks == [[3, 4]]:
                return None
            assert blocks == [[5, 6]]
            return staged_receive

        def finish_receive(self, staged, *, success):
            observed["finished_receive"] = (staged, success)

    connector = object.__new__(mooncake.FuriosaMooncakeConnector)
    connector._staging = _Staging()
    connector._send_staging = {}
    connector._pending_sends = {}
    connector._receive_staging = {}
    connector._staging_failed_receives = set()
    connector._empty_receives = set()
    connector._get_connector_metadata = lambda: metadata
    connector._require_staging = lambda: connector._staging

    def start_load(self, _forward_context, **_kwargs):
        observed["reqs_to_recv"] = metadata.reqs_to_recv
        observed["receive_ids"] = staged_request.local_block_ids

    monkeypatch.setattr(mooncake._MooncakeConnector, "start_load_kv", start_load)
    monkeypatch.setattr(
        mooncake._MooncakeConnector,
        "get_transfer_results",
        lambda self, _finished: result,
    )

    connector.start_load_kv(object())

    assert observed["reqs_to_recv"] == {"prefill": {"staged": staged_request}}
    assert observed["receive_ids"] == [[20, 21]]
    assert metadata.reqs_to_recv is original_reqs_to_recv
    assert connector.get_transfer_results(set()) is result
    assert result.finished_recving == {"full", "staged"}
    assert result.failed_recving == {"full"}
    assert observed["finished_receive"] == (staged_receive, True)


def test_connector_defers_receive_until_scheduler_staging_is_released(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from furiosa_opt_engine.cache_transfer import CacheStagingSlots
    from furiosa_opt_vllm.mooncake import FuriosaMooncakeConnector, _MooncakeConnector

    connector = object.__new__(FuriosaMooncakeConnector)
    connector._receive_slots = CacheStagingSlots(capacity_pages=1)
    connector._receive_leases = {}
    connector._active_receives = set()
    connector._request_pages = lambda _tokens: 1
    monkeypatch.setattr(
        _MooncakeConnector,
        "get_num_new_matched_tokens",
        lambda _self, _request, _computed: (4, True),
    )
    first = SimpleNamespace(request_id="first")
    second = SimpleNamespace(request_id="second")

    assert connector.get_num_new_matched_tokens(first, 0) == (4, True)
    assert connector.get_num_new_matched_tokens(second, 0) == (None, False)

    connector._release_receive(first.request_id)

    assert connector.get_num_new_matched_tokens(second, 0) == (4, True)
    connector._release_receive(second.request_id)


def test_mooncake_reservation_skips_page_list_copies_when_pool_is_full() -> None:
    from furiosa_opt_vllm.mooncake import _CacheStaging

    class _UncopyablePages:
        def __len__(self) -> int:
            return 1

        def __iter__(self):
            raise AssertionError("a full staging pool must not copy request page lists")

    staging = object.__new__(_CacheStaging)
    staging.group_count = 1
    staging.pool = SimpleNamespace(try_acquire=lambda _count: None)
    staging.layers = {}

    assert staging._try_reserve([_UncopyablePages()]) is None


def test_pending_send_enqueue_failure_preserves_queue_and_releases_staging() -> None:
    from furiosa_opt_vllm.mooncake import FuriosaMooncakeConnector

    pending = {"send": ("transfer", [[1, 2]])}
    staged = SimpleNamespace(
        lease=SimpleNamespace(identifier=1),
        staging_pages_by_group=[[31, 32]],
    )
    released = []

    class _Staging:
        capacity_pages = 4

        def try_stage(self, groups):
            assert groups == [[1, 2]]
            return staged

        def begin_send(self, item):
            assert item is staged

        def finish_send(self, item):
            released.append(item)

    def fail_enqueue(_metadata):
        raise RuntimeError("sender event loop is closed")

    connector = object.__new__(FuriosaMooncakeConnector)
    connector._pending_sends = dict(pending)
    connector._send_staging = {}
    connector.connector_worker = SimpleNamespace(start_load_kv=fail_enqueue)

    with pytest.raises(RuntimeError, match="sender event loop is closed"):
        connector._flush_pending_sends(_Staging())

    assert connector._pending_sends == pending
    assert connector._send_staging == {}
    assert released == [staged]


def test_connector_restores_receive_metadata_if_native_start_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import furiosa_opt_vllm.mooncake as mooncake

    request = SimpleNamespace(local_block_ids=[[3, 4]])
    metadata = SimpleNamespace(
        reqs_to_send={},
        reqs_to_recv={"prefill": {"receive": request}},
    )
    original_reqs_to_recv = metadata.reqs_to_recv
    observed: dict[str, object] = {}

    class _Staging:
        def try_receive(self, _blocks):
            return None

    connector = object.__new__(mooncake.FuriosaMooncakeConnector)
    connector._staging = _Staging()
    connector._send_staging = {}
    connector._pending_sends = {}
    connector._receive_staging = {}
    connector._staging_failed_receives = set()
    connector._empty_receives = set()
    connector._get_connector_metadata = lambda: metadata
    connector._require_staging = lambda: connector._staging

    def fail_native_start(self, _forward_context, **_kwargs):
        observed["reqs_to_recv"] = metadata.reqs_to_recv
        raise RuntimeError("native Mooncake start failed")

    monkeypatch.setattr(mooncake._MooncakeConnector, "start_load_kv", fail_native_start)

    with pytest.raises(RuntimeError, match="native Mooncake start failed"):
        connector.start_load_kv(object())

    assert observed["reqs_to_recv"] == {"prefill": {}}
    assert metadata.reqs_to_recv is original_reqs_to_recv


def test_native_start_failure_keeps_inflight_receives_and_rolls_back_sends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import furiosa_opt_vllm.mooncake as mooncake

    send = SimpleNamespace(
        lease=SimpleNamespace(identifier=1),
        staging_pages_by_group=[[31]],
    )
    receive = SimpleNamespace(
        lease=SimpleNamespace(identifier=2),
        device_pages_by_group=[[3]],
        staging_pages_by_group=[[41]],
    )
    receive_request = SimpleNamespace(local_block_ids=[[3]])
    original_sends = {"send": ("transfer", [[1]])}
    metadata = SimpleNamespace(
        reqs_to_send=dict(original_sends),
        reqs_to_recv={"prefill": {"receive": receive_request}},
    )
    finished_sends = []

    class _Staging:
        capacity_pages = 4
        pool = SimpleNamespace(release=lambda _lease: None)

        def try_stage(self, _groups):
            return send

        def begin_send(self, _staged):
            pass

        def try_receive(self, _groups):
            return receive

        def finish_send(self, staged):
            finished_sends.append(staged)

        def finish_receive(self, *_args, **_kwargs):
            pytest.fail("a receive may already be in flight")

    connector = object.__new__(mooncake.FuriosaMooncakeConnector)
    connector._staging = _Staging()
    connector._send_staging = {}
    connector._pending_sends = {}
    connector._receive_staging = {}
    connector._staging_failed_receives = set()
    connector._empty_receives = set()
    connector._get_connector_metadata = lambda: metadata
    connector._require_staging = lambda: connector._staging

    def fail_native_start(self, _forward_context, **_kwargs):
        assert metadata.reqs_to_send == {"send": ("transfer", [[31]])}
        assert receive_request.local_block_ids == [[41]]
        raise RuntimeError("native send enqueue failed")

    monkeypatch.setattr(mooncake._MooncakeConnector, "start_load_kv", fail_native_start)

    with pytest.raises(RuntimeError, match="native send enqueue failed"):
        connector.start_load_kv(object())

    assert metadata.reqs_to_send == original_sends
    assert connector._send_staging == {}
    assert finished_sends == [send]
    assert connector._receive_staging == {"receive": receive}
    assert receive_request.local_block_ids == [[41]]
