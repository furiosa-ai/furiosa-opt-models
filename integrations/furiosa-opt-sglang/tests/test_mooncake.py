from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest
import torch

from furiosa_opt_engine._host_region import HostRegionRegistrar, host_slot_region
from furiosa_opt_engine._mooncake import (
    MooncakeIOGuard,
    UnconfirmedMooncakeIO,
    require_confirmed_get,
    require_confirmed_put,
)
from furiosa_opt_sglang.cache import Pages
from furiosa_opt_sglang.mooncake import (
    FuriosaHiCacheMooncake,
    PoolName,
    PoolTransfer,
    _RegisteredHostPool,
)


class _HostCache:
    def __init__(self, slots: list[bytearray]) -> None:
        self.slots = slots

    def slot(self, index: int) -> _HostSlot:
        return _HostSlot(self.slots[index])


class _HostSlot:
    def __init__(self, buffer: bytearray) -> None:
        self.buffer = buffer

    def access(self) -> bytearray:
        return self.buffer


class _HostPool:
    page_num = 5
    page_size = 2
    dtype = torch.float16
    layer_num = 2
    start_layer = 0
    end_layer = 2
    chips = 1
    sizes = [4, 4]

    def __init__(self, slots: list[bytearray]) -> None:
        self.host_cache = _HostCache(slots)

    def _page_index(self, token_index: int) -> int:
        return token_index // self.page_size


class _FakeStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.regions: dict[int, int] = {}
        self.get_results: list[int] | None = None
        self.put_results: list[int] | None = None
        self.existence_results: list[int] | None = None
        self.put_error: BaseException | None = None
        self.events: list[tuple[str, int]] = []
        self.close_calls = 0

    def register_buffer(self, address: int, size: int) -> int:
        self.regions[address] = size
        self.events.append(("register", address))
        return 0

    def unregister_buffer(self, address: int) -> int:
        del self.regions[address]
        self.events.append(("unregister", address))
        return 0

    def close(self) -> None:
        self.close_calls += 1

    def batch_put_from(self, keys: list[str], addresses: list[int], sizes: list[int]) -> list[int]:
        if self.put_error is not None:
            raise self.put_error
        if self.put_results is not None:
            return self.put_results
        for key, address, size in zip(keys, addresses, sizes, strict=True):
            self.objects[key] = ctypes.string_at(address, size)
        return [0] * len(keys)

    def batch_get_into(self, keys: list[str], addresses: list[int], sizes: list[int]) -> list[int]:
        if self.get_results is not None:
            return self.get_results
        results = []
        for key, address, size in zip(keys, addresses, sizes, strict=True):
            value = self.objects.get(key)
            if value is None:
                results.append(-704)
                continue
            assert len(value) == size
            ctypes.memmove(address, value, size)
            results.append(size)
        return results


def _adapter(pool: _HostPool, store: _FakeStore) -> FuriosaHiCacheMooncake:
    adapter = FuriosaHiCacheMooncake.__new__(FuriosaHiCacheMooncake)
    adapter._registered_pool = _RegisteredHostPool(pool, 4, 2, "layout")
    adapter._pointer_io = MooncakeIOGuard()
    adapter.storage_config = SimpleNamespace(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
    )
    adapter.store = store
    adapter._tag_keys = lambda keys: keys
    adapter._batch_exist = lambda keys, *_args: (
        store.existence_results
        if store.existence_results is not None
        else [int(key in store.objects) for key in keys]
    )
    adapter._region_registrar = HostRegionRegistrar(
        adapter._register_mooncake_region, adapter._unregister_mooncake_region
    )
    return adapter


def test_sglang_mooncake_maps_and_transfers_whole_engine_pages() -> None:
    slots = [bytearray(b"0000") for _ in range(5)]
    slots[1][:] = b"page"
    slots[4][:] = b"data"
    store = _FakeStore()
    adapter = _adapter(_HostPool(slots), store)
    indices = torch.tensor([2, 3, 8, 9])

    assert adapter.batch_set_v1(["a", "b"], indices) == [True, True]
    keys = adapter._keys(["a", "b"])
    assert [store.objects[key] for key in keys] == [b"page", b"data"]

    slots[1][:] = b"0000"
    slots[4][:] = b"0000"
    assert adapter.batch_get_v1(["a", "b"], indices) == [True, True]
    assert (slots[1], slots[4]) == (b"page", b"data")
    assert len(store.regions) == 2


def test_sglang_put_checks_existence_before_leasing_only_missing_pages(monkeypatch) -> None:
    slots = [bytearray(b"zero") for _ in range(5)]
    slots[1][:] = b"page"
    store = _FakeStore()
    pool = _HostPool(slots)
    adapter = _adapter(pool, store)
    store.objects[adapter._keys(["present"])[0]] = b"old!"
    events: list[tuple[str, int | None]] = []

    def page_index(token_index: int) -> int:
        events.append(("page_index", token_index))
        return token_index // pool.page_size

    def slot(index: int) -> _HostSlot:
        events.append(("slot", index))
        return _HostSlot(slots[index])

    pool._page_index = page_index
    pool.host_cache.slot = slot
    original_region = host_slot_region

    def lease(buffer):
        events.append(("lease", None))
        return original_region(buffer)

    monkeypatch.setattr("furiosa_opt_sglang.mooncake.host_slot_region", lease)
    original_exists = adapter._batch_exist

    def exists(keys: list[str], *_args) -> list[int]:
        events.append(("exists", None))
        return original_exists(keys)

    adapter._batch_exist = exists

    result = adapter.batch_set_v1(["present", "missing"], torch.tensor([0, 1, 2, 3]))

    assert result == [True, True]
    assert store.objects[adapter._keys(["missing"])[0]] == b"page"
    assert events[0] == ("exists", None)
    assert [index for event, index in events if event == "page_index"] == [2]
    assert [index for event, index in events if event == "slot"] == [1]
    assert sum(event == "lease" for event, _index in events) == 1


def test_sglang_mooncake_v2_transfers_kv_pool_pages() -> None:
    slots = [bytearray(b"0000") for _ in range(5)]
    slots[1][:] = b"page"
    slots[4][:] = b"data"
    store = _FakeStore()
    adapter = _adapter(_HostPool(slots), store)
    transfer = SimpleNamespace(
        name=PoolName.KV,
        keys=["a", "b"],
        host_indices=torch.tensor([2, 3, 8, 9]),
    )

    assert adapter.batch_set_v2([transfer]) == {PoolName.KV: [True, True]}
    keys = adapter._keys(["a", "b"])
    slots[1][:] = b"0000"
    slots[4][:] = b"0000"
    assert adapter.batch_get_v2([transfer]) == {PoolName.KV: [True, True]}
    assert [store.objects[key] for key in keys] == [b"page", b"data"]
    assert (slots[1], slots[4]) == (b"page", b"data")


def test_sglang_mooncake_v2_registers_the_pool_before_transfer(monkeypatch) -> None:
    slots = [bytearray(8) for _ in range(5)]
    slots[1][:] = b"pagepage"
    pool = _HostPool(slots)
    store = _FakeStore()
    adapter = FuriosaHiCacheMooncake.__new__(FuriosaHiCacheMooncake)
    adapter._registered_pool = None
    adapter._pointer_io = MooncakeIOGuard()
    adapter.storage_config = SimpleNamespace(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
    )
    adapter.store = store
    adapter._tag_keys = lambda keys: keys
    adapter._batch_exist = lambda keys, *_args: [int(key in store.objects) for key in keys]
    adapter._region_registrar = HostRegionRegistrar(
        adapter._register_mooncake_region, adapter._unregister_mooncake_region
    )
    base_registrations = []

    def register_base(storage, registered_pool, name):
        base_registrations.append((storage, registered_pool, name))

    from sglang.srt.mem_cache.hicache_storage import HiCacheStorage

    monkeypatch.setattr(HiCacheStorage, "register_mem_host_pool_v2", register_base)
    adapter.register_mem_host_pool_v2(pool, PoolName.KV)

    assert base_registrations == [(adapter, pool, PoolName.KV)]
    assert adapter._registered_pool.pool is pool
    assert adapter._registered_pool.page_bytes == 8
    assert adapter.gb_per_page == 8 / (1 << 30)

    transfer = PoolTransfer(
        name=PoolName.KV,
        keys=["registered-page"],
        host_indices=torch.tensor([2, 3]),
    )
    try:
        assert adapter.batch_set_v2([transfer]) == {PoolName.KV: [True]}
        slots[1][:] = bytes(8)
        assert adapter.batch_get_v2([transfer]) == {PoolName.KV: [True]}
        assert slots[1] == b"pagepage"
    finally:
        adapter._region_registrar.close()


@pytest.mark.parametrize(
    "difference",
    ["same", "layout", "tp_rank", "pp_rank", "attn_cp_rank", "topology", "tag"],
)
def test_sglang_mooncake_keys_share_only_compatible_namespaces(difference: str) -> None:
    store = _FakeStore()
    writer_pool = _HostPool([bytearray(8) for _ in range(5)])
    writer = _adapter(writer_pool, store)
    writer.storage_config = SimpleNamespace(
        tp_rank=0,
        tp_size=2 if difference == "tp_rank" else 1,
        pp_rank=0,
        pp_size=2 if difference == "pp_rank" else 1,
        attn_cp_rank=0,
        attn_cp_size=2 if difference == "attn_cp_rank" else 1,
    )
    writer._registered_pool = writer._build_registered_pool(writer_pool)
    writer._tag_keys = lambda keys: [f"model:{key}" for key in keys]
    assert writer.batch_set_v1(["shared-page"], torch.tensor([2, 3])) == [True]

    reader_pool = _HostPool([bytearray(8) for _ in range(5)])
    reader = _adapter(reader_pool, store)
    reader.storage_config = SimpleNamespace(
        tp_rank=1 if difference == "tp_rank" else 0,
        tp_size=2 if difference in ("tp_rank", "topology") else 1,
        pp_rank=1 if difference == "pp_rank" else 0,
        pp_size=2 if difference == "pp_rank" else 1,
        attn_cp_rank=1 if difference == "attn_cp_rank" else 0,
        attn_cp_size=2 if difference == "attn_cp_rank" else 1,
    )
    if difference == "layout":
        reader_pool.dtype = torch.float32
    reader._registered_pool = reader._build_registered_pool(reader_pool)
    tag = "other-model" if difference == "tag" else "model"
    reader._tag_keys = lambda keys: [f"{tag}:{key}" for key in keys]

    assert reader.batch_exists(["shared-page"]) == (1 if difference == "same" else 0)


def test_sglang_mooncake_v2_reports_prefix_and_absent_pool_results() -> None:
    adapter = FuriosaHiCacheMooncake.__new__(FuriosaHiCacheMooncake)
    adapter._registered_pool = _RegisteredHostPool(_HostPool([]), 4, 2, "layout")
    adapter._pointer_io = MooncakeIOGuard()
    adapter.storage_config = SimpleNamespace(tp_rank=0, pp_rank=0, attn_cp_rank=0)
    adapter._tag_keys = lambda keys: keys
    adapter._batch_exist = lambda _keys, *_args: [1, 0]
    kv_transfer = SimpleNamespace(name=PoolName.KV)

    result = adapter.batch_exists_v2(["hit", "miss"], [kv_transfer])

    assert result.kv_hit_pages == 1
    assert result.extra_pool_hit_pages == {PoolName.KV: 1}

    adapter._batch_exist = lambda _keys, *_args: [0]
    result = adapter.batch_exists_v2(["miss"], [kv_transfer])
    assert result.kv_hit_pages == 0
    assert result.extra_pool_hit_pages == {}


@pytest.mark.parametrize("method", ["batch_exists_v2", "batch_get_v2", "batch_set_v2"])
def test_sglang_mooncake_v2_rejects_sidecar_pools(method: str) -> None:
    adapter = FuriosaHiCacheMooncake.__new__(FuriosaHiCacheMooncake)
    transfer = SimpleNamespace(name="sidecar", keys=["a"], host_indices=torch.tensor([0, 1]))

    with pytest.raises(ValueError, match="sidecar pool"):
        if method == "batch_exists_v2":
            adapter.batch_exists_v2(["a"], [transfer])
        else:
            getattr(adapter, method)([transfer])


@pytest.mark.parametrize("method", ["batch_get_v2", "batch_set_v2"])
@pytest.mark.parametrize("missing", ["keys", "host_indices"])
def test_sglang_mooncake_v2_requires_kv_transfer_data(method: str, missing: str) -> None:
    adapter = FuriosaHiCacheMooncake.__new__(FuriosaHiCacheMooncake)
    transfer = SimpleNamespace(
        name=PoolName.KV,
        keys=None if missing == "keys" else ["a"],
        host_indices=None if missing == "host_indices" else torch.tensor([0, 1]),
    )

    with pytest.raises(ValueError, match="require keys and host indices"):
        getattr(adapter, method)([transfer])


def test_sglang_mooncake_retains_page_when_completion_is_unknown() -> None:
    slots = [bytearray(b"0000") for _ in range(5)]
    store = _FakeStore()
    store.get_results = [-123456]
    adapter = _adapter(_HostPool(slots), store)

    with pytest.raises(UnconfirmedMooncakeIO):
        adapter.batch_get_v1(["a"], torch.tensor([2, 3]))

    with pytest.raises(BufferError):
        slots[1].extend(b"x")
    with pytest.raises(UnconfirmedMooncakeIO, match="worker must exit"):
        adapter._pointer_io.ensure_safe()


@pytest.mark.parametrize(
    ("results", "raises"),
    [([-123456], False), ([], False), (None, True)],
    ids=["unknown-code", "incomplete-result", "call-raised"],
)
def test_sglang_mooncake_retains_page_when_put_completion_is_unknown(
    results: list[int] | None, raises: bool
) -> None:
    slots = [bytearray(b"0000") for _ in range(5)]
    store = _FakeStore()
    store.put_results = results
    if raises:
        store.put_error = RuntimeError("transport failed")
    adapter = _adapter(_HostPool(slots), store)

    with pytest.raises(UnconfirmedMooncakeIO):
        adapter.batch_set_v1(["a"], torch.tensor([2, 3]))

    with pytest.raises(BufferError):
        slots[1].extend(b"x")
    with pytest.raises(UnconfirmedMooncakeIO, match="worker must exit"):
        adapter._pointer_io.ensure_safe()


def test_sglang_confirmed_put_rejection_releases_page_lease() -> None:
    slots = [bytearray(b"0000") for _ in range(5)]
    store = _FakeStore()
    store.put_results = [-705]
    adapter = _adapter(_HostPool(slots), store)

    assert adapter.batch_set_v1(["a"], torch.tensor([2, 3])) == [False]

    slots[1].extend(b"x")
    adapter._pointer_io.ensure_safe()


def test_sglang_rejects_malformed_store_existence_results() -> None:
    slots = [bytearray(b"0000") for _ in range(5)]
    store = _FakeStore()
    store.existence_results = [-1]
    adapter = _adapter(_HostPool(slots), store)

    with pytest.raises(RuntimeError, match="invalid existence result"):
        adapter.batch_set_v1(["key"], torch.tensor([2, 3]))

    assert not store.regions
    slots[1].extend(b"x")


def test_sglang_pages_parse_requires_contiguous_engine_layout() -> None:
    pool = SimpleNamespace(
        block=2,
        bytes=32,
        element_bytes=2,
        shape=[2, 2, 2, 2],
        strides=[16, 8, 4, 2],
    )
    decoder = SimpleNamespace(
        model=SimpleNamespace(kv_cache="bfloat16", parallel=1),
        pool=pool,
        layers=range(2),
    )

    pages = Pages.parse(decoder, heads=2, width=2)
    assert pages.dtype is torch.bfloat16

    pool.strides = [16, 4, 8, 2]
    with pytest.raises(ValueError, match="contiguous"):
        Pages.parse(decoder, heads=2, width=2)

    pool.strides = [16, 8, 4, 2]
    pool.shape = [2, 1, 4, 2]
    with pytest.raises(ValueError, match="contiguous"):
        Pages.parse(decoder, heads=2, width=2)

    pool.shape = [2, 2, 2, 2]
    pool.element_bytes = 1
    with pytest.raises(ValueError, match="contiguous"):
        Pages.parse(decoder, heads=2, width=2)

    pool.element_bytes = 2
    pool.bytes = 31
    with pytest.raises(ValueError, match="bytes a block"):
        Pages.parse(decoder, heads=2, width=2)


def test_sglang_registered_pool_requires_slot_size_to_match_layer_layout() -> None:
    slots = [bytearray(8) for _ in range(5)]
    pool = _HostPool(slots)
    adapter = FuriosaHiCacheMooncake.__new__(FuriosaHiCacheMooncake)
    adapter.storage_config = SimpleNamespace(tp_size=1, pp_size=1, attn_cp_size=1)

    registered = adapter._build_registered_pool(pool)
    assert registered.page_bytes == 8

    pool.sizes = [4, 3]
    with pytest.raises(ValueError, match="layer page sizes disagree"):
        adapter._build_registered_pool(pool)


def test_sglang_close_unregisters_pages_and_keeps_store_ownership() -> None:
    slots = [bytearray(b"0000") for _ in range(5)]
    store = _FakeStore()
    adapter = _adapter(_HostPool(slots), store)
    assert adapter.batch_set_v1(["a"], torch.tensor([2, 3])) == [True]

    adapter.close()

    assert not store.regions
    assert store.close_calls == 0


def test_sglang_close_preserves_unsafe_pages_and_store() -> None:
    slots = [bytearray(b"0000") for _ in range(5)]
    store = _FakeStore()
    store.get_results = [-123456]
    adapter = _adapter(_HostPool(slots), store)
    with pytest.raises(UnconfirmedMooncakeIO):
        adapter.batch_get_v1(["a"], torch.tensor([2, 3]))

    with pytest.raises(UnconfirmedMooncakeIO):
        adapter.close()

    assert len(store.regions) == 1
    assert store.close_calls == 0


def test_sglang_clear_does_not_clear_shared_mooncake_store() -> None:
    removed: list[bool] = []
    adapter = FuriosaHiCacheMooncake.__new__(FuriosaHiCacheMooncake)
    adapter.store = SimpleNamespace(remove_all=lambda: removed.append(True))

    adapter.clear()

    assert not removed


def test_sglang_rejects_non_kv_pool_before_mutating_registration() -> None:
    adapter = FuriosaHiCacheMooncake.__new__(FuriosaHiCacheMooncake)
    pool = object()

    with pytest.raises(ValueError, match="supports the engine KV pool"):
        adapter.register_mem_host_pool_v2(pool, "sidecar")

    assert not hasattr(adapter, "registered_pools")


def test_sglang_exists_rejects_incomplete_store_result() -> None:
    adapter = FuriosaHiCacheMooncake.__new__(FuriosaHiCacheMooncake)
    adapter._registered_pool = _RegisteredHostPool(_HostPool([]), 4, 2, "layout")
    adapter._pointer_io = MooncakeIOGuard()
    adapter.storage_config = SimpleNamespace(tp_rank=0, pp_rank=0, attn_cp_rank=0)
    adapter._tag_keys = lambda keys: keys
    adapter._batch_exist = lambda _keys: []

    with pytest.raises(RuntimeError, match="incomplete existence"):
        adapter.exists("key")


def test_mooncake_get_accepts_complete_and_safe_rejection_results() -> None:
    assert require_confirmed_get([4096, -704], [4096, 4096]) == [4096, -704]
    assert require_confirmed_get([0], [4096]) == [0]


@pytest.mark.parametrize("results", [[-123456], [4097], [True], []])
def test_mooncake_get_rejects_unconfirmed_results(results: list[object]) -> None:
    with pytest.raises(UnconfirmedMooncakeIO):
        require_confirmed_get(results, [4096])


@pytest.mark.parametrize("result", [-707, -900, -901])
def test_mooncake_get_treats_lease_and_rpc_errors_as_unconfirmed(result: int) -> None:
    with pytest.raises(UnconfirmedMooncakeIO):
        require_confirmed_get([result], [4096])


def test_mooncake_put_accepts_success_and_safe_rejection_results() -> None:
    assert require_confirmed_put([0, -705], 2) == [0, -705]


@pytest.mark.parametrize("results", [[-123456], [1], [False], []])
def test_mooncake_put_rejects_unconfirmed_results(results: list[object]) -> None:
    with pytest.raises(UnconfirmedMooncakeIO):
        require_confirmed_put(results, 1)


@pytest.mark.parametrize("result", [-900, -901])
def test_mooncake_put_treats_rpc_errors_as_unconfirmed(result: int) -> None:
    with pytest.raises(UnconfirmedMooncakeIO):
        require_confirmed_put([result], 1)


def test_mooncake_io_guard_retains_slots_after_unconfirmed_completion() -> None:
    slot = bytearray(b"page")
    guard = MooncakeIOGuard()

    with host_slot_region(slot) as region:
        with pytest.raises(UnconfirmedMooncakeIO):
            guard.run_pointer_io(
                owner=object(),
                regions=[region],
                operation=lambda: [-123456],
                validate=lambda result: require_confirmed_get(result, [4]),
                operation_name="GET",
            )

    with pytest.raises(BufferError):
        slot.extend(b"x")
    with pytest.raises(UnconfirmedMooncakeIO, match="worker must exit"):
        guard.ensure_safe()

    operation_started = False

    def operation_after_poison() -> list[int]:
        nonlocal operation_started
        operation_started = True
        return [4]

    with pytest.raises(UnconfirmedMooncakeIO, match="worker must exit"):
        guard.run_pointer_io(
            owner=object(),
            regions=[],
            operation=operation_after_poison,
            validate=lambda result: require_confirmed_get(result, [4]),
            operation_name="GET",
        )
    assert not operation_started
