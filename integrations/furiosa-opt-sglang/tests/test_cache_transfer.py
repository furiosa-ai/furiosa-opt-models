from __future__ import annotations

import ctypes
import mmap
import os
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from threading import Barrier
from types import SimpleNamespace
from typing import cast

import pytest

from furiosa_opt_engine import _host_region as host_region_module
from furiosa_opt_engine import cache_transfer as cache_transfer_module
from furiosa_opt_engine._host_region import (
    HostRegion,
    HostRegionLease,
    HostRegionRegistrar,
    host_slot_region,
)
from furiosa_opt_engine.cache_transfer import (
    AlignedFileIO,
    BoundedCacheStaging,
    _probe_direct_io,
    _probe_o_direct,
)


def _aligned_view(size: int, alignment: int) -> tuple[mmap.mmap, memoryview]:
    mapping = mmap.mmap(-1, size + alignment - 1)
    address = ctypes.addressof(ctypes.c_ubyte.from_buffer(mapping))
    offset = (-address) % alignment
    return mapping, memoryview(mapping)[offset : offset + size]


def _misaligned_view(size: int, alignment: int) -> tuple[bytearray, memoryview]:
    storage = bytearray(size + alignment)
    address = ctypes.addressof(ctypes.c_ubyte.from_buffer(storage))
    offset = (1 - address % alignment) % alignment
    view = memoryview(storage)[offset : offset + size]
    assert (address + offset) % alignment != 0
    return storage, view


def _leased_region(leases: ExitStack, size: int) -> HostRegionLease:
    return leases.enter_context(host_slot_region(bytearray(size)))


@pytest.mark.parametrize("use_direct", [False, True], ids=["buffered", "direct"])
def test_aligned_file_io_round_trips_pages(tmp_path, use_direct: bool) -> None:
    if use_direct and not _probe_o_direct(str(tmp_path)):
        pytest.skip("temporary filesystem does not support O_DIRECT")

    io = AlignedFileIO(str(tmp_path), use_direct=use_direct)
    source = bytearray(index % 251 for index in range(5003))
    destination = bytearray(len(source))
    path = tmp_path / "cache.bin"

    io.write(str(path), source)
    io.read(str(path), destination)

    assert destination == source
    assert path.stat().st_size == len(source)


def test_buffered_file_io_uses_host_slots_without_a_bounce_copy(tmp_path, monkeypatch) -> None:
    io = AlignedFileIO(str(tmp_path), use_direct=False)
    source = bytearray(range(128))
    destination = bytearray(len(source))

    def unexpected_bounce(_size: int) -> memoryview:
        pytest.fail("buffered I/O should pass the pinned slot directly")

    monkeypatch.setattr(io, "_buffer", unexpected_bounce)
    path = tmp_path / "buffered-slot.bin"
    io.write(str(path), source)
    io.read(str(path), destination)

    assert destination == source
    assert path.stat().st_size == len(source)


def test_aligned_file_io_reports_short_writes(tmp_path, monkeypatch) -> None:
    io = AlignedFileIO(str(tmp_path), use_direct=False)
    path = tmp_path / "short-write.bin"
    source = bytearray(5003)
    io.prepare(str(path), len(source), 1)
    monkeypatch.setattr(os, "pwrite", lambda fd, buffer, offset: len(buffer) - 1)

    with pytest.raises(OSError, match="short write"):
        io.write(str(path), source)


def test_aligned_file_io_reports_short_reads(tmp_path, monkeypatch) -> None:
    io = AlignedFileIO(str(tmp_path), use_direct=False)
    path = tmp_path / "short-read.bin"
    destination = bytearray(5003)
    io.prepare(str(path), len(destination), 1)
    monkeypatch.setattr(os, "preadv", lambda fd, buffers, offset: len(buffers[0]) - 1)

    with pytest.raises(OSError, match="short read"):
        io.read(str(path), destination)


def test_vectored_file_io_uses_o_direct_for_aligned_buffers(tmp_path, monkeypatch) -> None:
    alignment = _probe_direct_io(str(tmp_path))
    if alignment is None:
        pytest.skip("temporary filesystem does not support O_DIRECT")
    memory_alignment, offset_alignment = alignment
    io = AlignedFileIO(str(tmp_path), use_direct=True)
    assert io.use_direct
    path = tmp_path / "direct-vectors.bin"
    offset = 2 * offset_alignment
    length = offset_alignment
    io.prepare(str(path), length, 2, file_bytes=offset + 2 * length)

    source_owners = [_aligned_view(length, memory_alignment) for _ in range(2)]
    source_views = [view for _, view in source_owners]
    payloads = [bytes([index + 17]) * length for index in range(2)]
    for view, payload in zip(source_views, payloads, strict=True):
        view[:] = payload

    flags = []
    open_file = os.open

    def observe_open(path, mode, *args, **kwargs):
        flags.append(mode)
        return open_file(path, mode, *args, **kwargs)

    monkeypatch.setattr(os, "open", observe_open)
    destination_owners = [_aligned_view(length, memory_alignment) for _ in range(2)]
    destination_views = [view for _, view in destination_owners]
    try:
        io.writev_at(str(path), offset, source_views)
        io.readv_at(str(path), offset, destination_views)
        assert [bytes(view) for view in destination_views] == payloads
        assert any(mode & os.O_DIRECT for mode in flags)
    finally:
        for view in (*source_views, *destination_views):
            view.release()
        for mapping, _ in (*source_owners, *destination_owners):
            mapping.close()


def test_vectored_file_io_falls_back_for_misaligned_address(tmp_path, monkeypatch) -> None:
    alignment = _probe_direct_io(str(tmp_path))
    if alignment is None:
        pytest.skip("temporary filesystem does not support O_DIRECT")
    memory_alignment, offset_alignment = alignment
    if memory_alignment == 1:
        pytest.skip("filesystem accepts every buffer address")
    io = AlignedFileIO(str(tmp_path), use_direct=True)
    path = tmp_path / "buffered-vectors.bin"
    length = offset_alignment
    io.prepare(str(path), length, 2)
    storage, source = _misaligned_view(length, memory_alignment)
    source[:] = b"x" * length
    aligned_owner, aligned = _aligned_view(length, memory_alignment)
    aligned[:] = b"y" * length
    flags = []
    open_file = os.open

    def observe_open(path, mode, *args, **kwargs):
        flags.append(mode)
        return open_file(path, mode, *args, **kwargs)

    monkeypatch.setattr(os, "open", observe_open)
    try:
        io.writev_at(str(path), 0, (source, aligned))
        assert path.read_bytes() == b"x" * length + b"y" * length
        assert flags and all(not mode & os.O_DIRECT for mode in flags)
    finally:
        source.release()
        aligned.release()
        aligned_owner.close()


def test_vectored_file_io_many_reuses_direct_file_handle(tmp_path, monkeypatch) -> None:
    alignment = _probe_direct_io(str(tmp_path))
    if alignment is None:
        pytest.skip("temporary filesystem does not support O_DIRECT")
    memory_alignment, offset_alignment = alignment
    io = AlignedFileIO(str(tmp_path), use_direct=True)
    path = tmp_path / "direct-ranges.bin"
    length = offset_alignment
    io.prepare(str(path), length, 4)

    source_owners = [_aligned_view(length, memory_alignment) for _ in range(2)]
    source_views = [view for _, view in source_owners]
    payloads = [bytes([index + 31]) * length for index in range(2)]
    for view, payload in zip(source_views, payloads, strict=True):
        view[:] = payload
    destination_owners = [_aligned_view(length, memory_alignment) for _ in range(4)]
    destination_views = [view for _, view in destination_owners]

    flags = []
    open_file = os.open

    def observe_open(path, mode, *args, **kwargs):
        flags.append(mode)
        return open_file(path, mode, *args, **kwargs)

    monkeypatch.setattr(os, "open", observe_open)
    try:
        io.writev_at_many(str(path), ((0, source_views), (2 * length, source_views)))
        io.readv_at_many(
            str(path),
            ((0, destination_views[:2]), (2 * length, destination_views[2:])),
        )
        assert [bytes(view) for view in destination_views] == payloads * 2
        assert len(flags) == 2
        assert all(mode & os.O_DIRECT for mode in flags)
    finally:
        for view in (*source_views, *destination_views):
            view.release()
        for mapping, _ in (*source_owners, *destination_owners):
            mapping.close()


def test_vectored_file_io_many_keeps_per_range_alignment_fallback(tmp_path, monkeypatch) -> None:
    alignment = _probe_direct_io(str(tmp_path))
    if alignment is None:
        pytest.skip("temporary filesystem does not support O_DIRECT")
    memory_alignment, offset_alignment = alignment
    if memory_alignment == 1:
        pytest.skip("filesystem accepts every buffer address")
    io = AlignedFileIO(str(tmp_path), use_direct=True)
    path = tmp_path / "mixed-ranges.bin"
    length = offset_alignment
    io.prepare(str(path), length, 4)

    storage, misaligned = _misaligned_view(length, memory_alignment)
    aligned_owner, aligned = _aligned_view(length, memory_alignment)
    misaligned[:] = b"m" * length
    aligned[:] = b"a" * length
    flags = []
    open_file = os.open

    def observe_open(path, mode, *args, **kwargs):
        flags.append(mode)
        return open_file(path, mode, *args, **kwargs)

    monkeypatch.setattr(os, "open", observe_open)
    try:
        io.writev_at_many(
            str(path),
            ((0, (misaligned, aligned)), (2 * length, (aligned, aligned))),
        )
        assert path.read_bytes() == b"m" * length + b"a" * (3 * length)
        assert len(flags) == 2
        assert any(mode & os.O_DIRECT for mode in flags)
        assert any(not mode & os.O_DIRECT for mode in flags)
    finally:
        misaligned.release()
        aligned.release()
        aligned_owner.close()
        del storage


def test_aligned_file_io_uses_direct_io_for_aligned_pages(tmp_path, monkeypatch) -> None:
    alignment = cache_transfer_module._probe_direct_io(str(tmp_path))
    if alignment is None:
        pytest.skip("temporary filesystem does not support O_DIRECT")

    memory_alignment, offset_alignment = alignment
    size = offset_alignment
    io = AlignedFileIO(str(tmp_path), use_direct=True)
    assert io.use_direct
    opened_flags: list[tuple[str, int]] = []
    original_open = os.open

    def track_open(path, flags, *args, **kwargs):
        opened_flags.append((os.fspath(path), flags))
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", track_open)
    path = tmp_path / "direct-aligned.bin"
    mappings = []
    views = []
    offsets = []
    try:
        for _ in range(2):
            mapping = mmap.mmap(-1, size + 2 * memory_alignment)
            mappings.append(mapping)
            base_address = ctypes.addressof(ctypes.c_ubyte.from_buffer(mapping))
            offset = (-base_address) % memory_alignment
            offsets.append(offset)
            views.append(memoryview(mapping)[offset : offset + size])

        source, destination = views
        source[:] = bytes(index % 251 for index in range(size))
        io.write(str(path), source)
        io.read(str(path), destination)

        assert bytes(destination) == bytes(source)
        assert any(
            open_path == str(path) and flags & os.O_DIRECT for open_path, flags in opened_flags
        )

        if memory_alignment > 1:
            unaligned_source = memoryview(mappings[0])[offsets[0] + 1 : offsets[0] + 1 + size]
            unaligned_destination = memoryview(mappings[1])[offsets[1] + 1 : offsets[1] + 1 + size]
            views.extend((unaligned_source, unaligned_destination))
            assert io._direct_address(unaligned_source) is None
            unaligned_source[:] = bytes((index + 17) % 251 for index in range(size))
            bounced_path = tmp_path / "direct-unaligned-buffer.bin"
            io.write(str(bounced_path), unaligned_source)
            io.read(str(bounced_path), unaligned_destination)
            assert bytes(unaligned_destination) == bytes(unaligned_source)
            assert any(
                open_path == str(bounced_path) and flags & os.O_DIRECT
                for open_path, flags in opened_flags
            )
    finally:
        for view in views:
            view.release()
        for mapping in mappings:
            mapping.close()


def test_host_region_registrar_registers_once_and_unregisters_on_close() -> None:
    with ExitStack() as leases:
        registered = []
        unregistered = []
        region = _leased_region(leases, 64)
        registrar = HostRegionRegistrar(registered.append, unregistered.append)

        registrar.register([region, region])
        registrar.register([region])
        registrar.close()

        assert registered == [region.region]
        assert unregistered == [region.region]
        with pytest.raises(RuntimeError, match="closed"):
            registrar.register([region])


def test_host_region_registrar_releases_bound_owner_after_close() -> None:
    class Owner:
        def __init__(self) -> None:
            self.registrar = HostRegionRegistrar(self.register, self.unregister)
            self.registered: list[HostRegion] = []

        def register(self, region: HostRegion) -> None:
            self.registered.append(region)

        def unregister(self, region: HostRegion) -> None:
            self.registered.remove(region)

    with ExitStack() as leases:
        owner = Owner()
        owner_ref = weakref.ref(owner)
        registrar = owner.registrar
        registrar.register([_leased_region(leases, 64)])

        registrar.close()
        del owner

        assert owner_ref() is None


def test_host_region_registrar_keeps_slot_owner_until_unregistration() -> None:
    class SlotPool:
        pass

    buffer = bytearray(64)
    slot_owner = SlotPool()
    owner_ref = weakref.ref(slot_owner)
    region = HostRegionLease(memoryview(buffer), keepalive=slot_owner)
    registrar = HostRegionRegistrar(lambda _region: None, lambda _region: None)
    registrar.register([region])
    region.release()
    del region, slot_owner

    assert owner_ref() is not None

    registrar.close()

    assert owner_ref() is None


def test_host_slot_region_requires_an_allocation_owner() -> None:
    view = memoryview(bytearray(64))
    try:
        with pytest.raises(ValueError, match="require an allocation owner"):
            HostRegionLease(view, keepalive=None)
    finally:
        view.release()


def test_host_region_registrar_rejects_address_reuse_with_different_size() -> None:
    buffer = bytearray(64)
    owner = object()
    first = HostRegionLease(memoryview(buffer), keepalive=owner)
    second = HostRegionLease(memoryview(buffer)[:32], keepalive=owner)
    registrar = HostRegionRegistrar(lambda _region: None, lambda _region: None)

    try:
        registrar.register([first])
        with pytest.raises(RuntimeError, match="reused with a different size"):
            registrar.register([second])
        registrar.close()
    finally:
        first.release()
        second.release()


def test_host_region_registrar_keeps_successful_prefix_after_later_failure() -> None:
    with ExitStack() as leases:
        first = _leased_region(leases, 64)
        second = _leased_region(leases, 32)
        registered = []
        unregistered = []
        attempts = 0

        def register(region) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 2:
                raise RuntimeError("second registration failed")
            registered.append(region)

        registrar = HostRegionRegistrar(register, unregistered.append)
        with pytest.raises(RuntimeError, match="second registration failed"):
            registrar.register([first, second])

        registrar.register([first, second])
        registrar.close()

        assert registered == [first.region, second.region]
        assert set(unregistered) == {first.region, second.region}
        assert len(unregistered) == 2
        assert attempts == 3


def test_host_region_registrar_retains_owner_when_close_fails(monkeypatch) -> None:
    class Owner:
        def __init__(self) -> None:
            self.registrar = HostRegionRegistrar(lambda _region: None, self.unregister)
            self.fail_once = True

        def unregister(self, _region: HostRegion) -> None:
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("temporary unregistration failure")

    retained: list[object] = []
    monkeypatch.setattr(
        host_region_module, "retain_until_exit", lambda owner, _regions=(): retained.append(owner)
    )

    with ExitStack() as leases:
        owner = Owner()
        owner_ref = weakref.ref(owner)
        registrar = owner.registrar
        region = _leased_region(leases, 64)
        registrar.register([region])

        with pytest.raises(RuntimeError, match="failed to unregister 1 of 1"):
            registrar.close()
        assert retained == [registrar]
        del owner
        assert owner_ref() is not None

        registrar.close()
        assert owner_ref() is None
        assert retained == [registrar]


def test_host_region_registrar_rejects_address_only_records() -> None:
    registrar = HostRegionRegistrar(
        lambda _region: pytest.fail("registered an unleased region"), lambda _region: None
    )

    with pytest.raises(TypeError, match="requires a live lease"):
        registrar.register([cast(HostRegionLease, HostRegion(0x1000, 64))])


def test_host_region_registrar_rejects_released_slot_leases() -> None:
    with ExitStack() as leases:
        region = _leased_region(leases, 64)
        region.release()
        registrar = HostRegionRegistrar(lambda _region: None, lambda _region: None)

        with pytest.raises(RuntimeError, match="released host region"):
            registrar.register([region])


def test_host_region_registrar_retries_failed_registration() -> None:
    with ExitStack() as leases:
        attempts = 0
        region = _leased_region(leases, 64)

        def register(_region: HostRegion) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("transient registration failure")

        registrar = HostRegionRegistrar(register, lambda _region: None)
        with pytest.raises(RuntimeError, match="transient"):
            registrar.register([region])

        registrar.register([region])
        registrar.close()
        assert attempts == 2


def test_host_region_registrar_serializes_concurrent_first_use() -> None:
    with ExitStack() as leases:
        calls = []
        barrier = Barrier(3)
        region = _leased_region(leases, 64)
        registrar = HostRegionRegistrar(calls.append, lambda _region: None)

        def register() -> None:
            barrier.wait()
            registrar.register([region])

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(register)
            second = executor.submit(register)
            barrier.wait()
            first.result()
            second.result()

        registrar.close()
        assert calls == [region.region]


def test_host_region_registrar_close_attempts_all_regions_and_can_retry(monkeypatch) -> None:
    retained: list[object] = []
    monkeypatch.setattr(
        host_region_module, "retain_until_exit", lambda owner, _regions=(): retained.append(owner)
    )

    with ExitStack() as leases:
        regions = [_leased_region(leases, 64), _leased_region(leases, 128)]
        failures = {regions[0].region.address}
        attempted = []

        def unregister(region: HostRegion) -> None:
            attempted.append(region.address)
            if region.address in failures:
                failures.remove(region.address)
                raise RuntimeError(f"cannot unregister {region.address:#x}")

        registrar = HostRegionRegistrar(lambda _region: None, unregister)
        registrar.register(regions)

        with pytest.raises(RuntimeError, match="failed to unregister 1 of 2"):
            registrar.close()
        registrar.close()

        assert attempted == [
            regions[1].region.address,
            regions[0].region.address,
            regions[0].region.address,
        ]
        assert retained == [registrar]


def test_host_region_registrar_reports_total_regions_when_all_unregistration_fails(
    monkeypatch,
) -> None:
    retained: list[object] = []
    monkeypatch.setattr(
        host_region_module, "retain_until_exit", lambda owner, _regions=(): retained.append(owner)
    )

    with ExitStack() as leases:

        def fail_unregister(region: HostRegion) -> None:
            raise RuntimeError(hex(region.address))

        registrar = HostRegionRegistrar(lambda _region: None, fail_unregister)
        registrar.register([_leased_region(leases, 64), _leased_region(leases, 64)])

        with pytest.raises(RuntimeError, match="failed to unregister 2 of 2"):
            registrar.close()
        assert retained == [registrar]


def test_host_slot_region_holds_and_releases_buffer_lease() -> None:
    slot = bytearray(b"page")

    with host_slot_region(slot) as region:
        import ctypes

        assert ctypes.string_at(region.region.address, region.region.size) == b"page"
        with pytest.raises(BufferError):
            slot.extend(b"x")

    slot.extend(b"x")
    assert slot == b"pagex"


def test_host_slot_region_releases_lease_after_exception() -> None:
    slot = bytearray(b"page")

    with pytest.raises(RuntimeError, match="operation failed"):
        with host_slot_region(slot):
            raise RuntimeError("operation failed")

    slot.extend(b"x")
    assert slot == b"pagex"


def test_host_slot_region_leases_engine_slot_access(monkeypatch) -> None:
    class _FakeHostSlot:
        def __init__(self) -> None:
            self.buffer = bytearray(b"page")
            self.access_calls = 0

        def access(self) -> bytearray:
            self.access_calls += 1
            return self.buffer

    monkeypatch.setattr(cache_transfer_module.engine, "HostSlot", _FakeHostSlot)
    slot = _FakeHostSlot()

    with host_slot_region(slot) as region:
        assert region.region.size == 4
        assert slot.access_calls == 1
        with pytest.raises(BufferError):
            slot.buffer.extend(b"x")

    slot.buffer.extend(b"x")


class _FakeCopier:
    def __init__(self, _layers, _host_block_bytes) -> None:
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class _FakeHostCache:
    def __init__(self, _copier, _slots, *, backing) -> None:
        self.backing = backing
        self.backing_requests = 0

    def _backing_buffer(self):
        self.backing_requests += 1
        return self.backing

    def slot(self, _index: int):
        return SimpleNamespace(access=object)


def _fake_engine_cache():
    return SimpleNamespace(
        blocks=2,
        layers=[bytearray(8)],
        spec=SimpleNamespace(
            pools=[SimpleNamespace(bytes=4)],
            groups=[SimpleNamespace(pool=0, layers=(0,))],
        ),
    )


def _patch_bounded_staging_engine(monkeypatch):
    allocations = []
    copiers = []

    def allocate(size: int):
        backing = bytearray(size)
        allocations.append(backing)
        return backing

    def create_copier(layers, host_block_bytes):
        copier = _FakeCopier(layers, host_block_bytes)
        copiers.append(copier)
        return copier

    monkeypatch.setattr(cache_transfer_module.engine, "HostBuffer", allocate)
    monkeypatch.setattr(cache_transfer_module.engine, "CopierLayer", lambda **kwargs: kwargs)
    monkeypatch.setattr(cache_transfer_module.engine, "Copier", create_copier)
    monkeypatch.setattr(cache_transfer_module.engine, "HostCache", _FakeHostCache)
    return allocations, copiers


def test_bounded_staging_uses_hostcache_backing_across_cache_rebind(monkeypatch) -> None:
    allocations, _copiers = _patch_bounded_staging_engine(monkeypatch)
    first_cache = _fake_engine_cache()
    staging = BoundedCacheStaging(first_cache, chips=1, capacity_pages=2)
    old_host_cache = staging._host_cache
    assert old_host_cache is not None
    backing = old_host_cache.backing
    region = staging.lease_host_region()
    address = region.region.address

    assert len(allocations) == 1
    assert not hasattr(staging, "_buffer")
    assert region.region.size == staging.capacity_pages * staging.host_block_bytes
    assert old_host_cache.backing_requests == 1

    try:
        staging.release_engine_cache(first_cache)
        assert staging._host_cache is None
        assert staging._detached_backing is backing
        assert region.region.address == address

        staging.restore_engine_cache(_fake_engine_cache())
        assert staging._host_cache is not None
        assert staging._host_cache.backing is backing
        assert staging._detached_backing is None
        assert old_host_cache.backing_requests == 2
        assert len(allocations) == 1
        assert region.region.address == address
    finally:
        region.release()


@pytest.mark.parametrize("during_network_transfer", [False, True])
def test_bounded_staging_rejects_cache_release_with_active_lease(
    monkeypatch, during_network_transfer: bool
) -> None:
    _allocations, copiers = _patch_bounded_staging_engine(monkeypatch)
    cache = _fake_engine_cache()
    staging = BoundedCacheStaging(cache, chips=1, capacity_pages=2)
    lease = staging.try_acquire(1)
    assert lease is not None
    if during_network_transfer:
        staging.begin_transport(lease)

    with pytest.raises(RuntimeError, match="staging leases.*remain active"):
        staging.release_engine_cache(cache)
    assert copiers[0].shutdown_calls == 0

    if during_network_transfer:
        staging.complete_transport(lease)
    staging.release(lease)
    staging.release_engine_cache(cache)
    assert staging._host_cache is None
    assert staging._detached_backing is not None
