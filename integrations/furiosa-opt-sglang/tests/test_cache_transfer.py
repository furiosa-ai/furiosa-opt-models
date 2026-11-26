from __future__ import annotations

import ctypes
import mmap
import os

import pytest

from furiosa_opt_engine.cache_transfer import AlignedFileIO, _probe_direct_io, _probe_o_direct


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
