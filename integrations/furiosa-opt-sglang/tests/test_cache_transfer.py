from __future__ import annotations

import os

import pytest

from furiosa_opt_engine.cache_transfer import AlignedFileIO, _probe_o_direct, page_runs


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
    assert path.stat().st_size == io.aligned_size(len(source))


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
    assert path.stat().st_size == io.aligned_size(len(source))


def test_aligned_file_io_reports_short_writes(tmp_path, monkeypatch) -> None:
    io = AlignedFileIO(str(tmp_path), use_direct=False)
    path = tmp_path / "short-write.bin"
    source = bytearray(5003)
    io.prepare(str(path), len(source), 1)
    monkeypatch.setattr(os, "pwrite", lambda fd, buffer, offset: len(buffer) - 1)

    with pytest.raises(OSError, match="short write"):
        io.write_page(str(path), 0, source)


def test_aligned_file_io_reports_short_reads(tmp_path, monkeypatch) -> None:
    io = AlignedFileIO(str(tmp_path), use_direct=False)
    path = tmp_path / "short-read.bin"
    destination = bytearray(5003)
    io.prepare(str(path), len(destination), 1)
    monkeypatch.setattr(os, "preadv", lambda fd, buffers, offset: len(buffers[0]) - 1)

    with pytest.raises(OSError, match="short read"):
        io.read_page(str(path), 0, destination)


@pytest.mark.parametrize(
    ("device_pages", "host_pages", "chips", "expected"),
    [
        ([4, 5, 6, 7], [0, 1, 2, 3], 1, ((4, 0, 4),)),
        ([4, 5, 6], [10, 11, 12], 1, ((4, 10, 3),)),
        ([4, 5, 7, 8], [0, 1, 2, 3], 1, ((4, 0, 2), (7, 2, 2))),
        ([4, 5, 6, 7], [0, 1, 3, 4], 1, ((4, 0, 2), (6, 3, 2))),
        ([4, 5, 6], [0, 1, 3], 1, ((4, 0, 2), (6, 3, 1))),
        ([6, 5, 4], [0, 1, 2], 1, ((6, 0, 1), (5, 1, 1), (4, 2, 1))),
        ([4, 5], [0, 1], 2, ((4, 0, 1), (5, 1, 1))),
        ([], [], 1, ()),
    ],
    ids=[
        "adjacent",
        "nonzero-host-base",
        "device-gap",
        "host-gap",
        "host-gap-short-tail",
        "descending",
        "multi-chip",
        "empty",
    ],
)
def test_page_runs_preserve_device_and_host_order(
    device_pages: list[int],
    host_pages: list[int],
    chips: int,
    expected: tuple[tuple[int, int, int], ...],
) -> None:
    assert page_runs(device_pages, host_pages, chips) == expected


@pytest.mark.parametrize(
    ("device_pages", "host_pages", "chips", "message"),
    [
        ([], [], 0, "at least one chip"),
        ([1], [], 1, "same length"),
        ([-1], [0], 1, "cannot be negative"),
        ([0], [-1], 1, "cannot be negative"),
    ],
)
def test_page_runs_reject_invalid_routes(
    device_pages: list[int], host_pages: list[int], chips: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        page_runs(device_pages, host_pages, chips)
