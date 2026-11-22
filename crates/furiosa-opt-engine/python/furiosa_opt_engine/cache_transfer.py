"""DMA completion and aligned file I/O shared by serving-framework adapters."""

from __future__ import annotations

import contextlib
import ctypes
import mmap
import os
import threading
import uuid
from collections.abc import Iterable, Sequence
from itertools import pairwise
from typing import Protocol, TypeAlias, TypeVar, cast

from typing_extensions import Buffer

import furiosa_opt_engine as engine

ByteBuffer: TypeAlias = memoryview | engine.HostBuffer | engine.HostSlot
DEFAULT_READ_THREADS = 8
DEFAULT_WRITE_THREADS = 4


class PendingTransfer(Protocol):
    """The completion operations adapters need for a submitted host transfer."""

    def done(self) -> bool: ...

    def wait(self) -> None: ...


TransferT = TypeVar("TransferT", bound=PendingTransfer)


class AlignedFileIO:
    """Use page-aligned bounce buffers for O_DIRECT and pinned slots directly for buffered I/O."""

    def __init__(self, directory: str, *, use_direct: bool | None = None) -> None:
        self.alignment = mmap.PAGESIZE
        self.use_direct = _probe_o_direct(directory) if use_direct is None else use_direct
        self.local = threading.local()

    def write(self, path: str, source: ByteBuffer) -> None:
        source_view = _byte_view(source)
        try:
            self.prepare(path, len(source_view), 1)
            self._write_page(path, 0, source_view)
        finally:
            source_view.release()

    def read(self, path: str, destination: ByteBuffer) -> None:
        destination_view = _byte_view(destination)
        try:
            self._read_page(path, 0, destination_view)
        finally:
            destination_view.release()

    def prepare(self, path: str, page_bytes: int, pages: int) -> None:
        """Create a file with one page-aligned extent per cache page."""
        if pages <= 0:
            raise ValueError("file transfers must contain pages")
        size = self.aligned_size(page_bytes) * pages
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.ftruncate(descriptor, size)
        except Exception:
            os.close(descriptor)
            with contextlib.suppress(OSError):
                os.remove(path)
            raise
        os.close(descriptor)

    def write_page(self, path: str, index: int, source: ByteBuffer) -> None:
        """Write one page at its aligned file offset, bouncing only for direct I/O."""
        source_view = _byte_view(source)
        try:
            self._write_page(path, index, source_view)
        finally:
            source_view.release()

    def read_page(self, path: str, index: int, destination: ByteBuffer) -> None:
        """Read one page from its aligned file offset, bouncing only for direct I/O."""
        destination_view = _byte_view(destination)
        try:
            self._read_page(path, index, destination_view)
        finally:
            destination_view.release()

    def _write_page(self, path: str, index: int, source: memoryview) -> None:
        page_bytes = len(source)
        stride = self.aligned_size(page_bytes)
        bounce = self.use_direct
        target = self._buffer(stride) if bounce else source
        try:
            if bounce:
                source_address = ctypes.addressof(ctypes.c_ubyte.from_buffer(source))
                ctypes.memmove(self.local.address, source_address, page_bytes)
                target[page_bytes:] = bytes(stride - page_bytes)
            descriptor = os.open(path, os.O_WRONLY | self._direct_flag())
            try:
                written = os.pwrite(descriptor, target, index * stride)
                if written != len(target):
                    raise OSError(f"short write of {path}: {written} of {len(target)} bytes")
            finally:
                os.close(descriptor)
        finally:
            if bounce:
                target.release()

    def _read_page(self, path: str, index: int, destination: memoryview) -> None:
        page_bytes = len(destination)
        stride = self.aligned_size(page_bytes)
        bounce = self.use_direct
        source = self._buffer(stride) if bounce else destination
        try:
            descriptor = os.open(path, os.O_RDONLY | self._direct_flag())
            try:
                read = os.preadv(descriptor, [source], index * stride)
                if read != len(source):
                    raise OSError(f"short read of {path}: {read} of {len(source)} bytes")
            finally:
                os.close(descriptor)
            if bounce:
                destination_address = ctypes.addressof(ctypes.c_ubyte.from_buffer(destination))
                ctypes.memmove(destination_address, self.local.address, page_bytes)
        finally:
            if bounce:
                source.release()

    def _direct_flag(self) -> int:
        return getattr(os, "O_DIRECT", 0) if self.use_direct else 0

    def aligned_size(self, size: int) -> int:
        if size <= 0:
            raise ValueError("file transfers must contain bytes")
        return ((size + self.alignment - 1) // self.alignment) * self.alignment

    def _buffer(self, size: int) -> memoryview:
        if size <= 0:
            raise ValueError("file transfers must contain bytes")
        capacity = getattr(self.local, "capacity", 0)
        if capacity < size:
            aligned = ((size + self.alignment - 1) // self.alignment) * self.alignment
            previous = getattr(self.local, "buffer", None)
            if previous is not None:
                previous.close()
            self.local.buffer = None
            self.local.capacity = 0
            buffer = mmap.mmap(-1, aligned)
            address = ctypes.addressof(ctypes.c_ubyte.from_buffer(buffer))
            self.local.buffer = buffer
            self.local.address = address
            self.local.capacity = aligned
        return memoryview(self.local.buffer)[:size]


def _probe_o_direct(directory: str) -> bool:
    """Check direct I/O once; callers use buffered I/O when the filesystem rejects it."""
    direct = getattr(os, "O_DIRECT", 0)
    if not direct:
        return False
    path = os.path.join(directory, f".furiosa-o-direct-{uuid.uuid4().hex}")
    buffer = mmap.mmap(-1, mmap.PAGESIZE)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | direct, 0o600)
        try:
            return os.write(descriptor, buffer) == mmap.PAGESIZE
        finally:
            os.close(descriptor)
    except OSError:
        return False
    finally:
        buffer.close()
        with contextlib.suppress(OSError):
            os.remove(path)


def _byte_view(buffer: ByteBuffer) -> memoryview:
    """Export the engine buffer protocol, which Python 3.11 cannot express in a type stub."""
    if isinstance(buffer, engine.HostSlot):
        # The lease keeps DMA out until this view and any derived views are released.
        return memoryview(cast(Buffer, buffer.access())).cast("B")
    return memoryview(cast(Buffer, buffer)).cast("B")


def submit(transfers: Iterable[TransferT]) -> tuple[TransferT, ...]:
    """Submit copies and drain earlier work if a later copy cannot be submitted."""
    submitted: list[TransferT] = []
    try:
        for transfer in transfers:
            submitted.append(transfer)
    except Exception as submit_error:
        drain_error = settle(submitted)
        if drain_error is not None and drain_error is not submit_error:
            raise submit_error from drain_error
        raise
    return tuple(submitted)


def settle(transfers: Iterable[PendingTransfer]) -> BaseException | None:
    """Wait for every submitted copy and return the first failure after draining the rest."""
    failure = None
    for transfer in transfers:
        try:
            transfer.wait()
        except BaseException as error:
            if failure is None:
                failure = error
    return failure


def page_runs(
    device_pages: Sequence[int], host_pages: Sequence[int], chips: int
) -> tuple[tuple[int, int, int], ...]:
    """Group adjacent page pairs without changing device shard order.

    `chips` counts physical chips in the engine topology, not a framework setting.
    """
    if chips <= 0:
        raise ValueError("cache transfers require at least one chip")
    if len(device_pages) != len(host_pages):
        raise ValueError("device and host page lists must have the same length")
    starts: list[int] = []
    for index, (device_page, host_page) in enumerate(zip(device_pages, host_pages, strict=True)):
        if device_page < 0 or host_page < 0:
            raise ValueError("cache page indices cannot be negative")
        if (
            index == 0
            or chips > 1
            or (device_page - device_pages[index - 1], host_page - host_pages[index - 1]) != (1, 1)
        ):
            starts.append(index)
    return tuple(
        (device_pages[start], host_pages[start], end - start)
        for start, end in pairwise((*starts, len(device_pages)))
    )


def transfer_cache_pages(
    layer: engine.HbmBuffer,
    host: engine.HostBuffer,
    device_pages: Sequence[int],
    host_pages: Sequence[int],
    page_bytes: int,
    chips: int,
    *,
    to_host: bool,
) -> tuple[engine.Transfer, ...]:
    """Move one layer's cache pages through the supplied pinned host rows.

    This serves adapters, such as NIXL, that own their host buffers separately from the L3
    HostCache. L3 filesystem transfers use the batched Copier queue instead.
    """
    if page_bytes <= 0:
        raise ValueError("cache pages must contain bytes")
    host_page_bytes = page_bytes * chips

    def transfers() -> Iterable[engine.Transfer]:
        for device_page, host_page, count in page_runs(device_pages, host_pages, chips):
            device_range = layer.slice(device_page * page_bytes, count * page_bytes)
            host_range = host.slice(host_page * host_page_bytes, count * host_page_bytes)
            yield (device_range.to_host(host_range) if to_host else host_range.to_hbm(device_range))

    return submit(transfers())
