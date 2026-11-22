"""DMA completion and aligned file I/O shared by serving-framework adapters."""

from __future__ import annotations

import contextlib
import ctypes
import mmap
import os
import threading
import uuid
from collections.abc import Iterable
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
    """Use a page-aligned per-thread bounce buffer; O_DIRECT cannot use dma-buf slots directly."""

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
        """Write one page at its aligned file offset through the thread-local bounce buffer."""
        source_view = _byte_view(source)
        try:
            self._write_page(path, index, source_view)
        finally:
            source_view.release()

    def read_page(self, path: str, index: int, destination: ByteBuffer) -> None:
        """Read one page from its aligned file offset through the thread-local bounce buffer."""
        destination_view = _byte_view(destination)
        try:
            self._read_page(path, index, destination_view)
        finally:
            destination_view.release()

    def _write_page(self, path: str, index: int, source: memoryview) -> None:
        page_bytes = len(source)
        stride = self.aligned_size(page_bytes)
        target = self._buffer(stride)
        try:
            source_address = ctypes.addressof(ctypes.c_ubyte.from_buffer(source))
            ctypes.memmove(self.local.address, source_address, page_bytes)
            target[page_bytes:] = bytes(stride - page_bytes)
            descriptor = os.open(path, os.O_WRONLY | self._direct_flag())
            try:
                written = os.pwrite(descriptor, target, index * stride)
                if written != stride:
                    raise OSError(f"short write of {path}: {written} of {stride} bytes")
            finally:
                os.close(descriptor)
        finally:
            target.release()

    def _read_page(self, path: str, index: int, destination: memoryview) -> None:
        page_bytes = len(destination)
        stride = self.aligned_size(page_bytes)
        source = self._buffer(stride)
        try:
            descriptor = os.open(path, os.O_RDONLY | self._direct_flag())
            try:
                read = os.preadv(descriptor, [source], index * stride)
                if read != stride:
                    raise OSError(f"short read of {path}: {read} of {stride} bytes")
            finally:
                os.close(descriptor)
            destination_address = ctypes.addressof(ctypes.c_ubyte.from_buffer(destination))
            ctypes.memmove(destination_address, self.local.address, page_bytes)
        finally:
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


def settle(transfers: Iterable[PendingTransfer]) -> Exception | None:
    """Wait for every submitted copy and return the first failure after draining the rest."""
    failure = None
    for transfer in transfers:
        try:
            transfer.wait()
        except Exception as error:
            if failure is None:
                failure = error
    return failure
