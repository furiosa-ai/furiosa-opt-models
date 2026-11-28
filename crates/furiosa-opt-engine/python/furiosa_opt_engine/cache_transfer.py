"""DMA completion and aligned file I/O shared by serving-framework adapters."""

from __future__ import annotations

import contextlib
import ctypes
import mmap
import os
import struct
import sys
import threading
import uuid
from collections.abc import Iterable
from typing import Protocol, TypeAlias, TypeVar, cast

from typing_extensions import Buffer

import furiosa_opt_engine as engine

ByteBuffer: TypeAlias = Buffer | engine.HostSlot
DEFAULT_READ_THREADS = 8
DEFAULT_WRITE_THREADS = 4
_AT_FDCWD = -100
_STATX_DIOALIGN = 0x00002000
_STATX_SIZE = 256
_STATX_DIO_ALIGNMENT_OFFSET = 152


class PendingTransfer(Protocol):
    """The completion operations adapters need for a submitted host transfer."""

    def done(self) -> bool: ...

    def wait(self) -> None: ...


TransferT = TypeVar("TransferT", bound=PendingTransfer)


class AlignedFileIO:
    """Keep framework-native page bytes; use O_DIRECT only for aligned pages."""

    def __init__(self, directory: str, *, use_direct: bool | None = None) -> None:
        self.offset_alignment = mmap.PAGESIZE
        self.memory_alignment = mmap.PAGESIZE
        self.use_direct = False
        if use_direct is not False:
            alignment = _probe_direct_io(directory)
            if alignment is not None:
                self.memory_alignment, self.offset_alignment = alignment
                self.use_direct = True
        self.local = threading.local()

    def write(self, path: str, source: ByteBuffer) -> None:
        source_view = _byte_view(source)
        try:
            self.prepare(path, len(source_view), 1)
            self._write_at(path, 0, source_view)
        finally:
            source_view.release()

    def read(self, path: str, destination: ByteBuffer) -> None:
        destination_view = _byte_view(destination)
        try:
            self._read_at(path, 0, destination_view)
        finally:
            destination_view.release()

    def prepare(
        self, path: str, page_bytes: int, pages: int, *, file_bytes: int | None = None
    ) -> None:
        """Create compact pages, optionally padded to the framework's chunk size."""
        if pages <= 0:
            raise ValueError("file transfers must contain pages")
        if page_bytes <= 0:
            raise ValueError("file transfers must contain bytes")
        page_data_bytes = page_bytes * pages
        size = page_data_bytes if file_bytes is None else file_bytes
        if size < page_data_bytes:
            raise ValueError("file size cannot be smaller than its cache pages")
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            os.ftruncate(descriptor, size)
        except Exception:
            os.close(descriptor)
            with contextlib.suppress(OSError):
                os.remove(path)
            raise
        os.close(descriptor)

    def writev_at(self, path: str, offset: int, sources: Iterable[ByteBuffer]) -> None:
        """Write vectors contiguously at an exact framework-native file offset."""
        self._write_or_readv(path, offset, sources, writing=True)

    def readv_at(self, path: str, offset: int, destinations: Iterable[ByteBuffer]) -> None:
        """Read vectors contiguously at an exact framework-native file offset."""
        self._write_or_readv(path, offset, destinations, writing=False)

    def writev_at_many(
        self,
        path: str,
        writes: Iterable[tuple[int, Iterable[ByteBuffer]]],
    ) -> None:
        """Write several vector ranges while reusing direct and buffered file handles."""
        self._write_or_readv_many(path, writes, writing=True)

    def readv_at_many(
        self,
        path: str,
        reads: Iterable[tuple[int, Iterable[ByteBuffer]]],
    ) -> None:
        """Read several vector ranges while reusing direct and buffered file handles."""
        self._write_or_readv_many(path, reads, writing=False)

    def _write_at(self, path: str, offset: int, source: memoryview) -> None:
        if offset < 0 or not source:
            raise ValueError("file writes require a nonempty buffer and nonnegative offset")
        direct = self._direct_compatible(offset, (source,))
        source_address = self._direct_address(source) if direct else None
        bounce = direct and source_address is None
        target = self._buffer(len(source)) if bounce else source
        try:
            if bounce:
                target[:] = source
            descriptor = os.open(path, os.O_WRONLY | self._direct_flag(direct))
            try:
                written = os.pwrite(descriptor, target, offset)
                if written != len(target):
                    raise OSError(f"short write of {path}: {written} of {len(target)} bytes")
            finally:
                os.close(descriptor)
        finally:
            if bounce:
                target.release()

    def _read_at(self, path: str, offset: int, destination: memoryview) -> None:
        if offset < 0 or not destination:
            raise ValueError("file reads require a nonempty buffer and nonnegative offset")
        direct = self._direct_compatible(offset, (destination,))
        destination_address = self._direct_address(destination) if direct else None
        bounce = direct and destination_address is None
        source = self._buffer(len(destination)) if bounce else destination
        try:
            descriptor = os.open(path, os.O_RDONLY | self._direct_flag(direct))
            try:
                read = os.preadv(descriptor, [source], offset)
                if read != len(source):
                    raise OSError(f"short read of {path}: {read} of {len(source)} bytes")
            finally:
                os.close(descriptor)
            if bounce:
                if destination_address is None:
                    destination_address = ctypes.addressof(ctypes.c_ubyte.from_buffer(destination))
                ctypes.memmove(destination_address, self.local.address, len(destination))
        finally:
            if bounce:
                source.release()

    def _write_or_readv(
        self,
        path: str,
        offset: int,
        buffers: Iterable[ByteBuffer],
        *,
        writing: bool,
    ) -> None:
        self._write_or_readv_many(path, ((offset, buffers),), writing=writing)

    def _write_or_readv_many(
        self,
        path: str,
        transfers: Iterable[tuple[int, Iterable[ByteBuffer]]],
        *,
        writing: bool,
    ) -> None:
        descriptors: dict[bool, int] = {}
        vector_limit = None
        processed = False
        transfer_iter = iter(transfers)
        cleanup = contextlib.ExitStack()
        close_transfers = getattr(transfer_iter, "close", None)
        if close_transfers is not None:
            cleanup.callback(close_transfers)

        def descriptor(direct: bool) -> int:
            if direct not in descriptors:
                flags = os.O_WRONLY if writing else os.O_RDONLY
                file_descriptor = os.open(path, flags | self._direct_flag(direct))
                descriptors[direct] = file_descriptor
                cleanup.callback(os.close, file_descriptor)
            return descriptors[direct]

        def iov_limit() -> int:
            nonlocal vector_limit
            if vector_limit is None:
                try:
                    vector_limit = int(os.sysconf("SC_IOV_MAX"))
                except (AttributeError, OSError, ValueError):
                    vector_limit = 1024
                vector_limit = max(vector_limit, 1)
            return vector_limit

        with cleanup:
            for offset, buffers in transfer_iter:
                processed = True
                if offset < 0:
                    raise ValueError("file offsets cannot be negative")
                views = []
                try:
                    for buffer in buffers:
                        views.append(_byte_view(buffer))
                    if not views or any(not view for view in views):
                        raise ValueError("vectored file transfers require nonempty buffers")

                    direct = self._direct_compatible(offset, views)
                    if len(views) == 1:
                        view = views[0]
                        address = self._direct_address(view) if direct else None
                        bounce = direct and address is None
                        target = self._buffer(len(view)) if bounce else view
                        try:
                            if writing:
                                if bounce:
                                    target[:] = view
                                actual = os.pwrite(descriptor(direct), target, offset)
                            else:
                                actual = os.preadv(descriptor(direct), [target], offset)
                            if actual != len(target):
                                operation = "write" if writing else "read"
                                raise OSError(
                                    f"short {operation} of {path}: {actual} of {len(target)} bytes"
                                )
                            if bounce and not writing:
                                if address is None:
                                    address = ctypes.addressof(ctypes.c_ubyte.from_buffer(view))
                                ctypes.memmove(address, self.local.address, len(view))
                        finally:
                            if bounce:
                                target.release()
                        continue

                    direct = direct and all(
                        self._direct_address(view) is not None for view in views
                    )
                    file_descriptor = descriptor(direct)
                    current_offset = offset
                    limit = iov_limit()
                    for start in range(0, len(views), limit):
                        batch = views[start : start + limit]
                        expected = sum(map(len, batch))
                        actual = (
                            os.pwritev(file_descriptor, batch, current_offset)
                            if writing
                            else os.preadv(file_descriptor, batch, current_offset)
                        )
                        if actual != expected:
                            operation = "write" if writing else "read"
                            raise OSError(
                                f"short {operation} of {path}: {actual} of {expected} bytes"
                            )
                        current_offset += expected
                finally:
                    for view in views:
                        view.release()
            if not processed:
                raise ValueError("vectored file transfers require at least one range")

    def _direct_flag(self, direct: bool) -> int:
        return getattr(os, "O_DIRECT", 0) if direct else 0

    def _direct_compatible(self, offset: int, views: Iterable[memoryview]) -> bool:
        return (
            self.use_direct
            and offset >= 0
            and offset % self.offset_alignment == 0
            and all(len(view) > 0 and len(view) % self.offset_alignment == 0 for view in views)
        )

    def _direct_address(self, buffer: memoryview) -> int | None:
        """Return the pinned address only when it satisfies the probed memory alignment."""
        if buffer.readonly:
            return None
        address = ctypes.addressof(ctypes.c_ubyte.from_buffer(buffer))
        return address if address % self.memory_alignment == 0 else None

    def _buffer(self, size: int) -> memoryview:
        if size <= 0:
            raise ValueError("file transfers must contain bytes")
        capacity = getattr(self.local, "capacity", 0)
        if capacity < size:
            aligned = _align_up(size, mmap.PAGESIZE)
            previous = getattr(self.local, "buffer", None)
            if previous is not None:
                previous.close()
            self.local.buffer = None
            self.local.capacity = 0
            buffer = mmap.mmap(-1, aligned + self.memory_alignment - 1)
            base_address = ctypes.addressof(ctypes.c_ubyte.from_buffer(buffer))
            offset = (-base_address) % self.memory_alignment
            self.local.buffer = buffer
            self.local.address = base_address + offset
            self.local.offset = offset
            self.local.capacity = aligned
        offset = self.local.offset
        return memoryview(self.local.buffer)[offset : offset + size]


def _probe_o_direct(directory: str) -> bool:
    """Check whether aligned direct I/O works for this filesystem."""
    return _probe_direct_io(directory) is not None


def _probe_direct_io(directory: str) -> tuple[int, int] | None:
    """Return filesystem memory/offset alignments after one aligned direct-I/O probe."""
    direct = getattr(os, "O_DIRECT", 0)
    if not direct or not sys.platform.startswith("linux"):
        return None
    path = os.path.join(directory, f".furiosa-o-direct-{uuid.uuid4().hex}")
    descriptor = None
    direct_descriptor = None
    mapping = None
    view = None
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        alignment = _statx_dio_alignment(path)
        if alignment is None:
            return None
        memory_alignment, offset_alignment = alignment
        size = _align_up(mmap.PAGESIZE, offset_alignment)
        mapping = mmap.mmap(-1, size + memory_alignment - 1)
        base_address = ctypes.addressof(ctypes.c_ubyte.from_buffer(mapping))
        offset = (-base_address) % memory_alignment
        view = memoryview(mapping)[offset : offset + size]
        direct_descriptor = os.open(path, os.O_RDWR | direct)
        if os.pwrite(direct_descriptor, view, 0) != size:
            return None
        if os.preadv(direct_descriptor, [view], 0) != size:
            return None
        return alignment
    except (AttributeError, OSError, ValueError):
        return None
    finally:
        if view is not None:
            view.release()
        if mapping is not None:
            mapping.close()
        if direct_descriptor is not None:
            os.close(direct_descriptor)
        if descriptor is not None:
            os.close(descriptor)
        with contextlib.suppress(OSError):
            os.remove(path)


def _statx_dio_alignment(path: str) -> tuple[int, int] | None:
    """Read direct-I/O alignment from statx without adding work to page transfers."""
    try:
        statx = ctypes.CDLL(None, use_errno=True).statx
    except (AttributeError, OSError):
        return None
    statx.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_void_p,
    ]
    statx.restype = ctypes.c_int
    result = ctypes.create_string_buffer(_STATX_SIZE)
    if statx(_AT_FDCWD, os.fsencode(path), 0, _STATX_DIOALIGN, ctypes.byref(result)) != 0:
        return None
    values = result.raw
    mask = struct.unpack_from("=I", values)[0]
    if mask & _STATX_DIOALIGN == 0:
        return None
    memory_alignment, offset_alignment = struct.unpack_from(
        "=II", values, _STATX_DIO_ALIGNMENT_OFFSET
    )
    if memory_alignment <= 0 or offset_alignment <= 0:
        return None
    return memory_alignment, offset_alignment


def _align_up(size: int, alignment: int) -> int:
    return ((size + alignment - 1) // alignment) * alignment


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
    except BaseException as submit_error:
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
