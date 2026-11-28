"""Aligned file I/O over host-resident KV cache bytes."""

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

from typing_extensions import Buffer

DEFAULT_READ_THREADS = 8
DEFAULT_WRITE_THREADS = 4
_AT_FDCWD = -100
_STATX_DIOALIGN = 0x00002000
_STATX_SIZE = 256
_STATX_DIO_ALIGNMENT_OFFSET = 152


class AlignedFileIO:
    """Keep framework-native page bytes; bounce aligned O_DIRECT transfers through scratch."""

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

    def write(self, path: str, source: Buffer) -> None:
        source_view = _byte_view(source)
        try:
            self.prepare(path, len(source_view), 1)
            self._write_at(path, 0, source_view)
        finally:
            source_view.release()

    def read(self, path: str, destination: Buffer) -> None:
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

    def writev_at(self, path: str, offset: int, sources: Iterable[Buffer]) -> None:
        """Write vectors contiguously at an exact framework-native file offset."""
        self._write_or_readv(path, offset, sources, writing=True)

    def readv_at(self, path: str, offset: int, destinations: Iterable[Buffer]) -> None:
        """Read vectors contiguously at an exact framework-native file offset."""
        self._write_or_readv(path, offset, destinations, writing=False)

    def writev_at_many(
        self,
        path: str,
        writes: Iterable[tuple[int, Iterable[Buffer]]],
    ) -> None:
        """Write several vector ranges while reusing direct and buffered file handles."""
        self._write_or_readv_many(path, writes, writing=True)

    def readv_at_many(
        self,
        path: str,
        reads: Iterable[tuple[int, Iterable[Buffer]]],
    ) -> None:
        """Read several vector ranges while reusing direct and buffered file handles."""
        self._write_or_readv_many(path, reads, writing=False)

    def _write_at(self, path: str, offset: int, source: memoryview) -> None:
        if offset < 0 or not source:
            raise ValueError("file writes require a nonempty buffer and nonnegative offset")
        direct = self._direct_compatible(offset, (source,))
        bounce = direct
        # HostCache exposes DMA-BUF mappings, which O_DIRECT cannot use as its I/O buffer.
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
        bounce = direct
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
                destination[:] = source
        finally:
            if bounce:
                source.release()

    def _write_or_readv(
        self,
        path: str,
        offset: int,
        buffers: Iterable[Buffer],
        *,
        writing: bool,
    ) -> None:
        self._write_or_readv_many(path, ((offset, buffers),), writing=writing)

    def _write_or_readv_many(
        self,
        path: str,
        transfers: Iterable[tuple[int, Iterable[Buffer]]],
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
                        bounce = direct
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
                                view[:] = target
                        finally:
                            if bounce:
                                target.release()
                        continue

                    file_descriptor = descriptor(direct)
                    expected = sum(map(len, views))
                    if direct:
                        # Collapse vectors into aligned scratch; DMA-BUF views stay off O_DIRECT.
                        target = self._buffer(expected)
                        try:
                            if writing:
                                cursor = 0
                                for view in views:
                                    end = cursor + len(view)
                                    target[cursor:end] = view
                                    cursor = end
                                actual = os.pwrite(file_descriptor, target, offset)
                            else:
                                actual = os.preadv(file_descriptor, [target], offset)
                            if actual != expected:
                                operation = "write" if writing else "read"
                                raise OSError(
                                    f"short {operation} of {path}: {actual} of {expected} bytes"
                                )
                            if not writing:
                                cursor = 0
                                for view in views:
                                    end = cursor + len(view)
                                    part = target[cursor:end]
                                    try:
                                        view[:] = part
                                    finally:
                                        part.release()
                                    cursor = end
                        finally:
                            target.release()
                        continue

                    current_offset = offset
                    limit = iov_limit()
                    for start in range(0, len(views), limit):
                        batch = views[start : start + limit]
                        batch_bytes = sum(map(len, batch))
                        actual = (
                            os.pwritev(file_descriptor, batch, current_offset)
                            if writing
                            else os.preadv(file_descriptor, batch, current_offset)
                        )
                        if actual != batch_bytes:
                            operation = "write" if writing else "read"
                            raise OSError(
                                f"short {operation} of {path}: {actual} of {batch_bytes} bytes"
                            )
                        current_offset += batch_bytes
                finally:
                    for view in views:
                        view.release()
            if not processed:
                raise ValueError("vectored file transfers require at least one range")

    def _direct_flag(self, direct: bool) -> int:
        return getattr(os, "O_DIRECT", 0) if direct else 0

    def _direct_compatible(self, offset: int, views: Iterable[memoryview]) -> bool:
        if not self.use_direct or offset < 0 or offset % self.offset_alignment != 0:
            return False
        total = 0
        found = False
        for view in views:
            if not view:
                return False
            total += len(view)
            found = True
        return found and total % self.offset_alignment == 0

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


def _byte_view(buffer: Buffer) -> memoryview:
    """Normalize views to byte offsets for file I/O."""
    return memoryview(buffer).cast("B")
