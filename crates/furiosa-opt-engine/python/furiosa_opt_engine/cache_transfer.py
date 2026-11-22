"""Page routing and host copies shared by serving-framework adapters."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol, TypeAlias, TypeVar, cast

from typing_extensions import Buffer

import furiosa_opt_engine as engine

ByteBuffer: TypeAlias = memoryview | engine.HostBuffer


@dataclass(frozen=True, slots=True)
class CachePageRoute:
    """A repeated byte fragment from one cache page to its host representation."""

    # Offsets and strides are page-relative bytes; `fragments` copies each repeated fragment.
    cache_offset: int
    host_offset: int
    fragment_bytes: int
    fragments: int = 1
    cache_stride: int = 0
    host_stride: int = 0


class PendingTransfer(Protocol):
    """The completion operations adapters need for a submitted host transfer."""

    def done(self) -> bool: ...

    def wait(self) -> None: ...


TransferT = TypeVar("TransferT", bound=PendingTransfer)


def page_runs(
    device: Sequence[int], host: Sequence[int], chips: int
) -> tuple[tuple[int, int, int], ...]:
    """Return contiguous `(device_page, host_page, count)` runs safe for one DMA copy each."""
    if chips <= 0:
        raise ValueError("cache transfers require at least one chip")
    if len(device) != len(host):
        raise ValueError("device and host page lists must have the same length")
    starts: list[int] = []
    for index, (device_page, host_page) in enumerate(zip(device, host, strict=True)):
        if device_page < 0 or host_page < 0:
            raise ValueError("cache page indices cannot be negative")
        if (
            index == 0
            or chips > 1
            or (device_page - device[index - 1], host_page - host[index - 1]) != (1, 1)
        ):
            starts.append(index)
    return tuple(
        (device[start], host[start], end - start) for start, end in pairwise((*starts, len(device)))
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
    """Submit model-layout pages between one engine layer and chip-interleaved host rows."""
    if page_bytes <= 0:
        raise ValueError("cache pages must contain bytes")
    transfers: list[engine.Transfer] = []
    host_page_bytes = page_bytes * chips
    for device_page, host_page, count in page_runs(device_pages, host_pages, chips):
        device_range = layer.slice(device_page * page_bytes, count * page_bytes)
        host_range = host.slice(host_page * host_page_bytes, count * host_page_bytes)
        transfers.append(
            device_range.to_host(host_range) if to_host else host_range.to_hbm(device_range)
        )
    return submit(transfers)


def _byte_view(buffer: ByteBuffer) -> memoryview:
    """Export the engine buffer protocol, which Python 3.11 cannot express in a type stub."""
    return memoryview(cast(Buffer, buffer)).cast("B")


def copy_bytes(destination: ByteBuffer, source: ByteBuffer) -> None:
    """Copy equal-sized host byte ranges without changing their layout."""
    target = _byte_view(destination)
    value = _byte_view(source)
    if len(target) != len(value):
        raise ValueError("source and destination host ranges differ in size")
    target[:] = value


def copy_cache_pages(
    cache: ByteBuffer,
    host: ByteBuffer,
    routes: Sequence[CachePageRoute],
    *,
    page_count: int,
    cache_page_bytes: int,
    cache_page_stride: int,
    host_page_bytes: int,
    host_page_stride: int,
    to_host: bool,
) -> None:
    """Gather or scatter cache pages between engine and host layouts using certified byte routes."""
    if page_count <= 0:
        raise ValueError("cache page copies require at least one page")
    if not 0 < cache_page_bytes <= cache_page_stride:
        raise ValueError("cache page size must fit its stride")
    if not 0 < host_page_bytes <= host_page_stride:
        raise ValueError("host page size must fit its stride")

    cache_view = _byte_view(cache)
    host_view = _byte_view(host)
    if len(cache_view) < page_count * cache_page_stride:
        raise ValueError("cache buffer is smaller than its page range")
    if len(host_view) < page_count * host_page_stride:
        raise ValueError("host buffer is smaller than its page range")
    for route in routes:
        if route.cache_offset < 0 or route.host_offset < 0 or route.fragment_bytes <= 0:
            raise ValueError("cache page route has an invalid offset or fragment size")
        if route.fragments <= 0:
            raise ValueError("cache page route must contain at least one fragment")
        if route.cache_stride < 0 or route.host_stride < 0:
            raise ValueError("cache page route strides cannot be negative")
        cache_end = route.cache_offset + (route.fragments - 1) * route.cache_stride
        host_end = route.host_offset + (route.fragments - 1) * route.host_stride
        if cache_end + route.fragment_bytes > cache_page_bytes:
            raise ValueError("cache page route exceeds its engine page")
        if host_end + route.fragment_bytes > host_page_bytes:
            raise ValueError("cache page route exceeds its host page")

    if not to_host:
        empty = bytes(cache_page_stride)
        for page in range(page_count):
            start = page * cache_page_stride
            cache_view[start : start + cache_page_stride] = empty

    for route in routes:
        for page in range(page_count):
            cache_base = page * cache_page_stride
            host_base = page * host_page_stride
            for fragment in range(route.fragments):
                cache_start = cache_base + route.cache_offset + fragment * route.cache_stride
                host_start = host_base + route.host_offset + fragment * route.host_stride
                cache_range = slice(cache_start, cache_start + route.fragment_bytes)
                host_range = slice(host_start, host_start + route.fragment_bytes)
                if to_host:
                    host_view[host_range] = cache_view[cache_range]
                else:
                    cache_view[cache_range] = host_view[host_range]


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
