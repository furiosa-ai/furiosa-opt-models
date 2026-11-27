"""Pinned host-page leases and registration lifetime for external pointer APIs."""

from __future__ import annotations

import contextlib
import ctypes
import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import cast, final

from typing_extensions import Buffer

import furiosa_opt_engine as engine


@dataclass(frozen=True, slots=True)
class HostRegion:
    """Address and length passed to a foreign memory-registration API."""

    address: int
    size: int


@dataclass(slots=True, init=False)
class HostRegionLease:
    """A host memory lease retaining both its exported view and allocation owner."""

    region: HostRegion
    _view: memoryview = field(repr=False)
    _keepalive: object | None = field(default=None, repr=False)
    _retained: bool = field(default=False, init=False, repr=False)
    _released: bool = field(default=False, init=False, repr=False)

    def __init__(self, view: memoryview, *, keepalive: object) -> None:
        if keepalive is None:
            raise ValueError("registered host regions require an allocation owner")
        if not view:
            raise ValueError("registered host regions must contain bytes")
        address = ctypes.addressof(ctypes.c_ubyte.from_buffer(view))
        self.region = HostRegion(address, view.nbytes)
        self._view = view
        self._keepalive = keepalive
        self._retained = False
        self._released = False

    def retain(self) -> None:
        """Keep the lease alive when an external operation's completion is unknown."""
        if self._released:
            raise RuntimeError("cannot retain a released host region")
        self._retained = True

    def release(self) -> None:
        if not self._retained and not self._released:
            self._view.release()
            self._keepalive = None
            self._released = True


@dataclass(frozen=True, slots=True)
class _RegisteredHostRegion:
    size: int
    keepalive: object


@final
class HostRegionRegistrar:
    """Register stable host allocations; unregister them before releasing owners."""

    def __init__(
        self,
        register: Callable[[HostRegion], None],
        unregister: Callable[[HostRegion], None],
    ) -> None:
        self._register: Callable[[HostRegion], None] | None = register
        self._unregister: Callable[[HostRegion], None] | None = unregister
        self._retained_until_exit = False
        self._registered: dict[int, _RegisteredHostRegion] = {}
        self._lock = threading.Lock()
        self._closed = False

    def register(self, regions: Iterable[HostRegionLease]) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("host region registrar is closed")
            register = self._register
            if register is None:
                raise RuntimeError("host region registrar callbacks were released")
            for region in regions:
                if not isinstance(region, HostRegionLease):
                    raise TypeError("host region registration requires a live lease")
                if region._released:
                    raise RuntimeError("cannot register a released host region")
                keepalive = region._keepalive
                if keepalive is None:
                    raise RuntimeError("cannot register a host region without its owner")
                address = region.region.address
                registered = self._registered.get(address)
                if registered is not None:
                    if registered.size != region.region.size:
                        raise RuntimeError(
                            "registered host address was reused with a different size"
                        )
                    continue
                register(region.region)
                self._registered[address] = _RegisteredHostRegion(region.region.size, keepalive)

    def close(self) -> None:
        """Unregister all regions, trying every address before reporting failures."""
        with self._lock:
            self._closed = True
            region_count = len(self._registered)
            failures: list[BaseException] = []
            failed_regions: list[tuple[int, _RegisteredHostRegion]] = []
            unregister = self._unregister
            if self._registered and unregister is None:
                self._retain_until_exit()
                raise RuntimeError("host region registrar lost unregister callback")
            while self._registered:
                address, registered = self._registered.popitem()
                try:
                    assert unregister is not None
                    unregister(HostRegion(address, registered.size))
                except BaseException as error:
                    failures.append(error)
                    failed_regions.append((address, registered))
            for address, registered in reversed(failed_regions):
                self._registered[address] = registered
            if failures:
                self._retain_until_exit()
                close_error = RuntimeError(
                    f"failed to unregister {len(failures)} of {region_count} host regions"
                )
                for failure in failures:
                    close_error.add_note(repr(failure))
                failures.clear()
                del failure
                # The raised traceback keeps this frame alive; drop the bound callback so a later
                # successful close can release its owner even while the first error is retained.
                unregister = None
                raise close_error
            self._register = None
            self._unregister = None

    def _retain_until_exit(self) -> None:
        if not self._retained_until_exit:
            retain_until_exit(self)
            self._retained_until_exit = True


@contextlib.contextmanager
def host_slot_region(slot: engine.HostSlot) -> Iterator[HostRegionLease]:
    """Lease a pinned engine slot for a foreign API that reads or writes its address."""
    view = _byte_view(slot)
    try:
        region = HostRegionLease(view, keepalive=slot)
    except BaseException:
        view.release()
        raise
    try:
        yield region
    finally:
        region.release()


def _byte_view(buffer: Buffer | engine.HostSlot) -> memoryview:
    """Export a buffer and hold the engine's DMA lease when the input is a host slot."""
    source = buffer.access() if isinstance(buffer, engine.HostSlot) else buffer
    return memoryview(cast(Buffer, source)).cast("B")


# Intentional roots for foreign registrations or I/O with uncertain lifetimes.
_UNSAFE_POINTER_OWNERS: list[tuple[object, tuple[HostRegionLease, ...]]] = []
_UNSAFE_POINTER_OWNERS_LOCK = threading.Lock()


def retain_until_exit(owner: object, regions: Iterable[HostRegionLease] = ()) -> None:
    """Keep an owner and optional host regions alive while foreign access is unsafe."""
    retained_regions = tuple(regions)
    for region in retained_regions:
        region.retain()
    with _UNSAFE_POINTER_OWNERS_LOCK:
        _UNSAFE_POINTER_OWNERS.append((owner, retained_regions))
