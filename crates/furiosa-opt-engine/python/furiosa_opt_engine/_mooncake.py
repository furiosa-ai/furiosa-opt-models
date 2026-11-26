"""Completion checks shared by integrations that pass pinned pages to Mooncake."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import IntEnum
from threading import Lock
from typing import final

from ._host_region import HostRegionLease, retain_until_exit

ENGINE_PAGE_FORMAT_VERSION = "furiosa-engine-page-v1"


# Mooncake v0.3.12 ErrorCode values from types.h. Transport and lease errors fail
# closed because those results do not prove pointer I/O has stopped.
class _MooncakeErrorCode(IntEnum):
    NO_AVAILABLE_HANDLE = -200
    INVALID_PARAMS = -600
    INVALID_REPLICA = -702
    REPLICA_IS_NOT_READY = -703
    OBJECT_NOT_FOUND = -704
    OBJECT_ALREADY_EXISTS = -705
    TENANT_QUOTA_EXCEEDED = -1700


# Only deterministic pre-transfer rejections and object state outcomes release a
# page lease. RPC and lease failures are deliberately treated as unconfirmed.
_SAFE_IO_REJECTIONS = frozenset(
    {
        _MooncakeErrorCode.NO_AVAILABLE_HANDLE,
        _MooncakeErrorCode.REPLICA_IS_NOT_READY,
        _MooncakeErrorCode.OBJECT_NOT_FOUND,
        _MooncakeErrorCode.OBJECT_ALREADY_EXISTS,
    }
)
_SAFE_GET_REJECTIONS = _SAFE_IO_REJECTIONS | {
    _MooncakeErrorCode.INVALID_PARAMS,
    _MooncakeErrorCode.INVALID_REPLICA,
}
_SAFE_PUT_REJECTIONS = _SAFE_IO_REJECTIONS | {_MooncakeErrorCode.TENANT_QUOTA_EXCEEDED}


class UnconfirmedMooncakeIO(RuntimeError):
    """A Mooncake result does not prove that pointer I/O has ended."""


@final
class MooncakeIOGuard:
    """Poison a Store and retain its page leases when pointer I/O is uncertain."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._error: UnconfirmedMooncakeIO | None = None

    def ensure_safe(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise UnconfirmedMooncakeIO(
                "Mooncake I/O completion is unconfirmed; the worker must exit"
            ) from error

    def run_pointer_io(
        self,
        owner: object,
        regions: Sequence[HostRegionLease],
        operation: Callable[[], object],
        validate: Callable[[object], list[int]],
        operation_name: str,
    ) -> list[int]:
        # This lock check is the admission point; already admitted transfers run
        # concurrently, while calls admitted after poisoning are rejected. Holding
        # the lock across synchronous Store I/O would serialize all transfers.
        self.ensure_safe()
        try:
            return validate(operation())
        except BaseException as error:
            unsafe_error = UnconfirmedMooncakeIO(
                f"Mooncake {operation_name} completion could not be confirmed"
            )
            with self._lock:
                if self._error is None:
                    self._error = unsafe_error
            retain_until_exit(owner, regions)
            raise unsafe_error from error


def require_confirmed_get(results: object, capacities: Sequence[int]) -> list[int]:
    """Return GET byte counts, or reject results that cannot release host leases."""
    if not isinstance(results, list) or len(results) != len(capacities):
        raise UnconfirmedMooncakeIO("Mooncake returned an incomplete GET result")
    for result, capacity in zip(results, capacities, strict=True):
        if type(result) is not int or not (
            0 <= result <= capacity or result in _SAFE_GET_REJECTIONS
        ):
            raise UnconfirmedMooncakeIO(
                f"Mooncake GET completion is unconfirmed: result={result}, capacity={capacity}"
            )
    return results


def require_confirmed_put(results: object, count: int) -> list[int]:
    """Return PUT statuses, or reject results that cannot release host leases."""
    if not isinstance(results, list) or len(results) != count:
        raise UnconfirmedMooncakeIO("Mooncake returned an incomplete PUT result")
    for result in results:
        if type(result) is not int or not (result == 0 or result in _SAFE_PUT_REJECTIONS):
            raise UnconfirmedMooncakeIO(f"Mooncake PUT completion is unconfirmed: result={result}")
    return results


def require_confirmed_exists(results: object, count: int) -> list[bool]:
    """Return existence flags and reject malformed Store responses."""
    if not isinstance(results, list) or len(results) != count:
        raise RuntimeError("Mooncake returned an incomplete existence result")
    exists = []
    for result in results:
        if type(result) is not int or result not in (0, 1):
            raise RuntimeError(f"Mooncake returned an invalid existence result: {result}")
        exists.append(result == 1)
    return exists
