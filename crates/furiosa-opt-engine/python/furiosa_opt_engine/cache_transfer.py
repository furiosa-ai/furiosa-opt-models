"""Completion and error handling for KV cache transfers submitted by serving adapters."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, TypeVar


class PendingTransfer(Protocol):
    """The completion operations adapters need for a submitted host transfer."""

    def done(self) -> bool: ...

    def wait(self) -> None: ...


TransferT = TypeVar("TransferT", bound=PendingTransfer)


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
