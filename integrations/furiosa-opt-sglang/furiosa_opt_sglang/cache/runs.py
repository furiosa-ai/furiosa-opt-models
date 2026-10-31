"""Copies of a token pool's pages between the engine cache and host memory: which pages a slot
tensor names, which of them move in one copy, and submitting copies so that a failed one raises
only once every copy submitted with it ended."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import suppress
from itertools import pairwise
from typing import Final, Protocol

import torch
from pydantic import ConfigDict, TypeAdapter

# The dtypes in which SGLang's pools name token slots.
SLOTS: Final = frozenset({torch.int32, torch.int64})
PAGES: Final = TypeAdapter(list[int], config=ConfigDict(strict=True))


def holding(slots: torch.Tensor, size: int) -> list[int]:
    """The pages of `size` token slots that hold `slots`, in order."""
    if slots.dim() != 1 or slots.dtype not in SLOTS:
        raise ValueError("host image copies name token slots by a 1-D integer tensor")
    return sorted(set(PAGES.validate_python((slots // size).tolist())))


class _Transfer(Protocol):
    """A submitted copy as its submitter waits for it: `wait` returns once it ended and raises a
    failed copy."""

    def wait(self) -> None: ...


def submit(transfers: Iterable[_Transfer]) -> tuple[_Transfer, ...]:
    """The copies `transfers` submits as it is read. A copy that fails to submit raises once the
    copies submitted before it ended, ahead of their own failures."""
    submitted: list[_Transfer] = []
    try:
        submitted.extend(transfers)
    except BaseException:
        with suppress(RuntimeError):
            wait(submitted)
        raise
    return tuple(submitted)


def wait(transfers: Iterable[_Transfer]) -> None:
    """Wait until every copy of `transfers` ended, then raise the first failure among them."""
    failure: RuntimeError | None = None
    for transfer in transfers:
        try:
            transfer.wait()
        except RuntimeError as error:
            failure = failure or error
    if failure is not None:
        raise failure


def runs(device: list[int], host: list[int], chips: int) -> list[tuple[int, int, int]]:
    """`device` pages beside the `host` pages they move to or from, as `(block, page, length)`
    runs that each move in one copy: pages that both step by one. A copy moves each chip's part of
    its pages in turn while a host page holds every chip's part of one page, so on more than one
    chip a run is one page."""
    assert len(device) == len(host)
    starts = [
        i
        for i in range(len(device))
        if i == 0 or chips > 1 or (device[i] - device[i - 1], host[i] - host[i - 1]) != (1, 1)
    ]
    return [
        (device[start], host[start], end - start) for start, end in pairwise([*starts, len(device)])
    ]
