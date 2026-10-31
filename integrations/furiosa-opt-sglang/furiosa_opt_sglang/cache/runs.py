"""Copies of a token pool's pages between the engine cache and host memory, and which of the
pages move in one copy."""

from __future__ import annotations

from itertools import pairwise
from typing import Final

import torch
from pydantic import ConfigDict, TypeAdapter

# The dtypes in which SGLang's pools name token slots.
SLOTS: Final = frozenset({torch.int32, torch.int64})
PAGES: Final = TypeAdapter(list[int], config=ConfigDict(strict=True))


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
