"""Physical placement checks shared by framework adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import furiosa_opt_engine as engine

MAX_CHIP_ID: Final = 255
PE_COUNT: Final = 8

__all__ = [
    "MAX_CHIP_ID",
    "PE_COUNT",
    "draft_device",
    "has_distinct_chips",
    "is_nonempty_pe_range",
]


def has_distinct_chips(chips: Sequence[int]) -> bool:
    """Whether `chips` names one or more distinct physical chips."""
    return bool(chips) and len(set(chips)) == len(chips)


def is_nonempty_pe_range(pes: Sequence[int]) -> bool:
    """Whether `pes` is a nonempty half-open range."""
    return len(pes) == 2 and pes[0] < pes[1]


def draft_device(
    target: engine.Device, requested: engine.Placement | None, parallelism: int
) -> engine.Device:
    """Reuse the target by default; explicit matches reuse it and disjoint chips open another.

    Explicit placements must match `parallelism`; partial chip overlap is refused.
    """
    if requested is None:
        return target
    if len(requested.chips) != parallelism:
        raise ValueError(
            f"draft placement must provide {parallelism} chips; got {len(requested.chips)}"
        )
    if requested == target.placement:
        return target
    if set(requested.chips) & set(target.placement.chips):
        raise ValueError(
            f"draft placement {requested!r} overlaps target placement {target.placement!r}"
        )
    return engine.Device(requested)
