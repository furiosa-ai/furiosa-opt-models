"""Physical placement checks and opened devices shared by framework adapters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, final

import furiosa_opt_engine as engine

MAX_CHIP_ID: Final = 255
PE_COUNT: Final = 8

__all__ = [
    "MAX_CHIP_ID",
    "PE_COUNT",
    "PlacedDevice",
    "draft_device",
    "has_distinct_chips",
    "is_nonempty_pe_range",
    "open_placed_device",
]


def has_distinct_chips(chips: Sequence[int]) -> bool:
    """Whether `chips` names one or more distinct physical chips."""
    return bool(chips) and len(set(chips)) == len(chips)


def is_nonempty_pe_range(pes: Sequence[int]) -> bool:
    """Whether `pes` is a nonempty half-open range."""
    return len(pes) == 2 and pes[0] < pes[1]


@final
@dataclass(frozen=True, slots=True)
class PlacedDevice:
    """An opened device paired with the placement used to open it."""

    # Construction goes through the selection functions so adapters cannot split the pair.
    def __new__(cls, *args: object, **kwargs: object) -> PlacedDevice:
        raise TypeError("use open_placed_device or draft_device to select a device")

    device: engine.Device
    placement: engine.Placement


def _make_placed_device(device: engine.Device, placement: engine.Placement) -> PlacedDevice:
    """Build an opened device pair for the placement selection API."""
    selected = object.__new__(PlacedDevice)
    object.__setattr__(selected, "device", device)
    object.__setattr__(selected, "placement", placement)
    return selected


def open_placed_device(placement: engine.Placement) -> PlacedDevice:
    """Open `placement` and keep its device paired with that placement."""
    return _make_placed_device(engine.Device(placement), placement)


def draft_device(
    target: PlacedDevice, requested: engine.Placement | None, parallelism: int
) -> PlacedDevice:
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
    return open_placed_device(requested)
