"""The format of a token pool's pages over a decoder's paged cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Self, cast

import torch
from pydantic import ConfigDict, TypeAdapter

import furiosa_opt_engine as engine
from furiosa_opt_sglang.decoder import RELEASED

if TYPE_CHECKING:
    from furiosa_opt_sglang.decoder import Decoder


# The dtypes SGLang uses to name token slots.
SLOTS: Final = frozenset({torch.int32, torch.int64})
PAGES: Final = TypeAdapter(list[int], config=ConfigDict(strict=True))


def holding(slots: torch.Tensor, size: int) -> list[int]:
    """The pages of `size` token slots that hold `slots`, in order."""
    if size <= 0:
        raise ValueError("token page size must be positive")
    if slots.dim() != 1 or slots.dtype not in SLOTS:
        raise ValueError("cache page copies name token slots by a 1-D integer tensor")
    return sorted(set(PAGES.validate_python((slots // size).tolist())))


@dataclass(frozen=True, slots=True)
class Pages:
    """A decoder's paged keys and values as the token pool keeps them: blocks of `pool`'s format for
    each of `layers`, a part of each block on every one of `chips` chips, where a token's row holds
    `heads` keys, then as many values, of `width` elements of `dtype` each, which the engine does
    not report."""

    pool: engine.decoder.Pool
    layers: range
    chips: int
    dtype: torch.dtype
    heads: int
    width: int

    def __post_init__(self) -> None:
        """Reject layouts whose page size disagrees with the engine pool."""
        if self.chips <= 0:
            raise ValueError("cache pages require at least one chip")
        if self.heads <= 0 or self.width <= 0:
            raise ValueError("cache pages require positive head count and width")
        size = self.pool.block * 2 * self.heads * self.width * self.dtype.itemsize
        if size != self.pool.bytes:
            raise ValueError(f"the engine pages {self.pool.bytes} bytes a block, SGLang {size}")

    @classmethod
    def parse(cls, decoder: Decoder, heads: int, width: int) -> Self:
        """Refuse a row format whose blocks the engine sizes otherwise."""
        if decoder.model is None:
            raise RuntimeError(RELEASED)
        # PyTorch exposes its named dtypes through an untyped module lookup.
        dtype = cast("torch.dtype", getattr(torch, decoder.model.kv_cache))
        return cls(decoder.pool, decoder.layers, decoder.model.parallel, dtype, heads, width)
