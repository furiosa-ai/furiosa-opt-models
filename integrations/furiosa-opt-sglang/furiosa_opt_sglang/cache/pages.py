"""The format of a token pool's pages over a decoder's paged cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, cast

import torch

import furiosa_opt_engine as engine
from furiosa_opt_sglang.decoder import RELEASED

if TYPE_CHECKING:
    from furiosa_opt_sglang.decoder import Decoder


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

    @classmethod
    def parse(cls, decoder: Decoder, heads: int, width: int) -> Self:
        """Refuse a row format whose blocks the engine sizes otherwise."""
        if decoder.model is None:
            raise RuntimeError(RELEASED)
        # PyTorch exposes its named dtypes through an untyped module lookup.
        dtype = cast("torch.dtype", getattr(torch, decoder.model.kv_cache))
        size = decoder.pool.block * 2 * heads * width * dtype.itemsize
        if size != decoder.pool.bytes:
            raise ValueError(f"the engine pages {decoder.pool.bytes} bytes a block, SGLang {size}")
        return cls(decoder.pool, decoder.layers, decoder.model.parallel, dtype, heads, width)
