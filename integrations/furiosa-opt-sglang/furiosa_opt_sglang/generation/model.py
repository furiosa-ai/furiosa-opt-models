"""Bind a loaded text model to the framework's generation and pooling routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.layers.pooler import Pooler

    from furiosa_opt_sglang.cache import Cache
    from furiosa_opt_sglang.decoder import Decoder


@dataclass(frozen=True, slots=True, eq=False)
class Model:
    """A decoder, the framework's pooler of its rows and the cache storage the runner lends it."""

    core: Decoder
    pooler: Pooler
    cache: Cache | None = None
