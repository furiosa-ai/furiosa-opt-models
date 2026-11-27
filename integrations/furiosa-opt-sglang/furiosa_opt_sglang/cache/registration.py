"""Register one bounded placeholder page per engine layer with Mooncake."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Protocol

import furiosa_opt_engine as engine
from furiosa_opt_engine.cache_transfer import (
    cache_layer_offsets,
    cache_layer_page_bytes,
    cache_staging_page_bytes,
    host_buffer_region,
)

if TYPE_CHECKING:
    from furiosa_opt_sglang.cache.pages import Pages


class Pool(Protocol):
    pages: Pages
    storage: engine.decoder.Cache | None


class MooncakeRegistrationInfo(NamedTuple):
    addresses: list[int]
    lengths: list[int]
    page_sizes: list[int]


class MooncakeRegistration:
    """Stable one-page registrations; Mooncake moves KV through its staging ring."""

    def __init__(self, pool: Pool) -> None:
        storage = pool.storage
        if storage is None:
            raise RuntimeError("Mooncake registration requires resident engine KV storage")
        self._buffer = engine.HostBuffer(cache_staging_page_bytes(storage, pool.pages.chips))
        self._layers = tuple(
            self._buffer.slice(offset, size)
            for offset, size in zip(
                cache_layer_offsets(storage, pool.pages.chips),
                cache_layer_page_bytes(storage, pool.pages.chips),
                strict=True,
            )
        )
        self._regions = tuple(host_buffer_region(buffer) for buffer in self._layers)

    def infos(self) -> MooncakeRegistrationInfo:
        """Return the minimum valid region metadata needed by SGLang's Mooncake control plane."""
        return MooncakeRegistrationInfo(
            [region.region.address for region in self._regions],
            [region.region.size for region in self._regions],
            [region.region.size for region in self._regions],
        )
