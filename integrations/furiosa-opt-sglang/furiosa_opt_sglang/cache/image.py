"""A token pool's pinned host image, the memory a transfer backend registers as the pool under
prefill-decode disaggregation, since none reaches the device's.

- A prefill server stores the pages SGLang sends right before each send, whichever step or reload
  wrote them on the device.
- A decode server's transfer backend writes the pages a row allocation receives into the image
  (`Received`); the pages load once SGLang reports the transfer done, before any reader of the row.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Protocol, final

import torch

import furiosa_opt_engine as engine
from furiosa_opt_sglang.cache.runs import holding, runs, submit, wait
from furiosa_opt_sglang.decoder import RELEASED

if TYPE_CHECKING:
    from furiosa_opt_sglang.cache.pages import Pages


class Pool(Protocol):
    """The token pool an image mirrors: its pages' format, tokens a page, and its storage, none
    while the memory saver released it."""

    pages: Pages
    page_size: int
    storage: engine.decoder.Cache | None


@dataclass(slots=True)
class Received:
    """The pages a transfer backend writes into the host image for each request row, from the
    row's allocation until a load takes them."""

    _pages: dict[int, list[int]] = field(default_factory=dict)

    def allocate(self, rows: list[int]) -> None:
        """Open the allocation of `rows`, which holds no KV received for an earlier allocation."""
        for row in rows:
            _ = self._pages.pop(row, None)

    def receive(self, row: int, pages: list[int]) -> None:
        """Expect the KV of `pages` of `row` in the host image."""
        self._pages[row] = pages

    def take(self, rows: Iterable[int]) -> list[int]:
        """The received pages of `rows` the device does not hold yet, in order; the caller loads
        them."""
        return sorted(page for row in rows for page in self._pages.pop(row, ()))


@final
class Image:
    """The host image of `pool`'s pages, page after page in each layer, a page as the device holds
    it, and the KV request rows received in it."""

    def __init__(self, pool: Pool) -> None:
        self._pool = pool
        self.received = Received()

    @cached_property
    def buffers(self) -> list[engine.HostBuffer]:
        """Each layer's image, allocated when a transfer backend first asks for it."""
        storage = self._pool.storage
        if storage is None:
            raise RuntimeError(RELEASED)
        return [engine.HostBuffer(len(layer) * self._pool.pages.chips) for layer in storage.layers]

    def infos(self) -> tuple[list[int], list[int], list[int]]:
        """Each layer's image as a transfer backend registers it: its address, its length and the
        bytes of a page."""
        views = [torch.frombuffer(buffer, dtype=torch.uint8) for buffer in self.buffers]
        page = self._pool.pages.pool.bytes * self._pool.pages.chips
        return (
            [view.data_ptr() for view in views],
            [view.numel() for view in views],
            [page] * len(views),
        )

    def store(self, slots: torch.Tensor) -> None:
        """Copy the pages holding token `slots` of every layer to the image, which a transfer
        backend sends."""
        pages = holding(slots, self._pool.page_size)
        wait(submit(device.to_host(host) for device, host in self.copies(pages)))

    def receive(self, row: int, slots: torch.Tensor) -> None:
        """Expect the KV of token `slots` of `row` in the image."""
        self.received.receive(row, holding(slots, self._pool.page_size))

    def load(self, rows: Iterable[int]) -> None:
        """Copy the KV that `rows` received in the image to the device."""
        if pages := self.received.take(rows):
            wait(submit(host.to_hbm(device) for device, host in self.copies(pages)))

    def copies(self, pages: list[int]) -> list[tuple[engine.HbmBuffer, engine.HostBuffer]]:
        """The device and image ranges each copy of `pages` moves, a run of pages a layer."""
        storage = self._pool.storage
        if storage is None:
            raise RuntimeError(RELEASED)
        # An image page lies where the device's page does.
        chips, size = self._pool.pages.chips, self._pool.pages.pool.bytes
        return [
            (
                layer.slice(block * size, length * size),
                buffer.slice(block * size * chips, length * size * chips),
            )
            for layer, buffer in zip(storage.layers, self.buffers, strict=True)
            for block, _, length in runs(pages, pages, chips)
        ]
