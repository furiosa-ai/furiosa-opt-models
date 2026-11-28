"""Watch whether the test server reallocates host pages while their backups are still in flight."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Timer
from typing import TYPE_CHECKING, cast

from furiosa_opt_sglang import offload

if TYPE_CHECKING:
    import torch

    import furiosa_opt_engine as engine
    from furiosa_opt_sglang.cache import Cache


class WatchedCopier:
    """Hold a host view briefly so each backup stays pending after it is queued."""

    def __init__(self, copier: engine.Copier, written: dict[int, engine.Copy]) -> None:
        self.copier = copier
        self.written = written
        self.copy_in = copier.copy_in
        self.shutdown = copier.shutdown

    def copy_out(
        self,
        pages: list[tuple[int, list[tuple[int, int]]]],
        host: engine.HostCache,
        *,
        coalesce: bool = False,
    ) -> engine.Copy:
        if not pages:
            return self.copier.copy_out(pages, host, coalesce=coalesce)
        lease = host.access(pages[0][0])
        try:
            copy = self.copier.copy_out(pages, host, coalesce=coalesce)
            for slot, _ in pages:
                self.written[slot] = copy
            Timer(0.5, lease.release).start()
        except BaseException:
            lease.release()
            raise
        return copy


def register() -> None:
    log = Path(__file__).with_name("allocations.jsonl")
    written: dict[int, engine.Copy] = {}
    host = offload.Host
    init, alloc = host.__init__, host.alloc

    def initialize(
        self: offload.Host,
        cache: Cache,
        ratio: float,
        size: int,
        page_size: int,
        *,
        coalescing: bool = False,
    ) -> None:
        init(self, cache, ratio, size, page_size, coalescing=coalescing)
        self.copier = cast("engine.Copier", WatchedCopier(self.copier, written))

    def allocate(self: offload.Host, need_size: int) -> torch.Tensor | None:
        indices = alloc(self, need_size)
        if indices is not None:
            earlier = [written.pop(page, None) for page in offload.pages(indices, self.page_size)]
            allocation = {
                "reused": sum(copy is not None for copy in earlier),
                "running": sum(copy is not None and not copy.done() for copy in earlier),
            }
            with log.open("a") as lines:
                print(json.dumps(allocation), file=lines)
        return indices

    host.__init__ = initialize
    host.alloc = allocate
