"""An SGLang plugin that a test server loads to watch the host pages of the plugin's hierarchical
cache: each backup waits behind 1 GiB of scratch copies, so its pages stay in flight, and each
host-pool allocation logs how many of its pages held a copy and how many of those copies still
run, as a line of `allocations.jsonl` beside this file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypeAlias, final

import furiosa_opt_engine as engine

if TYPE_CHECKING:
    import torch

    from furiosa_opt_sglang.cache import Cache, Pages
    from furiosa_opt_sglang.decoder import Decoder
    from furiosa_opt_sglang.paging import Limits
    from furiosa_opt_sglang.runner import Runner

SCRATCH: Final = 1 << 30
# Each copy into host pages, beside the pages it writes.
Copies: TypeAlias = list[tuple[range, engine.Transfer]]


@final
class View:
    """A view of a layer's host buffer and the pages it covers."""

    def __init__(self, host: engine.HostBuffer, pages: range) -> None:
        self.host = host
        self.pages = pages


@final
class Buffer:
    """A layer's host buffer of pages `stride` bytes long, whose views name the pages they cover."""

    def __init__(self, buffer: engine.HostBuffer, stride: int) -> None:
        self.buffer = buffer
        self.stride = stride

    def slice(self, offset: int, size: int) -> View:
        pages = range(offset // self.stride, (offset + size) // self.stride)
        return View(self.buffer.slice(offset, size), pages)


@final
class Window:
    """A view of a layer's device buffer whose copies to host pages record their transfers."""

    def __init__(self, window: engine.HbmBuffer, copies: Copies) -> None:
        self.window = window
        self.copies = copies

    def to_host(self, view: View) -> engine.Transfer:
        transfer = self.window.to_host(view.host)
        self.copies.append((view.pages, transfer))
        return transfer


@final
class Layer:
    """A layer's device buffer whose views record their copies to host pages."""

    def __init__(self, layer: engine.HbmBuffer, copies: Copies) -> None:
        self.layer = layer
        self.copies = copies

    def slice(self, offset: int, size: int) -> Window:
        return Window(self.layer.slice(offset, size), self.copies)


def register() -> None:
    from furiosa_opt_sglang import offload, paging

    log = Path(__file__).with_name("allocations.jsonl")
    # Host page -> the transfers of the copies into it since it was last allocated.
    written: dict[int, list[engine.Transfer]] = {}
    # The device of each token pool lent, which its host tier's scratch copies run on.
    devices: list[engine.Device] = []
    scratch: list[tuple[engine.HbmBuffer, engine.HostBuffer]] = []
    host, tokens = offload.Host, paging.tokens
    init, alloc, backup = host.__init__, host.alloc, host.backup_from_device_all_layer

    def lend(runner: Runner, limits: Limits, decoder: Decoder, pages: Pages) -> Cache:
        devices.append(decoder.device)
        return tokens(runner, limits, decoder, pages)

    def initialize(
        self: offload.Host, cache: Cache, ratio: float, size: int, page_size: int
    ) -> None:
        init(self, cache, ratio, size, page_size)
        device = engine.HbmBuffer(devices[-1], SCRATCH)
        scratch.append((device, engine.HostBuffer(SCRATCH * cache.pages.chips)))

    def watched(
        self: offload.Host,
        device_pool: Cache,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        io_backend: str,
    ) -> None:
        # The device runs copies in submission order, so this backup's pages wait behind it.
        source, target = scratch[0]
        _ = source.to_host(target)
        copies: Copies = []
        layers, buffers = self.layers, self.buffers
        # The backup copies through these views as through the buffers they wrap.
        self.layers = [Layer(layer, copies) for layer in layers]  # pyright: ignore[reportAttributeAccessIssue]
        self.kv_buffer = [
            Buffer(buffer, size * self.chips)
            for buffer, size in zip(buffers, self.sizes, strict=True)
        ]
        try:
            backup(self, device_pool, host_indices, device_indices, io_backend)
        finally:
            self.layers, self.kv_buffer = layers, buffers
        for pages, transfer in copies:
            for page in pages:
                written.setdefault(page, []).append(transfer)

    def allocate(self: offload.Host, need_size: int) -> torch.Tensor | None:
        indices = alloc(self, need_size)
        if indices is not None:
            earlier = [written.pop(page, []) for page in offload.pages(indices, self.page_size)]
            allocation = {
                "reused": sum(1 for transfers in earlier if transfers),
                "running": sum(1 for transfers in earlier if not all(t.done() for t in transfers)),
            }
            with log.open("a") as lines:
                print(json.dumps(allocation), file=lines)
        return indices

    paging.tokens = lend
    host.__init__ = initialize
    host.alloc = allocate
    host.backup_from_device_all_layer = watched
