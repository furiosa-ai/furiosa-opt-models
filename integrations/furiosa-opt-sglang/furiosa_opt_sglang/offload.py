"""SGLang's hierarchical cache over the engine cache: a host-pool stack SGLang assembles for the
engine's token pool, whose pages move to and from the engine cache's layers.

- Each layer's host pages lie in one host buffer, page after page; a host page holds the layer's
  cache page as the device holds it, every chip's part chip after chip.
- SGLang moves whole pages: `pages` reads its tensor of token slots as pages of `page_size`
  consecutive slots from a multiple of `page_size`, on the device as on the host. Pages that follow
  each other on both move in one copy.
- A backup or load returns once submitted. The device orders each copy after the calls and copies
  that wrote its bytes and before the ones that read or overwrite them, which is all SGLang's
  completion events stand for here.
"""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING, Final, cast, final

import torch
from pydantic import ConfigDict, TypeAdapter
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import HybridCacheController
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    StackBuildResult,
    StackStrategy,
    build_pool_entry,
)
from sglang.srt.mem_cache.pool_host import HostKVCache, HostPoolGroup
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.runtime_context import get_memory
from typing_extensions import override

import furiosa_opt_engine as engine
from furiosa_opt_sglang.cache import Cache
from furiosa_opt_sglang.platform import Hierarchical

if TYPE_CHECKING:
    from threading import Event

    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.server_args import ServerArgs


@final
class Strategy(StackStrategy):
    """The host-pool stack of the engine's token pool: one host pool of its pages, anchoring the
    radix tree's one component."""

    @override
    def matches(self, kvcache: object, components: set[ComponentType]) -> bool:
        return isinstance(kvcache, Cache) and components == {ComponentType.FULL}

    @override
    def build(
        self,
        *,
        cache: UnifiedRadixCache,
        kvcache: object,
        params: CacheInitParams,
        server_args: ServerArgs,
        load_cache_event: Event,
        storage_backend: str | None = None,
        storage_backend_extra_config: dict[str, object] | None = None,
        prefetch_threshold: int = 256,
        model_name: str | None = None,
        enable_storage_metrics: bool = False,
    ) -> StackBuildResult:
        assert isinstance(kvcache, Cache)
        # The platform refuses storage backends, and the plugin's speculation packs no draft pools.
        assert storage_backend is None and not params.mtp_draft_device_pools
        # The scheduler's cache parameters carry its tensor-parallel group.
        assert params.tp_cache_group is not None
        options = Hierarchical.parse(get_memory())
        host = Host(kvcache, options.hicache_ratio, options.hicache_size, params.page_size)
        layers = kvcache.layer_num
        group = HostPoolGroup(
            [
                build_pool_entry(
                    name=PoolName.KV,
                    host_pool=host,
                    device_pool=kvcache,
                    layer_mapping={layer: layer for layer in range(layers)},
                    transfer_layer_id_max=layers,
                    is_anchor=True,
                )
            ]
        )
        controller = HybridCacheController(
            params.token_to_kv_pool_allocator,
            group,
            params.page_size,
            params.tp_cache_group,
            load_cache_event=load_cache_event,
            attn_cp_group=params.attn_cp_cache_group,
            attn_tp_group=params.attn_tp_cache_group,
            pp_group=params.pp_cache_group,
            write_policy=options.hicache_write_policy,
            io_backend=options.hicache_io_backend,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            transfer_layer_id_max=layers,
            enable_storage_metrics=enable_storage_metrics,
            host_memory_mode=options.hicache_host_memory_mode,
        )
        return StackBuildResult(
            host_pool_group=group,
            cache_controller=controller,
            component_host_pools={ComponentType.FULL: host},
            pools_desc="KV",
        )


@final
class Host(HostKVCache):
    """A host tier of `cache`'s pages, `ratio` times its tokens or `size` GB when that is set."""

    def __init__(self, cache: Cache, ratio: float, size: int, page_size: int) -> None:
        # The platform refuses the memory saver beside this tier, so the storage is never released.
        assert cache.storage is not None, "a token pool holds its storage until it is released"
        self.layers = cache.storage.layers
        # Bytes one page of each layer takes on each chip.
        self.sizes = [len(layer) // cache.storage.blocks for layer in self.layers]
        self.chips = cache.pages.chips
        # SGLang's pool reads these while it allocates the pages.
        self.device_pool = cache
        self.layer_num = self.target_layer_num = cache.layer_num
        super().__init__(
            dtype=cache.dtype,
            device_capacity_tokens=cache.size,
            size_per_token=sum(self.sizes) * self.chips // page_size,
            start_layer=cache.start_layer,
            end_layer=cache.end_layer,
            host_to_device_ratio=ratio,
            host_size=size,
            page_size=page_size,
            layout="page_first",
            pin_memory=False,
            device="cpu",
        )

    @override
    def get_size_per_token(self) -> int:
        return self.size_per_token

    @override
    def init_kv_buffer(self) -> list[engine.HostBuffer]:
        return [engine.HostBuffer(self.page_num * size * self.chips) for size in self.sizes]

    @override
    def backup_from_device_all_layer(
        self,
        device_pool: Cache,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        io_backend: str,
    ) -> None:
        device, host = pages(device_indices, self.page_size), pages(host_indices, self.page_size)
        moved = runs(device, host, self.chips)
        for layer, size, buffer in zip(self.layers, self.sizes, self.buffers, strict=True):
            # Bytes of one host page.
            stride = size * self.chips
            for block, page, length in moved:
                _ = layer.slice(block * size, length * size).to_host(
                    buffer.slice(page * stride, length * stride)
                )

    @override
    def load_to_device_per_layer(
        self,
        device_pool: Cache,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        layer_id: int,
        io_backend: str,
        *,
        is_draft: bool = False,
    ) -> None:
        layer, size, buffer = self.layers[layer_id], self.sizes[layer_id], self.buffers[layer_id]
        device, host = pages(device_indices, self.page_size), pages(host_indices, self.page_size)
        stride = size * self.chips
        for block, page, length in runs(device, host, self.chips):
            _ = buffer.slice(page * stride, length * stride).to_hbm(
                layer.slice(block * size, length * size)
            )

    @override
    def get_data_page(self, index: int, flat: bool = True) -> torch.Tensor:
        raise NotImplementedError("the engine's host pool serves no storage backend")

    @override
    def get_dummy_flat_data_page(self) -> torch.Tensor:
        raise NotImplementedError("the engine's host pool serves no storage backend")

    @override
    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        raise NotImplementedError("the engine's host pool serves no storage backend")

    @property
    def buffers(self) -> list[engine.HostBuffer]:
        """Each layer's pages, as `init_kv_buffer` allocated them."""
        return cast("list[engine.HostBuffer]", self.kv_buffer)


# The dtypes in which SGLang's pools name token slots.
SLOTS: Final = frozenset({torch.int32, torch.int64})
PAGES: Final = TypeAdapter(list[int], config=ConfigDict(strict=True))


def pages(indices: torch.Tensor, size: int) -> list[int]:
    """The pages of `size` token slots whose slots `indices` lists, each page whole and in order."""
    if indices.dim() != 1 or indices.dtype not in SLOTS:
        raise ValueError("hierarchical cache transfers name token slots by a 1-D integer tensor")
    if len(indices) % size:
        raise ValueError("hierarchical cache transfers move whole pages")
    rows = indices.reshape(-1, size)
    first = rows[:, :1]
    offsets = torch.arange(size, dtype=rows.dtype).expand_as(rows)
    if (first % size).any() or not torch.equal(rows - first, offsets):
        raise ValueError("a hierarchical cache page is a page's consecutive token slots")
    return PAGES.validate_python((first.flatten() // size).tolist())


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
