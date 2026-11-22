"""SGLang's host-pool adapter over the engine cache.

Each pinned slot holds one whole page across all layers. Six native workers batch up to two slots,
with every layer copy in one runtime DMA batch. SGLang keeps page selection and lifecycle; the
adapter submits stores asynchronously and reads storage through the same pinned slots. A load waits
for its all-layer batch before returning, so SGLang's per-layer ready events follow completed DMA.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, final

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

ENGINE_CACHE_GROUP = 0

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
        if storage_backend not in (None, "file"):
            raise ValueError(
                "the Furiosa engine cache supports SGLang HiCache storage only "
                "with its file backend"
            )
        if params.mtp_draft_device_pools:
            raise ValueError(
                "Furiosa HiCache cannot offload SGLang MTP draft pools: its host pool covers only "
                "the engine-owned target-model KV layers"
            )
        # The scheduler's cache parameters carry its tensor-parallel group.
        assert params.tp_cache_group is not None
        options = Hierarchical.parse(get_memory())
        host = Host(kvcache, options.hicache_ratio, options.hicache_size, params.page_size)
        if storage_backend == "file":
            storage_backend = "dynamic"
            storage_backend_extra_config = {
                **(storage_backend_extra_config or {}),
                "backend_name": "furiosa_file",
                "module_path": "furiosa_opt_sglang.storage",
                "class_name": "FuriosaHiCacheFile",
            }
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
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
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
        item_bytes = torch.empty((), dtype=cache.dtype).element_size()
        page_bytes = sum(self.sizes) * self.chips
        if page_size <= 0 or page_bytes % (page_size * item_bytes):
            raise ValueError("the host cache page must align to its token count and storage dtype")
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
    def init_kv_buffer(self) -> list[engine.HostCache]:
        offsets: list[int] = []
        offset = 0
        for size in self.sizes:
            offsets.append(offset)
            offset += size * self.chips
        self.copier = engine.Copier(
            self.layers,
            self.sizes,
            offsets,
            [0] * len(self.layers),
            offset,
        )
        self.host_cache = engine.HostCache(self.copier, self.page_num)
        self._pending_loads: dict[tuple[tuple[int, ...], tuple[int, ...]], set[int]] = {}
        return [self.host_cache]

    @override
    def get_hybrid_pool_buffer(self) -> list[engine.HostCache]:
        return [self.host_cache]

    @override
    def destroy(self) -> None:
        copier = getattr(self, "copier", None)
        if copier is not None:
            copier.shutdown()
        self.host_cache = None
        self.copier = None
        super().destroy()

    @override
    def backup_from_device_all_layer(
        self,
        device_pool: Cache,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        io_backend: str,
    ) -> None:
        device = pages(device_indices, self.page_size)
        host = pages(host_indices, self.page_size)
        if len(device) != len(host):
            raise ValueError("hierarchical cache backup source and destination pages differ")
        transfers = [
            (target, [(ENGINE_CACHE_GROUP, source)])
            for source, target in zip(device, host, strict=True)
        ]
        copy = self.copier.copy_out(transfers, self.host_cache)
        copy.submitted()

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
        if not 0 <= layer_id < self.layer_num:
            raise ValueError("hierarchical cache load names an unknown layer")
        device = tuple(pages(device_indices, self.page_size))
        host = tuple(pages(host_indices, self.page_size))
        if len(device) != len(host):
            raise ValueError("hierarchical cache load source and destination pages differ")
        if not device:
            return

        key = (tuple(device), tuple(host))
        seen = self._pending_loads.get(key)
        if seen is None:
            # SGLang calls once per layer; Copier moves every layer in one page batch.
            transfers = [
                (target, [(ENGINE_CACHE_GROUP, source)])
                for source, target in zip(device, host, strict=True)
            ]
            copy = self.copier.copy_in(transfers, self.host_cache)
            copy.submitted()
            # SGLang records layer-ready and allocator-finish events as soon as this callback
            # returns; neither event is ordered with the engine's PDMA queue. The engine batches
            # every layer into one DMA call, so wait for it before SGLang may consume or reuse pages.
            copy.wait()
            seen = set()
            self._pending_loads[key] = seen
        seen.add(layer_id)
        if len(seen) == self.layer_num:
            del self._pending_loads[key]

    @override
    def get_data_page(self, index: int, flat: bool = True) -> torch.Tensor:
        if not flat:
            raise ValueError("the engine cache exposes opaque flat pages to storage")
        page = self._page_index(index)
        slot = self.host_cache.slot(page).access()
        return torch.frombuffer(slot, dtype=torch.uint8).view(self.dtype)

    @override
    def get_dummy_flat_data_page(self) -> torch.Tensor:
        item_bytes = torch.empty((), dtype=self.dtype).element_size()
        return torch.empty(self._page_bytes // item_bytes, dtype=self.dtype, device="cpu")

    @override
    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        if data_page.device.type != "cpu":
            raise ValueError("the file storage backend returns host pages on CPU")
        page = self._page_index(index)
        source = memoryview(data_page.contiguous().view(torch.uint8).reshape(-1).numpy()).cast("B")
        if len(source) != self._page_bytes:
            raise ValueError("stored page size differs from the engine cache page")
        slot = self.host_cache.slot(page).access()
        destination = memoryview(slot).cast("B")
        try:
            destination[:] = source
        finally:
            destination.release()

    @property
    def _page_bytes(self) -> int:
        return sum(self.sizes) * self.chips

    def _page_index(self, index: int) -> int:
        if index < 0 or index % self.page_size:
            raise ValueError("storage pages must start at an aligned token slot")
        page = index // self.page_size
        if page >= self.page_num:
            raise ValueError("storage page is outside the engine host pool")
        return page


# The dtypes in which SGLang's pools name token slots.
SLOTS: Final = frozenset({torch.int32, torch.int64})
PAGES: Final = TypeAdapter(list[int], config=ConfigDict(strict=True))


def pages(indices: torch.Tensor, size: int) -> list[int]:
    """The pages of `size` token slots whose slots `indices` lists, each page whole and in order."""
    if size <= 0:
        raise ValueError("hierarchical cache page size must be positive")
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
