"""Own runner-lifetime engine storage and expose SGLang's paged cache to transfer adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast, final

import torch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator, paged
from sglang.srt.mem_cache.memory_pool import KVCache
from typing_extensions import override

import furiosa_opt_engine as engine
from furiosa_opt_engine.cache_transfer import cache_copier
from furiosa_opt_sglang.cache.pages import Pages
from furiosa_opt_sglang.cache.registration import MooncakeRegistration
from furiosa_opt_sglang.decoder import RELEASED

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention


@dataclass(slots=True)
class _SnapshotCopierBinding:
    """Own snapshot queue and HBM aliases for one engine-cache generation."""

    storage: engine.decoder.Cache
    copier: engine.Copier

    def close(self) -> None:
        self.copier.shutdown()


@final
class Cache(KVCache):
    """Keep engine storage alive for the runner, independently of request eviction; SGLang's memory
    saver drops the storage and allocates it again."""

    def __init__(
        self,
        spec: engine.decoder.CacheSpec,
        pages: Pages,
        size: int,
        transfer_backend: str | None = None,
    ) -> None:
        block, layers = pages.pool.block, pages.layers
        super().__init__(
            size,
            block,
            pages.dtype,
            len(layers),
            "cpu",
            enable_memory_saver=False,
            start_layer=layers.start,
            end_layer=layers.stop - 1,
        )
        self.pages = pages
        self.transfer_backend = transfer_backend
        self.mooncake_registration: MooncakeRegistration | None = None
        self._snapshot_binding: _SnapshotCopierBinding | None = None
        # The token pool's pages, rounded up, and its dummy page; None while released.
        self.storage: engine.decoder.Cache | None = engine.decoder.Cache(
            spec, (size + 2 * block - 1) // block
        )

    def _copy_copier_for(self, storage: engine.decoder.Cache) -> engine.Copier:
        """Keep synchronous snapshots on a queue independent of async transfer staging."""
        binding = self._snapshot_binding
        if binding is not None and binding.storage is storage:
            return binding.copier
        self.release_copy_copier()
        binding = _SnapshotCopierBinding(storage, cache_copier(storage, self.pages.chips))
        self._snapshot_binding = binding
        return binding.copier

    def release_copy_copier(self) -> None:
        """Drop the queue's HBM aliases before SGLang releases cache storage."""
        binding = self._snapshot_binding
        if binding is None:
            return
        self._snapshot_binding = None
        binding.close()

    @override
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError("NPU storage requires get_cpu_copy for a host snapshot")

    @override
    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError("NPU storage requires get_cpu_copy for a host snapshot")

    @override
    def get_kv_buffer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("NPU storage requires get_cpu_copy for a host snapshot")

    @override
    def set_kv_buffer(
        self, layer: RadixAttention, loc: torch.Tensor, cache_k: torch.Tensor, cache_v: torch.Tensor
    ) -> None:
        _ = copy(
            self, layer.layer_id - self.start_layer, loc, torch.stack((cache_k, cache_v), dim=1)
        )

    @property
    def head_num(self) -> int:
        """The key-value heads on each engine chip."""
        return self.pages.heads * self.pages.chips

    def get_kv_size_bytes(self) -> tuple[int, int]:
        if self.storage is None:
            raise RuntimeError(RELEASED)
        total = sum(len(layer) for layer in self.storage.layers)
        return total // 2, total // 2

    def get_contiguous_buf_infos(self) -> tuple[list[int], list[int], list[int]]:
        """Describe the selected transport's registered cache layout."""
        if self.transfer_backend == "nixl":
            return self._nixl_buffer_infos()
        if self.mooncake_registration is None:
            self.mooncake_registration = MooncakeRegistration(self)
        return self.mooncake_registration.infos()

    def _nixl_buffer_infos(self) -> tuple[list[int], list[int], list[int]]:
        if self.storage is None:
            raise RuntimeError(RELEASED)
        page_bytes = self.pages.pool.bytes * self.pages.chips
        if page_bytes % 2:
            raise ValueError(f"engine KV page size {page_bytes} cannot split into K and V")
        item_bytes = page_bytes // 2
        entries = 2 * len(self.storage.layers)
        return (
            [0] * entries,
            [item_bytes * self.storage.blocks] * entries,
            [item_bytes] * entries,
        )

    @override
    def get_cpu_copy(
        self,
        indices: torch.Tensor,
        mamba_indices: torch.Tensor | None = None,
        req_pool_index: torch.Tensor | None = None,
    ) -> list[list[list[torch.Tensor]]]:
        if mamba_indices is not None:
            raise ValueError("this cache contains attention state only")
        return [[list(copy(self, layer, indices).unbind(dim=1))] for layer in range(self.layer_num)]

    @override
    def load_cpu_copy(
        self,
        kv_cache_cpu: list[list[list[torch.Tensor]]],
        indices: torch.Tensor,
        mamba_indices: torch.Tensor | None = None,
        req_pool_index: torch.Tensor | None = None,
    ) -> None:
        if mamba_indices is not None:
            raise ValueError("this cache contains attention state only")
        for layer, chunks in enumerate(kv_cache_cpu):
            _ = copy(self, layer, indices, torch.cat([torch.stack(pair, dim=1) for pair in chunks]))

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor) -> None:
        # A verified chain's accepted rows already sit in the slots they commit to.
        if torch.equal(tgt_loc, src_loc):
            return
        snapshot = self.get_cpu_copy(src_loc)
        self.load_cpu_copy(snapshot, tgt_loc)


@final
class Allocator(PagedTokenToKVPoolAllocator):
    """Use SGLang's block allocator with its CPU reference index function."""

    @override
    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        num_new_pages: int | None = None,
    ) -> torch.Tensor | None:
        counts = (seq_lens_cpu + self.page_size - 1) // self.page_size - (
            prefix_lens_cpu + self.page_size - 1
        ) // self.page_size
        needed = int(counts.sum())
        if needed > len(self.free_pages):
            return None
        output = torch.empty(extend_num_tokens, dtype=torch.int64, device="cpu")
        # SGLang leaves its CPU reference index function unannotated.
        extend = cast("Callable[..., None]", paged.alloc_extend_naive)
        extend(prefix_lens, seq_lens, last_loc, self.free_pages, output, self.page_size, "cpu")
        self.free_pages = self.free_pages[needed:]
        return output

    @override
    def alloc_decode(
        self, seq_lens: torch.Tensor, seq_lens_cpu: torch.Tensor, last_loc: torch.Tensor
    ) -> torch.Tensor | None:
        return self.alloc_extend(
            seq_lens - 1, seq_lens_cpu - 1, seq_lens, seq_lens_cpu, last_loc, len(seq_lens)
        )


@final
class Attention(AttentionBackend):
    """Expose scheduler defaults; the engine model executes attention."""


def copy(
    cache: Cache, layer: int, indices: torch.Tensor, values: torch.Tensor | None = None
) -> torch.Tensor:
    """The `layer` rows at `indices` on the host, after writing `values` there when given."""
    pages, page, dtype = cache.pages, cache.page_size, cache.dtype
    shape = (2, pages.heads * pages.chips, pages.width)
    if indices.numel() == 0:
        # Match torch.frombuffer's CPU output even if PyTorch's default device changed.
        return torch.empty((0, *shape), dtype=dtype, device="cpu")
    if values is not None and (values.dtype != dtype or values.shape != (indices.numel(), *shape)):
        raise ValueError("snapshot shape or dtype differs from the cache")
    # PyTorch's overloads leave the inverse-index form of unique untyped.
    unique = cast("Callable[..., tuple[torch.Tensor, torch.Tensor]]", torch.unique)
    blocks, inverse = unique(indices // page, sorted=True, return_inverse=True)
    # PyTorch erases element types from integer tensor list conversion.
    integers = cast("Callable[[torch.Tensor], list[int]]", torch.Tensor.tolist)
    if cache.storage is None:
        raise RuntimeError(RELEASED)
    storage = cache.storage
    block_ids = integers(blocks)
    copier = cache._copy_copier_for(storage)
    host_page_bytes = (len(storage.layers[layer]) // storage.blocks) * pages.chips
    host = engine.HostBuffer(len(block_ids) * host_page_bytes)
    host_cache = engine.HostCache(copier, len(block_ids), backing=host, layers=[layer])
    routes = [(slot, [(layer, block)]) for slot, block in enumerate(block_ids)]
    copier.copy_out(routes, host_cache).wait()
    # Engine order is [chip, K/V, local head, width]. SGLang flattens heads chip-major
    # (2 chips × 4 heads -> chip 0's heads, then chip 1's).
    data = torch.frombuffer(host, dtype=dtype).reshape(
        len(block_ids), page, pages.chips, 2, pages.heads, pages.width
    )
    offsets = indices % page
    if values is not None:
        per_chip = (
            values.contiguous()
            .reshape(indices.numel(), 2, pages.chips, pages.heads, pages.width)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
        data[inverse, offsets] = per_chip
        copier.copy_in(routes, host_cache).wait()
    selected = data[inverse, offsets].permute(0, 2, 1, 3, 4).contiguous()
    return selected.reshape(-1, *shape)
