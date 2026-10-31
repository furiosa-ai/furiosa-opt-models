"""Own runner-lifetime engine storage and expose SGLang's cache transport."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, cast, final

import torch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator, paged
from sglang.srt.mem_cache.memory_pool import KVCache
from typing_extensions import override

import furiosa_opt_engine as engine
from furiosa_opt_sglang.decoder import RELEASED

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention

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
        return cls(decoder.pool, decoder.layers, decoder.model.chips, dtype, heads, width)


@final
class Cache(KVCache):
    """Keep engine storage alive for the runner, independently of request eviction; SGLang's memory
    saver drops the storage and allocates it again."""

    def __init__(self, spec: engine.decoder.CacheSpec, pages: Pages, size: int) -> None:
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
        # The token pool's pages, rounded up, and its dummy page; None while released.
        self.storage: engine.decoder.Cache | None = engine.decoder.Cache(
            spec, (size + 2 * block - 1) // block
        )

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

    def get_kv_size_bytes(self) -> tuple[int, int]:
        if self.storage is None:
            raise RuntimeError(RELEASED)
        total = sum(len(layer) for layer in self.storage.layers)
        return total // 2, total // 2

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
    shape = (2, pages.heads, pages.width)
    if indices.numel() == 0:
        return torch.empty((0, *shape), dtype=dtype)
    # PyTorch's overloads leave the inverse-index form of unique untyped.
    unique = cast("Callable[..., tuple[torch.Tensor, torch.Tensor]]", torch.unique)
    blocks, inverse = unique(indices // page, sorted=True, return_inverse=True)
    # PyTorch erases element types from integer tensor list conversion.
    integers = cast("Callable[[torch.Tensor], list[int]]", torch.Tensor.tolist)
    if cache.storage is None:
        raise RuntimeError(RELEASED)
    size, storage = pages.pool.bytes, cache.storage.layers[layer]
    windows = [storage.slice(block * size, size) for block in integers(blocks)]

    def read(window: engine.HbmBuffer) -> torch.Tensor:
        host = engine.HostBuffer(len(window))
        window.to_host(host).wait()
        return torch.frombuffer(host, dtype=torch.uint8)

    data = torch.cat([read(window) for window in windows])
    data = data.reshape(-1, page, 2 * pages.heads * pages.width * dtype.itemsize)
    offsets = indices % page
    if values is not None:
        if values.dtype != dtype or values.shape != (indices.numel(), *shape):
            raise ValueError("snapshot shape or dtype differs from the cache")
        data[inverse, offsets] = values.contiguous().view(torch.uint8).reshape(indices.numel(), -1)
        for window, value in zip(windows, data, strict=True):
            _ = engine.HostBuffer(value.numpy().data).to_hbm(window)
    return data[inverse, offsets].contiguous().view(dtype).reshape(-1, *shape)
