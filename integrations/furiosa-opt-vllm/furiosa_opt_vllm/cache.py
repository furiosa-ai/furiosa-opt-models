"""vLLM's cache layouts that the plugin reports, the spec vLLM assigns a group's layers, and the
blocks a copy to or from host memory moves at once."""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING, Final

from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm.v1.kv_cache_layout import KVCacheLayout

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheSpec


# The layouts in which each layer's blocks fill one contiguous range, as a layer's view reads them.
LAYOUTS: Final = tuple(
    layout.name for layout in KVCacheLayout if layout.is_layer_compact and layout.is_block_compact
)


def assigned(group: KVCacheGroupSpec, layer: str) -> KVCacheSpec:
    """The cache spec vLLM assigned to `layer` within `group`."""
    match group.kv_cache_spec:
        case UniformTypeKVCacheSpecs(kv_cache_specs=specs):
            return specs[layer]
        case shared:
            return shared


def runs(device: list[int], host: list[int], chips: int) -> list[tuple[int, int, int]]:
    """`device` blocks beside the `host` rows they move to or from, as `(block, row, length)`
    runs that each move in one copy: blocks and rows that both step by one. A copy moves each
    chip's part of its blocks in turn while a row holds every chip's part of one block, so on more
    than one chip a run is one block."""
    starts = [
        i
        for i in range(len(device))
        if i == 0 or chips > 1 or (device[i] - device[i - 1], host[i] - host[i - 1]) != (1, 1)
    ]
    return [
        (device[start], host[start], end - start) for start, end in pairwise([*starts, len(device)])
    ]
