"""vLLM's cache layouts that the plugin reports, its names of the engine's cached layers, the spec
vLLM assigns a group's layers, the block table each engine group reads, and the blocks a copy to
or from host memory moves at once."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Final

from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm.v1.kv_cache_layout import KVCacheLayout

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheSpec

    import furiosa_opt_engine as engine


# The layouts in which each layer's blocks fill one contiguous range, as a layer's view reads them.
LAYOUTS: Final = tuple(
    layout.name for layout in KVCacheLayout if layout.is_layer_compact and layout.is_block_compact
)


def names(group: engine.decoder.Group) -> list[str]:
    """vLLM's names of the layers the engine's `group` caches."""
    return [f"layers.{index}" for index in group.layers]


def assigned(group: KVCacheGroupSpec, layer: str) -> KVCacheSpec:
    """The cache spec vLLM assigned to `layer` within `group`."""
    match group.kv_cache_spec:
        case UniformTypeKVCacheSpecs(kv_cache_specs=specs):
            return specs[layer]
        case shared:
            return shared


@dataclass(frozen=True, slots=True)
class Tables:
    """For each engine cache group in order, the index of the vLLM cache group whose block table a
    request names the engine group's blocks by. vLLM splits an engine group's layers among groups
    of its own that page alike, so one of their tables addresses them all; the others' blocks stay
    unwritten."""

    groups: tuple[int, ...]

    @classmethod
    def parse(cls, cache: engine.decoder.CacheSpec, groups: Sequence[KVCacheGroupSpec]) -> Tables:
        """The first of vLLM's `groups` within each group of `cache`. Refuses a vLLM group with
        layers of two engine groups, and groups that leave out an engine group's layer."""
        held = [set(names(group)) for group in cache.groups]
        # The engine group that holds each vLLM group's layers, or none when no one group does.
        owners = [
            next(
                (index for index, layers in enumerate(held) if layers >= set(group.layer_names)),
                None,
            )
            for group in groups
        ]
        covered = [
            {
                name
                for group, owner in zip(groups, owners, strict=True)
                if owner == index
                for name in group.layer_names
            }
            for index in range(len(held))
        ]
        if None in owners or covered != held:
            raise ValueError("vLLM's cache groups differ from the loaded model's")
        return cls(tuple(owners.index(index) for index in range(len(held))))

    def rows(self, blocks: Sequence[list[int]]) -> list[list[int]]:
        """A request's block IDs for each engine cache group, from its vLLM block tables."""
        return [blocks[group] for group in self.groups]


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
