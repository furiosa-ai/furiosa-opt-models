"""vLLM's cache layouts that the plugin reports, the spec vLLM assigns a group's layers, and the
blocks a copy to or from host memory moves at once."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Final

from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm.v1.kv_cache_layout import KVCacheLayout

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheSpec

    import furiosa_opt_engine as engine


# Preserve each upstream layout name for vLLM's cache layout selection.
LAYOUTS: Final = tuple(layout.name for layout in KVCacheLayout)


def group_indices(
    cache: engine.decoder.CacheSpec, groups: list[KVCacheGroupSpec]
) -> dict[int, int]:
    """Map stable engine group IDs to vLLM's current group order by layer set."""
    engine_groups: dict[int, set[str]] = defaultdict(set)
    layer_owner: dict[int, int] = {}
    for group in cache.groups:
        for layer_id in group.layers:
            if layer_id in layer_owner:
                raise ValueError(f"engine cache layer {layer_id} belongs to multiple groups")
            layer_owner[layer_id] = group.vllm_group
            engine_groups[group.vllm_group].add(f"layers.{layer_id}")

    vllm_groups: dict[frozenset[str], int] = {}
    for index, group in enumerate(groups):
        names = frozenset(group.layer_names)
        if not names or len(names) != len(group.layer_names):
            raise ValueError(f"vLLM cache group {index} has no layers or duplicate layer names")
        if names in vllm_groups:
            raise ValueError("vLLM cache groups contain duplicate layer sets")
        vllm_groups[names] = index

    if len(engine_groups) != len(vllm_groups):
        raise ValueError("engine and vLLM cache groups differ")
    indices: dict[int, int] = {}
    for engine_group, names in engine_groups.items():
        index = vllm_groups.get(frozenset(names))
        if index is None:
            raise ValueError(f"engine cache group {engine_group} differs from vLLM's layer groups")
        indices[engine_group] = index
    return indices


def assigned(group: KVCacheGroupSpec, layer: str) -> KVCacheSpec:
    """The cache spec vLLM assigned to `layer` within `group`."""
    match group.kv_cache_spec:
        case UniformTypeKVCacheSpecs(kv_cache_specs=specs):
            return specs[layer]
        case shared:
            return shared
