from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.kv_cache_utils import compute_layer_kv_cache_shape_bytes, compute_layout_strides
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, KVCacheTensor
from vllm.v1.kv_cache_layout import KVCacheLayout

from furiosa_opt_vllm.cache import LAYOUTS
from furiosa_opt_vllm.generation.paged import storage_group_block_ids
from furiosa_opt_vllm.paging import _validate_groups, _validate_kv_cache_tensors


def test_reuses_logical_block_ids_for_each_mapped_engine_group() -> None:
    logical = ([1, 4, 6], [2, 3])
    assert storage_group_block_ids(logical, (0, 0, 1, 0)) == [
        [1, 4, 6],
        [1, 4, 6],
        [2, 3],
        [1, 4, 6],
    ]


def test_rejects_an_engine_group_without_a_vllm_group() -> None:
    with pytest.raises(ValueError, match="no matching vLLM block table"):
        storage_group_block_ids(([1],), (1,))


def test_matches_logical_group_ids_to_vllm_group_order_by_layer_set() -> None:
    cache = SimpleNamespace(
        groups=[
            SimpleNamespace(vllm_group=0, layers=[4, 0]),
            SimpleNamespace(vllm_group=1, layers=[3]),
        ]
    )
    first, second = object(), object()
    config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=["layers.3"], kv_cache_spec=second, host_resident=False),
            SimpleNamespace(
                layer_names=["layers.0", "layers.4"], kv_cache_spec=first, host_resident=False
            ),
        ]
    )
    expected = {"layers.0": first, "layers.4": first, "layers.3": second}
    assert _validate_groups(cache, config, expected) == {0: 1, 1: 0}


def test_rejects_a_vllm_cache_group_placed_in_host_memory() -> None:
    cache = SimpleNamespace(groups=[SimpleNamespace(vllm_group=0, layers=[0])])
    config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=["layers.0"], kv_cache_spec=object(), host_resident=True)
        ]
    )
    with pytest.raises(ValueError, match="placed engine cache group 0 in host memory"):
        _validate_groups(cache, config, {"layers.0": object()})


def test_rejects_a_vllm_cache_tensor_placed_in_host_memory() -> None:
    config = SimpleNamespace(
        num_blocks=1,
        kv_cache_tensors=[SimpleNamespace(host_resident=True)],
    )
    with pytest.raises(ValueError, match="placed an engine-owned cache layer in host memory"):
        _validate_kv_cache_tensors(config, KVCacheLayout.LBHNC, {})


@pytest.mark.parametrize("name", LAYOUTS)
def test_validates_upstream_tensor_strides_for_every_global_layout(name: str) -> None:
    spec = FullAttentionSpec(block_size=4, num_kv_heads=2, head_size=4, dtype=torch.bfloat16)
    group = KVCacheGroupSpec(layer_names=["layers.0", "layers.1"], kv_cache_spec=spec)
    selected = KVCacheLayout[name]
    num_blocks = 3
    aggregate_block_stride = spec.page_size_bytes * len(group.layer_names)
    block_stride = aggregate_block_stride if selected.is_block_outermost else None
    strides = compute_layout_strides(
        spec,
        num_blocks,
        len(group.layer_names),
        selected,
        fixed_strides=(None, block_stride, None, None, None),
    )
    shape = (len(group.layer_names), *compute_layer_kv_cache_shape_bytes(spec, num_blocks))
    size = sum((dim - 1) * stride for dim, stride in zip(shape, strides, strict=True)) + shape[-1]
    tensor = KVCacheTensor(
        size=size,
        layers=group.layer_names,
        layer_stride=strides[0],
        block_stride=strides[1],
    )
    config = SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_tensors=[tensor],
        kv_cache_groups=[group],
    )

    _validate_kv_cache_tensors(config, selected, {name: spec for name in group.layer_names})
