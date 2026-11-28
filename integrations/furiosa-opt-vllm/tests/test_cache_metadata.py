from __future__ import annotations

import pytest
import torch
from vllm.v1.core.kv_cache_utils import validate_kv_cache_layout
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
from vllm.v1.kv_cache_layout import KVCacheLayout

from furiosa_opt_vllm.cache import LAYOUTS


def test_advertises_every_vllm_layout() -> None:
    assert set(LAYOUTS) == {layout.name for layout in KVCacheLayout}
    assert len(LAYOUTS) == 6


@pytest.mark.parametrize("name", LAYOUTS)
def test_vllm_accepts_every_layout_for_uniform_page_sizes(name: str) -> None:
    spec = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=8, dtype=torch.bfloat16)
    groups = [KVCacheGroupSpec(layer_names=["layers.0"], kv_cache_spec=spec)]
    validate_kv_cache_layout(KVCacheLayout[name], groups)


@pytest.mark.parametrize(
    ("name", "accepted"),
    [(layout.name, layout.name in {"BLHNC", "BLNHC"}) for layout in KVCacheLayout],
)
def test_vllm_layout_legality_for_mixed_pages_uses_upstream_rules(
    name: str, accepted: bool
) -> None:
    small = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=8, dtype=torch.bfloat16)
    large = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=16, dtype=torch.bfloat16)
    groups = [
        KVCacheGroupSpec(layer_names=["layers.0"], kv_cache_spec=small),
        KVCacheGroupSpec(layer_names=["layers.1"], kv_cache_spec=large),
    ]
    if accepted:
        validate_kv_cache_layout(KVCacheLayout[name], groups)
    else:
        with pytest.raises(ValueError, match="cannot express this model's mixed page sizes"):
            validate_kv_cache_layout(KVCacheLayout[name], groups)
