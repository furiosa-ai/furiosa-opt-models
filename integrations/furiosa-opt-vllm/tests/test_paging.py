from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

import furiosa_opt_engine as engine
from furiosa_opt_vllm.decoder import Decoder
from furiosa_opt_vllm.paging import available

if TYPE_CHECKING:
    from furiosa_opt_vllm.checkpoint import Location
    from furiosa_opt_vllm.runner import Runner

# One group of 40 layers, each page 16 tokens of 4 E4M3 KV heads of 128 channels.
LAYERS, PAGE, SCRATCH = 40, 16384, 256
BLOCK = LAYERS * PAGE


class Model:
    """The head geometry vLLM's cache spec reads."""

    def get_num_kv_heads(self, _: object) -> int:
        return 4

    def get_head_size(self) -> int:
        return 128


@pytest.mark.parametrize(
    ("memory", "blocks"),
    [
        # One byte short of an eleventh block beside the scratch pages.
        (LAYERS * SCRATCH * PAGE + 11 * BLOCK - 1, 10),
        # Past what a layer's I32 byte offsets address, its scratch pages included.
        (2**40, 2**31 // PAGE - SCRATCH),
    ],
)
def test_reserves_scratch_pages(memory: int, blocks: int) -> None:
    """vLLM's cache blocks fit the memory and byte offsets the decoder's scratch pages leave."""
    group = SimpleNamespace(
        pool=0, layers=range(LAYERS), kind=engine.decoder.Kind.Paged(), window=None, scratch=SCRATCH
    )
    cache = SimpleNamespace(pools=[SimpleNamespace(block=16, bytes=PAGE)], groups=[group])
    loaded = cast("object", SimpleNamespace(cache=cache, kv_cache="float8_e4m3fn"))
    device = cast("object", SimpleNamespace(available=2**41, capacity=2**41))
    unread = object()
    model = Decoder(
        cast("engine.Decoder", loaded),
        None,
        cast("Location", unread),
        cast("engine.LoadOptions", unread),
        cast("engine.Device", device),
        0,
    )
    runner = cast(
        "object",
        SimpleNamespace(
            model=model,
            state=SimpleNamespace(workspace=0),
            selection=SimpleNamespace(capacity=int),
            vllm_config=SimpleNamespace(
                model_config=Model(),
                parallel_config=None,
                cache_config=SimpleNamespace(
                    kv_cache_memory_bytes=memory, gpu_memory_utilization=1.0
                ),
            ),
        ),
    )

    assert available(cast("Runner", runner)) == blocks * BLOCK
