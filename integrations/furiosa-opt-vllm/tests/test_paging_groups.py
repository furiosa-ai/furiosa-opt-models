"""vLLM's hybrid cache manager pages a decoder whose cache groups differ in head geometry and
window at the engine's block size, and counts its memory in the blocks the engine holds.

The cache is Gemma 4 12B's: 40 sliding-window layers of 8 KV heads of 256 values that attend 1024
positions, then 8 full-attention layers of one KV head of 512 values, in 16-token BF16 blocks.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import pytest
import torch
from transformers import AutoConfig
from vllm.config import DeviceConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, SlidingWindowSpec
from vllm.v1.sample.sampler import Sampler

import furiosa_opt_engine as engine
from furiosa_opt_vllm import cache, paging
from furiosa_opt_vllm.admission import Refused, offload
from furiosa_opt_vllm.checkpoint import Location
from furiosa_opt_vllm.decoder import Decoder
from furiosa_opt_vllm.runner import Runner
from furiosa_opt_vllm.selection import Sampling

pytestmark = pytest.mark.usefixtures("platform_env")


# The engine makes a cache spec only for a loaded decoder; these carry the fields paging reads.
@dataclass(frozen=True)
class Pool:
    block: int
    bytes: int


@dataclass(frozen=True)
class Group:
    pool: int
    layers: range
    kind: engine.decoder.Kind
    window: int | None


@dataclass(frozen=True)
class CacheSpec:
    pools: list[Pool]
    groups: list[Group]


@dataclass(frozen=True)
class Loaded:
    cache: CacheSpec
    kv_cache: str


@dataclass(frozen=True)
class Memory:
    capacity: int
    available: int


# A page holds a key and a value of each head for each of a block's tokens, two bytes a value.
SPEC: Final = CacheSpec(
    [Pool(16, 16 * 2 * 8 * 256 * 2), Pool(16, 16 * 2 * 1 * 512 * 2)],
    [
        Group(0, range(40), engine.decoder.Kind.Paged(), 1024),
        Group(1, range(40, 48), engine.decoder.Kind.Paged(), None),
    ],
)
# Device bytes a block takes, since the engine allocates each block in every layer.
FOOTPRINT: Final = 40 * SPEC.pools[0].bytes + 8 * SPEC.pools[1].bytes


@pytest.fixture
def runner(tmp_path: Path) -> Runner:
    """A runner of Gemma 4 12B's language model on an engine decoder that declares `SPEC`."""
    AutoConfig.from_pretrained("unsloth/gemma-4-12b-it-NVFP4").save_pretrained(tmp_path)
    # The model joins after vLLM's config check, which verifies a checkpoint the test lacks.
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    config.model_config = EngineArgs(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=4096,
        language_model_only=True,
    ).create_model_config()
    runner = Runner(config)
    runner.model = Decoder(
        cast("engine.Decoder", cast("object", Loaded(SPEC, "bfloat16"))),
        None,
        Location(str(tmp_path), {}),
        engine.LoadOptions(),
        cast("engine.Device", cast("object", Memory(capacity=64 << 30, available=48 << 30))),
        config.model_config.get_vocab_size(),
    )
    runner.selection = Sampling(Sampler())
    # vLLM's engine core resolves the layout before it asks for the cache's memory.
    _ = resolve_kv_cache_layout(config, [list(cache.LAYOUTS)], paging.describe(runner).values())
    return runner


def plan(runner: Runner, memory: int) -> KVCacheConfig:
    """vLLM's cache config of the layers `runner` reports, as vLLM's engine core plans it for
    `memory`."""
    [config] = get_kv_cache_configs(runner.vllm_config, [paging.describe(runner)], [memory])
    return config


def test_pads_smaller_pages_to_the_largest(runner: Runner) -> None:
    """Each group reports its own geometry and window at the engine's block size, a page smaller
    than another group's padded to it, so vLLM need not grow the group's block to match."""
    specs = paging.describe(runner)
    assert specs.keys() == {f"layers.{index}" for index in range(48)}
    assert specs["layers.0"] == SlidingWindowSpec(
        block_size=16, num_kv_heads=8, head_size=256, dtype=torch.bfloat16, sliding_window=1024
    )
    full = specs["layers.40"]
    assert isinstance(full, FullAttentionSpec)
    assert full == FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        page_size_padded=SPEC.pools[0].bytes,
    )
    assert (full.page_size_bytes, full.unpadded_page_size_bytes) == (
        SPEC.pools[0].bytes,
        SPEC.pools[1].bytes,
    )


def test_reads_a_table_of_its_own_layers_for_each_group(runner: Runner) -> None:
    """vLLM splits the sliding layers among groups of its own, and a request names each engine
    group's blocks by the table of a vLLM group that holds only that engine group's layers."""
    groups = plan(runner, 4096 * FOOTPRINT).kv_cache_groups
    assert len(groups) == 6
    assert {group.kv_cache_spec.block_size for group in groups} == {16}
    tables = cache.Tables.parse(cast("engine.decoder.CacheSpec", cast("object", SPEC)), groups)
    # Each vLLM group's table holds a block numbered after the group.
    blocks = tuple([index] for index in range(len(groups)))
    rows = tables.rows(blocks)
    assert len(rows) == len(SPEC.groups)
    for group, [block] in zip(SPEC.groups, rows, strict=True):
        assert set(groups[block].layer_names) <= {f"layers.{index}" for index in group.layers}


def test_counts_memory_in_the_blocks_the_engine_holds(runner: Runner) -> None:
    """vLLM allocates as many blocks as the engine fits in the cache's memory, though vLLM counts
    fewer bytes a block than the engine allocates."""
    runner.vllm_config.cache_config.kv_cache_memory_bytes = 2048 * FOOTPRINT + FOOTPRINT // 2
    assert plan(runner, paging.available(runner)).num_blocks == 2048


def test_refuses_groups_across_engine_groups(runner: Runner) -> None:
    """Without vLLM's hybrid manager, one vLLM group holds layers of both engine groups, whose
    blocks the engine reads apart."""
    runner.vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = True
    groups = plan(runner, 4096 * FOOTPRINT).kv_cache_groups
    with pytest.raises(ValueError, match="vLLM's cache groups differ from the loaded model's"):
        _ = cache.Tables.parse(cast("engine.decoder.CacheSpec", cast("object", SPEC)), groups)


@pytest.mark.feature("KV cache offloading")
def test_refuses_offloading_several_cache_groups(runner: Runner) -> None:
    """The host tier copies a block of every layer, which holds for one cache group only."""
    config = runner.vllm_config
    config.cache_config.kv_offloading_size = 1
    offload(config)
    planned = plan(runner, 4096 * FOOTPRINT)
    with pytest.raises(Refused) as refusal:
        _ = KVConnectorFactory.create_connector(config, KVConnectorRole.WORKER, planned)
    assert (refusal.value.option, refusal.value.requirement) == (
        "kv_offloading_size",
        "offloads the KV cache of a model with one cache group: the host tier copies a block "
        + "of every layer",
    )
