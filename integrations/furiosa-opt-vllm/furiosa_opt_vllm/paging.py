"""vLLM's view of a decoder's cache: the spec of each cached layer, the memory its blocks take in
vLLM's count, and the storage and block tables `initialize` gives the runner.

- vLLM pages every cache group at one page size, so a group of smaller pages reports them padded
  to the largest, at the engine's block size.
- vLLM's groups share each block ID, so vLLM counts a block as the pages of its largest group,
  while the engine allocates each block in every layer.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, assert_never, cast

import torch
from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
from vllm.v1.sample.logits_processor import build_logitsprocs
from vllm.v1.worker.gpu_input_batch import InputBatch

import furiosa_opt_engine as engine
from furiosa_opt_vllm.cache import Tables, assigned, names
from furiosa_opt_vllm.decoder import RELEASED, Decoder
from furiosa_opt_vllm.multimodal import Media

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, KVCacheSpec

    from furiosa_opt_vllm.runner import Runner


def describe(runner: Runner) -> dict[str, KVCacheSpec]:
    loaded = paged(runner)
    if loaded is None:
        return {}
    cache = loaded.cache
    model = runner.vllm_config.model_config
    parallel = runner.vllm_config.parallel_config
    # PyTorch exposes its named dtypes through an untyped module lookup.
    dtype = cast("torch.dtype", getattr(torch, loaded.kv_cache))

    def attention(group: engine.decoder.Group, heads: int, head: int) -> AttentionSpec:
        block = cache.pools[group.pool].block
        match group.kind, group.window:
            case engine.decoder.Kind.Paged(), None:
                return FullAttentionSpec(
                    block_size=block, num_kv_heads=heads, head_size=head, dtype=dtype
                )
            case engine.decoder.Kind.Paged(), int() as window:
                return SlidingWindowSpec(
                    block_size=block,
                    num_kv_heads=heads,
                    head_size=head,
                    dtype=dtype,
                    sliding_window=window,
                )
            case _:
                raise NotImplementedError("the plugin pages only attention groups")

    # The engine names a group's page, not its heads: the group takes the KV heads and head size,
    # as vLLM's config gives them for each layer, that fill its page.
    layers = [
        model.model_arch_config[index] for index in range(model.get_total_num_hidden_layers())
    ]
    shapes = dict.fromkeys(
        (model.get_num_kv_heads(parallel, layer), layer.head_size) for layer in layers
    )
    largest = max((cache.pools[group.pool].bytes for group in cache.groups), default=0)
    specs: dict[str, KVCacheSpec] = {}
    for group in cache.groups:
        page = cache.pools[group.pool].bytes
        shaped = [attention(group, heads, head) for heads, head in shapes]
        fitting = [spec for spec in shaped if spec.unpadded_page_size_bytes == page]
        if not fitting:
            pages = ", ".join(str(spec.unpadded_page_size_bytes) for spec in shaped)
            raise ValueError(f"the engine pages {page} bytes a block, vLLM {pages}")
        spec = fitting[0] if page == largest else replace(fitting[0], page_size_padded=largest)
        specs |= {name: spec for name in names(group)}
    return specs


def available(runner: Runner) -> int:
    """The memory vLLM may fill with cache blocks, in its count: the blocks the engine fits in the
    memory left for the cache, each at the pages of vLLM's largest cache group."""
    specs = describe(runner)
    loaded = paged(runner)
    if loaded is None or not specs:
        return 0
    cache = loaded.cache
    config = runner.vllm_config.cache_config
    device = runner.state.device
    memory = device.available - runner.state.workspace
    if config.kv_cache_memory_bytes is not None:
        if config.kv_cache_memory_bytes > memory:
            raise ValueError("requested KV memory exceeds available memory after workspace")
        memory = config.kv_cache_memory_bytes
    else:
        memory -= int(device.capacity * (1 - config.gpu_memory_utilization))
    # A cache layer addresses its pages by I32 byte offsets.
    reach = min((2**31) // cache.pools[group.pool].bytes for group in cache.groups)
    blocks = runner.selection.capacity(min(max(0, memory) // footprint(cache), reach))
    counted = max(
        sum(assigned(group, name).page_size_bytes for name in group.layer_names)
        for group in get_kv_cache_groups(runner.vllm_config, specs)
    )
    return blocks * counted


def initialize(runner: Runner, config: KVCacheConfig) -> None:
    model = runner.model
    if model is None:
        raise RuntimeError("load_model must precede cache configuration")
    settings = runner.vllm_config.model_config
    scheduler = runner.vllm_config.scheduler_config
    capacity = settings.max_model_len
    blocks = [group.kv_cache_spec.block_size for group in config.kv_cache_groups]
    batch = InputBatch(
        max_num_reqs=scheduler.max_num_seqs,
        max_model_len=capacity,
        max_num_batched_tokens=scheduler.max_num_batched_tokens,
        device=torch.device("cpu"),
        vocab_size=settings.get_vocab_size(),
        block_sizes=blocks,
        kernel_block_sizes=blocks,
        max_num_blocks_per_req=[(capacity + block - 1) // block for block in blocks],
        logitsprocs=build_logitsprocs(
            runner.vllm_config,
            torch.device("cpu"),
            False,
            False,
            custom_logitsprocs=settings.logits_processors or (),
        ),
        logitsprocs_need_output_token_ids=True,
        is_pooling_model=runner.pooler is not None,
        reasoning_config=runner.vllm_config.reasoning_config,
        num_spec_tokens=runner.vllm_config.num_speculative_tokens,
    )
    match model:
        case Media():
            if config.kv_cache_groups:
                raise ValueError("the encoder has no persistent cache")
        case Decoder(model=loaded):
            if loaded is None:
                raise RuntimeError(RELEASED)
            expected = describe(runner)
            tables = Tables.parse(loaded.cache, config.kv_cache_groups)
            if config.num_blocks <= 0:
                raise ValueError("the scheduler must allocate at least one cache block")
            if any(
                assigned(group, name) != expected[name]
                for group in config.kv_cache_groups
                for name in group.layer_names
            ):
                raise ValueError("vLLM's cache spec differs from the loaded model's")
            runner.cache = engine.decoder.Cache(loaded.cache, config.num_blocks)
            runner.tables = tables
        case _:
            assert_never(model)
    runner.input_batch = batch
    runner.selection.allocate(config.num_blocks, runner.requests)


def paged(runner: Runner) -> engine.Decoder | None:
    """The loaded decoder whose cache vLLM pages, or none for a media encoder, which caches
    nothing."""
    match runner.model:
        case None:
            raise RuntimeError("load_model must precede cache configuration")
        case Media():
            return None
        case Decoder(model=loaded):
            if loaded is None:
                raise RuntimeError(RELEASED)
            return loaded
        case _:
            assert_never(runner.model)


def footprint(cache: engine.decoder.CacheSpec) -> int:
    """Device bytes a block of `cache` takes: a page in every layer it caches."""
    return sum(len(group.layers) * cache.pools[group.pool].bytes for group in cache.groups)
