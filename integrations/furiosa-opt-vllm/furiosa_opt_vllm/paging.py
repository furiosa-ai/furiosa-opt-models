from __future__ import annotations

from typing import TYPE_CHECKING, assert_never, cast

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.sample.logits_processor import build_logitsprocs
from vllm.v1.worker.gpu_input_batch import InputBatch

import furiosa_opt_engine as engine
from furiosa_opt_vllm.cache import assigned
from furiosa_opt_vllm.decoder import RELEASED, Decoder
from furiosa_opt_vllm.multimodal import Media

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec

    from furiosa_opt_vllm.runner import Runner


def describe(runner: Runner) -> dict[str, KVCacheSpec]:
    match runner.model:
        case None:
            raise RuntimeError("load_model must precede cache configuration")
        case Media():
            return {}
        case Decoder(model=loaded):
            if loaded is None:
                raise RuntimeError(RELEASED)
            cache = loaded.cache
        case _:
            assert_never(runner.model)
    config = runner.vllm_config.model_config
    # PyTorch exposes its named dtypes through an untyped module lookup.
    dtype = cast("torch.dtype", getattr(torch, loaded.kv_cache))
    specs: dict[str, KVCacheSpec] = {}
    for group in cache.groups:
        pool = cache.pools[group.pool]
        match group.kind:
            case engine.decoder.Kind.Paged() if group.window is None:
                spec = FullAttentionSpec(
                    block_size=pool.block,
                    num_kv_heads=config.get_num_kv_heads(runner.vllm_config.parallel_config),
                    head_size=config.get_head_size(),
                    dtype=dtype,
                )
            case _:
                raise NotImplementedError("the plugin pages only full-history attention groups")
        if spec.page_size_bytes != pool.bytes:
            raise ValueError(
                f"the engine pages {pool.bytes} bytes a block, vLLM {spec.page_size_bytes}"
            )
        specs |= {f"layers.{index}": spec for index in group.layers}
    return specs


def available(runner: Runner) -> int:
    model = runner.model
    if model is None:
        raise RuntimeError("load_model must precede cache configuration")
    spec = describe(runner)
    if not spec:
        return 0
    config = runner.vllm_config.cache_config
    block_bytes = sum(layer.page_size_bytes for layer in spec.values())
    device = model.device
    memory = device.available - runner.state.workspace
    if config.kv_cache_memory_bytes is not None:
        if config.kv_cache_memory_bytes > memory:
            raise ValueError("requested KV memory exceeds available memory after workspace")
        memory = config.kv_cache_memory_bytes
    else:
        memory -= int(device.capacity * (1 - config.gpu_memory_utilization))
    # A cache layer addresses its pages by I32 byte offsets.
    max_blocks = min((2**31) // layer.page_size_bytes for layer in spec.values())
    match model:
        case Decoder():
            blocks = min(max(0, memory) // block_bytes, max_blocks)
            return runner.selection.capacity(blocks) * block_bytes
        case Media():
            return max(0, min(memory, max_blocks * block_bytes))
        case _:
            assert_never(model)


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
            # A request names its blocks group by group, as vLLM's block tables list them.
            if [set(group.layer_names) for group in config.kv_cache_groups] != [
                {f"layers.{index}" for index in group.layers} for group in loaded.cache.groups
            ]:
                raise ValueError("vLLM's cache groups differ from the loaded model's")
            if config.num_blocks <= 0:
                raise ValueError("the scheduler must allocate at least one cache block")
            if any(
                assigned(group, name) != expected[name]
                for group in config.kv_cache_groups
                for name in group.layer_names
            ):
                raise ValueError("vLLM's cache spec differs from the loaded model's")
            runner.cache = engine.decoder.Cache(loaded.cache, config.num_blocks)
        case _:
            assert_never(model)
    runner.input_batch = batch
    runner.selection.allocate(config.num_blocks, runner.requests)
