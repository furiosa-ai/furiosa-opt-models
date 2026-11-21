from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING, assert_never, cast

import torch
from vllm.v1.core.kv_cache_utils import (
    compute_layout_strides,
    validate_kv_cache_layout,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
    compute_layer_kv_cache_shape_bytes,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.sample.logits_processor import build_logitsprocs
from vllm.v1.worker.gpu_input_batch import InputBatch

import furiosa_opt_engine as engine
from furiosa_opt_vllm.cache import assigned, group_indices
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
    # PyTorch exposes its named dtypes through an untyped module lookup.
    dtype = cast("torch.dtype", getattr(torch, loaded.kv_cache))
    specs: dict[str, KVCacheSpec] = {}
    for group in cache.groups:
        pool = cache.pools[group.pool]
        shape = tuple(pool.shape)
        if len(shape) != 4 or shape[0] != pool.block or shape[1] != 2:
            raise ValueError(
                f"engine pool {group.pool} must describe [tokens, key/value, heads, features]"
            )
        if pool.element_bytes != dtype.itemsize:
            raise ValueError(
                f"engine pool {group.pool} stores {pool.element_bytes}-byte values, "
                f"but vLLM cache dtype {dtype} uses {dtype.itemsize} bytes"
            )
        expected_strides = (
            shape[1] * shape[2] * shape[3] * pool.element_bytes,
            shape[2] * shape[3] * pool.element_bytes,
            shape[3] * pool.element_bytes,
            pool.element_bytes,
        )
        if tuple(pool.strides) != expected_strides:
            raise ValueError(f"engine pool {group.pool} has a non-dense cache page")
        dense_page_bytes = prod(shape) * pool.element_bytes
        common = {
            "block_size": pool.block,
            "num_kv_heads": shape[2],
            "head_size": shape[3],
            "dtype": dtype,
            "page_size_padded": pool.bytes if pool.bytes != dense_page_bytes else None,
        }
        match group.kind:
            case engine.decoder.Kind.Paged() if group.window is None:
                spec: KVCacheSpec = FullAttentionSpec(**common)
            case engine.decoder.Kind.Paged() if group.window is not None and group.window > 0:
                spec = SlidingWindowSpec(**common, sliding_window=group.window)
            case engine.decoder.Kind.State():
                raise NotImplementedError("the plugin does not serve recurrent State cache groups")
            case engine.decoder.Kind.Cross():
                raise NotImplementedError("the plugin does not serve encoder Cross cache groups")
            case _:
                raise ValueError(f"engine group {group.vllm_group} has an invalid cache window")
        if spec.page_size_bytes != pool.bytes:
            raise ValueError(
                f"the engine pages {pool.bytes} bytes a block, vLLM {spec.page_size_bytes}"
            )
        for index in group.layers:
            name = f"layers.{index}"
            if name in specs:
                raise ValueError(f"engine cache metadata names layer {name} more than once")
            specs[name] = spec
    if set(specs) != {f"layers.{index}" for index in range(cache.layer_count)}:
        raise ValueError("engine cache metadata does not cover every model layer exactly once")
    return specs


def _validate_kv_cache_tensors(
    config: KVCacheConfig,
    layout: KVCacheLayout,
    expected: dict[str, KVCacheSpec],
) -> None:
    """Check vLLM's physical layer/block strides against its pinned layout rules."""
    if config.num_blocks <= 0:
        raise ValueError("the scheduler must allocate at least one cache block")
    actual: set[str] = set()
    for tensor in config.kv_cache_tensors:
        if tensor.host_resident:
            raise ValueError("vLLM placed an engine-owned cache layer in host memory")
        if not tensor.layers or len(set(tensor.layers)) != len(tensor.layers):
            raise ValueError("vLLM KVCacheTensor has no layers or duplicate layer names")
        if tensor.size <= 0 or tensor.offset < 0:
            raise ValueError("vLLM KVCacheTensor has an invalid size or offset")
        for name in tensor.layers:
            if name not in expected or name in actual:
                raise ValueError(f"vLLM KVCacheTensor has unexpected or duplicate layer {name}")
            actual.add(name)
    if actual != set(expected):
        raise ValueError("vLLM KVCacheTensor metadata does not cover every engine layer")

    bytes_per_block = sum(
        group.kv_cache_spec.page_size_bytes
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        else group.kv_cache_spec.page_size_bytes * len(group.layer_names)
        for group in config.kv_cache_groups
    )
    interleaved_block_stride = bytes_per_block if layout.is_block_outermost else None
    for tensor in config.kv_cache_tensors:
        layer_specs = [expected[name] for name in tensor.layers]
        if any(spec != layer_specs[0] for spec in layer_specs[1:]):
            raise ValueError("vLLM KVCacheTensor combines layers with different page specs")
        spec = layer_specs[0]
        expected_strides = compute_layout_strides(
            spec,
            config.num_blocks,
            len(tensor.layers),
            layout,
            fixed_strides=(None, interleaved_block_stride, None, None, None),
        )
        if (tensor.layer_stride, tensor.block_stride) != expected_strides[:2]:
            raise ValueError("vLLM KVCacheTensor strides disagree with its selected layout")
        shape = (len(tensor.layers), *compute_layer_kv_cache_shape_bytes(spec, config.num_blocks))
        last = sum((dim - 1) * stride for dim, stride in zip(shape, expected_strides, strict=True))
        end = tensor.offset + last + shape[-1]
        if end > tensor.size:
            raise ValueError("vLLM KVCacheTensor layer pages exceed their backing allocation")


def _validate_groups(
    cache: engine.decoder.CacheSpec,
    config: KVCacheConfig,
    expected_specs: dict[str, KVCacheSpec],
) -> dict[int, int]:
    resolved = group_indices(cache, config.kv_cache_groups)
    for group_id, index in resolved.items():
        group = config.kv_cache_groups[index]
        if group.host_resident:
            raise ValueError(f"vLLM placed engine cache group {group_id} in host memory")
        layer_specs = {name: expected_specs[name] for name in group.layer_names}
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            wrapped = group.kv_cache_spec.kv_cache_specs
            if set(wrapped) != set(group.layer_names):
                raise ValueError(
                    f"vLLM UniformTypeKVCacheSpecs for group {group_id} has a different layer set"
                )
            if not UniformTypeKVCacheSpecs.is_uniform_type(wrapped):
                raise ValueError(
                    f"vLLM cache group {group_id} mixes incompatible attention spec types"
                )
        elif any(spec != group.kv_cache_spec for spec in layer_specs.values()):
            raise ValueError(
                f"vLLM cache group {group_id} needs UniformTypeKVCacheSpecs for its layer pages"
            )
    return resolved


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
            vllm_group_indices = _validate_groups(loaded.cache, config, expected)
            if any(
                assigned(group, name) != expected[name]
                for group in config.kv_cache_groups
                for name in group.layer_names
            ):
                raise ValueError("vLLM's cache spec differs from the loaded model's")
            layout = runner.vllm_config.cache_config.get_resolved_kv_cache_layout()
            validate_kv_cache_layout(layout, config.kv_cache_groups)
            _validate_kv_cache_tensors(config, layout, expected)
            runner.cache = engine.decoder.Cache(loaded.cache, config.num_blocks)
            runner.engine_group_block_table_indices = tuple(
                vllm_group_indices[group.vllm_group] for group in loaded.cache.groups
            )
        case _:
            assert_never(model)
    runner.input_batch = batch
    runner.selection.allocate(config.num_blocks, runner.requests)
