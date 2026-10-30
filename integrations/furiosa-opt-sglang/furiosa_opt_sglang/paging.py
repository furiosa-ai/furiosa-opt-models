"""Size the framework's token and request pools for the loaded model and lend it their storage."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, assert_never, cast, final

import torch
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.model_runner_components.layer_setup import ModelLayerInfo
from sglang.srt.runtime_context import get_memory, get_schedule
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from typing_extensions import override

from furiosa_opt_sglang.cache import Allocator, Cache, Pages
from furiosa_opt_sglang.decoder import RELEASED, Decoder
from furiosa_opt_sglang.generation.model import Model
from furiosa_opt_sglang.speculative import Speculative
from furiosa_opt_sglang.speculative.standalone import Draft

if TYPE_CHECKING:
    from furiosa_opt_sglang.generation import Token
    from furiosa_opt_sglang.runner import Runner


@dataclass(frozen=True, slots=True)
class Limits:
    """The framework's resolved token, request and recurrent-state limits."""

    tokens: int | None
    requests: int | None
    page: int
    states: int | None
    radix: bool
    layers: int
    head: int | None


def tokens(runner: Runner, limits: Limits, decoder: Decoder, pages: Pages) -> Cache:
    """Lend the framework a token pool over `pages` of `decoder`'s cache."""
    if decoder.model is None:
        raise RuntimeError(RELEASED)
    runner.kv_cache_dtype = pages.dtype
    runner.kv_cache_dtype_str = (
        "fp8_e4m3" if runner.kv_cache_dtype == torch.float8_e4m3fn else "auto"
    )
    device = decoder.device
    free = device.available - int(device.capacity * (1 - runner.mem_fraction_static))
    match runner.speculative:
        case Speculative(proposer=Draft() as draft):
            pass
        case _:
            draft = None
    count = decoder.blocks(free, *([draft.decoder] if draft is not None else []))
    # The token pool keeps one dummy block.
    block = decoder.pool.block
    capacity = max(0, count - 1) * block
    if limits.tokens is not None and not 0 < limits.tokens <= capacity:
        raise ValueError(f"max_total_tokens must be in 1..{capacity}")
    runner.max_total_num_tokens = (limits.tokens or capacity) // block * block
    runner.max_running_requests = limits.requests or decoder.rows
    cache = Cache(decoder.model.cache, pages, runner.max_total_num_tokens)
    if draft is not None:
        assert cache.storage is not None, "a new token pool holds its storage"
        # The draft caches each request in the blocks the target's table names.
        draft.allocate(cache.storage.blocks)
    runner.token_to_kv_pool = cache
    return cache


@final
class Requests(ReqToTokenPool):
    """SGLang's request rows; releasing a row drops the token a greedy step left for its next step.

    A released row's request runs no further step from that row, so its token has no reader."""

    def __init__(self, size: int, context: int, selected: dict[int, Token]) -> None:
        super().__init__(size, context, "cpu", enable_memory_saver=False)
        self.selected = selected

    @override
    def free_rows(self, indices: list[int]) -> None:
        for index in indices:
            _ = self.selected.pop(index, None)
        super().free_rows(indices)

    @override
    def clear(self) -> None:
        self.selected.clear()
        super().clear()


def assign(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
    batch_size: int,
) -> None:
    """Write each request's new cache slots into its row of the request table."""
    offset = 0
    for i in range(batch_size):
        start, end = int(start_offset[i]), int(end_offset[i])
        req_to_token[req_pool_indices[i], start:end] = out_cache_loc[offset : offset + end - start]
        offset += end - start


def requests(runner: Runner) -> None:
    from sglang.srt.mem_cache.allocation_sizing import get_req_to_token_extra_context_len

    # SGLang reserves table entries past the context for a speculative step's draft slots.
    runner.req_to_token_pool = Requests(
        runner.max_running_requests,
        runner.model_config.context_len + get_req_to_token_extra_context_len(),
        runner.selected,
    )


def initialize(runner: Runner, loaded: Model) -> Model:
    schedule = get_schedule()
    # SGLang's runtime namespace bags erase the declared ServerArgs field types.
    limits = Limits(
        cast("int | None", schedule.max_total_tokens),
        cast("int | None", schedule.max_running_requests),
        cast("int", schedule.page_size),
        cast("int | None", schedule.max_mamba_cache_size),
        not cast("bool", get_memory().disable_radix_cache),
        cast("int", runner.model_config.num_hidden_layers),
        cast("int | None", runner.model_config.head_dim),
    )
    runner.dtype = torch.bfloat16
    runner.sliding_window_size = None
    runner.is_hybrid_swa = False
    runner.weight_load_mem_usage = 0
    runner.memory_saver_adapter = TorchMemorySaverAdapter.create(enable=False)
    runner.layer_info = layers(limits)
    loaded, page = model(runner, limits, loaded)
    runner.token_to_kv_pool_allocator = Allocator(
        size=runner.max_total_num_tokens,
        page_size=page,
        dtype=runner.kv_cache_dtype,
        device="cpu",
        kvcache=runner.token_to_kv_pool,
        need_sort=False,
    )
    return loaded


def layers(limits: Limits) -> ModelLayerInfo:
    """Every checkpoint layer, all owned by this runner."""
    return ModelLayerInfo(
        start_layer=0, end_layer=limits.layers, num_effective_layers=limits.layers
    )


def model(runner: Runner, limits: Limits, loaded: Model) -> tuple[Model, int]:
    """`loaded` with the cache pools its core reads, and their page size."""
    core = loaded.core
    match core:
        case Decoder(pool=pool):
            # Engine block IDs are request-table page starts divided by the page size.
            if limits.page != pool.block:
                raise ValueError(f"the compiled cache pages hold {pool.block} tokens")
            if limits.head is None:
                raise ValueError("the framework must resolve the attention head width")
            heads = runner.model_config.get_num_kv_heads(1)
            pages = Pages.parse(core, heads, limits.head)
            loaded = replace(loaded, cache=tokens(runner, limits, core, pages))
            requests(runner)
            return loaded, pool.block
        case _:
            assert_never(core)
