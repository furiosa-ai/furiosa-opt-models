"""The stock TeaCache policy over the selected H3 FL2VA engine's retained rows.

The CFG-distilled pipeline uses only the positive context. The engine owns probe
and residual storage; Python receives completed means and chooses full blocks.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeAlias, cast, final

import torch
from torch import nn
from typing_extensions import override
from vllm_omni.diffusion.cache.teacache.hook import (
    TeaCacheHook,
    _average_l1_stats_across_sp,  # pyright: ignore[reportPrivateUsage] -- retain the stock reduction.
)
from vllm_omni.diffusion.cache.teacache.state import TeaCacheState
from vllm_omni.diffusion.hooks import HookRegistry

from .components import Adapter

_TRANSFORMER = "MiniMaxH3DiTModel"
_HOOK = "teacache"
_Statistics: TypeAlias = tuple[float, float] | None


class _CachedTransformer(Protocol):
    """The native component's decision and reset contract.

    The callback runs exactly once, with None when prior rows are invalid or
    with distance/norm means already rounded to BF16 and represented in FP32.
    True requests full blocks. Reset drains accepted readers before clearing rows.
    """

    def cached_forward(
        self, decide: Callable[[_Statistics], bool], /, **kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def reset_cache(self) -> None: ...


def _component(module: nn.Module) -> _CachedTransformer:
    if not isinstance(module, Adapter):
        raise TypeError("engine TeaCache requires an Adapter")
    return cast(_CachedTransformer, module.component)


@final
class Hook(TeaCacheHook):
    """Keep the original scalar policy while native code retains all tensor rows.

    Inherited initialization configures the policy and extractor; the extractor
    is never invoked. The selected pipeline sets state.do_true_cfg=False.
    """

    @override
    def new_forward(
        self, module: nn.Module, *args: object, **kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if args:
            raise TypeError("transformer takes forward keyword arguments")
        component = _component(module)
        self.state_manager.set_context("teacache_positive")
        state = cast(TeaCacheState, self.state_manager.get_state())
        output = component.cached_forward(lambda means: self._decide(state, means), **kwargs)
        state.cnt += 1
        self._forward_cnt += 1
        return output

    def _decide(self, state: TeaCacheState, means: _Statistics) -> bool:
        if state.cnt == 0 or means is None:
            # A failed or cancelled native call invalidates its rows. The required full
            # computation establishes a new residual anchor, just as the first call does.
            state.accumulated_rel_l1_distance = 0.0
            return True

        mean_diff = torch.tensor(means[0], dtype=torch.bfloat16)
        mean_prev = torch.tensor(means[1], dtype=torch.bfloat16)
        mean_diff, mean_prev = _average_l1_stats_across_sp(mean_diff, mean_prev)
        rel_distance = (mean_diff / (mean_prev + 1e-8)).item()
        state.accumulated_rel_l1_distance += abs(float(self.rescale_func(rel_distance)))
        threshold = self.config.rel_l1_thresh
        assert threshold is not None
        if state.accumulated_rel_l1_distance < threshold:
            return False
        state.accumulated_rel_l1_distance = 0.0
        return True

    @override
    def reset_state(self, module: nn.Module) -> nn.Module:
        _component(module).reset_cache()
        return super().reset_state(module)


def adopt(pipeline: nn.Module) -> None:
    """Replace existing stock H3 hooks after the runner enables TeaCache.

    Preserve each config and registry's other hooks. Other architectures are
    untouched; a selected H3 adapter without its expected stock hook is an error.
    """
    replacements: list[tuple[HookRegistry, TeaCacheHook]] = []
    for module in pipeline.modules():
        if not isinstance(module, Adapter):
            continue
        registry = getattr(module, "_hook_registry", None)
        hook = registry.get_hook(_HOOK) if isinstance(registry, HookRegistry) else None
        configured_h3 = (
            isinstance(hook, TeaCacheHook) and hook.config.transformer_type == _TRANSFORMER
        )
        if type(module.skeleton).__name__ != _TRANSFORMER and not configured_h3:
            continue
        if (
            not isinstance(registry, HookRegistry)
            or type(hook) is not TeaCacheHook
            or not configured_h3
        ):
            raise RuntimeError("selected H3 Adapter has no stock MiniMaxH3DiTModel TeaCache hook")
        replacements.append((registry, hook))

    if not replacements:
        if type(pipeline).__name__ == "MiniMaxH3Pipeline":
            raise RuntimeError("selected H3 pipeline has no stock TeaCache hook on an Adapter")
        return
    if getattr(pipeline, "partition", None) != "fl2va":
        raise NotImplementedError("engine TeaCache requires the selected H3 FL2VA partition")

    for registry, hook in replacements:
        registry.remove_hook(_HOOK)
        registry.register_hook(_HOOK, Hook(hook.config))
