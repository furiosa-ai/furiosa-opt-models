"""The H3 hook preserves the framework's policy over native cache statistics."""

import itertools
from collections.abc import Callable
from typing import Self, cast, final

import numpy as np
import numpy.typing as npt
import pytest
import torch
from torch import nn
from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook, apply_teacache_hook
from vllm_omni.diffusion.cache.teacache.state import TeaCacheState
from vllm_omni.diffusion.hooks import HookRegistry

from furiosa_opt_vllm.omni import teacache
from furiosa_opt_vllm.omni.components import Adapter


def test_decisions_match_stock() -> None:
    """BF16 means preserve every decision and accumulated distance of the original workload."""
    configuration = TeaCacheConfig(transformer_type="MiniMaxH3DiTModel")
    stock = TeaCacheHook(configuration)
    native = Component()
    pipeline = MiniMaxH3Pipeline(native)
    apply_teacache_hook(pipeline.transformer, configuration)
    teacache.adopt(pipeline)
    registry = HookRegistry.get_or_create(pipeline.transformer)
    hook = registry.get_hook("teacache")
    assert isinstance(hook, teacache.Hook)
    state = TeaCacheState()
    base = torch.randn(24, 5376, generator=torch.Generator().manual_seed(5)).to(torch.bfloat16)
    steps = [0.01, 0.02, 0.03, 0.05, 0.08, 0.02, 0.03, 0.06, 0.1, 0.012, 0.04]
    scales = itertools.accumulate(steps, lambda scale, step: scale * (1 + step), initial=1.0)
    previous = None
    for scale in scales:
        probe = (base.float() * scale).to(torch.bfloat16)
        # Compare against the pinned upstream policy entry, rather than restating its formula.
        expected = stock._should_compute_full_transformer(state, probe)  # pyright: ignore[reportPrivateUsage]
        native.means = (
            None
            if previous is None
            else (float((probe - previous).abs().mean()), float(previous.abs().mean()))
        )
        _ = pipeline.transformer()
        assert native.decisions[-1] == expected
        assert (
            cast(TeaCacheState, hook.state_manager.get_state()).accumulated_rel_l1_distance
            == state.accumulated_rel_l1_distance
        )
        state.previous_modulated_input = previous = probe
        state.cnt += 1
    assert True in native.decisions[1:] and False in native.decisions[1:]


def test_failure_and_new_generation() -> None:
    """Failed native calls preserve the exception and count; invalid rows force a new anchor."""
    native = Component()
    pipeline = MiniMaxH3Pipeline(native)
    configuration = TeaCacheConfig(transformer_type="MiniMaxH3DiTModel", rel_l1_thresh=1e9)
    apply_teacache_hook(pipeline.transformer, configuration)
    teacache.adopt(pipeline)
    registry = HookRegistry.get_or_create(pipeline.transformer)
    hook = registry.get_hook("teacache")
    assert isinstance(hook, teacache.Hook)
    _ = pipeline.transformer()
    native.means = (0.0, 1.0)
    _ = pipeline.transformer()
    assert native.decisions == [True, False]
    state = cast(TeaCacheState, hook.state_manager.get_state())
    assert state.cnt == 2
    native.fail = True
    with pytest.raises(LookupError, match="native failure"):
        _ = pipeline.transformer()
    assert state.cnt == 2
    native.fail = False
    native.means = None
    _ = pipeline.transformer()
    assert native.decisions[-1] is True
    assert state.cnt == 3
    assert state.accumulated_rel_l1_distance == 0.0
    registry.reset_hook("teacache")
    assert native.resets == 1
    _ = pipeline.transformer()
    assert native.decisions[-1] is True
    assert cast(TeaCacheState, hook.state_manager.get_state()).cnt == 1


@final
class Component:
    """The native boundary: one completed-statistics callback per transformer call."""

    def __init__(self) -> None:
        self.means: tuple[float, float] | None = None
        self.decisions: list[bool] = []
        self.fail = False
        self.resets = 0

    def prepare(self, *args: object, **kwargs: object) -> Self:
        del args, kwargs
        return self

    def forward(self) -> Self:
        raise AssertionError("the installed cache hook must call forward_cached")

    def forward_cached(
        self,
        decide: Callable[[tuple[float, float] | None], bool],
        /,
        **kwargs: object,
    ) -> Self:
        del kwargs
        self.decisions.append(decide(self.means))
        if self.fail:
            raise LookupError("native failure")
        return self

    def to_host(self) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        return np.array([1.0], dtype=np.float32), np.array([2.0], dtype=np.float32)

    def reset_cache(self) -> None:
        self.resets += 1
        self.means = None


class MiniMaxH3DiTModel(nn.Module):
    """An architecture marker without upstream code or weights."""


@final
class MiniMaxH3Pipeline(nn.Module):
    def __init__(self, component: Component) -> None:
        super().__init__()
        self.partition = "fl2va"
        self.transformer = Adapter(component, MiniMaxH3DiTModel())
