"""Compose a framework pipeline whose architecture modules compute through engine components."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterable, Mapping
from contextlib import ExitStack
from copy import copy
from functools import partial
from pathlib import Path
from typing import ClassVar, Protocol, TypeAlias, cast, final
from unittest.mock import patch

import safetensors
import safetensors.torch
import torch
from pydantic import BaseModel, ConfigDict
from safetensors.torch import load_file
from torch import nn
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from typing_extensions import override
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.offloader.module_collector import ModuleDiscovery
from vllm_omni.diffusion.registry import initialize_model

# A component takes the arguments that the framework passes to the module it replaces.
Component: TypeAlias = Callable[..., object]


class Pretrained(Protocol):
    """Checkpoint code that builds a component and reads its weights."""

    @classmethod
    def from_pretrained(cls, path: str, /) -> nn.Module: ...


class Config(BaseModel):
    """The part of a component's `config.json` that names the checkpoint code building it."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    auto_map: dict[str, str] = {}


@final
class Adapter(nn.Module):
    """A tensor-free module that computes through an engine component in place of `skeleton`."""

    def __init__(self, component: Component, skeleton: nn.Module) -> None:
        super().__init__()
        self.component = component
        # Outside the module registry, so traversals and device moves skip the replaced module.
        object.__setattr__(self, "skeleton", skeleton)

    @override
    def forward(self, *args: object, **kwargs: object) -> object:
        return self.component(*args, **kwargs)

    # The engine answers architecture calls; the skeleton keeps host state such as a processor.
    @override
    def __getattr__(self, name: str) -> nn.Module | torch.Tensor:
        try:
            return super().__getattr__(name)
        except AttributeError:
            try:
                return getattr(self.component, name)
            except AttributeError:
                return getattr(self.skeleton, name)


def construct(od_config: OmniDiffusionConfig, directory: Path, names: Iterable[str]) -> nn.Module:
    """Build the registry's pipeline of `directory` on meta tensors, the module of each component
    in `names`, by its `model_index.json` name, without its weights. Keep framework model
    resolution on this local directory without changing the caller's configuration."""
    # TODO(upstream): pipelines take no prebuilt components; a component argument to their
    # constructor would replace this build-then-swap, the checkpoint-code stand-in and the check.
    with ExitStack() as stack:
        for name in names:
            path = directory / name
            config = Config.model_validate_json((path / "config.json").read_bytes())
            reference = config.auto_map.get("AutoModel")
            if reference is None:
                continue
            if not od_config.trust_remote_code:
                raise ValueError(f"building {path} runs checkpoint code; pass trust_remote_code")
            remote = cast("type[Pretrained]", get_class_from_dynamic_module(reference, str(path)))
            _ = stack.enter_context(
                patch.object(remote, "from_pretrained", partial(skeleton, remote.from_pretrained))
            )
        _ = stack.enter_context(torch.device("meta"))
        resolved = copy(od_config)
        resolved.model = str(directory)
        return initialize_model(resolved)


def compose(pipeline: nn.Module, components: Mapping[str, Component]) -> nn.Module:
    """`pipeline` with each engine component, by its `model_index.json` name, in place of its
    module."""
    discovered = ModuleDiscovery.discover(pipeline)
    modules = dict(
        zip(
            [*discovered.dit_names, *discovered.encoder_names, *discovered.vae_names],
            [*discovered.dits, *discovered.encoders, *discovered.vaes],
            strict=True,
        )
    )
    if missing := components.keys() - modules.keys():
        raise ValueError(f"{type(pipeline).__name__} has no module for {sorted(missing)}")
    # The loader reads this list after construction; the engine already holds these weights.
    sources: list[DiffusersPipelineLoader.ComponentSource] = getattr(
        pipeline, "weights_sources", []
    )
    if unserved := {source.prefix.split(".")[0] for source in sources} - components.keys():
        raise ValueError(
            f"{type(pipeline).__name__} reads {sorted(unserved)}, which no engine component serves"
        )
    for name, component in components.items():
        # A framework wrapper keeps its tiling and reaches the checkpoint architecture as `model`.
        architecture = dict(modules[name].named_children()).get("model", modules[name])
        adapter = Adapter(component, architecture)
        for alias, child in [*pipeline.named_modules(remove_duplicate=False)]:
            if child is architecture:
                pipeline.set_submodule(alias, adapter)
    sources.clear()
    return pipeline


def skeleton(from_pretrained: Callable[[str], nn.Module], path: str, /) -> nn.Module:
    """Run checkpoint code that builds a module, on meta tensors and without reading weights."""
    # TODO(upstream): checkpoint `from_pretrained` reads every weight with no option to skip them;
    # the constructor argument named in `construct` would leave this code unused.
    module = sys.modules[from_pretrained.__module__]
    # Checkpoint code reaches `load_file` through the safetensors module or its own import of it.
    imports = [name for name, value in vars(module).items() if value is load_file]
    with ExitStack() as stack:
        _ = stack.enter_context(torch.device("meta"))
        _ = stack.enter_context(patch.object(safetensors.torch, "load_file", shapes))
        for name in imports:
            _ = stack.enter_context(patch.object(module, name, shapes))
        built = from_pretrained(path)
    # The framework moves the module next, and meta tensors cannot move.
    for child in built.modules():
        for name, _ in [*child.named_parameters(recurse=False)]:
            child.register_parameter(name, None)
        for name, _ in [*child.named_buffers(recurse=False)]:
            child.register_buffer(name, None)
    return built


def shapes(filename: str | os.PathLike[str], device: str | int = "cpu") -> dict[str, torch.Tensor]:
    """`load_file` without the data: meta tensors of the stored shapes, enough for a strict load."""
    del device
    with safetensors.safe_open(filename, framework="pt") as file:
        return {
            key: torch.empty(file.get_slice(key).get_shape(), device="meta") for key in file.keys()
        }
