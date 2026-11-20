"""Compose the framework's pipeline over one engine owner on four RNGD chips."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from shutil import copyfile
from typing import TYPE_CHECKING, final

from torch import nn
from typing_extensions import override
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.offloader.config import OffloadStrategy, resolve_offload
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

from .components import compose, construct

if TYPE_CHECKING:
    import furiosa_opt_engine as engine


def build(*, od_config: OmniDiffusionConfig) -> nn.Module:
    """Check the load, construct architecture skeletons, then place one complete engine pipeline.

    One framework worker drives all four chips. The engine streams the text encoder beside
    resident denoiser/decoders; component views keep this same pipeline and context alive.
    """
    import furiosa_opt_engine as engine

    path = directory(od_config)
    assert od_config.model is not None
    remote = not Path(od_config.model).is_dir()
    if remote and od_config.model_class_name == "MiniMaxH3Pipeline":
        validate(od_config)
    options = engine.LoadOptions()
    # Metadata suffices for registry selection and H3's component-document checks.
    # Preserve the registry's refusal for other architectures before fetching weights.
    context = configured(path) if remote else nullcontext(engine.Checkpoint(path))
    with context as checkpoint:
        options.parse(checkpoint)
    if od_config.model_class_name != "MiniMaxH3Pipeline":
        raise NotImplementedError(f"{od_config.model_class_name}: the engine loads no pipeline")
    if remote:
        fetch(od_config, path)
        checkpoint = engine.Checkpoint(path)
    else:
        validate(od_config)
    profile = engine.minimax_h3.Pipeline
    served = [name for name in profile.names if od_config.model_loaded.get(name, True)]
    registered = construct(od_config, path, served)
    [chips] = place([profile.chips], {chip: numa_node(chip) for chip in visible()})
    device = engine.Device(engine.Placement(chips, range(8)))
    pipeline = options.load(checkpoint, device)
    if not isinstance(pipeline, profile):
        raise TypeError(f"{type(pipeline).__name__} is not the selected H3 pipeline")
    od_config.max_num_seqs = pipeline.max_num_seqs
    return compose(registered, {name: getattr(pipeline, name) for name in served})


def validate(od_config: OmniDiffusionConfig) -> None:
    """Refuse unsupported H3 framework choices before fetching checkpoint weights."""
    if od_config.parallel_config.world_size != 1:
        raise NotImplementedError("one Omni worker drives the H3 pipeline's four RNGD chips")
    if od_config.enable_sleep_mode:
        raise NotImplementedError("H3 pipeline sleep/wake is not implemented")
    if resolve_offload(od_config).strategy is not OffloadStrategy.NONE:
        raise NotImplementedError(
            "the H3 engine owns encoder streaming and resident decoder weights"
        )
    if od_config.quantization_config is not None or od_config.lora_path:
        raise NotImplementedError("H3 loads its checkpoint precision without LoRA adapters")
    if od_config.cache_backend not in (None, "none", "tea_cache"):
        raise NotImplementedError(
            f"H3 cache backend {od_config.cache_backend!r} is not implemented"
        )
    if not od_config.enforce_eager:
        raise NotImplementedError("the H3 engine requires enforce_eager=True")


def directory(od_config: OmniDiffusionConfig) -> Path:
    """Resolve the framework's pipeline path using only remote JSON metadata.

    Run path resolvers against the local snapshot so they select the same partition without
    fetching component weights themselves. Nested configs and sidecars retain their paths.
    """
    from vllm.utils.import_utils import resolve_obj_by_qualname
    from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
    from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType
    from vllm_omni.diffusion.registry import DiffusionModelRegistry
    from vllm_omni.model_executor.model_loader.weight_utils import (
        download_weights_from_hf_specific,
    )

    if od_config.model is None:
        raise ValueError("a pipeline checkpoint is required")
    classes = DiffusionModelRegistry.models
    served = classes.get(od_config.model_class_name or "")
    resolvers = {
        stage.model_path_resolver
        for pipeline in OMNI_PIPELINES.values()
        if isinstance(pipeline, PipelineConfig)
        for stage in pipeline.stages
        if stage.execution_type == StageExecutionType.DIFFUSION
        and served is not None
        and classes.get(stage.model_arch or "") == served
        and stage.model_path_resolver is not None
    }
    resolve: Callable[[str, str | None, str | None], str] | None = None
    match sorted(resolvers):
        case []:
            pass
        case [qualname]:
            resolve = resolve_obj_by_qualname(qualname)
        case several:
            raise ValueError(f"{od_config.model_class_name}: several path resolvers {several}")
    path = Path(od_config.model)
    if not path.is_dir():
        path = Path(
            download_weights_from_hf_specific(
                od_config.model,
                None,
                ["*.json"],
                od_config.revision,
                # An index makes Checkpoint open its shards, which are not fetched yet.
                ignore_patterns=["*.safetensors.index.json"],
                require_all=True,
            )
        )
    return (
        Path(resolve(str(path), od_config.revision, od_config.task_type))
        if resolve is not None
        else path
    )


def fetch(od_config: OmniDiffusionConfig, path: Path) -> None:
    """Fetch the validated remote partition at its already resolved Hub commit."""
    from vllm_omni.model_executor.model_loader.weight_utils import (
        download_weights_from_hf_specific,
    )

    assert od_config.model is not None
    if Path(od_config.model).is_dir():
        return
    snapshot = next(parent for parent in (path, *path.parents) if parent.parent.name == "snapshots")
    relative = path.relative_to(snapshot)
    _ = download_weights_from_hf_specific(
        od_config.model,
        None,
        [f"{relative}/**" if relative.parts else "**"],
        snapshot.name,
        require_all=True,
    )


@contextmanager
def configured(path: Path) -> Generator[engine.Checkpoint, None, None]:
    """The pipeline's JSON configuration, isolated from any cached weight files or indexes."""
    import furiosa_opt_engine as engine

    with tempfile.TemporaryDirectory() as temporary:
        metadata = Path(temporary)
        for document in path.rglob("*.json"):
            if not document.is_file() or document.name.endswith(".safetensors.index.json"):
                continue
            target = metadata / document.relative_to(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            _ = copyfile(document, target)
        yield engine.Checkpoint(metadata)


def visible() -> list[int]:
    """Installed chip IDs in the runtime's visible order."""
    listed = os.environ.get("FURIOSA_VISIBLE_CHIPS", "").split(",")
    if chips := [int(chip) for chip in listed if chip.strip()]:
        return chips
    nodes = Path("/dev/rngd").glob("npu*mgmt")
    return sorted(int(node.name.removeprefix("npu").removesuffix("mgmt")) for node in nodes)


def numa_node(chip: int) -> int:
    """The NUMA node of the chip's PCI device."""
    bus = Path(f"/sys/class/rngd_mgmt/rngd!npu{chip}mgmt/busname").read_text().strip()
    return int(Path("/sys/bus/pci/devices", bus, "numa_node").read_text())


def place(counts: Sequence[int], sockets: Mapping[int, int]) -> list[list[int]]:
    """Preserve the repro's placement: prefer one socket, placing the largest loads first."""
    free, placed = [*sockets], [list[int]() for _ in counts]
    for load in sorted(range(len(counts)), key=lambda load: -counts[load]):
        groups = [
            [chip for chip in free if sockets[chip] == socket]
            for socket in dict.fromkeys(sockets[chip] for chip in free)
        ]
        group = next((group for group in reversed(groups) if len(group) >= counts[load]), free)
        if len(group) < counts[load]:
            raise ValueError(f"loads of {[*counts]} chips need more chips than {[*sockets]}")
        placed[load] = group[: counts[load]]
        free = [chip for chip in free if chip not in placed[load]]
    return placed


@final
class Runner(DiffusionModelRunner):
    """The diffusion runner that loads its pipeline through `build`."""

    @override
    def load_model(
        self,
        memory_pool_context_fn: Callable[..., AbstractContextManager[object]] | None = None,
        load_format: str = "default",
        custom_pipeline_name: str | None = None,
    ) -> None:
        if load_format != "default" or custom_pipeline_name is not None:
            raise ValueError(
                "a pipeline loads through the engine, without a load format or custom pipeline"
            )
        # The loader calls a custom pipeline with `od_config` alone, so `build` loads the engine.
        super().load_model(
            memory_pool_context_fn, "custom_pipeline", f"{build.__module__}.{build.__qualname__}"
        )
        if self.od_config.cache_backend == "tea_cache":
            from .teacache import adopt

            assert isinstance(self.pipeline, nn.Module)
            adopt(self.pipeline)
