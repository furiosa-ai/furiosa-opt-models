"""Refuse vLLM-Omni's diffusion pipelines at load: the engine loads no pipeline."""

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import NoReturn, final

from typing_extensions import override
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner


def build(*, od_config: OmniDiffusionConfig) -> NoReturn:
    """Refuse the checkpoint's pipeline before reading a weight, with the engine's reason when its
    registry refuses the pipeline's architecture."""
    import furiosa_opt_engine as engine

    if od_config.model is None:
        raise ValueError("a pipeline checkpoint is required")
    engine.LoadOptions().parse(engine.Checkpoint(od_config.model))
    raise NotImplementedError(f"{od_config.model_class_name}: the engine loads no pipeline")


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
