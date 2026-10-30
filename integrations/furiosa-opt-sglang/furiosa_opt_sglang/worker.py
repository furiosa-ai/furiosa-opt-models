"""SGLang worker that runs the engine model for the framework sampler and pooler."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, final

from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.layers.sparse_pooler import SparseEmbeddingOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker

# The upstream type-only circular import shadows this module's own class declaration.
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_schedule
from typing_extensions import override

from furiosa_opt_sglang import embedding, generation
from furiosa_opt_sglang.runner import Runner

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        PPProxyTensors,
    )
    from sglang.srt.model_executor.model_runner import ModelRunner


@final
class Worker(TpModelWorker):
    """SGLang worker that builds the plugin runner and runs its scheduled batches."""

    @override
    def _init_model_runner(self) -> None:
        self._model_runner: ModelRunner = Runner(
            model_config=self.model_config,
            # SGLang's runtime schedule bag leaves its fields untyped.
            mem_fraction_static=cast("float", get_schedule().mem_fraction_static),
            gpu_id=self.gpu_id,
            nccl_port=self.nccl_port,
            server_args=self.server_args,
        )

    @override
    # SGLang's inferred annotation omits the pooler result consumed by its embedding scheduler.
    def forward_batch_embedding(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, batch: ScheduleBatch
    ) -> tuple[EmbeddingPoolerOutput | SparseEmbeddingOutput, bool]:
        runner = self.runner
        # SGLang's forward batch of the request, keeping normalized hidden rows.
        prepared = ForwardBatch.init_new(batch, runner, return_hidden_states_before_norm=False)
        return embedding.pool(runner, runner.model.loaded, batch, prepared), False

    @override
    def forward_batch_generation(
        self,
        batch: ScheduleBatch | None,
        forward_batch: ForwardBatch | None = None,
        pp_proxy_tensors: PPProxyTensors | None = None,
        is_verify: bool = False,
        skip_attn_backend_init: bool | None = None,
        *,
        capture_hidden_mode: CaptureHiddenMode | None = None,
        return_kv_loc_plan: bool = False,
    ) -> GenerationBatchResult:
        # The engine call builds its own operands; prebuilt batches and pipeline proxies would
        # be ignored, and the platform disables the schedules that pass them.
        if (
            batch is None
            or forward_batch is not None
            or pp_proxy_tensors is not None
            or is_verify
            or capture_hidden_mode is not None
            or return_kv_loc_plan
        ):
            raise ValueError("furiosa-opt-sglang requires ordinary scheduled generation")
        runner = self.runner
        return generation.issue(runner, runner.model.loaded, batch, overlap=self.enable_overlap)

    @property
    def runner(self) -> Runner:
        # SGLang fixes this mutable attribute to ModelRunner, even for replacement workers.
        return cast(Runner, self._model_runner)
