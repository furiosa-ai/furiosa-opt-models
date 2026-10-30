"""vLLM worker that forwards each worker call to the plugin model runner."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never, final

from typing_extensions import override
from vllm.distributed import (
    ensure_model_parallel_initialized,
    get_pp_group,
    get_tp_group,
    init_distributed_environment,
)
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.ec_transfer.ec_transfer_state import (
    ensure_ec_transfer_initialized,
    ensure_ec_transfer_shutdown,
)
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_initialized,
    ensure_kv_transfer_shutdown,
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.model_executor.models import ModelRegistry
from vllm.platforms.interface import set_assigned_physical_gpu_ids
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

from furiosa_opt_vllm import cache, residency
from furiosa_opt_vllm.admission import Refused
from furiosa_opt_vllm.decoder import Decoder
from furiosa_opt_vllm.model_runner import FuriosaModelRunner
from furiosa_opt_vllm.multimodal import Media
from furiosa_opt_vllm.selection import Sampling, Speculation

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorHandshakeMetadata
    from vllm.lora.request import LoRARequest
    from vllm.tasks import SupportedTask
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
    from vllm.v1.outputs import AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput

    from furiosa_opt_vllm.lora import Adapters


@final
class FuriosaWorker(WorkerBase):
    """vLLM worker whose calls go to the model runner it creates in `init_device`."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        super().__init__(
            vllm_config,
            local_rank,
            rank,
            distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        self.runner: FuriosaModelRunner | None = None
        self.failure: Exception | None = None

    @override
    def init_device(self) -> None:
        # Publish the chips vLLM assigns this data-parallel replica, as its own workers do, so the
        # replica's local ranks map into them.
        if (chips := self.parallel_config.assigned_physical_gpu_ids) is not None:
            set_assigned_physical_gpu_ids(chips)
        set_random_seed(self.model_config.seed)
        if self.vllm_config.kv_transfer_config is not None:
            # A KV connector reads its rank from vLLM's process groups: this process is rank 0
            # of 1, holding every chip's cache.
            init_distributed_environment(1, 0, self.distributed_init_method, 0, backend="gloo")
            ensure_model_parallel_initialized(1, 1)
        self.runner = FuriosaModelRunner(self.vllm_config)
        # The worker half of an encoder cache connector, when the config names one.
        ensure_ec_transfer_initialized(self.vllm_config)

    @override
    def shutdown(self) -> None:
        try:
            # Ends the connector's transfers, raising the first that failed.
            ensure_kv_transfer_shutdown()
        finally:
            ensure_ec_transfer_shutdown()
            self.runner = None

    def get_kv_connector_handshake_metadata(
        self,
    ) -> dict[tuple[int, int], KVConnectorHandshakeMetadata] | None:
        """The KV connector's handshake metadata, keyed by this worker's pipeline and tensor
        rank, as vLLM's engine core collects it from its workers."""
        if not has_kv_transfer_group():
            return None
        metadata = get_kv_transfer_group().get_handshake_metadata()
        if metadata is None:
            return None
        return {(get_pp_group().rank_in_group, get_tp_group().rank_in_group): metadata}

    @override
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        self._runner().load_model(load_dummy_weights=load_dummy_weights)
        if has_ec_transfer():
            get_ec_transfer().start_worker_services()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        runner = self._runner()
        if runner.pooler is not None:
            return tuple(sorted(runner.pooler.get_supported_tasks()))
        config = runner.vllm_config.model_config
        if ModelRegistry.is_transcription_only_model(config.architectures, config):
            return ("transcription",)
        return ("generate",)

    @override
    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self._runner().get_kv_cache_spec()

    @override
    def get_supported_kv_cache_layouts(self) -> list[str]:
        # The runner has no torch attention layers for vLLM to ask.
        return list(cache.LAYOUTS)

    def determine_available_memory(self) -> int:
        return self._runner().available_memory()

    def reset_encoder_cache(self) -> None:
        model = self._runner().model
        match model:
            case Media():
                model.features.clear()
                model.transfers.clear()
            case None | Decoder():
                pass
            case _:
                assert_never(model)

    def update_max_model_len(self, max_model_len: int) -> None:
        """Take the context length vLLM fits to cache memory for `--max-model-len -1`, before the
        cache is allocated."""
        self.model_config.max_model_len = max_model_len

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        # As vLLM's workers do, the connector starts before the cache it registers.
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)
        self._runner().initialize_kv_cache(kv_cache_config)

    def sleep(self, level: int = 1) -> None:
        residency.sleep(self._runner(), level)

    def wake_up(self, tags: list[str] | None = None) -> None:
        residency.wake_up(self._runner(), tags)

    def discard(self, tags: tuple[str, ...]) -> None:
        residency.release(self._runner(), tags)

    def reload_weights(
        self,
        weights_iterator: object = None,
        weights_path: str | None = None,
        is_checkpoint_format: bool = True,
    ) -> None:
        """Load the weights again from the served checkpoint's files; the engine loads no other
        source."""
        for name, value, default in (
            ("weights_iterator", weights_iterator, None),
            ("weights_path", weights_path, None),
            ("is_checkpoint_format", is_checkpoint_format, True),
        ):
            if value != default:
                raise Refused(name, f"reload_weights reads only the served checkpoint; omit {name}")
        residency.reload(self._runner())

    @override
    def compile_or_warm_up_model(self) -> CompilationTimes:
        # As in vLLM's workers, sampling starts from the seed whatever loading drew.
        set_random_seed(self.model_config.seed)
        return CompilationTimes(language_model=0.0, encoder=0.0)

    @override
    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput | None:
        if self.failure is not None:
            raise RuntimeError("an earlier step failed to sample") from self.failure
        return self._runner().execute_model(scheduler_output)

    @override
    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        """The step's output: a greedy step's tokens are read when vLLM collects them, after it
        issues the next step, which takes them on the device."""
        try:
            return self._runner().sample_tokens(grammar_output)
        except Exception as error:
            # Async scheduling issues the next step before it collects this error; that step
            # raises it, since its requests lack the tokens this one failed to sample.
            self.failure = error
            raise

    @override
    def get_cache_block_size_bytes(self) -> int:
        return sum(spec.page_size_bytes for spec in self.get_kv_cache_spec().values())

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        selection = self._runner().selection
        match selection:
            case Sampling():
                return None
            case Speculation():
                drafts, selection.drafts = selection.drafts, None
                return drafts
            case _:
                assert_never(selection)

    @override
    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self._adapters().add(lora_request)

    @override
    def remove_lora(self, lora_id: int) -> bool:
        return self._adapters().remove(lora_id)

    @override
    def pin_lora(self, lora_id: int) -> bool:
        return self._adapters().pin(lora_id)

    @override
    def list_loras(self) -> set[int]:
        return self._adapters().ids()

    def _runner(self) -> FuriosaModelRunner:
        if self.runner is None:
            raise RuntimeError("init_device must run before any other worker call")
        return self.runner

    def _adapters(self) -> Adapters:
        adapters = self._runner().adapters
        if adapters is None:
            raise RuntimeError("LoRA is not enabled; start the server with --enable-lora")
        return adapters
