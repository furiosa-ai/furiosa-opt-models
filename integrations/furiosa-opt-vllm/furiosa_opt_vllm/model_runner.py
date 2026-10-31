from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, assert_never, final

from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.v1.worker.gpu.kv_connector import get_kv_connector

from furiosa_opt_vllm import embedding, generation, load, multimodal, paging, request
from furiosa_opt_vllm.decoder import Decoder
from furiosa_opt_vllm.generation import sequence
from furiosa_opt_vllm.multimodal import Media
from furiosa_opt_vllm.offload import RNGDOffloadingConnector
from furiosa_opt_vllm.runner import Runner

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
    from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput


@final
class FuriosaModelRunner(Runner):
    def __init__(self, vllm_config: VllmConfig) -> None:
        super().__init__(vllm_config)
        embedding.configure(self)

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        load.bind(self, load_dummy_weights=load_dummy_weights)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return paging.describe(self)

    def available_memory(self) -> int:
        return paging.available(self)

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        paging.initialize(self, kv_cache_config)
        if not has_kv_transfer_group():
            return
        connector = get_kv_transfer_group()
        # The platform admits only the offloading connector, for the cache of a decoder the engine
        # holds.
        assert isinstance(connector, RNGDOffloadingConnector) and self.cache is not None
        assert isinstance(self.model, Decoder) and self.model.model is not None
        connector.register(self.cache, self.model.model.chips)
        # vLLM's attention layers, which the engine replaces, hold no cache tensors to register.
        self.kv_connector = get_kv_connector(self.vllm_config, {})

    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput | None:
        if self.model is None:
            raise RuntimeError("load_model must precede execution")
        if self.pending is not None:
            raise RuntimeError("sample_tokens must collect the previous execution")
        request.update(self, scheduler_output)
        if isinstance(self.model, Media):
            return multimodal.step(self.model, scheduler_output, self.requests)
        if not scheduler_output.num_scheduled_tokens:
            return self.kv_connector.no_forward(scheduler_output)
        self.kv_connector.pre_forward(scheduler_output)
        match self.model:
            case Decoder():
                output = sequence.forward(self, self.model, scheduler_output)
            case _:
                assert_never(self.model)
        self.kv_connector.finish_forward()
        if output is None:
            # Sampling reports the step's transfers with its tokens.
            return None
        transferred = self.kv_connector.post_forward(scheduler_output.finished_req_ids)
        return output if transferred is None else replace(output, kv_connector_output=transferred)

    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        return generation.sample(self, grammar_output)
