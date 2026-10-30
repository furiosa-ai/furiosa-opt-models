"""vLLM platform that keeps framework tensors on the host and refuses configurations the plugin
does not serve."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, final

import torch
from pydantic import ValidationError
from typing_extensions import override
from vllm.platforms.interface import Platform, PlatformEnum

from furiosa_opt_vllm.additional import Section
from furiosa_opt_vllm.admission import (
    CONTRACTS,
    Refused,
    adapters,
    offload,
    options,
    speculative,
    verify_checkpoint,
)

if TYPE_CHECKING:
    from vllm.config import CacheConfig, ModelConfig, VllmConfig
    from vllm.inputs import EngineInput
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.selector import AttentionSelectorConfig


@final
class FuriosaPlatform(Platform):
    """vLLM platform for RNGD: host tensors, the plugin worker and no attention backend."""

    _enum: PlatformEnum = PlatformEnum.OOT
    # vLLM's process groups place their tensors on this torch device; framework tensors are on
    # the host.
    device_name: str = "cpu"
    device_type: str = "furiosa"
    # PyTorch needs a fallback dispatch key because RNGD has no registered backend.
    dispatch_key: str = "CPU"
    ray_device_key: str = ""
    # The runtime's mask of the chips a process may open, which vLLM's local ranks index.
    device_control_env_var: str = "FURIOSA_VISIBLE_CHIPS"

    @classmethod
    @override
    def pre_register_and_update(cls, parser: FlexibleArgumentParser | None = None) -> None:
        from vllm.model_executor.models import ModelRegistry

        ModelRegistry.register_model("BambaForCausalLM", "furiosa_opt_vllm.recurrent:Bamba")

    @classmethod
    @override
    def get_device_name(cls, device_id: int = 0) -> str:
        return "rngd"

    @override
    def uses_host_device_handling(self) -> bool:
        return True

    @classmethod
    @override
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        if vllm_config.speculative_config is not None:
            # Proposers draft from each step's accepted tokens on the host, so steps never overlap.
            vllm_config.scheduler_config.async_scheduling = False
        try:
            section = Section.parse(vllm_config)
        except ValidationError as error:
            raise Refused(
                "additional_config.furiosa",
                "; ".join(
                    ".".join(map(str, ("additional_config", *detail["loc"]))) + f": {detail['msg']}"
                    for detail in error.errors()
                ),
            ) from None
        offload(vllm_config)
        for contract in CONTRACTS:
            if not contract.accepts(vllm_config):
                raise Refused(contract.option, contract.requirement)
        _ = speculative(vllm_config, section)
        # One process drives every chip of a replica; vLLM would start a process per rank.
        parallel_config.distributed_executor_backend = "uni"
        # Host-side sampling runs torch CPU kernels on OpenMP, whose thread pool does not survive
        # fork: an engine core forked after the parent ran one crashes in its first parallel region.
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        # vLLM's ngram proposer runs numba, whose OpenMP layer resizes that pool to numba's thread
        # count, every CPU by default, on which each later sampling kernel waits for busy cores.
        os.environ.setdefault("NUMBA_NUM_THREADS", str(torch.get_num_threads()))
        if model_config:
            recurrent_state(model_config, vllm_config.cache_config)
            verify_checkpoint(model_config, vllm_config.load_config, "model", options(vllm_config))
            adapters(vllm_config)
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "furiosa_opt_vllm.worker.FuriosaWorker"

    @classmethod
    @override
    def update_block_size_for_backend(cls, vllm_config: VllmConfig) -> None:
        if (
            vllm_config.model_config
            and vllm_config.model_config.hf_text_config.model_type == "bamba"
        ):
            from vllm.v1.attention.backend import AttentionBackend

            # Indexed rows accept the base backend's unrestricted positive block size.
            cls._align_hybrid_block_size(vllm_config, AttentionBackend)

    @classmethod
    @override
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @classmethod
    @override
    def get_attn_backend_cls(
        cls,
        selected_backend: AttentionBackendEnum,
        attn_selector_config: AttentionSelectorConfig,
        num_heads: int | None = None,
    ) -> str:
        raise NotImplementedError(
            "furiosa-opt-vllm runs attention inside the extension, so vLLM selects no backend"
        )

    @classmethod
    @override
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        raise NotImplementedError(
            "furiosa-opt-vllm sizes KV through FuriosaWorker.determine_available_memory"
        )

    @classmethod
    @override
    def is_pin_memory_available(cls) -> bool:
        return False

    @override
    def is_sleep_mode_available(self) -> bool:
        # The worker drops and loads the engine's weights and cache; vLLM's allocator holds neither.
        return True

    @classmethod
    @override
    def num_compute_units(cls, device_id: int = 0) -> int:
        # vLLM's sampler kernels size their grids by this; they run on the host's threads.
        return torch.get_num_threads()

    @classmethod
    @override
    def set_device(cls, device: torch.device) -> None:
        # The model runner opens the device when loading the model.
        pass

    @classmethod
    @override
    def manual_seed_all(cls, seed: int) -> None:
        # The device draws no random numbers; vLLM seeds the host generators that sampling uses.
        pass

    @classmethod
    @override
    def inference_mode(cls) -> torch.autograd.grad_mode.inference_mode:
        return torch.inference_mode(mode=True)

    @classmethod
    @override
    def validate_request(
        cls, processed_inputs: EngineInput, params: SamplingParams | PoolingParams
    ) -> None:
        # The engine's calls take token IDs and return each row's greedy token, not its scores.
        from vllm.sampling_params import SamplingParams

        from furiosa_opt_vllm.generation import greedy

        if isinstance(params, SamplingParams) and not greedy(params):
            raise Refused(
                "sampling_params", "serves only greedy decoding, without logprobs or logit changes"
            )
        if processed_inputs["type"] == "embeds":
            raise Refused("prompt_embeds", "takes token IDs, not prompt embeddings")


def recurrent_state(model: ModelConfig, cache: CacheConfig) -> None:
    """Refuse recurrent layouts the engine cannot read and pin its float32 state dtypes."""
    if model.hf_text_config.model_type not in ("mamba", "mamba2", "bamba"):
        return
    if model.hf_text_config.model_type in ("mamba2", "bamba"):
        from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

        if is_conv_state_dim_first():
            raise ValueError("recurrent models require the SD SSM conv state layout")
    if any(
        dtype not in ("auto", "float32")
        for dtype in (cache.mamba_cache_dtype, cache.mamba_ssm_cache_dtype)
    ):
        raise ValueError("furiosa-opt-vllm requires float32 recurrent state")
    cache.mamba_cache_dtype = "float32"
    cache.mamba_ssm_cache_dtype = "float32"
