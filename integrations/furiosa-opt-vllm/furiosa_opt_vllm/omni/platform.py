"""vLLM-Omni platform that hands diffusion pipelines to the engine, which refuses each one."""

from typing import final

import psutil
import torch
from typing_extensions import override
from vllm.platforms.cpu import CpuPlatform
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum


@final
class Platform(OmniPlatform, CpuPlatform):
    _omni_enum = OmniPlatformEnum.OOT

    @classmethod
    @override
    def get_default_stage_config_path(cls) -> str:
        return "vllm_omni/deploy"

    @classmethod
    @override
    def get_diffusion_model_runner_cls(cls) -> str:
        return "furiosa_opt_vllm.omni.pipeline.Runner"

    @classmethod
    @override
    def get_torch_device(cls, local_rank: int | None = None) -> torch.device:
        return torch.device("cpu")

    @classmethod
    @override
    def get_device_count(cls) -> int:
        return 1

    @classmethod
    @override
    def get_current_memory_usage(cls, device: torch.device | str | int | None = None) -> int:
        return psutil.Process().memory_info().rss

    @classmethod
    @override
    def get_device_version(cls) -> str | None:
        return None

    @classmethod
    @override
    def synchronize(cls) -> None:
        pass

    # Torch executes on CPU, whose allocator has no device reservation counters or cache.
    @classmethod
    def reset_peak_memory_stats(cls) -> None:
        pass

    @classmethod
    def empty_cache(cls) -> None:
        pass

    @classmethod
    def max_memory_reserved(cls) -> int:
        return 0

    @classmethod
    def max_memory_allocated(cls) -> int:
        return 0

    @classmethod
    @override
    def get_device_memory(cls, device: torch.device | None = None) -> tuple[int, int]:
        memory = psutil.virtual_memory()
        return memory.available, memory.total

    @classmethod
    @override
    def get_free_memory(cls, device: torch.device | None = None) -> int:
        return cls.get_device_memory(device)[0]

    @classmethod
    @override
    def supports_torch_inductor(cls) -> bool:
        return False

    @classmethod
    @override
    def get_diffusion_attn_backend_cls(
        cls, selected_backend: str | None, head_size: int, allow_trtllm_default: bool = False
    ) -> str:
        return "vllm_omni.diffusion.attention.backends.sdpa.SDPABackend"
