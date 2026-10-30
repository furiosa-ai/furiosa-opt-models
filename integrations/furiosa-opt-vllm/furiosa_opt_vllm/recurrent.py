"""vLLM's model class of a hybrid attention and Mamba-2 checkpoint, which the platform registers so
that vLLM reads the checkpoint's recurrent state layout; vLLM builds no model of it."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal, final

import torch
from pydantic import BaseModel, ConfigDict
from typing_extensions import override
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    SupportsMambaPrefixCaching,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFunc


class Common(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    num_hidden_layers: int
    vocab_size: int
    hidden_size: int
    num_attention_heads: int = 1
    num_key_value_heads: int = 1


class Hybrid(Common):
    model_type: Literal["bamba"]
    attn_layer_indices: tuple[int, ...] = ()
    mamba_expand: int
    mamba_n_groups: int
    mamba_n_heads: int
    mamba_d_head: int
    mamba_d_state: int
    mamba_d_conv: int

    def shape(self, ranks: int) -> tuple[tuple[int, int], tuple[int, int, int]]:
        """The convolution and recurrence state shapes of one of `ranks` tensor-parallel ranks."""
        return MambaStateShapeCalculator.mamba2_state_shape(
            tp_world_size=ranks,
            intermediate_size=self.mamba_expand * self.hidden_size,
            n_groups=self.mamba_n_groups,
            num_heads=self.mamba_n_heads,
            head_dim=self.mamba_d_head,
            state_size=self.mamba_d_state,
            conv_kernel=self.mamba_d_conv,
        )


@final
class Bamba(torch.nn.Module, HasInnerState, IsHybrid, SupportsMambaPrefixCaching):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        raise RuntimeError("this architecture requires the furiosa-opt-vllm model runner")

    @override
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("this architecture requires the furiosa-opt-vllm model runner")

    def embed_input_ids(self, _input_ids: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("this architecture requires the furiosa-opt-vllm model runner")

    def compute_logits(self, _hidden_states: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("this architecture requires the furiosa-opt-vllm model runner")

    @classmethod
    @override
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        config = Hybrid.model_validate(vllm_config.model_config.hf_text_config.to_dict())
        return config.shape(vllm_config.parallel_config.tensor_parallel_size)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        cache = vllm_config.cache_config
        if any(
            dtype not in ("auto", "float32")
            for dtype in (cache.mamba_cache_dtype, cache.mamba_ssm_cache_dtype)
        ):
            raise ValueError("recurrent models require float32 state")
        return torch.float32, torch.float32

    @classmethod
    @override
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.mamba2_state_copy_func()
