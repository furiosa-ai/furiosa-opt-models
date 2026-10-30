"""Describe the recurrent architectures' state layout to the framework's linear-attention pools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import final

import torch
from sglang.srt.configs.mamba_utils import (
    BaseLinearStateParams,
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from transformers import BambaConfig as TransformersBambaConfig
from transformers import Mamba2Config as TransformersMamba2Config
from transformers import MambaConfig


@dataclass(kw_only=True, frozen=True)
class Shape:
    conv: list[tuple[int, int]]
    temporal: tuple[int, int]


@dataclass(kw_only=True, frozen=True)
class Parameters(BaseLinearStateParams):
    shape: Shape


@final
class Config(MambaConfig):
    # SGLang reads attention dimensions even when the attention layer list is empty.
    num_attention_heads = 1
    num_key_value_heads = 1

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return []

    @property
    def mamba2_cache_params(self) -> Parameters:
        return Parameters(
            layers=list(range(self.num_hidden_layers)),
            dtype=Mamba2StateDType(conv=torch.float32, temporal=torch.float32),
            shape=Shape(
                conv=[(self.intermediate_size, self.conv_kernel)],
                temporal=(self.intermediate_size, self.state_size),
            ),
        )


@final
class Mamba2Config(TransformersMamba2Config):
    num_attention_heads = 1
    num_key_value_heads = 1

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return []

    @property
    def mamba2_cache_params(self) -> Mamba2CacheParams:
        return Mamba2CacheParams(
            layers=list(range(self.num_hidden_layers)),
            dtype=Mamba2StateDType(conv=torch.float32, temporal=torch.float32),
            shape=Mamba2StateShape.create(
                tp_world_size=1,
                intermediate_size=self.expand * self.hidden_size,
                n_groups=self.n_groups,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                state_size=self.state_size,
                conv_kernel=self.conv_kernel,
            ),
        )


@final
class BambaConfig(TransformersBambaConfig):
    @property
    def full_attention_layer_ids(self) -> list[int]:
        return list(self.attn_layer_indices or [])

    @property
    def mamba2_cache_params(self) -> Mamba2CacheParams:
        def dimension(value: str | int | None, name: str) -> int:
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"hybrid state requires a positive integer {name}")
            return value

        return Mamba2CacheParams(
            layers=[
                layer
                for layer in range(self.num_hidden_layers)
                if layer not in self.full_attention_layer_ids
            ],
            dtype=Mamba2StateDType(conv=torch.float32, temporal=torch.float32),
            shape=Mamba2StateShape.create(
                tp_world_size=1,
                intermediate_size=dimension(self.mamba_expand, "mamba_expand") * self.hidden_size,
                n_groups=dimension(self.mamba_n_groups, "mamba_n_groups"),
                num_heads=dimension(self.mamba_n_heads, "mamba_n_heads"),
                head_dim=dimension(self.mamba_d_head, "mamba_d_head"),
                state_size=dimension(self.mamba_d_state, "mamba_d_state"),
                conv_kernel=dimension(self.mamba_d_conv, "mamba_d_conv"),
            ),
        )
