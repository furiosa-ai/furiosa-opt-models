"""vLLM attention backend name for the engine, which runs attention itself.

The bounded NIXL adapter and its cache staging live in `nixl.py`.
"""

from __future__ import annotations

from typing import NoReturn, final

from typing_extensions import override
from vllm.v1.attention.backend import AttentionBackend


@final
class Attention(AttentionBackend):
    """The attention backend of the engine's cache pages, which the engine attends to itself."""

    @staticmethod
    @override
    def get_name() -> str:
        return "FURIOSA"

    @staticmethod
    @override
    def get_impl_cls() -> NoReturn:
        raise NotImplementedError("the engine runs attention")

    @staticmethod
    @override
    def get_builder_cls() -> NoReturn:
        raise NotImplementedError("the engine runs attention")
