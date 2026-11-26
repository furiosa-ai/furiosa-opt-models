"""A vLLM option and the condition the platform requires it to meet."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(frozen=True, slots=True)
class Contract:
    option: str
    accepts: Callable[[VllmConfig], bool]
    requirement: str
