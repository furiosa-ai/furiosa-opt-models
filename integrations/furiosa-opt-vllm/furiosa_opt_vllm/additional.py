"""The plugin's section of vLLM's `additional_config`: the options under its `furiosa` key."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_core import PydanticCustomError

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# A chip or PE index, which no bool or float stands for.
Index = Annotated[int, Field(strict=True, ge=0)]


class Placement(BaseModel):
    """Physical chips, in rank order, and the half-open range `pes` of the PEs used on each."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    chips: tuple[Index, ...]
    pes: tuple[Index, Index]

    @field_validator("chips")
    @classmethod
    def distinct(cls, chips: tuple[int, ...]) -> tuple[int, ...]:
        if not chips or len(set(chips)) < len(chips):
            raise PydanticCustomError("distinct", "Input should be one or more distinct chips")
        return chips

    @field_validator("pes")
    @classmethod
    def ascending(cls, pes: tuple[int, int]) -> tuple[int, int]:
        if pes[0] >= pes[1]:
            raise PydanticCustomError("ascending", "Input should be a nonempty half-open range")
        return pes


class Section(BaseModel):
    """The plugin's options, each at its default unless `additional_config["furiosa"]` sets it."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    # Where the draft model loads; by default on the chips the target leaves free.
    draft_placement: Placement | None = None

    @classmethod
    def parse(cls, config: VllmConfig) -> Section:
        """The section of `config`'s `additional_config`, at its defaults without a `furiosa`
        key."""
        return Config.model_validate(config.additional_config).furiosa


class Config(BaseModel):
    """vLLM's `additional_config`, of which the plugin reads only the `furiosa` key."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    furiosa: Section = Section()
