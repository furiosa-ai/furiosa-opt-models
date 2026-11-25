"""The plugin's section of vLLM's `additional_config`: the options under its `furiosa` key."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_core import PydanticCustomError

from furiosa_opt_engine.placement import (
    MAX_CHIP_ID,
    PE_COUNT,
    has_distinct_chips,
    is_nonempty_pe_range,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# A physical chip ID in the engine's `u8` range.
Chip = Annotated[int, Field(strict=True, ge=0, le=MAX_CHIP_ID)]
# A half-open PE range endpoint; a chip has PEs 0 through 7.
PE = Annotated[int, Field(strict=True, ge=0, le=PE_COUNT)]


class Placement(BaseModel):
    """Physical chips, in rank order, and the half-open range `pes` of the PEs used on each."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    chips: tuple[Chip, ...]
    pes: tuple[PE, PE]

    @field_validator("chips")
    @classmethod
    def distinct(cls, chips: tuple[int, ...]) -> tuple[int, ...]:
        if not has_distinct_chips(chips):
            raise PydanticCustomError("distinct", "Input should be one or more distinct chips")
        return chips

    @field_validator("pes")
    @classmethod
    def ascending(cls, pes: tuple[int, int]) -> tuple[int, int]:
        if not is_nonempty_pe_range(pes):
            raise PydanticCustomError("ascending", "Input should be a nonempty half-open range")
        return pes


class Section(BaseModel):
    """The plugin's options, each at its default unless `additional_config["furiosa"]` sets it."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    # Where the draft model loads; by default on the target's chips.
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
