"""A LoRA adapter's PEFT data as a load brings it, parsed once into the values the manager checks.

- The configuration and added tokens come from an adapter directory's JSON files or from a
  `/load_lora_adapter_from_tensors` request, the factors from that request's deserialized tensors.
  Each is refused with a `ValueError` naming what is malformed, before anything reaches the device.
- The engine reads the whole configuration again when it loads the adapter, with its own refusals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, ClassVar, Final, TypeVar, cast

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
    TypeAdapter,
    ValidationError,
)
from sglang.srt.lora.utils import get_normalized_target_modules

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

T = TypeVar("T")

JSON: Final = TypeAdapter(JsonValue)
CONFIG: Final = TypeAdapter(dict[str, JsonValue])
TOKENS: Final = TypeAdapter(dict[str, StrictInt])


class Fields(BaseModel):
    """The fields of a PEFT LoRA configuration that SGLang's manager reads."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", strict=True)

    r: Annotated[int, Field(gt=0)]
    lora_alpha: int | float
    target_modules: str | list[str]


@dataclass(frozen=True, slots=True)
class Peft:
    """A PEFT LoRA configuration, its rank, SGLang's normalized names of the modules it adapts, and
    how many of its added tokens lie past the base vocabulary."""

    config: dict[str, JsonValue]
    rank: int
    modules: set[str]
    added: int

    @classmethod
    def parse(cls, config: object, tokens: object, vocabulary: int) -> Peft:
        """The configuration `config` with the added tokens `tokens`, of a model whose token IDs
        lie below `vocabulary`; SGLang counts only tokens past it as added."""
        parsed = valid("the configuration", lambda: CONFIG.validate_python(config))
        fields = valid("the configuration's", lambda: Fields.model_validate(parsed))
        added = (
            {} if tokens is None else valid("added token", lambda: TOKENS.validate_python(tokens))
        )
        modules = get_normalized_target_modules(fields.target_modules)
        return cls(parsed, fields.r, modules, sum(token >= vocabulary for token in added.values()))

    @classmethod
    def read(cls, directory: Path, vocabulary: int) -> Peft:
        """The configuration of the adapter in `directory`, with its `added_tokens.json` if any."""
        path = directory / "adapter_config.json"
        config = valid(path.name, lambda: JSON.validate_json(path.read_bytes()))
        tokens = directory / "added_tokens.json"
        added = (
            valid(tokens.name, lambda: JSON.validate_json(tokens.read_bytes()))
            if tokens.is_file()
            else None
        )
        return cls.parse(config, added, vocabulary)


def valid(what: str, parse: Callable[[], T]) -> T:
    """The value `parse` returns, or a `ValueError` naming the first malformed field of `what`."""
    try:
        return parse()
    except ValidationError as error:
        first = error.errors()[0]
        where = " ".join([what, *(str(part) for part in first["loc"])])
        raise ValueError(f"{where}: {first['msg']}") from None


def factors(tensors: object) -> dict[str, torch.Tensor]:
    """`tensors` as host copies the engine reads: dense, contiguous, sharing no storage."""
    if not isinstance(tensors, dict):
        raise ValueError("the factors are no mapping of names to tensors")
    copies: dict[str, torch.Tensor] = {}
    for name, tensor in cast("dict[object, object]", tensors).items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name!r} names no tensor")
        if tensor.device.type != "cpu" or tensor.layout != torch.strided:
            raise ValueError(f"{name} is no dense host tensor")
        if tensor.is_quantized or tensor.is_nested:
            raise ValueError(f"{name} is a quantized or nested tensor")
        copies[name] = tensor.detach().clone(memory_format=torch.contiguous_format)
    return copies
