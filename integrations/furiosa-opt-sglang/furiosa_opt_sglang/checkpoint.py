"""Resolve a checkpoint and bind the text model it serves to SGLang's pooling."""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

from pydantic import JsonValue, TypeAdapter

if TYPE_CHECKING:
    import furiosa_opt_engine as engine
    from furiosa_opt_sglang.decoder import Decoder
    from furiosa_opt_sglang.generation.model import Model


def configured(
    directory: str, revision: str | None, overrides: Mapping[str, JsonValue]
) -> engine.Checkpoint:
    """The checkpoint at `directory`, local or on the Hub, as its config alone with SGLang's model
    overrides: no weight is mapped or downloaded."""
    import huggingface_hub

    import furiosa_opt_engine as engine

    path = Path(directory) / "config.json"
    if not path.parent.is_dir():
        path = Path(huggingface_hub.hf_hub_download(directory, "config.json", revision=revision))
    config = TypeAdapter(dict[str, JsonValue]).validate_json(path.read_bytes())
    with tempfile.TemporaryDirectory() as empty:
        return engine.Checkpoint(empty, config=config | dict(overrides))


def checkout(directory: str, revision: str | None) -> str:
    """A local checkpoint directory, downloading safetensors, else binary weights, when remote."""
    import huggingface_hub

    if Path(directory).is_dir():
        return directory
    # Hub overloads include untyped progress and user-agent parameters.
    download = cast("Callable[..., str]", huggingface_hub.snapshot_download)

    def fetch(suffix: str) -> str:
        return download(
            directory, revision=revision, allow_patterns=["config.json", f"*.{suffix}*", "*.pt"]
        )

    local = fetch("safetensors")
    return local if any(Path(local).glob("*.safetensors")) else fetch("bin")


def text(core: Decoder) -> Model:
    """A text model over `core`, pooling the last row of each sequence."""
    from sglang.srt.layers.pooler import Pooler, PoolingType

    from furiosa_opt_sglang.generation.model import Model

    return Model(core, Pooler(PoolingType.LAST, normalize=True))
