"""Resolve a framework model's checkpoint and map it for the engine."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import huggingface_hub

import furiosa_opt_engine as engine

if TYPE_CHECKING:
    from vllm.config import ModelConfig


def checkout(model: str, revision: str | None, download_dir: str | None) -> str:
    """The local directory of checkpoint `model`: its path, or its config and safetensors, else
    binary weights, downloaded from the Hub into `download_dir`."""
    if Path(model).is_dir():
        return model

    def fetch(suffix: str) -> str:
        return huggingface_hub.snapshot_download(
            model,
            revision=revision,
            cache_dir=download_dir,
            allow_patterns=["config.json", f"*.{suffix}*", "*.pt"],
        )

    directory = fetch("safetensors")
    return directory if any(Path(directory).glob("*.safetensors")) else fetch("bin")


@dataclass(frozen=True, slots=True)
class Location:
    """A checkpoint's local directory and the config it loads under, which every map of it
    keeps, as vLLM keeps the config it resolved at startup."""

    directory: str
    config: dict[str, object]

    def mapped(self) -> engine.Checkpoint:
        """The checkpoint's tensors, mapped from its files as they are now."""
        return engine.Checkpoint(self.directory, config=self.config)


def configured(model: ModelConfig, download_dir: str | None) -> engine.Checkpoint:
    """The checkpoint of `model` as its config alone, from its directory or the Hub, with the
    user's Hugging Face overrides: no weight is mapped or downloaded."""
    path = Path(model.model) / "config.json"
    if not path.parent.is_dir():
        path = Path(
            huggingface_hub.hf_hub_download(
                model.model, "config.json", revision=model.revision, cache_dir=download_dir
            )
        )
    config = json.loads(path.read_bytes()) | overrides(model)
    with tempfile.TemporaryDirectory() as empty:
        return engine.Checkpoint(empty, config=config)


def located(model: ModelConfig, download_dir: str | None) -> Location:
    """Where the checkpoint of `model` is, in its directory or downloaded into `download_dir`,
    under its own config with the user's Hugging Face overrides."""
    directory = checkout(model.model, model.revision, download_dir)
    config = json.loads((Path(directory) / "config.json").read_bytes())
    return Location(directory, config | overrides(model))


def overrides(model: ModelConfig) -> dict[str, object]:
    """The user's Hugging Face overrides of `model`'s checkpoint config."""
    # The engine reads the checkpoint's format; vLLM's resolved config is Transformers' form of it,
    # with fields renamed and set for vLLM's loader.
    # A callable override rewrites a config for vLLM's own model classes, such as a draft's MTP.
    return {} if callable(model.hf_overrides) else model.hf_overrides
