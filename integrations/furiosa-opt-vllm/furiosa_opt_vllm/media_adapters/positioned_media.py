"""Translate position-based media features into engine inputs for vLLM."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import torch

import furiosa_opt_engine as engine

if TYPE_CHECKING:
    from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalKwargsItem


class PositionedMediaAdapter:
    """Convert position-based processor features into engine media inputs."""

    architectures = ("Gemma4UnifiedForConditionalGeneration",)

    def prepare_vision(
        self,
        feature: MultiModalFeatureSpec,
        device: engine.Device,
    ) -> engine.encoder.Image:
        data = _data(feature)
        match feature.modality:
            case "image":
                return _image(
                    device,
                    cast(torch.Tensor, data["pixel_values"]),
                    cast(torch.Tensor, data["pixel_position_ids"]),
                )
            case "video":
                return _image(
                    device,
                    cast(torch.Tensor, data["pixel_values_videos"]).flatten(0, -2),
                    cast(torch.Tensor, data["pixel_position_ids_videos"]).flatten(0, -2),
                )
            case modality:
                raise NotImplementedError(f"the vision adapter embeds no {modality}")

    def prepare_audio(
        self,
        feature: MultiModalFeatureSpec,
        device: engine.Device,
    ) -> engine.encoder.Audio:
        data = _data(feature)
        frames = cast(torch.Tensor, data["input_features_padded"])
        mask = cast(torch.Tensor, data["input_features_mask"])
        rows = frames[mask.to(torch.bool)].to(torch.bfloat16)
        return engine.encoder.audio_from_dlpack(rows.detach(), device)


def _image(
    device: engine.Device,
    patches: torch.Tensor,
    positions: torch.Tensor,
) -> engine.encoder.Image:
    valid = ~(positions == -1).all(dim=-1)
    rows = patches[valid].to(torch.bfloat16)
    xy = positions[valid].tolist()
    return engine.encoder.image_from_dlpack(rows.detach(), xy, device)


def _data(feature: MultiModalFeatureSpec) -> Mapping[str, object]:
    return cast("MultiModalKwargsItem", feature.data).get_data()
