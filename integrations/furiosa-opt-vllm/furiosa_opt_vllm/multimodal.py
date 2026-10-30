"""A vision-language model's media encoder alone, as vLLM's encoder-only instance runs it: each step
embeds its requests' scheduled images, videos and audio on the device and hands the rows to the
encoder cache connector, which another instance's language model reads.

- vLLM schedules the items to embed and frees a hash once no request reads it; an item's rows stay
  on the device and on the host, where the connector saves them from, until then.
- An image's or a video's patches of a position, the processor's padding left out, embed as one
  image of the engine; a clip's valid frames as one clip.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import torch
from vllm.v1.outputs import make_empty_encoder_model_runner_output
from vllm.v1.worker.ec_connector_model_runner_mixin import ECConnectorModelRunnerMixin

import furiosa_opt_engine as engine

if TYPE_CHECKING:
    from vllm.multimodal.inputs import MultiModalFeatureSpec
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.worker.gpu_input_batch import CachedRequestState


@dataclass(frozen=True, slots=True, eq=False)
class Media:
    """A loaded media encoder, its device, and the rows of every hash vLLM has not freed."""

    encoder: engine.Encoder
    device: engine.Device
    features: dict[str, engine.HbmTensor] = field(default_factory=dict)
    # Host rows by hash: what the encoder cache connector saves from.
    transfers: dict[str, torch.Tensor] = field(default_factory=dict)
    # An encoder-only instance holds no cache and allocates as it runs.
    workspace: ClassVar[int] = 0


def step(
    model: Media, scheduler: SchedulerOutput, requests: dict[str, CachedRequestState]
) -> ModelRunnerOutput:
    """An encoder-only step: the scheduled items embedded and each saved through the connector,
    then vLLM's output of a step that samples nothing, which finishes a request once embedded."""
    for identifier in scheduler.free_encoder_mm_hashes:
        _ = model.features.pop(identifier, None)
        _ = model.transfers.pop(identifier, None)
    with ECConnectorModelRunnerMixin.maybe_get_ec_connector_output(
        scheduler, encoder_cache=model.transfers
    ):
        for req_id, indices in scheduler.scheduled_encoder_inputs.items():
            for index in indices:
                feature = requests[req_id].mm_features[index]
                if feature.identifier in model.features:
                    continue
                rows = embed(model, feature)
                model.features[feature.identifier] = rows
                model.transfers[feature.identifier] = torch.as_tensor(rows.to_host().numpy()).to(
                    torch.bfloat16
                )
                ECConnectorModelRunnerMixin.maybe_save_ec_to_connector(
                    model.transfers, feature.identifier
                )
        return make_empty_encoder_model_runner_output(scheduler)


def embed(model: Media, feature: MultiModalFeatureSpec) -> engine.HbmTensor:
    """An item's rows from its processed inputs, one for each of its placeholder positions."""
    if feature.data is None:
        raise ValueError(f"item {feature.identifier} has no inputs to embed")
    data = feature.data.get_data()

    def tensor(key: str) -> torch.Tensor:
        value = data[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"the processor's {key} is no tensor")
        return value

    match feature.modality:
        case "image":
            patches = image(model, tensor("pixel_values"), tensor("pixel_position_ids"))
            (rows,) = model.encoder.vision([patches])
        case "video":
            frames = image(
                model,
                tensor("pixel_values_videos").flatten(0, -2),
                tensor("pixel_position_ids_videos").flatten(0, -2),
            )
            (rows,) = model.encoder.vision([frames])
        case "audio":
            frames, mask = tensor("input_features_padded"), tensor("input_features_mask")
            valid = int(mask.sum())
            if not bool(mask[:valid].all()):
                raise ValueError("a clip's valid frames precede its padding")
            clip = upload(model, frames[:valid].to(torch.bfloat16))
            (rows,) = model.encoder.audio([engine.vision_language.Audio(clip)])
        case modality:
            raise NotImplementedError(f"the media encoder embeds no {modality}")
    if (count := rows.shape[0]) != (positions := feature.mm_position.get_num_embeds()):
        raise ValueError(f"{feature.modality} has {count} rows for {positions} placeholders")
    return rows


def image(
    model: Media, pixels: torch.Tensor, positions: torch.Tensor
) -> engine.vision_language.Image:
    """The patches of a position, the processor's padding (both coordinates -1) left out."""
    valid = ~(positions == -1).all(dim=-1)
    patches = upload(model, pixels[valid].to(torch.bfloat16))
    return engine.vision_language.Image(patches, positions[valid].tolist())


def upload(model: Media, rows: torch.Tensor) -> engine.HbmTensor:
    return engine.HostTensor.from_dlpack(rows.detach().contiguous()).to_hbm(model.device)
