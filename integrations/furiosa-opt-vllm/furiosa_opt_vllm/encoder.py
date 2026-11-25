"""A model's media encoder alone, as vLLM's encoder-only instance runs it: each step
embeds its requests' scheduled images, videos and audio on the device and hands the rows to the
encoder cache connector, which another instance's language model reads.

- vLLM schedules the items to embed and frees a hash once no request reads it; host rows stay here
  for the connector until then, while each device output is released after its transfer.
- An image's or a video's patches of a position, the processor's padding left out, embed as one
  image of the engine; a clip's valid frames as one clip.
- Video input supplies sampled frames; audio is encoded when the framework supplies it as a separate
  audio item. This path does not extract a soundtrack from a video.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import torch
from vllm.v1.outputs import make_empty_encoder_model_runner_output
from vllm.v1.worker.ec_connector_model_runner_mixin import ECConnectorModelRunnerMixin

import furiosa_opt_engine as engine
from furiosa_opt_vllm.media_adapter import Adapter, AudioAdapter, VisionAdapter

if TYPE_CHECKING:
    from vllm.multimodal.inputs import MultiModalFeatureSpec
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.worker.gpu_input_batch import CachedRequestState


@dataclass(frozen=True, slots=True, eq=False)
class Encoder:
    """A loaded media encoder, its processor adapter, and host rows vLLM has not freed."""

    encoder: engine.Encoder
    device: engine.Device
    adapter: Adapter
    # Host rows by hash are both the processed-hash set and the connector's source.
    transfers: dict[str, torch.Tensor] = field(default_factory=dict)
    # An encoder-only instance holds no cache and allocates as it runs.
    workspace: ClassVar[int] = 0


def step(
    model: Encoder, scheduler: SchedulerOutput, requests: dict[str, CachedRequestState]
) -> ModelRunnerOutput:
    """An encoder-only step: the scheduled items embedded and each saved through the connector,
    then vLLM's output of a step that samples nothing, which finishes a request once embedded."""
    for identifier in scheduler.free_encoder_mm_hashes:
        _ = model.transfers.pop(identifier, None)
    with ECConnectorModelRunnerMixin.maybe_get_ec_connector_output(
        scheduler, encoder_cache=model.transfers
    ):
        vision_features: list[MultiModalFeatureSpec] = []
        audio_features: list[MultiModalFeatureSpec] = []
        for req_id, indices in scheduler.scheduled_encoder_inputs.items():
            for index in indices:
                feature = requests[req_id].mm_features[index]
                # vLLM 0.31's _try_schedule_encoder_inputs omits cached and duplicate hashes;
                # _batch_mm_inputs_from_scheduler skips data-less specs, which this path mirrors.
                if feature.data is None:
                    continue
                match feature.modality:
                    case "image" | "video":
                        vision_features.append(feature)
                    case "audio":
                        audio_features.append(feature)
                    case modality:
                        raise NotImplementedError(f"the media encoder embeds no {modality}")

        for features in (vision_features, audio_features):
            if not features:
                continue
            # Encoder returns one output per input item; see engine.encoder.Encoder.
            for feature, rows in zip(features, _embed(model, features)):  # noqa: B905
                host_rows = _host_rows(rows)
                model.transfers[feature.identifier] = host_rows
                ECConnectorModelRunnerMixin.maybe_save_ec_to_connector(
                    model.transfers, feature.identifier
                )
        return make_empty_encoder_model_runner_output(scheduler)


def _embed(model: Encoder, features: Sequence[MultiModalFeatureSpec]) -> list[engine.HbmTensor]:
    """Embed one nonempty, single-modality group from the scheduler."""
    modality = features[0].modality
    match modality:
        case "image" | "video":
            if not isinstance(model.adapter, VisionAdapter):
                raise NotImplementedError("the vLLM adapter cannot prepare vision inputs")
            # The engine call reports if this loaded model has no vision capability.
            # vLLM passes sampled video frames here; soundtrack has its own audio feature.
            items = [model.adapter.prepare_vision(feature, model.device) for feature in features]
            rows = model.encoder.vision(items)
        case "audio":
            if not isinstance(model.adapter, AudioAdapter):
                raise NotImplementedError("the vLLM adapter cannot prepare audio inputs")
            # The engine call reports if this loaded model has no audio capability.
            items = [model.adapter.prepare_audio(feature, model.device) for feature in features]
            rows = model.encoder.audio(items)
        case modality:
            raise NotImplementedError(f"the media encoder embeds no {modality}")
    # This direct path bypasses vLLM's sanity_check_mm_encoder_outputs. The engine guarantees one
    # tensor per item; check vLLM's PlaceholderRange row count here.
    for feature, item_rows in zip(features, rows):  # noqa: B905
        if (count := item_rows.shape[0]) != (positions := feature.mm_position.get_num_embeds()):
            raise ValueError(f"{feature.modality} has {count} rows for {positions} placeholders")
    return rows


def _host_rows(rows: engine.HbmTensor) -> torch.Tensor:
    """Expose BF16 host rows without widening through NumPy or copying back into PyTorch."""
    shape = tuple(rows.shape)
    buffer = rows.buffer
    size = len(buffer)
    if size == 0:
        return torch.empty(shape, dtype=torch.bfloat16)
    host = engine.HostBuffer(size)
    buffer.to_host(host).wait()
    # PyTorch retains this buffer owner while the encoder cache connector consumes the tensor.
    return torch.frombuffer(memoryview(host), dtype=torch.bfloat16).reshape(shape)
