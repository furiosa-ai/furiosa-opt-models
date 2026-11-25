"""Run SGLang's encoder-only media service through a registered engine adapter."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch

from furiosa_opt_sglang.media_adapter import (
    Adapter,
    AudioAdapter,
    EncoderPreprocessor,
    PreprocessResult,
    VisionAdapter,
    adapter_for_model_type,
    find_adapter,
    require_adapter,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem

    import furiosa_opt_engine as engine


class EncoderModel:
    """Generic SGLang EPD wrapper; a processor adapter supplies normalized media items."""

    def __init__(self, encoder: engine.Encoder, device: engine.Device, adapter: Adapter) -> None:
        self.encoder = encoder
        self.device = device
        self.adapter = adapter
        self._parameter = torch.empty((), dtype=torch.bfloat16)

    def parameters(self) -> Iterator[torch.Tensor]:
        # SGLang reads only the dtype here; encoder weights live on the NPU.
        yield self._parameter

    def get_image_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        return self._vision_feature(items, "image")

    def get_video_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        return self._vision_feature(items, "video")

    def get_audio_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        return self._audio_feature(items)

    def _vision_feature(
        self,
        items: list[MultimodalDataItem],
        modality: Literal["image", "video"],
    ) -> torch.Tensor:
        input_width = self.encoder.vision_input_width
        if input_width is None:
            raise NotImplementedError("the loaded encoder has no vision capability")
        if not isinstance(self.adapter, VisionAdapter):
            raise NotImplementedError("the SGLang adapter cannot prepare vision inputs")
        images = [
            self.adapter.prepare_vision(item, modality, input_width, self.device) for item in items
        ]
        rows = self.encoder.vision(images)
        # Engine returns one tensor per item; MMEncoder.slice_embedding checks
        # the flattened row count against preprocessing token_counts.
        return _host_rows(rows, self.encoder.output_width)

    def _audio_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        input_width = self.encoder.audio_input_width
        if input_width is None:
            raise NotImplementedError("the loaded encoder has no audio capability")
        if not isinstance(self.adapter, AudioAdapter):
            raise NotImplementedError("the SGLang adapter cannot prepare audio inputs")
        clips = [self.adapter.prepare_audio(item, input_width, self.device) for item in items]
        rows = self.encoder.audio(clips)
        return _host_rows(rows, self.encoder.output_width)


def load_encoder_model(
    original: Callable[..., object],
    *,
    model_config: ModelConfig,
    load_config: object,
    device_config: object,
) -> object:
    """Replace the framework model on encoder-only workers with a registered media adapter."""
    from sglang.srt.runtime_context import get_disagg

    if not get_disagg().encoder_only:
        return original(
            model_config=model_config,
            load_config=load_config,
            device_config=device_config,
        )

    architectures = getattr(model_config.hf_config, "architectures", ()) or ()
    adapter = require_adapter(architectures)
    return _load_encoder_model(model_config, int(device_config.gpu_id), adapter)


def allow_registered_encoder_disaggregation(
    original: Callable[..., object], server_args: object
) -> object:
    """Admit registered engine media adapters through SGLang's EPD architecture allowlist."""
    from sglang.srt.arg_groups.pd_disaggregation_hook import model_config_of, resolving_view

    config = resolving_view(server_args)
    if not config.encoder_only:
        return original(server_args)
    model_config = model_config_of(server_args)
    architectures = getattr(model_config.hf_config, "architectures", ()) or ()
    adapter = find_adapter(architectures)
    if adapter is None:
        return original(server_args)

    try:
        return original(server_args)
    except ValueError as error:
        # SGLang 0.5.22 exposes its architecture allowlist as this ValueError;
        # no structured allowlist hook exists, so suppress only this exact refusal.
        refused_architectures = (
            f"Model type {architecture} is not supported for encoder disaggregation."
            for architecture in adapter.architectures
        )
        if any(str(error) == message for message in refused_architectures):
            # All other transfer/backend checks ran before SGLang's architecture allowlist.
            return None
        raise


async def update_media_token_counts(
    original: Callable[..., Awaitable[PreprocessResult]],
    preprocessor: EncoderPreprocessor,
    items: Sequence[object],
    modality: Modality,
) -> PreprocessResult:
    """Use model-declared preprocessing when needed, then set exact placeholder counts."""
    adapter = adapter_for_model_type(preprocessor.model_type)
    if adapter is not None:
        result = await adapter.preprocess_encoder_items(preprocessor, items, modality)
        if result is not None:
            return result

    result = await original(preprocessor, items, modality)
    if adapter is not None:
        adapter.update_token_counts(result, modality)
    return result


# The hook delegates normal workers; this helper constructs the engine-backed producer.
def _load_encoder_model(
    model_config: ModelConfig,
    first_chip: int,
    adapter: Adapter,
) -> EncoderModel:
    from sglang.srt.runtime_context import get_model

    import furiosa_opt_engine as engine
    from furiosa_opt_sglang.checkpoint import Location, checkout
    from furiosa_opt_sglang.decoder import PES, chips

    settings = get_model()
    directory = checkout(settings.model_path, settings.revision)
    config_path = Path(directory) / "config.json"
    config = json.loads(config_path.read_bytes())
    config.update(model_config.model_override_args or {})
    checkpoint = Location(directory, config).mapped()
    device = engine.Device(engine.Placement(chips(first_chip, 1), PES))
    encoder = engine.LoadOptions().encoder().load(checkpoint, device)
    return EncoderModel(encoder, device, adapter)


def _host_rows(rows: Sequence[engine.HbmTensor], width: int) -> torch.Tensor:
    import furiosa_opt_engine as engine

    # Engine Encoder guarantees BF16 [rows, output_width] tensors.
    parts = [(row.buffer, len(row.buffer), row.shape[0]) for row in rows]
    row_count = sum(rows for _, _, rows in parts)
    total_size = sum(size for _, size, _ in parts)

    if total_size == 0:
        return torch.empty((row_count, width), dtype=torch.bfloat16)

    host = engine.HostBuffer(total_size)
    transfers = []
    # Retain each destination view until its asynchronous transfer finishes.
    host_slices = []
    offset = 0
    for buffer, size, _ in parts:
        if size:
            target = host.slice(offset, size)
            host_slices.append(target)
            transfers.append(buffer.to_host(target))
        offset += size

    for transfer in transfers:
        transfer.wait()

    # The tensor retains the buffer owner while SGLang consumes its host rows.
    return torch.frombuffer(memoryview(host), dtype=torch.bfloat16).reshape(row_count, width)
