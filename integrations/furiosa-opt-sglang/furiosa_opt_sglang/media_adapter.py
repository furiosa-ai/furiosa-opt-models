"""Select model processor adapters at the SGLang boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Sized
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, runtime_checkable

import furiosa_opt_engine as engine
from furiosa_opt_sglang.media_adapters.positioned_media import PositionedMediaAdapter

if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Executor

    from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem


class PreprocessResult(Protocol):
    """SGLang's fields shared by processor adapters and the encoder server."""

    mm_inputs: Mapping[str, object]
    grid_thw: Sized
    token_counts: list[int]


class EncoderPreprocessor(Protocol):
    """The pinned SGLang preprocessor surface used by media adapters."""

    model_type: str
    video_processor: Callable[..., Mapping[str, object]] | None
    preproc_executor: Executor | None

    async def _process_image_items(
        self, items: Sequence[object], model_preprocessor: object
    ) -> Mapping[str, object]: ...

    async def _process_audio_items(
        self, items: Sequence[object], model_preprocessor: object
    ) -> Mapping[str, object]: ...

    async def _flatten_and_load_data_by_modality(
        self, items: Sequence[object], modality: Modality
    ) -> Sequence[object]: ...


class Adapter(Protocol):
    """Architecture registration for engine-backed SGLang media models."""

    architectures: ClassVar[tuple[str, ...]]


class PreprocessorAdapter(Protocol):
    """Optional model-specific preprocessing and token-count rules."""

    model_types: ClassVar[tuple[str, ...]]

    async def preprocess_encoder_items(
        self, preprocessor: EncoderPreprocessor, items: Sequence[object], modality: Modality
    ) -> PreprocessResult | None: ...

    def update_token_counts(self, result: PreprocessResult, modality: Modality) -> None: ...


@runtime_checkable
class VisionAdapter(Protocol):
    def prepare_vision(
        self,
        item: MultimodalDataItem,
        modality: Literal["image", "video"],
        input_width: int,
        device: engine.Device,
    ) -> engine.encoder.Image: ...


@runtime_checkable
class AudioAdapter(Protocol):
    def prepare_audio(
        self,
        item: MultimodalDataItem,
        input_width: int,
        device: engine.Device,
    ) -> engine.encoder.Audio: ...


_BY_ARCHITECTURE: dict[str, type[Adapter]] = {
    architecture: PositionedMediaAdapter for architecture in PositionedMediaAdapter.architectures
}
_BY_MODEL_TYPE: dict[str, type[PreprocessorAdapter]] = {
    model_type: PositionedMediaAdapter for model_type in PositionedMediaAdapter.model_types
}


def find_adapter(architectures: Sequence[str]) -> Adapter | None:
    """Find the registered adapter matching a checkpoint architecture, if one exists."""
    adapter_type = next(
        (
            _BY_ARCHITECTURE[architecture]
            for architecture in architectures
            if architecture in _BY_ARCHITECTURE
        ),
        None,
    )
    return adapter_type() if adapter_type is not None else None


def require_adapter(architectures: Sequence[str]) -> Adapter:
    """Select a registered processor adapter for an encoder-only checkpoint."""
    adapter = find_adapter(architectures)
    if adapter is None:
        raise NotImplementedError(
            f"no SGLang media input adapter is registered for architectures {tuple(architectures)}"
        )
    return adapter


def adapter_for_model_type(model_type: str) -> PreprocessorAdapter | None:
    """Find an adapter whose preprocessing rules use this Transformers model type."""
    adapter_type = _BY_MODEL_TYPE.get(model_type)
    return adapter_type() if adapter_type is not None else None
