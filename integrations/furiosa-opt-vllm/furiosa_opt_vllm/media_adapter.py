"""Select model processor adapters at the vLLM boundary."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

import furiosa_opt_engine as engine
from furiosa_opt_vllm.media_adapters.positioned_media import PositionedMediaAdapter

if TYPE_CHECKING:
    from vllm.multimodal.inputs import MultiModalFeatureSpec


class Adapter(Protocol):
    """A model's processor contract at the vLLM boundary."""

    architectures: ClassVar[tuple[str, ...]]


@runtime_checkable
class VisionAdapter(Protocol):
    def prepare_vision(
        self,
        feature: MultiModalFeatureSpec,
        device: engine.Device,
    ) -> engine.encoder.Image: ...


@runtime_checkable
class AudioAdapter(Protocol):
    def prepare_audio(
        self,
        feature: MultiModalFeatureSpec,
        device: engine.Device,
    ) -> engine.encoder.Audio: ...


_ADAPTERS: dict[str, type[Adapter]] = {
    architecture: PositionedMediaAdapter for architecture in PositionedMediaAdapter.architectures
}


def adapter_for(architectures: Sequence[str]) -> Adapter:
    """Select the processor adapter declared for one of the checkpoint's architectures."""
    adapter_type = next(
        (_ADAPTERS[architecture] for architecture in architectures if architecture in _ADAPTERS),
        None,
    )
    if adapter_type is None:
        raise NotImplementedError(
            f"no vLLM media input adapter is registered for architectures {tuple(architectures)}"
        )
    return adapter_type()
