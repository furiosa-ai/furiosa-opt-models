"""Build encoder media inputs from host tensors that export DLPack."""

from __future__ import annotations

from collections.abc import Sequence

from ._native import Device, HbmTensor, HostTensor
from ._native import encoder as _encoder

Audio = _encoder.Audio
Image = _encoder.Image

__all__ = ["Audio", "Image", "audio_from_dlpack", "image_from_dlpack"]


def image_from_dlpack(
    patches: object,
    positions: Sequence[Sequence[int]],
    device: Device,
) -> Image:
    """Upload host BF16 patch-feature rows and pair them with their valid positions.

    `patches` must export host rows through DLPack. Its rows must match `positions`.
    """
    return Image(_upload(patches, device), positions)


def audio_from_dlpack(frames: object, device: Device) -> Audio:
    """Upload host BF16 audio feature rows as one encoder clip.

    `frames` must export host rows through DLPack.
    """
    return Audio(_upload(frames, device))


def _upload(rows: object, device: Device) -> HbmTensor:
    return HostTensor.from_dlpack(rows).to_hbm(device)
