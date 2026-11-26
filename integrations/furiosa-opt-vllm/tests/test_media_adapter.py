"""Framework media fields mapped to engine inputs without creating an NPU device."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_video_adapter_uses_processor_position_field(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    import furiosa_opt_engine as engine
    from furiosa_opt_vllm.media_adapters.positioned_media import PositionedMediaAdapter

    data = {
        "pixel_values_videos": torch.tensor(
            [[[[1.0, 10.0], [2.0, 20.0]], [[3.0, 30.0], [4.0, 40.0]]]]
        ),
        "pixel_position_ids_videos": torch.tensor([[[[0, 0], [0, 1]], [[1, 0], [-1, -1]]]]),
    }
    feature = SimpleNamespace(
        identifier="video",
        modality="video",
        data=SimpleNamespace(get_data=lambda: data),
    )
    captured: dict[str, object] = {}
    device = object()

    def image_from_dlpack(rows: torch.Tensor, positions: list[list[int]], actual_device: object):
        captured["rows"] = rows
        captured["positions"] = positions
        captured["device"] = actual_device
        return object()

    monkeypatch.setattr(engine.encoder, "image_from_dlpack", image_from_dlpack)

    PositionedMediaAdapter().prepare_vision(feature, device)

    rows = captured["rows"]
    assert isinstance(rows, torch.Tensor)
    assert rows.dtype == torch.bfloat16
    assert rows.tolist() == [[1, 10], [2, 20], [3, 30]]
    assert captured["positions"] == [[0, 0], [0, 1], [1, 0]]
    assert captured["device"] is device


def test_audio_adapter_filters_padded_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    import furiosa_opt_engine as engine
    from furiosa_opt_vllm.media_adapters.positioned_media import PositionedMediaAdapter

    data = {
        "input_features_padded": torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]]),
        "input_features_mask": torch.tensor([[True, False, True]]),
    }
    feature = SimpleNamespace(
        identifier="audio",
        modality="audio",
        data=SimpleNamespace(get_data=lambda: data),
    )
    captured: dict[str, object] = {}
    device = object()

    def audio_from_dlpack(rows: torch.Tensor, actual_device: object):
        captured["rows"] = rows
        captured["device"] = actual_device
        return object()

    monkeypatch.setattr(engine.encoder, "audio_from_dlpack", audio_from_dlpack)

    PositionedMediaAdapter().prepare_audio(feature, device)

    rows = captured["rows"]
    assert isinstance(rows, torch.Tensor)
    assert rows.dtype == torch.bfloat16
    assert rows.tolist() == [[1, 10], [3, 30]]
    assert captured["device"] is device
