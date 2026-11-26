"""Gemma 4's host processor outputs normalized into SGLang encoder items."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
import torch
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.encoder_preprocessing import get_encoder_preprocessed_items

from furiosa_opt_sglang.encoder import update_media_token_counts
from furiosa_opt_sglang.media_adapter import PreprocessResult
from furiosa_opt_sglang.media_adapters.positioned_media import _audio_rows, _vision_rows


class Processor:
    model_type = "gemma4_unified"

    def __init__(self, *, image: object, video: object, audio: object) -> None:
        self.outputs = {
            Modality.IMAGE: image,
            Modality.VIDEO: video,
            Modality.AUDIO: audio,
        }
        self.video_processor = self._process_video
        self.preproc_executor = None

    async def _process_image_items(
        self, _items: object, _model_preprocessor: object
    ) -> dict[str, object]:
        return self.outputs[Modality.IMAGE]

    async def _flatten_and_load_data_by_modality(
        self, _items: object, _modality: Modality
    ) -> list[object]:
        return []

    def _process_video(self, *, videos: object, do_sample_frames: bool) -> dict[str, object]:
        assert videos == []
        assert not do_sample_frames
        return self.outputs[Modality.VIDEO]

    async def _process_audio_items(
        self, _items: object, _model_preprocessor: object
    ) -> dict[str, object]:
        return self.outputs[Modality.AUDIO]


@pytest.fixture
def processor() -> Processor:
    return Processor(
        image={
            "pixel_values": torch.arange(24).reshape(2, 4, 3),
            "image_position_ids": torch.tensor(
                [[[0, 0], [0, 1], [-1, -1], [-1, -1]], [[1, 0], [1, 1], [1, 2], [-1, -1]]]
            ),
        },
        video={
            "pixel_values_videos": torch.arange(24).reshape(2, 2, 2, 3),
            "video_position_ids": torch.tensor(
                [
                    [[[0, 0], [-1, -1]], [[1, 0], [1, 1]]],
                    [[[0, 0], [0, 1]], [[1, 0], [1, 1]]],
                ]
            ),
        },
        audio={
            "input_features": torch.arange(24).reshape(2, 4, 3),
            "feature_attention_mask": torch.tensor(
                [[True, True, True, False], [True, True, False, False]]
            ),
            "audio_feature_lens_raw": torch.tensor([3, 2]),
        },
    )


def preprocess(processor: Processor, modality: Modality) -> object:
    async def unused_original(*_args: object) -> PreprocessResult:
        raise AssertionError("Gemma 4 uses its registered host preprocessor")

    return asyncio.run(update_media_token_counts(unused_original, processor, [], modality))


def test_image_processor_batches_become_one_rowwise_item_per_image(processor: Processor) -> None:
    """Gemma 4 emits [images, patches, width] tensors without image_grid_thw."""
    result = preprocess(processor, Modality.IMAGE)

    items = get_encoder_preprocessed_items(result.mm_inputs)
    assert items is not None
    assert [tuple(item.feature.shape) for item in items] == [(4, 3), (4, 3)]
    assert [tuple(item.image_position_ids.shape) for item in items] == [(4, 2), (4, 2)]
    assert result.grid_thw.tolist() == [[1, 1, 4], [1, 1, 4]]
    assert result.token_counts == [2, 3]
    rows, positions = _vision_rows(items[0].feature, items[0].image_position_ids, width=3)
    assert rows.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert positions.tolist() == [[0, 0], [0, 1]]


def test_video_processor_batch_preserves_each_video_and_frame(processor: Processor) -> None:
    """Gemma 4 emits [videos, frames, patches, width] with matching position IDs."""
    result = preprocess(processor, Modality.VIDEO)

    items = get_encoder_preprocessed_items(result.mm_inputs)
    assert items is not None
    assert [tuple(item.feature.shape) for item in items] == [(4, 3), (4, 3)]
    assert [tuple(item.video_position_ids.shape) for item in items] == [(4, 2), (4, 2)]
    assert items[0].feature.tolist() == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
    ]
    assert items[1].feature.tolist() == [
        [12, 13, 14],
        [15, 16, 17],
        [18, 19, 20],
        [21, 22, 23],
    ]
    assert items[0].video_position_ids.tolist() == [
        [0, 0],
        [-1, -1],
        [1, 0],
        [1, 1],
    ]
    assert items[1].video_position_ids.tolist() == [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]
    rows, positions = _vision_rows(items[0].feature, items[0].video_position_ids, width=3)
    assert rows.tolist() == [[0, 1, 2], [6, 7, 8], [9, 10, 11]]
    assert positions.tolist() == [[0, 0], [1, 0], [1, 1]]
    assert result.grid_thw.tolist() == [[2, 1, 2], [2, 1, 2]]
    assert result.token_counts == [3, 4]


def test_audio_masks_trim_token_counts(processor: Processor) -> None:
    """Gemma 4 audio uses batched features, a valid-frame mask, and raw lengths."""
    result = preprocess(processor, Modality.AUDIO)

    items = get_encoder_preprocessed_items(result.mm_inputs)
    assert items is not None
    assert [tuple(item.feature.shape) for item in items] == [(4, 3), (4, 3)]
    assert [item.audio_feature_lens_raw.tolist() for item in items] == [[3], [2]]
    assert result.token_counts == [3, 2]
    assert _audio_rows(items[0].feature, items[0].feature_attention_mask, width=3).tolist() == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ]
    assert _audio_rows(items[1].feature, items[1].feature_attention_mask, width=3).tolist() == [
        [12, 13, 14],
        [15, 16, 17],
    ]


def test_video_rejects_positions_that_do_not_match_patches(processor: Processor) -> None:
    processor.outputs[Modality.VIDEO]["video_position_ids"] = torch.zeros((2, 2, 3, 2))
    with pytest.raises(ValueError, match="video processor"):
        preprocess(processor, Modality.VIDEO)


def test_video_decoder_sampling_matches_sglang_gemma_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SGLang video decoders use Gemma's configured uniform frame sampling."""
    from sglang.srt.utils import video_decoder

    from furiosa_opt_sglang.media_adapters.positioned_media import PositionedMediaAdapter

    class VideoDecoder:
        def __init__(self) -> None:
            self.indices: list[int] = []
            self.closed = False

        def __len__(self) -> int:
            return 5

        def get_frames_at(self, indices: list[int]) -> np.ndarray:
            self.indices = indices
            return np.asarray(indices, dtype=np.uint8).reshape(-1, 1, 1, 1).repeat(3, axis=-1)

        def close(self) -> None:
            self.closed = True

    class VideoProcessor:
        num_frames = 2

        def __call__(
            self, *, videos: list[torch.Tensor], do_sample_frames: bool
        ) -> dict[str, object]:
            assert not do_sample_frames
            frames = videos[0][:, 0, 0, 0].to(torch.float32)
            pixels = frames.reshape(1, -1, 1, 1).expand(1, -1, 2, 3)
            positions = torch.tensor([[[[0, 0], [0, 1]], [[1, 0], [1, 1]]]])
            return {
                "pixel_values_videos": pixels,
                "video_position_ids": positions,
            }

    decoder = VideoDecoder()
    monkeypatch.setattr(video_decoder, "VideoDecoderWrapper", VideoDecoder)

    class Preprocessor:
        model_type = "gemma4_unified"
        video_processor = VideoProcessor()
        preproc_executor = None

        async def _flatten_and_load_data_by_modality(
            self, _items: object, _modality: Modality
        ) -> list[object]:
            return [decoder]

    result = asyncio.run(
        PositionedMediaAdapter().preprocess_encoder_items(Preprocessor(), [], Modality.VIDEO)
    )

    assert result is not None
    assert decoder.indices == [0, 2]
    assert decoder.closed
    assert result.mm_inputs["pixel_values_videos"][:, 0].tolist() == [0, 0, 2, 2]


@pytest.mark.parametrize(
    ("modality", "position_key", "position_shape"),
    [
        (Modality.IMAGE, "image_position_ids", (2, 4, 4)),
        (Modality.VIDEO, "video_position_ids", (3, 2, 4)),
    ],
)
def test_visual_processor_rejects_positions_without_xy_pairs(
    processor: Processor, modality: Modality, position_key: str, position_shape: tuple[int, ...]
) -> None:
    processor.outputs[modality][position_key] = torch.zeros(position_shape)
    with pytest.raises(ValueError, match=r"patches, 2\]"):
        preprocess(processor, modality)
