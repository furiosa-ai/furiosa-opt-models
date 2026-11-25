"""Translate position-based media features into SGLang EPD and engine inputs."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import torch

import furiosa_opt_engine as engine

POSITION_COMPONENTS = 2
DEFAULT_VIDEO_FRAME_COUNT = 32

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
    from sglang.srt.utils.video_decoder import VideoDecoderWrapper

    from furiosa_opt_sglang.media_adapter import EncoderPreprocessor, PreprocessResult


@dataclass(slots=True)
class EncoderResult:
    """Fields SGLang's encoder server reads from an encoder preprocessing result."""

    mm_inputs: Mapping[str, object]
    grid_thw: torch.Tensor
    token_counts: list[int]


class PositionedMediaAdapter:
    """Adapt position-based processor features to SGLang EPD and engine inputs."""

    architectures = ("Gemma4UnifiedForConditionalGeneration",)
    model_types = ("gemma4_unified",)

    async def preprocess_encoder_items(
        self, preprocessor: EncoderPreprocessor, items: Sequence[object], modality: Modality
    ) -> EncoderResult | None:
        """Adapt position-based processor output to SGLang's grid-based EPD API."""
        from sglang.srt.managers.schedule_batch import Modality as SGLangModality

        # SGLang's generic process_mm_items requires conventional grids and derives token counts
        # from generic model formulas. This processor needs per-item positions and counts of valid
        # engine rows; EPD still validates item/grid cardinality and final output row counts.
        match modality:
            case SGLangModality.IMAGE:
                inputs = await preprocessor._process_image_items(items, None)
                return _visual_result(
                    inputs,
                    modality,
                    feature_key="pixel_values",
                    position_key="image_position_ids",
                    grid_key="image_grid_thw",
                )
            case SGLangModality.VIDEO:
                inputs = await _process_video_items(preprocessor, items)
                return _video_result(inputs, modality)
            case SGLangModality.AUDIO:
                inputs = await preprocessor._process_audio_items(items, None)
                return _audio_result(inputs, modality)
            case _:
                return None

    def prepare_vision(
        self,
        item: MultimodalDataItem,
        modality: Literal["image", "video"],
        input_width: int,
        device: engine.Device,
    ) -> engine.encoder.Image:
        position_name = f"{modality}_position_ids"
        # EncoderPreprocessOutput preserves items normalized by _visual_result.
        rows, xy = _vision_rows(
            item.feature,
            getattr(item, position_name),
            input_width,
        )
        return _engine_image(rows, xy, device)

    def prepare_audio(
        self,
        item: MultimodalDataItem,
        input_width: int,
        device: engine.Device,
    ) -> engine.encoder.Audio:
        # EncoderPreprocessOutput preserves feature/mask alignment checked by _audio_result.
        rows = _audio_rows(item.feature, item.feature_attention_mask, input_width)
        return _engine_audio(rows, device)

    def update_token_counts(self, result: PreprocessResult, modality: Modality) -> None:
        from sglang.srt.managers.schedule_batch import Modality as SGLangModality

        if modality != SGLangModality.AUDIO:
            return
        mm_inputs = result.mm_inputs
        mask = mm_inputs.get("feature_attention_mask")
        if mask is None:
            raise ValueError("audio preprocessing has no feature_attention_mask")
        mask = torch.as_tensor(mask)
        if mask.ndim == 1:
            counts = [int(mask.to(torch.bool).sum())]
        else:
            counts = [int(row.to(torch.bool).sum()) for row in mask.reshape(mask.shape[0], -1)]
        # MMEncoder checks token_counts length against grid_thw before encoding.
        result.token_counts = counts


def _visual_result(
    inputs: Mapping[str, object],
    modality: Modality,
    *,
    feature_key: str,
    position_key: str,
    grid_key: str,
) -> EncoderResult:
    from sglang.srt.managers.schedule_batch import MultimodalDataItem

    features = _processor_tensor(inputs, feature_key)
    positions = _processor_tensor(inputs, position_key)
    # EPD checks grid/token counts, but cannot verify processor rows align with position IDs.
    if (
        features.ndim != 3
        or positions.ndim != 3
        or positions.shape[-1] != POSITION_COMPONENTS
        or features.shape[:2] != positions.shape[:2]
    ):
        raise ValueError(
            f"positioned {modality.name.lower()} processor must return matching "
            f"[items, patches, width] and [items, patches, {POSITION_COMPONENTS}] tensors; got "
            f"{tuple(features.shape)} and {tuple(positions.shape)}"
        )

    items = []
    grids = []
    token_counts = []
    patch_count = features.shape[1]
    # The shape check above establishes equal image and position batch counts.
    # EPD uses grid products to slice rows; this processor exposes them only via position IDs.
    grid = torch.tensor([[1, 1, patch_count]], dtype=torch.long)
    for rows, xy in zip(features, positions):  # noqa: B905
        rows = rows.reshape(-1, features.shape[-1]).contiguous()
        xy = xy.reshape(-1, POSITION_COMPONENTS).contiguous()
        grids.append(grid)
        token_counts.append(int((~_padding_positions(xy)).sum()))
        items.append(
            MultimodalDataItem(
                modality=modality,
                feature=rows,
                model_specific_data={position_key: xy, grid_key: grid},
            )
        )
    return _preprocess_result(inputs, items, grids, token_counts, feature_key, grid_key)


def _video_result(inputs: Mapping[str, object], modality: Modality) -> EncoderResult:
    """Use Gemma's processor video axis to restore one EPD item per video."""
    from sglang.srt.managers.schedule_batch import MultimodalDataItem

    features = _processor_tensor(inputs, "pixel_values_videos")
    positions = _processor_tensor(inputs, "video_position_ids")
    # EPD checks grid/token counts, but cannot verify processor frames align with position IDs.
    if (
        features.ndim != 4
        or positions.ndim != 4
        or positions.shape[-1] != POSITION_COMPONENTS
        or features.shape[:3] != positions.shape[:3]
    ):
        raise ValueError(
            "video processor must return matching "
            f"[videos, frames, patches, width] and "
            f"[videos, frames, patches, {POSITION_COMPONENTS}] tensors; got "
            f"{tuple(features.shape)} and {tuple(positions.shape)}"
        )

    items = []
    grids = []
    token_counts = []
    frame_count = features.shape[1]
    patch_count = features.shape[2]
    for video, video_positions in zip(features, positions):  # noqa: B905
        rows = video.reshape(-1, features.shape[-1]).contiguous()
        xy = video_positions.reshape(-1, POSITION_COMPONENTS).contiguous()
        grid = torch.tensor([[frame_count, 1, patch_count]], dtype=torch.long)
        grids.append(grid)
        token_counts.append(int((~_padding_positions(xy)).sum()))
        items.append(
            MultimodalDataItem(
                modality=modality,
                feature=rows,
                model_specific_data={"video_position_ids": xy, "video_grid_thw": grid},
            )
        )
    return _preprocess_result(
        inputs,
        items,
        grids,
        token_counts,
        "pixel_values_videos",
        "video_grid_thw",
    )


def _audio_result(inputs: Mapping[str, object], modality: Modality) -> EncoderResult:
    from sglang.srt.managers.schedule_batch import MultimodalDataItem

    features = _processor_tensor(inputs, "input_features")
    masks = _processor_tensor(inputs, "feature_attention_mask")
    # SGLang 0.5.22 derives audio_feature_lens_raw from feature_attention_mask.sum(-1); use that
    # same mask as the row-count source instead of rechecking the copied length metadata.
    if features.ndim != 3 or masks.ndim != 2 or features.shape[:2] != masks.shape:
        raise ValueError(
            "audio processor must return matching [clips, frames, width] "
            f"and [clips, frames] tensors; got {tuple(features.shape)} and {tuple(masks.shape)}"
        )

    items = []
    grids = []
    token_counts = []
    for rows, mask in zip(features, masks):  # noqa: B905
        valid = mask.to(torch.bool)
        count = int(valid.sum())
        grid = torch.tensor([count], dtype=torch.long)
        grids.append(grid)
        items.append(
            MultimodalDataItem(
                modality=modality,
                feature=rows.contiguous(),
                model_specific_data={
                    "audio_feature_lens_raw": grid,
                    "feature_attention_mask": valid,
                },
            )
        )
        token_counts.append(count)
    return _preprocess_result(
        inputs,
        items,
        grids,
        token_counts,
        "input_features",
        "audio_feature_lens_raw",
    )


def _preprocess_result(
    inputs: Mapping[str, object],
    items: list[MultimodalDataItem],
    item_grids: list[torch.Tensor],
    token_counts: list[int],
    feature_key: str,
    grid_key: str,
) -> EncoderResult:
    from sglang.srt.multimodal.encoder_preprocessing import EncoderPreprocessOutput

    # Let SGLang enforce its non-empty item contract before concatenating item tensors.
    output = EncoderPreprocessOutput(inputs, mm_items=items)
    output[feature_key] = torch.cat([item.feature for item in items], dim=0)
    grids = torch.cat(item_grids, dim=0)
    output[grid_key] = grids
    return EncoderResult(mm_inputs=output, grid_thw=grids, token_counts=token_counts)


def _processor_tensor(inputs: Mapping[str, object], key: str) -> torch.Tensor:
    return torch.as_tensor(cast(torch.Tensor, inputs[key])).detach().cpu()


async def _process_video_items(
    preprocessor: EncoderPreprocessor, items: Sequence[object]
) -> Mapping[str, object]:
    """Use Gemma's SGLang frame sampling before its Hugging Face video processor."""
    from sglang.srt.managers.schedule_batch import Modality as SGLangModality
    from sglang.srt.utils.video_decoder import VideoDecoderWrapper

    video_processor = preprocessor.video_processor
    if video_processor is None:
        raise ValueError("No video processor available")

    loaded = await preprocessor._flatten_and_load_data_by_modality(items, SGLangModality.VIDEO)
    try:
        videos = [
            _video_decoder_to_tensor(video, video_processor)
            if isinstance(video, VideoDecoderWrapper)
            else video
            for video in loaded
        ]
        result = await asyncio.get_running_loop().run_in_executor(
            preprocessor.preproc_executor,
            functools.partial(
                video_processor,
                videos=videos,
                do_sample_frames=False,
            ),
        )
    except BaseException as error:
        if close_errors := _close_video_decoders(loaded):
            raise error from close_errors
        raise
    if close_errors := _close_video_decoders(loaded):
        raise close_errors
    return result


def _close_video_decoders(videos: Sequence[object]) -> ExceptionGroup[Exception] | None:
    """Close every loaded video, collecting failures so cleanup never masks earlier work."""
    errors = []
    for video in videos:
        try:
            close = getattr(video, "close", None)
            if callable(close):
                close()
        except Exception as error:
            errors.append(error)
    if errors:
        return ExceptionGroup("failed to close video decoders", errors)
    return None


def _video_decoder_to_tensor(video: VideoDecoderWrapper, processor: object) -> torch.Tensor:
    """Sample a SGLang video decoder as Gemma4SGLangProcessor does."""
    total = len(video)
    num_frames = getattr(processor, "num_frames", DEFAULT_VIDEO_FRAME_COUNT)
    indices = (
        list(range(total))
        if total <= num_frames
        else torch.arange(0, total, total / num_frames).int().tolist()
    )
    frames = video.get_frames_at(indices)
    return torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()


def _vision_rows(
    features: object,
    positions: object,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = _rows(features, width)
    xy = _rows(positions, POSITION_COMPONENTS)
    padding = _padding_positions(xy)
    return rows[~padding], xy[~padding]


def _padding_positions(positions: torch.Tensor) -> torch.Tensor:
    return (positions == -1).all(dim=-1)


def _audio_rows(features: object, mask: object, width: int) -> torch.Tensor:
    return _rows(features, width)[_rows(mask, 1).reshape(-1).to(torch.bool)]


def _rows(value: object, width: int) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().cpu()
    return tensor.reshape(-1, width)


def _engine_image(
    rows: torch.Tensor,
    positions: torch.Tensor,
    device: engine.Device,
) -> engine.encoder.Image:
    rows = rows.to(dtype=torch.bfloat16)
    return engine.encoder.image_from_dlpack(rows, positions.tolist(), device)


def _engine_audio(rows: torch.Tensor, device: engine.Device) -> engine.encoder.Audio:
    rows = rows.to(dtype=torch.bfloat16)
    return engine.encoder.audio_from_dlpack(rows, device)
