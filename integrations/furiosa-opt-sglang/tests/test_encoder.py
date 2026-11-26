"""SGLang's Gemma 4 encoder adapter submits its real media rows to the engine."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem

from furiosa_opt_sglang.decoder import PES, chips
from furiosa_opt_sglang.encoder import EncoderModel
from furiosa_opt_sglang.media_adapter import require_adapter
from furiosa_opt_sglang.media_adapters.positioned_media import PositionedMediaAdapter

MODEL = "unsloth/gemma-4-12b-it-NVFP4"
REVISION = "b1f649734b34aa5575b03d186abd1b9be3d0d5c4"


def test_gemma4_audio_token_counts_follow_valid_frames() -> None:
    """SGLang's generic convolution estimate must not shorten Gemma 4's raw audio frames."""
    result = SimpleNamespace(
        mm_inputs={
            "feature_attention_mask": torch.tensor([[True, True, False], [True, False, False]])
        },
        grid_thw=torch.tensor([3, 3]),
        token_counts=[1, 1],
    )
    PositionedMediaAdapter().update_token_counts(result, Modality.AUDIO)
    assert result.token_counts == [2, 1]


@pytest.mark.feature("Disaggregated encoder and encoder caches")
def test_gemma4_adapter_runs_sglang_image_video_and_audio_items() -> None:
    """The adapter filters processor padding and returns each live row from actual NPU calls."""
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig

    import furiosa_opt_engine as engine

    directory = snapshot_download(
        MODEL,
        revision=REVISION,
        allow_patterns=["*.json", "*.safetensors"],
    )
    config = AutoConfig.from_pretrained(directory)
    device = engine.Device(engine.Placement(chips(0, 1), PES))
    encoder = engine.EncoderLoadOptions().load(engine.Checkpoint(directory), device)
    assert isinstance(encoder, engine.Encoder)
    adapter = require_adapter(config.architectures)
    model = EncoderModel(encoder, device, adapter)

    image = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=torch.zeros((2, encoder.vision_input_width)),
        model_specific_data={
            "image_grid_thw": torch.tensor([[1, 1, 2]]),
            "image_position_ids": torch.tensor([[0, 0], [-1, -1]]),
        },
    )
    video = MultimodalDataItem(
        modality=Modality.VIDEO,
        feature=torch.zeros((2, encoder.vision_input_width)),
        model_specific_data={
            "video_grid_thw": torch.tensor([[1, 1, 2]]),
            "video_position_ids": torch.tensor([[0, 0], [-1, -1]]),
        },
    )
    audio = MultimodalDataItem(
        modality=Modality.AUDIO,
        feature=torch.zeros((3, encoder.audio_input_width)),
        model_specific_data={
            "audio_feature_lens_raw": torch.tensor([2]),
            "feature_attention_mask": torch.tensor([True, True, False]),
        },
    )

    outputs = (
        model.get_image_feature([image]),
        model.get_video_feature([video]),
        model.get_audio_feature([audio]),
    )
    assert [tuple(output.shape) for output in outputs] == [
        (1, encoder.output_width),
        (1, encoder.output_width),
        (2, encoder.output_width),
    ]
    assert all(output.dtype == torch.bfloat16 for output in outputs)


def test_encoder_model_preserves_adapter_item_order_without_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrapper preserves one rowwise engine input per SGLang item, in order."""
    import furiosa_opt_engine as engine

    monkeypatch.setattr(engine, "HostBuffer", bytearray)

    class Transfer:
        def wait(self) -> None:
            pass

    class Buffer:
        def __init__(self, values: torch.Tensor) -> None:
            self.raw = values.to(torch.bfloat16).view(torch.uint8).numpy().tobytes()

        def __len__(self) -> int:
            return len(self.raw)

        def to_host(self, host: bytearray) -> Transfer:
            memoryview(host)[:] = self.raw
            return Transfer()

    class Rows:
        def __init__(self, values: list[int]) -> None:
            self.shape = (len(values), 1)
            self.buffer = Buffer(torch.tensor(values).reshape(-1, 1))

    class Encoder:
        vision_input_width = 1
        audio_input_width = 1
        output_width = 1

        def vision(self, items: list[list[int]]) -> list[Rows]:
            return [Rows(item) for item in items]

        def audio(self, items: list[list[int]]) -> list[Rows]:
            return [Rows(item) for item in items]

    class Adapter:
        def prepare_vision(
            self, item: object, _modality: str, _input_width: int, _device: object
        ) -> list[int]:
            return item.vision_rows

        def prepare_audio(self, item: object, _input_width: int, _device: object) -> list[int]:
            return item.audio_rows

    model = EncoderModel(Encoder(), None, Adapter())
    first = SimpleNamespace(vision_rows=[2, 3], audio_rows=[7, 8])
    second = SimpleNamespace(vision_rows=[5], audio_rows=[9])

    vision = model.get_image_feature([first, second])
    audio = model.get_audio_feature([first, second])

    assert torch.equal(vision, torch.tensor([[2], [3], [5]], dtype=torch.bfloat16))
    assert torch.equal(audio, torch.tensor([[7], [8], [9]], dtype=torch.bfloat16))


def test_encoder_disaggregation_allowlist_bypasses_only_registered_architecture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hook skips SGLang's allowlist refusal without swallowing unrelated errors."""
    from sglang.srt.arg_groups import pd_disaggregation_hook

    from furiosa_opt_sglang.encoder import allow_registered_encoder_disaggregation

    server_args = object()
    monkeypatch.setattr(
        pd_disaggregation_hook,
        "resolving_view",
        lambda _arguments: SimpleNamespace(encoder_only=True),
    )
    monkeypatch.setattr(
        pd_disaggregation_hook,
        "model_config_of",
        lambda _arguments: SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["Gemma4UnifiedForConditionalGeneration"])
        ),
    )

    def rejected_by_architecture(_arguments: object) -> None:
        raise ValueError(
            "Model type Gemma4UnifiedForConditionalGeneration is not supported "
            "for encoder disaggregation."
        )

    assert allow_registered_encoder_disaggregation(rejected_by_architecture, server_args) is None

    def rejected_for_another_reason(_arguments: object) -> None:
        raise ValueError("encoder transfer backend is unsupported")

    with pytest.raises(ValueError, match="encoder transfer backend is unsupported"):
        allow_registered_encoder_disaggregation(rejected_for_another_reason, server_args)
