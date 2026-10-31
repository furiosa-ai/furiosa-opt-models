"""A vision-language checkpoint's media encoder gives an image, a video and audio the rows
Transformers' embedders give them, called through the engine's API and on vLLM's encoder-only
instance, which hands them to the encoder cache connector."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import pytest

if TYPE_CHECKING:
    import torch

# vLLM's processing of a Gemma 4 prompt pads with a length its tokenizer ignores, which it warns of.
pytestmark = pytest.mark.filterwarnings(
    "ignore:`max_length` is ignored when `padding`=`True`:UserWarning:transformers.*"
)
# The checkpoints whose media encoder the engine loads alone; the engine serves no language model of
# theirs yet, so the inventory lists none of them for media.
ENCODERS = [
    pytest.param(
        "unsloth/gemma-4-12b-it-NVFP4", "b1f649734b34aa5575b03d186abd1b9be3d0d5c4", id="gemma4"
    )
]
QUESTIONS = {
    "image": "What is shown in this image? Answer in one sentence.",
    "video": "What happens in this video? Answer in one sentence.",
    "audio": "Transcribe this audio.",
}


class Medium(NamedTuple):
    """One of vLLM's test assets: its modality, the item vLLM takes, the chat item Transformers'
    processor takes and the processor's options for it."""

    modality: str
    served: object
    item: dict[str, object]
    options: dict[str, object]


@pytest.fixture(scope="module")
def media() -> dict[str, Medium]:
    """vLLM's test assets by name: two images, a video of 32 frames and a clip of audio."""
    from vllm.assets.audio import AudioAsset
    from vllm.assets.image import ImageAsset, ImageAssetName
    from vllm.assets.video import VideoAsset

    def image(name: ImageAssetName) -> Medium:
        image = ImageAsset(name).pil_image.convert("RGB")
        return Medium("image", image, {"type": "image", "image": image}, {})

    video = VideoAsset("baby_reading", num_frames=32)
    frames, metadata = video.np_ndarrays, video.metadata
    # vLLM reads its asset's `do_sample_frames` as the processor option, which Transformers takes
    # beside the metadata: the frames are already sampled.
    sampled = {key: value for key, value in metadata.items() if key != "do_sample_frames"}
    options: dict[str, object] = {
        "video_metadata": sampled,
        "do_sample_frames": metadata["do_sample_frames"],
    }
    samples, rate = AudioAsset("mary_had_lamb").audio_and_sample_rate
    return {
        "stop_sign": image("stop_sign"),
        "cherry_blossom": image("cherry_blossom"),
        "baby_reading": Medium(
            "video", (frames, metadata), {"type": "video", "video": frames}, options
        ),
        "mary_had_lamb": Medium("audio", (samples, rate), {"type": "audio", "audio": samples}, {}),
    }


def prompt(medium: Medium) -> list[dict[str, object]]:
    """One user message: the item, then the question about its modality."""
    question = {"type": "text", "text": QUESTIONS[medium.modality]}
    return [{"role": "user", "content": [medium.item, question]}]


@pytest.mark.feature("Multimodal inputs (images, video, audio)")
@pytest.mark.parametrize(("model", "revision"), ENCODERS)
def test_encoder_embeds_each_item_as_transformers_does(
    model: str, revision: str, media: dict[str, Medium]
) -> None:
    """The media encoder, loaded alone, returns for each item of a call the rows Transformers'
    embedders give it: in one call two images, which share a launch, and a video's frames, which
    take another; then a clip of audio."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor

    import furiosa_opt_engine as engine

    directory = snapshot_download(
        model, revision=revision, allow_patterns=["*.json", "*.safetensors"]
    )

    def embedded(
        images: list[tuple[torch.Tensor, torch.Tensor]], clips: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """The rows of each of `images`, patches and their positions, from one call, and of each
        of `clips` from another, read before the device objects drop."""
        device = engine.Device(engine.Placement.visible([0], range(8)))
        encoder = engine.LoadOptions().encoder().load(engine.Checkpoint(directory), device)
        assert isinstance(encoder, engine.Encoder)

        def upload(rows: torch.Tensor) -> engine.HbmTensor:
            rows = rows.to(torch.bfloat16).contiguous()
            return engine.HostTensor.from_dlpack(rows).to_hbm(device)

        def host(call: list[engine.HbmTensor]) -> list[torch.Tensor]:
            return [torch.from_numpy(rows.to_host().numpy()) for rows in call]

        vision = encoder.vision(
            [engine.vision_language.Image(upload(p), xy.tolist()) for p, xy in images]
        )
        audio = encoder.audio([engine.vision_language.Audio(upload(frames)) for frames in clips])
        return host(vision), host(audio)

    def patches(pixels: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The patches of every frame that have a position, the processor's padding left out."""
        pixels, positions = pixels.flatten(0, -2), positions.flatten(0, -2)
        valid = ~(positions == -1).all(dim=-1)
        return pixels[valid], positions[valid]

    processor = AutoProcessor.from_pretrained(model, revision=revision)
    inputs = {
        name: processor.apply_chat_template(
            prompt(medium),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs=medium.options,
        )
        for name, medium in media.items()
    }
    names = ["stop_sign", "cherry_blossom", "baby_reading"]
    stop_sign, cherry_blossom, video = (inputs[name] for name in names)
    images = [
        patches(stop_sign["pixel_values"][0], stop_sign["image_position_ids"][0]),
        patches(cherry_blossom["pixel_values"][0], cherry_blossom["image_position_ids"][0]),
        patches(video["pixel_values_videos"][0], video["video_position_ids"][0]),
    ]
    audio = inputs["mary_had_lamb"]
    clips = [audio["input_features"][0][audio["input_features_mask"][0]]]
    image_rows, clip_rows = embedded(images, clips)
    assert (len(image_rows), len(clip_rows)) == (len(images), len(clips))
    for name, rows in zip(names, image_rows, strict=True):
        check(rows, model, revision, inputs[name], media[name].modality)
    check(clip_rows[0], model, revision, audio, "audio")


@pytest.mark.feature("Disaggregated encoder and encoder caches")
@pytest.mark.parametrize(("model", "revision"), ENCODERS)
@pytest.mark.parametrize("name", ["stop_sign", "baby_reading", "mary_had_lamb"])
def test_encoder_instance_saves_transformers_rows(
    tmp_path: Path, model: str, revision: str, media: dict[str, Medium], name: str
) -> None:
    """An encoder-only instance saves, for the item of a request, the rows Transformers' embedders
    give it: one row for each placeholder position, accepted against Transformers' BF16 rows in
    magnitude and in direction."""
    import safetensors.torch
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.config import ECTransferConfig

    medium = media[name]
    processor = AutoProcessor.from_pretrained(model, revision=revision)
    text = processor.apply_chat_template(prompt(medium), add_generation_prompt=True, tokenize=False)
    transfer = ECTransferConfig(
        ec_connector="ECExampleConnector",
        ec_role="ec_producer",
        ec_connector_extra_config={"shared_storage_path": str(tmp_path)},
    )
    producer = LLM(model=model, revision=revision, max_num_seqs=1, ec_transfer_config=transfer)
    _ = producer.generate(
        {"prompt": text, "multi_modal_data": {medium.modality: medium.served}},
        SamplingParams(temperature=0, max_tokens=1),
    )
    producer.llm_engine.engine_core.shutdown()
    (saved,) = tmp_path.glob("*/encoder_cache.safetensors")
    rows = safetensors.torch.load_file(saved)["ec_cache"]
    inputs = processor.apply_chat_template(
        prompt(medium),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs=medium.options,
    )
    check(rows, model, revision, inputs, medium.modality)


def check(
    rows: torch.Tensor, model: str, revision: str, inputs: dict[str, torch.Tensor], modality: str
) -> None:
    """An item's `rows` held to Transformers' BF16 rows `t` of its `inputs` by an acceptance policy,
    fixed in advance from Transformers' rows alone and not an error bound: with `x` the FP32 rows
    of the same inputs and `e = |t - x| / |t|`, one row for each of `t`'s, a relative L2 distance
    from `t` of at most `2e`, and in each row a cosine to `t` of at least `2 cos(t, x)^2 - 1`."""
    import torch

    expected, exact = (
        features(model, revision, inputs, modality, dtype).double()
        for dtype in (torch.bfloat16, torch.float32)
    )
    rows = rows.double()
    assert rows.shape == expected.shape
    allowed = 2 * float((expected - exact).norm() / expected.norm())
    distance = float((rows - expected).norm() / expected.norm())
    cosine = torch.nn.functional.cosine_similarity
    cosines = cosine(rows, expected, dim=1)
    thresholds = 2 * cosine(expected, exact, dim=1) ** 2 - 1
    worst = int(cosines.argmin())
    print(
        f"{modality}: {len(rows)} rows, relative L2 {distance:.3e} of {allowed:.3e},",
        f"worst row cosine {cosines[worst]:.7f} of {thresholds[worst]:.7f}",
    )
    assert distance <= allowed, f"relative L2 {distance:.3e} above {allowed:.3e}"
    below = (cosines < thresholds).nonzero().flatten().tolist()
    assert not below, f"rows {below} below their cosine thresholds"


def features(
    model: str,
    revision: str,
    inputs: dict[str, torch.Tensor],
    modality: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Transformers' rows of the item's placeholder positions in `dtype`, from its embedder modules
    loaded with the checkpoint's weights and reading the item's inputs rounded to BF16, as a BF16
    model reads them: the whole model's quantized language model is not needed.
    """
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.gemma4_unified import modeling_gemma4_unified as hf

    directory = snapshot_download(model, revision=revision, allow_patterns=["*.json"])
    config = AutoConfig.from_pretrained(directory)
    weights = snapshot_download(model, revision=revision, allow_patterns=["*.safetensors"])
    with safe_open(f"{weights}/model.safetensors", "pt") as file:

        def state(prefix: str) -> dict[str, torch.Tensor]:
            names = (name for name in file.keys() if name.startswith(prefix))  # noqa: SIM118
            return {name.removeprefix(prefix): file.get_tensor(name) for name in names}

        vision = hf.Gemma4UnifiedVisionEmbedder(config.vision_config, config.text_config)
        projection = state("model.embed_vision.")
        vision.load_state_dict(
            state("model.vision_embedder.")
            | {f"multimodal_embedder.{name}": value for name, value in projection.items()},
            strict=True,
        )
        audio = hf.Gemma4UnifiedMultimodalEmbedder(config.audio_config, config.text_config)
        audio.load_state_dict(state("model.embed_audio."), strict=True)
    vision, audio = vision.to(dtype).eval(), audio.to(dtype).eval()

    def patches(pixels: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        valid = ~(positions == -1).all(dim=-1)
        return vision(pixels[None].to(torch.bfloat16).to(dtype), positions[None])[0][valid]

    with torch.inference_mode():
        match modality:
            case "image":
                return patches(inputs["pixel_values"][0], inputs["image_position_ids"][0])
            case "video":
                frames = zip(
                    inputs["pixel_values_videos"][0], inputs["video_position_ids"][0], strict=True
                )
                return torch.cat([patches(pixels, positions) for pixels, positions in frames])
            case "audio":
                frames = inputs["input_features"][0].to(torch.bfloat16).to(dtype)
                return audio(frames)[inputs["input_features_mask"][0]]
            case _:
                raise ValueError(modality)
