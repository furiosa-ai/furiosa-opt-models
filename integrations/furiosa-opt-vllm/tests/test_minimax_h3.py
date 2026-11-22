"""Opt-in H3 external-reference and native callback contracts.

Run this module with --h3, FURIOSA_H3_CHECKPOINT (local FL2VA directory or repository root),
FURIOSA_H3_GOLDEN (the private opt request's manifest/inputs/outputs), and four explicitly selected
FURIOSA_VISIBLE_CHIPS. No checkpoint or reference is downloaded;
no media or trace files are written.
"""

from __future__ import annotations

import faulthandler
import gc
import hashlib
import json
import os
from collections.abc import Callable, Iterator
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from typing import TYPE_CHECKING, TypeAlias, cast
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pytest
import torch
from torch import nn

if TYPE_CHECKING:
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.worker.utils import StepRequestState

    import furiosa_opt_engine as engine

pytestmark = [
    pytest.mark.h3,
    pytest.mark.filterwarnings("ignore:`torch.jit.script` is deprecated:DeprecationWarning"),
]

REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
CHECKPOINT = f"MiniMaxAI/MiniMax-H3@{REVISION} (FL2VA/)"
CAPTURED = [0, 31, 62, 93, 123]
# Original opt raw-output bounds, before media encoding; no tolerance fitted to port outputs.
FRAMES, WAVEFORM = (0.975045, 0.2231), (0.970748, 0.2419)
Statistics: TypeAlias = tuple[float, float] | None


@dataclass(repr=False)
class Golden:
    checkpoint: Path
    inputs: dict[str, torch.Tensor]
    outputs: dict[str, torch.Tensor]


@dataclass(repr=False)
class Loaded:
    model: MiniMaxH3Pipeline
    transformer: engine.minimax_h3.Transformer
    text: engine.minimax_h3.TextEncoder
    video: engine.minimax_h3.VideoDecoder
    audio: engine.minimax_h3.AudioDecoder


@dataclass(repr=False)
class Forward:
    kwargs: dict[str, object]


def tensor(value: object, shape: tuple[int, ...], dtype: torch.dtype, name: str) -> torch.Tensor:
    """Report contracts without putting private tensor values in assertion output."""
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != dtype:
        pytest.fail(f"{name}: unexpected tensor shape or dtype", pytrace=False)
    if not bool(torch.isfinite(value).all()):
        pytest.fail(f"{name}: nonfinite tensor", pytrace=False)
    return value


@pytest.fixture(scope="module")
def h3_golden(pytestconfig: pytest.Config) -> Golden:
    if not pytestconfig.getoption("h3", default=False):
        pytest.skip("H3 requires --h3")
    paths: dict[str, Path] = {}
    for name in ("FURIOSA_H3_CHECKPOINT", "FURIOSA_H3_GOLDEN"):
        value = os.environ.get(name)
        if not value or not Path(value).is_dir():
            pytest.fail(f"--h3 requires {name} to name an existing local directory", pytrace=False)
        paths[name] = Path(value).resolve()
    checkpoint, reference = paths["FURIOSA_H3_CHECKPOINT"], paths["FURIOSA_H3_GOLDEN"]
    from safetensors import SafetensorError
    from safetensors.torch import load_file

    # Check an available Hub snapshot identity; copied local directories still undergo the
    # engine's full component-schema and strict weight validation when loaded below.
    for parent in (checkpoint, *checkpoint.parents):
        if parent.parent.name == "snapshots" and parent.name != REVISION:
            pytest.fail("local H3 snapshot revision differs from the opt reference", pytrace=False)
    try:
        manifest = json.loads((reference / "manifest.json").read_text())
        if not isinstance(manifest, dict) or not isinstance(manifest.get("workload"), dict):
            raise ValueError("expected an opt reference manifest")
        if manifest["checkpoint"] != CHECKPOINT:
            pytest.fail(
                "H3 golden checkpoint identity differs from the selected FL2VA revision",
                pytrace=False,
            )
        workload = manifest["workload"]
        expected = {
            "task": "t2va",
            "seed": 424242424251,
            "steps": 50,
            "width": 1344,
            "height": 768,
            "frames": 124,
            "fps": 24,
            "flow_shift": 12,
            "audio_flow_shift": 3,
            "sampler": "euler",
        }
        if any(workload.get(key) != value for key, value in expected.items()):
            pytest.fail("H3 golden workload differs from the original opt request", pytrace=False)
        for name in ("inputs.safetensors", "outputs.safetensors"):
            with (reference / name).open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != manifest["sha256"][name]:
                pytest.fail(f"H3 golden hash mismatch: {name}", pytrace=False)
        inputs = load_file(reference / "inputs.safetensors")
        outputs = load_file(reference / "outputs.safetensors")
    except (OSError, ValueError, KeyError, TypeError, SafetensorError):
        pytest.fail(
            "H3 golden requires a valid opt manifest and its two hashed tensor files",
            pytrace=False,
        )
    if set(inputs) != {"text_embeddings", "text_tags"} or set(outputs) != {"frames", "waveform"}:
        pytest.fail("H3 golden has unexpected input/output tensor names", pytrace=False)
    _ = tensor(inputs["text_embeddings"], (77, 5120), torch.bfloat16, "text embeddings")
    _ = tensor(inputs["text_tags"], (77,), torch.int64, "text tags")
    _ = tensor(outputs["frames"], (5, 768, 1344, 3), torch.uint8, "reference frames")
    _ = tensor(outputs["waveform"], (1, 2, 165600), torch.float32, "reference waveform")
    return Golden(checkpoint, inputs, outputs)


@pytest.fixture(scope="module")
def h3_distributed(h3_golden: Golden, tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    import huggingface_hub.constants
    import vllm.platforms
    import vllm_omni.platforms
    from vllm.platforms.cpu import CpuPlatform
    from vllm.utils import mem_utils
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )

    from furiosa_opt_vllm.omni.platform import Platform

    del h3_golden  # Validate local inputs before any model/device setup.
    try:
        chips = [int(chip) for chip in os.environ["FURIOSA_VISIBLE_CHIPS"].split(",")]
    except (KeyError, ValueError):
        pytest.fail("H3 requires four explicit FURIOSA_VISIBLE_CHIPS", pytrace=False)
    if len(chips) != 4 or len(set(chips)) != 4 or min(chips) < 0:
        pytest.fail("H3 requires four distinct nonnegative FURIOSA_VISIBLE_CHIPS", pytrace=False)
    rendezvous = tmp_path_factory.mktemp("h3-distributed") / "rendezvous"
    with (
        patch.object(vllm.platforms, "_current_platform", CpuPlatform()),
        patch.object(vllm_omni.platforms, "_current_omni_platform", Platform()),
        patch.object(mem_utils, "current_platform", Platform()),
        pytest.MonkeyPatch.context() as environment,
    ):
        environment.setenv("HF_HUB_OFFLINE", "1")
        environment.setenv("TRANSFORMERS_OFFLINE", "1")
        environment.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
        try:
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                backend="gloo",
                distributed_init_method=rendezvous.as_uri(),
            )
            initialize_model_parallel(
                data_parallel_size=1, sequence_parallel_size=1, backend="gloo"
            )
            yield
        finally:
            # Runs after the loaded fixture's cached value is released, including framework
            # cycles that would otherwise retain native views and their chips.
            _ = gc.collect()
            destroy_model_parallel()
            destroy_distributed_environment()


@pytest.fixture(scope="module")
def h3_pipeline(h3_golden: Golden, h3_distributed: None) -> Iterator[Loaded]:
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.vllm_config import create_base_diffusion_vllm_config

    import furiosa_opt_engine as engine
    from furiosa_opt_vllm.omni.components import Adapter, compose, construct
    from furiosa_opt_vllm.omni.pipeline import Runner, directory

    del h3_distributed
    configuration = OmniDiffusionConfig(
        model=str(h3_golden.checkpoint),
        model_class_name="MiniMaxH3Pipeline",
        trust_remote_code=True,
        enforce_eager=True,
    )
    runner = Runner(
        create_base_diffusion_vllm_config(torch.device("cpu"), configuration),
        configuration,
        torch.device("cpu"),
    )
    # Runner opens the selected four-chip/all-eight-PE context before preparing host weights.
    runner.load_model()
    model = runner.pipeline
    if not isinstance(model, MiniMaxH3Pipeline):
        pytest.fail("Runner did not load the H3 framework pipeline", pytrace=False)
    # Composition replaced the modules described by the upstream pipeline's annotations.
    views: tuple[object, ...] = (
        model.transformer,
        model.text_encoder,
        model.video_vae.model,
        model.audio_vae.model,
    )
    transformer, text, video, audio = (cast(Adapter, view).component for view in views)
    if not (
        isinstance(transformer, engine.minimax_h3.Transformer)
        and isinstance(text, engine.minimax_h3.TextEncoder)
        and isinstance(video, engine.minimax_h3.VideoDecoder)
        and isinstance(audio, engine.minimax_h3.AudioDecoder)
    ):
        pytest.fail("Runner components are not the public H3 native bindings", pytrace=False)
    loaded = Loaded(model, transformer, text, video, audio)
    device = loaded.transformer.device
    for component in (loaded.text, loaded.video, loaded.audio):
        if component.to_hbm(device) is not component:
            pytest.fail("H3 component does not retain the pipeline's device", pytrace=False)
    # Preserve recorded()'s external-conditioning path while retaining the same owner's
    # text view for callback tests. This constructs a framework wrapper, not another engine.
    external = OmniDiffusionConfig(
        model=str(h3_golden.checkpoint),
        model_class_name="MiniMaxH3Pipeline",
        trust_remote_code=True,
        enforce_eager=True,
        model_loaded={"text_encoder": False, "vae_encoder": False},
    )
    components = {
        "transformer": loaded.transformer,
        "video_vae": loaded.video,
        "audio_vae": loaded.audio,
    }
    wrapper = compose(construct(external, directory(external), components), components)
    if not isinstance(wrapper, MiniMaxH3Pipeline):
        pytest.fail("external-conditioning wrapper is not an H3 pipeline", pytrace=False)
    loaded.model = wrapper
    runner.pipeline = wrapper
    del model
    try:
        yield loaded
    finally:
        loaded.transformer.reset_cache()
        device.synchronize()


def recorded(golden: Golden) -> StepRequestState:
    """Original opt recorded-conditioning workload; scheduling and noise stay in Omni."""
    from vllm_omni.diffusion.worker.utils import StepRequestState
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.model_executor.models.minimax_h3.conditioning import MiniMaxH3EncoderConditioning

    conditioning = MiniMaxH3EncoderConditioning(
        hidden_states=golden.inputs["text_embeddings"],
        token_tags=golden.inputs["text_tags"],
        task="t2va",
        height=768,
        width=1344,
        num_frames=124,
        latent_t=37,
        audio_t=207,
    )
    return StepRequestState(
        request_id="reference",
        sampling=OmniDiffusionSamplingParams(seed=424242424251, num_inference_steps=50, fps=24),
        prompt={"additional_information": {"encoder_output": conditioning.to_omni_payload()}},
    )


def compare(
    actual: torch.Tensor, expected: torch.Tensor, bounds: tuple[float, float], name: str
) -> None:
    actual, expected = actual.double().flatten(), expected.double().flatten()
    squared = (expected * expected).sum()
    cosine = float((actual * expected).sum() / ((actual * actual).sum() * squared).sqrt())
    relative = float((((actual - expected) ** 2).sum() / squared).sqrt())
    print(f"{name}: cosine={cosine:.9f}, relative L2={relative:.9f}", flush=True)
    if not (cosine >= bounds[0] and relative <= bounds[1]):
        pytest.fail(f"{name}: cosine={cosine:.9f}, relative L2={relative:.9f}", pytrace=False)


def test_raw_reference(h3_pipeline: Loaded, h3_golden: Golden) -> None:
    from vllm_omni.diffusion.worker.input_batch import InputBatch

    h3_pipeline.transformer.reset_cache()
    request = recorded(h3_golden)
    with torch.inference_mode():
        _ = h3_pipeline.model.prepare_encode(request)
        for _ in range(50):
            noise = h3_pipeline.model.denoise_step(
                InputBatch.make_batch([request]), states=[request]
            )
            if not isinstance(noise, torch.Tensor) or not bool(torch.isfinite(noise).all()):
                pytest.fail("H3 denoising returned no finite noise tensor", pytrace=False)
            h3_pipeline.model.step_scheduler(request, noise)
        output = h3_pipeline.model.post_decode(request).output
    if not isinstance(output, tuple) or len(output) != 2:
        pytest.fail("H3 raw request returned no video/audio pair", pytrace=False)
    video = tensor(output[0], (1, 124, 768, 1344, 3), torch.uint8, "generated video")
    waveform = tensor(output[1], (1, 2, 165600), torch.float32, "generated waveform")
    compare(video[0, CAPTURED], h3_golden.outputs["frames"], FRAMES, "video")
    compare(waveform, h3_golden.outputs["waveform"], WAVEFORM, "audio")


@pytest.fixture(scope="module")
def h3_forward(h3_pipeline: Loaded, h3_golden: Golden) -> Forward:
    from vllm_omni.diffusion.worker.input_batch import InputBatch

    captured = Forward({})

    def capture(module: nn.Module, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        del module
        if args:
            pytest.fail("H3 transformer no longer receives keyword arguments", pytrace=False)
        prompt = tensor(kwargs.get("prompt_embeds"), (77, 5120), torch.bfloat16, "forward prompt")
        if not torch.equal(prompt, h3_golden.inputs["text_embeddings"]):
            pytest.fail("Omni did not preserve the supplied opt text conditioning", pytrace=False)
        captured.kwargs = deepcopy(kwargs)

    h3_pipeline.transformer.reset_cache()
    request = recorded(h3_golden)
    with torch.inference_mode():
        _ = h3_pipeline.model.prepare_encode(request)
        handle = h3_pipeline.model.transformer.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            _ = h3_pipeline.model.denoise_step(InputBatch.make_batch([request]), states=[request])
        finally:
            handle.remove()
    if not captured.kwargs:
        pytest.fail("framework transformer hook captured no forward arguments", pytrace=False)
    return captured


def finite_output(
    output: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]],
) -> None:
    for value in output:
        if not bool(np.isfinite(value).all()):
            pytest.fail("native H3 conversion returned no finite tensor", pytrace=False)


@pytest.fixture
def callback_deadline(h3_forward: Forward) -> Iterator[None]:
    """Terminate a callback deadlock without arming a deadline during loading or raw50."""
    del h3_forward
    faulthandler.dump_traceback_later(30, exit=True)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


@pytest.mark.usefixtures("callback_deadline")
def test_callback_refuses_same_pipeline_reentry(h3_pipeline: Loaded, h3_forward: Forward) -> None:
    native, kwargs = h3_pipeline.transformer, h3_forward.kwargs
    ids = torch.tensor([1], dtype=torch.int64)
    video = torch.zeros(1, 24, 37, 48, 84, dtype=torch.float32)
    audio = torch.zeros(2, 32, 207, dtype=torch.float32)
    calls: list[tuple[str, Callable[[], object]]] = [
        ("reset", native.reset_cache),
        ("forward", lambda: native.prepare(**kwargs)),
        ("cached_forward", lambda: native.prepare(**kwargs).forward_cached(lambda _: True)),
        ("text", lambda: h3_pipeline.text.prepare(ids)),
        ("video", lambda: h3_pipeline.video.prepare(video, frame_num=1)),
        ("audio", lambda: h3_pipeline.audio.prepare(audio)),
    ]
    invoked = False

    def decide(means: Statistics) -> bool:
        nonlocal invoked
        del means
        invoked = True
        for _, call in calls:
            with pytest.raises(RuntimeError, match="cannot reenter"):
                _ = call()
        return True

    native.reset_cache()
    finite_output(native.prepare(**kwargs).forward_cached(decide).to_host())
    if not invoked:
        pytest.fail("native H3 did not invoke its decision callback", pytrace=False)
    # The caught refusals leave the outer evaluation and subsequent ordinary calls usable.
    finite_output(native.prepare(**kwargs).forward().to_host())
    _ = tensor(
        torch.from_numpy(h3_pipeline.text.prepare(ids).forward().to_host()).view(torch.bfloat16),
        (1, 5120),
        torch.bfloat16,
        "recovered encoder",
    )


@pytest.mark.parametrize("failure_kind", ["python", "reentry"])
@pytest.mark.usefixtures("callback_deadline")
def test_callback_exception_invalidates_cache(
    h3_pipeline: Loaded, h3_forward: Forward, failure_kind: str
) -> None:
    native, kwargs = h3_pipeline.transformer, h3_forward.kwargs
    native.reset_cache()
    expected = native.prepare(**kwargs).forward_cached(lambda _: True).to_host()
    failure = LookupError("opt H3 callback failure")

    def fail(means: Statistics) -> bool:
        if means is None:
            pytest.fail("successful cached call retained no probe", pytrace=False)
        if failure_kind == "reentry":
            native.reset_cache()
            pytest.fail("callback reset did not refuse reentry", pytrace=False)
        raise failure

    kind = RuntimeError if failure_kind == "reentry" else LookupError
    reason = "cannot reenter" if failure_kind == "reentry" else "opt H3 callback failure"
    with pytest.raises(kind, match=reason) as caught:
        _ = native.prepare(**kwargs).forward_cached(fail).to_host()
    if failure_kind == "python" and caught.value is not failure:
        pytest.fail("native binding replaced the callback exception", pytrace=False)

    def recover(means: Statistics) -> bool:
        if means is not None:
            pytest.fail("failed callback left a published cache", pytrace=False)
        return True

    actual = native.prepare(**kwargs).forward_cached(recover).to_host()
    finite_output(actual)
    if any(not np.array_equal(left, right) for left, right in zip(actual, expected, strict=True)):
        pytest.fail("callback recovery changed a forced-compute output", pytrace=False)
    # The recovered anchor can also serve the cached branch on the next call.
    finite_output(native.prepare(**kwargs).forward_cached(lambda _: False).to_host())


@pytest.mark.usefixtures("callback_deadline")
def test_callback_allows_queued_other_thread_reset(
    h3_pipeline: Loaded, h3_forward: Forward
) -> None:
    native, kwargs = h3_pipeline.transformer, h3_forward.kwargs
    started, finished = Event(), Event()
    errors: list[BaseException] = []

    def reset() -> None:
        started.set()
        try:
            native.reset_cache()
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    worker = Thread(target=reset, daemon=True)

    def decide(means: Statistics) -> bool:
        del means
        worker.start()
        if not started.wait(5):
            pytest.fail("reset worker did not start", pytrace=False)
        if finished.wait(0.1):
            pytest.fail(
                "other-thread reset did not queue behind the active callback", pytrace=False
            )
        return True

    native.reset_cache()
    try:
        finite_output(native.prepare(**kwargs).forward_cached(decide).to_host())
    finally:
        # Never join inside the callback: the reset legitimately waits for this call to finish.
        if worker.ident is not None:
            worker.join(timeout=30)
    if worker.is_alive() or errors or not finished.is_set():
        pytest.fail("other-thread reset did not complete normally after the call", pytrace=False)

    def fresh(means: Statistics) -> bool:
        if means is not None:
            pytest.fail("queued reset did not invalidate the completed call's cache", pytrace=False)
        return True

    finite_output(native.prepare(**kwargs).forward_cached(fresh).to_host())
