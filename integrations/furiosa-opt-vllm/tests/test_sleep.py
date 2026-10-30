"""Sleep mode frees the weights' and the KV cache's device memory and wakes to the same answers."""

from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest
import torch
from huggingface_hub import snapshot_download
from pydantic import JsonValue, TypeAdapter
from safetensors.torch import load, save
from typing_extensions import TypedDict
from vllm import LLM, SamplingParams

import furiosa_opt_engine as engine

if TYPE_CHECKING:
    from collections.abc import Generator

    from furiosa_opt_vllm.worker import FuriosaWorker

CHECKPOINT: Final = "Qwen/Qwen3-0.6B"
PROMPTS: Final = ("The capital of France is", "List the planets of the solar system:")
# Blocks of the KV cache: fewer bytes than the weights, so device memory can fit one but not the
# other.
BLOCKS: Final = 64
GIB: Final = 1 << 30


class Entry(TypedDict):
    """A tensor's byte range in its safetensors file."""

    data_offsets: tuple[int, int]


HEADER: Final = TypeAdapter(dict[str, JsonValue])
ENTRIES: Final = TypeAdapter(dict[str, Entry])


@pytest.fixture(scope="module")
def files(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A copy of the checkpoint, whose files the tests rewrite."""
    copy = tmp_path_factory.mktemp("checkpoint") / "Qwen3-0.6B"
    _ = shutil.copytree(snapshot_download(CHECKPOINT), copy)
    return copy


@pytest.fixture(scope="module")
def llm(files: Path) -> Generator[LLM]:
    with pytest.MonkeyPatch.context() as patch:
        # The tests reach the worker's device through functions of their own.
        patch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        llm = LLM(
            str(files),
            dtype="bfloat16",
            enable_sleep_mode=True,
            max_model_len=512,
            num_gpu_blocks_override=BLOCKS,
        )
        yield llm
        llm.llm_engine.engine_core.shutdown()


def answers(llm: LLM) -> list[list[int]]:
    """Each prompt's greedy tokens."""
    params = SamplingParams(temperature=0, max_tokens=24, ignore_eos=True)
    return [list(output.outputs[0].token_ids) for output in llm.generate(list(PROMPTS), params)]


def available(llm: LLM) -> int:
    """Free device memory per chip, as the worker's engine device reports it."""

    def read(worker: FuriosaWorker) -> int:
        runner = worker.runner
        assert runner is not None and runner.model is not None
        return runner.model.device.available

    [free] = llm.collective_rpc(read)
    return free


def cache(llm: LLM) -> int:
    """Bytes of the KV cache per chip, which the worker reads from the awake model."""
    blocks = llm.llm_engine.vllm_config.cache_config.num_gpu_blocks
    [block] = llm.collective_rpc("get_cache_block_size_bytes")
    assert blocks is not None
    return blocks * block


def weights() -> int:
    """Bytes of the checkpoint's distinct weights: its tensors, less a head tied to the token
    table, which the model reads once."""
    directory = Path(snapshot_download(CHECKPOINT))
    tied = HEADER.validate_json((directory / "config.json").read_bytes())["tie_word_embeddings"]
    total = 0
    for file in directory.glob("*.safetensors"):
        with file.open("rb") as reader:
            header = HEADER.validate_json(reader.read(int.from_bytes(reader.read(8), "little")))
        _ = header.pop("__metadata__", None)
        total += sum(
            end - start
            for name, entry in ENTRIES.validate_python(header).items()
            if not (tied and name == "lm_head.weight")
            for start, end in [entry["data_offsets"]]
        )
    return total


@contextmanager
def rewritten(files: Path, content: bytes) -> Generator[None]:
    """The checkpoint's weight files holding `content` in place of their own, until the block
    ends."""
    original = Path(snapshot_download(CHECKPOINT))
    tensors = sorted(files.glob("*.safetensors"))
    try:
        for file in tensors:
            _ = file.write_bytes(content)
        yield
    finally:
        for file in tensors:
            _ = shutil.copyfile(original / file.name, file)


def hold(llm: LLM, name: str, size: int) -> None:
    """Allocate `size` bytes of the worker's device memory, held as `name` until `drop`."""

    def allocate(worker: FuriosaWorker) -> None:
        runner = worker.runner
        assert runner is not None and runner.model is not None
        sizes = [GIB] * (size // GIB) + [size % GIB]
        device = runner.model.device
        vars(worker)[name] = [engine.HbmBuffer(device, part) for part in sizes if part]

    _ = llm.collective_rpc(allocate)


def drop(llm: LLM, name: str) -> None:
    """Free the device memory `hold` allocated as `name`."""

    def free(worker: FuriosaWorker) -> None:
        del vars(worker)[name]

    _ = llm.collective_rpc(free)


@pytest.mark.feature("Sleep mode")
@pytest.mark.parametrize("level", [1, 2])
def test_wakes_to_the_same_answers(llm: LLM, level: int) -> None:
    """Sleep frees at least the distinct weights and the KV cache, and each level wakes to the
    answers of before; level 2 in vLLM's documented order: weights, reload, cache."""
    expected = answers(llm)
    awake, kv = available(llm), cache(llm)
    llm.sleep(level=level)
    assert available(llm) - awake >= weights() + kv
    if level == 1:
        llm.wake_up()
    else:
        llm.wake_up(tags=["weights"])
        _ = llm.collective_rpc("reload_weights")
        llm.wake_up(tags=["kv_cache"])
    assert answers(llm) == expected


@pytest.mark.feature("Sleep mode")
def test_wakes_to_the_loaded_weights_at_level_1(llm: LLM, files: Path) -> None:
    """Level 1 wakes to the weights the server loaded: checkpoint files that hold other weights
    from before the sleep to after the wake leave the answers as before."""
    expected = answers(llm)
    [file] = Path(snapshot_download(CHECKPOINT)).glob("*.safetensors")
    tensors = load(file.read_bytes())
    # Other weights of the same names and shapes, which answer otherwise.
    tensors["model.norm.weight"] = -tensors["model.norm.weight"]
    with rewritten(files, save(tensors)):
        llm.sleep(level=1)
        llm.wake_up()
        assert answers(llm) == expected


@pytest.mark.feature("Sleep mode")
def test_wakes_from_the_files_at_level_2(llm: LLM, files: Path) -> None:
    """Level 2 keeps nothing: waking reads the checkpoint's files, fails while they are empty, and
    answers as before once they are back."""
    expected = answers(llm)
    llm.sleep(level=2)
    with rewritten(files, b""), pytest.raises(Exception, match="model.safetensors"):
        llm.wake_up()
    llm.wake_up()
    assert answers(llm) == expected


@pytest.mark.feature("Sleep mode")
def test_wakes_each_tag_alone(llm: LLM) -> None:
    """The KV cache wakes before the weights into its own memory, the weights wake after it, and
    the engine answers as before."""
    expected, kv = answers(llm), cache(llm)
    llm.sleep(level=1)
    asleep = available(llm)
    llm.wake_up(tags=["kv_cache"])
    assert asleep - available(llm) == kv
    llm.wake_up(tags=["weights"])
    assert asleep - available(llm) >= kv + weights()
    assert answers(llm) == expected


@pytest.mark.feature("Sleep mode")
def test_wakes_again_after_the_cache_fails(llm: LLM) -> None:
    """A wake whose KV cache finds no memory keeps the weights it loaded, so a retry with room for
    the cache but not for a second copy of the weights wakes the engine to the same answers."""
    expected = answers(llm)
    awake, kv = available(llm), cache(llm)
    llm.sleep(level=1)
    asleep = available(llm)
    loaded = asleep - awake - kv
    assert kv < loaded
    # In the 4-KiB units large buffers take: room for the weights and half the cache, and once
    # `small` drops, room for the cache but not for the weights.
    unit = 4096
    half = kv // 2 // unit * unit
    small = half + (loaded - kv) // 2 // unit * unit
    hold(llm, "big", (asleep - loaded - half - small) // unit * unit)
    hold(llm, "small", small)
    try:
        with pytest.raises(Exception, match="no contiguous space"):
            llm.wake_up()
        assert available(llm) < kv
        drop(llm, "small")
        llm.wake_up()
    finally:
        drop(llm, "big")
    assert answers(llm) == expected


@pytest.mark.feature("Sleep mode")
def test_releases_the_cache_alone(llm: LLM) -> None:
    """A paused engine releases its KV cache's memory and keeps the weights, then answers as
    before once the cache wakes."""
    expected = answers(llm)
    awake, kv = available(llm), cache(llm)
    llm.sleep(level=0)
    llm.release_kv_cache_memory()
    freed = available(llm) - awake
    assert kv <= freed < kv + weights()
    llm.wake_up()
    assert answers(llm) == expected


@pytest.mark.feature("Sleep mode")
def test_keeps_the_weights_when_a_reload_cannot_read(llm: LLM, files: Path) -> None:
    """A reload whose checkpoint files cannot be read keeps the served weights."""
    expected = answers(llm)
    with rewritten(files, b""), pytest.raises(Exception, match="model.safetensors"):
        _ = llm.collective_rpc("reload_weights")
    assert answers(llm) == expected


@pytest.mark.feature("Sleep mode")
def test_reloads_without_room_for_a_second_copy(llm: LLM) -> None:
    """A reload frees the served weights' device memory before it loads the new ones, so it loads
    them with less free memory than one copy of the weights and answers as before."""
    expected = answers(llm)
    # In the 4-KiB units large buffers take: free memory just short of one copy of the weights.
    unit = 4096
    hold(llm, "filler", (available(llm) - weights()) // unit * unit + unit)
    try:
        assert available(llm) < weights()
        _ = llm.collective_rpc("reload_weights")
    finally:
        drop(llm, "filler")
    assert answers(llm) == expected


@pytest.mark.feature("Sleep mode")
def test_reloads_again_after_a_failed_load(llm: LLM, files: Path) -> None:
    """A reload whose checkpoint lacks the model's tensors leaves the engine awake without
    weights, and a reload of the restored checkpoint answers as before."""
    expected = answers(llm)
    awake = available(llm)
    with (
        rewritten(files, save({"unrelated": torch.zeros(1)})),
        pytest.raises(Exception, match="missing from the checkpoint"),
    ):
        _ = llm.collective_rpc("reload_weights")
    assert not llm.llm_engine.is_sleeping()
    assert available(llm) - awake >= weights()
    _ = llm.collective_rpc("reload_weights")
    assert answers(llm) == expected


@pytest.mark.feature("Sleep mode")
def test_restores_the_weights_a_failed_reload_dropped(llm: LLM, files: Path) -> None:
    """A failed reload keeps the copy of the weights it dropped, so level 1 wakes to them while
    the checkpoint's files still lack the model's tensors."""
    expected = answers(llm)
    with rewritten(files, save({"unrelated": torch.zeros(1)})):
        with pytest.raises(Exception, match="missing from the checkpoint"):
            _ = llm.collective_rpc("reload_weights")
        llm.sleep(level=1)
        llm.wake_up()
    assert answers(llm) == expected


@pytest.mark.feature("Sleep mode")
@pytest.mark.parametrize(
    ("option", "value"),
    [("weights_iterator", []), ("weights_path", "/elsewhere"), ("is_checkpoint_format", False)],
)
def test_refuses_reload_sources(llm: LLM, option: str, value: object) -> None:
    """Reloading reads only the served checkpoint, in its own format."""
    with pytest.raises(Exception, match=f"omit {option}"):
        _ = llm.collective_rpc("reload_weights", kwargs={option: value})
