import json
from fnmatch import fnmatchcase
from pathlib import Path
from shutil import copyfile

import pytest
from vllm_omni.diffusion.data import OmniDiffusionConfig

from furiosa_opt_vllm.omni.pipeline import build


@pytest.fixture
def remote_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, list[str]]:
    """Replace only Hub transfer; keep framework configuration, resolution and engine parsing."""
    from vllm_omni.model_executor.model_loader import weight_utils

    repository = tmp_path / "repository"
    snapshot = tmp_path / "hub" / "snapshots" / "test-revision"
    files = {
        "model_index.json": json.dumps({"_class_name": "MiniMaxH3Pipeline"}),
        "FL2VA/model_index.json": json.dumps({"_class_name": "MiniMaxH3Pipeline"}),
        "Ref2VA/model_index.json": json.dumps({"_class_name": "MiniMaxH3Pipeline"}),
        "FL2VA/transformer/model.safetensors.index.json": json.dumps(
            {"weight_map": {"weight": "model-00001-of-00001.safetensors"}}
        ),
        "FL2VA/transformer/model-00001-of-00001.safetensors": "unfetched weights",
    }
    for name, content in files.items():
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_text(content)
    transferred: list[str] = []

    def download(
        repo_id: str,
        *,
        allow_patterns: str | list[str],
        ignore_patterns: str | list[str] | None = None,
        revision: str | None = None,
        **_: object,
    ) -> str:
        assert repo_id == "test-org/checkpoint"
        assert revision == "test-revision"
        allowed = [allow_patterns] if isinstance(allow_patterns, str) else allow_patterns
        ignored = [ignore_patterns] if isinstance(ignore_patterns, str) else ignore_patterns or []
        snapshot.mkdir(parents=True, exist_ok=True)
        for source in repository.rglob("*"):
            if not source.is_file():
                continue
            name = source.relative_to(repository).as_posix()
            if not any(fnmatchcase(name, pattern) for pattern in allowed) or any(
                fnmatchcase(name, pattern) for pattern in ignored
            ):
                continue
            assert name.endswith(".json") and not name.endswith(".safetensors.index.json"), (
                f"refused load attempted bulk transfer: {name}"
            )
            destination = snapshot / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            _ = copyfile(source, destination)
            transferred.append(name)
        return str(snapshot)

    monkeypatch.setattr(weight_utils, "snapshot_download", download)
    return repository, snapshot, transferred


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        (
            {"parallel_config": {"tensor_parallel_size": 2}},
            "one Omni worker drives the H3 pipeline's four RNGD chips",
        ),
        ({"enable_sleep_mode": True}, "H3 pipeline sleep/wake is not implemented"),
        (
            {"diffusion_offload_config": {"mode": "module", "components": ["dit"]}},
            "the H3 engine owns encoder streaming and resident decoder weights",
        ),
        (
            {"quantization_config": "fp8"},
            "H3 loads its checkpoint precision without LoRA adapters",
        ),
        (
            {"lora_path": "test-org/adapter"},
            "H3 loads its checkpoint precision without LoRA adapters",
        ),
        (
            {"cache_backend": "cache_dit"},
            "H3 cache backend 'cache_dit' is not implemented",
        ),
        ({"enforce_eager": False}, "the H3 engine requires enforce_eager=True"),
    ],
    ids=["workers", "sleep", "offload", "quantization", "lora", "cache", "eager"],
)
def test_refuses_remote_options_before_weights(
    remote_checkpoint: tuple[Path, Path, list[str]], overrides: dict[str, object], reason: str
) -> None:
    config = OmniDiffusionConfig(
        **{
            "model": "test-org/checkpoint",
            "model_class_name": "MiniMaxH3Pipeline",
            "revision": "test-revision",
            "enforce_eager": True,
            **overrides,
        }
    )
    with pytest.raises(NotImplementedError) as refused:
        build(od_config=config)
    assert str(refused.value) == reason
    _, _, transferred = remote_checkpoint
    assert transferred  # Resolution may fetch metadata, but must never fetch weights.


@pytest.mark.parametrize("model_class", ["FluxPipeline", "MiniMaxH3Pipeline"])
@pytest.mark.parametrize("cached_index", [False, True], ids=["empty-cache", "partial-cache"])
def test_preserves_remote_architecture_refusal(
    remote_checkpoint: tuple[Path, Path, list[str]], model_class: str, cached_index: bool
) -> None:
    repository, snapshot, transferred = remote_checkpoint
    if cached_index:
        index = "FL2VA/transformer/model.safetensors.index.json"
        target = snapshot / index
        target.parent.mkdir(parents=True)
        _ = copyfile(repository / index, target)
    # Both a root pipeline and a framework-resolved partition must keep the engine's reason.
    for name in ("model_index.json", "FL2VA/model_index.json"):
        _ = (repository / name).write_text(json.dumps({"_class_name": "FluxPipeline"}))
    config = OmniDiffusionConfig(
        model="test-org/checkpoint",
        model_class_name=model_class,
        revision="test-revision",
        enforce_eager=True,
    )
    with pytest.raises(NotImplementedError) as refused:
        build(od_config=config)
    assert str(refused.value) == "FluxPipeline: unsupported, no model type is implemented"
    assert transferred


def test_refuses_pipelines(tmp_path: Path) -> None:
    """vLLM-Omni's load of a diffusion pipeline is refused before a weight is read, with the
    engine's reason, which names the pipeline's architecture."""
    _ = (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "FluxPipeline"}))
    config = OmniDiffusionConfig(model=str(tmp_path), model_class_name="FluxPipeline")
    with pytest.raises(NotImplementedError) as refused:
        build(od_config=config)
    assert str(refused.value) == "FluxPipeline: unsupported, no model type is implemented"
