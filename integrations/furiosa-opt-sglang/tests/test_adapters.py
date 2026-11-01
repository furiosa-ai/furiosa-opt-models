"""SGLang's `lora_manager` builds each adapter revision once on host, under the ID SGLang gives its
load, and refuses what SGLang or the engine refuses before any of it is built."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, cast, final

import pytest
import torch
from safetensors.torch import save_file
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.runtime_context import get_parallel

import furiosa_opt_engine as engine
from furiosa_opt_sglang.decoder import Decoder
from furiosa_opt_sglang.lora import Adapters
from furiosa_opt_sglang.lora.options import Lora

if TYPE_CHECKING:
    from transformers import PretrainedConfig

# The engine adapter type of a loaded decoder; the engine double below never reads it.
SPEC: Final = cast("engine.decoder.AdapterSpec", object())
VOCABULARY: Final = 64
# The device bytes the cache leaves to calls, and those a device adapter takes.
HEADROOM: Final = 1 << 20
SIZE: Final = 1 << 16
# What a device adapter holds after a load into it failed partway.
TORN: Final = "torn"
FACTOR: Final = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
CONFIG: Final[dict[str, object]] = {
    "peft_type": "LORA",
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def options(**changes: object) -> Lora:
    """SGLang's LoRA options with LoRA on and nothing else set, but `changes`."""
    fields: dict[str, object] = {
        "enable_lora_overlap_loading": False,
        "experts_shared_outer_loras": False,
        "lora_use_virtual_experts": False,
        "lora_target_modules": None,
        "lora_eviction_policy": "lru",
        "max_lora_rank": None,
        "max_loras_per_batch": 8,
    }
    return Lora.model_validate(fields | changes)


def peft(directory: Path, **changes: object) -> Path:
    """A PEFT adapter directory of `CONFIG` with `changes`, a `None` value dropping its field."""
    directory.mkdir()
    config = {key: value for key, value in (CONFIG | changes).items() if value is not None}
    _ = (directory / "adapter_config.json").write_text(json.dumps(config))
    save_file({FACTOR: torch.zeros(8, 16)}, directory / "adapter_model.safetensors")
    return directory


def ref(name: str, path: Path | str) -> LoRARef:
    """SGLang's record of a load of the adapter `name` at `path`, under a new ID."""
    return LoRARef(lora_name=name, lora_path=str(path))


@dataclass(frozen=True)
class Host:
    """A revision in host memory, which nothing changes, of a checkpoint or of no updates."""

    checkpoint: engine.Checkpoint | None


@dataclass
class Device:
    """The engine's storage the host tests stand in for: the revisions built on host, the device
    adapters taken, the revisions written into them, and the checkpoint directory of each."""

    available: int
    hosts: list[Host] = field(default_factory=list)
    adapters: list[Adapter] = field(default_factory=list)
    writes: list[tuple[Adapter, Host]] = field(default_factory=list)
    sources: list[tuple[engine.Checkpoint, str]] = field(default_factory=list)
    # Whether the next load into a device adapter fails partway.
    failing: bool = False

    def name(self, revision: object) -> str | None:
        """The directory of the checkpoint `revision` holds; None for no updates."""
        if revision == TORN:
            return TORN
        source = revision.checkpoint if isinstance(revision, Host) else revision
        if source is None:
            return None
        return next(name for checkpoint, name in self.sources if checkpoint is source)


@final
class Adapter:
    """A device adapter of the double: its memory taken once, its revision replaced by a load."""

    def __init__(self, device: Device) -> None:
        device.available -= SIZE
        self.device = device
        self.revision: object = None

    def load(self, host: Host) -> None:
        self.device.writes.append((self, host))
        if self.device.failing:
            self.revision = TORN
            raise RuntimeError("the device fails")
        self.revision = host


@pytest.fixture
def device(monkeypatch: pytest.MonkeyPatch) -> Device:
    """Room for many device adapters past the calls' headroom, and the engine types over it."""
    device = Device(HEADROOM + 64 * SIZE)
    checkpoint = engine.Checkpoint

    def read(path: str | Path, config: dict[str, object] | None = None) -> engine.Checkpoint:
        device.sources.append((checkpoint(path, config), Path(path).name))
        return device.sources[-1][0]

    def host(spec: object, source: object) -> Host:
        assert spec is SPEC, "a revision is of its decoder's adapter type"
        assert isinstance(source, checkpoint), source
        device.hosts.append(Host(source))
        return device.hosts[-1]

    def adapter(spec: object) -> Adapter:
        assert spec is SPEC, "a device adapter is of its decoder's adapter type"
        device.adapters.append(Adapter(device))
        return device.adapters[-1]

    monkeypatch.setattr(engine, "Checkpoint", read)
    monkeypatch.setattr(engine.decoder, "HostAdapter", host, raising=False)
    monkeypatch.setattr(engine.decoder, "Adapter", adapter)
    return device


@pytest.fixture
def manager(device: Device) -> Callable[..., Adapters]:
    """Managers built where the runner builds one, before the cache sizes, in a server of one
    tensor-parallel rank, under LoRA options with the given changes."""
    _ = device

    def make(**changes: object) -> Adapters:
        # The decoder fields the manager reads, and the model configuration SGLang holds.
        config = SimpleNamespace(num_hidden_layers=2, hidden_size=16, vocab_size=VOCABULARY)
        decoder = SimpleNamespace(model=SimpleNamespace(adapters=SPEC), vocabulary=VOCABULARY)
        with get_parallel().override(tp_size=1, tp_rank=0, attn_tp_size=1):
            return Adapters(
                cast("Decoder", cast("object", decoder)),
                options(**changes),
                cast("PretrainedConfig", cast("object", config)),
            )

    return make


def held(device: Device, adapters: tuple[engine.decoder.Adapter, ...]) -> list[str | None]:
    """The adapter each listed device adapter holds a revision of."""
    return [device.name(cast("Adapter", cast("object", listed)).revision) for listed in adapters]


@pytest.fixture
def adapter(tmp_path: Path) -> Path:
    return peft(tmp_path / "sky")


def test_loads_a_peft_directory(
    manager: Callable[..., Adapters], device: Device, adapter: Path
) -> None:
    """A load builds the adapter on host once and names it with every loaded adapter, as SGLang's
    does; a step's calls then read it from an entry."""
    adapters = manager()
    sky = ref("sky", adapter)
    result = adapters.load_lora_adapter(sky)
    assert (result.success, result.loaded_adapters) == (True, {"sky": str(adapter)})
    assert len(device.hosts) == 1
    assert held(device, adapters.bind([sky.lora_id]).adapters) == ["sky"]


@pytest.mark.parametrize(
    ("changes", "setting", "dropped", "reason"),
    [
        ({"r": 16}, {"max_lora_rank": 8}, None, "rank 16 exceeds --max-lora-rank 8"),
        (
            {"target_modules": ["q_proj", "gate_proj"]},
            {"lora_target_modules": {"q_proj", "k_proj", "v_proj", "o_proj"}},
            None,
            "['gate_up_proj'] are not in --lora-target-modules",
        ),
        ({"r": None}, {}, None, "configuration's r: Field required"),
        ({}, {}, "adapter_model.safetensors", "reads adapter_model.safetensors"),
    ],
    ids=["rank", "targets", "config", "weights"],
)
def test_refuses_before_building(
    manager: Callable[..., Adapters],
    device: Device,
    tmp_path: Path,
    changes: dict[str, object],
    setting: dict[str, object],
    dropped: str | None,
    reason: str,
) -> None:
    """A load SGLang's manager refuses, or one of weights the engine does not read, leaves the
    loaded adapters as they were and builds nothing."""
    adapters = manager(**setting)
    sky = peft(tmp_path / "sky")
    _ = adapters.load_lora_adapter(ref("sky", sky))
    other = peft(tmp_path / "other", **changes)
    if dropped is not None:
        (other / dropped).unlink()
    result = adapters.load_lora_adapter(ref("other", other))
    assert not result.success
    assert reason in str(result.error_message), result.error_message
    assert result.loaded_adapters == {"sky": str(sky)}
    assert len(device.hosts) == 1


def test_refuses_a_loaded_name(
    manager: Callable[..., Adapters], device: Device, adapter: Path
) -> None:
    """A second load of a loaded name is refused in the wording SGLang's tokenizer tells a
    concurrent reload by."""
    adapters = manager()
    _ = adapters.load_lora_adapter(ref("sky", adapter))
    result = adapters.load_lora_adapter(ref("sky", adapter))
    assert not result.success and "already loaded" in str(result.error_message)
    assert result.loaded_adapters == {"sky": str(adapter)}
    assert len(device.hosts) == 1


def test_refuses_adapters_that_add_tokens(
    manager: Callable[..., Adapters], device: Device, adapter: Path
) -> None:
    """Tokens past the base vocabulary are refused; SGLang drops listed tokens already in it."""
    tokens = {"<inside>": VOCABULARY - 1, "<outside>": VOCABULARY}
    _ = (adapter / "added_tokens.json").write_text(json.dumps(tokens))
    result = manager().load_lora_adapter(ref("sky", adapter))
    assert not result.success and "adds tokens" in str(result.error_message)
    assert device.hosts == []


def test_never_replaces_the_revision_of_an_id(
    manager: Callable[..., Adapters], device: Device, tmp_path: Path
) -> None:
    """SGLang mints an ID for each load and keys cached prefixes by it, so a second load under a
    held ID is a broken invariant, not a replacement."""
    adapters = manager()
    sky = ref("sky", peft(tmp_path / "sky"))
    _ = adapters.load_lora_adapter(sky)
    again = LoRARef(lora_id=sky.lora_id, lora_name="other", lora_path=str(peft(tmp_path / "b")))
    with pytest.raises(AssertionError):
        _ = adapters.load_lora_adapter(again)
    assert held(device, adapters.bind([sky.lora_id]).adapters) == ["sky"]


def test_reloads_as_a_new_revision(
    manager: Callable[..., Adapters], device: Device, adapter: Path
) -> None:
    """A name loaded again after its unload is a new revision under a new ID; its old ID names
    nothing a call can bind."""
    adapters = manager()
    first, second = ref("sky", adapter), ref("sky", adapter)
    _ = adapters.load_lora_adapter(first)
    assert adapters.unload_lora_adapter(first).loaded_adapters == {}
    assert adapters.load_lora_adapter(second).success
    assert held(device, adapters.bind([second.lora_id]).adapters) == ["sky"]
    assert device.hosts[1] is not device.hosts[0]
    with pytest.raises(KeyError):
        _ = adapters.bind([first.lora_id])


def test_admits_adapters_of_every_module_under_named_targets(
    manager: Callable[..., Adapters], device: Device, tmp_path: Path
) -> None:
    """An adapter that adapts `all-linear` loads under named `--lora-target-modules`, as SGLang's
    memory pool admits it; the engine still refuses modules its terms lack."""
    adapters = manager(lora_target_modules={"q_proj"})
    every = peft(tmp_path / "every", target_modules="all-linear")
    assert adapters.load_lora_adapter(ref("every", every)).success
    assert len(device.hosts) == 1
