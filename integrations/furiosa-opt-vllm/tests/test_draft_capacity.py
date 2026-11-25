"""Draft placement and cache-budget decisions, without opening an NPU device."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, NoReturn, cast

import pytest

import furiosa_opt_engine as engine
from furiosa_opt_engine.placement import PlacedDevice
from furiosa_opt_vllm import checkpoint
from furiosa_opt_vllm.additional import Placement, Section
from furiosa_opt_vllm.admission import Draft, Refused, speculative
from furiosa_opt_vllm.decoder import Decoder
from furiosa_opt_vllm.drafting import Drafter, _select_device  # pyright: ignore[reportPrivateUsage]

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def target(chips: list[int]) -> Decoder:
    """A target placement and device identity for placement decisions."""
    device = object()
    placement = engine.Placement(chips, range(8))
    selected = cast(
        PlacedDevice,
        cast(object, SimpleNamespace(device=device, placement=placement)),
    )
    return cast(
        Decoder,
        cast(
            object,
            SimpleNamespace(
                selected=selected,
                device=device,
                placement=placement,
            ),
        ),
    )


def draft(placement: Placement | None, *, parallelism: int) -> Draft:
    """The explicit placement a draft model requests."""
    parallel = SimpleNamespace(tensor_parallel_size=parallelism)
    return cast(Draft, cast(object, SimpleNamespace(placement=placement, parallel=parallel)))


@pytest.mark.parametrize("method", [None, "ngram", "suffix"])
def test_refuses_draft_placement_without_draft_model(method: str | None) -> None:
    """A draft-only placement cannot silently do nothing for another method."""
    settings = None if method is None else SimpleNamespace(method=method)
    config = cast(
        "VllmConfig",
        cast(object, SimpleNamespace(speculative_config=settings, model_config=None)),
    )
    section = Section(draft_placement=Placement(chips=(3,), pes=(0, 8)))

    with pytest.raises(Refused) as error:
        _ = speculative(config, section)

    assert error.value.option == "additional_config.furiosa.draft_placement"
    assert error.value.requirement == "requires speculative_config.method=draft_model"


def test_reuses_target_device_for_the_same_placement() -> None:
    """An explicit match reuses the target's open device and shares its HBM budget."""
    loaded_target = target([1, 2])

    selected = _select_device(
        draft(Placement(chips=(1, 2), pes=(0, 8)), parallelism=2), loaded_target
    )

    assert selected.device is loaded_target.device
    assert selected.placement == loaded_target.placement


@pytest.mark.parametrize(
    "placement",
    [
        Placement(chips=(2, 3), pes=(0, 8)),
        Placement(chips=(1, 2), pes=(0, 4)),
    ],
    ids=["partial-chips", "different-pes"],
)
def test_refuses_partial_chip_overlap(placement: Placement) -> None:
    """An overlapping placement that differs by chip or PE cannot open a second device."""
    loaded_target = target([1, 2])
    with pytest.raises(Refused) as error:
        _ = _select_device(draft(placement, parallelism=2), loaded_target)

    assert error.value.option == "additional_config.furiosa.draft_placement"
    assert repr(loaded_target.placement) in error.value.requirement
    assert (
        repr(engine.Placement(list(placement.chips), range(*placement.pes)))
        in error.value.requirement
    )


def test_opens_a_device_for_disjoint_chips(monkeypatch: pytest.MonkeyPatch) -> None:
    """A disjoint placement opens its own device rather than reusing the target's."""
    opened = object()
    placements: list[engine.Placement] = []

    def open_device(placement: engine.Placement) -> object:
        placements.append(placement)
        return opened

    monkeypatch.setattr(engine, "Device", open_device)
    loaded_target = target([1, 2])

    selected = _select_device(
        draft(Placement(chips=(3,), pes=(0, 8)), parallelism=1), loaded_target
    )

    assert selected.device is opened
    assert selected.placement.chips == [3]
    assert placements[0].chips == [3]


def test_refuses_draft_placement_with_wrong_chip_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disjoint placement needs one chip per draft tensor-parallel rank."""

    def fail_to_open(_placement: engine.Placement) -> NoReturn:
        pytest.fail("opened device")

    monkeypatch.setattr(engine, "Device", fail_to_open)
    loaded_target = target([1, 2])

    with pytest.raises(Refused) as error:
        _ = _select_device(draft(Placement(chips=(3,), pes=(0, 8)), parallelism=2), loaded_target)

    assert error.value.option == "additional_config.furiosa.draft_placement"
    assert "must provide 2 chips; got 1" in error.value.requirement


def drafter(
    *, shared_device: bool, available: int, capacity: int = 1000, utilization: float = 0.8
) -> Drafter:
    """A drafter with two cache groups whose total per-block size is 32 bytes."""
    cache = cache_spec()
    target_device = SimpleNamespace(available=available, capacity=capacity)
    device = (
        target_device if shared_device else SimpleNamespace(available=available, capacity=capacity)
    )
    target_placement = engine.Placement([0], range(8))
    placement = target_placement if shared_device else engine.Placement([1], range(8))
    selected = cast(
        PlacedDevice,
        cast(object, SimpleNamespace(device=device, placement=placement)),
    )
    return Drafter(
        cast(engine.Decoder, cast(object, SimpleNamespace(cache=cache))),
        selected,
        target_placement,
        cast(Draft, cast(object, SimpleNamespace(memory_utilization=utilization))),
        checkpoint.Location("", {}),
        engine.LoadOptions(),
    )


def cache_spec() -> SimpleNamespace:
    """Two pools with five total layer pages per block."""
    return SimpleNamespace(
        pools=[SimpleNamespace(bytes=8), SimpleNamespace(bytes=4)],
        groups=[
            SimpleNamespace(pool=0, layers=range(3)),
            SimpleNamespace(pool=1, layers=range(2)),
        ],
    )


def test_sizes_shared_cache_budgets_together() -> None:
    """Colocated caches split resolved target capacity, including an explicit cache-size limit."""
    assert drafter(shared_device=True, available=100).capacity(100, 16) == 33


def test_caps_separate_cache_by_its_free_memory() -> None:
    """A separate placement leaves the configured 20% outside its cache budget."""
    assert drafter(shared_device=False, available=900).capacity(100, 16) == 21


def test_separate_cache_respects_headroom_larger_than_available_memory() -> None:
    """The draft cache gets no blocks when utilization reserve exceeds free memory."""
    assert drafter(shared_device=False, available=100).capacity(100, 16) == 0
