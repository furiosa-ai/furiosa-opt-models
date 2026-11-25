"""SGLang draft placement selection and cache sizing, without opening an NPU device."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import furiosa_opt_engine as engine
from furiosa_opt_engine.placement import PlacedDevice
from furiosa_opt_sglang.decoder import Decoder, Location
from furiosa_opt_sglang.paging import cache_blocks
from furiosa_opt_sglang.platform import DRAFT_PLACEMENT_ENV, DraftPlacement, Refused, Standalone
from furiosa_opt_sglang.speculative import standalone
from furiosa_opt_sglang.speculative.standalone import Draft


def decoder(
    device: object,
    blocks: int,
    calls: list[tuple[int, tuple[Decoder, ...]]],
    placement: engine.Placement,
) -> Decoder:
    """A decoder budget with a recorded block-count request."""

    def count(memory: int, *others: Decoder) -> int:
        calls.append((memory, others))
        return blocks

    return cast(
        Decoder,
        cast(object, SimpleNamespace(device=device, placement=placement, blocks=count)),
    )


def draft_for(decoder: Decoder) -> Draft:
    """A draft wrapper around a decoder without loading an NPU model."""
    draft = object.__new__(Draft)
    draft._decoder = decoder
    return draft


def causal_model() -> SimpleNamespace:
    """Model metadata the shared compatibility check reads after draft load."""
    return SimpleNamespace(
        parallel=2,
        output=engine.decoder.Output.TOKENS,
        window=2048,
        cache=SimpleNamespace(
            pools=[SimpleNamespace(block=16)],
            groups=[
                SimpleNamespace(pool=0, kind=engine.decoder.Kind.Paged()),
            ],
        ),
    )


def test_sizes_shared_draft_against_target_device() -> None:
    """A colocated draft retains the existing combined target-and-draft budget."""
    device = object()
    placement = engine.Placement([0], range(8))
    calls: list[tuple[int, tuple[Decoder, ...]]] = []
    target = decoder(device, 7, calls, placement)
    draft_decoder = decoder(device, 5, [], placement)
    draft = draft_for(draft_decoder)

    count = cache_blocks(target, 600, 0.8, draft)

    assert count == 7
    assert calls == [(600, (draft_decoder,))]


@pytest.mark.parametrize(
    ("target_blocks", "expected"), [(12, 7), (4, 4)], ids=["draft-limit", "target-limit"]
)
def test_caps_disjoint_draft_by_each_device(target_blocks: int, expected: int) -> None:
    """A disjoint draft uses its own reserved-memory budget, then the tighter device limit."""
    calls: list[tuple[int, tuple[Decoder, ...]]] = []
    target = decoder(object(), target_blocks, calls, engine.Placement([0], range(8)))
    draft_device = SimpleNamespace(capacity=1000, available=900)
    draft_calls: list[tuple[int, tuple[Decoder, ...]]] = []
    draft_decoder = decoder(draft_device, 7, draft_calls, engine.Placement([1], range(8)))
    draft = draft_for(draft_decoder)

    count = cache_blocks(target, 800, 0.8, draft)

    assert count == expected
    assert calls == [(800, ())]
    assert draft_calls == [(700, ())]


def test_sizes_cache_from_saved_spec_after_weights_are_released() -> None:
    """Cache planning uses saved layout metadata while the memory saver released model weights."""
    spec = SimpleNamespace(
        pools=[SimpleNamespace(bytes=8)],
        groups=[SimpleNamespace(pool=0, layers=range(2))],
    )
    released = cast(
        Decoder,
        cast(object, SimpleNamespace(model=None, cache_spec=spec)),
    )

    assert Decoder.blocks(released, 64) == 4


@pytest.mark.parametrize(
    ("chips", "opened", "overlap", "tokenizer_mismatch", "draft_vocabulary"),
    [
        ((0, 1), False, False, False, 3),
        ((2, 3), True, False, False, 3),
        ((1, 2), False, True, False, 3),
        ((0, 1), False, False, True, 3),
        ((2, 3), False, False, False, 4),
    ],
    ids=[
        "explicit-target-placement",
        "disjoint-placement",
        "partial-overlap",
        "mismatched-tokenizer",
        "mismatched-config-vocabulary",
    ],
)
def test_validates_tokenizer_and_selected_placement_before_draft_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    chips: tuple[int, int],
    opened: bool,
    overlap: bool,
    tokenizer_mismatch: bool,
    draft_vocabulary: int,
) -> None:
    """Reject incompatible tokenizer or vocabulary before opening/loading the draft."""
    target_device = object()
    target_placement = engine.Placement([0, 1], range(8))
    target_selected = cast(
        PlacedDevice,
        cast(object, SimpleNamespace(device=target_device, placement=target_placement)),
    )
    target = cast(
        Decoder,
        cast(
            object,
            SimpleNamespace(
                model=causal_model(),
                device=target_device,
                placement=target_placement,
                selected=target_selected,
                vocabulary=3,
                pool=SimpleNamespace(block=16),
            ),
        ),
    )
    _ = (tmp_path / "config.json").write_text(f'{{"vocab_size":{draft_vocabulary}}}')

    def checkout(_directory: str, _revision: str | None) -> str:
        return str(tmp_path)

    vocabularies = {
        "draft": {"a": 0, "b": 1, "c": 2},
        "target": {"a": 0, "b": 1, "c": 2},
    }

    def get_tokenizer(path: str, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(get_vocab=lambda: vocabularies[path])

    mapped_calls: list[Location] = []

    def mapped(location: Location) -> object:
        mapped_calls.append(location)
        return object()

    monkeypatch.setattr(standalone, "checkout", checkout)
    monkeypatch.setattr(
        standalone,
        "get_serving",
        lambda: SimpleNamespace(
            tokenizer_mode="auto", tokenizer_backend="auto", tokenizer_path="target"
        ),
    )
    monkeypatch.setattr(
        standalone,
        "get_model",
        lambda: SimpleNamespace(trust_remote_code=False, revision=None, kv_cache_dtype="auto"),
    )
    monkeypatch.setattr(
        standalone,
        "get_tokenizer",
        get_tokenizer,
    )
    monkeypatch.setattr(Location, "mapped", mapped)

    class LoadOptions:
        def chips(self, count: int) -> LoadOptions:
            assert count == 2
            return self

        def parse(self, _checkpoint: object) -> None:
            pass

    monkeypatch.setattr(engine, "LoadOptions", LoadOptions)
    selected_device = object()
    opened_placements: list[engine.Placement] = []

    def open_device(placement: engine.Placement) -> object:
        opened_placements.append(placement)
        return selected_device

    monkeypatch.setattr(engine, "Device", open_device)
    requested_placement = engine.Placement(list(chips), range(8))
    options = cast(
        Standalone,
        cast(
            object,
            SimpleNamespace(
                speculative_draft_model_path="draft",
                speculative_draft_model_revision=None,
                speculative_draft_kv_cache_dtype=None,
                draft_placement=DraftPlacement(chips=chips, pes=(0, 8)),
            ),
        ),
    )
    if tokenizer_mismatch:
        vocabularies["draft"] = {"a": 0, "c": 1, "b": 2}
        with pytest.raises(Refused, match="target's token-to-id mapping"):
            _ = Draft(options, target, context=1024)
        assert not opened_placements
        assert not mapped_calls
        return
    if draft_vocabulary != target.vocabulary:
        with pytest.raises(Refused, match="does not match target vocabulary size"):
            _ = Draft(options, target, context=1024)
        assert not opened_placements
        assert not mapped_calls
        return
    loaded = cast(
        Decoder,
        cast(
            object,
            SimpleNamespace(
                model=causal_model(),
                vocabulary=3,
                pool=SimpleNamespace(block=16),
            ),
        ),
    )
    loads: list[PlacedDevice] = []

    def load(
        _cls: type[Decoder],
        _checkpoint: object,
        _options: object,
        selected: PlacedDevice,
        _vocabulary: int,
    ) -> Decoder:
        loads.append(selected)
        return loaded

    monkeypatch.setattr(Decoder, "load", classmethod(load))

    if overlap:
        with pytest.raises(Refused) as error:
            _ = Draft(options, target, context=1024)
        assert error.value.option == DRAFT_PLACEMENT_ENV
        assert not opened_placements
        assert not loads
        return

    _ = Draft(options, target, context=1024)

    assert len(loads) == 1
    assert loads[0].placement == requested_placement
    if opened:
        assert loads[0].device is selected_device
        assert opened_placements == [requested_placement]
    else:
        assert loads[0].device is target_device
        assert not opened_placements
