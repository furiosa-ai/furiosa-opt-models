"""Framework-neutral draft compatibility and cache sizing contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

import furiosa_opt_engine as engine
import furiosa_opt_engine.speculative as engine_speculative
from furiosa_opt_engine.speculative import (
    DraftRequest,
    cache_blocks_for_memory,
    cache_memory_budget,
    draft_blocks_for_shared_budget,
    same_tokenizer_vocabulary,
    validate_draft_decoder,
    validate_draft_vocabulary,
    validate_mask_token,
)


@dataclass(frozen=True, slots=True)
class Tensor:
    """Host stand-in for engine token tensors; it has no device or kernel behavior."""

    values: tuple[int, ...]

    @staticmethod
    def from_dlpack(value: NDArray[np.int32]) -> Tensor:
        return Tensor(tuple(int(cast(np.int32, item)) for item in value))

    def to_hbm(self, _device: object) -> Tensor:
        return self

    def to_host(self) -> Tensor:
        return self

    def numpy(self) -> NDArray[np.int32]:
        return np.asarray(self.values, dtype=np.int32)

    def __getitem__(self, rows: slice) -> Tensor:
        return Tensor(self.values[rows])


@dataclass(frozen=True, slots=True)
class Request:
    """A request recorded by the host stand-in decoder."""

    rows: Sequence[Sequence[int]]
    cached: int
    tokens: Tensor
    keep: engine.decoder.Keep | range


@dataclass(slots=True)
class Cache:
    """Allocation metadata only; the host stand-in stores no cache pages."""

    spec: object
    blocks: int


@dataclass(slots=True)
class Model:
    """A decoder-call recorder returning one zero for each row the session asks it to keep."""

    window: int = 2048
    rows: int = 8
    output: engine.decoder.Output = field(default_factory=lambda: engine.decoder.Output.TOKENS)
    kv_cache: str = "bfloat16"
    cache: SimpleNamespace = field(default_factory=lambda: cache_layout((16,)))
    calls: list[list[Request]] = field(default_factory=list)

    def forward(self, requests: Sequence[Request], _cache: Cache) -> Tensor:
        self.calls.append(list(requests))
        kept = 0
        for request in requests:
            match request.keep:
                case range():
                    kept += len(request.keep)
                case engine.decoder.Keep.ALL:
                    kept += len(request.tokens.values)
                case engine.decoder.Keep.LAST:
                    kept += 1
                case engine.decoder.Keep.NONE:
                    pass
        return Tensor((0,) * kept)


def cache_layout(block_sizes: tuple[int, ...]) -> SimpleNamespace:
    """Paged attention groups with the supplied request block sizes."""
    return SimpleNamespace(
        pools=[SimpleNamespace(block=size, bytes=8) for size in block_sizes],
        groups=[
            SimpleNamespace(
                pool=index,
                kind=engine.decoder.Kind.Paged(),
                layers=range(1),
                window=None,
            )
            for index in range(len(block_sizes))
        ],
    )


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> Model:
    """A host-only decoder call recorder; it does not model or execute an RNGD kernel."""
    fake = SimpleNamespace(
        HostTensor=Tensor,
        decoder=SimpleNamespace(
            Request=Request,
            Cache=Cache,
            Kind=engine.decoder.Kind,
            Keep=engine.decoder.Keep,
            Output=engine.decoder.Output,
        ),
    )
    monkeypatch.setattr(engine_speculative, "engine", fake)
    return Model()


def decoder(block_sizes: tuple[int, ...], window: int = 2048) -> engine.Decoder:
    """A token decoder exposing cache metadata without loading a device model."""
    return cast(
        engine.Decoder,
        cast(
            object,
            SimpleNamespace(
                output=engine.decoder.Output.TOKENS,
                window=window,
                cache=SimpleNamespace(
                    pools=[SimpleNamespace(block=size) for size in block_sizes],
                    groups=[
                        SimpleNamespace(pool=index, kind=engine.decoder.Kind.Paged())
                        for index in range(len(block_sizes))
                    ],
                ),
            ),
        ),
    )


def cache_spec() -> engine.decoder.CacheSpec:
    """Two cache groups with five total layer pages per shared request block."""
    return cast(
        engine.decoder.CacheSpec,
        cast(
            object,
            SimpleNamespace(
                pools=[SimpleNamespace(bytes=8), SimpleNamespace(bytes=4)],
                groups=[
                    SimpleNamespace(pool=0, layers=range(3)),
                    SimpleNamespace(pool=1, layers=range(2)),
                ],
            ),
        ),
    )


@pytest.mark.parametrize(
    ("tokens", "reusable", "count", "reason"),
    [
        ((1,), 0, -1, "draft count must be a nonnegative integer"),
        ((1,), 0, True, "draft count must be a nonnegative integer"),
        ((), 0, 1, "a draft request must contain at least one known token"),
        ((1,), -1, 1, "reusable prefix length must lie within the known tokens"),
        ((1,), 2, 1, "reusable prefix length must lie within the known tokens"),
    ],
    ids=["negative-count", "boolean-count", "empty-tokens", "negative-prefix", "long-prefix"],
)
def test_refuses_invalid_draft_request(
    tokens: tuple[int, ...], reusable: int, count: int, reason: str
) -> None:
    """Reject malformed request boundaries before an adapter sends them to the engine."""
    with pytest.raises(ValueError) as error:
        _ = DraftRequest("sequence", tokens, reusable, ((0,),), count)
    assert str(error.value) == reason


def test_draft_request_snapshots_block_tables() -> None:
    blocks = [[3, 4]]
    request = DraftRequest("sequence", (1,), 0, blocks, 1)

    blocks[0][0] = 9
    assert request.blocks == ((3, 4),)


def test_session_rebuilds_prefix_after_block_table_change(model: Model) -> None:
    session = engine_speculative.DraftSession[str](
        cast(engine.Decoder, cast(object, model)),
        cast(engine.Device, cast(object, None)),
        8,
        4,
    )
    first = DraftRequest("sequence", (1, 2, 3), 0, ((3, 4),), 1)
    changed = DraftRequest("sequence", (1, 2, 3, 4), 3, ((8, 9),), 1)

    _ = session.propose((first,))
    model.calls.clear()
    _ = session.propose((changed,))

    request = model.calls[0][0]
    assert request.cached == 0
    assert request.rows == ((8, 9),)


def test_session_prefills_cold_cache_and_reuses_known_prefix(model: Model) -> None:
    """A cold cache replays input; a warm mask cache reuses only the known prefix."""
    session = engine_speculative.DraftSession[str](
        cast(engine.Decoder, cast(object, model)),
        cast(engine.Device, cast(object, None)),
        8,
        4,
        mask=3,
    )
    first = DraftRequest("sequence", (1, 2, 3), 2, ((3, 4),), 2)

    _ = session.propose((first,))

    assert model.calls[0][0].cached == 0
    model.calls.clear()

    next_step = DraftRequest("sequence", (1, 2, 3, 0), 3, ((3, 4),), 2)
    _ = session.propose((next_step,))

    assert model.calls[0][0].cached == 3
    model.calls.clear()

    limited = DraftRequest("sequence", (1, 2, 3, 0), 0, ((3, 4),), 2)
    _ = session.propose((limited,))

    assert model.calls[0][0].cached == 0


def test_session_rebuilds_after_failed_proposal(
    model: Model, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed draft call cannot leave prefix metadata that skips rebuilding its cache."""
    session = engine_speculative.DraftSession[str](
        cast(engine.Decoder, cast(object, model)),
        cast(engine.Device, cast(object, None)),
        8,
        4,
    )
    first = DraftRequest("sequence", (1, 2, 3), 0, ((3, 4),), 1)
    _ = session.propose((first,))
    model.calls.clear()

    def fail_after_cache_write(*_args: object) -> None:
        raise RuntimeError("draft forward failed")

    with monkeypatch.context() as patch:
        patch.setattr(session, "_draft_sequential", fail_after_cache_write)
        interrupted = DraftRequest("sequence", (1, 2, 3, 0), 3, ((3, 4),), 2)
        with pytest.raises(RuntimeError, match="draft forward failed"):
            _ = session.propose((interrupted,))

    model.calls.clear()
    retry = DraftRequest("sequence", (1, 2, 3, 0, 1), 4, ((3, 4),), 1)
    _ = session.propose((retry,))

    assert model.calls[0][0].cached == 0


def test_session_suspends_weights_and_cache_independently(model: Model) -> None:
    session = engine_speculative.DraftSession[str](
        cast(engine.Decoder, cast(object, model)),
        cast(engine.Device, cast(object, None)),
        8,
        4,
    )
    first = DraftRequest("sequence", (1, 2, 3), 0, ((3, 4),), 1)
    _ = session.propose((first,))
    model.calls.clear()
    next_step = DraftRequest("sequence", (1, 2, 3, 0), 3, ((3, 4),), 1)

    session.release_weights()
    with pytest.raises(RuntimeError, match="must be awake"):
        _ = session.propose((next_step,))
    session.restore_weights(cast(engine.Decoder, cast(object, model)))
    _ = session.propose((next_step,))
    assert model.calls[0][0].cached == 3

    session.release_cache()
    with pytest.raises(RuntimeError, match="must be awake"):
        _ = session.propose((next_step,))
    session.restore_cache()
    model.calls.clear()
    _ = session.propose((next_step,))
    assert model.calls[0][0].cached == 0


def test_zero_drafts_skip_the_decoder(model: Model) -> None:
    session = engine_speculative.DraftSession[str](
        cast(engine.Decoder, cast(object, model)),
        cast(engine.Device, cast(object, None)),
        8,
        4,
    )
    request = DraftRequest("sequence", (1,), 0, ((0,),), 0)

    assert session.propose((request,)) == ((),)
    assert model.calls == []


def test_session_refuses_recurrent_cache_groups(model: Model) -> None:
    model.cache = SimpleNamespace(
        pools=[SimpleNamespace(block=16)],
        groups=[SimpleNamespace(pool=0, kind=engine.decoder.Kind.State())],
    )

    with pytest.raises(ValueError, match="requires paged attention cache groups"):
        _ = engine_speculative.DraftSession[str](
            cast(engine.Decoder, cast(object, model)),
            cast(engine.Device, cast(object, None)),
            8,
            4,
        )


def test_session_clips_drafts_to_remaining_context(model: Model) -> None:
    model.window = 5
    session = engine_speculative.DraftSession[str](
        cast(engine.Decoder, cast(object, model)),
        cast(engine.Device, cast(object, None)),
        8,
        4,
    )
    request = DraftRequest("sequence", (1, 2, 3, 0), 0, ((0,),), 3)

    assert len(session.propose((request,))[0]) == 2


def test_session_clips_drafts_to_every_cache_group_capacity(model: Model) -> None:
    model.cache = cache_layout((16, 8))
    session = engine_speculative.DraftSession[str](
        cast(engine.Decoder, cast(object, model)),
        cast(engine.Device, cast(object, None)),
        8,
        4,
    )
    request = DraftRequest("sequence", (1, 2, 3, 0, 1), 0, ((10, 11), (20,)), 6)

    assert len(session.propose((request,))[0]) == 4
    assert all(call_request.rows == request.blocks for call in model.calls for call_request in call)


def test_validates_vocabulary_cache_layout_and_context() -> None:
    target = decoder((16, 32))
    draft = decoder((16, 32))

    validate_draft_vocabulary(2, 2)
    validate_draft_decoder(target, draft, 2048)

    with pytest.raises(ValueError, match="draft vocabulary size"):
        validate_draft_vocabulary(2, 3)
    with pytest.raises(ValueError, match="block sizes differ"):
        validate_draft_decoder(target, decoder((16, 16)), 2048)
    with pytest.raises(ValueError, match="shorter than required context"):
        validate_draft_decoder(target, decoder((16, 32), window=1024), 2048)
    with pytest.raises(ValueError, match="paged attention cache groups"):
        validate_draft_decoder(
            target,
            cast(
                engine.Decoder,
                cast(
                    object,
                    SimpleNamespace(
                        output=engine.decoder.Output.TOKENS,
                        window=2048,
                        cache=SimpleNamespace(
                            pools=[SimpleNamespace(block=16)],
                            groups=[SimpleNamespace(pool=0, kind=engine.decoder.Kind.State())],
                        ),
                    ),
                ),
            ),
            2048,
        )


def test_validates_tokenizer_mappings_and_mask_ids() -> None:
    assert same_tokenizer_vocabulary({"a": 0, "b": 1}, {"a": 0, "b": 1})
    assert not same_tokenizer_vocabulary({"a": 0, "b": 1}, {"b": 0, "a": 1})
    assert validate_mask_token(None, 3) is None
    assert validate_mask_token(0, 3) == 0
    assert validate_mask_token(2, 3) == 2

    for invalid in (True, -1, 3, "2"):
        with pytest.raises(ValueError, match="mask token ID"):
            _ = validate_mask_token(invalid, 3)


def test_cache_sizing_accounts_for_all_groups_and_i32_offsets() -> None:
    cache = cache_spec()

    assert cache_blocks_for_memory(900, cache) == 28
    assert cache_blocks_for_memory(2**40, cache) == 2**31 // 8
    assert draft_blocks_for_shared_budget(100, 16, cache) == 33


def test_memory_budget_respects_headroom_and_workspace() -> None:
    assert cache_memory_budget(101, 101, 0.8) == 80
    assert cache_memory_budget(101, 101, 0.8, workspace=10) == 70

    for available, capacity, workspace in [(-1, 100, 0), (100, -1, 0), (100, 100, -1)]:
        with pytest.raises(ValueError, match="must be nonnegative"):
            _ = cache_memory_budget(available, capacity, 0.8, workspace=workspace)
