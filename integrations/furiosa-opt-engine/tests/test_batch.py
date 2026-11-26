"""Framework-neutral framing, call planning and capacity-bounded row batching."""

import pickle
from contextlib import contextmanager
from typing import cast

import pytest

import furiosa_opt_engine.batch as batch


def test_call_preserves_chunks_and_kept_order() -> None:
    rows = (batch.Row("a", 5, False), batch.Row("a", 6, True), batch.Row("b", 0, False))
    call = batch.Call[str].plan(rows, every=False)
    assert call.requests == (
        batch.Request("a", 5, (0, 1), True),
        batch.Request("b", 0, (2,), False),
    )
    assert call.kept == (1,)


def test_call_keeps_all_rows_for_pooling() -> None:
    call = batch.Call[int].plan((batch.Row(0, 4, False), batch.Row(1, 0, True)), every=True)
    assert call.kept == (0, 1)


def test_batch_snapshots_framework_tokens() -> None:
    tokens = [1, 2, 3]
    parsed = batch.Batch.parse(tokens, [2, 1], [7, 1])
    tokens[0] = 9
    assert parsed.tokens == ((1, 2), (3,))
    assert parsed.cached == (5, 0)


def test_batch_refuses_wrong_framing() -> None:
    with pytest.raises(batch.Framing) as refused:
        _ = batch.Batch.parse([1], [2], [2])
    assert refused.value.tokens == 1
    assert refused.value.scheduled == 2


@pytest.mark.parametrize(("lengths", "totals"), [((0, 1), (1, 2)), ((-1, 2), (1, 2)), ((1,), (0,))])
def test_batch_refuses_invalid_lengths(lengths: tuple[int, ...], totals: tuple[int, ...]) -> None:
    with pytest.raises(batch.Framing):
        _ = batch.Batch.parse([1] * sum(lengths), lengths, totals)


def test_refusal_survives_framework_error_handling() -> None:
    """Context managers rewrite a raised refusal's traceback, and process pools pickle it."""

    @contextmanager
    def scope():
        yield

    with pytest.raises(batch.Framing) as refused, scope():
        _ = batch.Batch.parse([1], [2], [2])
    # pickle restores an untyped object.
    restored = cast("batch.Framing", pickle.loads(pickle.dumps(refused.value)))
    assert (restored.tokens, restored.scheduled) == (1, 2)


@pytest.mark.parametrize(("ids", "vocabulary"), [([-1], 32), ([32], 32), ([0], None)])
def test_operand_refuses_ids_outside_the_vocabulary(ids: list[int], vocabulary: int | None) -> None:
    """An engine call reads its token IDs unchecked, so its operand holds only vocabulary IDs."""
    with pytest.raises(ValueError, match="vocabulary"):
        _ = batch.operand(ids, vocabulary)


def test_operand_holds_vocabulary_ids_as_int32() -> None:
    assert batch.operand([0, 31], 32).tolist() == [0, 31]
    assert batch.operand([0, 31], 32).dtype == "int32"


@pytest.mark.parametrize(
    ("lengths", "rows", "first", "second"),
    [
        ((35, 18, 2, 0), 8, ((0, 3), (1, 3), (2, 2)), ((0, 4), (1, 4))),
        ((2, 3, 4), 1, ((0, 1),), ((1, 1),)),
    ],
)
def test_batches_sequence_rows_within_decoder_capacity(
    lengths: tuple[int, ...],
    rows: int,
    first: tuple[tuple[int, int], ...],
    second: tuple[tuple[int, int], ...],
) -> None:
    """Batch work across requests without duplicating a sequence or exceeding compiled rows."""
    consumed = [[] for _ in lengths]
    remaining = sum(lengths)
    batches = tuple(batch.chunks(lengths, rows))
    assert tuple((chunk.sequence, chunk.length) for chunk in batches[0]) == first
    assert tuple((chunk.sequence, chunk.length) for chunk in batches[1]) == second
    for row_batch in batches:
        used = sum(chunk.length for chunk in row_batch)
        assert used == min(rows, remaining)
        assert len({chunk.sequence for chunk in row_batch}) == len(row_batch)
        for chunk in row_batch:
            consumed[chunk.sequence].extend(range(chunk.start, chunk.start + chunk.length))
        remaining -= used

    assert consumed == [list(range(length)) for length in lengths]
    assert remaining == 0


def test_chunks_rotates_spare_rows_between_sequences() -> None:
    """Carry the round-robin turn across calls when requests leave spare decoder rows."""
    batches = tuple(batch.chunks((20, 20, 20), 4))[:3]
    allocated = tuple(
        sum(chunk.length for row_batch in batches for chunk in row_batch if chunk.sequence == seq)
        for seq in range(3)
    )

    assert allocated == (4, 4, 4)
