"""The binding parses its batch, plans its calls and makes its token operand."""

import pickle
from contextlib import contextmanager
from typing import cast

import pytest

from furiosa_opt_vllm import batch


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
