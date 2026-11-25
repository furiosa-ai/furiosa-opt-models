"""Framework-neutral planning for the rows and token operands of engine calls."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import accumulate, groupby, islice
from typing import Generic, TypeVar, final

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

Identity = TypeVar("Identity")
__all__ = [
    "Batch",
    "Call",
    "Chunk",
    "Framing",
    "Identity",
    "Request",
    "Row",
    "admit",
    "chunks",
    "operand",
]


@dataclass(frozen=True, slots=True)
class Batch:
    """A scheduled token batch with immutable token rows and cache positions.

    Use `parse` to snapshot framework sequences. Direct construction requires tuple rows and a
    tuple of cache positions.
    """

    tokens: tuple[tuple[int, ...], ...]
    cached: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.tokens) != len(self.cached):
            raise ValueError("batch token rows and cache positions must have the same length")
        if any(not tokens for tokens in self.tokens):
            raise ValueError("batch token rows must not be empty")
        if any(position < 0 for position in self.cached):
            raise ValueError("batch cache positions must be nonnegative")

    @classmethod
    def parse(cls, tokens: Sequence[int], lengths: Sequence[int], totals: Sequence[int]) -> Batch:
        """Refine framework row framing, without applying engine serving limits."""
        offsets = (0, *accumulate(lengths))
        if (
            offsets[-1] != len(tokens)
            or len(lengths) != len(totals)
            or any(length < 1 for length in lengths)
            or any(total < length for total, length in zip(totals, lengths, strict=True))
        ):
            raise Framing(len(tokens), offsets[-1], tuple(lengths), tuple(totals))
        remaining = iter(tokens)
        return cls(
            tuple(tuple(islice(remaining, length)) for length in lengths),
            tuple(total - length for total, length in zip(totals, lengths, strict=True)),
        )


@final
class Framing(ValueError):
    """Framework row counts do not partition its flat token buffer within each sequence."""

    # Context managers and pickle restore exception attributes; keep this a plain ValueError.
    def __init__(
        self, tokens: int, scheduled: int, lengths: tuple[int, ...], totals: tuple[int, ...]
    ) -> None:
        super().__init__(tokens, scheduled, lengths, totals)
        self.tokens = tokens
        self.scheduled = scheduled
        self.lengths = lengths
        self.totals = totals

    @override
    def __str__(self) -> str:
        return (
            f"scheduled input lengths {self.lengths} must be positive, within the sequence "
            f"lengths {self.totals} and partition {self.tokens} tokens; their sum is "
            f"{self.scheduled}"
        )


@dataclass(frozen=True, slots=True)
class Row(Generic[Identity]):
    """One sequence position and whether the caller reads its output."""

    sequence: Identity
    position: int
    complete: bool


@dataclass(frozen=True, slots=True)
class Request(Generic[Identity]):
    """Consecutive input rows belonging to one sequence in a call."""

    sequence: Identity
    cached: int
    rows: tuple[int, ...]
    complete: bool


@dataclass(frozen=True, slots=True)
class Call(Generic[Identity]):
    """Input runs and output row order for one engine call."""

    requests: tuple[Request[Identity], ...]
    kept: tuple[int, ...]

    @staticmethod
    def plan(rows: Sequence[Row[Identity]], *, every: bool) -> Call[Identity]:
        """Make one request per contiguous sequence run.

        For each run, callers must supply consecutive positions. `every` keeps all rows; otherwise,
        only rows marked complete are kept.
        """
        requests: list[Request[Identity]] = []
        for sequence, group in groupby(range(len(rows)), key=lambda i: rows[i].sequence):
            run = tuple(group)
            requests.append(Request(sequence, rows[run[0]].position, run, rows[run[-1]].complete))
        return Call(
            tuple(requests),
            tuple(i for i, row in enumerate(rows) if every or row.complete),
        )


@dataclass(frozen=True, slots=True)
class Chunk:
    """A sequence's consecutive rows in one capacity-bounded call."""

    sequence: int
    start: int
    length: int


def chunks(lengths: Sequence[int], rows: int) -> Iterator[tuple[Chunk, ...]]:
    """Fairly divide row work across calls, with at most one run per sequence per call."""
    if rows < 1:
        raise ValueError("a decoder call must serve at least one row")
    if any(length < 0 for length in lengths):
        raise ValueError("sequence work cannot have a negative row count")

    pending = deque((sequence, 0) for sequence, length in enumerate(lengths) if length)
    while pending:
        selected = [pending.popleft() for _ in range(min(len(pending), rows))]
        sizes = [1] * len(selected)
        capacity = rows - len(selected)
        active = deque(
            index
            for index, (sequence, start) in enumerate(selected)
            if start + sizes[index] < lengths[sequence]
        )
        carry_turn = bool(capacity and active)
        while capacity and active:
            index = active.popleft()
            sequence, start = selected[index]
            sizes[index] += 1
            capacity -= 1
            if start + sizes[index] < lengths[sequence]:
                active.append(index)
        next_sequence = selected[active[0]][0] if carry_turn and active else None

        yield tuple(
            Chunk(sequence, start, size)
            for (sequence, start), size in zip(selected, sizes, strict=True)
        )
        pending.extend(
            (sequence, start + size)
            for (sequence, start), size in zip(selected, sizes, strict=True)
            if start + size < lengths[sequence]
        )
        if next_sequence is not None:
            # Continue from the next sequence that would have received a spare row.
            while pending[0][0] != next_sequence:
                pending.rotate(-1)


def admit(ids: Sequence[int], vocabulary: int | None) -> None:
    """Refuse token IDs outside `vocabulary`; a model without a vocabulary takes no IDs."""
    if vocabulary is None or any(id < 0 or id >= vocabulary for id in ids):
        raise ValueError(f"token IDs must lie in the vocabulary of {vocabulary} tokens")


def operand(ids: Sequence[int], vocabulary: int | None) -> NDArray[np.int32]:
    """Make the int32 token operand of an engine call after admitting its IDs."""
    admit(ids, vocabulary)
    return np.asarray(ids, dtype=np.int32)
