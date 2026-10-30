"""What a sequence call takes from the framework's batch.

- `Batch` parses the framework's flat token buffer once, before the engine runs.
- `Call.plan` groups a call's rows into one request per sequence and names the rows it returns.
- `admit` refuses token IDs outside the vocabulary, which the engine reads unchecked, and
  `operand` makes a call's token operand of admitted IDs.

`furiosa_opt_sglang.batch` is a copy of this module; a change goes into both.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate, groupby, pairwise
from typing import Generic, TypeVar, final

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

Identity = TypeVar("Identity", int, str)


@dataclass(frozen=True, slots=True)
class Batch:
    """A scheduled token batch, copied before engine execution."""

    tokens: tuple[tuple[int, ...], ...]
    cached: tuple[int, ...]

    @classmethod
    def parse(cls, tokens: Sequence[int], lengths: Sequence[int], totals: Sequence[int]) -> "Batch":
        """Refine the framework's flat batch framing, not the engine's serving limits."""
        offsets = (0, *accumulate(lengths))
        if (
            offsets[-1] != len(tokens)
            or any(length < 1 for length in lengths)
            or any(total < length for total, length in zip(totals, lengths, strict=True))
        ):
            raise Framing(len(tokens), offsets[-1], tuple(lengths), tuple(totals))
        return cls(
            tuple(tuple(tokens[start:stop]) for start, stop in pairwise(offsets)),
            tuple(total - length for total, length in zip(totals, lengths, strict=True)),
        )


@final
class Framing(ValueError):
    """The framework's row counts do not partition its flat token buffer within each sequence."""

    # A plain exception: context managers and pickling set and read exception attributes.
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
            f"lengths {self.totals} and partition {self.tokens} tokens, got {self.scheduled} rows"
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
    def plan(rows: Sequence[Row[Identity]], *, every: bool) -> "Call[Identity]":
        """One request per run of a sequence's rows; `every` returns all rows, not only the
        complete ones."""
        return Call(
            tuple(
                Request(sequence, rows[run[0]].position, run, rows[run[-1]].complete)
                for sequence, group in groupby(range(len(rows)), key=lambda i: rows[i].sequence)
                for run in [tuple(group)]
            ),
            tuple(i for i, row in enumerate(rows) if every or row.complete),
        )


def admit(ids: Sequence[int], vocabulary: int | None) -> None:
    """Refuse `ids` unless each lies below `vocabulary`; a model without a vocabulary (`None`)
    takes no token IDs."""
    if vocabulary is None or any(id < 0 or id >= vocabulary for id in ids):
        raise ValueError(f"token IDs must lie in the vocabulary of {vocabulary} tokens")


def operand(ids: Sequence[int], vocabulary: int | None) -> NDArray[np.int32]:
    """`ids` as an engine call's int32 token operand, once `admit` takes them."""
    admit(ids, vocabulary)
    return np.asarray(ids, dtype=np.int32)
