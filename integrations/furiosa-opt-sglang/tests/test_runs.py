"""A token pool's page copies name the pages that hold its slots, move adjacent pages in one copy,
and raise a failed copy only once every copy submitted with it ended."""

from collections.abc import Iterator
from typing import final

import pytest


@pytest.mark.parametrize(
    ("device", "host", "chips", "moved"),
    [
        ([4, 5, 6], [10, 11, 12], 1, [(4, 10, 3)]),
        ([4, 5, 7, 8], [0, 1, 2, 3], 1, [(4, 0, 2), (7, 2, 2)]),
        ([4, 5, 6], [0, 1, 3], 1, [(4, 0, 2), (6, 3, 1)]),
        ([6, 5, 4], [0, 1, 2], 1, [(6, 0, 1), (5, 1, 1), (4, 2, 1)]),
        ([4, 5], [0, 1], 2, [(4, 0, 1), (5, 1, 1)]),
    ],
    ids=["adjacent", "device-gap", "host-gap", "descending", "two-chips"],
)
def test_adjacent_pages_move_in_one_copy(
    device: list[int], host: list[int], chips: int, moved: list[tuple[int, int, int]]
) -> None:
    """Pages that follow each other on the device and on the host move in one copy on a cache of
    one chip, and one page a copy on more chips, whose copy writes chip after chip."""
    from furiosa_opt_sglang.cache import runs

    assert runs.runs(device, host, chips) == moved


@pytest.mark.parametrize(
    ("slots", "pages"), [([17, 3, 18, 40], [0, 1, 2]), ([], [])], ids=["unordered", "none"]
)
def test_host_image_copies_the_pages_holding_slots(slots: list[int], pages: list[int]) -> None:
    """The host image copies each page that holds one of the token slots once, in order."""
    import torch

    from furiosa_opt_sglang.cache import runs

    assert runs.holding(torch.tensor(slots, dtype=torch.int32), 16) == pages


@pytest.mark.parametrize(
    ("shape", "floating"), [((16,), True), ((2, 8), False)], ids=["float", "two-dimensional"]
)
def test_host_image_refuses_other_slot_tensors(shape: tuple[int, ...], floating: bool) -> None:
    """Token slots in a tensor other than one row of integers name no pages of the host image."""
    import torch

    from furiosa_opt_sglang.cache import runs

    dtype = torch.float32 if floating else torch.int64
    with pytest.raises(ValueError, match="by a 1-D integer tensor"):
        _ = runs.holding(torch.arange(16, 32, dtype=dtype).reshape(shape), 16)


@final
class Transfer:
    """A stand-in for the engine's `Transfer` that raises `failure`, if the test gives one, when
    waited for."""

    def __init__(self, failure: RuntimeError | None = None) -> None:
        self.failure = failure
        self.waited = False

    def wait(self) -> None:
        self.waited = True
        if self.failure is not None:
            raise self.failure


def test_failed_submission_waits_for_submitted_copies() -> None:
    """A host image copy that fails to submit raises, ahead of a submitted copy's own failure, once
    the copies submitted before it ended."""
    from furiosa_opt_sglang.cache import runs

    first, second = Transfer(RuntimeError("copy failed")), Transfer()

    def submitted() -> Iterator[Transfer]:
        yield first
        yield second
        raise ValueError("submission failed")

    with pytest.raises(ValueError, match="submission failed"):
        _ = runs.submit(submitted())
    assert (first.waited, second.waited) == (True, True)


def test_failed_copy_waits_for_the_other_copies() -> None:
    """A wait over host image copies that finds a failed one waits for the copies after it, then
    raises the first failure."""
    from furiosa_opt_sglang.cache import runs

    transfers = (Transfer(RuntimeError("first")), Transfer(), Transfer(RuntimeError("third")))
    with pytest.raises(RuntimeError, match="first"):
        runs.wait(transfers)
    assert all(transfer.waited for transfer in transfers)
