"""A token pool's page copies move pages that follow each other in one copy."""

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
