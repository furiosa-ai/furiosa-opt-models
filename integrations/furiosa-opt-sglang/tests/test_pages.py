import pytest


@pytest.mark.parametrize(
    ("slots", "pages"), [([17, 3, 18, 40], [0, 1, 2]), ([], [])], ids=["unordered", "none"]
)
def test_host_image_copies_the_pages_holding_slots(slots: list[int], pages: list[int]) -> None:
    """A page copy selects each page that holds one of the token slots once, in order."""
    import torch

    from furiosa_opt_sglang.cache.pages import holding

    assert holding(torch.tensor(slots, dtype=torch.int32), 16) == pages


@pytest.mark.parametrize(
    ("shape", "floating"), [((16,), True), ((2, 8), False)], ids=["float", "two-dimensional"]
)
def test_host_image_refuses_other_slot_tensors(shape: tuple[int, ...], floating: bool) -> None:
    """Token slots in other tensor shapes do not name cache pages."""
    import torch

    from furiosa_opt_sglang.cache.pages import holding

    dtype = torch.float32 if floating else torch.int64
    with pytest.raises(ValueError, match="by a 1-D integer tensor"):
        _ = holding(torch.arange(16, 32, dtype=dtype).reshape(shape), 16)


def test_pages_reject_a_directly_constructed_mismatched_layout() -> None:
    """Every Pages instance has the same byte-size invariant as one parsed from a decoder."""
    from types import SimpleNamespace

    import torch

    from furiosa_opt_sglang.cache.pages import Pages

    fields = (range(2), 1, torch.bfloat16, 2, 4)
    valid_pool = SimpleNamespace(block=16, bytes=512)
    pages = Pages(valid_pool, *fields)
    assert pages.pool is valid_pool

    invalid_pool = SimpleNamespace(block=16, bytes=511)
    with pytest.raises(ValueError, match="511 bytes a block, SGLang 512"):
        Pages(invalid_pool, *fields)
