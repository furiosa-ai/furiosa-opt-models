import torch
from sglang.srt.mem_cache.memory_pool import NoOpMHATokenToKVPool

from furiosa_opt_sglang.cache import Allocator


def test_extension_preserves_the_partial_prefix_page() -> None:
    pool = NoOpMHATokenToKVPool(16, 4, torch.bfloat16, 1, 8, 0, "cpu", enable_memory_saver=False)
    allocator = Allocator(
        size=16,
        page_size=4,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=pool,
        need_sort=False,
    )
    first = allocator.alloc_extend(
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([3]),
        torch.tensor([3]),
        torch.tensor([-1]),
        3,
    )
    assert first is not None

    extension = allocator.alloc_extend(
        torch.tensor([3]),
        torch.tensor([3]),
        torch.tensor([6]),
        torch.tensor([6]),
        first[-1:],
        3,
    )

    assert extension is not None
    assert extension[0] == first[-1] + 1
    assert extension[1] % 4 == 0
    ordered = torch.sort(torch.cat((first, extension))).values
    assert torch.all(ordered[1:] != ordered[:-1])
