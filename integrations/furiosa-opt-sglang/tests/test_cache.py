"""SGLang's cache hooks copy the token rows it selects to and from the engine cache."""

import subprocess
import sys
from collections.abc import Sequence

import pytest


def test_offload_hooks_copy_cache_rows() -> None:
    # Isolate the device owner from the HTTP server started by the other tests.
    _ = subprocess.run([sys.executable, __file__, "copy"], check=True, timeout=180)


def test_copy_hooks_follow_the_engine_layout() -> None:
    """The rows the hooks move into other blocks serve later tokens bit for bit as where the engine
    wrote them."""
    _ = subprocess.run([sys.executable, __file__, "layout"], check=True, timeout=300)


@pytest.mark.parametrize(
    ("slots", "pages"),
    [(range(16, 32), [1]), ([*range(48, 64), *range(16, 32)], [3, 1]), ([], [])],
    ids=["one", "out-of-order", "none"],
)
def test_hierarchical_cache_moves_whole_pages(slots: Sequence[int], pages: list[int]) -> None:
    """SGLang's hierarchical cache names the pages it moves by their token slots, in its order."""
    import torch

    from furiosa_opt_sglang import offload

    assert offload.pages(torch.tensor(list(slots), dtype=torch.int64), 16) == pages


@pytest.mark.parametrize(
    ("slots", "reason"),
    [
        (range(16, 24), "hierarchical cache transfers move whole pages"),
        (range(8, 24), "a hierarchical cache page is a page's consecutive token slots"),
        ([*range(16, 31), 33], "a hierarchical cache page is a page's consecutive token slots"),
    ],
    ids=["partial", "unaligned", "gap"],
)
def test_hierarchical_cache_refuses_partial_pages(slots: Sequence[int], reason: str) -> None:
    """Token slots that are not whole pages of consecutive slots name no pages."""
    import torch

    from furiosa_opt_sglang import offload

    with pytest.raises(ValueError, match=reason):
        _ = offload.pages(torch.tensor(list(slots), dtype=torch.int64), 16)


@pytest.mark.parametrize(
    ("shape", "floating"), [((16,), True), ((2, 8), False)], ids=["float", "two-dimensional"]
)
def test_hierarchical_cache_refuses_other_slot_tensors(
    shape: tuple[int, ...], floating: bool
) -> None:
    """Token slots in a tensor other than one row of integers name no pages, though their values
    fill a whole page."""
    import torch

    from furiosa_opt_sglang import offload

    dtype = torch.float32 if floating else torch.int64
    with pytest.raises(ValueError, match="by a 1-D integer tensor"):
        _ = offload.pages(torch.arange(16, 32, dtype=dtype).reshape(shape), 16)


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
def test_hierarchical_cache_moves_adjacent_pages_in_one_copy(
    device: list[int], host: list[int], chips: int, moved: list[tuple[int, int, int]]
) -> None:
    """Pages that follow each other on the device and on the host move in one copy on a cache of
    one chip, and one page a copy on more chips, whose copy writes chip after chip."""
    from furiosa_opt_sglang import offload

    assert offload.runs(device, host, chips) == moved


def test_mamba_pool_copies_and_restores_slots() -> None:
    from array import array
    from collections.abc import Callable
    from typing import cast

    import torch
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.memory_pool import (
        HybridReqToTokenPool,
        NoOpMHATokenToKVPool,
    )
    from sglang.srt.sampling.sampling_params import SamplingParams

    from furiosa_opt_sglang.recurrent import Config

    config = Config(hidden_size=8, num_hidden_layers=2)
    pool = HybridReqToTokenPool(
        size=2,
        mamba_size=3,
        mamba_spec_state_size=2,
        max_context_len=32,
        device="cpu",
        enable_memory_saver=False,
        cache_params=config.mamba2_cache_params,
        mamba_layer_ids=[0, 1],
        enable_mamba_extra_buffer=False,
        enable_overlap_schedule=False,
    )
    requests = [Req(str(i), "", array("i", [1, 2]), SamplingParams()) for i in range(2)]
    rows = pool.alloc(requests)
    assert rows is not None
    slots = pool.get_mamba_indices(torch.tensor(rows))
    # PyTorch erases element types from integer tensor list conversion.
    integers = cast("Callable[[torch.Tensor], list[int]]", torch.Tensor.tolist)
    for req, slot in zip(requests, integers(slots), strict=True):
        assert req.kv.mamba_pool_idx is not None
        assert slot == int(req.kv.mamba_pool_idx)
    cache = pool.mamba_pool.mamba_cache
    assert cache.conv[0].shape == (2, 4, 16, 4)
    assert cache.temporal.shape == (2, 4, 16, 16)
    for layer in range(2):
        cache.conv[0][layer, slots[0]] = layer + 1
        cache.temporal[layer, slots[0]] = layer + 3
    # SGLang leaves its state snapshot API unannotated.
    save = cast(
        "Callable[[torch.Tensor], tuple[list[torch.Tensor], torch.Tensor]]",
        pool.mamba_pool.get_cpu_copy,
    )
    restore = cast(
        "Callable[[tuple[list[torch.Tensor], torch.Tensor], torch.Tensor], None]",
        pool.mamba_pool.load_cpu_copy,
    )
    snapshot = save(slots[:1])
    pool.mamba_pool.copy_from(slots[:1], slots[1:])
    pool.mamba_pool.clear_slots(slots[:1])
    assert not cache.conv[0][:, slots[0]].any()
    assert not cache.temporal[:, slots[0]].any()
    for layer in range(2):
        assert torch.all(cache.conv[0][layer, slots[1]] == layer + 1)
        assert torch.all(cache.temporal[layer, slots[1]] == layer + 3)
    restore(snapshot, slots[:1])
    torch.testing.assert_close(cache.conv[0][:, slots[0]], cache.conv[0][:, slots[1]])
    torch.testing.assert_close(cache.temporal[:, slots[0]], cache.temporal[:, slots[1]])
    for req in requests:
        pool.free_mamba_cache(req)
        pool.free(req)
    assert pool.available_size() == 2
    assert pool.mamba_allocator.available_size() == 3
    pool.clear()
    assert not pool.req_index_to_mamba_index_mapping.any()
    tokens = NoOpMHATokenToKVPool(32, 1, torch.bfloat16, 1, 8, 0, "cpu", enable_memory_saver=False)
    # SGLang leaves its token-pool snapshot API unannotated.
    copy = cast("Callable[..., list[torch.Tensor]]", tokens.get_cpu_copy)
    load = cast("Callable[..., None]", tokens.load_cpu_copy)
    assert copy(torch.tensor([1, 31]), mamba_indices=slots) == []
    load([], torch.tensor([1, 31]), mamba_indices=slots)
    assert tokens.get_kv_size_bytes() == (0, 0)


def copy_blocks() -> None:
    from collections.abc import Callable
    from typing import cast

    import huggingface_hub
    import torch
    from transformers import AutoConfig

    import furiosa_opt_engine as engine
    from furiosa_opt_sglang.cache import Cache, Pages
    from furiosa_opt_sglang.decoder import PES, Decoder, Location, chips

    # Hub overloads include untyped progress and user-agent parameters.
    download = cast("Callable[[str], str]", huggingface_hub.snapshot_download)
    directory = download("Qwen/Qwen3-1.7B")
    config = AutoConfig.from_pretrained(directory)
    device = engine.Device(engine.Placement(chips(0, 1), PES))
    decoder = Decoder.load(
        Location(directory, None), engine.LoadOptions(), device, config.vocab_size
    )
    assert decoder.model is not None
    heads, head = config.num_key_value_heads, config.head_dim
    block, layers = decoder.pool.block, len(decoder.layers)
    cache = Cache(decoder.model.cache, Pages.parse(decoder, heads, head), 4 * block)
    source = torch.tensor([block + 1, block + 3])
    target = torch.tensor([2 * block + 2, 2 * block + 5])
    expected = [
        [[torch.full((2, heads, head), layer + kind + 1, dtype=torch.bfloat16) for kind in (0, 1)]]
        for layer in range(layers)
    ]
    cache.load_cpu_copy(expected, source)
    cache.move_kv_cache(target, source)
    for layer, chunks in enumerate(cache.get_cpu_copy(target)):
        for got, want in zip(chunks[0], expected[layer][0], strict=True):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert cache.get_cpu_copy(torch.empty(0, dtype=torch.long))[0][0][0].numel() == 0


def follow_layout() -> None:
    from collections.abc import Callable
    from typing import cast

    import huggingface_hub
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoTokenizer

    import furiosa_opt_engine as engine
    from furiosa_opt_sglang.cache import Cache, Pages
    from furiosa_opt_sglang.decoder import PES, Decoder, Location, chips

    # Hub overloads include untyped progress and user-agent parameters.
    download = cast("Callable[[str], str]", huggingface_hub.snapshot_download)
    directory = download("Qwen/Qwen3-0.6B")
    config = AutoConfig.from_pretrained(directory)
    tokenizer = AutoTokenizer.from_pretrained(directory)
    device = engine.Device(engine.Placement(chips(0, 1), PES))
    # The base model's hidden rows tell apart attention over any other cache rows.
    options = engine.LoadOptions().architecture("Qwen3Model")
    decoder = Decoder.load(Location(directory, None), options, device, config.vocab_size)
    block, model = decoder.pool.block, decoder.model
    assert model is not None
    pages = Pages.parse(decoder, config.num_key_value_heads, config.head_dim)
    cache = Cache(model.cache, pages, 9 * block)
    storage = cache.storage
    assert storage is not None
    text = "The capital of France is Paris. The capital of Japan is Tokyo. The capital of Italy is"
    prompt = tokenizer(text).input_ids
    other = tokenizer("Water boils at one hundred degrees. " * 4).input_ids
    # The prompt ends inside a block, so a wrong row order moves bytes of other rows.
    assert block < len(prompt) < len(other) and len(prompt) % block
    source, target = [1, 2, 3, 4], [5, 6, 7, 8]

    def call(
        blocks: list[int], cached: int, ids: list[int], keep: engine.decoder.Keep
    ) -> torch.Tensor:
        tokens = engine.HostTensor.from_dlpack(np.asarray(ids, dtype=np.int32)).to_hbm(device)
        request = engine.decoder.Request([blocks], cached, tokens, keep)
        return torch.as_tensor(model.forward([request], storage).to_host().numpy())

    def prefill(blocks: list[int], ids: list[int]) -> None:
        for start in range(0, len(ids), decoder.rows):
            _ = call(blocks, start, ids[start : start + decoder.rows], engine.decoder.Keep.NONE)

    def extend(blocks: list[int]) -> torch.Tensor:
        """The hidden rows of eight more tokens after the prompt, a call each."""
        positions = enumerate(other[:8], start=len(prompt))
        return torch.cat([call(blocks, p, [t], engine.decoder.Keep.LAST) for p, t in positions])

    def slots(blocks: list[int]) -> torch.Tensor:
        return torch.tensor([blocks[p // block] * block + p % block for p in range(len(prompt))])

    # Rows another sequence left in the target blocks stand in for whatever a transfer misses.
    prefill(target, other)
    prefill(source, prompt)
    cache.move_kv_cache(slots(target), slots(source))
    assert torch.equal(extend(target), extend(source))


if __name__ == "__main__":
    {"copy": copy_blocks, "layout": follow_layout}[sys.argv[1]]()
