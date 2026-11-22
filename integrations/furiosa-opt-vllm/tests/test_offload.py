"""vLLM's KV cache offloading keeps evicted prompt blocks in host memory, and a prompt answers with
them as with its blocks in the device cache.

- The jobs that copy blocks report each ended job once and raise a failed copy once every copy
  submitted before it ended, here over stand-in transfers; blocks whose host rows follow each other
  share one copy.
- On RNGD, `vllm serve --kv-offloading-size` answers an evicted prompt as a server without
  offloading answers it from its device cache. Under batch invariance a prompt reloaded from
  chunks, or resumed after preemption, answers as it does with no block cached.
"""

from __future__ import annotations

import urllib.request
from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, final

import pytest
import torch
from conftest import post, serve
from test_preemption import CHECKPOINT, CONTEXT, PREEMPTED, counter, greedy
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.inputs import TokensPrompt
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.base import CopyRun, GPULoadStoreSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

from furiosa_opt_engine.cache_transfer import (
    CachePageRoute,
    copy_cache_pages,
    page_runs,
)
from furiosa_opt_engine.cache_transfer import (
    submit as submit_transfers,
)
from furiosa_opt_vllm import cache
from furiosa_opt_vllm.offload import worker

if TYPE_CHECKING:
    from collections.abc import Iterator
    from http.client import HTTPResponse

    from transformers import PreTrainedTokenizerBase
    from vllm import RequestOutput
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
        OffloadingConnectorMetadata,
    )
    from vllm.entrypoints.openai.completion.protocol import CompletionResponse

ANSWER: Final = 32
BLOCK: Final = 16
# Whose home and hobby each prompt asks about; the first prompt ends inside a block, so a hit on
# either tier leaves the same tokens to compute.
NOTES: Final = (
    (("Dmitri", "Kyiv", "painting"), ("Esra", "Izmir", "cycling"), ("Femi", "Lagos", "drumming")),
    (("Alice", "Paris", "the violin"), ("Bruno", "Lima", "chess"), ("Chen", "Oslo", "rowing")),
    (("Gita", "Pune", "hiking"), ("Hugo", "Lyon", "baking"), ("Ines", "Porto", "surfing")),
)
# Notes that fill three chunks of two blocks before the question about their last person.
LONGER: Final = (NOTES[0] + NOTES[1], NOTES[2] + NOTES[0], NOTES[1] + NOTES[2])
# Notes of nine people, in two orders, that fill eight blocks and part of a ninth.
NINE: Final = (NOTES[0] + NOTES[1] + NOTES[2], NOTES[1] + NOTES[2] + NOTES[0])
# Notes of ten people, whose prompt's last chunk of four blocks fills a ninth block and part of a
# tenth.
TEN: Final = tuple(people + (("Jun", "Seoul", "climbing"),) for people in NINE)
LOADED: Final = "vllm:kv_offload_load_bytes"
# A server whose device holds 8 blocks beside vLLM's null block, which one request and its answer
# mostly fill, and which reports each prompt's cached tokens.
SERVED: Final = [
    CHECKPOINT,
    "--dtype",
    "bfloat16",
    "--generation-config",
    "vllm",
    "--max-model-len",
    "128",
    "--num-gpu-blocks-override",
    "9",
    "--enable-prompt-tokens-details",
]


@final
class Copies:
    """A stand-in for the engine's `Transfer` whose copies end, or fail, when the test says."""

    def __init__(self, failure: RuntimeError | None = None) -> None:
        self.ended = False
        self.waited = False
        self.failure = failure

    def done(self) -> bool:
        if self.failure is not None:
            raise self.failure
        return self.ended

    def wait(self) -> None:
        self.waited = True
        if self.failure is not None:
            raise self.failure
        self.ended = True


def jobs(*copies: Copies) -> worker.Jobs:
    """Jobs 0, 1, ... each running one of `copies`, of 64 bytes."""
    running = worker.Jobs()
    for job_id, transfer in enumerate(copies):
        running.submit(job_id, [transfer], 64)
    return running


def test_reports_each_job_once_it_ends() -> None:
    """A poll reports no job whose copies still run, and each ended job once, in the order the
    jobs end."""
    first, second = Copies(), Copies()
    running = jobs(first, second)
    assert running.finished() == []
    second.ended = True
    assert [result.job_id for result in running.finished()] == [1]
    assert running.finished() == []
    first.ended = True
    (result,) = running.finished()
    assert (result.job_id, result.success, result.transfer_size) == (0, True, 64)
    assert running.finished() == []


def test_reports_job_once_every_copy_ends() -> None:
    """A job's copies end in any order, and the job ends with the last of them."""
    first, middle, last = Copies(), Copies(), Copies()
    running = worker.Jobs()
    running.submit(0, [first, middle, last], 192)
    last.ended = True
    assert running.finished() == []
    first.ended = True
    assert running.finished() == []
    middle.ended = True
    assert [result.job_id for result in running.finished()] == [0]


def test_waits_only_for_named_jobs() -> None:
    """A wait ends the named jobs alone and leaves their reports to the next poll."""
    first, second = Copies(), Copies()
    running = jobs(first, second)
    running.wait({0, 7})
    assert (first.waited, second.waited) == (True, False)
    assert [result.job_id for result in running.finished()] == [0]


def test_waits_for_each_copy_of_a_failed_job() -> None:
    """A wait that finds a failed copy waits for the job's later copies, then raises the first
    failure."""
    copies = (Copies(RuntimeError("first")), Copies(), Copies(RuntimeError("third")))
    running = worker.Jobs()
    running.submit(0, copies, 192)
    with pytest.raises(RuntimeError, match="first"):
        running.wait({0})
    assert all(transfer.waited for transfer in copies)


def test_failed_submission_waits_for_submitted_copies() -> None:
    """A copy that fails to submit raises, ahead of a submitted copy's failure, once the job's
    copies submitted before it ended; the job reports nothing."""
    first, second = Copies(RuntimeError("copy failed")), Copies()

    def submitted() -> Iterator[Copies]:
        yield first
        yield second
        raise RuntimeError("submission failed")

    running = worker.Jobs()
    with pytest.raises(RuntimeError, match="submission failed"):
        running.submit(0, submitted(), 192)
    assert (first.waited, second.waited) == (True, True)
    assert running.finished() == []


def test_shared_core_submission_drains_copies_on_failure() -> None:
    first = Copies(RuntimeError("copy failed"))

    def submitted() -> Iterator[Copies]:
        yield first
        raise RuntimeError("submission failed")

    with pytest.raises(RuntimeError, match="submission failed") as error:
        _ = submit_transfers(submitted())
    assert first.waited
    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "copy failed"


def test_shared_core_routes_pages_in_both_directions() -> None:
    route = CachePageRoute(
        cache_offset=0,
        host_offset=1,
        fragment_bytes=1,
        fragments=2,
        cache_stride=2,
        host_stride=3,
    )
    routes = (route,)
    cache = bytearray([10, 11, 12, 99, 20, 21, 22, 99])
    host = bytearray(10)
    copy_cache_pages(
        cache,
        host,
        routes,
        page_count=2,
        cache_page_bytes=3,
        cache_page_stride=4,
        host_page_bytes=5,
        host_page_stride=5,
        to_host=True,
    )
    assert host == bytearray([0, 10, 0, 0, 12, 0, 20, 0, 0, 22])

    cache[:] = bytes([255]) * len(cache)
    copy_cache_pages(
        cache,
        host,
        routes,
        page_count=2,
        cache_page_bytes=3,
        cache_page_stride=4,
        host_page_bytes=5,
        host_page_stride=5,
        to_host=False,
    )
    assert cache == bytearray([10, 0, 12, 0, 20, 0, 22, 0])


def test_raises_failed_copy() -> None:
    """A poll that finds a failed copy raises its error instead of reporting the job."""
    failure = RuntimeError("copy failed")
    running = jobs(Copies(), Copies(failure))
    with pytest.raises(RuntimeError, match="copy failed"):
        _ = running.finished()


def test_poll_drains_every_copy_of_a_failed_job() -> None:
    """A poll that finds one failed copy waits for every submitted copy before it raises."""
    copies = (Copies(), Copies(RuntimeError("copy failed")), Copies())
    copies[0].ended = True
    running = worker.Jobs()
    running.submit(0, copies, 192)

    with pytest.raises(RuntimeError, match="copy failed"):
        _ = running.finished()

    assert all(transfer.waited for transfer in copies)


def test_shutdown_ends_every_job_then_raises() -> None:
    """Shutdown waits for every job, then raises the first failure among them."""
    copies = (Copies(RuntimeError("first")), Copies(), Copies(RuntimeError("second")))
    running = jobs(*copies)
    with pytest.raises(RuntimeError, match="first"):
        running.shutdown()
    assert all(transfer.waited for transfer in copies)
    assert running.finished() == []


@pytest.mark.parametrize(
    ("blocks", "rows", "chips", "moved"),
    [
        ([4, 5, 6], [10, 11, 12], 1, [(4, 10, 3)]),
        ([4, 5, 7, 8], [0, 1, 2, 3], 1, [(4, 0, 2), (7, 2, 2)]),
        ([4, 5, 6], [0, 1, 3], 1, [(4, 0, 2), (6, 3, 1)]),
        ([6, 5, 4], [0, 1, 2], 1, [(6, 0, 1), (5, 1, 1), (4, 2, 1)]),
        ([4, 5], [0, 1], 2, [(4, 0, 1), (5, 1, 1)]),
    ],
    ids=["adjacent", "block-gap", "row-gap", "descending", "two-chips"],
)
def test_moves_adjacent_blocks_in_one_copy(
    blocks: list[int], rows: list[int], chips: int, moved: list[tuple[int, int, int]]
) -> None:
    """Blocks whose host rows follow theirs move in one copy on a cache of one chip, and one block
    a copy on more chips, whose copy writes chip after chip."""
    assert page_runs(blocks, rows, chips) == tuple(moved)


def test_staged_transfer_commits_shared_bytes_after_dma_completion() -> None:
    """The shared primary row is committed only after its DMA staging copy finishes."""
    copy = Copies()
    committed: list[str] = []
    released: list[str] = []
    staged = worker._StagedTransfer(
        copy,
        object(),
        lambda: committed.append("stored"),
        lambda: released.append("staging"),
    )

    assert not staged.done()
    assert committed == released == []
    copy.ended = True
    assert staged.done()
    assert copy.waited
    assert committed == ["stored"]
    assert released == ["staging"]
    assert staged.done()
    assert committed == ["stored"]
    assert released == ["staging"]


def test_staged_transfer_does_not_commit_failed_dma() -> None:
    """A failed engine copy never publishes unverified staging bytes into the shared region."""
    failure = RuntimeError("DMA failed")
    released: list[str] = []
    staged = worker._StagedTransfer(
        Copies(failure),
        object(),
        lambda: pytest.fail("committed"),
        lambda: released.append("staging"),
    )

    with pytest.raises(RuntimeError, match="DMA failed"):
        staged.wait()
    assert released == ["staging"]


class _ImmediateCopy:
    def done(self) -> bool:
        return True

    def wait(self) -> None:
        return


class _Buffer:
    def __init__(self, source: int | bytearray, start: int = 0, size: int | None = None) -> None:
        self.data = source if isinstance(source, bytearray) else bytearray(source)
        self.start = start
        self.size = len(self.data) - start if size is None else size

    def __len__(self) -> int:
        return self.size

    def slice(self, offset: int, size: int) -> _Buffer:
        if offset < 0 or size <= 0 or offset + size > self.size:
            raise ValueError("invalid test buffer range")
        return type(self)(self.data, self.start + offset, size)

    @property
    def view(self) -> memoryview:
        return memoryview(self.data)[self.start : self.start + self.size]

    def to_hbm(self, device: _FakeHbmBuffer) -> _ImmediateCopy:
        device.view[:] = self.view
        return _ImmediateCopy()


class _FakeHostBuffer(_Buffer):
    pass


class _FakeHbmBuffer(_Buffer):
    def to_host(self, host: _FakeHostBuffer) -> _ImmediateCopy:
        host.view[:] = self.view
        return _ImmediateCopy()


def _attention_page(page_bytes: int) -> FullAttentionSpec:
    head_size = page_bytes // 4
    return FullAttentionSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=head_size,
        dtype=torch.uint8,
    )


def test_routes_multigroup_ranges_to_heterogeneous_engine_subgroups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One logical group fans out by layer page size; group offsets keep independent block IDs."""
    monkeypatch.setattr(worker.engine, "HostBuffer", _FakeHostBuffer)
    (small, large, other) = (8, 16, 8)
    layers = [_FakeHbmBuffer(8 * size) for size in (small, large, other)]
    for block in range(8):
        for layer, size, value in zip(layers, (small, large, other), (10, 30, 50), strict=True):
            layer.view[block * size : (block + 1) * size] = bytes([value + block]) * size

    cache_spec = SimpleNamespace(
        pools=[
            SimpleNamespace(bytes=small),
            SimpleNamespace(bytes=large),
            SimpleNamespace(bytes=other),
        ],
        groups=[
            SimpleNamespace(pool=0, layers=[0], logical_group=1),
            SimpleNamespace(pool=1, layers=[1], logical_group=1),
            SimpleNamespace(pool=2, layers=[2], logical_group=0),
        ],
    )
    cache_instance = SimpleNamespace(spec=cache_spec, layers=layers, blocks=8)
    cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["layers.2"], kv_cache_spec=_attention_page(other)),
            KVCacheGroupSpec(
                layer_names=["layers.0", "layers.1"],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=2,
                    kv_cache_specs={
                        "layers.0": _attention_page(small),
                        "layers.1": _attention_page(large),
                    },
                ),
            ),
        ],
    )
    spec = SimpleNamespace(
        blocks_per_chunk=2,
        num_chunks=4,
        kv_bytes_per_chunk=64,
        cpu_page_size_per_worker=64,
        config=SimpleNamespace(canonical_layout=False),
    )
    host = worker.Host(cache_instance, 1, spec, None, cache_config, (0, 1))  # type: ignore[arg-type]
    gpu = GPULoadStoreSpec([2, 3, 4], group_sizes=[1, 2], block_indices=[0, 1])
    cpu = CPULoadStoreSpec([2, 3, 1])

    assert host.submit_store(0, gpu, cpu)
    (stored,) = host.get_finished()
    assert stored.transfer_size == 56
    assert bytes(host.rows[0].slice(7 * small, small).view) == bytes([13]) * small
    assert bytes(host.rows[0].slice(2 * small, small).view) == bytes([14]) * small
    assert bytes(host.rows[1].slice(7 * large, large).view) == bytes([33]) * large
    assert bytes(host.rows[1].slice(2 * large, large).view) == bytes([34]) * large
    assert bytes(host.rows[2].slice(2 * other, other).view) == bytes([52]) * other

    for layer in layers:
        layer.view[:] = bytes(len(layer))
    assert host.submit_load(1, cpu, gpu)
    (loaded,) = host.get_finished()
    assert loaded.transfer_size == stored.transfer_size
    assert bytes(layers[0].slice(3 * small, small).view) == bytes([13]) * small
    assert bytes(layers[0].slice(4 * small, small).view) == bytes([14]) * small
    assert bytes(layers[1].slice(3 * large, large).view) == bytes([33]) * large
    assert bytes(layers[2].slice(2 * other, other).view) == bytes([52]) * other
    host.shutdown()

    shared = bytearray(4 * 64)
    region = SimpleNamespace(
        rank=0,
        create_kv_memoryview=lambda: memoryview(shared),
        cleanup=lambda: None,
    )
    tiered = worker.Host(
        cache_instance,
        1,
        spec,
        None,
        cache_config,
        (0, 1),
        mmap_region=region,  # type: ignore[arg-type]
    )
    assert tiered._staging_pool_bytes == 0
    assert tiered.submit_store(2, gpu, cpu)
    assert not any(shared)
    (tiered_store,) = tiered.get_finished()
    assert tiered_store.transfer_size == stored.transfer_size
    staged_ids = {id(stage) for buffers in tiered._staging_pool.values() for stage in buffers}
    assert tiered._staging_pool_bytes == 32
    assert shared[128:136] == bytes([52]) * other
    assert shared[216:224] == bytes([13]) * small
    assert shared[240:256] == bytes([33]) * large
    assert shared[96:112] == bytes([34]) * large

    for layer in layers:
        layer.view[:] = bytes(len(layer))
    assert tiered.submit_load(3, cpu, gpu)
    assert tiered._staging_pool_bytes == 0
    (tiered_load,) = tiered.get_finished()
    assert tiered_load.transfer_size == stored.transfer_size
    assert {
        id(stage) for buffers in tiered._staging_pool.values() for stage in buffers
    } == staged_ids
    assert bytes(layers[0].slice(3 * small, small).view) == bytes([13]) * small
    assert bytes(layers[0].slice(4 * small, small).view) == bytes([14]) * small
    assert bytes(layers[1].slice(3 * large, large).view) == bytes([33]) * large
    assert bytes(layers[1].slice(4 * large, large).view) == bytes([34]) * large
    assert bytes(layers[2].slice(2 * other, other).view) == bytes([52]) * other
    tiered.shutdown()
    assert tiered._staging_pool_bytes == 0


@pytest.mark.parametrize(
    ("strides", "portable"),
    [((4, 2, 2, 1), False), ((2, 4, 2, 1), True)],
    ids=["opaque-layout", "certified-layout"],
)
def test_canonical_mapping_uses_metadata_views_and_fails_closed(
    strides: tuple[int, int, int, int], portable: bool
) -> None:
    """Canonical mapping consumes a meta view of the exact page strides, then certifies or falls
    back through vLLM's own tiling check."""
    pool = SimpleNamespace(
        bytes=8,
        element_bytes=1,
        shape=[2, 2, 1, 2],
        strides=list(strides),
    )
    engine_group = SimpleNamespace(pool=0, layers=[0], logical_group=0)
    cache_instance = SimpleNamespace(spec=SimpleNamespace(pools=[pool], groups=[engine_group]))
    layer_spec = FullAttentionSpec(block_size=2, num_kv_heads=1, head_size=2, dtype=torch.uint8)
    cache_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layers.0"], kv_cache_spec=layer_spec)],
    )
    group_indices = cache.group_indices(cache_instance.spec, cache_config.kv_cache_groups)
    view = worker._metadata_cache_views(cache_instance, cache_config, group_indices)["layers.0"]
    assert view.device.type == "meta"
    assert tuple(view.shape) == (4, 2, 2, 1, 2)
    assert tuple(view.stride()) == (8, strides[1], strides[0], strides[2], strides[3])

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            world_size=1,
            cp_kv_cache_interleave_size=1,
            rank=0,
        ),
        model_config=SimpleNamespace(get_total_num_kv_heads=lambda: 1),
    )
    mappings = worker.derive_canonical_mappings(config, cache_config, {"layers.0": view})
    mapping = mappings["layers.0"]
    assert (mapping.local_page_size_bytes, mapping.canonical_page_size_bytes) == (8, 8)
    assert mapping.parallelism_agnostic is portable


def test_canonical_copy_reorders_fragments_and_zeros_padding() -> None:
    """Only upstream-certified page bytes persist, and loading restores local offsets."""
    mapping = SimpleNamespace(
        local_page_size_bytes=3,
        canonical_page_size_bytes=3,
        runs=(
            CopyRun(0, 2, 1, 1, 1, 1),
            CopyRun(1, 0, 1, 1, 1, 1),
            CopyRun(2, 1, 1, 1, 1, 1),
        ),
    )
    local = bytearray([10, 20, 30, 99, 40, 50, 60, 99])
    canonical = bytearray(6)
    worker._copy_mapping(canonical, local, mapping, 2, local_page_stride=4, store=True)
    assert canonical == bytes([20, 30, 10, 50, 60, 40])

    restored = bytearray(8)
    worker._copy_mapping(restored, canonical, mapping, 2, local_page_stride=4, store=False)
    assert restored == bytes([10, 20, 30, 0, 40, 50, 60, 0])


def ask(tokenizer: PreTrainedTokenizerBase, people: tuple[tuple[str, str, str], ...]) -> list[int]:
    """A chat question whose answer only the notes at its start hold."""
    facts = " ".join(
        f"{name} lives in {city} and spends every weekend on {hobby}."
        for name, city, hobby in people
    )
    question = (
        f"Read these notes. {facts} "
        f"Where does {people[1][0]} live, and what does {people[-1][0]} do on weekends?"
    )
    match tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ):
        case str() as text:
            return tokenizer.encode(text, add_special_tokens=False)
        case _:
            raise TypeError("untokenized chat rendering must return text")


def complete(url: str, prompt: list[int]) -> CompletionResponse:
    """The server's greedy answer to `prompt`, with its token IDs."""
    from vllm.entrypoints.openai.completion.protocol import (
        CompletionRequest,
        CompletionResponse,
    )

    request = CompletionRequest.model_validate(
        {
            "model": CHECKPOINT,
            "prompt": prompt,
            "max_tokens": ANSWER,
            "temperature": 0,
            "ignore_eos": True,
            "return_token_ids": True,
        }
    )
    return post(f"{url}/v1/completions", request, CompletionResponse)


def cached(response: CompletionResponse) -> int | None:
    """The prompt tokens the server reports cached for `response`."""
    details = response.usage.prompt_tokens_details if response.usage else None
    return None if details is None else details.cached_tokens


def loaded(url: str) -> float:
    """The bytes the server reports loaded from host memory."""
    from prometheus_client.parser import text_string_to_metric_families

    response: HTTPResponse = urllib.request.urlopen(f"{url}/metrics", timeout=30)
    with response:
        text = response.read().decode()
    return sum(
        sample.value
        for family in text_string_to_metric_families(text)
        if family.name == LOADED
        for sample in family.samples
        if sample.name == f"{LOADED}_total"
    )


@pytest.mark.feature("KV cache offloading")
def test_serves_evicted_prefix_from_host_memory(
    tokenizer: PreTrainedTokenizerBase, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A prompt whose blocks other prompts evicted from the device reloads them from host memory:
    `vllm serve --kv-offloading-size` answers it as a server without offloading answers it from its
    device cache."""
    first, *others = (ask(tokenizer, people) for people in NOTES)
    hit = len(first) // BLOCK * BLOCK
    with serve(SERVED, tmp_path_factory) as url:
        _ = complete(url, first)
        resident = complete(url, first)
    assert cached(resident) == hit
    with serve([*SERVED, "--kv-offloading-size", "1"], tmp_path_factory) as url:
        _ = complete(url, first)
        for prompt in others:
            _ = complete(url, prompt)
        assert loaded(url) == 0
        reloaded = complete(url, first)
        assert loaded(url) > 0
    assert cached(reloaded) == hit
    assert reloaded.choices[0].token_ids == resident.choices[0].token_ids


def generate(llm: LLM, prompt: list[int]) -> RequestOutput:
    """The greedy answer of `llm` to `prompt` alone."""
    params = SamplingParams(temperature=0, max_tokens=ANSWER, ignore_eos=True)
    (output,) = llm.generate(TokensPrompt(prompt_token_ids=prompt), params)
    return output


@pytest.mark.feature("KV cache offloading")
def test_reloads_part_of_a_chunk(
    tokenizer: PreTrainedTokenizerBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Host chunks of two blocks reload whole, and from the middle of a chunk whose first block
    the device holds, and the prompt answers as it did before any of its blocks was cached."""
    # A batch-invariant load computes a row alike whatever calls wrote the blocks it reads.
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    llm = LLM(
        CHECKPOINT,
        dtype="bfloat16",
        generation_config="vllm",
        max_model_len=160,
        num_gpu_blocks_override=160 // BLOCK + 1,
        disable_log_stats=False,
        kv_offloading_size=1,
        kv_transfer_config=KVTransferConfig(kv_connector_extra_config={"blocks_per_chunk": 2}),
    )
    try:
        first, *others = (ask(tokenizer, people) for people in LONGER)
        assert len(first) // BLOCK == 6
        computed = generate(llm, first)
        for prompt in others:
            _ = generate(llm, prompt)
        # Sharing three blocks, a prompt reloads the first chunk whole and computes the third block.
        shared = first[: 3 * BLOCK] + others[0][3 * BLOCK :]
        assert generate(llm, shared).num_cached_tokens == 2 * BLOCK
        before = counter(llm, LOADED)
        # Three blocks hit on the device; the other three reload from the second and third chunks.
        reloaded = generate(llm, first)
        assert counter(llm, LOADED) > before
        assert reloaded.num_cached_tokens == 6 * BLOCK
        assert reloaded.outputs[0].token_ids == computed.outputs[0].token_ids
    finally:
        llm.llm_engine.engine_core.shutdown()


def sequential(offloading: float | None = None) -> LLM:
    """An engine that runs one request at a time, in a device cache of 12 blocks that a request and
    its answer mostly fill, with `offloading` GiB of host memory if set; each synchronous step frees
    a finished request's blocks before the next step schedules another request."""
    return LLM(
        CHECKPOINT,
        dtype="bfloat16",
        generation_config="vllm",
        max_model_len=12 * BLOCK,
        max_num_seqs=1,
        max_num_batched_tokens=64,
        num_gpu_blocks_override=13,
        async_scheduling=False,
        disable_log_stats=False,
        kv_offloading_size=offloading,
    )


@pytest.mark.feature("KV cache offloading")
def test_reuses_blocks_once_their_store_ends(
    tokenizer: PreTrainedTokenizerBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A request takes over a finished request's blocks while their store to host memory is still
    pending: the step that hands them over submits the store and waits for it before its calls, so
    the finished prompt, reloaded, answers as its device prefix hit does."""
    first, second = (ask(tokenizer, people) for people in TEN)
    oracle = sequential()
    try:
        _ = generate(oracle, first)
        resident = generate(oracle, first)
        alone = generate(oracle, second)
    finally:
        oracle.llm_engine.engine_core.shutdown()
    # The engine core runs in this process, so the test sees the stores each step's fence covers.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    pending: list[int] = []
    flush = worker.Worker.flush

    def spy(self: worker.Worker, metadata: OffloadingConnectorMetadata) -> None:
        # The flushed stores that no earlier step submitted.
        unsubmitted = {job_id for job_id, _ in self.deferred}.union(metadata.store_jobs)
        pending.append(len((metadata.jobs_to_flush or set()) & unsubmitted))
        flush(self, metadata)

    monkeypatch.setattr(worker.Worker, "flush", spy)
    llm = sequential(1)
    try:
        # The first request ends at its first token, in the step whose prefill fills its ninth
        # block, before that block's store starts; the second takes the block in the next step.
        params = [
            SamplingParams(temperature=0, max_tokens=1),
            SamplingParams(temperature=0, max_tokens=ANSWER, ignore_eos=True),
        ]
        prompts = [TokensPrompt(prompt_token_ids=prompt) for prompt in (first, second)]
        _, taken = llm.generate(prompts, params)
        reloaded = generate(llm, first)
        assert counter(llm, LOADED) > 0
        assert reloaded.num_cached_tokens == resident.num_cached_tokens == 9 * BLOCK
        assert reloaded.outputs[0].token_ids == resident.outputs[0].token_ids
        assert taken.outputs[0].token_ids == alone.outputs[0].token_ids
        assert any(pending), pending
    finally:
        # vLLM's in-process shutdown empties torch's accelerator caches, which RNGD has none of.
        with monkeypatch.context() as cleanup:
            cleanup.setattr(torch.accelerator, "empty_cache", lambda: None)
            cleanup.setattr(torch.accelerator, "empty_host_cache", lambda: None)
            llm.llm_engine.engine_core.shutdown()


@pytest.mark.feature("KV cache offloading")
def test_keeps_preempted_requests(
    tokenizer: PreTrainedTokenizerBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requests preempted for cache blocks resume from the blocks they stored in host memory and
    answer as an engine without offloading answers them."""
    # A batch-invariant load computes a row alike whatever calls wrote the blocks it reads.
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    orders = (*NINE, NOTES[2] + NOTES[0] + NOTES[1], NOTES[0] + NOTES[2] + NOTES[1])
    requests = [TokensPrompt(prompt_token_ids=ask(tokenizer, people)) for people in orders]

    def start(blocks: int | None = None, offloading: float | None = None) -> LLM:
        """An engine that runs every request at once, in `blocks` cache blocks and with
        `offloading` GiB of host memory if set."""
        return LLM(
            CHECKPOINT,
            dtype="bfloat16",
            generation_config="vllm",
            max_model_len=CONTEXT,
            max_num_seqs=len(requests),
            max_num_batched_tokens=64,
            num_gpu_blocks_override=blocks,
            disable_log_stats=False,
            kv_offloading_size=offloading,
        )

    def answers(llm: LLM) -> list[list[int]]:
        return [list(output.outputs[0].token_ids) for output in llm.generate(requests, greedy())]

    oracle = start()
    try:
        expected = answers(oracle)
    finally:
        oracle.llm_engine.engine_core.shutdown()
    llm = start(blocks=40, offloading=1)
    try:
        assert answers(llm) == expected
        assert counter(llm, PREEMPTED) > 0
        assert counter(llm, LOADED) > 0
    finally:
        llm.llm_engine.engine_core.shutdown()
