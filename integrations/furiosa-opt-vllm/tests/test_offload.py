"""vLLM's KV cache offloading keeps evicted prompt blocks in host memory, and a prompt answers with
them as with its blocks in the device cache.

- The transfer tracker reports each ended job once and drains submitted copies after a submission
  failure, here over stand-in completions.
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
)

from furiosa_opt_engine.cache_transfer import (
    submit as submit_transfers,
)
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


def test_vllm_routes_preserve_transfer_page_order() -> None:
    """The engine groups consecutive page pairs before sorting slots within each DMA batch."""
    host = worker.Host.__new__(worker.Host)
    host.blocks = 8
    host.group_indices = {4: 0}
    groups = [
        worker.GroupWork(
            group_id=4,
            chunks=(
                worker.ChunkWork(
                    chunk_id=1,
                    device_blocks=(9, 3, 7),
                    host_slots=(2, 0, 1),
                ),
            ),
        )
    ]

    assert host.routed_pages(groups) == [
        (10, [(0, 9)]),
        (8, [(0, 3)]),
        (9, [(0, 7)]),
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
    cache_spec = SimpleNamespace(pools=[pool], groups=[engine_group])
    layer_spec = FullAttentionSpec(block_size=2, num_kv_heads=1, head_size=2, dtype=torch.uint8)
    cache_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layers.0"], kv_cache_spec=layer_spec)],
    )
    layer_owner = {0: (0, 0)}
    view = worker._metadata_cache_views(cache_spec, cache_config, layer_owner)["layers.0"]
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


def test_canonical_layout_requires_an_identity_page_mapping(monkeypatch) -> None:
    class FakeCopier:
        slot_bytes = 8
        layer_ranges = [[(0, 8)]]

        def __init__(self, *_args: object) -> None:
            pass

        def shutdown(self) -> None:
            pass

    class FakeHostCache:
        def __init__(self, _copier: FakeCopier, _slots: int) -> None:
            pass

    monkeypatch.setattr(worker.engine, "Copier", FakeCopier)
    monkeypatch.setattr(worker.engine, "HostCache", FakeHostCache)
    monkeypatch.setattr(worker.host_pool, "register", lambda *_args: None)
    monkeypatch.setattr(worker.host_pool, "unregister", lambda *_args: None)

    group = KVCacheGroupSpec(
        layer_names=["layers.0"],
        kv_cache_spec=FullAttentionSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=2,
            dtype=torch.uint8,
        ),
    )
    cache_config = KVCacheConfig(num_blocks=4, kv_cache_tensors=[], kv_cache_groups=[group])
    pool = SimpleNamespace(
        block=2,
        bytes=8,
        element_bytes=1,
        shape=[2, 2, 1, 2],
        strides=[4, 2, 2, 1],
    )
    cache = SimpleNamespace(
        blocks=4,
        layers=[bytearray(32)],
        spec=SimpleNamespace(
            pools=[pool],
            groups=[SimpleNamespace(pool=0, layers=[0], logical_group=0)],
        ),
    )
    spec = SimpleNamespace(
        blocks_per_chunk=2,
        num_chunks=2,
        kv_bytes_per_chunk=16,
        config=SimpleNamespace(
            canonical_layout=True,
            parallel=SimpleNamespace(is_parallelism_agnostic=False),
            engine_id="canonical-layout-test",
        ),
    )

    def canonical_mapping(canonical_offset: int) -> SimpleNamespace:
        return SimpleNamespace(
            local_page_size_bytes=8,
            canonical_page_size_bytes=8,
            num_writers=1,
            writer_index=0,
            runs=[
                SimpleNamespace(
                    local_offset=0,
                    canonical_offset=canonical_offset,
                    local_stride=1,
                    canonical_stride=1,
                )
            ],
        )

    monkeypatch.setattr(
        worker,
        "derive_canonical_mappings",
        lambda *_args: {"layers.0": canonical_mapping(0)},
    )
    host = worker.Host(cache, 1, spec, SimpleNamespace(), cache_config, (0,))
    host.shutdown()

    monkeypatch.setattr(
        worker,
        "derive_canonical_mappings",
        lambda *_args: {"layers.0": canonical_mapping(1)},
    )
    with pytest.raises(ValueError, match="not the engine page byte-for-byte"):
        worker.Host(cache, 1, spec, SimpleNamespace(), cache_config, (0,))


def test_vllm_tiering_uses_engine_owned_primary_layout(tmp_path) -> None:
    from vllm.v1.kv_offload.tiering.manager import TieringOffloadingManager

    from furiosa_opt_vllm.offload.spec import _PinnedPrimaryTier
    from furiosa_opt_vllm.offload.storage import FuriosaFileSystemTierManager

    config = SimpleNamespace(
        canonical_layout=False,
        replicated_layout=False,
        model=SimpleNamespace(name="engine-cache-test", dtype="bfloat16"),
        cache=SimpleNamespace(tokens_per_hash=16),
        groups=[],
        parallel=SimpleNamespace(
            tp_size=1,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            rank=0,
            is_parallelism_agnostic=False,
        ),
        engine_id="engine-cache-test",
    )
    spec = SimpleNamespace(
        num_chunks=2,
        kv_bytes_per_chunk=4096,
        eviction_policy="lru",
        cache_policy_module_path=None,
        kv_events_config=SimpleNamespace(enable_kv_cache_events=False),
        blocks_per_chunk=2,
        config=config,
    )
    primary = _PinnedPrimaryTier(spec)
    storage = None
    try:
        view = primary.get_kv_memoryview()
        assert view.strides == (4096, 1)

        storage = FuriosaFileSystemTierManager(
            spec,
            view,
            "fs",
            root_dir=str(tmp_path),
        )
        manager = TieringOffloadingManager(primary, [storage])

        assert manager.primary_tier is primary
        assert storage._primary_kv_view is view
        assert storage.block_size_bytes == 4096
    finally:
        if storage is not None:
            storage.shutdown()
        primary.shutdown()


def test_vllm_chunk_store_round_trips_layer_major_pages(tmp_path) -> None:
    from furiosa_opt_engine.cache_file import AlignedFileIO
    from furiosa_opt_vllm.offload.pool import FilePageSegment
    from furiosa_opt_vllm.offload.storage import (
        FuriosaFileSystemTierManager,
        _ChunkStore,
        _LoadProgress,
    )

    class HostCache:
        def __init__(self, *pages: bytes) -> None:
            self.pages = [bytearray(page) for page in pages]

        def access(self, index: int) -> memoryview:
            return memoryview(self.pages[index])

    file_io = AlignedFileIO(str(tmp_path), use_direct=False)
    path = tmp_path / "cache" / "chunk.bin"
    layout = (FilePageSegment(0, 3), FilePageSegment(3, 4))
    cache = HostCache(b"abcWXYZ", b"def1234")
    chunk = _ChunkStore(file_io, str(path), 7, 2, 16, layout)

    chunk.write(cache, range(2))

    assert path.read_bytes() == b"abcdefWXYZ1234\0\0"
    restored = HostCache(bytes(7), bytes(7))
    manager = FuriosaFileSystemTierManager.__new__(FuriosaFileSystemTierManager)
    manager._file_io = file_io
    manager._load_progress = {}
    progress = _LoadProgress(job_id=9, files=1)
    manager._load_chunk(str(path), 0, restored, range(2), layout, progress)
    assert [bytes(page) for page in restored.pages] == [b"abcWXYZ", b"def1234"]
    assert manager._load_progress == {9: 1}


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
