"""A prompt prefilled by one server and decoded by another, whose NIXL connector moves the prompt's
cache through host buffers, continues as a server without a connector continues it.

- The runner's copy between the engine cache and NIXL's host buffers moves every named block in
  NIXL's order and ends before it returns; a failed copy raises once every copy submitted with it
  ended. Here over stand-in buffers whose bytes land only when a copy is waited for.
- The offloading host tier, over the same stand-ins, copies the cache the runner holds at each copy,
  as the copy above does.
- Under vLLM's MultiConnector, the runner gives the plugin's offloading connector its host tier
  and a connector staging blocks in host buffers the views and the copy above, and a step reports
  each connector's received requests and jobs.
- On RNGD, a decode server's greedy tokens of a prompt a prefill server sent equal a server's own,
  and the decode server took the prompt's blocks from its connector. Under a MultiConnector that
  also offloads, the decode server answers the prompt the same once its blocks were evicted and
  reloaded from host memory.
"""

from __future__ import annotations

import json
import os
import secrets
import urllib.request
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, Literal, cast, final

import numpy as np
import pytest
import torch
from conftest import post, serve
from typing_extensions import override
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer import kv_transfer_state
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorMetadata,
    MultiKVConnectorWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    TransferJob,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.openai.completion.protocol import CompletionRequest, CompletionResponse
from vllm.utils import random_uuid
from vllm.utils.network_utils import get_open_port
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.kv_offload.base import GPULoadStoreSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

import furiosa_opt_engine as engine
from furiosa_opt_vllm import connector, offload, paging
from furiosa_opt_vllm.decoder import Decoder
from furiosa_opt_vllm.model_runner import FuriosaModelRunner
from furiosa_opt_vllm.offload import worker

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from contextlib import AbstractContextManager
    from http.client import HTTPResponse
    from pathlib import Path

    from numpy.typing import NDArray
    from pydantic import JsonValue
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import CopyBlocksOp
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

    from furiosa_opt_vllm.checkpoint import Location
    from furiosa_opt_vllm.runner import Runner

# A cache of `BLOCKS` blocks of `TOKENS` tokens in each of `LAYERS` layers, `HEADS` KV heads of
# `WIDTH` BF16 values each; a request holds at most `LENGTH` tokens.
BLOCKS: Final = 8
TOKENS: Final = 4
HEADS: Final = 2
WIDTH: Final = 8
LAYERS: Final = 2
LENGTH: Final = 16
SPEC: Final = FullAttentionSpec(
    block_size=TOKENS, num_kv_heads=HEADS, head_size=WIDTH, dtype=torch.bfloat16
)
NAMES: Final = tuple(f"layers.{index}" for index in range(LAYERS))
CONFIG: Final = KVCacheConfig(
    num_blocks=BLOCKS,
    kv_cache_tensors=[],
    kv_cache_groups=[KVCacheGroupSpec(list(NAMES), SPEC)],
)


@final
class Copy:
    """A stand-in for the engine's `Transfer`: its bytes land when it is first waited for, unless
    the test made it fail."""

    def __init__(self, land: Callable[[], None], failure: RuntimeError | None) -> None:
        self.land = land
        self.failure = failure
        self.ended = False

    def wait(self) -> None:
        if not self.ended:
            self.ended = True
            if self.failure is None:
                self.land()
        if self.failure is not None:
            raise self.failure

    def done(self) -> bool:
        if self.ended and self.failure is not None:
            raise self.failure
        return self.ended


@dataclass
class Chip:
    """The copies the stand-in buffers submit, in order: copy `fail` fails when waited for, and
    submitting copy `refuse` raises."""

    fail: int | None = None
    refuse: int | None = None
    copies: list[Copy] = field(default_factory=list[Copy])

    def submit(self, land: Callable[[], None]) -> Copy:
        if len(self.copies) == self.refuse:
            raise RuntimeError("submission failed")
        failure = RuntimeError("copy failed") if len(self.copies) == self.fail else None
        copy = Copy(land, failure)
        self.copies.append(copy)
        return copy

    def settled(self) -> bool:
        """Whether every submitted copy ended."""
        return all(copy.ended for copy in self.copies)


class Host(np.ndarray[tuple[int], np.dtype[np.uint8]]):
    """A stand-in for the engine's `HostBuffer` over NumPy bytes, whose copies `chip` submits."""

    chip: Chip | None = None

    @override
    def __array_finalize__(self, obj: object) -> None:
        self.chip = cast("Chip | None", getattr(obj, "chip", None))

    def slice(self, offset: int, size: int) -> Host:
        assert 0 <= offset and size > 0 and offset + size <= len(self)
        return cast("Host", self[offset : offset + size])

    def to_hbm(self, buffer: Hbm) -> Copy:
        assert len(buffer) == len(self) and self.chip is not None
        return self.chip.submit(lambda: np.copyto(buffer.data, self))


@dataclass(frozen=True)
class Hbm:
    """A stand-in for the engine's `HbmBuffer` over NumPy bytes, whose copies `chip` submits."""

    data: NDArray[np.uint8]
    chip: Chip

    def __len__(self) -> int:
        return len(self.data)

    def slice(self, offset: int, size: int) -> Hbm:
        assert offset % 8 == 0 and size % 8 == 0 and offset + size <= len(self)
        return Hbm(self.data[offset : offset + size], self.chip)

    def to_host(self, host: Host) -> Copy:
        assert len(host) == len(self)
        return self.chip.submit(lambda: np.copyto(host, self.data))


@dataclass(frozen=True)
class Group:
    layers: range


@dataclass(frozen=True)
class Spec:
    groups: list[Group]


@dataclass(frozen=True)
class Cache:
    """A stand-in for the engine's cache: `LAYERS` layers of `BLOCKS` blocks in one group."""

    layers: list[Hbm]
    blocks: int = BLOCKS
    spec: Spec = Spec([Group(range(LAYERS))])

    @classmethod
    def on(cls, chip: Chip) -> Cache:
        return cls([Hbm(np.zeros(BLOCKS * SPEC.page_size_bytes, np.uint8), chip) for _ in NAMES])

    def blocks_of(self, layer: int) -> torch.Tensor:
        """Layer `layer`'s blocks in the engine's order: `[block, token, key or value, head,
        width]`."""
        data = torch.from_numpy(self.layers[layer].data).view(torch.bfloat16)
        return data.view(BLOCKS, TOKENS, 2, HEADS, WIDTH)


@dataclass
class Holder:
    """A runner's part that the copy reads: the cache it holds now."""

    cache: Cache | None


@pytest.fixture
def chip(monkeypatch: pytest.MonkeyPatch) -> Chip:
    """The copies of this test's stand-in buffers; host buffers the connector allocates are
    stand-ins too."""
    chip = Chip()

    def allocate(size: int) -> Host:
        host = np.zeros(size, np.uint8).view(Host)
        host.chip = chip
        return host

    monkeypatch.setattr(engine, "HostBuffer", allocate)
    return chip


def layers(holder: Holder) -> connector.Layers:
    # The copy reads only the cache a runner holds.
    return connector.Layers(cast("Runner", cast("object", holder)), CONFIG, LENGTH)


def nixl_order(block: torch.Tensor) -> torch.Tensor:
    """An engine-order block `[token, key or value, head, width]` as NIXL holds it: each head's
    tokens, a token's key then its value."""
    return torch.stack(
        [
            torch.stack([torch.cat([block[t, 0, h], block[t, 1, h]]) for t in range(TOKENS)])
            for h in range(HEADS)
        ]
    )


def random_blocks(seed: int, shape: tuple[int, ...]) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator).to(torch.bfloat16)


def host_buffers(views: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Host buffers of each registered layer's shape, as NIXL allocates them, zeroed."""
    return {name: torch.zeros(view.shape, dtype=view.dtype) for name, view in views.items()}


def test_registers_layers_as_blocks_of_heads(chip: Chip) -> None:
    """NIXL sizes its host buffers from each layer's registered tensor, which holds no memory:
    `[block, head, token, key and value]` of the cache's dtype."""
    views = layers(Holder(Cache.on(chip))).views
    assert set(views) == set(NAMES)
    for view in views.values():
        assert view.device.type == "meta"
        assert view.dtype == torch.bfloat16
        assert view.shape == (BLOCKS, HEADS, TOKENS, 2 * WIDTH)


def test_sends_every_named_block_in_nixl_order(chip: Chip) -> None:
    """A send copies each named block of each layer, after its copies ended, into NIXL's host
    buffer of that layer in NIXL's order, and no other block."""
    cache = Cache.on(chip)
    for index in range(LAYERS):
        _ = cache.blocks_of(index).copy_(random_blocks(index, (BLOCKS, TOKENS, 2, HEADS, WIDTH)))
    copier = layers(Holder(cache))
    sent = host_buffers(copier.views)
    blocks = [5, 6, 2]
    copier.copy(copier.views, sent, blocks, blocks, "d2h")
    for index, name in enumerate(NAMES):
        for block in range(BLOCKS):
            expected = (
                nixl_order(cache.blocks_of(index)[block])
                if block in blocks
                else torch.zeros(HEADS, TOKENS, 2 * WIDTH, dtype=torch.bfloat16)
            )
            assert torch.equal(sent[name][block], expected), (name, block)


def test_receives_every_named_block_before_returning(chip: Chip) -> None:
    """A receive copies each named block of NIXL's host buffers into the engine's order of the
    cache, and its copies ended once it returns; other blocks keep their bytes."""
    cache = Cache.on(chip)
    copier = layers(Holder(cache))
    received = {
        name: random_blocks(index, (BLOCKS, HEADS, TOKENS, 2 * WIDTH))
        for index, name in enumerate(NAMES)
    }
    blocks = [1, 2, 7]
    copier.copy(received, copier.views, blocks, blocks, "h2d")
    assert chip.settled()
    for index, name in enumerate(NAMES):
        for block in range(BLOCKS):
            actual = nixl_order(cache.blocks_of(index)[block])
            expected = (
                received[name][block]
                if block in blocks
                else torch.zeros(HEADS, TOKENS, 2 * WIDTH, dtype=torch.bfloat16)
            )
            assert torch.equal(actual, expected), (name, block)


Direction = Literal["h2d", "d2h"]


def copy(copier: connector.Layers, direction: Direction) -> None:
    """Copy blocks 3, 4 and 0 of every layer in `direction`."""
    host = host_buffers(copier.views)
    blocks = [3, 4, 0]
    match direction:
        case "d2h":
            copier.copy(copier.views, host, blocks, blocks, direction)
        case "h2d":
            copier.copy(host, copier.views, blocks, blocks, direction)


@pytest.mark.parametrize("direction", ["h2d", "d2h"])
def test_failed_copy_raises_once_every_copy_ended(chip: Chip, direction: Direction) -> None:
    """A copy that fails raises from the call, once every copy the call submitted ended."""
    copier = layers(Holder(Cache.on(chip)))
    chip.fail = 1
    with pytest.raises(RuntimeError, match="copy failed"):
        copy(copier, direction)
    assert len(chip.copies) > 2
    assert chip.settled()


@pytest.mark.parametrize("direction", ["h2d", "d2h"])
def test_failed_submission_raises_once_submitted_copies_ended(
    chip: Chip, direction: Direction
) -> None:
    """A copy that fails to submit raises from the call, ahead of a submitted copy's own failure,
    once the copies submitted before it ended."""
    copier = layers(Holder(Cache.on(chip)))
    chip.fail, chip.refuse = 0, 2
    with pytest.raises(RuntimeError, match="submission failed"):
        copy(copier, direction)
    assert len(chip.copies) == 2
    assert chip.settled()


def test_failed_staging_raises_once_submitted_copies_ended(chip: Chip) -> None:
    """A receive whose host buffer of a later layer lacks a named block raises from the call, once
    the copies it submitted for the earlier layers ended."""
    copier = layers(Holder(Cache.on(chip)))
    received = host_buffers(copier.views)
    received[NAMES[1]] = received[NAMES[1]][:2]
    with pytest.raises(IndexError):
        copier.copy(received, copier.views, [3, 4, 0], [3, 4, 0], "h2d")
    assert chip.copies
    assert chip.settled()


def test_copies_the_cache_waking_allocates(chip: Chip) -> None:
    """After sleep mode frees the cache and waking allocates another, a receive writes the cache
    the runner holds, not the freed one."""
    freed = Cache.on(chip)
    holder = Holder(freed)
    copier = layers(holder)
    holder.cache = None
    woken = Cache.on(chip)
    holder.cache = woken
    received = {name: random_blocks(9, (BLOCKS, HEADS, TOKENS, 2 * WIDTH)) for name in NAMES}
    copier.copy(received, copier.views, [4], [4], "h2d")
    assert torch.equal(nixl_order(woken.blocks_of(0)[4]), received[NAMES[0]][4])
    assert not freed.layers[0].data.any()


def test_host_tier_copies_the_cache_waking_allocates(chip: Chip) -> None:
    """After sleep mode frees the cache and waking allocates another, the offloading host tier
    stores and loads blocks of the cache the runner holds, not the freed one."""
    freed = Cache.on(chip)
    holder = Holder(freed)
    # The host tier reads only the cache a runner holds.
    tier = worker.Host(cast("Runner", cast("object", holder)), 1, 1, 2)
    holder.cache = None
    woken = Cache.on(chip)
    holder.cache = woken
    stored = random_blocks(5, (TOKENS, 2, HEADS, WIDTH))
    _ = woken.blocks_of(0)[4].copy_(stored)
    _ = tier.submit_store(1, GPULoadStoreSpec([4], [1], [0]), CPULoadStoreSpec([0]))
    tier.wait({1})
    _ = tier.submit_load(2, CPULoadStoreSpec([0]), GPULoadStoreSpec([6], [1], [0]))
    tier.wait({2})
    assert torch.equal(woken.blocks_of(0)[6], stored)
    assert not freed.layers[0].data.any()


# Through vLLM's MultiConnector


@final
class Plan(KVConnectorMetadata):
    """A step's metadata for `Staged`, which plans no transfer."""


@final
class Staged(KVConnectorBase_V1, SupportsHMA):
    """A connector that stages blocks in host buffers as NIXL does, which this host lacks: it keeps
    the caches and the copy the runner registers, and reports `received` as received."""

    def __init__(
        self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        self.caches: dict[str, torch.Tensor] = {}
        self.copy: CopyBlocksOp | None = None
        self.received: set[str] = set()

    @override
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self.caches = kv_caches

    @override
    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp) -> None:
        self.copy = copy_operation

    @override
    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        received, self.received = self.received, set()
        return set(), received

    @override
    def start_load_kv(self, forward_context: ForwardContext, **kwargs: object) -> None:
        pass

    @override
    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    @override
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: object,
    ) -> None:
        pass

    @override
    def wait_for_save(self) -> None:
        pass

    @override
    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int, bool]:
        return 0, False

    @override
    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ) -> None:
        pass

    @override
    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        return Plan()

    @override
    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, object] | None]:
        return False, None


@dataclass(frozen=True)
class Connected:
    """A runner whose engine cache vLLM's MultiConnector connects through `staged` and the
    plugin's `offloading` connector, in that order."""

    runner: FuriosaModelRunner
    cache: Cache
    staged: Staged
    offloading: offload.RNGDOffloadingConnector

    @classmethod
    def on(cls, chip: Chip, served: Path, monkeypatch: pytest.MonkeyPatch) -> Connected:
        """The runner after vLLM's worker initialized its cache, a stand-in on `chip`."""
        config = EngineArgs(
            model=str(served),
            dtype="bfloat16",
            skip_tokenizer_init=True,
            block_size=TOKENS,
            kv_offloading_size=1,
        ).create_engine_config()
        # Set past the platform's check, which admits NIXL in place of `Staged`.
        config.kv_transfer_config = KVTransferConfig(
            kv_connector="MultiConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "connectors": [
                    {
                        "kv_connector": "Staged",
                        "kv_connector_module_path": __name__,
                        "kv_role": "kv_both",
                    },
                    {
                        "kv_connector": "RNGDOffloadingConnector",
                        "kv_connector_module_path": offload.__name__,
                        "kv_role": "kv_both",
                        "kv_connector_extra_config": {"cpu_bytes_to_use": 1 << 16},
                    },
                ]
            },
        )
        multi = KVConnectorFactory.create_connector(config, KVConnectorRole.WORKER, CONFIG)
        assert isinstance(multi, MultiConnector)
        staged, offloading = multi.sub_connectors
        assert isinstance(staged, Staged)
        assert isinstance(offloading, offload.RNGDOffloadingConnector)
        monkeypatch.setattr(kv_transfer_state, "_KV_CONNECTOR_AGENT", multi)
        runner = FuriosaModelRunner(config)
        # The connectors read only the parallel chips of the engine's decoder.
        unread = cast("object", None)
        runner.model = Decoder(
            cast("engine.Decoder", cast("object", SimpleNamespace(parallel=1))),
            None,
            cast("Location", unread),
            cast("engine.LoadOptions", unread),
            cast("engine.Device", unread),
            0,
        )
        cache = Cache.on(chip)

        def initialize(runner: Runner, _: KVCacheConfig) -> None:
            runner.cache = cast("engine.decoder.Cache", cast("object", cache))

        monkeypatch.setattr(paging, "initialize", initialize)
        runner.initialize_kv_cache(CONFIG)
        return cls(runner, cache, staged, offloading)

    def step(self, offloading: OffloadingConnectorMetadata) -> KVConnectorOutput:
        """A step without tokens, as vLLM's runner drives the connector around it, whose metadata
        gives the offloading connector `offloading`."""
        metadata = MultiKVConnectorMetadata((Plan(), offloading))
        scheduled = replace(SchedulerOutput.make_empty(), kv_connector_metadata=metadata)
        output = self.runner.kv_connector.no_forward(scheduled).kv_connector_output
        assert output is not None
        return output


@pytest.mark.usefixtures("platform_env")
@pytest.mark.feature("KV cache offloading")
def test_connects_the_cache_to_each_child_connector(
    chip: Chip, served: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under vLLM's MultiConnector, the plugin's offloading connector copies the engine cache
    through its host tier, and a connector staging blocks in host buffers takes the cache's
    NIXL views and the engine's copy."""
    connected = Connected.on(chip, served, monkeypatch)
    assert isinstance(connected.offloading.worker.worker, worker.Host)
    staged = connected.staged
    assert {name: (view.shape, view.device.type) for name, view in staged.caches.items()} == {
        name: ((BLOCKS, HEADS, TOKENS, 2 * WIDTH), "meta") for name in NAMES
    }
    cache = connected.cache
    _ = cache.blocks_of(0).copy_(random_blocks(3, (BLOCKS, TOKENS, 2, HEADS, WIDTH)))
    sent = host_buffers(staged.caches)
    assert staged.copy is not None
    staged.copy(staged.caches, sent, [2], [2], "d2h")
    assert torch.equal(sent[NAMES[0]][2], nixl_order(cache.blocks_of(0)[2]))


@pytest.mark.usefixtures("platform_env")
@pytest.mark.feature("KV cache offloading")
def test_reports_each_child_connectors_transfers(
    chip: Chip, served: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under vLLM's MultiConnector, a step reports each connector's received requests: the staging
    connector's at once, and an offloaded load's once its copies ended, with its job in the
    offloading connector's report alone."""
    connected = Connected.on(chip, served, monkeypatch)
    connected.staged.received = {"remote"}
    load = TransferJob("local", CPULoadStoreSpec([0]), GPULoadStoreSpec([1], [1], [0]))
    first = connected.step(OffloadingConnectorMetadata({3: load}, {}))
    assert first.finished_recving == {"remote"}
    assert first.kv_connector_worker_meta is None
    for copy in chip.copies:
        copy.wait()
    second = connected.step(OffloadingConnectorMetadata({}, {}))
    assert second.finished_recving == {"local"}
    report = second.kv_connector_worker_meta
    assert isinstance(report, MultiKVConnectorWorkerMetadata)
    staged, offloaded = report.metadata
    assert staged is None
    assert isinstance(offloaded, OffloadingWorkerMetadata)
    assert offloaded.completed_jobs == {3: 1}


# On RNGD

MODEL: Final = "Qwen/Qwen3-0.6B"
# Several cache blocks, the last one partial. A run's own words keep a chip's memory from holding
# its cache from an earlier run, which a missed transfer would then read.
PROMPT: Final = f"Run {secrets.token_hex(8)}. " + " ".join(
    f"Item {i} of the list is a word for the color of the sky." for i in range(24)
)
ANSWER: Final = 16


def metric(url: str, name: str) -> float:
    """The sum of the server's samples of metric `name`, labelled or not."""
    response: HTTPResponse = urllib.request.urlopen(f"{url}/metrics", timeout=30)
    with response:
        lines = response.read().decode().splitlines()
    return sum(
        float(line.rsplit(" ", 1)[1])
        for line in lines
        if line.startswith((f"{name} ", f"{name}{{"))
    )


@dataclass(frozen=True)
class Routed:
    """One leg of a request as vLLM's disaggregation proxies route it: the request ID both legs
    share, by which NIXL's push connector pairs a decode with its prefill, and the leg's KV
    connector parameters."""

    request_id: str
    transfer: dict[str, JsonValue]


def complete(
    url: str,
    model: str,
    tokens: int,
    routed: Routed | None = None,
    *,
    headers: Mapping[str, str] | None = None,
) -> CompletionResponse:
    """The greedy completion of `PROMPT` under `model` by the server at `url`, its tokens listed,
    routed as `routed` says, with `headers`."""
    body: dict[str, JsonValue] = {
        "model": model,
        "prompt": PROMPT,
        "max_tokens": tokens,
        "temperature": 0,
        "ignore_eos": True,
        "return_token_ids": True,
    }
    if routed is not None:
        body |= {"request_id": routed.request_id, "kv_transfer_params": routed.transfer}
    request = CompletionRequest.model_validate(body)
    return post(f"{url}/v1/completions", request, CompletionResponse, headers=headers)


def generated(response: CompletionResponse) -> list[int]:
    tokens = response.choices[0].token_ids
    assert tokens is not None
    return tokens


@dataclass(frozen=True)
class Case:
    """A served configuration: NIXL's connector, the flags of every server and those of the decode
    server alone, the chips the decode server takes, and whether its requests run an adapter."""

    connector: str = "NixlConnector"
    flags: tuple[str, ...] = ()
    decode: tuple[str, ...] = ()
    chips: int = 1
    adapted: bool = False
    # Sleep mode frees and allocates the decode server's cache before the prompt arrives.
    slept: bool = False


CASES: Final = {
    "pull": Case(),
    "push": Case(connector="NixlPushConnector"),
    "lora": Case(flags=("--enable-lora", "--max-lora-rank", "32"), adapted=True),
    "fp8": Case(flags=("--quantization", "fp8_per_tensor")),
    "ngram": Case(
        flags=(
            "--speculative-config",
            json.dumps({"method": "ngram", "num_speculative_tokens": 3, "prompt_lookup_max": 3}),
        )
    ),
    "data-parallel": Case(decode=("--data-parallel-size", "2"), chips=2),
    "sleep": Case(decode=("--enable-sleep-mode",), slept=True),
}


@pytest.fixture(scope="module")
def adapter(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A PEFT adapter of rank 32 on every projection, whose `0.02 * N(0, 1)` factors change the
    greedy tokens."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    projections = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    model = get_peft_model(
        AutoModelForCausalLM.from_pretrained(MODEL),
        LoraConfig(r=32, lora_alpha=45, target_modules=projections),
    )
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, value in model.named_parameters():
            if ".lora_" in name:
                _ = value.copy_(0.02 * torch.randn(value.shape, generator=generator))
    directory = tmp_path_factory.mktemp("adapter")
    model.save_pretrained(str(directory))
    return directory


@pytest.mark.feature("Disaggregated prefill and KV connectors (NIXL, Mooncake, MoRIIO, LMCache)")
@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_decodes_a_remote_prefill(
    case: Case, tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
) -> None:
    """The decoder's tokens of a prompt whose cache the prefill server sent through NIXL equal a
    server's own greedy tokens of it, the decoder took the prompt's blocks from its connector, and
    NIXL moved bytes between the servers' host buffers."""
    chips = os.environ["FURIOSA_VISIBLE_CHIPS"].split(",")
    assert len(chips) >= 1 + case.chips, chips
    flags = [MODEL, "--max-model-len", "1024", *case.flags]
    model = MODEL
    if case.adapted:
        path = cast("Path", request.getfixturevalue("adapter"))
        flags += ["--lora-modules", f"adapted={path}"]
        model = "adapted"
    with serve(flags, tmp_path_factory, {"FURIOSA_VISIBLE_CHIPS": chips[0]}) as single:
        expected = generated(complete(single, model, ANSWER))

    def connected(
        role: str, visible: list[str], extra: tuple[str, ...]
    ) -> AbstractContextManager[str]:
        transfer = {"kv_connector": case.connector, "kv_role": role, "kv_buffer_device": "cpu"}
        env = {
            "FURIOSA_VISIBLE_CHIPS": ",".join(visible),
            "VLLM_NIXL_SIDE_CHANNEL_HOST": "127.0.0.1",
            "VLLM_NIXL_SIDE_CHANNEL_PORT": str(get_open_port()),
            "VLLM_SERVER_DEV_MODE": "1",
        }
        arguments = [*flags, *extra, "--kv-transfer-config", json.dumps(transfer)]
        return serve(arguments, tmp_path_factory, env)

    with (
        connected("kv_producer", chips[:1], ()) as prefill,
        connected("kv_consumer", chips[1 : 1 + case.chips], case.decode) as decode,
    ):
        if case.slept:
            for action in ("/sleep?level=1", "/wake_up"):
                answer: HTTPResponse = urllib.request.urlopen(
                    urllib.request.Request(f"{decode}{action}", method="POST"), timeout=300
                )
                with answer:
                    assert answer.status == 200
        # Routed as vLLM's disaggregation proxies route a request: both legs under one ID.
        pair = random_uuid()
        prefilled = Routed(pair, {"do_remote_decode": True, "do_remote_prefill": False})
        sent = complete(prefill, model, 1, prefilled)
        assert sent.kv_transfer_params is not None
        decoded = generated(complete(decode, model, ANSWER, Routed(pair, sent.kv_transfer_params)))
        taken = metric(decode, "vllm:external_prefix_cache_hits_total")
        moved = metric(prefill, "vllm:nixl_bytes_transferred_sum") + metric(
            decode, "vllm:nixl_bytes_transferred_sum"
        )
    assert decoded == expected
    assert len(expected) == ANSWER
    assert taken > 0
    assert moved > 0


# Notes longer than `PROMPT`, whose blocks evict most of its blocks from a decode server of
# `DEVICE_BLOCKS` blocks.
FILLER: Final = f"Run {secrets.token_hex(8)}. " + " ".join(
    f"Line {i} names a fruit that grows on a tall tree." for i in range(30)
)
DEVICE_BLOCKS: Final = 34


@dataclass(frozen=True)
class Composed:
    """A decode server's MultiConnector over NIXL and the plugin's offloading connector: the NIXL
    case of both servers, and whether the offloading connector comes first."""

    case: Case = field(default_factory=Case)
    offloading_first: bool = False


COMPOSED: Final = {
    "pull": Composed(),
    "push": Composed(CASES["push"]),
    "offloading-first": Composed(offloading_first=True),
    "data-parallel": Composed(CASES["data-parallel"]),
    "sleep": Composed(CASES["sleep"]),
}


@pytest.mark.feature("Disaggregated prefill and KV connectors (NIXL, Mooncake, MoRIIO, LMCache)")
@pytest.mark.feature("KV cache offloading")
@pytest.mark.parametrize("composed", COMPOSED.values(), ids=COMPOSED.keys())
def test_offloads_a_remote_prefill(
    composed: Composed, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A decode server whose MultiConnector runs NIXL and the plugin's offloading connector
    decodes a prompt the prefill server sent as a server decodes it alone, then answers it the same
    after other notes evicted its blocks, from the blocks its host tier stored."""
    case = composed.case
    chips = os.environ["FURIOSA_VISIBLE_CHIPS"].split(",")
    assert len(chips) >= 1 + case.chips, chips
    flags = [MODEL, "--max-model-len", "512", "--num-gpu-blocks-override", str(DEVICE_BLOCKS)]
    with serve(flags, tmp_path_factory, {"FURIOSA_VISIBLE_CHIPS": chips[0]}) as single:
        expected = generated(complete(single, MODEL, ANSWER))
    nixl = {"kv_connector": case.connector, "kv_buffer_device": "cpu"}
    offloading = {
        "kv_connector": "RNGDOffloadingConnector",
        "kv_connector_module_path": offload.__name__,
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"cpu_bytes_to_use": 1 << 30},
    }
    entries = [{**nixl, "kv_role": "kv_consumer"}, offloading]
    consumer = {
        "kv_connector": "MultiConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "connectors": entries[::-1] if composed.offloading_first else entries
        },
    }

    def connected(
        transfer: Mapping[str, object], visible: list[str], extra: tuple[str, ...]
    ) -> AbstractContextManager[str]:
        env = {
            "FURIOSA_VISIBLE_CHIPS": ",".join(visible),
            "VLLM_NIXL_SIDE_CHANNEL_HOST": "127.0.0.1",
            "VLLM_NIXL_SIDE_CHANNEL_PORT": str(get_open_port()),
            "VLLM_SERVER_DEV_MODE": "1",
        }
        arguments = [*flags, *extra, "--kv-transfer-config", json.dumps(transfer)]
        return serve(arguments, tmp_path_factory, env)

    # Every decode request goes to one replica, whose host tier alone stores the prompt's blocks.
    pinned = {"X-data-parallel-rank": "0"} if case.chips > 1 else None
    producer = {**nixl, "kv_role": "kv_producer"}
    with (
        connected(producer, chips[:1], ()) as prefill,
        connected(consumer, chips[1 : 1 + case.chips], case.decode) as decode,
    ):
        if case.slept:
            for action in ("/sleep?level=1", "/wake_up"):
                answer: HTTPResponse = urllib.request.urlopen(
                    urllib.request.Request(f"{decode}{action}", method="POST"), timeout=300
                )
                with answer:
                    assert answer.status == 200
        pair = random_uuid()
        prefilled = Routed(pair, {"do_remote_decode": True, "do_remote_prefill": False})
        sent = complete(prefill, MODEL, 1, prefilled)
        assert sent.kv_transfer_params is not None
        routed = Routed(pair, sent.kv_transfer_params)
        decoded = generated(complete(decode, MODEL, ANSWER, routed, headers=pinned))
        taken = metric(decode, "vllm:external_prefix_cache_hits_total")
        moved = metric(prefill, "vllm:nixl_bytes_transferred_sum") + metric(
            decode, "vllm:nixl_bytes_transferred_sum"
        )
        filled = CompletionRequest.model_validate(
            {"model": MODEL, "prompt": FILLER, "max_tokens": 1, "temperature": 0}
        )
        _ = post(f"{decode}/v1/completions", filled, CompletionResponse, headers=pinned)
        reloaded = generated(complete(decode, MODEL, ANSWER, headers=pinned))
        stored = metric(decode, "vllm:kv_offload_store_bytes_total")
        loaded = metric(decode, "vllm:kv_offload_load_bytes_total")
    assert decoded == expected
    assert reloaded == expected
    assert taken > 0
    assert moved > 0
    assert stored > 0
    assert loaded > 0
