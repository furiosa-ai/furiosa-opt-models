"""The offloading connector's worker side on the engine cache, as vLLM's offloading worker runs it.

- `Worker` starts a step's stores in the next step, ends the stores a step flushes before its calls,
  and reports a load's request once the load landed.
- `Host` keeps each layer's host rows in one host buffer, row after row; a row holds every chip's
  part of one block, chip after chip, and a host chunk is `blocks_per_chunk` rows of every layer.
  Blocks whose rows follow each other move in one copy.
- A copy returns once submitted: the device orders it after the calls and copies that wrote its
  bytes and before the ones that write or read them later. `Jobs` keeps each copy from its
  submission until the job reports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from time import monotonic
from typing import TYPE_CHECKING, Protocol, final

from typing_extensions import override
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
    TransferJob,
)
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

import furiosa_opt_engine as engine
from furiosa_opt_vllm.cache import runs

if TYPE_CHECKING:
    from collections.abc import Iterable

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
        OffloadingConnectorMetadata,
    )


@final
class Worker:
    """The connector's worker side over `host`, as vLLM's offloading worker runs it: a step's
    stores start in the next step, after the step's tokens are read."""

    def __init__(self, host: Host) -> None:
        self.host = host
        self.deferred: list[tuple[int, TransferJob]] = []
        # job_id -> the request whose blocks the load fills.
        self.loads: dict[int, str] = {}
        self.completed = OffloadingWorkerMetadata()

    def flush(self, metadata: OffloadingConnectorMetadata) -> None:
        """Submit the deferred stores and the step's stores that `metadata` flushes, then wait for
        the flushed jobs, whose blocks the step hands to other requests."""
        flushed = metadata.jobs_to_flush or set()
        self.deferred.extend(
            (job_id, job)
            for job_id in flushed
            if (job := metadata.store_jobs.pop(job_id, None)) is not None
        )
        self.submit()
        self.host.wait(flushed)

    def start(self, metadata: OffloadingConnectorMetadata) -> None:
        """Submit the deferred stores, then the loads of `metadata`."""
        self.submit()
        for job_id, job in metadata.load_jobs.items():
            assert isinstance(job.dst_spec, GPULoadStoreSpec)
            self.loads[job_id] = job.req_id
            _ = self.host.submit_load(job_id, job.src_spec, job.dst_spec)

    def defer(self, metadata: OffloadingConnectorMetadata) -> None:
        """Keep the stores of `metadata` for the next step."""
        self.deferred.extend(metadata.store_jobs.items())

    def submit(self) -> None:
        for job_id, job in self.deferred:
            assert isinstance(job.src_spec, GPULoadStoreSpec)
            _ = self.host.submit_store(job_id, job.src_spec, job.dst_spec)
        self.deferred.clear()

    def finished(self) -> set[str]:
        """The requests whose loads ended since the last call, each job's end recorded for
        `report`."""
        received: set[str] = set()
        for result in self.host.get_finished():
            request = self.loads.pop(result.job_id, None)
            stats = self.completed.transfer_stats
            assert result.transfer_size is not None and result.transfer_time is not None
            (stats.store if request is None else stats.load).record(
                result.transfer_size, result.transfer_time
            )
            self.completed.mark_completed(result.job_id)
            if request is not None:
                received.add(request)
        return received

    def report(self) -> OffloadingWorkerMetadata | None:
        """The jobs that ended since the last report, once each."""
        if not self.completed.completed_jobs:
            return None
        report, self.completed = self.completed, OffloadingWorkerMetadata()
        return report

    def shutdown(self) -> None:
        """Drop the stores not yet submitted, then end the submitted jobs."""
        self.deferred.clear()
        self.host.shutdown()


@final
class Host(OffloadingWorker):
    """The host tier of `cache`, whose layers span `chips` chips: `chunks` chunks of `blocks` rows
    of every layer, and the jobs that copy blocks between them."""

    def __init__(self, cache: engine.decoder.Cache, chips: int, chunks: int, blocks: int) -> None:
        self.layers = cache.layers
        # Bytes one block of each layer takes on each chip.
        self.sizes = [len(layer) // cache.blocks for layer in self.layers]
        self.chips = chips
        self.blocks = blocks
        self.rows = [engine.HostBuffer(chunks * blocks * size * chips) for size in self.sizes]
        self.jobs = Jobs()

    @override
    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        pieces = self.pieces(src_spec, dst_spec)
        transfers = (block.to_host(row) for block, row in pieces)
        self.jobs.submit(job_id, transfers, sum(len(row) for _, row in pieces))
        return True

    @override
    def submit_load(self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec) -> bool:
        pieces = self.pieces(dst_spec, src_spec)
        transfers = (row.to_hbm(block) for block, row in pieces)
        self.jobs.submit(job_id, transfers, sum(len(row) for _, row in pieces))
        return True

    @override
    def get_finished(self) -> list[TransferResult]:
        return self.jobs.finished()

    @override
    def wait(self, job_ids: set[int]) -> None:
        self.jobs.wait(job_ids)

    @override
    def shutdown(self) -> None:
        self.jobs.shutdown()

    def pieces(
        self, device: GPULoadStoreSpec, host: LoadStoreSpec
    ) -> list[tuple[engine.HbmBuffer, engine.HostBuffer]]:
        """Each layer's runs of the cache blocks that `device` names, beside the host rows `host`
        holds them in, in order: a group's first block takes its chunk's row of the block's index
        in the request."""
        assert isinstance(host, CPULoadStoreSpec)
        # The engine holds one cache group.
        (count,) = device.group_sizes
        (index,) = device.block_indices
        skip = index % self.blocks
        rows = list(
            islice(
                (
                    int(chunk) * self.blocks + offset
                    for chunk in host.block_ids
                    for offset in range(self.blocks)
                ),
                skip,
                skip + count,
            )
        )
        blocks = [int(block) for block in device.block_ids]
        assert len(rows) == len(blocks) == count
        moved = runs(blocks, rows, self.chips)
        return [
            (
                layer.slice(block * size, length * size),
                buffer.slice(row * size * self.chips, length * size * self.chips),
            )
            for layer, size, buffer in zip(self.layers, self.sizes, self.rows, strict=True)
            for block, row, length in moved
        ]


class _Transfer(Protocol):
    """A submitted copy as a job follows it: `done` tells without waiting whether it ended, `wait`
    waits for its end, and both raise a failed copy."""

    def done(self) -> bool: ...

    def wait(self) -> None: ...


@dataclass(frozen=True, slots=True)
class Job:
    """A submitted store or load: its copies, their bytes and when they were submitted."""

    transfers: tuple[_Transfer, ...]
    size: int
    started: float

    def done(self) -> bool:
        """Whether every copy ended, without waiting; a failed copy it reaches raises."""
        return all(transfer.done() for transfer in self.transfers)

    def settle(self) -> RuntimeError | None:
        """Wait until every copy ended, then return the first failure among them."""
        failure: RuntimeError | None = None
        for transfer in self.transfers:
            try:
                transfer.wait()
            except RuntimeError as error:
                failure = failure or error
        return failure


@final
class Jobs:
    """Submitted jobs, each reported once after its copies ended.

    A copy that fails raises from the call that finds it, once every copy that call waits for
    ended: it poisoned the memory it touched, and vLLM asserts that each job succeeded.
    """

    def __init__(self) -> None:
        self.running: dict[int, Job] = {}

    def submit(self, job_id: int, transfers: Iterable[_Transfer], size: int) -> None:
        """Run job `job_id`, whose copies move `size` bytes: `transfers` submits each copy as it
        is read, and the job keeps it at once. A copy that fails to submit raises once the copies
        submitted before it ended."""
        started = monotonic()
        submitted: list[_Transfer] = []
        try:
            for transfer in transfers:
                submitted.append(transfer)
        except RuntimeError:
            _ = Job(tuple(submitted), size, started).settle()
            raise
        self.running[job_id] = Job(tuple(submitted), size, started)

    def finished(self) -> list[TransferResult]:
        """The jobs whose copies ended since the last call, without waiting."""
        ended = [(job_id, job) for job_id, job in self.running.items() if job.done()]
        for job_id, _ in ended:
            del self.running[job_id]
        return [
            TransferResult(job_id, True, job.size, monotonic() - job.started)
            for job_id, job in ended
        ]

    def wait(self, job_ids: set[int]) -> None:
        """Wait until the running jobs of `job_ids` ended; `finished` still reports them."""
        self.drain(job_ids & self.running.keys())

    def shutdown(self) -> None:
        """Wait until every running job ended, then forget them."""
        try:
            self.drain(set(self.running))
        finally:
            self.running.clear()

    def drain(self, job_ids: set[int]) -> None:
        """Wait for each copy of the jobs of `job_ids`, then raise the first failure among them."""
        failures = [self.running[job_id].settle() for job_id in sorted(job_ids)]
        failure = next((error for error in failures if error is not None), None)
        if failure is not None:
            raise failure
