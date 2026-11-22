"""vLLM filesystem tier I/O directly over the engine's pinned cache slots."""

from __future__ import annotations

import functools
import logging
import os
import uuid
from collections.abc import Iterable
from threading import Lock
from typing import TYPE_CHECKING

from typing_extensions import override
from vllm.v1.kv_offload.tiering.base import JobResult, TransferJob
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager

from furiosa_opt_engine.cache_transfer import (
    DEFAULT_READ_THREADS,
    DEFAULT_WRITE_THREADS,
    AlignedFileIO,
)
from furiosa_opt_vllm.offload import pool as host_pool

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

    import furiosa_opt_engine as engine

logger = logging.getLogger(__name__)


def _noop() -> None:
    pass


class _ChunkStore:
    """Write one vLLM file as independently aligned cache-page transfers."""

    def __init__(self, file_io: AlignedFileIO, path: str, page_bytes: int, pages: int) -> None:
        self.file_io = file_io
        self.path = path
        self.temporary = f"{path}.{uuid.uuid4().hex}.tmp"
        self.remaining = pages
        self.failed = False
        self.lock = Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        file_io.prepare(self.temporary, page_bytes, pages)

    def write(self, index: int, slot: engine.HostSlot) -> None:
        with self.lock:
            failed = self.failed
        succeeded = False
        try:
            if not failed:
                self.file_io.write_page(self.temporary, index, slot)
                succeeded = True
        finally:
            self._finish(succeeded)

    def abort(self) -> None:
        try:
            os.remove(self.temporary)
        except OSError:
            pass

    def _finish(self, succeeded: bool) -> None:
        with self.lock:
            self.failed |= not succeeded
            self.remaining -= 1
            if self.remaining:
                return
            if self.failed:
                self.abort()
                return
            try:
                os.replace(self.temporary, self.path)
            except Exception:
                self.abort()
                raise


class _LoadProgress:
    """Report only the longest key prefix whose every cache page loaded successfully."""

    def __init__(self, job_id: int, files: int, pages: int) -> None:
        self.job_id = job_id
        self.failed = [False] * files
        self.remaining = [pages] * files
        self.prefix = 0
        self.lock = Lock()

    def page_done(self, progress: dict[int, int], file_index: int, succeeded: bool) -> None:
        with self.lock:
            self.failed[file_index] |= not succeeded
            self.remaining[file_index] -= 1
            while self.prefix < len(self.remaining):
                if self.remaining[self.prefix] or self.failed[self.prefix]:
                    break
                self.prefix += 1
            progress[self.job_id] = self.prefix


class FuriosaFileSystemTierManager(FileSystemTierManager):
    """Keep vLLM's lookup and file naming while reading pinned slots with aligned I/O."""

    def __init__(
        self,
        offloading_spec: OffloadingSpec,
        primary_kv_view: memoryview,
        tier_type: str,
        **options: object,
    ) -> None:
        options.setdefault("n_read_threads", DEFAULT_READ_THREADS)
        options.setdefault("n_write_threads", DEFAULT_WRITE_THREADS)
        super().__init__(offloading_spec, primary_kv_view, tier_type, **options)
        directory = os.path.dirname(self.file_mapper.get_config_file_path())
        self._file_io = AlignedFileIO(directory, use_direct=self._use_o_direct)
        self._blocks_per_chunk = offloading_spec.blocks_per_chunk
        self._engine_id = offloading_spec.config.engine_id
        self._job_transfer_bytes: dict[int, int] = {}

    @override
    def submit_store(self, job_metadata: TransferJob) -> None:
        keys = list(job_metadata.keys)
        chunk_ids = [int(chunk) for chunk in job_metadata.chunk_ids]
        if len(keys) != len(chunk_ids):
            raise ValueError("vLLM filesystem store keys and host chunks differ in count")
        if self.events is not None:
            self._store_job_keys[job_metadata.job_id] = keys
        paths = [self.file_mapper.get_file_name(key) for key in keys]
        self._job_block_counts[job_metadata.job_id] = len(keys)
        self._job_transfer_bytes[job_metadata.job_id] = len(keys) * self._block_size
        cache = host_pool.get(self._engine_id)
        tasks = []
        batches = []
        try:
            for path, chunk_id in zip(paths, chunk_ids, strict=True):
                if os.path.exists(path):
                    tasks.extend([_noop] * self._blocks_per_chunk)
                    continue
                first = chunk_id * self._blocks_per_chunk
                slots = [cache.slot(first + index) for index in range(self._blocks_per_chunk)]
                batch = _ChunkStore(self._file_io, path, len(slots[0]), len(slots))
                batches.append(batch)
                tasks.extend(
                    functools.partial(batch.write, index, slot) for index, slot in enumerate(slots)
                )
        except Exception:
            for batch in batches:
                batch.abort()
            raise
        if not tasks:
            tasks.append(_noop)
        self._pool.enqueue_store(job_metadata.job_id, len(tasks), tasks)

    @override
    def submit_load(self, job_metadata: TransferJob) -> None:
        job_id = job_metadata.job_id
        keys = list(job_metadata.keys)
        chunk_ids = [int(chunk) for chunk in job_metadata.chunk_ids]
        if len(keys) != len(chunk_ids):
            raise ValueError("vLLM filesystem load keys and host chunks differ in count")
        paths = [self.file_mapper.get_file_name(key) for key in keys]
        self._load_job_keys[job_id] = keys
        self._job_block_counts[job_id] = len(keys)
        self._job_transfer_bytes[job_id] = len(keys) * self._block_size
        cache = host_pool.get(self._engine_id)
        progress = _LoadProgress(job_id, len(paths), self._blocks_per_chunk)
        self._load_progress[job_id] = 0
        tasks = []
        for file_index, (path, chunk_id) in enumerate(zip(paths, chunk_ids, strict=True)):
            first = chunk_id * self._blocks_per_chunk
            tasks.extend(
                functools.partial(
                    self._load_page,
                    path,
                    file_index,
                    page_index,
                    first + page_index,
                    cache,
                    progress,
                )
                for page_index in range(self._blocks_per_chunk)
            )
        if not tasks:
            tasks.append(_noop)
        self._pool.enqueue_load(job_id, len(tasks), tasks)

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        for result in super().get_finished_jobs():
            transfer_bytes = self._job_transfer_bytes.pop(result.job_id, None)
            if transfer_bytes is not None:
                result.transfer_bytes = transfer_bytes
            yield result

    def _load_page(
        self,
        path: str,
        file_index: int,
        page_index: int,
        slot_index: int,
        cache: engine.HostCache,
        progress: _LoadProgress,
    ) -> None:
        succeeded = False
        try:
            self._file_io.read_page(path, page_index, cache.slot(slot_index))
            succeeded = True
        except OSError as error:
            if "short read" in str(error).lower():
                try:
                    os.remove(path)
                except OSError:
                    logger.debug("Could not remove short cache file %s", path)
            raise
        finally:
            progress.page_done(self._load_progress, file_index, succeeded)
