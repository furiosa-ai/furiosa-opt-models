"""vLLM filesystem tier I/O directly over the engine's pinned cache slots."""

from __future__ import annotations

import functools
import logging
import os
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class _ChunkLayer:
    host_offset: int
    page_bytes: int


class _ChunkStore:
    """Write one vLLM chunk in its native layer-major layout."""

    def __init__(
        self,
        file_io: AlignedFileIO,
        path: str,
        page_bytes: int,
        pages: int,
        file_bytes: int,
        file_layout: tuple[host_pool.FilePageSegment, ...],
    ) -> None:
        self.file_io = file_io
        self.path = path
        self.temporary = f"{path}.{uuid.uuid4().hex}.tmp"
        self.page_bytes = page_bytes
        self.pages = pages
        self.file_bytes = file_bytes
        self.layout: list[_ChunkLayer] = []
        file_offset = 0
        host_bytes = 0
        for segment in file_layout:
            if segment.host_offset < 0 or segment.page_bytes <= 0:
                raise ValueError("vLLM filesystem layout has an invalid layer segment")
            if segment.host_offset + segment.page_bytes > page_bytes:
                raise ValueError("vLLM filesystem layer segment exceeds its host slot")
            self.layout.append(_ChunkLayer(segment.host_offset, segment.page_bytes))
            host_bytes += segment.page_bytes
            file_offset += segment.page_bytes * pages
        if host_bytes != page_bytes:
            raise ValueError("vLLM filesystem layer pages do not cover the host slot")
        if file_offset > file_bytes:
            raise ValueError("vLLM filesystem layer pages exceed the chunk file size")

    def write(self, slots: list[engine.HostSlot]) -> None:
        accesses = []
        views = []
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self.file_io.prepare(
                self.temporary,
                self.page_bytes,
                self.pages,
                file_bytes=self.file_bytes,
            )
            for slot in slots:
                access = slot.access()
                accesses.append(access)
                views.append(_slot_view(access))
            if len(views) == 1:
                # One-block chunks already have the native layer-major byte order.
                self.file_io.writev_at(self.temporary, 0, views)
            else:
                self.file_io.writev_at_many(self.temporary, _layer_vectors(views, self.layout))
            os.replace(self.temporary, self.path)
        except BaseException:
            self.abort()
            raise
        finally:
            for view in views:
                view.release()
            accesses.clear()

    def abort(self) -> None:
        try:
            os.remove(self.temporary)
        except OSError:
            pass


class _LoadProgress:
    """Report only the longest key prefix whose files loaded successfully."""

    def __init__(self, job_id: int, files: int) -> None:
        self.job_id = job_id
        self.completed: list[bool | None] = [None] * files
        self.prefix = 0
        self.lock = Lock()

    def file_done(self, progress: dict[int, int], file_index: int, succeeded: bool) -> None:
        with self.lock:
            self.completed[file_index] = succeeded
            while self.prefix < len(self.completed) and self.completed[self.prefix] is True:
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
        job_id = job_metadata.job_id
        paths = [self.file_mapper.get_file_name(key) for key in keys]
        registration = host_pool.get(self._engine_id)
        cache = registration.cache
        file_layout = registration.file_layout
        tasks = []
        batches = []
        try:
            for path, chunk_id in zip(paths, chunk_ids, strict=True):
                if os.path.exists(path):
                    tasks.append(_noop)
                    continue
                first = chunk_id * self._blocks_per_chunk
                slots = [cache.slot(first + index) for index in range(self._blocks_per_chunk)]
                batch = _ChunkStore(
                    self._file_io,
                    path,
                    len(slots[0]),
                    len(slots),
                    self._block_size,
                    file_layout,
                )
                batches.append(batch)
                tasks.append(functools.partial(batch.write, slots))
        except Exception:
            for batch in batches:
                batch.abort()
            raise
        if not tasks:
            tasks.append(_noop)
        if self.events is not None:
            self._store_job_keys[job_id] = keys
        self._job_block_counts[job_id] = len(keys)
        self._job_transfer_bytes[job_id] = len(keys) * self._block_size
        self._pool.enqueue_store(job_id, len(tasks), tasks)

    @override
    def submit_load(self, job_metadata: TransferJob) -> None:
        job_id = job_metadata.job_id
        keys = list(job_metadata.keys)
        chunk_ids = [int(chunk) for chunk in job_metadata.chunk_ids]
        if len(keys) != len(chunk_ids):
            raise ValueError("vLLM filesystem load keys and host chunks differ in count")
        paths = [self.file_mapper.get_file_name(key) for key in keys]
        registration = host_pool.get(self._engine_id)
        cache = registration.cache
        file_layout = registration.file_layout
        progress = _LoadProgress(job_id, len(paths))
        tasks = []
        for file_index, (path, chunk_id) in enumerate(zip(paths, chunk_ids, strict=True)):
            first = chunk_id * self._blocks_per_chunk
            slots = [cache.slot(first + index) for index in range(self._blocks_per_chunk)]
            tasks.append(
                functools.partial(
                    self._load_chunk,
                    path,
                    file_index,
                    slots,
                    file_layout,
                    progress,
                )
            )
        if not tasks:
            tasks.append(_noop)
        self._load_job_keys[job_id] = keys
        self._job_block_counts[job_id] = len(keys)
        self._job_transfer_bytes[job_id] = len(keys) * self._block_size
        self._load_progress[job_id] = 0
        self._pool.enqueue_load(job_id, len(tasks), tasks)

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        for result in super().get_finished_jobs():
            transfer_bytes = self._job_transfer_bytes.pop(result.job_id, None)
            if transfer_bytes is not None:
                result.transfer_bytes = transfer_bytes
            yield result

    def _load_chunk(
        self,
        path: str,
        file_index: int,
        slots: list[engine.HostSlot],
        file_layout: tuple[host_pool.FilePageSegment, ...],
        progress: _LoadProgress,
    ) -> None:
        succeeded = False
        accesses = []
        views = []
        try:
            for slot in slots:
                access = slot.access()
                accesses.append(access)
                views.append(_slot_view(access))
            if len(views) == 1:
                self._file_io.readv_at(path, 0, views)
            else:
                self._file_io.readv_at_many(path, _layer_vectors(views, file_layout))
            succeeded = True
        except OSError as error:
            if "short read" in str(error).lower():
                try:
                    os.remove(path)
                except OSError:
                    logger.debug("Could not remove short cache file %s", path)
            raise
        finally:
            for view in views:
                view.release()
            accesses.clear()
            progress.file_done(self._load_progress, file_index, succeeded)


def _slot_view(access: engine.HostSlotAccess) -> memoryview:
    base = memoryview(access)
    try:
        return base.cast("B")
    finally:
        base.release()


def _layer_vectors(
    slots: list[memoryview],
    layout: Iterable[_ChunkLayer | host_pool.FilePageSegment],
) -> Iterable[tuple[int, list[memoryview]]]:
    file_offset = 0
    for layer in layout:
        vectors = [slot[layer.host_offset : layer.host_offset + layer.page_bytes] for slot in slots]
        try:
            yield file_offset, vectors
        finally:
            for vector in vectors:
                vector.release()
        file_offset += layer.page_bytes * len(slots)
