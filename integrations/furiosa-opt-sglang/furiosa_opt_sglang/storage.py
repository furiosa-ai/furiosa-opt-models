"""SGLang file storage using the engine's aligned page I/O."""

from __future__ import annotations

import logging
import os
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageExtraInfo,
    PoolTransfer,
)
from typing_extensions import override

from furiosa_opt_engine.cache_transfer import (
    DEFAULT_READ_THREADS,
    DEFAULT_WRITE_THREADS,
    AlignedFileIO,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig

logger = logging.getLogger(__name__)


class FuriosaHiCacheFile(HiCacheFile):
    """Keep SGLang's keys and eviction policy, moving engine pages with O_DIRECT when available."""

    def __init__(self, storage_config: HiCacheStorageConfig, options: Any = None) -> None:
        super().__init__(storage_config)
        self.file_io = AlignedFileIO(self.file_path)
        settings = options if isinstance(options, dict) else {}
        self._read_io = ThreadPoolExecutor(
            max_workers=int(settings.get("n_read_threads", DEFAULT_READ_THREADS)),
            thread_name_prefix="furiosa_hicache_read",
        )
        self._write_io = ThreadPoolExecutor(
            max_workers=int(settings.get("n_write_threads", DEFAULT_WRITE_THREADS)),
            thread_name_prefix="furiosa_hicache_write",
        )

    def _batch_io_v2(
        self,
        transfers: list[PoolTransfer],
        op_fn: Any,
        executor: ThreadPoolExecutor,
        *,
        deduplicate: bool = False,
    ) -> dict[str, list[bool]]:
        results: dict[str, list[bool]] = {}
        pending = []
        scheduled: dict[tuple[str, str], Future[bool]] = {}
        for transfer in transfers:
            host_pool = self.registered_pools[transfer.name]
            keys = transfer.keys or []
            page_size = getattr(host_pool, "page_size", 1) or 1
            expected = len(keys) * page_size
            host_indices = transfer.host_indices
            if host_indices is None or host_indices.numel() != expected:
                logger.error(
                    "%s indices length mismatch for %s: expected %s, got %s",
                    op_fn.__name__,
                    transfer.name,
                    expected,
                    host_indices.numel() if host_indices is not None else 0,
                )
                results[transfer.name] = [False] * len(keys)
                continue

            page_results = results[transfer.name] = [False] * len(keys)
            for index, key in enumerate(keys):
                page_offset = host_indices[index * page_size].item()
                task_key = (str(transfer.name), key)
                future = scheduled.get(task_key) if deduplicate else None
                if future is None:
                    future = executor.submit(op_fn, transfer.name, key, host_pool, page_offset)
                    if deduplicate:
                        scheduled[task_key] = future
                pending.append(
                    (
                        page_results,
                        index,
                        future,
                    )
                )

        failure = None
        for page_results, index, future in pending:
            try:
                page_results[index] = future.result()
            except Exception as error:
                if failure is None:
                    failure = error
        if failure is not None:
            raise failure
        return results

    @override
    def batch_get_v2(
        self,
        transfers: list[PoolTransfer],
        extra_info: HiCacheStorageExtraInfo | None = None,
    ) -> dict[str, list[bool]]:
        return self._batch_io_v2(transfers, self._read_page, self._read_io)

    @override
    def batch_set_v2(
        self,
        transfers: list[PoolTransfer],
        extra_info: HiCacheStorageExtraInfo | None = None,
    ) -> dict[str, list[bool]]:
        return self._batch_io_v2(transfers, self._write_page, self._write_io, deduplicate=True)

    def _read_page(self, pool_name: str, key: str, host_pool: Any, page_offset: int) -> bool:
        if not hasattr(host_pool, "host_cache"):
            return super()._read_page(pool_name, key, host_pool, page_offset)

        storage_key = self._log_key(pool_name, key)
        suffixed = self._get_suffixed_key(storage_key)
        path = os.path.join(self.file_path, f"{suffixed}.bin")
        try:
            self.file_io.read(path, host_pool.host_cache.slot(host_pool._page_index(page_offset)))
        except FileNotFoundError:
            if self.metadata_cache is not None:
                self.metadata_cache.remove(suffixed)
            logger.warning("Failed to fetch %s from Furiosa file storage.", storage_key)
            return False
        self._evictor.touch(suffixed, path)
        if self.metadata_cache is not None:
            self.metadata_cache.add(suffixed)
        return True

    def _write_page(self, pool_name: str, key: str, host_pool: Any, page_offset: int) -> bool:
        if not hasattr(host_pool, "host_cache"):
            return super()._write_page(pool_name, key, host_pool, page_offset)

        storage_key = self._log_key(pool_name, key)
        suffixed = self._get_suffixed_key(storage_key)
        path = os.path.join(self.file_path, f"{suffixed}.bin")
        if self.exists(storage_key):
            self._evictor.touch(suffixed, path)
            return True

        slot = host_pool.host_cache.slot(host_pool._page_index(page_offset))
        size = self.file_io.aligned_size(len(slot))
        if not self._evictor.reserve(suffixed, size, key=storage_key):
            return False

        temporary = os.path.join(self.file_path, f".{uuid.uuid4().hex}.tmp")
        try:
            self.file_io.write(temporary, slot)
            os.replace(temporary, path)
            self._evictor.commit(suffixed)
            if self.metadata_cache is not None:
                self.metadata_cache.add(suffixed)
            return True
        except Exception as error:
            logger.error("Failed to store %s in Furiosa file storage: %s", storage_key, error)
            self._evictor.abort(suffixed)
            try:
                os.remove(temporary)
            except OSError:
                pass
            if self.metadata_cache is not None:
                self.metadata_cache.remove(suffixed)
            return False
