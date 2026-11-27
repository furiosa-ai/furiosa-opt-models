"""SGLang's native Mooncake store over Furiosa's pinned host-cache slots."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from typing_extensions import override

from furiosa_opt_engine._host_region import (
    HostRegion,
    HostRegionRegistrar,
    host_slot_region,
)
from furiosa_opt_engine._mooncake import (
    ENGINE_PAGE_FORMAT_VERSION,
    MooncakeIOGuard,
    require_confirmed_exists,
    require_confirmed_get,
    require_confirmed_put,
)

if TYPE_CHECKING:
    import torch
    from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
    from sglang.srt.mem_cache.pool_host import HostKVCache

    import furiosa_opt_engine as engine


class _FuriosaHostPool(Protocol):
    """Furiosa-only fields required to map SGLang's page indices to engine slots."""

    host_cache: engine.HostCache
    page_num: int
    page_size: int | None
    dtype: torch.dtype
    layer_num: int
    start_layer: int
    end_layer: int
    chips: int
    sizes: Sequence[int]

    def _page_index(self, token_index: int) -> int: ...


@dataclass(frozen=True, slots=True)
class _RegisteredHostPool:
    pool: _FuriosaHostPool
    page_bytes: int
    page_size: int
    layout_suffix: str


class FuriosaHiCacheMooncake(MooncakeStore):
    """Keep SGLang's Mooncake client/config and key lifecycle over engine pages.

    Integration deliberately uses MooncakeStore's pinned private hooks
    ``_load_config``, ``_tag_keys`` and ``_batch_exist`` from SGLang 0.5.22.
    """

    def __init__(self, storage_config: HiCacheStorageConfig, mem_pool: Any = None) -> None:
        self._registered_pool: _RegisteredHostPool | None = None
        self._pointer_io = MooncakeIOGuard()
        try:
            super().__init__(storage_config, mem_pool)
        except ImportError as error:
            cause = error.__cause__
            if isinstance(cause, ModuleNotFoundError) and cause.name == "mooncake":
                raise ImportError(
                    "Install furiosa-opt-sglang[mooncake] to use the Mooncake HiCache backend"
                ) from error
            raise
        # Install bound callbacks only after parent initialization succeeds;
        # the registrar otherwise forms a self-cycle on partial construction.
        self._region_registrar = HostRegionRegistrar(
            self._register_mooncake_region, self._unregister_mooncake_region
        )

    @override
    def _load_config(self, storage_config: Any = None) -> Any:
        config = super()._load_config(storage_config)
        if config.standalone_storage:
            raise ValueError(
                "Furiosa HiCache Mooncake requires embedded storage; standalone storage "
                "needs Mooncake-owned host tensors"
            )
        return config

    @override
    def register_mem_pool_host(self, mem_pool_host: HostKVCache) -> None:
        registered_pool = self._build_registered_pool(mem_pool_host)
        HiCacheStorage.register_mem_pool_host(self, mem_pool_host)
        self._registered_pool = registered_pool
        self.gb_per_page = registered_pool.page_bytes / (1 << 30)

    @override
    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name: str) -> None:
        if host_pool_name != PoolName.KV:
            raise ValueError(
                f"Furiosa HiCache Mooncake supports the engine KV pool, not {host_pool_name}"
            )
        registered_pool = self._build_registered_pool(host_pool)
        HiCacheStorage.register_mem_host_pool_v2(self, host_pool, host_pool_name)
        self._registered_pool = registered_pool
        self.gb_per_page = registered_pool.page_bytes / (1 << 30)

    def _build_registered_pool(self, host_pool: HostKVCache) -> _RegisteredHostPool:
        required_fields = (
            "host_cache",
            "_page_index",
            "page_num",
            "page_size",
            "dtype",
            "chips",
            "sizes",
            "layer_num",
            "start_layer",
            "end_layer",
        )
        missing = [field for field in required_fields if not hasattr(host_pool, field)]
        if missing or not callable(getattr(host_pool, "_page_index", None)):
            detail = ", ".join(missing) if missing else "_page_index is not callable"
            raise TypeError(
                "Furiosa Mooncake requires the engine-backed HiCache host pool; "
                f"missing or invalid fields: {detail}"
            )
        furiosa_pool = cast(_FuriosaHostPool, host_pool)
        if furiosa_pool.page_num <= 0:
            raise ValueError("Furiosa Mooncake requires at least one host cache page")
        page_size = furiosa_pool.page_size
        if page_size is None or page_size <= 0:
            raise ValueError("Furiosa Mooncake requires a positive HiCache page size")
        slot_size = len(furiosa_pool.host_cache.slot(0))
        if slot_size <= 0:
            raise ValueError("Furiosa Mooncake cache pages must contain bytes")
        if furiosa_pool.chips <= 0:
            raise ValueError("Furiosa Mooncake cache layout must have a positive chip count")
        if furiosa_pool.layer_num <= 0 or len(furiosa_pool.sizes) != furiosa_pool.layer_num:
            raise ValueError("Furiosa Mooncake layer count must match its per-layer page sizes")
        if any(size <= 0 for size in furiosa_pool.sizes):
            raise ValueError("Furiosa Mooncake cache layout must have positive layer page sizes")
        expected_slot_size = sum(furiosa_pool.sizes) * furiosa_pool.chips
        if slot_size != expected_slot_size:
            raise ValueError(
                "Furiosa Mooncake layer page sizes disagree with the host cache slot size"
            )
        config = self.storage_config
        layout = {
            "format": ENGINE_PAGE_FORMAT_VERSION,
            "dtype": str(furiosa_pool.dtype),
            "page_size": page_size,
            "page_bytes": slot_size,
            "layers": furiosa_pool.layer_num,
            "start_layer": furiosa_pool.start_layer,
            "end_layer": furiosa_pool.end_layer,
            "chips": furiosa_pool.chips,
            "layer_page_bytes": list(furiosa_pool.sizes),
            "tp_size": config.tp_size,
            "pp_size": config.pp_size,
            "attn_cp_size": config.attn_cp_size,
        }
        encoded = json.dumps(layout, sort_keys=True, separators=(",", ":"))
        return _RegisteredHostPool(
            pool=furiosa_pool,
            page_bytes=slot_size,
            page_size=page_size,
            layout_suffix=hashlib.sha256(encoded.encode()).hexdigest(),
        )

    def _keys(self, keys: list[str]) -> list[str]:
        registered_pool = self._registered_pool
        if registered_pool is None or self.storage_config is None:
            raise RuntimeError("SGLang registered the Mooncake backend before the host pool")
        tagged = self._tag_keys(keys)
        return [
            f"{key}_{self.storage_config.tp_rank}_{self.storage_config.pp_rank}_"
            f"{self.storage_config.attn_cp_rank}_{registered_pool.layout_suffix}"
            for key in tagged
        ]

    @override
    def exists(self, key: str) -> bool:
        self._pointer_io.ensure_safe()
        return require_confirmed_exists(self._batch_exist(self._keys([key])), 1)[0]

    def close(self) -> None:
        self._pointer_io.ensure_safe()
        # Unregister before allowing upstream Store teardown. If unregistering
        # fails, keep the Store live so the next close can safely retry.
        self._region_registrar.close()
        super().close()

    @override
    def clear(self) -> None:
        """Keep SGLang's default no-op instead of clearing the shared Mooncake Store."""
        HiCacheStorage.clear(self)

    @override
    def get(
        self,
        key: str,
        target_location: Any = None,
        target_sizes: Any = None,
    ) -> Any:
        raise NotImplementedError(
            "Furiosa Mooncake uses HiCache page-index APIs, not tensor-address storage"
        )

    @override
    def batch_get(
        self,
        keys: list[str],
        target_locations: Any = None,
        target_sizes: Any = None,
    ) -> Any:
        raise NotImplementedError(
            "Furiosa Mooncake uses HiCache page-index APIs, not tensor-address storage"
        )

    @override
    def set(
        self,
        key: str,
        value: Any = None,
        target_location: Any = None,
        target_sizes: Any = None,
    ) -> bool:
        raise NotImplementedError(
            "Furiosa Mooncake uses HiCache page-index APIs, not tensor-address storage"
        )

    @override
    def batch_set(
        self,
        keys: list[str],
        values: Any = None,
        target_locations: Any = None,
        target_sizes: Any = None,
    ) -> bool:
        raise NotImplementedError(
            "Furiosa Mooncake uses HiCache page-index APIs, not tensor-address storage"
        )

    @override
    def batch_exists(
        self,
        keys: list[str],
        extra_info: HiCacheStorageExtraInfo | None = None,
    ) -> int:
        if not keys:
            return 0
        self._pointer_io.ensure_safe()
        results = self._batch_exist(self._keys(keys), extra_info)
        for index, exists in enumerate(require_confirmed_exists(results, len(keys))):
            if not exists:
                return index
        return len(keys)

    @override
    def batch_get_v1(
        self,
        keys: list[str],
        host_indices: torch.Tensor,
        extra_info: HiCacheStorageExtraInfo | None = None,
    ) -> list[bool]:
        return self._batch_transfer(keys, host_indices, is_set=False)

    @override
    def batch_set_v1(
        self,
        keys: list[str],
        host_indices: torch.Tensor,
        extra_info: HiCacheStorageExtraInfo | None = None,
    ) -> list[bool]:
        return self._batch_transfer(keys, host_indices, is_set=True)

    @override
    def batch_exists_v2(
        self,
        keys: list[str],
        pool_transfers: list[PoolTransfer] | None = None,
        extra_info: HiCacheStorageExtraInfo | None = None,
    ) -> PoolTransferResult:
        unsupported = [
            transfer.name for transfer in pool_transfers or [] if transfer.name != PoolName.KV
        ]
        if unsupported:
            raise ValueError(f"Furiosa Mooncake does not support sidecar pools: {unsupported}")
        hit_pages = self.batch_exists(keys, extra_info)
        return PoolTransferResult(
            kv_hit_pages=hit_pages,
            extra_pool_hit_pages={PoolName.KV: hit_pages} if hit_pages else {},
        )

    @override
    def batch_get_v2(
        self,
        transfers: list[PoolTransfer],
        extra_info: HiCacheStorageExtraInfo | None = None,
    ) -> dict[str, list[bool]]:
        return self._batch_transfer_v2(transfers, is_set=False)

    @override
    def batch_set_v2(
        self,
        transfers: list[PoolTransfer],
        extra_info: HiCacheStorageExtraInfo | None = None,
    ) -> dict[str, list[bool]]:
        return self._batch_transfer_v2(transfers, is_set=True)

    def _batch_transfer_v2(
        self, transfers: list[PoolTransfer], *, is_set: bool
    ) -> dict[str, list[bool]]:
        results = {}
        for transfer in transfers:
            if transfer.name != PoolName.KV:
                raise ValueError(f"Furiosa Mooncake does not support sidecar pool {transfer.name}")
            if transfer.keys is None or transfer.host_indices is None:
                raise ValueError("Mooncake KV transfers require keys and host indices")
            results[transfer.name] = self._batch_transfer(
                transfer.keys, transfer.host_indices, is_set=is_set
            )
        return results

    def _batch_transfer(
        self,
        keys: list[str],
        host_indices: torch.Tensor,
        *,
        is_set: bool,
    ) -> list[bool]:
        if not keys:
            return []
        self._pointer_io.ensure_safe()
        registered_pool = self._registered_pool
        if registered_pool is None:
            raise RuntimeError("SGLang has not registered the Furiosa host cache")
        host_pool = registered_pool.pool
        page_size = registered_pool.page_size
        if host_indices.numel() != len(keys) * page_size:
            raise ValueError("SGLang Mooncake keys and host indices differ in cache-page count")

        object_keys = self._keys(keys)
        if is_set:
            # SGLang retains each host page through this synchronous storage call.
            # Check the Store first, then lease only pages that need PUT.
            succeeded = require_confirmed_exists(self._batch_exist(object_keys), len(keys))
            transfer_indices = [index for index, found in enumerate(succeeded) if not found]
            if not transfer_indices:
                return succeeded
        else:
            succeeded = []
            transfer_indices = list(range(len(keys)))

        first_token_indices = host_indices[::page_size].tolist()
        slots = [
            host_pool.host_cache.slot(host_pool._page_index(first_token_indices[index]))
            for index in transfer_indices
        ]

        with ExitStack() as leases:
            regions = tuple(leases.enter_context(host_slot_region(slot)) for slot in slots)
            if len({region.region.size for region in regions}) != 1:
                raise ValueError("Furiosa Mooncake cache pages have inconsistent sizes")
            addresses = [region.region.address for region in regions]
            sizes = [region.region.size for region in regions]
            if is_set:
                write_regions = regions
                self._region_registrar.register(write_regions)
                writes = self._pointer_io.run_pointer_io(
                    self,
                    write_regions,
                    lambda: self.store.batch_put_from(
                        [object_keys[index] for index in transfer_indices], addresses, sizes
                    ),
                    lambda result: require_confirmed_put(result, len(transfer_indices)),
                    "PUT",
                )
                for index, result in zip(transfer_indices, writes, strict=True):
                    succeeded[index] = result == 0
                if not all(succeeded):
                    succeeded = require_confirmed_exists(self._batch_exist(object_keys), len(keys))
            else:
                self._region_registrar.register(regions)
                reads = self._pointer_io.run_pointer_io(
                    self,
                    regions,
                    lambda: self.store.batch_get_into(
                        [object_keys[index] for index in transfer_indices], addresses, sizes
                    ),
                    lambda result: require_confirmed_get(result, sizes),
                    "GET",
                )
                succeeded = [result == registered_pool.page_bytes for result in reads]
        return succeeded

    def _register_mooncake_region(self, region: HostRegion) -> None:
        result = self.store.register_buffer(region.address, region.size)
        if result != 0:
            raise RuntimeError(
                f"Mooncake failed to register host slot {region.address:#x} "
                f"({region.size} bytes): {result}"
            )

    def _unregister_mooncake_region(self, region: HostRegion) -> None:
        result = self.store.unregister_buffer(region.address)
        if result != 0:
            raise RuntimeError(
                f"Mooncake failed to unregister host slot {region.address:#x}: {result}"
            )
