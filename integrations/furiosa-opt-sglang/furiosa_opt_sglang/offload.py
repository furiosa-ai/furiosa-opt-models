"""SGLang's host-pool adapter over the engine cache.

Each pinned slot holds one whole page across all layers. Six native workers batch up to two slots,
with every layer copy in one runtime DMA batch. SGLang keeps page selection and lifecycle; the
adapter tracks engine copies so SGLang's native events are recorded only after DMA completion.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from threading import Lock, local
from typing import TYPE_CHECKING, Final, Protocol, final

import torch
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import HybridCacheController
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    StackBuildResult,
    StackStrategy,
    build_pool_entry,
)
from sglang.srt.mem_cache.pool_host import HostKVCache, HostPoolGroup
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.runtime_context import get_memory
from typing_extensions import override

import furiosa_opt_engine as engine
from furiosa_opt_sglang.cache import Cache
from furiosa_opt_sglang.platform import Hierarchical
from furiosa_opt_sglang.transfer import AsyncL2TransferEngine

ENGINE_CACHE_GROUP = 0


@dataclass(frozen=True)
class _LoadRoute:
    """One SGLang page batch, reused by its per-layer callbacks."""

    device_indices: torch.Tensor
    host_indices: torch.Tensor
    page_key: _PageKey
    transfers: list[tuple[int, list[tuple[int, int]]]]


@dataclass(frozen=True, slots=True, eq=False)
class _PageKey:
    """Content key with a cached hash for repeated per-layer lookups."""

    device: tuple[int, ...]
    host: tuple[int, ...]
    _hash: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_hash", hash((self.device, self.host)))

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        return self is other or (
            isinstance(other, _PageKey) and self.device == other.device and self.host == other.host
        )


class _CopySubmit(Protocol):
    def __call__(
        self,
        pages: Sequence[tuple[int, Sequence[tuple[int, int]]]],
        host: engine.HostCache,
        *,
        coalesce: bool = False,
    ) -> engine.Copy: ...


def _take_coalescing_option(
    options: dict[str, object] | None,
) -> tuple[bool, dict[str, object] | None]:
    if options is None:
        return False, None
    storage_options = dict(options)
    coalescing = storage_options.pop("coalescing", False)
    if type(coalescing) is not bool:
        raise ValueError("coalescing must be a boolean")
    return coalescing, storage_options


if TYPE_CHECKING:
    from threading import Event

    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.server_args import ServerArgs


@final
class Strategy(StackStrategy):
    """The host-pool stack of the engine's token pool: one host pool of its pages, anchoring the
    radix tree's one component."""

    @override
    def matches(self, kvcache: object, components: set[ComponentType]) -> bool:
        return isinstance(kvcache, Cache) and components == {ComponentType.FULL}

    @override
    def build(
        self,
        *,
        cache: UnifiedRadixCache,
        kvcache: object,
        params: CacheInitParams,
        server_args: ServerArgs,
        load_cache_event: Event,
        storage_backend: str | None = None,
        storage_backend_extra_config: dict[str, object] | None = None,
        prefetch_threshold: int = 256,
        model_name: str | None = None,
        enable_storage_metrics: bool = False,
    ) -> StackBuildResult:
        assert isinstance(kvcache, Cache)
        if storage_backend not in (None, "file"):
            raise ValueError(
                "the Furiosa engine cache supports SGLang HiCache storage only "
                "with its file backend"
            )
        if params.mtp_draft_device_pools:
            raise ValueError(
                "Furiosa HiCache cannot offload SGLang MTP draft pools: its host pool covers only "
                "the engine-owned target-model KV layers"
            )
        # The scheduler's cache parameters carry its tensor-parallel group.
        assert params.tp_cache_group is not None
        coalescing, storage_options = _take_coalescing_option(storage_backend_extra_config)
        storage_backend_extra_config = storage_options
        options = Hierarchical.parse(get_memory())
        host = Host(
            kvcache,
            options.hicache_ratio,
            options.hicache_size,
            params.page_size,
            coalescing=coalescing,
        )
        if storage_backend == "file":
            storage_backend = "dynamic"
            storage_backend_extra_config = {
                **(storage_options or {}),
                "backend_name": "furiosa_file",
                "module_path": "furiosa_opt_sglang.storage",
                "class_name": "FuriosaHiCacheFile",
            }
        layers = kvcache.layer_num
        group = HostPoolGroup(
            [
                build_pool_entry(
                    name=PoolName.KV,
                    host_pool=host,
                    device_pool=kvcache,
                    layer_mapping={layer: layer for layer in range(layers)},
                    transfer_layer_id_max=layers,
                    is_anchor=True,
                )
            ]
        )
        controller = HybridCacheController(
            params.token_to_kv_pool_allocator,
            group,
            params.page_size,
            params.tp_cache_group,
            load_cache_event=load_cache_event,
            attn_cp_group=params.attn_cp_cache_group,
            attn_tp_group=params.attn_tp_cache_group,
            pp_group=params.pp_cache_group,
            write_policy=options.hicache_write_policy,
            io_backend=options.hicache_io_backend,
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            transfer_layer_id_max=layers,
            enable_storage_metrics=enable_storage_metrics,
            host_memory_mode=options.hicache_host_memory_mode,
        )
        transfer_engine = AsyncL2TransferEngine(controller.l2_transfer_engine)
        controller.l2_transfer_engine = transfer_engine
        host.transfer_engine = transfer_engine
        return StackBuildResult(
            host_pool_group=group,
            cache_controller=controller,
            component_host_pools={ComponentType.FULL: host},
            pools_desc="KV",
        )


@final
class Host(HostKVCache):
    """A host tier of `cache`'s pages, `ratio` times its tokens or `size` GB when that is set."""

    def __init__(
        self,
        cache: Cache,
        ratio: float,
        size: int,
        page_size: int,
        *,
        coalescing: bool = False,
    ) -> None:
        # The platform refuses the memory saver beside this tier, so the storage is never released.
        assert cache.storage is not None, "a token pool holds its storage until it is released"
        self.layers = cache.storage.layers
        # Bytes one page of each layer takes on each chip.
        self.sizes = [len(layer) // cache.storage.blocks for layer in self.layers]
        self.chips = cache.pages.chips
        item_bytes = cache.dtype.itemsize
        page_bytes = sum(self.sizes) * self.chips
        if page_size <= 0 or page_bytes % (page_size * item_bytes):
            raise ValueError("the host cache page must align to its token count and storage dtype")
        # SGLang's pool reads these while it allocates the pages.
        self.device_pool = cache
        self.layer_num = self.target_layer_num = cache.layer_num
        self.transfer_engine = None
        self.coalescing = coalescing
        super().__init__(
            dtype=cache.dtype,
            device_capacity_tokens=cache.size,
            size_per_token=sum(self.sizes) * self.chips // page_size,
            start_layer=cache.start_layer,
            end_layer=cache.end_layer,
            host_to_device_ratio=ratio,
            host_size=size,
            page_size=page_size,
            layout="page_first",
            pin_memory=False,
            device="cpu",
        )

    @override
    def get_size_per_token(self) -> int:
        return self.size_per_token

    @override
    def init_kv_buffer(self) -> list[engine.HostCache]:
        copy_layers: list[engine.CopierLayer] = []
        offset = 0
        for layer, page_bytes in zip(self.layers, self.sizes, strict=True):
            copy_layers.append(
                engine.CopierLayer(
                    buffer=layer,
                    page_bytes=page_bytes,
                    host_offset=offset,
                    group=0,
                )
            )

            offset += page_bytes * self.chips
        self.copier = engine.Copier(copy_layers, offset)
        self.host_cache = engine.HostCache(self.copier, self.page_num)
        self._pending_loads: dict[_PageKey, set[int]] = {}
        self._pending_loads_lock = Lock()
        self._copier_lock = Lock()
        self._copy_batch = local()
        return [self.host_cache]

    @override
    def get_hybrid_pool_buffer(self) -> list[engine.HostCache]:
        return [self.host_cache]

    @override
    def destroy(self) -> None:
        transfer_engine = self.transfer_engine
        failure: BaseException | None = None
        try:
            if transfer_engine is not None:
                transfer_engine.shutdown()
        except BaseException as error:
            failure = error
        copier = getattr(self, "copier", None)
        try:
            if copier is not None:
                copier.shutdown()
        except BaseException as error:
            if failure is None:
                failure = error
            else:
                failure = BaseExceptionGroup("multiple cache shutdown failures", [failure, error])
        self.transfer_engine = None
        self.host_cache = None
        self.copier = None
        try:
            super().destroy()
        except BaseException as error:
            if failure is None:
                failure = error
            else:
                failure = BaseExceptionGroup("multiple cache shutdown failures", [failure, error])
        if failure is not None:
            raise failure

    @override
    def backup_from_device_all_layer(
        self,
        device_pool: Cache,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        io_backend: str,
    ) -> None:
        device = pages(device_indices, self.page_size)
        host = pages(host_indices, self.page_size)
        if len(device) != len(host):
            raise ValueError("hierarchical cache backup source and destination pages differ")
        transfers = [
            (target, [(ENGINE_CACHE_GROUP, source)])
            for source, target in zip(device, host, strict=True)
        ]
        copy = self._create_copy(self.copier.copy_out, transfers)
        self._submit_copy(copy)

    @override
    def load_to_device_per_layer(
        self,
        device_pool: Cache,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        layer_id: int,
        io_backend: str,
        *,
        is_draft: bool = False,
    ) -> None:
        if not 0 <= layer_id < self.layer_num:
            raise ValueError("hierarchical cache load names an unknown layer")
        copy_batch = getattr(self, "_copy_batch", None)
        load_routes = None if copy_batch is None else getattr(copy_batch, "load_routes", None)
        # SGLang reuses each transfer's tensors across its per-layer callbacks. Keeping the refs
        # prevents id reuse while this batch memo avoids a CUDA-to-host sync for every layer.
        route_id = (id(device_indices), id(host_indices))
        route = None if load_routes is None else load_routes.get(route_id)
        if (
            route is None
            or route.device_indices is not device_indices
            or route.host_indices is not host_indices
        ):
            device = tuple(pages(device_indices, self.page_size))
            host = tuple(pages(host_indices, self.page_size))
            if len(device) != len(host):
                raise ValueError("hierarchical cache load source and destination pages differ")
            transfers = [
                (target, [(ENGINE_CACHE_GROUP, source)])
                for source, target in zip(device, host, strict=True)
            ]
            route = _LoadRoute(device_indices, host_indices, _PageKey(device, host), transfers)
            if load_routes is not None:
                load_routes[route_id] = route
        if not route.page_key.device:
            return

        key = route.page_key
        pending_loads_lock = getattr(self, "_pending_loads_lock", None)
        if pending_loads_lock is None:
            pending_loads_lock = self._pending_loads_lock = Lock()
        with pending_loads_lock:
            seen = self._pending_loads.get(key)
            new_copy = seen is None
            if seen is None:
                seen = set()
                self._pending_loads[key] = seen
            seen.add(layer_id)
            if len(seen) == self.layer_num:
                del self._pending_loads[key]

        if new_copy:
            # SGLang calls once per layer; Copier moves every layer in one page batch.
            try:
                copy = self._create_copy(
                    self.copier.copy_in, route.transfers, coalesce=self.coalescing
                )
                self._submit_copy(copy)
            except BaseException:
                with pending_loads_lock:
                    self._pending_loads.pop(key, None)
                raise
            if copy_batch is not None and hasattr(copy_batch, "load_keys"):
                copy_batch.load_keys.append(key)

    def begin_async_copy_batch(self) -> None:
        if hasattr(self._copy_batch, "copies"):
            raise RuntimeError("nested SGLang cache copy batch")
        self._copy_batch.copies = []
        self._copy_batch.load_keys = []
        self._copy_batch.load_routes = {}

    def finish_async_copy_batch(self) -> list[engine.Copy]:
        batch = getattr(self, "_copy_batch", None)
        copies = None if batch is None else getattr(batch, "copies", None)
        if copies is None:
            return []
        del batch.copies
        load_keys = batch.load_keys
        del batch.load_keys
        del batch.load_routes
        with self._pending_loads_lock:
            for key in load_keys:
                self._pending_loads.pop(key, None)
        return copies

    def _submit_copy(self, copy: engine.Copy) -> None:
        batch = getattr(self, "_copy_batch", None)
        copies = None if batch is None else getattr(batch, "copies", None)
        if copies is not None:
            copies.append(copy)
            return
        copy.wait()

    def _create_copy(
        self,
        submit: _CopySubmit,
        transfers: list[tuple[int, list[tuple[int, int]]]],
        *,
        coalesce: bool = False,
    ) -> engine.Copy:
        copier_lock = getattr(self, "_copier_lock", None)
        if copier_lock is None:
            return submit(transfers, self.host_cache, coalesce=coalesce)
        with copier_lock:
            return submit(transfers, self.host_cache, coalesce=coalesce)

    @override
    def get_data_page(self, index: int, flat: bool = True) -> torch.Tensor:
        if not flat:
            raise ValueError("the engine cache exposes opaque flat pages to storage")
        page = self._page_index(index)
        slot = self.host_cache.slot(page).access()
        return torch.frombuffer(slot, dtype=torch.uint8).view(self.dtype)

    @override
    def get_dummy_flat_data_page(self) -> torch.Tensor:
        item_bytes = self.dtype.itemsize
        return torch.empty(self._page_bytes // item_bytes, dtype=self.dtype, device="cpu")

    @override
    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        if data_page.device.type != "cpu":
            raise ValueError("the file storage backend returns host pages on CPU")
        page = self._page_index(index)
        source = memoryview(data_page.contiguous().view(torch.uint8).reshape(-1).numpy()).cast("B")
        if len(source) != self._page_bytes:
            raise ValueError("stored page size differs from the engine cache page")
        slot = self.host_cache.slot(page).access()
        destination = memoryview(slot).cast("B")
        try:
            destination[:] = source
        finally:
            destination.release()

    @property
    def _page_bytes(self) -> int:
        return sum(self.sizes) * self.chips

    def _page_index(self, index: int) -> int:
        if index < 0 or index % self.page_size:
            raise ValueError("storage pages must start at an aligned token slot")
        page = index // self.page_size
        if page >= self.page_num:
            raise ValueError("storage page is outside the engine host pool")
        return page


# The dtypes in which SGLang's pools name token slots.
SLOTS: Final = frozenset({torch.int32, torch.int64})


def pages(indices: torch.Tensor, size: int) -> list[int]:
    """The pages of `size` token slots whose slots `indices` lists, each page whole and in order."""
    if size <= 0:
        raise ValueError("hierarchical cache page size must be positive")
    if indices.dim() != 1 or indices.dtype not in SLOTS:
        raise ValueError("hierarchical cache transfers name token slots by a 1-D integer tensor")
    if len(indices) % size:
        raise ValueError("hierarchical cache transfers move whole pages")
    rows = indices.reshape(-1, size)
    first = rows[:, :1]
    offsets = torch.arange(size, dtype=rows.dtype, device=rows.device).expand_as(rows)
    valid = (first >= 0) & (first % size == 0) & (rows - first == offsets)
    result = torch.cat(
        ((valid.all().reshape(1).to(indices.dtype)), first.flatten() // size)
    ).tolist()
    if not result[0]:
        raise ValueError("a hierarchical cache page is a page's consecutive token slots")
    return result[1:]
