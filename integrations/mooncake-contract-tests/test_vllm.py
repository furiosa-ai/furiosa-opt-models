from __future__ import annotations

import threading
from contextlib import ExitStack
from types import SimpleNamespace
from uuid import uuid4

from mooncake.store import ReplicateConfig

from furiosa_opt_engine._host_region import HostRegionLease, HostRegionRegistrar
from furiosa_opt_engine._mooncake import MooncakeIOGuard
from furiosa_opt_engine.cache_transfer import BoundedCacheStaging
from furiosa_opt_vllm.mooncake_store import (
    FuriosaMooncakeStoreWorker,
    _PageOperation,
)
from support import (
    PAGE_BYTES,
    PinnedPage,
    contents_slot,
    fill_slot,
    open_device,
    open_store,
    pinned_page,
)


def _cache(page: PinnedPage) -> object:
    return SimpleNamespace(
        blocks=page.slots,
        layers=[page.buffer],
        spec=SimpleNamespace(
            groups=[SimpleNamespace(logical_group=0, pool=0, layers=(0,))],
            pools=[SimpleNamespace(bytes=PAGE_BYTES)],
        ),
    )


def _worker(
    page: PinnedPage,
    store: object,
    key: str,
) -> tuple[FuriosaMooncakeStoreWorker, BoundedCacheStaging, HostRegionLease]:
    cache = _cache(page)
    staging = BoundedCacheStaging(cache, chips=1, capacity_pages=page.slots)
    worker = object.__new__(FuriosaMooncakeStoreWorker)
    worker._cache = cache
    worker._staging = staging
    worker._staging_region = None
    worker._unsafe_staging_leases = []
    worker._region_registrar = HostRegionRegistrar(
        worker._register_mooncake_region, worker._unregister_mooncake_region
    )
    worker._pointer_io = MooncakeIOGuard()
    worker._lease_slots = threading.BoundedSemaphore(page.slots)
    worker._max_batch_pages = page.slots
    worker._disk_oversized_groups = frozenset()
    worker._kv_cache_groups = [SimpleNamespace(layer_names=("layers.0",))]
    worker.store = store
    worker.store_replicate_config = ReplicateConfig()
    worker._closed = False
    worker._closing = False

    region = staging.lease_host_region()
    worker._staging_address = region.region.address
    worker._region_registrar.register((region,))
    worker._staging_region = region
    operation = _PageOperation(
        group=0,
        block_id=0,
        key=key,
        start=0,
        end=1,
        block_hash=b"contract",
    )
    worker._contract_operation = operation
    return worker, staging, region


def _engine_copy_in(page: PinnedPage, slot: int) -> None:
    page.copier.copy_in([(slot, [(0, 0)])], page.cache).wait()


def _engine_copy_out(page: PinnedPage, slot: int) -> None:
    page.copier.copy_out([(slot, [(0, 0)])], page.cache).wait()


def test_vllm_native_store_moves_engine_pages_between_instances(
    mooncake_master: str,
) -> None:
    """Exercise the adapter with real RNGD HBM pages and one registered host slab."""
    from furiosa_opt_vllm.decoder import PES

    with ExitStack() as stack:
        device = open_device(PES)
        writer_page = pinned_page(device, slots=2)
        reader_page = pinned_page(device, slots=2)
        stack.callback(writer_page.copier.shutdown)
        stack.callback(reader_page.copier.shutdown)
        writer_store = open_store(mooncake_master)
        reader_store = open_store(mooncake_master)
        stack.callback(writer_store.close)
        stack.callback(reader_store.close)

        namespace = f"ci-vllm-store-{uuid4().hex}"
        writer, writer_staging, _ = _worker(writer_page, writer_store, f"{namespace}@prompt")
        reader, reader_staging, _ = _worker(reader_page, reader_store, f"{namespace}@prompt")
        stack.callback(_release_adapter, writer, writer_staging)
        stack.callback(_release_adapter, reader, reader_staging)

        fill_slot(writer_page, 0, 37)
        _engine_copy_in(writer_page, 0)
        fill_slot(reader_page, 0, 0)
        _engine_copy_in(reader_page, 0)

        assert writer._put_pages([writer._contract_operation])
        assert reader._get_pages("contract-request", [reader._contract_operation])

        _engine_copy_out(reader_page, 0)
        assert contents_slot(reader_page, 0) == bytes([37]) * PAGE_BYTES

        # Each adapter registered one entire contiguous staging slab.
        assert writer._staging_region.region.size == writer_staging.capacity_pages * PAGE_BYTES
        assert reader._staging_region.region.size == reader_staging.capacity_pages * PAGE_BYTES


def _release_adapter(
    worker: FuriosaMooncakeStoreWorker,
    staging: BoundedCacheStaging,
) -> None:
    worker._region_registrar.close()
    worker._staging_region.release()
    staging.release_engine_cache(worker._cache)
