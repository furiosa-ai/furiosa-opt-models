from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from threading import Event
from types import SimpleNamespace
from uuid import uuid4

import torch

import furiosa_opt_engine as engine
from furiosa_opt_engine._host_region import HostRegionRegistrar
from furiosa_opt_engine._mooncake import MooncakeIOGuard
from furiosa_opt_sglang.mooncake import FuriosaHiCacheMooncake
from support import (
    PAGE_BYTES,
    PinnedPage,
    contents_slot,
    fill_slot,
    open_device,
    open_store,
    pinned_page,
)


class _EngineHostPool:
    def __init__(self, page: PinnedPage) -> None:
        self.host_cache: engine.HostCache = page.cache
        self.page_num = page.slots
        self.page_size = 1
        self.dtype = torch.uint8
        self.layer_num = 1
        self.start_layer = 0
        self.end_layer = 1
        self.chips = 1
        self.sizes = [PAGE_BYTES]

    @staticmethod
    def _page_index(token_index: int) -> int:
        return token_index


def _adapter(page: PinnedPage, store: object, namespace: str) -> FuriosaHiCacheMooncake:
    adapter = FuriosaHiCacheMooncake.__new__(FuriosaHiCacheMooncake)
    adapter.storage_config = SimpleNamespace(
        tp_rank=0,
        pp_rank=0,
        attn_cp_rank=0,
        tp_size=1,
        pp_size=1,
        attn_cp_size=1,
    )
    adapter.store = store
    adapter._tag_keys = lambda keys: [f"{namespace}:{key}" for key in keys]
    adapter._batch_exist = lambda keys, *_args: store.batch_is_exist(keys)
    adapter._pointer_io = MooncakeIOGuard()
    adapter._region_registrar = HostRegionRegistrar(
        adapter._register_mooncake_region, adapter._unregister_mooncake_region
    )
    adapter._registered_pool = adapter._build_registered_pool(_EngineHostPool(page))
    return adapter


def test_sglang_store_and_load_between_adapter_instances(mooncake_master: str) -> None:
    """A second SGLang storage adapter reads bytes written by the first from engine host slots."""
    from furiosa_opt_sglang.decoder import PES

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

        namespace = f"ci-sglang-{uuid4().hex}"
        writer = _adapter(writer_page, writer_store, namespace)
        reader = _adapter(reader_page, reader_store, namespace)
        stack.callback(writer._region_registrar.close)
        stack.callback(reader._region_registrar.close)
        fill_slot(writer_page, 0, 83)
        fill_slot(writer_page, 1, 84)
        fill_slot(reader_page, 0, 0)
        fill_slot(reader_page, 1, 0)
        keys = ["shared-prompt-0", "shared-prompt-1"]
        indices = torch.tensor([0, 1])

        assert writer.batch_set_v1(keys, indices) == [True, True]
        assert reader.batch_exists(keys) == 2
        assert reader.batch_get_v1(keys, indices) == [True, True]
        assert contents_slot(reader_page, 0) == bytes([83]) * PAGE_BYTES
        assert contents_slot(reader_page, 1) == bytes([84]) * PAGE_BYTES


def test_sglang_store_lease_blocks_engine_dma_until_store_completion(
    mooncake_master: str,
) -> None:
    """A Store lease through HostSlot.access holds the same Rust gate as Copier DMA."""
    from furiosa_opt_sglang.decoder import PES

    class BlockingStore:
        def __init__(self, store: object) -> None:
            self.store = store
            self.entered = Event()
            self.release = Event()

        def __getattr__(self, name: str) -> object:
            return getattr(self.store, name)

        def batch_put_from(self, keys: list[str], addresses: list[int], sizes: list[int]):
            self.entered.set()
            if not self.release.wait(timeout=30):
                raise TimeoutError("test did not release the Store call")
            return self.store.batch_put_from(keys, addresses, sizes)

    with ExitStack() as stack:
        device = open_device(PES)
        page = pinned_page(device)
        stack.callback(page.copier.shutdown)
        raw_store = open_store(mooncake_master)
        stack.callback(raw_store.close)
        store = BlockingStore(raw_store)
        adapter = _adapter(page, store, f"ci-sglang-lease-{uuid4().hex}")
        stack.callback(adapter._region_registrar.close)
        fill_slot(page, 0, 91)

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                store_call = executor.submit(
                    adapter.batch_set_v1, ["leased-page"], torch.tensor([0])
                )
                assert store.entered.wait(timeout=30)
                dma = page.copier.copy_out([(0, [(0, 0)])], page.cache)
                assert not dma.done(timeout=0.1)
                store.release.set()
                assert store_call.result(timeout=30) == [True]
                assert dma.done(timeout=30)
        finally:
            store.release.set()
