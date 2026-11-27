from __future__ import annotations

import os
from dataclasses import dataclass

from mooncake.store import MooncakeDistributedStore

import furiosa_opt_engine as engine

PAGE_BYTES = 64
GLOBAL_SEGMENT_BYTES = 64 * 1024 * 1024
LOCAL_BUFFER_BYTES = 16 * 1024 * 1024


@dataclass
class PinnedPage:
    """One real engine-owned pinned host slot and its backing device page."""

    buffer: engine.HbmBuffer
    copier: engine.Copier
    cache: engine.HostCache
    slots: int


def open_device(pes: range) -> engine.Device:
    visible_chips = os.environ.get("FURIOSA_VISIBLE_CHIPS", "0")
    chip = int(visible_chips.split(",", maxsplit=1)[0])
    return engine.Device(engine.Placement([chip], pes))


def pinned_page(device: engine.Device, slots: int = 1) -> PinnedPage:
    buffer = engine.HbmBuffer(device, PAGE_BYTES)
    copier = engine.Copier(
        [engine.CopierLayer(buffer, PAGE_BYTES, host_offset=0, group=0)], PAGE_BYTES
    )
    return PinnedPage(
        buffer=buffer,
        copier=copier,
        cache=engine.HostCache(copier, slots),
        slots=slots,
    )


def fill(page: PinnedPage, value: int) -> None:
    fill_slot(page, 0, value)


def fill_slot(page: PinnedPage, index: int, value: int) -> None:
    access = page.cache.slot(index).access()
    view = memoryview(access)
    try:
        view[:] = bytes([value]) * PAGE_BYTES
    finally:
        view.release()


def contents(page: PinnedPage) -> bytes:
    return contents_slot(page, 0)


def contents_slot(page: PinnedPage, index: int) -> bytes:
    access = page.cache.slot(index).access()
    view = memoryview(access)
    try:
        return bytes(view)
    finally:
        view.release()


def open_store(master_address: str) -> MooncakeDistributedStore:
    """Connect a non-CUDA TCP client to the per-job metadata service."""
    store = MooncakeDistributedStore()
    try:
        result = store.setup(
            "127.0.0.1",
            "P2PHANDSHAKE",
            GLOBAL_SEGMENT_BYTES,
            LOCAL_BUFFER_BYTES,
            "tcp",
            "",
            master_address,
        )
        if result != 0:
            raise RuntimeError(f"Mooncake Store setup failed: {result}")
    except BaseException as setup_error:
        try:
            close_result = store.close()
            if close_result not in (None, 0):
                setup_error.add_note(f"Mooncake Store close failed: {close_result}")
        except BaseException as cleanup_error:
            setup_error.add_note(f"Mooncake Store close raised: {cleanup_error!r}")
        raise
    return store
