"""Worker host slots shared with the vLLM scheduler and filesystem tier."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import furiosa_opt_engine as engine


@dataclass(frozen=True, slots=True)
class FilePageSegment:
    """One layer page's location in an engine slot and order in a vLLM chunk file."""

    host_offset: int
    page_bytes: int


@dataclass(frozen=True, slots=True)
class _Registration:
    cache: engine.HostCache
    file_layout: tuple[FilePageSegment, ...]


_LOCK = Lock()
_POOLS: dict[str, _Registration] = {}


def register(
    engine_id: str,
    host: engine.HostCache,
    file_layout: tuple[FilePageSegment, ...],
) -> None:
    with _LOCK:
        if engine_id in _POOLS:
            raise RuntimeError(f"host cache is already registered for engine {engine_id!r}")
        if not file_layout:
            raise ValueError("vLLM filesystem layout needs at least one layer page")
        _POOLS[engine_id] = _Registration(host, file_layout)


def get(engine_id: str) -> _Registration:
    with _LOCK:
        try:
            return _POOLS[engine_id]
        except KeyError as error:
            raise RuntimeError(
                f"engine host cache {engine_id!r} has not been registered"
            ) from error


def unregister(engine_id: str, host: engine.HostCache) -> None:
    with _LOCK:
        registration = _POOLS.get(engine_id)
        if registration is not None and registration.cache is host:
            del _POOLS[engine_id]
