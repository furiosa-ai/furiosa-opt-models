"""Worker host slots shared with the vLLM scheduler and filesystem tier."""

from __future__ import annotations

from threading import Lock

import furiosa_opt_engine as engine

_LOCK = Lock()
_POOLS: dict[str, engine.HostCache] = {}


def register(engine_id: str, host: engine.HostCache) -> None:
    with _LOCK:
        if engine_id in _POOLS:
            raise RuntimeError(f"host cache is already registered for engine {engine_id!r}")
        _POOLS[engine_id] = host


def get(engine_id: str) -> engine.HostCache:
    with _LOCK:
        try:
            return _POOLS[engine_id]
        except KeyError as error:
            raise RuntimeError(
                f"engine host cache {engine_id!r} has not been registered"
            ) from error


def unregister(engine_id: str, host: engine.HostCache) -> None:
    with _LOCK:
        if _POOLS.get(engine_id) is host:
            del _POOLS[engine_id]
