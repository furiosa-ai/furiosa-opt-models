"""vLLM's sleep mode on the engine: freeing device memory is dropping what holds it, and waking
loads or allocates it again.

- Sleep mode needs `enable_sleep_mode`, under which a load keeps the copy of the checkpoint's
  tensors in host memory that it loaded. Level 1 keeps that copy and waking loads it again;
  level 2 and `release` drop it, and waking reads the checkpoint's files into a new copy.
- A released KV cache keeps its spec and block count, from which waking allocates it at once.
- Waking restores one tag after another, each awake once restored, so a retry after a failure
  restores only what still sleeps.
- `reload` loads the awake weights again from the checkpoint's files, into the device memory the
  served ones free first.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Final

import furiosa_opt_engine as engine
from furiosa_opt_vllm.admission import Refused
from furiosa_opt_vllm.decoder import Decoder
from furiosa_opt_vllm.runner import Storage, Weights

if TYPE_CHECKING:
    from collections.abc import Collection

    from furiosa_opt_vllm.runner import Runner, Tag

# The memory sleep mode moves, in the order waking restores it; releasing goes the other way, so a
# cache that was never allocated is refused before the weights leave.
TAGS: Final[tuple[Tag, ...]] = ("weights", "kv_cache")


def decoder(runner: Runner) -> Decoder:
    """The loaded decoder, whose weights and cache storage sleep mode moves."""
    match runner.model:
        case Decoder():
            return runner.model
        case None:
            raise RuntimeError("load_model must precede sleep mode")
        case _:
            raise NotImplementedError("sleep mode moves the memory of a decoder only")


def cache(runner: Runner) -> engine.decoder.Cache:
    """The KV cache the runner holds now, which sleep mode releases and waking allocates anew."""
    if runner.cache is None:
        raise RuntimeError("wake_up must restore the KV cache before its blocks move")
    return runner.cache


def sleep(runner: Runner, level: int) -> None:
    """Release the weights and the KV cache; level 1 keeps the weights' copy."""
    release(runner, TAGS, decoder(runner).copy if level == 1 else None)


def release(runner: Runner, tags: Collection[str], copy: engine.Checkpoint | None = None) -> None:
    """Release the memory of the awake `tags`; released weights keep `copy`."""
    if not runner.vllm_config.model_config.enable_sleep_mode:
        raise Refused(
            "enable_sleep_mode",
            "sleeps only with enable_sleep_mode, whose loads keep the weights' copy for level 1",
        )
    served = decoder(runner)
    for tag in reversed(TAGS):
        if tag not in tags or tag in runner.asleep:
            continue
        match tag:
            case "weights":
                runner.model = replace(served, model=None, copy=None)
                runner.asleep[tag] = Weights(copy)
            case "kv_cache":
                if runner.cache is None:
                    raise RuntimeError("initialize_kv_cache must precede sleep mode")
                cache = runner.cache
                _release_furiosa_cache_connectors(cache)
                runner.asleep[tag] = Storage(cache.spec, cache.blocks)
                runner.cache = None


def wake_up(runner: Runner, tags: Collection[str] | None) -> None:
    """Restore the memory of `tags`, or of every tag."""
    for tag in TAGS:
        if tags is not None and tag not in tags:
            continue
        match runner.asleep.get(tag):
            case Weights(copy):
                released = decoder(runner)
                copy = released.checkpoint.mapped().read() if copy is None else copy
                runner.model = replace(released, copy=copy).loaded(copy)
            case Storage(spec, blocks):
                if runner.cache is None:
                    runner.cache = engine.decoder.Cache(spec, blocks)
                _restore_furiosa_cache_connectors(runner.cache)
            case None:
                continue
        del runner.asleep[tag]


def _cache_connectors() -> list[Any]:
    from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group

    if not has_kv_transfer_group():
        return []
    from furiosa_opt_vllm.model_runner import leaves

    return leaves(get_kv_transfer_group())


def _cache_connector_hooks() -> list[tuple[Any, Any]]:
    hooks = []
    for connector in _cache_connectors():
        release_cache = getattr(connector, "_release_furiosa_engine_cache", None)
        restore_cache = getattr(connector, "_restore_furiosa_engine_cache", None)
        if release_cache is None and restore_cache is None:
            continue
        if not callable(release_cache) or not callable(restore_cache):
            raise TypeError(
                "Furiosa cache connectors must implement release and restore hooks together"
            )
        hooks.append((release_cache, restore_cache))
    return hooks


def _release_furiosa_cache_connectors(cache: engine.decoder.Cache) -> None:
    attempted = []
    try:
        for release_cache, restore_cache in _cache_connector_hooks():
            attempted.append(restore_cache)
            release_cache(cache)
    except BaseException as release_error:
        restore_errors = []
        for restore_cache in reversed(attempted):
            try:
                restore_cache(cache)
            except BaseException as error:
                restore_errors.append(error)
        if restore_errors:
            raise BaseExceptionGroup(
                "connector cache release and rollback failed",
                [release_error, *restore_errors],
            ) from None
        raise


def _restore_furiosa_cache_connectors(cache: engine.decoder.Cache) -> None:
    for _, restore_cache in _cache_connector_hooks():
        restore_cache(cache)


def reload(runner: Runner) -> None:
    """Load the awake weights again from the checkpoint's files into the device memory the served
    ones free first; under sleep mode from a new copy, which replaces the decoder's once they load.
    The files are read before the served weights leave the device, so files that cannot be read
    keep them serving; a failed load leaves no weights on the device and the decoder its copy."""
    if "weights" in runner.asleep:
        raise RuntimeError("reload_weights requires awake weights; wake them up first")
    served = decoder(runner)
    mapped = served.checkpoint.mapped()
    copy = mapped.read() if runner.vllm_config.model_config.enable_sleep_mode else None
    # The load takes the served weights' memory: their calls end (vLLM does not pause its steps
    # for reload_weights), then their last reference drops.
    served.device.synchronize()
    runner.model = released = replace(served, model=None)
    del served
    runner.model = replace(released, copy=copy).loaded(mapped if copy is None else copy)
