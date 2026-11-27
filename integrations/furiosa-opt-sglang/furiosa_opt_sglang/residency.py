"""SGLang's memory saver on the engine: releasing memory by tag drops what holds it, and resuming
loads or allocates it again.

- Only `--enable-memory-saver` releases memory, as with SGLang's own allocator.
- With `--enable-weights-cpu-backup`, released weights keep the copy of the checkpoint's tensors in
  host memory that the server loaded, and resuming loads it again. Without it, resuming reads the
  checkpoint's files as they are then, so releasing the weights also flushes the cache SGLang
  keeps of them, as its own weight updates do.
- A released KV cache keeps its spec and block count, from which resuming allocates it.
- Resuming restores one tag after another, each awake once restored; `cuda_graph` names no memory
  of the engine.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import TYPE_CHECKING, Final, assert_never, cast

import furiosa_opt_engine as engine
from furiosa_opt_sglang.cache import Cache
from furiosa_opt_sglang.decoder import Decoder
from furiosa_opt_sglang.generation.model import Model
from furiosa_opt_sglang.runner import Storage, Weights

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

    from furiosa_opt_sglang.runner import Module, Tag

# The memory the memory saver moves, in the order resuming restores it, as SGLang resumes it.
TAGS: Final[tuple[Tag, ...]] = ("weights", "kv_cache")


def served(module: Module) -> tuple[Model, Decoder, Cache]:
    """The model `module` serves, the decoder that holds its weights and the token pool that holds
    its cache storage."""
    match module.loaded:
        case Model(core=Decoder() as decoder, cache=Cache() as cache) as model:
            return model, decoder, cache
        case _:
            raise NotImplementedError("the memory saver moves the memory of a decoder only")


def release(
    module: Module,
    tags: Collection[str] | None,
    flush: Callable[[], bool],
    *,
    transfer_managers: Iterable[object] = (),
) -> None:
    """Release the memory of the awake `tags`, or of every tag when none are named, as SGLang
    names them; `flush` drops SGLang's cache."""
    from sglang.srt.runtime_context import get_exec

    # SGLang's configuration bags leave their fields untyped.
    if not cast("bool", get_exec().features.enable_memory_saver):
        return
    model, decoder, cache = served(module)
    for tag in reversed(TAGS):
        if (tags and tag not in tags) or tag in module.asleep:
            continue
        match tag:
            case "kv_cache":
                storage = cache.storage
                assert storage is not None, "an awake KV cache holds its storage"
                from furiosa_opt_sglang.staging import (
                    attach_manager_staging,
                    release_manager_staging,
                )

                managers = tuple(
                    manager
                    for manager in transfer_managers
                    if getattr(manager, "engine_cache", None) is cache
                )
                errors = []
                for manager in managers:
                    try:
                        release_manager_staging(manager)
                    except BaseException as error:
                        errors.append(error)
                try:
                    cache.release_copy_copier()
                except BaseException as error:
                    errors.append(error)
                if errors:
                    restore_errors = []
                    for manager in managers:
                        try:
                            attach_manager_staging(manager, storage, cache.pages.chips)
                        except BaseException as error:
                            restore_errors.append(error)
                    if restore_errors:
                        raise BaseExceptionGroup(
                            "SGLang cache release and staging restore failed",
                            [*errors, *restore_errors],
                        ) from None
                    raise BaseExceptionGroup(
                        "SGLang cache release and staging teardown failed", errors
                    ) from None
                module.asleep[tag] = Storage(storage.spec, storage.blocks)
                cache.storage = None
            case "weights":
                module.loaded = replace(model, core=replace(decoder, model=None, copy=None))
                module.asleep[tag] = Weights(decoder.copy)
                # Resumed from the files, the weights may no longer be those the cached rows hold.
                if decoder.copy is None and "kv_cache" not in module.asleep and not flush():
                    raise RuntimeError("SGLang kept requests while releasing the weights")
            case _:
                assert_never(tag)


def resume(
    module: Module,
    tags: Collection[str] | None,
    *,
    transfer_managers: Iterable[object] = (),
) -> None:
    """Restore the memory of `tags`, or of every tag when none are named."""
    for tag in TAGS:
        if tags and tag not in tags:
            continue
        model, decoder, cache = served(module)
        released = module.asleep.get(tag)
        match released:
            case Weights(copy):
                source = decoder.checkpoint.mapped() if copy is None else copy
                module.loaded = replace(model, core=replace(decoder, copy=copy).loaded(source))
            case Storage(spec, blocks):
                storage = engine.decoder.Cache(spec, blocks)
                cache.storage = storage
                from furiosa_opt_sglang.staging import (
                    attach_manager_staging,
                    release_manager_staging,
                )

                attached = []
                try:
                    for manager in transfer_managers:
                        if getattr(manager, "engine_cache", None) is not cache:
                            continue
                        attached.append(manager)
                        attach_manager_staging(manager, storage, cache.pages.chips)
                except BaseException as attach_error:
                    cleanup_errors = []
                    for manager in reversed(attached):
                        try:
                            release_manager_staging(manager)
                        except BaseException as error:
                            cleanup_errors.append(error)
                    cache.storage = None
                    if cleanup_errors:
                        raise BaseExceptionGroup(
                            "SGLang cache resume and staging cleanup failed",
                            [attach_error, *cleanup_errors],
                        ) from None
                    raise
            case None:
                continue
            case _:
                assert_never(released)
        del module.asleep[tag]
