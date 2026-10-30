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
from typing import TYPE_CHECKING, Final

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
                runner.asleep[tag] = Storage(runner.cache.spec, runner.cache.blocks)
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
                runner.cache = engine.decoder.Cache(spec, blocks)
            case None:
                continue
        del runner.asleep[tag]


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
