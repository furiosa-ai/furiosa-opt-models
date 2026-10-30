"""A decoder the engine loaded onto a scheduler's chips, and the rows and blocks its calls take.

- `Decoder.load` loads a checkpoint as its causal model, whose calls return each kept row's greedy
  token, or as its base model, whose calls return the kept rows' hidden values.
- A served decoder caches every layer in one paged group of full-history attention, which SGLang's
  token pool pages as one pool.
- A decoder keeps where its checkpoint is (`Location`) and the options it loads with: SGLang's
  memory saver drops the loaded model to free its device memory, and `loaded` brings the weights
  back by a new load. With a backup, the decoder loads a copy of the checkpoint's tensors in host
  memory and keeps it, so the weights come back as the server loaded them whatever happens to the
  files.
- `chips` maps SGLang's device index of a scheduler to the physical chips it drives.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

import furiosa_opt_engine as engine

if TYPE_CHECKING:
    from pydantic import JsonValue

# Every PE of an RNGD chip.
PES: Final = range(8)

RELEASED: Final = "SGLang's memory saver released the device memory this call reads: resume it"


def chips(first: int, count: int) -> list[int]:
    """The `count` chips from position `first` of `FURIOSA_VISIBLE_CHIPS`, or of every installed
    chip in index order when it is unset, as SGLang maps a device index to a visible device."""
    visible = os.environ.get("FURIOSA_VISIBLE_CHIPS", "")
    listed = [int(chip) for chip in visible.split(",") if chip.strip()]
    chosen = listed[first : first + count] if listed else list(range(first, first + count))
    if len(chosen) != count:
        raise ValueError(
            f"FURIOSA_VISIBLE_CHIPS lists {len(listed)} chips; the scheduler needs {first + count}"
        )
    return chosen


@dataclass(frozen=True, slots=True)
class Location:
    """A checkpoint's local directory and the config it loads under, which every map of it keeps,
    as SGLang keeps the config it resolved at startup; None is the directory's own."""

    directory: str
    config: dict[str, JsonValue] | None

    def mapped(self) -> engine.Checkpoint:
        """The checkpoint's tensors, mapped from its files as they are now."""
        return engine.Checkpoint(self.directory, config=self.config)


@dataclass(frozen=True, slots=True)
class Decoder:
    """A loaded decoder, where its checkpoint is and the options it loads with, the device it runs
    on, the vocabulary its token IDs index, and the block format and layers of its paged cache
    group."""

    # None while SGLang's memory saver released the weights.
    model: engine.Decoder | None
    # With a backup, the checkpoint read into host memory that the weights last loaded from; None
    # without one, and while the released weights hold it.
    copy: engine.Checkpoint | None
    checkpoint: Location
    options: engine.LoadOptions
    device: engine.Device
    vocabulary: int
    pool: engine.decoder.Pool
    layers: range

    @classmethod
    def load(
        cls,
        checkpoint: Location,
        options: engine.LoadOptions,
        device: engine.Device,
        vocabulary: int,
        *,
        backup: bool = False,
    ) -> Decoder:
        """`checkpoint` loaded with `options` onto `device`, whose token IDs lie below `vocabulary`;
        with `backup`, loaded from a copy of its tensors in host memory, which the decoder keeps.
        Refuses a decoder whose calls return scores or whose cache is not one paged full-history
        group."""
        mapped = checkpoint.mapped()
        copy = mapped.read() if backup else None
        loaded = options.load(mapped if copy is None else copy, device)
        match loaded:
            case engine.Decoder(output=engine.decoder.Output.TOKENS | engine.decoder.Output.HIDDEN):
                pass
            case _:
                raise NotImplementedError(f"{type(loaded).__name__} is no served decoder")
        match loaded.cache.groups:
            case [engine.decoder.Group(kind=engine.decoder.Kind.Paged(), window=None) as group]:
                pool = loaded.cache.pools[group.pool]
                return cls(
                    loaded, copy, checkpoint, options, device, vocabulary, pool, group.layers
                )
            case _:
                raise NotImplementedError("the plugin pages only one full-history attention group")

    def loaded(self, checkpoint: engine.Checkpoint) -> Decoder:
        """This decoder with the weights of `checkpoint` loaded onto its device by its options: the
        line of its first load, whose cache format it keeps."""
        return replace(self, model=self.options.load(checkpoint, self.device))

    @property
    def rows(self) -> int:
        """Rows one call runs at most."""
        if self.model is None:
            raise RuntimeError(RELEASED)
        return self.model.rows

    def blocks(self, memory: int, *others: Decoder) -> int:
        """Cache blocks that `memory` bytes a chip hold for this decoder and `others`, which number
        their blocks alike, within the I32 byte offsets a layer addresses its blocks by."""
        decoders = (self, *others)
        page = sum(decoder.pool.bytes * len(decoder.layers) for decoder in decoders)
        return min(max(0, memory) // page, *(2**31 // decoder.pool.bytes for decoder in decoders))
