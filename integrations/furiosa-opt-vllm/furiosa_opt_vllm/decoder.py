"""A decoder the engine loaded onto the chips of vLLM's local ranks, and the rows its calls take.

- `Decoder.load` loads the checkpoint as its causal model, whose calls return each kept row's
  greedy token, or as its base model, whose calls return the kept rows' hidden values, when vLLM
  pools the hidden rows of a causal model.
- A decoder keeps where its checkpoint is and the options it loads with: sleep mode drops the
  loaded model to free its device memory, and `loaded` brings the weights back by a new load.
- Under sleep mode the decoder loads a copy of the checkpoint's tensors in host memory and keeps
  it, so level 1 wakes to the weights it loaded whatever later happens to the files.
- Under `VLLM_BATCH_INVARIANT=1` the load is batch invariant, and so is every load again: a row's
  values keep their bits whatever other rows share its calls.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, Final

from vllm.platforms import current_platform

import furiosa_opt_engine as engine
from furiosa_opt_vllm import checkpoint
from furiosa_opt_vllm.admission import Refused, options

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from furiosa_opt_vllm.checkpoint import Location

# Every PE of an RNGD chip.
PES: Final = range(8)

RELEASED: Final = "no weights are on the device: wake them up, or reload them after a failed reload"


@dataclass(frozen=True, slots=True)
class Decoder:
    """A loaded decoder, where its checkpoint is and the options it loads with, the device it runs
    on and the vocabulary its token IDs index."""

    # None while sleep mode, or a reload that failed, left the weights off the device.
    model: engine.Decoder | None
    # Under sleep mode, the checkpoint read into host memory that the weights last loaded from;
    # None without sleep mode, and while sleep mode holds it for the released weights.
    copy: engine.Checkpoint | None
    checkpoint: Location
    options: engine.LoadOptions
    device: engine.Device
    vocabulary: int
    # A call allocates its buffers as it runs, from the memory the cache leaves free.
    workspace: ClassVar[int] = 0

    @classmethod
    def load(cls, config: VllmConfig) -> Decoder:
        """The checkpoint of `config` on the chips vLLM assigns its local ranks, every PE of each,
        with the options the platform checked at startup; under sleep mode, loaded from its copy.
        Refuses a resolved `max_model_len` past the decoder's window."""
        model = config.model_config
        tensor = config.parallel_config.tensor_parallel_size
        chips = [current_platform.device_id_to_physical_device_id(rank) for rank in range(tensor)]
        device = engine.Device(engine.Placement(chips, PES))
        located = checkpoint.located(model, config.load_config.download_dir)
        mapped = located.mapped()
        copy = mapped.read() if model.enable_sleep_mode else None
        decoder = cls(None, copy, located, options(config), device, model.get_vocab_size())
        loaded = decoder.loaded(mapped if copy is None else copy)
        match loaded.model:
            case engine.Decoder(window=window) if model.max_model_len > window:
                raise Refused("max_model_len", f"requires max_model_len at most {window}")
            case _:
                return loaded

    def loaded(self, checkpoint: engine.Checkpoint) -> Decoder:
        """This decoder with the weights of `checkpoint` loaded onto its device by its options."""
        model = self.options.load(checkpoint, self.device)
        match model:
            case engine.Decoder(output=engine.decoder.Output.TOKENS | engine.decoder.Output.HIDDEN):
                return replace(self, model=model)
            case _:
                raise NotImplementedError(f"{type(model).__name__} is no served decoder")

    @property
    def rows(self) -> int:
        """Rows one call runs at most."""
        if self.model is None:
            raise RuntimeError(RELEASED)
        return self.model.rows
