"""SGLang's LoRA adapters as engine adapters, and the ones each step's calls take.

- SGLang's tokenizer names an adapter and gives each load of it a new LoRA ID, by which its radix
  cache keys cached prefixes. `--lora-paths`, `/load_lora_adapter`,
  `/load_lora_adapter_from_tensors` and `/unload_lora_adapter` reach `Adapters`, SGLang's
  `lora_manager`, which builds the adapter of an ID on host as an engine host adapter, or drops
  it. A reload is another ID and another host adapter.
- The device holds adapters in the entries of SGLang's own pool (`pool`), taken before the cache
  sizes. Binding a step's adapters makes the pool load each one no entry holds into the entry its
  policy picks, before the step's first call; a call submitted earlier still reads what its entry
  held then.
- A step's `Binding` lists its requests' entries, and each engine request names its adapter by its
  index there; the base model takes none. SGLang admits a request to a batch when
  `validate_lora_batch` accepts the batch's adapters: at most `AdapterSpec.SLOTS`, and SGLang's own
  bound of the pool's entries and pins.
- A load's configuration, added tokens and factors are parsed once (`peft`). Malformed data,
  SGLang's refusals and the engine's are answered with no effect on the device; a device failure
  raises out of the scheduler, which then serves nothing without the adapter.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Final, NamedTuple, cast, final

import safetensors.torch
import torch
from sglang.srt.lora.utils import get_normalized_target_modules
from sglang.srt.managers.io_struct import LoRAUpdateOutput

import furiosa_opt_engine as engine
from furiosa_opt_sglang.lora.peft import Peft, factors
from furiosa_opt_sglang.lora.pool import Pool

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from sglang.srt.lora.lora_registry import LoRARef
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from transformers import PretrainedConfig

    from furiosa_opt_sglang.decoder import Decoder
    from furiosa_opt_sglang.lora.options import Lora

# The file of an adapter's factors that the engine reads.
WEIGHTS: Final = "adapter_model.safetensors"


class Binding(NamedTuple):
    """The adapters one step's calls take, and each LoRA ID's slot among them."""

    adapters: tuple[engine.decoder.Adapter, ...]
    slots: Mapping[str, int]

    def slot(self, lora_id: str | None) -> int | None:
        """The slot of `lora_id`, which the step bound; the base model, `None`, takes none."""
        return None if lora_id is None else self.slots[lora_id]


UNADAPTED: Final = Binding((), {})


@final
class Adapters:
    """SGLang's `lora_manager`: the adapters loaded for a decoder's calls, by LoRA ID, and SGLang's
    pool of the entries on its device."""

    # The platform serves no data-parallel attention, under which SGLang prepares adapters itself.
    attn_dp_enabled: Final = False

    def __init__(self, decoder: Decoder, options: Lora, config: PretrainedConfig) -> None:
        """The adapters of `decoder`'s calls, of the model of `config`, under SGLang's LoRA
        `options`; the pool takes its entries on the device now."""
        spec = decoder.model.adapters if decoder.model is not None else None
        assert spec is not None, "a load for adapters selects a model that runs them"
        self.spec: engine.decoder.AdapterSpec = spec
        self.vocabulary: int = decoder.vocabulary
        self.options: Lora = options
        # SGLang's normalized names of the modules an adapter may adapt; None admits any.
        targets = options.lora_target_modules
        self.targets: set[str] | None = (
            None if targets is None or "all" in targets else get_normalized_target_modules(targets)
        )
        # SGLang's manager's records: each loaded ID's load, and how many of them are pinned.
        self.lora_refs: dict[str, LoRARef] = {}
        self.num_pinned_loras: int = 0
        # Each loaded ID's adapter on host, which the pool loads into the entry it picks.
        self.hosts: dict[str, engine.decoder.HostAdapter] = {}
        self.memory_pool: Pool = Pool(spec, self.hosts, config, options)
        # SGLang seats the base model first.
        _ = self.bind([None])

    def load_lora_adapter(self, ref: LoRARef) -> LoRAUpdateOutput:
        """Build the PEFT checkpoint directory or hub repository `ref` names on host."""
        try:
            directory = Path(cast("str", ref.lora_path))
            if not directory.is_dir():
                import huggingface_hub

                # Hub overloads include untyped progress and user-agent parameters.
                download = cast("Callable[..., str]", huggingface_hub.snapshot_download)
                directory = Path(download(str(directory)))
            peft = self._admit(ref, Peft.read(directory, self.vocabulary))
            if not (directory / WEIGHTS).is_file():
                raise ValueError(f"reads {WEIGHTS}")
            checkpoint = engine.Checkpoint(directory, peft.config)
        except (OSError, ValueError) as error:
            return self._refused(ref, error)
        return self._build(ref, checkpoint)

    def load_lora_adapter_from_tensors(
        self,
        ref: LoRARef,
        tensors: object,
        config_dict: object,
        added_tokens_config: object = None,
    ) -> LoRAUpdateOutput:
        """Build the adapter of the factors `tensors`, the PEFT configuration `config_dict` and the
        added tokens `added_tokens_config` of a request on host, as the directory holding them
        would load."""
        with TemporaryDirectory() as directory:
            try:
                peft = Peft.parse(config_dict, added_tokens_config, self.vocabulary)
                _ = self._admit(ref, peft)
                # Safetensors leaves its metadata parameter's dict untyped.
                save = cast(
                    "Callable[[dict[str, torch.Tensor], Path], None]", safetensors.torch.save_file
                )
                save(factors(tensors), Path(directory) / WEIGHTS)
                checkpoint = engine.Checkpoint(directory, peft.config)
            except (OSError, ValueError) as error:
                return self._refused(ref, error)
            return self._build(ref, checkpoint)

    def unload_lora_adapter(self, ref: LoRARef) -> LoRAUpdateOutput:
        """Drop the adapter of `ref`'s ID, which SGLang unloads once no request names it, and free
        its entry; a step already running keeps reading what the entry holds."""
        _ = self.memory_pool.remove_lora(ref.lora_id)
        _ = self.hosts.pop(ref.lora_id, None)
        if (loaded := self.lora_refs.pop(ref.lora_id, None)) is not None:
            self.num_pinned_loras -= bool(loaded.pinned)
        return self._result()

    def validate_lora_batch(self, lora_ids: set[str | None]) -> bool:
        """Whether one batch runs every adapter of `lora_ids`: SGLang's bound of the pool's
        entries, of which the base model, `None`, takes one and pinned adapters keep theirs, and
        the engine's slots of a call, which the base model takes none of."""
        entries = self.memory_pool.max_loras_per_batch
        if len(lora_ids) > entries or len(lora_ids - {None}) > engine.decoder.AdapterSpec.SLOTS:
            return False
        if self.num_pinned_loras == 0:
            return True
        pinned = sum(
            bool(self.lora_refs[lora_id].pinned) for lora_id in lora_ids if lora_id is not None
        )
        return len(lora_ids) - pinned <= entries - self.num_pinned_loras

    def fetch_new_loras(self, new_loras: set[str | None]) -> None:
        """Nothing to fetch here: binding a step's adapters loads those its entries miss."""
        _ = new_loras

    def prepare_lora_batch(self, forward_batch: ForwardBatch) -> None:
        """Nothing to prepare here: each engine call binds its requests' adapters itself."""
        _ = forward_batch

    def bind(self, lora_ids: Iterable[str | None]) -> Binding:
        """The entries holding the adapters of `lora_ids`, in the order they first name them, after
        SGLang's pool loaded those it missed; `None` names the base model."""
        ids = list(lora_ids)
        self.memory_pool.prepare_lora_batch(set(ids), {}, [], self.lora_refs, None, None)
        named = list(dict.fromkeys(lora_id for lora_id in ids if lora_id is not None))
        entries = self.memory_pool.uid_to_buffer_id
        adapters = tuple(self.memory_pool.entries[entries[lora_id]] for lora_id in named)
        return Binding(adapters, {lora_id: slot for slot, lora_id in enumerate(named)})

    def _admit(self, ref: LoRARef, peft: Peft) -> Peft:
        """`peft`, the configuration of `ref`, after SGLang's refusals: a name already loaded, added
        tokens, a rank above `--max-lora-rank`, modules outside `--lora-target-modules` and a pin
        of the pool's last unpinned entry."""
        assert ref.lora_id not in self.lora_refs, "SGLang gives each load a new LoRA ID"
        assert ref.lora_name is not None and ref.lora_path is not None, "SGLang names each load"
        if any(held.lora_name == ref.lora_name for held in self.lora_refs.values()):
            # SGLang's tokenizer tells a concurrent reload by this wording.
            raise ValueError("it is already loaded")
        if ref.pinned and self.num_pinned_loras >= self.memory_pool.max_loras_per_batch - 1:
            # SGLang keeps one entry of the pool for the base model and unpinned adapters.
            raise ValueError("pins the last unpinned entry of --max-loras-per-batch")
        if peft.added > 0:
            raise ValueError("serves no adapter that adds tokens to the vocabulary")
        if (bound := self.options.max_lora_rank) is not None and peft.rank > bound:
            raise ValueError(f"rank {peft.rank} exceeds --max-lora-rank {bound}")
        # An adapter of every module adapts those the targets name; the engine refuses the others.
        targets = self.targets
        if targets is not None and "all" not in peft.modules and not peft.modules <= targets:
            raise ValueError(f"{sorted(peft.modules - targets)} are not in --lora-target-modules")
        return peft

    def _build(self, ref: LoRARef, checkpoint: engine.Checkpoint) -> LoRAUpdateOutput:
        """Build the adapter of `ref` from `checkpoint` on host, answering the engine's refusals;
        any other failure raises."""
        try:
            host = engine.decoder.HostAdapter(self.spec, checkpoint)
        except (OSError, ValueError, NotImplementedError) as error:
            return self._refused(ref, error)
        self.hosts[ref.lora_id] = host
        self.lora_refs[ref.lora_id] = ref
        self.num_pinned_loras += bool(ref.pinned)
        return self._result()

    def _refused(self, ref: LoRARef, error: Exception) -> LoRAUpdateOutput:
        """SGLang's answer to a load of `ref` that `error` refused."""
        return self._result(f"LoRA adapter {ref.lora_name}: {error}")

    def _result(self, error: str | None = None) -> LoRAUpdateOutput:
        """SGLang's answer to a load or an unload, naming every loaded adapter."""
        loaded: dict[str, str | LoRARef] = {
            held.lora_name: held.lora_path
            for held in self.lora_refs.values()
            if held.lora_name is not None and held.lora_path is not None
        }
        return LoRAUpdateOutput(
            success=error is None, error_message=error or "", loaded_adapters=loaded
        )
