"""SGLang's LoRA memory pool over engine adapters: SGLang's own code decides which adapter each
entry holds, and the engine stores it.

- An entry is an engine adapter on the device, taken when the pool is built, before the cache
  sizes; SGLang's `--max-loras-per-batch` sets their count, the base model's entry included.
- A batch that names an adapter no entry holds makes SGLang pick an entry, by its eviction policy
  and pins, into which the adapter's host revision loads in place; SGLang publishes the entry as
  holding it only after the load returned, so a failed load leaves it unpublished.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, final

import torch
from sglang.srt.lora.mem_pool import LoRAMemoryPool
from typing_extensions import override

import furiosa_opt_engine as engine

if TYPE_CHECKING:
    from collections.abc import Mapping

    from sglang.srt.lora.layers import BaseLayerWithLoRA
    from sglang.srt.lora.lora import LoRAAdapter
    from transformers import AutoConfig, PretrainedConfig

    from furiosa_opt_sglang.lora.options import Lora


@final
class Model(torch.nn.Module):
    """The base model as SGLang's pool reads it: its configuration, its weights the engine's."""

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.config: PretrainedConfig = config


@final
class Pool(LoRAMemoryPool):
    """SGLang's pool whose entries are adapters of `spec` on its decoder's device, each loaded from
    the host revision `hosts` holds of the LoRA ID SGLang assigns it."""

    def __init__(
        self,
        spec: engine.decoder.AdapterSpec,
        hosts: Mapping[str, engine.decoder.HostAdapter],
        config: PretrainedConfig,
        options: Lora,
    ) -> None:
        """SGLang's pool of the model of `config` under the LoRA `options`, as SGLang builds one."""
        self.spec: engine.decoder.AdapterSpec = spec
        self.hosts: Mapping[str, engine.decoder.HostAdapter] = hosts
        self.entries: list[engine.decoder.Adapter] = []
        super().__init__(
            # SGLang annotates its model configuration as the factory that makes it.
            base_hf_config=cast("AutoConfig", cast("object", config)),
            max_loras_per_batch=options.max_loras_per_batch,
            dtype=torch.bfloat16,
            # The rank and modules every entry holds, which SGLang's pool only records.
            max_lora_rank=engine.decoder.AdapterSpec.RANK,
            target_modules=options.modules(),
            base_model=Model(config),
            eviction_policy=options.lora_eviction_policy,
            lora_added_tokens_size=0,
        )

    @override
    def init_buffers(self, base_model: torch.nn.Module) -> None:
        """Take every entry on the device, each updating no row."""
        _ = base_model
        self.entries = [engine.decoder.Adapter(self.spec) for _ in range(self.max_loras_per_batch)]

    @override
    def load_lora_weight_to_buffer(
        self,
        uid: str | None,
        buffer_id: int,
        lora_adapter: LoRAAdapter | None,
        lora_modules: list[dict[str, torch.nn.Module]],
        lora_embed_tokens_module: BaseLayerWithLoRA | None,
        lora_lm_head_module: BaseLayerWithLoRA | None,
    ) -> None:
        """Load the host revision of `uid` into the entry `buffer_id`; the base model, `None`,
        writes nothing, as no call lists its entry. A device failure raises out of the batch."""
        _ = lora_adapter, lora_modules, lora_embed_tokens_module, lora_lm_head_module
        if uid is not None:
            self.entries[buffer_id].load(self.hosts[uid])
