"""The LoRA server options the plugin serves, which the platform checks at startup."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Final

from pydantic import AfterValidator, Field
from sglang.srt.lora.utils import get_normalized_target_modules

from furiosa_opt_sglang.admission import Refused, Served, Unset

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


# SGLang's names of the projections an adapter adds to; `all` names every one the model adapts.
PROJECTIONS: Final = frozenset(
    {
        *("q_proj", "k_proj", "v_proj", "qkv_proj", "o_proj"),
        *("gate_proj", "up_proj", "gate_up_proj", "down_proj", "all"),
    }
)


def projections(modules: set[str] | None) -> set[str] | None:
    """`modules` when each is an attention or MLP projection."""
    if modules is not None and not modules <= PROJECTIONS:
        raise ValueError(sorted(modules - PROJECTIONS))
    return modules


class Lora(Served):
    """The LoRA options the plugin serves: the engine adds each adapter's updates to the attention
    and MLP projections, from the entries of SGLang's own adapter pool."""

    enable_lora_overlap_loading: Unset = Field(
        description=(
            "loads adapters into the pool's entries on the scheduler thread; omit "
            "--enable-lora-overlap-loading"
        )
    )
    experts_shared_outer_loras: Unset = Field(description="applies no LoRA to experts")
    lora_use_virtual_experts: Unset = Field(description="applies no LoRA to experts")
    lora_target_modules: Annotated[set[str] | None, AfterValidator(projections)] = Field(
        description="applies LoRA to the attention and MLP projections only"
    )
    lora_eviction_policy: str
    max_lora_rank: int | None
    max_loras_per_batch: int

    def modules(self) -> set[str]:
        """SGLang's normalized names of the modules an adapter may adapt: those
        `--lora-target-modules` names, or every projection."""
        targets = self.lora_target_modules
        if targets is None or "all" in targets:
            targets = PROJECTIONS - {"all"}
        return get_normalized_target_modules(targets)


def admit(server_args: ServerArgs) -> None:
    """Refuse LoRA options past the adapter terms the engine compiles."""
    import furiosa_opt_engine as engine

    settings = Lora.parse(server_args)
    if (settings.max_lora_rank or 0) > (rank := engine.decoder.AdapterSpec.RANK):
        raise Refused("max_lora_rank", f"supports --max-lora-rank up to {rank}")
