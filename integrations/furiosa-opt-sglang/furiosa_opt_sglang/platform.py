"""Keep scheduler tensors on the host; the extension owns device execution."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, cast, final

from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform
from typing_extensions import override

from furiosa_opt_sglang.admission import (
    Hierarchical,
    Recurrent,
    Refused,
    Requested,
    Resolved,
    Speculations,
)

if TYPE_CHECKING:
    from pydantic import JsonValue
    from sglang.srt.server_args import ServerArgs
    from transformers import PretrainedConfig


@final
class Platform(SRTPlatform):
    """Use CPU tensor operations for framework scheduling, not for model execution."""

    # Out of tree: SGLang takes memory capacity from this class instead of probing the host.
    _enum = PlatformEnum.OOT
    # Names the platform in SGLang's record of the server defaults it applied.
    device_name = "furiosa"
    # Framework tensors are torch CPU tensors; SGLang's process groups use gloo for this type.
    device_type = "cpu"

    @override
    def get_device(self, device_id: int = 0) -> str:
        # An unset `--device` resolves to this name, which keeps framework tensors on the host.
        return "cpu"

    @override
    def get_device_total_memory(self, device_id: int = 0) -> int:
        # SGLang sizes memory-dependent defaults from it; framework tensors live in host memory.
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")

    @override
    def apply_server_args_defaults(self, server_args: ServerArgs) -> None:
        # Refuses options the plugin does not serve and pins the scheduling the engine requires.
        from sglang.srt.arg_groups.model_override_base import model_config_of

        raw = requested(server_args)
        _ = Requested.parse(raw)
        resolved = Resolved.parse(server_args)
        speculative = Speculations.parse(server_args) is not None
        # One scheduler drives every chip of a split model, as SGLang's would drive one rank;
        # the engine reads the operator's degree, which SGLang's ranks would each load whole.
        # Each data-parallel replica's first chip then lies that many chips past the last one's.
        tensor = cast("int", raw.get("tp_size") or 1)
        server_args.gpu_id_step = tensor
        server_args.tp_size = 1
        # The engine allocates each chip's cache whole, so a model split over chips needs the
        # headroom vLLM's default utilization leaves; an operator's fraction stays.
        if raw.get("mem_fraction_static") is None:
            server_args.mem_fraction_static = 0.9
        server_args.device = "cpu"
        server_args.dtype = "bfloat16"
        model_config = model_config_of(server_args)
        # The engine refuses from the config alone, before a scheduler starts or a weight downloads,
        # what the scheduler's load of these options would.
        from furiosa_opt_sglang.checkpoint import configured
        from furiosa_opt_sglang.load import options

        overrides = cast("dict[str, JsonValue]", model_config.model_override_args)
        try:
            options(
                model_config,
                tensor,
                resolved.kv_cache_dtype,
                deterministic=resolved.enable_deterministic_inference,
            ).parse(configured(server_args.model_path, server_args.revision, overrides))
        except (NotImplementedError, ValueError) as error:
            raise Refused("model_path", str(error)) from None
        # SGLang leaves the checkpoint's Transformers config untyped.
        hf_config = cast("PretrainedConfig", model_config.hf_config)
        model_type = hf_config.model_type
        state = recurrent(model_type)
        if state:
            _ = Recurrent.parse(server_args)
        if resolved.enable_hierarchical_cache:
            _ = Hierarchical.parse(server_args)
        server_args.page_size = 1 if state else 16
        # A generation step returns before the device ends it; recurrent state, embeddings and
        # verified drafts are read back within the step, so overlapping only delays their results.
        server_args.disable_overlap_schedule = (
            resolved.disable_overlap_schedule
            or speculative
            or state
            or not model_config.is_generation
        )
        # Prompt chunks share a step with decode rows, so long prompts neither stall them nor wait
        # for a prefill wave; a speculative step verifies decode rows only.
        server_args.enable_mixed_chunk = (
            model_config.is_generation and not state and not speculative
        )
        server_args.disable_cuda_graph = True
        server_args.disable_piecewise_cuda_graph = True
        server_args.sampling_backend = "pytorch"


def requested(server_args: ServerArgs) -> dict[str, object]:
    """The options the operator passed, before resolution rules declared their defaults."""
    # Resolution snapshots every operator field on the record before its first rule runs.
    raw = cast("object", getattr(server_args, "_raw_input", None))
    if not isinstance(raw, dict):
        raise RuntimeError("SGLang resolved the server options without a snapshot of its input")
    return cast("dict[str, object]", raw)


def recurrent(model_type: str) -> bool:
    """Whether `model_type` keeps recurrent state."""
    return model_type in ("mamba", "mamba2", "bamba")
