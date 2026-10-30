"""Keep scheduler tensors on the host; the extension owns device execution."""

from __future__ import annotations

import os
from collections.abc import Mapping
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Annotated,
    ClassVar,
    Final,
    Literal,
    Self,
    cast,
    final,
)

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, RootModel, ValidationError
from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform
from typing_extensions import override

if TYPE_CHECKING:
    from pydantic import JsonValue
    from sglang.srt.server_args import ServerArgs
    from transformers import PretrainedConfig

CacheDtype = Literal["auto", "bf16", "bfloat16", "fp8", "fp8_e4m3"]
# SGLang's spellings of a `--kv-cache-dtype` of one element type, under the engine's dtype name;
# `auto` keeps the engine's. Whether a model caches in that dtype is the engine's to decide.
CACHE_DTYPES: Final[Mapping[CacheDtype, str | None]] = MappingProxyType(
    {
        "auto": None,
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp8": "float8_e4m3fn",
        "fp8_e4m3": "float8_e4m3fn",
    }
)


def unset(value: object) -> object:
    """`value` when it leaves its feature off: false, empty or none."""
    if value:
        raise ValueError("set")
    return value


# A feature the plugin does not serve: any false, empty or missing value leaves it off.
Unset = Annotated[object, AfterValidator(unset)]


class Served(BaseModel):
    """Values the plugin serves of the fields it reads; parsing a record refuses any other value,
    naming the field and, when it has one, its description as the requirement."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, from_attributes=True)

    @classmethod
    def parse(cls, source: object) -> Self:
        try:
            return cls.model_validate(source)
        except ValidationError as error:
            raise cls.refused(str(error.errors()[0]["loc"][0])) from None

    @classmethod
    def refused(cls, field: str) -> Refused:
        """The refusal of `field`'s value, naming its description as the requirement."""
        requirement = cls.model_fields[field].description
        return Refused(field, requirement or f"does not support {field}")


class Requested(Served):
    """The operator's choice of the options the platform pins or the engine replaces; SGLang's
    own defaults for them stand."""

    dtype: Literal["auto", "bfloat16"] = Field(description="requires bfloat16 weights")
    load_format: Literal["auto", "safetensors"] = Field(description="loads safetensors checkpoints")
    device: Literal["cpu"] | None = Field(
        description="keeps framework tensors on the host; omit --device"
    )
    attention_backend: Literal["torch_native"] | None = Field(
        description="runs attention in the engine; omit --attention-backend"
    )
    prefill_attention_backend: Literal["torch_native"] | None = Field(
        description="runs attention in the engine; omit --prefill-attention-backend"
    )
    decode_attention_backend: Literal["torch_native"] | None = Field(
        description="runs attention in the engine; omit --decode-attention-backend"
    )


LORA: Final = "serves no LoRA adapter; omit --enable-lora and --lora-paths"
QUANTIZATION: Final = "loads each checkpoint's own weights and quantizes none; omit --quantization"


class Resolved(Served):
    """The options SGLang derived from the operator's flags, among the values the plugin serves."""

    ep_size: Literal[1] = Field(description="requires ep_size=1")
    pp_size: Literal[1] = Field(description="requires pp_size=1")
    nnodes: Literal[1] = Field(description="requires nnodes=1")
    # The platform spaces data-parallel replicas a tensor degree of chips apart.
    gpu_id_step: Literal[1] = Field(description="requires gpu_id_step=1")
    dcp_size: Literal[1] = Field(description="requires dcp_size=1")
    attn_cp_size: Literal[1] = Field(description="requires attn_cp_size=1")
    attn_dp_size: Literal[1] = Field(description="requires attn_dp_size=1")
    enable_lora: Unset = Field(description=LORA)
    lora_paths: Unset = Field(description=LORA)
    quantization: Unset = Field(description=QUANTIZATION)
    quantization_param_path: Unset
    enable_hisparse: Unset
    enable_return_hidden_states: Unset
    enable_return_routed_experts: Unset
    enable_return_indexer_topk: Unset
    enable_watermark: Unset = Field(
        description="serves no watermarking, which samples from scores the engine does not return"
    )
    enable_strict_thinking: Unset = Field(
        description="serves no strict thinking, which masks scores the engine does not return"
    )
    dllm_algorithm: Unset
    enable_unified_memory: Unset
    enable_lmcache: Unset
    enable_flexkv: Unset
    enable_unified_cache_external_linker: Unset
    prefill_only_disable_kv_cache: Unset
    # No encoder output leaves the device for another instance.
    encoder_only: Unset
    language_only: Unset
    # The engine places every weight it loads; SGLang's per-layer embedding offload moves tables of
    # its own model, and its backend and directory options take effect only with it.
    ple_offload_embedding: Unset
    disaggregation_mode: Literal["null"] = Field(description="requires disaggregation_mode=null")
    startup_weight_load_mode: Literal["serial"] = Field(
        description="requires startup_weight_load_mode=serial"
    )
    kv_cache_dtype: CacheDtype = Field(
        description=(
            "names the engine a cache dtype only for --kv-cache-dtype auto, bf16, bfloat16, fp8 "
            "or fp8_e4m3"
        )
    )
    disable_overlap_schedule: bool
    enable_deterministic_inference: bool
    enable_hierarchical_cache: bool


class Hierarchical(Served):
    """The hierarchical cache's options: a host tier of the engine cache's pages, which the engine
    moves. The platform parses them to refuse, and the host-pool strategy to build the tier."""

    # The host tier holds `hicache_size` GB when set, else `hicache_ratio` times the device pool.
    hicache_ratio: float
    hicache_size: int
    hicache_write_policy: str
    hicache_storage_backend: Unset = Field(
        description="keeps the hierarchical cache in host memory; omit --hicache-storage-backend"
    )
    hicache_host_memory_mode: Literal["cache"] = Field(
        description="keeps host pages as a cache tier; omit --hicache-host-memory-mode"
    )
    hicache_io_backend: Literal["kernel"] = Field(
        description="moves host pages through the engine; omit --hicache-io-backend"
    )
    hicache_mem_layout: Literal["page_first"] = Field(
        description="keeps host pages in the engine's layout; omit --hicache-mem-layout"
    )
    # SGLang resolves the fraction to none for an explicit size; its auto-sizing reads only the
    # device pools it builds itself.
    hicache_host_memory_fraction: None = Field(
        description="sizes the hierarchical cache by --hicache-ratio or --hicache-size"
    )
    # The memory saver frees the cache storage and resumes into new storage, while the host tier
    # keeps copying the storage it was built over; SGLang's memory options, which the strategy
    # parses, leave it out.
    enable_memory_saver: Unset = Field(
        default=False,
        description="releases no memory under a hierarchical cache yet; omit --enable-memory-saver",
    )


class Speculation(Served):
    """The speculative decoding options the plugin serves: one chain of drafts a request, which the
    target verifies within the step that proposed it."""

    dp_size: Literal[1] = Field(description="requires dp_size=1 with speculative decoding")
    # The speculative worker and a draft model hold device memory the memory saver does not release.
    enable_memory_saver: Unset = Field(
        description="releases no memory under speculative decoding yet; omit --enable-memory-saver"
    )
    speculative_adaptive: Unset
    speculative_ngram_external_corpus_path: Unset
    speculative_ngram_external_sam_budget: Unset


class Ngram(Speculation):
    """NGRAM speculation's options: a corpus of the requests' own n-grams, searched one candidate a
    position."""

    speculative_algorithm: Literal["NGRAM"]
    speculative_ngram_min_bfs_breadth: Literal[1] = Field(
        description="requires speculative_ngram_min_bfs_breadth=1"
    )
    speculative_ngram_max_bfs_breadth: Literal[1] = Field(
        description="requires speculative_ngram_max_bfs_breadth=1"
    )
    speculative_ngram_match_type: str
    speculative_ngram_capacity: int
    speculative_ngram_max_trie_depth: int
    speculative_ngram_external_corpus_max_tokens: int


class Standalone(Speculation):
    """STANDALONE speculation's options: a draft model, which drafts one candidate a position."""

    speculative_algorithm: Literal["STANDALONE"]
    speculative_eagle_topk: Literal[1] | None = Field(
        description="requires speculative_eagle_topk=1"
    )
    speculative_draft_model_path: str
    speculative_draft_model_revision: str | None


class Speculations(
    RootModel[Annotated[Ngram | Standalone, Field(discriminator="speculative_algorithm")]]
):
    """The options of each speculative algorithm the plugin serves."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, from_attributes=True)

    @classmethod
    def parse(cls, source: object) -> Ngram | Standalone | None:
        """`source`'s speculative decoding, as its algorithm's options, or none."""
        if getattr(source, "speculative_algorithm", None) is None:
            return None
        try:
            return cls.model_validate(source).root
        except ValidationError as error:
            match error.errors()[0]:
                case {"type": "union_tag_invalid", "ctx": {"expected_tags": str() as tags}}:
                    raise Refused(
                        "speculative_algorithm", f"speculative_algorithm supports {tags}"
                    ) from None
                case {"loc": (_, str() as field, *_)}:
                    raise next(
                        options.refused(field)
                        for options in Speculation.__subclasses__()
                        if field in options.model_fields
                    ) from None
                case _:
                    raise


class Recurrent(Served):
    """The recurrent-state options of a model that keeps recurrent state."""

    enable_hierarchical_cache: Unset = Field(
        description="keeps attention pages in host memory, not recurrent state"
    )
    enable_int8_mamba_checkpoint: Unset
    enable_linear_replayssm: Unset
    enable_linear_replayssm_spec: Unset
    mamba_radix_cache_strategy: Literal["auto", "no_buffer"] = Field(
        description="requires mamba_radix_cache_strategy in ('auto', 'no_buffer')"
    )
    mamba_ssm_dtype: Literal["float32"] | None = Field(
        description="requires mamba_ssm_dtype in (None, 'float32')"
    )


@final
class Refused(ValueError):
    """A server option the plugin cannot honor, and the requirement it violates."""

    # A plain exception: context managers and pickling set and read exception attributes.
    def __init__(self, option: str, requirement: str) -> None:
        super().__init__(option, requirement)
        self.option = option
        self.requirement = requirement

    @override
    def __str__(self) -> str:
        return f"furiosa-opt-sglang {self.requirement}"


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
