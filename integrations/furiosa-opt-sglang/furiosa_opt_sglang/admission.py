"""The server options the plugin serves, as records whose parsing refuses any other value by the
option's name and requirement; the platform parses them before a scheduler starts."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, ClassVar, Final, Literal, Self, TypeAlias, final

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, RootModel, ValidationError
from typing_extensions import override

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
# A server's part in prefill-decode disaggregation, as SGLang names it.
Mode: TypeAlias = Literal["null", "prefill", "decode"]


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
# SGLang's quantizations at load whose computation the engine serves: E4M3 weights restored to
# BF16 before each product, the activations kept in BF16.
Quantization = Literal["fp8"]
QUANTIZATION: Final = (
    "quantizes a BF16 checkpoint at load only with --quantization fp8, one scale per output channel"
)


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
    quantization: Quantization | None = Field(description=QUANTIZATION)
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
    disaggregation_mode: Mode = Field(
        description="requires disaggregation_mode null, prefill or decode"
    )
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
    # SGLang gives the draft the target's quantization unless the operator names the draft's own.
    speculative_draft_model_quantization: Quantization | None = Field(
        description=(
            "quantizes a BF16 draft model at load only with fp8, one scale per output channel"
        )
    )


class Speculations(
    RootModel[Annotated[Ngram | Standalone, Field(discriminator="speculative_algorithm")]]
):
    """The options of each speculative algorithm the plugin serves."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, from_attributes=True)

    @classmethod
    def parse(cls, source: object) -> Ngram | Standalone | None:
        """`source`'s speculative decoding as SGLang resolved it, as its algorithm's options, or
        none."""
        from sglang.srt.arg_groups.model_override_base import resolving_view

        # A server record keeps the operator's spellings beside what SGLang resolved from them,
        # such as a draft model's `unquant`, which resolves to no quantization.
        resolved = resolving_view(source)
        if getattr(resolved, "speculative_algorithm", None) is None:
            return None
        try:
            return cls.model_validate(resolved).root
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


BACKEND: Final = "moves the cache only over --disaggregation-transfer-backend mooncake_tcp"
FAKE: Final = (
    "moves no KV over the fake transfer backend, a synthetic decode benchmark; use "
    + "--disaggregation-transfer-backend mooncake_tcp"
)


class Disaggregation(Served):
    """The prefill-decode disaggregation options the plugin serves: a transfer backend that
    registers the token pool's host image, whose pages the plugin keeps equal to the device's."""

    # SGLang's nixl and mori backends register the token pool as device memory, and the ascend
    # backend moves Ascend NPU memory.
    disaggregation_transfer_backend: Literal["mooncake", "fake"] = Field(description=BACKEND)
    # SGLang's host receive registers the hierarchical cache's host pool beside the token pool's
    # key and value buffers, which only its dense MHA pool exposes.
    disaggregation_decode_host_receive_threshold: float = Field(
        ge=1,
        description="receives KV only into the token pool's host image: SGLang receives into a "
        + "host pool only beside an MHA token pool's separate key and value buffers",
    )
    # SGLang's radix cache backs a retracted request up to its host pool only over an MHA token
    # pool (unified_radix_cache.supports_retraction_backup) and raises once the weights loaded.
    disaggregation_decode_retraction_backup: Literal["cpu_tensor", "none"] | None = Field(
        description="backs a retracted request up only by the token pool's cpu_tensor copies: "
        + "SGLang's host-pool backup needs an MHA token pool"
    )
    disaggregation_decode_enable_offload_kvcache: Unset = Field(
        description="offloads no decode server's cache: the engine's host pool serves no storage "
        + "backend"
    )
    disaggregation_enable_kv_checksum: Unset = Field(
        description="computes no KV checksum: SGLang's checksum runs a CUDA kernel over the "
        + "transfer regions"
    )
    enable_pd_role_switch: bool
    # SGLang's decode server stashes a received request's first token as a token relay
    # (ScheduleBatch.process_prebuilt), which its NGRAM future map reads as accepted drafts it
    # lacks (FutureMap.stash), and its scheduler raises.
    speculative_algorithm: Literal["STANDALONE"] | None = Field(
        description="drafts no NGRAM candidates with disaggregation: SGLang's decode server hands "
        + "a received request's first token to an NGRAM schedule in a form it cannot read"
    )
    # The platform reports tp_size 1 to SGLang, so neither server can check the other's chips.
    tp_size: Literal[1] = Field(
        description="requires tp_size=1 with disaggregation: an image page holds each chip's part "
        + "in turn, which a server split over other chips reads differently"
    )


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
