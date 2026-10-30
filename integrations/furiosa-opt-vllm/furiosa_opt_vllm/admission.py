"""The vLLM options the plugin serves, each with the requirement the platform's refusal names.

- `CONTRACTS` checks one option each, after `offload` makes `--kv-offloading-size` select the
  plugin's KV connector, `CONNECTOR`.
- `speculative` parses vLLM's speculative config into the method the plugin drafts with and its
  settings; the platform and the runner, which run in different processes, both parse it.
- `quantization` parses vLLM's quantization options into the quantization a load gives a BF16
  checkpoint, which the platform and the runner both parse too.
- `options` is the engine's options of the load of vLLM's checkpoint, which the startup check and
  the load both take, so the engine alone decides which chips, cache dtypes, models and LoRA
  adapters load.
- `verify_checkpoint` refuses a checkpoint the engine cannot load, before the engine core starts,
  whose startup error vLLM reports without its reason; `adapters` refuses LoRA settings past the
  adapter terms the engine compiles.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, assert_never, final

import torch
from typing_extensions import override
from vllm import envs

if TYPE_CHECKING:
    from vllm.config import (
        LoadConfig,
        LoRAConfig,
        ModelConfig,
        ParallelConfig,
        SpeculativeConfig,
        VllmConfig,
    )

    import furiosa_opt_engine as engine
    from furiosa_opt_vllm.additional import Placement, Section

# The engine's name of the quantization a load gives a BF16 checkpoint's projections.
Quantization: TypeAlias = Literal["fp8_per_tensor"]

# vLLM's spellings of a `--kv-cache-dtype` of one element type, under the engine's dtype name;
# `auto` keeps the engine's. Whether a model caches in that dtype is the engine's to decide.
CACHE_DTYPES: Final[Mapping[str, str | None]] = MappingProxyType(
    {
        "auto": None,
        "bfloat16": "bfloat16",
        "fp8": "float8_e4m3fn",
        "fp8_e4m3": "float8_e4m3fn",
    }
)


# The KV connector `--kv-offloading-size` selects, by the class and module vLLM's factory loads.
CONNECTOR: Final = "RNGDOffloadingConnector"
CONNECTOR_MODULE: Final = "furiosa_opt_vllm.offload"


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
        return f"furiosa-opt-vllm {self.requirement}"


@dataclass(frozen=True, slots=True)
class Contract:
    option: str
    accepts: Callable[[VllmConfig], bool]
    requirement: str


def offload(config: VllmConfig) -> None:
    """Turn `--kv-offloading-size` into the transfer config vLLM derives from it for its own
    connector, naming `CONNECTOR` instead, and take the size: vLLM derives its config after the
    platform's check, from a size it still sees."""
    from vllm.config import KVTransferConfig

    cache = config.cache_config
    if (
        cache.kv_offloading_size is None
        or cache.kv_offloading_backend != "native"
        or envs.VLLM_USE_SIMPLE_KV_OFFLOAD
    ):
        return
    transfer = config.kv_transfer_config or KVTransferConfig()
    transfer.kv_connector = CONNECTOR
    transfer.kv_connector_module_path = CONNECTOR_MODULE
    transfer.kv_role = "kv_both"
    transfer.kv_connector_extra_config["cpu_bytes_to_use"] = cache.kv_offloading_size * (1 << 30)
    config.kv_transfer_config = transfer
    cache.kv_offloading_size = None


@dataclass(frozen=True, slots=True)
class Offloading:
    """What the contracts check of an offloading transfer config through `CONNECTOR`, with its
    `kv_connector_extra_config` keys read as vLLM's offloading reads them."""

    # No spec_name but vLLM's CPUOffloadingSpec, and no spec_module_path.
    cpu_spec: bool
    canonical_layout: bool
    load_failure_policy: Literal["recompute", "fail"]


def offloading(config: VllmConfig) -> Offloading | None:
    """The offloading settings of `config` when it runs through `CONNECTOR`, or none without it."""
    transfer = config.kv_transfer_config
    if transfer is None or transfer.kv_connector != CONNECTOR:
        return None
    extra: Mapping[str, object] = transfer.kv_connector_extra_config
    return Offloading(
        extra.get("spec_name", "CPUOffloadingSpec") == "CPUOffloadingSpec"
        and "spec_module_path" not in extra,
        bool(extra.get("canonical_layout")),
        transfer.kv_load_failure_policy,
    )


def engine_cache(model: ModelConfig | None) -> bool:
    """Whether the engine holds `model`'s cache: the paged attention cache of a generating decoder
    with no recurrent state or encoder memory."""
    return model is None or (
        model.runner_type == "generate" and not (model.has_inner_state or model.is_encoder_decoder)
    )


# The projections the engine adds an adapter's updates to, in BF16, and the packed projections
# vLLM names instead of some of them.
PROJECTIONS: Final = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)
PACKED: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {"qkv_proj": ("q_proj", "k_proj", "v_proj"), "gate_up_proj": ("gate_proj", "up_proj")}
)

LORA: Final[tuple[tuple[str, Callable[[LoRAConfig], bool], str], ...]] = (
    (
        "lora_target_modules",
        lambda c: (
            c.target_modules is None
            or PROJECTIONS
            <= {name for target in c.target_modules for name in PACKED.get(target, (target,))}
        ),
        "applies LoRA to every attention and MLP projection together",
    ),
    (
        "lora_dtype",
        lambda c: c.lora_dtype in ("auto", "bfloat16", torch.bfloat16),
        "keeps LoRA weights in bfloat16",
    ),
    ("fully_sharded_loras", lambda c: not c.fully_sharded_loras, "shards no LoRA computation"),
    ("default_mm_loras", lambda c: not c.default_mm_loras, "applies no LoRA per modality"),
    (
        "enable_tower_connector_lora",
        lambda c: not c.enable_tower_connector_lora,
        "applies no LoRA to vision towers or connectors",
    ),
    (
        "enable_mixed_moe_lora_format",
        lambda c: not c.enable_mixed_moe_lora_format,
        "applies no LoRA to experts",
    ),
    (
        "enable_moe_shared_loras",
        lambda c: not c.enable_moe_shared_loras,
        "applies no LoRA to experts",
    ),
    (
        "max_lora_cls_labels",
        lambda c: c.max_lora_cls_labels is None,
        "applies no LoRA to classification heads",
    ),
)


CONTRACTS: Final = (
    Contract(
        "data_parallel_size",
        lambda c: (
            c.parallel_config.data_parallel_size == 1
            or c.model_config is None
            or not c.model_config.is_moe
        ),
        "requires data_parallel_size=1 for a mixture-of-experts model",
    ),
    Contract(
        "data_parallel_backend",
        lambda c: c.parallel_config.data_parallel_backend == "mp",
        "requires data_parallel_backend=mp",
    ),
    Contract(
        "data_parallel_size",
        lambda c: c.parallel_config.data_parallel_size == 1 or c.speculative_config is None,
        "requires data_parallel_size=1 with speculative decoding",
    ),
    Contract(
        "pipeline_parallel_size",
        lambda c: c.parallel_config.pipeline_parallel_size == 1,
        "requires pipeline_parallel_size=1",
    ),
    Contract(
        "prefill_context_parallel_size",
        lambda c: c.parallel_config.prefill_context_parallel_size == 1,
        "requires prefill_context_parallel_size=1",
    ),
    Contract(
        "decode_context_parallel_size",
        lambda c: c.parallel_config.decode_context_parallel_size == 1,
        "requires decode_context_parallel_size=1",
    ),
    Contract("nnodes", lambda c: c.parallel_config.nnodes == 1, "requires nnodes=1"),
    Contract(
        "enable_expert_parallel",
        lambda c: not c.parallel_config.enable_expert_parallel,
        "serves no expert parallelism",
    ),
    *(
        Contract(
            option,
            lambda c, accepts=accepts: c.lora_config is None or accepts(c.lora_config),
            requirement,
        )
        for option, accepts, requirement in LORA
    ),
    Contract(
        "enable_return_routed_experts",
        lambda c: not c.aux_output_config.enable_return_routed_experts,
        "returns no routed experts",
    ),
    Contract(
        "kv_offloading_backend",
        lambda c: (
            c.cache_config.kv_offloading_size is None
            or c.cache_config.kv_offloading_backend == "native"
        ),
        "offloads the KV cache with kv_offloading_backend=native",
    ),
    Contract(
        "VLLM_USE_SIMPLE_KV_OFFLOAD",
        lambda c: c.cache_config.kv_offloading_size is None or not envs.VLLM_USE_SIMPLE_KV_OFFLOAD,
        f"offloads through {CONNECTOR}, so it serves no VLLM_USE_SIMPLE_KV_OFFLOAD=1",
    ),
    Contract(
        "kv_transfer_config",
        lambda c: (
            c.kv_transfer_config is None
            or c.kv_transfer_config.kv_connector != "OffloadingConnector"
        ),
        f"offloads through {CONNECTOR}, which --kv-offloading-size selects: vLLM's "
        + "OffloadingConnector copies through CUDA streams",
    ),
    Contract(
        "kv_transfer_config",
        lambda c: (
            c.kv_transfer_config is None
            or (
                c.kv_transfer_config.kv_connector == CONNECTOR
                and c.kv_transfer_config.kv_connector_module_path == CONNECTOR_MODULE
            )
        ),
        f"serves no KV connector but {CONNECTOR}, which --kv-offloading-size selects",
    ),
    Contract(
        "kv_connector_extra_config",
        lambda c: (settings := offloading(c)) is None or settings.cpu_spec,
        "keeps offloaded blocks in host memory by vLLM's CPUOffloadingSpec, so it takes no other "
        + "spec_name or spec_module_path",
    ),
    Contract(
        "kv_connector_extra_config",
        lambda c: (settings := offloading(c)) is None or not settings.canonical_layout,
        "keeps offloaded blocks in the device's order, so it takes no canonical_layout",
    ),
    # A failed copy poisons the blocks and host rows it touched, so no request reruns on them.
    Contract(
        "kv_load_failure_policy",
        lambda c: (settings := offloading(c)) is None or settings.load_failure_policy == "fail",
        "stops the engine when an offloaded block fails to copy: kv_load_failure_policy=fail",
    ),
    # vLLM's offloading scheduler ends a job once each of its ranks' workers reports it, and one
    # worker drives every chip of the replica.
    Contract(
        "tensor_parallel_size",
        lambda c: offloading(c) is None or c.parallel_config.tensor_parallel_size == 1,
        "offloads at tensor_parallel_size=1: vLLM awaits each rank; one worker drives all chips",
    ),
    Contract(
        "pipeline_parallel_size",
        lambda c: offloading(c) is None or c.parallel_config.pipeline_parallel_size == 1,
        "offloads at pipeline_parallel_size=1: vLLM awaits each rank; one worker drives all chips",
    ),
    # The worker joins vLLM's process groups as rank 0 of 1, which data parallelism renumbers.
    Contract(
        "data_parallel_size",
        lambda c: offloading(c) is None or c.parallel_config.data_parallel_size == 1,
        "offloads at data_parallel_size=1: vLLM ranks each replica's worker among all replicas",
    ),
    Contract(
        "kv_offloading_size",
        lambda c: offloading(c) is None or engine_cache(c.model_config),
        "offloads the attention cache of a decoder, not recurrent state or encoder memory",
    ),
    Contract(
        "kv_offloading_size",
        lambda c: offloading(c) is None or not c.is_mm_encoder_only,
        "offloads the attention cache of a decoder, which an encoder-only instance does not run",
    ),
    # Sleep mode frees the cache storage and allocates new storage on waking, while the host tier
    # keeps copying the storage it was built over.
    Contract(
        "enable_sleep_mode",
        lambda c: (
            offloading(c) is None or not (c.model_config and c.model_config.enable_sleep_mode)
        ),
        "offloads the KV cache without sleep mode, which frees the cache its host tier copies",
    ),
    # A consumer needs the checkpoint's language model, which the engine does not serve; of vLLM's
    # connectors, the example one saves a producer's host rows as they are.
    Contract(
        "ec_transfer_config",
        lambda c: (
            c.ec_transfer_config is None
            or (
                c.ec_transfer_config.is_encode_only
                and c.ec_transfer_config.ec_connector == "ECExampleConnector"
            )
        ),
        "serves an encoder cache connector only as an encoder-only ECExampleConnector producer",
    ),
    Contract(
        "cpu_offload_gb",
        lambda c: c.offload_config.uva.cpu_offload_gb == 0,
        "keeps every weight on the device, so it takes no --cpu-offload-gb",
    ),
    Contract(
        "offload_group_size",
        lambda c: c.offload_config.prefetch.offload_group_size == 0,
        "keeps every weight on the device, so it takes no --offload-group-size",
    ),
    Contract(
        "load_format",
        lambda c: c.load_config.load_format == "auto",
        "finds the checkpoint's weight files itself, so it takes only --load-format auto",
    ),
    Contract(
        "dtype",
        lambda c: not c.model_config or c.model_config.dtype == torch.bfloat16,
        "uses bfloat16 activations",
    ),
    Contract(
        "cache_dtype",
        lambda c: c.cache_config.cache_dtype in CACHE_DTYPES,
        "names the engine a cache dtype only for --kv-cache-dtype auto, bfloat16, fp8 or fp8_e4m3",
    ),
    Contract(
        "kv_cache_dtype_skip_layers",
        lambda c: not c.cache_config.kv_cache_dtype_skip_layers,
        "does not support --kv-cache-dtype-skip-layers",
    ),
    # Under Model Runner V2 vLLM admits only V2's processors, which read per-request buffers that
    # vLLM allocates on a torch device; the runner's batch holds V1's on the host.
    Contract(
        "logits_processors",
        lambda c: (
            not c.model_config or not c.model_config.logits_processors or not c.use_v2_model_runner
        ),
        "serves custom logits processors only under VLLM_USE_V2_MODEL_RUNNER=0",
    ),
    # A hybrid prefix hit at a match unit finer than the block asks the runner to copy the block;
    # the block size is aligned only after this check, so every explicit unit is refused.
    Contract(
        "prefix_match_unit",
        lambda c: (
            c.cache_config.prefix_match_unit is None
            or c.cache_config.mamba_cache_mode != "align"
            or not c.model_config
            or not c.model_config.is_hybrid
        ),
        "copies no cache blocks, so hybrid prefix caching takes no --prefix-match-unit",
    ),
)


QUANTIZATION: Final = (
    "quantizes a BF16 checkpoint at load only with --quantization fp8_per_tensor: E4M3 weights "
    "under one scale per tensor, which the projections restore to BF16"
)


def quantization(config: ModelConfig, scope: str = "") -> Quantization | None:
    """The quantization vLLM's `--quantization` and `--quantization-config` ask a load of
    `config`'s BF16 checkpoint for, or none to load the checkpoint's own weights; refuses what the
    engine does not apply, naming the option within `scope`."""
    from vllm.config.quantization import ONLINE_QUANT_SHORTHAND_NAMES, resolve_quantization_config
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8StaticChannelSym,
        kFp8StaticTensorSym,
    )

    if config.model_arch_config.quantization_config is not None:
        # vLLM itself refuses a --quantization other than the checkpoint's own method.
        if config.quantization_config is not None:
            raise Refused(
                f"{scope}quantization_config",
                "quantizes no layer a quantized checkpoint leaves unquantized",
            )
        return None
    if config.quantization is None:
        return None
    if config.quantization not in ONLINE_QUANT_SHORTHAND_NAMES:
        raise Refused(f"{scope}quantization", QUANTIZATION)
    settings = resolve_quantization_config(config.quantization, config.quantization_config)
    if settings is not None and settings.ignore:
        raise Refused(f"{scope}quantization_config.ignore", "quantizes every projection at load")
    if settings is not None and settings.targets is not None:
        raise Refused(f"{scope}quantization_config.targets", "quantizes every projection at load")
    weight = None if settings is None or settings.linear is None else settings.linear.weight
    if weight == kFp8StaticChannelSym:
        # vLLM serves per-channel FP8 only with activations quantized per token (no BF16 inputs).
        raise Refused(
            f"{scope}quantization",
            (
                "keeps activations in BF16, which vLLM's fp8_per_channel quantizes per token; "
                "use --quantization fp8_per_tensor"
            ),
        )
    if settings is None or settings.linear is None or weight != kFp8StaticTensorSym:
        raise Refused(f"{scope}quantization", QUANTIZATION)
    # vLLM picks each online method's activation quantization itself; an override asks for FP8 ones.
    if settings.linear.activation is not None:
        raise Refused(
            f"{scope}quantization_config.linear.activation",
            "takes no activation quantization; the projections take BF16 activations",
        )
    if config.is_moe and settings.moe is not None:
        raise Refused(f"{scope}quantization", "quantizes no routed experts at load")
    return "fp8_per_tensor"


class Method(StrEnum):
    """A speculative method the plugin serves, under its vLLM name."""

    NGRAM = "ngram"
    SUFFIX = "suffix"
    MODEL = "draft_model"

    @classmethod
    def parse(cls, name: str | None) -> Method:
        """The method `name` names, refused unless the plugin serves it."""
        try:
            return cls(name)
        except ValueError:
            raise Refused(
                "speculative_config.method",
                "supports only the ngram, suffix and draft_model speculative methods",
            ) from None


@dataclass(frozen=True, slots=True)
class Draft:
    """A draft model to load beside the target: its checkpoint, parallelism and cache dtype, where
    it loads, and the context its cache must hold."""

    model: ModelConfig
    parallel: ParallelConfig
    # By default the draft model loads on the chips the target leaves free.
    placement: Placement | None
    download_dir: str | None
    # The engine's dtype name; `None` keeps the engine's.
    cache_dtype: str | None
    # The target's context length, to which drafting extends a request.
    max_model_len: int

    @classmethod
    def parse(cls, config: VllmConfig, settings: SpeculativeConfig, section: Section) -> Draft:
        """The draft model of `config`'s speculative `settings`, whose load config and cache dtype
        default to the target's; refuses a setting the drafter would not honor, naming its
        option."""
        if settings.draft_sample_method != "greedy":
            raise Refused(
                "speculative_config.draft_sample_method",
                "speculative_config.draft_sample_method requires greedy drafting",
            )
        if settings.use_heterogeneous_vocab:
            raise Refused(
                "speculative_config.use_heterogeneous_vocab",
                "takes no use_heterogeneous_vocab; a draft model needs the target vocabulary",
            )
        # vLLM's model loader imports the platform, which imports this module.
        from vllm.model_executor.model_loader.utils import get_draft_load_config

        load = get_draft_load_config(config)
        if load.load_format != "auto":
            raise Refused(
                "speculative_config.draft_load_config",
                "speculative_config.draft_load_config requires load_format auto",
            )
        dtype = settings.kv_cache_dtype or config.cache_config.cache_dtype
        if dtype not in CACHE_DTYPES:
            raise Refused(
                "speculative_config.kv_cache_dtype",
                "names the engine a draft cache dtype only for auto, bfloat16, fp8 or fp8_e4m3",
            )
        if quantization(settings.draft_model_config, "speculative_config.") is not None:
            raise Refused("speculative_config.quantization", "quantizes no draft model at load")
        # The engine loads once vLLM checks a configuration, not when vLLM imports the platform.
        import furiosa_opt_engine as engine

        cache_dtype = CACHE_DTYPES[dtype]
        loads = engine.LoadOptions().chips(settings.draft_parallel_config.tensor_parallel_size)
        verify_checkpoint(
            settings.draft_model_config,
            load,
            "speculative_config.model",
            loads if cache_dtype is None else loads.kv_cache(cache_dtype),
        )
        return cls(
            settings.draft_model_config,
            settings.draft_parallel_config,
            section.draft_placement,
            load.download_dir,
            cache_dtype,
            config.model_config.max_model_len,
        )


def options(config: VllmConfig) -> engine.LoadOptions:
    """The engine's options of the load of `config`'s checkpoint: its media encoder alone on an
    encoder-only instance, else the model on vLLM's tensor degree of chips with vLLM's cache dtype
    and batch invariance, as its base model when vLLM pools the hidden rows of a causal model, and
    for calls that run LoRA adapters when vLLM enables LoRA; under the quantization vLLM asks of a
    BF16 checkpoint. Refuses a base model the engine lacks."""
    from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES
    from vllm import envs
    from vllm.model_executor.models import ModelRegistry

    # The engine loads once vLLM checks a configuration, not when vLLM imports the platform.
    import furiosa_opt_engine as engine

    model = config.model_config
    loads = engine.LoadOptions()
    if config.is_mm_encoder_only:
        # An encoder cache producer runs no language model, so no cache, chip split or tiling.
        loads = loads.encoder()
    else:
        loads = loads.chips(config.parallel_config.tensor_parallel_size)
        if (dtype := CACHE_DTYPES[config.cache_config.cache_dtype]) is not None:
            loads = loads.kv_cache(dtype)
        if envs.VLLM_BATCH_INVARIANT:
            loads = loads.batch_invariant()
        if model.convert_type == "embed" and not ModelRegistry.is_pooling_model(
            model.architectures, model
        ):
            base = MODEL_MAPPING_NAMES.get(model.hf_text_config.model_type)
            if base is None:
                raise Refused(
                    "convert", f"finds no base model of {model.hf_text_config.model_type} to embed"
                )
            try:
                loads = loads.architecture(base)
            except (NotImplementedError, ValueError) as error:
                raise Refused("convert", str(error)) from None
        if config.lora_config is not None:
            loads = loads.adapters()
    quantized = quantization(model)
    return loads if quantized is None else loads.quantization(quantized)


def verify_checkpoint(
    model: ModelConfig, load: LoadConfig, option: str, options: engine.LoadOptions
) -> None:
    """Refuse under `option` the checkpoint of `model`, where `load` finds it, if the engine cannot
    load it with `options`: an architecture the engine does not serve, a configuration that no
    listed model matches, or a choice of `options` its model refuses. The config alone decides
    first, so a refused checkpoint downloads no weight."""
    from furiosa_opt_vllm import checkpoint

    try:
        options.parse(checkpoint.configured(model, load.download_dir))
        options.parse(checkpoint.located(model, load.download_dir).mapped())
    except (NotImplementedError, ValueError) as error:
        raise Refused(option, str(error)) from None


def adapters(config: VllmConfig) -> None:
    """Refuse LoRA settings past the adapter terms the engine compiles: a rank above theirs, or
    more adapters in a batch than a call runs."""
    lora = config.lora_config
    if lora is None:
        return
    # The engine loads once vLLM checks a configuration, not when vLLM imports the platform.
    import furiosa_opt_engine as engine

    spec = engine.decoder.AdapterSpec
    if lora.max_lora_rank > spec.RANK:
        raise Refused("max_lora_rank", f"supports max_lora_rank up to {spec.RANK}")
    if lora.max_loras > spec.SLOTS:
        raise Refused("max_loras", f"supports max_loras up to {spec.SLOTS}")


# vLLM's ngram or suffix proposer, or a draft model.
Speculative: TypeAlias = Literal[Method.NGRAM, Method.SUFFIX] | Draft


def speculative(config: VllmConfig, section: Section) -> Speculative | None:
    """The speculative method `config` asks for, with the settings it drafts with, or `None`
    without speculation; refuses a setting the plugin would not honor, naming its option."""
    settings = config.speculative_config
    if settings is None:
        return None
    method = Method.parse(settings.method)
    if settings.rejection_sample_method != "standard":
        raise Refused(
            "speculative_config.rejection_sample_method",
            "requires speculative_config.rejection_sample_method=standard",
        )
    model = config.model_config
    if model and (
        model.is_hybrid or model.hf_text_config.model_type in ("mamba", "mamba2", "bamba")
    ):
        raise Refused(
            "speculative_config", "cannot roll back recurrent state for speculative decoding"
        )
    match method:
        case Method.NGRAM:
            return method
        case Method.SUFFIX:
            # vLLM's suffix proposer drafts its configured count, which the schedule would shrink.
            if settings.uses_dynamic_speculative_decoding():
                raise Refused(
                    "speculative_config.num_speculative_tokens_per_batch_size",
                    "takes no num_speculative_tokens_per_batch_size for suffix decoding",
                )
            return method
        case Method.MODEL:
            return Draft.parse(config, settings, section)
        case _:
            assert_never(method)
