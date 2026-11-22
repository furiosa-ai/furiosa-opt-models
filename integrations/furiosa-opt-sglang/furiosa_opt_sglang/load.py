"""Translate framework load options into engine load options.

- `options` is the engine's options of the load of SGLang's checkpoint, which the platform checks
  at startup and the scheduler loads with, so the engine alone decides which chips, cache dtypes,
  quantizations and models load.
- `quantization` resolves SGLang's online weight quantization setting to an engine load method for
  target and draft checkpoints.
- `quantized_options` applies that method to engine load options.
- `bind` loads the checkpoint onto a scheduler's chips.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from pydantic import JsonValue, TypeAdapter

from furiosa_opt_sglang.platform import CACHE_DTYPES, CacheDtype, Refused

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from transformers import PretrainedConfig

    import furiosa_opt_engine as engine
    from furiosa_opt_sglang.generation.model import Model


def options(
    model_config: ModelConfig, tensor: int, dtype: CacheDtype, *, deterministic: bool
) -> engine.LoadOptions:
    """The engine's options of the load of `model_config`'s checkpoint: on `tensor` chips, with the
    cache dtype SGLang spells `dtype`, batch invariant under deterministic inference, quantized
    where SGLang quantizes it at load, as its base model when SGLang pools the hidden rows of a
    causal checkpoint."""
    from sglang.srt.configs.model_config import is_generation_model
    from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES

    import furiosa_opt_engine as engine

    hf_config = cast("PretrainedConfig", model_config.hf_config)
    loads = engine.LoadOptions().chips(tensor)
    if (cache_dtype := CACHE_DTYPES[dtype]) is not None:
        loads = loads.kv_cache(cache_dtype)
    if deterministic:
        loads = loads.batch_invariant()
    # SGLang's model config leaves the type of the method it resolves partly unknown: the operator's
    # `--quantization`, or the checkpoint's own when it declares one.
    method = cast("str | None", model_config.quantization)
    loads = quantized_options(loads, method, hf_config)
    if not model_config.is_generation and is_generation_model(hf_config.architectures or []):
        # SGLang pools the hidden rows of a causal checkpoint's base model.
        base = MODEL_MAPPING_NAMES.get(hf_config.model_type)
        if base is None:
            raise ValueError(f"{hf_config.architectures} has no base model to embed with")
        loads = loads.architecture(base)
    return loads


def quantization(
    method: str | None, config: Mapping[str, JsonValue] | PretrainedConfig
) -> Literal["fp8_per_tensor"] | None:
    """The engine's quantization of a checkpoint of `config` that SGLang loads under `method`, the
    operator's or the checkpoint's own, or none: SGLang quantizes at load only with `fp8` and only a
    checkpoint whose config declares no quantization. When SGLang's optimized channel-wise kernels
    are unavailable, its fallback uses one scale for each fused projection tensor.

    Raises:
        ValueError: The checkpoint's quantization metadata is malformed.
    """
    from sglang.srt.model_loader.checkpoint_quantization import resolve_checkpoint_quant_spec

    if method != "fp8":
        return None
    config_mapping = (
        config if isinstance(config, Mapping) else cast("Mapping[str, JsonValue]", config.to_dict())
    )
    _validate_quantization_metadata(config_mapping)
    declared = resolve_checkpoint_quant_spec(config)
    return "fp8_per_tensor" if declared is None else None


def quantized_options(
    loads: engine.LoadOptions,
    method: str | None,
    config: Mapping[str, JsonValue] | PretrainedConfig,
) -> engine.LoadOptions:
    """`loads` with the online quantization SGLang applies to `config` under `method`, if any."""
    if (name := quantization(method, config)) is not None:
        return loads.quantization(name)
    return loads


def bind(model_config: ModelConfig, first: int) -> Model:
    """The checkpoint of `model_config` on the chips from SGLang's device index `first`, with the
    options the platform checked at startup: its causal model to generate with, or its base model
    when SGLang pools hidden rows."""
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_exec, get_model, get_server_args

    import furiosa_opt_engine as engine
    from furiosa_opt_sglang.checkpoint import checkout, text
    from furiosa_opt_sglang.decoder import PES, Decoder, Location, chips

    settings, features = get_model(), get_exec().features
    # SGLang's namespace bags and model overrides do not preserve their field types.
    revision = cast("str | None", settings.revision)
    # The platform admitted only cache dtypes it spells for the engine.
    dtype = cast("CacheDtype", settings.kv_cache_dtype)
    deterministic = cast("bool", get_exec().deterministic.enable_deterministic_inference)
    overrides = cast("dict[str, JsonValue]", model_config.model_override_args)
    # SGLang's environment accessor returns Any even for string options.
    if cast("str | None", envs.SGLANG_EMBEDDINGS_SPARSE_HEAD.get()) is not None:
        raise Refused("SGLANG_EMBEDDINGS_SPARSE_HEAD", "serves no sparse embedding head")
    directory = checkout(cast("str", settings.model_path), revision)
    # The checkpoint config retains head geometry that SGLang rewrites for its own kernels.
    config = TypeAdapter(dict[str, JsonValue]).validate_json(
        (Path(directory) / "config.json").read_bytes()
    )
    config.update(overrides)
    checkpoint = Location(directory, config)
    # The operator's options keep the tensor degree the platform resolved to one scheduler.
    tensor = cast("int", get_server_args().tp_size)
    try:
        loads = options(model_config, tensor, dtype, deterministic=deterministic)
        mapped = checkpoint.mapped()
        loads.parse(mapped.source)
    except (NotImplementedError, ValueError) as error:
        raise Refused("model_path", str(error)) from None
    device = engine.Device(engine.Placement(chips(first, tensor), PES))
    # Only the memory saver releases the weights, so only it keeps the backup they resume from.
    backup = cast("bool", features.enable_memory_saver and features.enable_weights_cpu_backup)
    try:
        decoder = Decoder.load(mapped, loads, device, model_config.vocab_size, backup=backup)
    except NotImplementedError as error:
        raise Refused("model_path", str(error)) from None
    match decoder.model:
        case engine.Decoder(window=window) if model_config.context_len > window:
            raise Refused("context_length", f"requires context_length at most {window}")
        case engine.Decoder(output=engine.decoder.Output.HIDDEN) if model_config.is_generation:
            raise ValueError("the checkpoint has no output head")
        case engine.Decoder(output=engine.decoder.Output.TOKENS) if not model_config.is_generation:
            raise NotImplementedError(
                "SGLang pools hidden rows, which the calls of a causal model do not return"
            )
        case _:
            return text(decoder)


def _metadata_mapping(value: JsonValue, source: str) -> Mapping[str, JsonValue]:
    if isinstance(value, Mapping):
        return cast("Mapping[str, JsonValue]", value)
    raise ValueError(
        f"invalid checkpoint quantization metadata: {source} must be a mapping, "
        f"got {type(value).__name__}"
    )


def _validate_quantization_metadata(config: Mapping[str, JsonValue]) -> None:
    # Match SGLang's lookup precedence, validating only the metadata it will inspect.
    quantization_config = config.get("quantization_config")
    if quantization_config is not None:
        _metadata_mapping(quantization_config, "quantization_config")
        return

    text_config = config.get("text_config")
    if text_config is not None:
        text_config_mapping = _metadata_mapping(text_config, "text_config")
        quantization_config = text_config_mapping.get("quantization_config")
        if quantization_config is not None:
            _metadata_mapping(quantization_config, "text_config.quantization_config")
            return

    compression_config = config.get("compression_config")
    if compression_config is not None:
        _metadata_mapping(compression_config, "compression_config")
