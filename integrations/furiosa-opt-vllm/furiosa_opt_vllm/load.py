"""Load the decoder vLLM's model config names, or its media encoder alone on an encoder-only
instance, and bind it to the runner's serving path."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.platforms import current_platform

import furiosa_opt_engine as engine
from furiosa_opt_vllm import checkpoint, lora, selection
from furiosa_opt_vllm.admission import options
from furiosa_opt_vllm.decoder import PES, Decoder
from furiosa_opt_vllm.multimodal import Media
from furiosa_opt_vllm.runner import Labels, Vocabulary

if TYPE_CHECKING:
    from furiosa_opt_vllm.runner import Runner


def bind(runner: Runner, *, load_dummy_weights: bool = False) -> None:
    if load_dummy_weights:
        raise ValueError("the backend loads checkpoint parameters")
    config = runner.vllm_config
    match runner.classifier:
        case Labels():
            raise ValueError(
                f"{config.model_config.architectures} has no score head to classify with"
            )
        case Vocabulary():
            raise NotImplementedError(
                "classifier_from_token reads vocabulary scores, which the engine does not return"
            )
        case None:
            pass
    if config.is_mm_encoder_only:
        # An encoder cache producer runs no language model: rank 0's chip holds the encoder.
        device = engine.Device(
            engine.Placement([current_platform.device_id_to_physical_device_id(0)], PES)
        )
        located = checkpoint.located(config.model_config, config.load_config.download_dir)
        match options(config).load(located.mapped(), device):
            case engine.Encoder() as encoder:
                runner.model = Media(encoder, device)
            case loaded:
                raise NotImplementedError(f"{type(loaded).__name__} is no media encoder")
        runner.selection = selection.choose(config, runner.speculative, None, runner.requests)
        return
    decoder = Decoder.load(config)
    model = decoder.model
    if model is None:
        raise RuntimeError("a load returns its model")
    match model.output:
        case engine.decoder.Output.HIDDEN if runner.pooler is None:
            raise ValueError("the checkpoint has no output head")
        case engine.decoder.Output.TOKENS if runner.pooler is not None:
            raise NotImplementedError(
                "vLLM pools hidden rows, which the calls of a causal model do not return"
            )
        case _:
            pass
    cache = model.cache
    if any(pool.block != config.cache_config.block_size for pool in cache.pools):
        raise ValueError(f"the compiled cache uses block_size={cache.pools[0].block}")
    if config.lora_config is not None:
        spec = model.adapters
        assert spec is not None, "a load for adapters selects a model that runs them"
        runner.adapters = lora.Adapters(config.lora_config, spec, config.model_config.max_model_len)
    runner.model = decoder
    runner.selection = selection.choose(config, runner.speculative, cache, runner.requests)
