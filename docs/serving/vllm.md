# vLLM options

This page lists the vLLM options the plugin overrides or checks; other options keep vLLM's behavior.
[Serving](index.md) shows each serving task, and [Integrating frameworks](../engine/frameworks.md#return-the-rows-the-framework-needs) explains which steps take their tokens from the device.

## Options

The platform accepts these values:

| Option | Accepted values |
| --- | --- |
| `--dtype` | `auto` (for BF16 checkpoints), `bfloat16` |
| `--block-size` | The model's compiled cache block; a refused size names it |
| `--max-model-len` | At most the decoder's compiled window, which startup names otherwise |
| `--kv-cache-dtype` | `auto`, `bfloat16`; startup refuses a dtype the model does not cache in with the engine's reason ([FP8 KV cache](index.md#features-not-supported-yet) is not supported yet) |
| `--tensor-parallel-size` | 1, or a chip count the model is compiled to split over; a refused count names the accepted ones |
| `--convert classify` | A checkpoint with a score head, or `classifier_from_token` over its vocabulary head |
| `head_dtype` in `--hf-overrides` | `float32`, the pooling default, for a score head |
| `rope_parameters` and `max_position_embeddings` in `--hf-overrides` | A `rope_type` the engine computes, with `rope_theta`, within the compiled window ([Extend the context](index.md#extend-the-context-past-the-trained-length)) |
| Data, prefill context and decode context parallel degrees, `--nnodes` | 1 |
| `--logits-processors` | Processors of vLLM's V1 interface, under `VLLM_USE_V2_MODEL_RUNNER=0` |
| `VLLM_BATCH_INVARIANT` | `0`, or `1` to load the model [batch invariant](index.md#keep-outputs-independent-of-the-batch) |
| `--ec-transfer-config` | An encoder-only instance (`ec_role` `ec_producer`) through `ECExampleConnector`, for a checkpoint whose media encoder the engine loads alone |
| `--kv-offloading-size` | Native offloading through `RNGDOffloadingConnector` ([Offload the KV cache to host memory](index.md#offload-the-kv-cache-to-host-memory)), at tensor, pipeline and data parallel degrees 1, without sleep mode, for a model without recurrent state or encoder memory and not on an encoder-only instance; `--kv-transfer-config` beside it adds `kv_connector_extra_config` options of vLLM's CPU offloading |

These features are not supported yet, and the plugin refuses them at startup rather than ignore them:

- Dynamic KV scales and mixed layer KV dtypes (`kv_cache_dtype_skip_layers`).
- LoRA settings past the engine's adapter terms: `--enable-lora` with a model or an online quantization whose load runs no adapter, such as Qwen3 8B or `--quantization fp8_per_tensor`, `--max-lora-rank` above 32, `--max-loras` above 2, `--lora-target-modules` that leave out an attention or MLP projection, `--lora-dtype` other than `auto` or `bfloat16`, `--fully-sharded-loras`, and LoRA on vision towers, experts or classification heads; an adapter saved as `adapter_model.bin` is refused when it loads.
- Speculative decoding with a method other than `ngram`, `suffix` or `draft_model`, with a `rejection_sample_method` other than `standard`, or with a recurrent model; suffix decoding with a batch-size schedule (`num_speculative_tokens_per_batch_size`); a draft model with `draft_sample_method` `probabilistic`, `use_heterogeneous_vocab`, or a `kv_cache_dtype` outside the values above.
- Expert parallelism.
- KV connectors other than `RNGDOffloadingConnector` (`--kv-transfer-config`), among them vLLM's own `OffloadingConnector`, an encoder cache connector other than the encoder-only one above (`--ec-transfer-config`), and weight offloading to the host (`--cpu-offload-gb`, `--offload-group-size`).
- KV offloading through LMCache (`--kv-offloading-backend lmcache`), the simple CPU connector (`VLLM_USE_SIMPLE_KV_OFFLOAD=1`), an offloading spec other than `CPUOffloadingSpec` or `canonical_layout` in `kv_connector_extra_config`, `kv_load_failure_policy` `recompute`, over several chips (`--tensor-parallel-size`, `--pipeline-parallel-size`) or replicas (`--data-parallel-size`), or with sleep mode (`--enable-sleep-mode`).
- Quantizing at load: `--quantization` must name a quantized checkpoint's own method; a BF16 checkpoint takes only `fp8_per_tensor`, or `online` with `quantization_config.linear` `fp8_per_tensor`, without `ignore`, `targets` or an activation override.
- `--prefix-match-unit` for a hybrid model with `--mamba-cache-mode align`, because the plugin copies no cache blocks.

Watermarking and requested prompt-token scoring are supported.
