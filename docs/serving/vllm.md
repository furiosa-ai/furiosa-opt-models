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
| `--kv-offloading-size` | Enables native offloading through `RNGDOffloadingConnector`; [KV cache offloading](#kv-cache-offloading) lists supported layouts, settings and limits |

## KV cache offloading

### Configure the host tier

Set `--kv-offloading-size` to the pinned host-tier capacity in GiB. The plugin uses vLLM's CPU offloading contract through `RNGDOffloadingConnector`. See the [serving task guide](index.md#offload-the-kv-cache-to-host-memory) for a filesystem launch command.

The filesystem tier uses vLLM's `TieringOffloadingSpec`. Its native `secondary_tiers` field belongs in `kv_connector_extra_config`; Furiosa supplies the tier manager over engine-pinned slots.

Keep `blocks_per_chunk` and `eviction_policy` in vLLM's `kv_connector_extra_config`.

### Coalesce worker-step loads

The Furiosa-only `coalescing` option defaults to `false`; set it to `true` beside `secondary_tiers` in `kv_connector_extra_config`. vLLM still tracks each load job separately while the engine combines that worker step's loads into one staging transfer. The [shared transfer guide](../engine/frameworks.md#optional-coalescing) documents staging cost and memory growth.

### Filesystem tier

The filesystem tier writes vLLM's native layer-major chunks, including trailing alignment padding. It leaves capacity and stale-directory cleanup to the operator. See [Aligned filesystem I/O](../engine/frameworks.md#aligned-filesystem-io) for aligned I/O rules.

### Page layout

`canonical_layout` requires a byte-identical engine page and one engine chip per vLLM worker. A prefix key can match while two tensor-parallel layouts put the same attention heads in different byte ranges. To keep direct DMA, startup checks the mapping and rejects any case that would need page conversion. With `canonical_layout` disabled, the adapter uses the worker's local layout and copies chip pages in rank order.

### Cache groups and parallelism

Offloading supports causal full-attention and sliding-window `Paged` cache groups; it does not support recurrent `State` or encoder `Cross` groups. It requires `CPUOffloadingSpec`, a compiled tensor-parallel chip count, pipeline and data parallel degrees of one, `kv_load_failure_policy` `fail`, and sleep mode off. A deployment remains single-replica because vLLM ranks each replica's worker among all data-parallel workers.

### Failures and metrics

The adapter propagates transfer errors through vLLM's native offload worker; the [shared transfer guide](../engine/frameworks.md#completion-and-failures) documents page reuse after failure. The `vllm:kv_offload_store_bytes` and `vllm:kv_offload_load_bytes` metrics count bytes moved in each direction.

These features are not supported yet, and the plugin refuses them at startup rather than ignore them:

- Dynamic KV scales and mixed layer KV dtypes (`kv_cache_dtype_skip_layers`).
- LoRA settings past the engine's adapter terms: `--enable-lora` with a model or an online quantization whose load runs no adapter, such as Qwen3 8B or `--quantization fp8_per_tensor`, `--max-lora-rank` above 32, `--max-loras` above 2, `--lora-target-modules` that leave out an attention or MLP projection, `--lora-dtype` other than `auto` or `bfloat16`, `--fully-sharded-loras`, and LoRA on vision towers, experts or classification heads; an adapter saved as `adapter_model.bin` is refused when it loads.
- Speculative decoding with a method other than `ngram`, `suffix` or `draft_model`, with a `rejection_sample_method` other than `standard`, or with a recurrent model; suffix decoding with a batch-size schedule (`num_speculative_tokens_per_batch_size`); a draft model with `draft_sample_method` `probabilistic`, `use_heterogeneous_vocab`, or a `kv_cache_dtype` outside the values above.
- Expert parallelism.
- KV connectors other than `RNGDOffloadingConnector` (`--kv-transfer-config`), among them vLLM's own `OffloadingConnector`, an encoder cache connector other than the encoder-only one above (`--ec-transfer-config`), and weight offloading to the host (`--cpu-offload-gb`, `--offload-group-size`).
- Alternative KV-offload paths: LMCache (`--kv-offloading-backend lmcache`) and the simple CPU connector (`VLLM_USE_SIMPLE_KV_OFFLOAD=1`).
- Quantizing at load: `--quantization` must name a quantized checkpoint's own method; a BF16 checkpoint takes only `fp8_per_tensor`, or `online` with `quantization_config.linear` `fp8_per_tensor`, without `ignore`, `targets` or an activation override.
- `--prefix-match-unit` for a hybrid model with `--mamba-cache-mode align`, because the plugin copies no cache blocks.

Watermarking and requested prompt-token scoring are supported.
