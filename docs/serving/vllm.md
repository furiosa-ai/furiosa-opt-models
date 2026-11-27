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

vLLM keeps block selection, admission and eviction; the plugin implements its native CPU offloading contract through `RNGDOffloadingConnector`. Without a secondary tier, `--kv-offloading-size` sets pinned host-tier capacity in GiB. The [serving task guide](index.md#offload-the-kv-cache-to-host-memory) has the filesystem launch command.

The filesystem tier uses vLLM's `TieringOffloadingSpec`. Its native `secondary_tiers` field belongs in `kv_connector_extra_config`; Furiosa supplies the tier manager over engine-pinned slots.

Keep `blocks_per_chunk` and `eviction_policy` in vLLM's `kv_connector_extra_config`.

### Coalesce worker-step loads

The Furiosa-only `coalescing` option defaults to `false`; set it to `true` beside `secondary_tiers` in `kv_connector_extra_config`. vLLM still tracks each load job separately while the engine combines that worker step's loads into one staging transfer. The [shared transfer guide](../engine/frameworks.md#optional-coalescing) documents staging cost and memory growth.

### Filesystem tier

The filesystem tier has no capacity limit or automatic stale-directory cleanup; operators remove old directories. Its file layout and direct-I/O rules are in the [shared filesystem guide](../engine/frameworks.md#filesystem-layout-and-io).

### Page layout

`canonical_layout` is accepted only when its page mapping is byte-identical to the engine layout and each vLLM worker maps to one engine chip. Startup rejects other mappings instead of converting page fragments. Without `canonical_layout`, each worker uses its local page layout. Offloading supports the model's compiled tensor-parallel chip counts, transfers each chip's page in rank order and reports completion only after the full transfer.

### Cache groups and parallelism

Offloading supports causal full-attention and sliding-window `Paged` cache groups; it does not support recurrent `State` or encoder `Cross` groups. It requires `CPUOffloadingSpec`, a compiled tensor-parallel chip count, pipeline and data parallel degrees of one, `kv_load_failure_policy` `fail`, and sleep mode off. A deployment remains single-replica because vLLM ranks each replica's worker among all data-parallel workers.

### Failures and metrics

The adapter propagates transfer errors through vLLM's native offload worker; the [shared transfer guide](../engine/frameworks.md#completion-and-failures) documents page reuse after failure. The `vllm:kv_offload_store_bytes` and `vllm:kv_offload_load_bytes` metrics count bytes moved in each direction.

## Shared Mooncake Store prefix cache

The native Store connector lets a server reuse prompt KV written by another server. Install the optional `mooncake` extra as described in [Installation](../getting-started/installation.md#install-optional-mooncake-storage-support), configure the Mooncake Store client, and select the Furiosa adapter for vLLM's native `MooncakeStoreConnector`:

```bash
MOONCAKE_CONFIG_PATH=/etc/mooncake/vllm.json \
vllm serve Qwen/Qwen3-0.6B \
  --kv-transfer-config '{"kv_connector":"FuriosaMooncakeStoreConnector","kv_connector_module_path":"furiosa_opt_vllm.mooncake_store","kv_role":"kv_both","kv_connector_extra_config":{"cache_prefix":"production","staging_buffer_size_mb":1024}}'
```

### Compatibility

The adapter must match the vLLM version pinned in `integrations/furiosa-opt-vllm/pyproject.toml`;
that manifest's optional `mooncake` extra owns the Mooncake version range. The [shared Mooncake
compatibility contract](../engine/frameworks.md#shared-mooncake-storage-compatibility) describes
upgrade validation.

Use the same `cache_prefix` and model identifier on instances that share KV. The adapter adds an engine page-format, cache-spec, dtype, layout and parallelism namespace. vLLM's Store key does not fingerprint checkpoint revisions; change `cache_prefix` when weights change under the same model identifier. Use `kv_role` `kv_producer` or `kv_consumer` when an instance should only write or read.

The adapter registers one bounded staging slab and passes page buffers to Mooncake's native multi-buffer Store API. `staging_buffer_size_mb` controls that pinned slab and defaults to 1024 MiB. If the Mooncake Store service has its own disk tier, configure it on the Mooncake deployment; the vLLM connector uses that tier through the same Store API. This option is separate from the local filesystem `secondary_tiers` configuration above. The Mooncake disk path has not been performance-compared with the local filesystem tier.

The native Store connector requires asynchronous loads and supports Paged full and sliding-window attention. It does not support recurrent `State` or encoder `Cross` cache groups. It uses one Store send worker, matching vLLM's native sender; GET worker count follows `VLLM_MOONCAKE_LOAD_RECV_THREADS`. Do not combine it with `--kv-offloading-size`; vLLM selects one KV connector for a worker.

## Features not supported yet

The plugin refuses these framework features at startup rather than ignore them:

- Dynamic KV scales and mixed layer KV dtypes (`kv_cache_dtype_skip_layers`).
- LoRA settings past the engine's adapter terms: `--enable-lora` with a model or an online quantization whose load runs no adapter, such as Qwen3 8B or `--quantization fp8_per_tensor`, `--max-lora-rank` above 32, `--max-loras` above 2, `--lora-target-modules` that leave out an attention or MLP projection, `--lora-dtype` other than `auto` or `bfloat16`, `--fully-sharded-loras`, and LoRA on vision towers, experts or classification heads; an adapter saved as `adapter_model.bin` is refused when it loads.
- Speculative decoding with a method other than `ngram`, `suffix` or `draft_model`, with a `rejection_sample_method` other than `standard`, or with a recurrent model; suffix decoding with a batch-size schedule (`num_speculative_tokens_per_batch_size`); a draft model with `draft_sample_method` `probabilistic`, `use_heterogeneous_vocab`, or a `kv_cache_dtype` outside the values above.
- Expert parallelism.
- Direct KV connectors selected through `--kv-transfer-config` other than `RNGDOffloadingConnector` and `FuriosaMooncakeStoreConnector`, including vLLM's own `OffloadingConnector`. The `secondary_tiers` setting above supports local filesystem storage within native offloading; use the direct Mooncake Store connector above for shared prefix storage.
- Encoder cache connectors other than the encoder-only one above (`--ec-transfer-config`).
- Weight offloading to the host (`--cpu-offload-gb`, `--offload-group-size`).
- KV offloading through LMCache (`--kv-offloading-backend lmcache`), the simple CPU connector (`VLLM_USE_SIMPLE_KV_OFFLOAD=1`), a custom offloading spec, `kv_load_failure_policy` `recompute`, over several vLLM workers (`--pipeline-parallel-size`) or replicas (`--data-parallel-size`), or with sleep mode (`--enable-sleep-mode`).
- Quantizing at load: `--quantization` must name a quantized checkpoint's own method; a BF16 checkpoint takes only `fp8_per_tensor`, or `online` with `quantization_config.linear` `fp8_per_tensor`, without `ignore`, `targets` or an activation override.
- `--prefix-match-unit` for a hybrid model with `--mamba-cache-mode align`, because the plugin copies no cache blocks.

## Offline generation

The offline `vllm.LLM.generate` API supports watermarking and requested prompt-token scoring; this does not change the serving API's greedy-only request contract.
