# vLLM options

This page lists the vLLM options the plugin overrides or checks; other options keep vLLM's behavior.
[Serving](index.md) compares supported features, [Disaggregated KV transfer](disaggregated-kv-transfer.md) gives paired-role recipes, and [Integrating frameworks](../engine/frameworks.md) explains the shared engine data paths.

## Compatibility

The plugin targets the vLLM 0.31.0 CPU wheel and its V1 connector lifecycle. The pinned wheel and optional NIXL/Mooncake dependency bounds are declared in `integrations/furiosa-opt-vllm/pyproject.toml`; the [transfer guide](disaggregated-kv-transfer.md#compatibility) records shared transport constraints and upgrade policy. Recheck connector and runner APIs, then run the vLLM adapter suite in [Testing](../develop/testing.md#run-the-framework-tests) before changing the framework pin.

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
| `--kv-transfer-config` | vLLM's `NixlConnector`, `NixlPullConnector` or `NixlPushConnector` with `kv_buffer_device=cpu` at tensor parallel degree 1, or the plugin's standalone `FuriosaMooncakeConnector` with `mooncake_protocol=tcp` at tensor parallel degree 1; decoder attention cache only, without `--kv-offloading-size` for Mooncake; NIXL can combine with `RNGDOffloadingConnector` in a `MultiConnector` ([paired-role recipes](disaggregated-kv-transfer.md); [transfer behavior](#disaggregated-serving); [compatibility](#compatibility)) |
| `--kv-offloading-size` | Enables native offloading through `RNGDOffloadingConnector`; accepts vLLM's `CPUOffloadingSpec` and `TieringOffloadingSpec`; [KV cache offloading](#kv-cache-offloading) lists supported layouts, settings and limits |

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

## Disaggregated serving

The plugin keeps vLLM's native connector scheduling and request lifecycle while staging only active KV pages in a bounded engine-pinned host pool.

### Mooncake TCP

The connector supports the decoder attention cache at `tensor_parallel_size=1`; it cannot combine with `--kv-offloading-size`. `staging_buffer_size_mb` in `kv_connector_extra_config` sets its fixed staging capacity, which defaults to 1024 MiB. Temporary receive-pool pressure defers and retries a request through vLLM's scheduler. A receive larger than the pool follows vLLM's `kv_load_failure_policy` (local recompute by default). Temporary send-pool pressure queues a transfer while vLLM retains the request blocks; a send larger than the pool errors with its required capacity. Sustained send pressure can keep device cache blocks allocated longer.

The upstream connector pairs producer and consumer metadata through `engine_id`, `remote_engine_id`, `remote_bootstrap_addr` and `transfer_id`; the launch example is in the [paired-role guide](disaggregated-kv-transfer.md#mooncake-over-tcp).

### NIXL

`kv_buffer_device` must be `cpu`. The adapter uses vLLM's native NIXL handshake and pull/push control flow, and maps NIXL block IDs to reusable slots in a fixed-size host pool. The pool defaults to 1024 MiB and `staging_buffer_size_mb` in `kv_connector_extra_config` sizes it. It does not mirror the device cache. If a request cannot fit, vLLM recomputes it locally; the copy completes before NIXL sends blocks or reports them received. The decode server's `vllm:external_prefix_cache_hits` counts connector-hit prompt tokens, and `vllm:nixl_bytes_transferred` counts transferred bytes.

The handshake compares model, quantization and speculative method; name a request's LoRA adapter on both servers. NIXL supports data-parallel replicas and sleep mode. vLLM uses one worker for all chips while it awaits tensor-parallel ranks, so each connector serves one chip per replica. The adapter moves only attention cache pages; it refuses recurrent state, encoder memory, encoder-only instances, native device-memory connectors such as Mooncake and MoRIIO, and LMCache, which copies torch tensors directly.

### Combine NIXL with local offloading

With a direct NIXL config, `--kv-offloading-size` preserves NIXL and adds the plugin's `RNGDOffloadingConnector`. Use `MultiConnector` to set connector order explicitly:

```bash
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 vllm serve Qwen/Qwen3-0.6B --port 8200 --kv-transfer-config '{
  "kv_connector": "MultiConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"connectors": [
    {"kv_connector": "NixlConnector", "kv_role": "kv_consumer", "kv_buffer_device": "cpu"},
    {"kv_connector": "RNGDOffloadingConnector", "kv_connector_module_path": "furiosa_opt_vllm.offload",
     "kv_role": "kv_both", "kv_connector_extra_config": {"cpu_bytes_to_use": 17179869184}}]}}'
```

vLLM asks connectors in order and uses the first one reporting matched tokens; it does not merge hit lengths. It sends store hooks to each connector, which applies its own scheduling and eviction policy. Keep NIXL first for remote-first lookup, or put offloading first for local-first lookup. The NIXL entry may be `NixlPushConnector`.

The NIXL pool and vLLM's offloading region have separate ownership. A request stored by both paths can copy KV blocks twice, using extra host bandwidth and capacity; hits do not share copies. NIXL alone supports data-parallel replicas and sleep mode. With offloading, the offloading path requires data-parallel degree 1 and sleep mode off. For NIXL data-parallel replicas, route reloads to the replica that decoded the request (`X-data-parallel-rank`).

These features are not supported yet, and the plugin refuses them at startup rather than ignore them:

- Dynamic KV scales and mixed layer KV dtypes (`kv_cache_dtype_skip_layers`).
- LoRA settings past the engine's adapter terms: `--enable-lora` with a model or an online quantization whose load runs no adapter, such as Qwen3 8B or `--quantization fp8_per_tensor`, `--max-lora-rank` above 32, `--max-loras` above 2, `--lora-target-modules` that leave out an attention or MLP projection, `--lora-dtype` other than `auto` or `bfloat16`, `--fully-sharded-loras`, and LoRA on vision towers, experts or classification heads; an adapter saved as `adapter_model.bin` is refused when it loads.
- Speculative decoding with a method other than `ngram`, `suffix` or `draft_model`, with a `rejection_sample_method` other than `standard`, or with a recurrent model; suffix decoding with a batch-size schedule (`num_speculative_tokens_per_batch_size`); a draft model with `draft_sample_method` `probabilistic`, `use_heterogeneous_vocab`, or a `kv_cache_dtype` outside the values above.
- Expert parallelism.
- KV connectors not listed in the accepted `--kv-transfer-config` values above, including vLLM's own `OffloadingConnector`, native `MooncakeConnector`, MoRIIO, LMCache, arbitrary modules and unsupported encoder-cache connectors; weight offloading to the host (`--cpu-offload-gb`, `--offload-group-size`) is unsupported too.
- KV offloading through LMCache (`--kv-offloading-backend lmcache`), the simple CPU connector (`VLLM_USE_SIMPLE_KV_OFFLOAD=1`), an offloading spec other than `CPUOffloadingSpec` or stock `TieringOffloadingSpec` with filesystem secondary tiers, `kv_load_failure_policy` `recompute`, over several vLLM workers (`--pipeline-parallel-size`) or replicas (`--data-parallel-size`), or with sleep mode (`--enable-sleep-mode`).
- Quantizing at load: `--quantization` must name a quantized checkpoint's own method; a BF16 checkpoint takes only `fp8_per_tensor`, or `online` with `quantization_config.linear` `fp8_per_tensor`, without `ignore`, `targets` or an activation override.
- `--prefix-match-unit` for a hybrid model with `--mamba-cache-mode align`, because the plugin copies no cache blocks.

Watermarking and requested prompt-token scoring are supported.
