# SGLang options

This page lists the SGLang options the plugin overrides or checks; other options keep SGLang's behavior.
[Serving](index.md) compares supported features, [Disaggregated KV transfer](disaggregated-kv-transfer.md) gives paired-role recipes, and [Integrating frameworks](../engine/frameworks.md) explains the shared engine data paths.

## Compatibility

The plugin targets SGLang 0.5.22's native manager, receiver and staging APIs. The pinned framework and optional NIXL/Mooncake dependency bounds are declared in `integrations/furiosa-opt-sglang/pyproject.toml`; the [transfer guide](disaggregated-kv-transfer.md#compatibility) owns shared transport constraints and upgrade policy. Recheck upstream lifecycle APIs, then run the SGLang adapter suite in [Testing](../develop/testing.md#run-the-framework-tests) before changing the framework pin.

## Options

The platform accepts these values:

| Option | Accepted values |
| --- | --- |
| `--kv-cache-dtype` | `auto`, `bf16`, `bfloat16`; startup refuses a dtype the model does not cache in with the engine's reason ([FP8 KV cache](index.md#features-not-supported-yet) is not supported yet) |
| `--dtype` | `auto`, `bfloat16` |
| `--load-format` | `auto`, `safetensors` |
| `--device` | Unset or `cpu` (framework tensors stay on the host; the model runs on RNGD) |
| `--attention-backend`, `--prefill-attention-backend`, `--decode-attention-backend` | Unset or `torch_native` (the engine runs attention) |
| `startup_weight_load_mode` | `serial` |
| `disaggregation_mode` | `null`, `prefill` or `decode`; uses `--disaggregation-transfer-backend mooncake_tcp` (Mooncake TCP) or `nixl`, with `SGLANG_DISAGG_STAGING_BUFFER=1` for either backend ([paired-role recipes](disaggregated-kv-transfer.md); [compatibility](#compatibility)) |
| Attention data, expert, context parallelism (`attn_dp_size`, `ep_size`, `dcp_size`, `attn_cp_size`) and `nnodes` | 1 |
| `--tp-size` | A chip count the model is compiled for; one scheduler drives every chip, so `/server_info` reports `tp_size` 1 |
| `--dp-size` | Any, or 1 with speculative decoding; replica `i` runs on the `--tp-size` chips from position `i` × `--tp-size` of `FURIOSA_VISIBLE_CHIPS`, or of every installed chip when it is unset |
| `--gpu-id-step` | Unset; the platform spaces the replicas by the tensor degree |
| `--context-length` | At most the decoder's compiled window, which startup names otherwise |
| `rope_parameters` or `rope_scaling` and `max_position_embeddings` in `--json-model-override-args` | A `rope_type` the engine computes, within the compiled window ([Extend the context](index.md#extend-the-context-past-the-trained-length)) |
| `--mem-fraction-static` | Unset defaults to 0.9, leaving the headroom a model split over chips needs |
| `--enable-deterministic-inference` | Off, or on to load the model [batch invariant](index.md#keep-outputs-independent-of-the-batch) with the radix cache on |
| `--enable-hierarchical-cache` | A model without recurrent state, with the host tier sized by `--hicache-ratio` or `--hicache-size`; see [Hierarchical cache](#hierarchical-cache) for required modes and storage options |

## Hierarchical cache

### Configure the host tier

The plugin connects SGLang's HiCache page selection and lifecycle to engine-pinned host slots. Choose a host-tier size with `--hicache-ratio` or `--hicache-size`; the [serving task guide](index.md#offload-the-kv-cache-to-host-memory) has the filesystem launch command.

The filesystem tier uses SGLang's `file` option and `FuriosaHiCacheFile`. Set `--hicache-host-memory-mode cache`, `--hicache-io-backend kernel` and `--hicache-mem-layout page_first`. The storage backend must be unset or `file`; other backends need their own host-buffer integration. Automatic sizing from free host memory is available only for pools SGLang allocates itself.

The adapter loads through SGLang's `dynamic` storage-backend factory and preserves SGLang's cache keys and file-tier lifecycle. The [shared filesystem guide](../engine/frameworks.md#filesystem-layout-and-io) defines the file format and I/O alignment behavior.

### Coalesce page restores

The Furiosa-only `coalescing` option defaults to `false`; set it to `true` beside `file_storage_path` in `--hicache-storage-backend-extra-config` to stage each selected all-layer page batch.

The adapter submits one all-layer engine copy for the selected page batch when its first per-layer callback arrives. Separate restore calls remain separate.

See [Optional coalescing](../engine/frameworks.md#optional-coalescing) for the shared costs and mechanics.

### Restore timing

SGLang normally makes a restored layer available as soon as it is ready. This adapter reports all layer-ready callbacks only after the full engine copy finishes, so early layers may wait for later layers and lose compute/transfer overlap. Splitting the copy per layer could restore that overlap, but would add runtime submissions and change the all-layer DMA schedule.

### Supported cache pages

The adapter transfers only engine-owned target-model attention KV pages, so recurrent state and MTP draft pools are unsupported.

### Memory saver

The memory saver cannot run with HiCache because it frees cache storage that the host tier still uses.

### Failures

Failed copies surface at SGLang's next HiCache completion check after the engine copy drains.

### Observe cache hits

A `/generate` response reports reused tokens by tier in `meta_info.cached_tokens_details`.

These features are not supported yet, and the plugin refuses them at startup rather than ignore them:

- LoRA and speculative algorithms other than NGRAM and STANDALONE.
- Speculative decoding with more than one candidate a position, adaptive speculation or an external n-gram corpus.
- Watermarking (`--enable-watermark`) and strict thinking (`--enable-strict-thinking`), which change the scores of each step.
- Quantization and quantization parameter files.
- The memory saver (`--enable-memory-saver`) with speculative decoding, whose worker and draft model keep device memory a release does not reach yet.
- Per-layer embedding offload (`ple_offload_embedding`).
- Returning hidden states, routed experts or indexer top-k.
- Diffusion LM algorithms (`dllm_algorithm`) and `prefill_only_disable_kv_cache`.
- Encoder disaggregation (`--encoder-only`, `--language-only`).
- With prefill-decode disaggregation: a backend other than `mooncake_tcp` or `nixl` with `SGLANG_DISAGG_STAGING_BUFFER=1` (`fake` moves no KV), a model split over chips, host receive (`--disaggregation-decode-host-receive-threshold` below 1: SGLang receives into a host pool only beside a token pool of separate key and value buffers), decode-side cache offload, which needs a storage backend, KV checksums, which SGLang computes with a CUDA kernel, a `host_pool` retraction backup, which SGLang backs up only from an MHA token pool, and speculative decoding with NIXL.

The platform disables CUDA graphs and piecewise CUDA graphs, and sets the scheduling options by model kind:

| Model | Page size | Overlap schedule | Mixed chunks |
| --- | --- | --- | --- |
| Non-recurrent generation | 16 | On, unless you disable it | On |
| Recurrent-state generation | 1 | Off | Off |
| Embedding | 16 | Off | Off |
| Speculative decoding | 16 | Off | Off |

A model that keeps recurrent state also requires `mamba_ssm_dtype` unset or `float32` and `mamba_radix_cache_strategy` `auto` or `no_buffer`, and does not support `enable_int8_mamba_checkpoint`, `enable_linear_replayssm` or `enable_linear_replayssm_spec` yet.
[Integrating frameworks](../engine/frameworks.md#sglang-specifics) explains the overlap schedule and mixed chunks.

## Disaggregated serving

The adapter preserves SGLang's bounded staging ring and transfer-completion lifecycle for NIXL and Mooncake TCP. The prefill side copies each chunk's selected pages into its registered staging region, and decode copies received pages into the device cache before the scheduler publishes or runs the request. Cached prefixes and pages restored from hierarchical cache use the same path. The prefill side pins one chunk per native transfer worker; `SGLANG_DISAGG_STAGING_POOL_SIZE_MB` sets the decode ring size (4096 MiB by default), and SGLang's watermarks control backpressure.

Both roles require the same model, engine chip topology (`--tp-size`) and cache page geometry because the adapter transfers raw engine pages without converting attention-TP layouts. The engine owns the chip topology while SGLang's scheduler metadata remains 1; the adapter preserves the peer's actual TP metadata during chunk transfer. The common pinned-host staging path and its per-chunk DMA cost are described in [Integrating frameworks](../engine/frameworks.md#disaggregated-kv-transfer).

See the [Disaggregated KV transfer guide](disaggregated-kv-transfer.md) for NIXL and Mooncake TCP launch recipes. Decode host receive is not enabled; received pages land in the engine cache after transfer completion. Filesystem HiCache remains a separate tier with separate host buffers, so a page may also be copied for the L3 tier when both are enabled. HiCache does not reuse the transfer ring's transient staging regions.

SGLang's early send of a cached prefix records a CUDA event. Prefill roles and servers started with `--enable-pd-role-switch` use non-overlap scheduling. Retraction with `/pause_generation` can receive pages again; memory retraction restores from the decode server's host backup. If the post-transfer device-cache copy fails, the adapter uses SGLang's native failed-transfer path so the request is aborted and its cache rows are released. The memory saver, hierarchical cache on both roles, decode-side radix cache, data-parallel replicas, STANDALONE speculative decoding and role switching work with disaggregation; startup refusals are listed in [Options](#options).

## Speculative decoding

The plugin verifies one chain of drafts a request on a causal attention language model:

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3-1.7B \
  --speculative-algorithm NGRAM --speculative-ngram-max-bfs-breadth 1
python -m sglang.launch_server --model-path Qwen/Qwen3-1.7B \
  --speculative-algorithm STANDALONE --speculative-draft-model-path Qwen/Qwen3-0.6B
```

A STANDALONE draft model loads on the target's chips and shares their memory; it must share the target's token-to-id mapping, vocabulary and cache block size, and its window must hold the context.

## Requests

A decoder call takes token IDs and returns each row's greedy token, without the scores of the vocabulary.
The plugin answers a request that asks for any of these with HTTP 400 and the reason, before the scheduler runs it:

- Sampling other than greedy decoding (`temperature` 0 or `top_k` 1), penalties, `min_new_tokens`, `logit_bias` (even an empty one), grammars (`regex`, `json_schema`, `ebnf`, `structural_tag`, and so a `tool_choice` of `required` or a named function), beam search, watermarks, logprobs, sampling masks and a request's own logit processor.
- Input embeddings and embedding overrides (`input_embeds`, `positional_embed_overrides`, `embed_overrides`) in place of token IDs.
- Images, video and audio.
