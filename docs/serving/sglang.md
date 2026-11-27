# SGLang options

This page lists the SGLang options the plugin overrides or checks; other options keep SGLang's behavior.
[Serving](index.md) shows each serving task, and [Integrating frameworks](../engine/frameworks.md#return-the-rows-the-framework-needs) explains which steps take their tokens from the device.

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
| `disaggregation_mode` | `null` |
| Attention data, expert, context parallelism (`attn_dp_size`, `ep_size`, `dcp_size`, `attn_cp_size`) and `nnodes` | 1 |
| `--tp-size` | A chip count the model is compiled for; one scheduler drives every chip, so `/server_info` reports `tp_size` 1 |
| `--dp-size` | Any, or 1 with speculative decoding; replica `i` runs on the `--tp-size` chips from position `i` × `--tp-size` of `FURIOSA_VISIBLE_CHIPS`, or of every installed chip when it is unset |
| `--gpu-id-step` | Unset; the platform spaces the replicas by the tensor degree |
| `--context-length` | At most the decoder's compiled window, which startup names otherwise |
| `rope_parameters` or `rope_scaling` and `max_position_embeddings` in `--json-model-override-args` | A `rope_type` the engine computes, within the compiled window ([Extend the context](index.md#extend-the-context-past-the-trained-length)) |
| `--mem-fraction-static` | Unset defaults to 0.9, leaving the headroom a model split over chips needs |
| `--enable-deterministic-inference` | Off, or on to load the model [batch invariant](index.md#keep-outputs-independent-of-the-batch) with the radix cache on |
| `--enable-hierarchical-cache` | Models without recurrent state; configure host-tier size, storage backend, and required modes as described in [Hierarchical cache](#hierarchical-cache) |

## Hierarchical cache

### Configure the host tier

The plugin connects SGLang's HiCache page selection and lifecycle to engine-pinned host slots. Choose a host-tier size with `--hicache-ratio` or `--hicache-size` and use `--hicache-host-memory-mode cache`, `--hicache-io-backend kernel`, and `--hicache-mem-layout page_first`.

The accepted storage backends are unset, `file`, and `mooncake`; other backends need their own host-buffer integration. Automatic sizing from free host memory is available only for pools SGLang allocates itself.

### Filesystem tier

The `file` backend selects `FuriosaHiCacheFile`; the [serving task guide](index.md#offload-the-kv-cache-to-host-memory) has its launch command. The adapter uses SGLang's `dynamic` storage-backend factory and preserves SGLang's cache keys and file-tier lifecycle. The [shared filesystem guide](../engine/frameworks.md#filesystem-layout-and-io) owns the file format and I/O alignment rules.

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

### Mooncake storage

Install the optional `mooncake` extra as described in [Installation](../getting-started/installation.md#install-optional-mooncake-storage-support), configure SGLang's Mooncake Store, and select the native HiCache backend:

```bash
SGLANG_HICACHE_MOONCAKE_CONFIG_PATH=/etc/mooncake/sglang.json \
python -m sglang.launch_server --model-path Qwen/Qwen3-1.7B \
  --enable-hierarchical-cache --hicache-ratio 2 \
  --hicache-host-memory-mode cache --hicache-io-backend kernel --hicache-mem-layout page_first \
  --hicache-storage-backend mooncake
```

The Mooncake backend preserves SGLang's lookup lifecycle and stores one complete engine page per key. Its namespace includes model, parallelism and page layout, so instances share entries only when those match. It does not fingerprint checkpoint revisions; change `extra_backend_tag` when weights change under the same model name. `standalone_storage` is unsupported because Mooncake must register engine-owned host pages.

## Features not supported yet

The plugin refuses these framework features at startup rather than ignore them:

- LoRA and speculative algorithms other than NGRAM and STANDALONE.
- Speculative decoding with more than one candidate a position, adaptive speculation or an external n-gram corpus.
- Watermarking (`--enable-watermark`) and strict thinking (`--enable-strict-thinking`), which change the scores of each step.
- Quantization and quantization parameter files.
- HiSparse, unified memory, LMCache, FlexKV and the unified cache external linker; this plugin supports `file` and `mooncake` storage backends.
- The memory saver (`--enable-memory-saver`) with speculative decoding, whose worker and draft model keep device memory a release does not reach yet.
- Per-layer embedding offload (`ple_offload_embedding`).
- Returning hidden states, routed experts or indexer top-k.
- Diffusion LM algorithms (`dllm_algorithm`) and `prefill_only_disable_kv_cache`.
- Encoder disaggregation (`--encoder-only`, `--language-only`).

The platform disables CUDA graphs and piecewise CUDA graphs, and sets the scheduling options by model kind:

| Model | Page size | Overlap schedule | Mixed chunks |
| --- | --- | --- | --- |
| Non-recurrent generation | 16 | On, unless you disable it | On |
| Recurrent-state generation | 1 | Off | Off |
| Embedding | 16 | Off | Off |
| Speculative decoding | 16 | Off | Off |

A model that keeps recurrent state also requires `mamba_ssm_dtype` unset or `float32` and `mamba_radix_cache_strategy` `auto` or `no_buffer`, and does not support `enable_int8_mamba_checkpoint`, `enable_linear_replayssm` or `enable_linear_replayssm_spec` yet.
[Integrating frameworks](../engine/frameworks.md#sglang-specifics) explains the overlap schedule and mixed chunks.

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
