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
| `disaggregation_mode` | `null`, or `prefill` and `decode` with `--disaggregation-transfer-backend mooncake_tcp` ([Disaggregate prefill and decode](index.md#disaggregate-prefill-and-decode)) |
| Attention data, expert, context parallelism (`attn_dp_size`, `ep_size`, `dcp_size`, `attn_cp_size`) and `nnodes` | 1 |
| `--tp-size` | A chip count the model is compiled for; one scheduler drives every chip, so `/server_info` reports `tp_size` 1 |
| `--dp-size` | Any, or 1 with speculative decoding; replica `i` runs on the `--tp-size` chips from position `i` × `--tp-size` of `FURIOSA_VISIBLE_CHIPS`, or of every installed chip when it is unset |
| `--gpu-id-step` | Unset; the platform spaces the replicas by the tensor degree |
| `--context-length` | At most the decoder's compiled window, which startup names otherwise |
| `rope_parameters` or `rope_scaling` and `max_position_embeddings` in `--json-model-override-args` | A `rope_type` the engine computes, within the compiled window ([Extend the context](index.md#extend-the-context-past-the-trained-length)) |
| `--mem-fraction-static` | Unset defaults to 0.9, leaving the headroom a model split over chips needs |
| `--enable-deterministic-inference` | Off, or on to load the model [batch invariant](index.md#keep-outputs-independent-of-the-batch) with the radix cache on |
| `--enable-hierarchical-cache` | A model without recurrent state, with the host tier sized by `--hicache-ratio` or `--hicache-size` and kept in memory: `--hicache-host-memory-mode cache`, `--hicache-io-backend kernel`, `--hicache-mem-layout page_first`, no `--hicache-storage-backend` and no memory saver |
| `--quantization` | Unset, or `fp8` to [quantize a BF16 checkpoint at load](index.md#quantize-a-bf16-checkpoint-at-load) |
| `--speculative-draft-model-quantization` | Unset, which takes the target's, `unquant`, or `fp8` |

These features are not supported yet, and the plugin refuses them at startup rather than ignore them:

- LoRA and speculative algorithms other than NGRAM and STANDALONE.
- Speculative decoding with more than one candidate a position, adaptive speculation or an external n-gram corpus.
- Watermarking (`--enable-watermark`) and strict thinking (`--enable-strict-thinking`), which change the scores of each step.
- Quantization methods other than `fp8`, of the target or of a STANDALONE draft model, and quantization parameter files.
- Hierarchical cache storage backends, HiSparse, unified memory, LMCache, FlexKV and the unified cache external linker.
- The memory saver (`--enable-memory-saver`) with speculative decoding, whose worker and draft model keep device memory a release does not reach yet, and with the hierarchical cache, whose host tier copies the cache storage a release frees.
- Per-layer embedding offload (`ple_offload_embedding`).
- Returning hidden states, routed experts or indexer top-k.
- Diffusion LM algorithms (`dllm_algorithm`) and `prefill_only_disable_kv_cache`.
- Encoder disaggregation (`--encoder-only`, `--language-only`).
- With prefill-decode disaggregation: a transfer backend other than `mooncake_tcp` (`nixl` and `mori` register the token pool as device memory, and `fake` moves no KV), a model split over chips (an image page holds each chip's part in turn, which a server split otherwise reads differently), host receive (`--disaggregation-decode-host-receive-threshold` below 1: SGLang receives into a host pool only beside a token pool of separate key and value buffers), decode-side cache offload, which needs a storage backend, KV checksums, which SGLang computes with a CUDA kernel, a `host_pool` retraction backup, which SGLang backs up only from an MHA token pool, and NGRAM speculative decoding, whose decode server cannot read the first token SGLang hands a received request.

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
