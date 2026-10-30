{% from "commands.md" import vllm_serve, sglang_serve with context %}
{%- set embedding = benchmarks.models | selectattr("id", "equalto", "Qwen/Qwen3-Embedding-0.6B") | first -%}
# Serving

The vLLM and SGLang plugins serve model functions through each framework's own Python and HTTP APIs, and the model computation runs on RNGD.
This page shows each serving task for both frameworks; the tabs switch together.
Install a plugin first ([Installation](../getting-started/installation.md)); [Quickstart](../getting-started/quickstart.md) shows text generation.

## Feature support

| Feature | vLLM | SGLang | Details |
| --- | --- | --- | --- |
| Text generation | <span class="support support-yes">Supported</span> (greedy decoding) | <span class="support support-yes">Supported</span> (greedy decoding) | [Quickstart](../getting-started/quickstart.md#serve-over-http) |
| Embeddings | <span class="support support-yes">Supported</span> | <span class="support support-yes">Supported</span> | [Serve embeddings](#serve-embeddings) |
| Reranking | <span class="support support-no">Not yet</span> | <span class="support support-no">Not yet</span> | [Not supported yet](#features-not-supported-yet) |
| Classification and rewards from a score head | <span class="support support-no">Not yet</span> | <span class="support support-no">Not yet</span> | [Not supported yet](#features-not-supported-yet) |
| Multimodal inputs (images, video, audio) | <span class="support support-no">Not yet</span> | <span class="support support-no">Not yet</span> | [Not supported yet](#features-not-supported-yet) |
| Structured outputs | <span class="support support-no">Not yet</span> | <span class="support support-no">Not yet</span> | [Not supported yet](#features-not-supported-yet) |
| Tool calling and reasoning outputs | <span class="support support-no">Not yet</span> | <span class="support support-yes">Supported</span> (`tool_choice` `auto`) | [Call tools and separate reasoning](#call-tools-and-separate-reasoning) |
| Prompt embeddings | <span class="support support-no">Not yet</span> | <span class="support support-no">Not yet</span> | [Not supported yet](#features-not-supported-yet) |
| Custom logits processors | <span class="support support-no">Not yet</span> | <span class="support support-no">Not yet</span> | [Not supported yet](#features-not-supported-yet) |
| Watermarking | <span class="support support-no">Not yet</span> | <span class="support support-no">Not yet</span> | [Not supported yet](#features-not-supported-yet) |
| Speculative decoding (ngram, suffix, draft model) | <span class="support support-yes">Supported</span> (ngram and suffix; no draft model yet) | <span class="support support-yes">Supported</span> (ngram and draft model; no suffix decoding in SGLang) | [ngram or suffix](#speculate-with-ngram-or-suffix-decoding), [draft model](#speculate-with-a-draft-model) |
| Speculative decoding (EAGLE, MTP, Medusa, MLP, parallel draft models) | <span class="support support-no">Not yet</span> | <span class="support support-no">Not yet</span> | [Not supported yet](#features-not-supported-yet) |
| Automatic prefix caching | <span class="support support-yes">Supported</span> | <span class="support support-yes">Supported</span> | [Reuse cached prefixes](#reuse-cached-prefixes) |
| FP8 KV cache | <span class="support support-no">Not yet</span> | <span class="support support-no">Not yet</span> | [Not supported yet](#features-not-supported-yet) |
| Quantized checkpoints (FP8, NVFP4) | <span class="support support-no">Not yet</span> | <span class="support support-no">Not yet</span> | [Not supported yet](#features-not-supported-yet) |
| Online quantization | <span class="support support-yes">Supported</span> | <span class="support support-no">Not yet</span> | [Quantize a BF16 checkpoint at load](#quantize-a-bf16-checkpoint-at-load) |
| KV cache offloading | <span class="support support-yes">Supported</span> | <span class="support support-yes">Supported</span> | [Offload the KV cache to host memory](#offload-the-kv-cache-to-host-memory) |
| Sleep mode | <span class="support support-yes">Supported</span> | <span class="support support-yes">Supported</span> (without speculative decoding) | [Put the server to sleep](#put-the-server-to-sleep) |
| Long contexts, up to the compiled window | <span class="support support-yes">Supported</span> | <span class="support support-yes">Supported</span> | [Serve long contexts](#serve-long-contexts) |
| Context extension past the trained length | <span class="support support-yes">Supported</span> | <span class="support support-yes">Supported</span> | [Extend the context past the trained length](#extend-the-context-past-the-trained-length) |
| Tensor parallelism over a compiled chip count | <span class="support support-no">Not yet</span> | <span class="support support-no">Not yet</span> | [Not supported yet](#features-not-supported-yet) |
| Data parallelism | <span class="support support-yes">Supported</span> | <span class="support support-yes">Supported</span> | [Run data-parallel replicas](#run-data-parallel-replicas) |
| Pipeline, expert and context parallelism | <span class="support support-no">Not yet</span> | <span class="support support-no">Not yet</span> | [Not supported yet](#features-not-supported-yet) |
| LoRA | <span class="support support-yes">Supported</span> | <span class="support support-no">Not yet</span> | [Serve LoRA adapters](#serve-lora-adapters) |
| Disaggregated prefill and KV connectors (NIXL, Mooncake, MoRIIO, LMCache) | <span class="support support-no">Not yet</span> | <span class="support support-no">Not yet</span> | [Not supported yet](#features-not-supported-yet) |
| Disaggregated encoder and encoder caches | <span class="support support-no">Not yet</span> | <span class="support support-no">Not yet</span> | [Not supported yet](#features-not-supported-yet) |
| Per-request metrics | <span class="support support-yes">Supported</span> | <span class="support support-yes">Supported</span> | [Report per-request metrics](#report-per-request-metrics) |
| Batch invariance | <span class="support support-yes">Supported</span> | <span class="support support-yes">Supported</span> | [Keep outputs independent of the batch](#keep-outputs-independent-of-the-batch) |

A feature that is not supported yet is refused only in the ways [Features not supported yet](#features-not-supported-yet) lists.

## Serve embeddings

=== "vLLM"

    ```bash
    {{ vllm_serve(embedding) | indent(4) }}
    ```

    To send inputs from another shell:

    ```bash
    curl http://127.0.0.1:8000/v1/embeddings \
      -H 'Content-Type: application/json' \
      -d '{"model": "Qwen/Qwen3-Embedding-0.6B", "input": ["The capital of France is Paris."]}'
    ```

    vLLM selects and normalizes the embedding by the model's pooling configuration.

=== "SGLang"

    ```bash
    {{ sglang_serve(embedding) }}
    ```

    To send inputs from another shell, with `dimensions` for a shorter embedding:

    ```bash
    curl http://127.0.0.1:30000/v1/embeddings \
      -H 'Content-Type: application/json' \
      -d '{"model": "Qwen/Qwen3-Embedding-0.6B", "input": ["The capital of France is Paris."], "dimensions": 256}'
    ```

    SGLang's pooler selects the last token, truncates dimensions and normalizes the result.

The response holds each input's vector in `data[i].embedding`.
[Qwen3 Embedding](../models/qwen3-embedding.md) lists every embedding checkpoint.

## Call tools and separate reasoning

With SGLang, a tool call parser turns the checkpoint's tool-call text into the response's `tool_calls`, and a reasoning parser moves its thinking apart from the answer:

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --tool-call-parser qwen25 --reasoning-parser qwen3
```

Send `tools` with `/v1/chat/completions`; the parser returns the call the model writes, and the thinking is in `message.reasoning_content`.
A `tool_choice` of `required` or a named function constrains the answer with a grammar, and `--enable-strict-thinking` constrains the thinking, so both are refused as structured outputs are.
The parser names depend on the checkpoint's chat template.

## Speculate with ngram or suffix decoding

With vLLM, `--speculative-config '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_min": 2, "prompt_lookup_max": 4}'` drafts tokens by copying from the request's own text, and `{"method": "suffix"}` also from earlier responses; the target verifies the drafts, so greedy output does not change.

With SGLang, `--speculative-algorithm NGRAM --speculative-ngram-max-bfs-breadth 1` drafts from the requests' own text; SGLang has no suffix decoding.
[SGLang options](sglang.md#speculative-decoding) lists the flags it accepts.

Suffix decoding runs through Arctic Inference, which builds from source with a C++ compiler, such as `sudo apt install g++`.
To install Arctic Inference, add the plugin's `suffix` extra:

```bash
CARGO=cargo-furiosa-opt pip install --extra-index-url https://download.pytorch.org/whl/cpu './integrations/furiosa-opt-vllm[suffix]'
```

Without it, vLLM refuses `{"method": "suffix"}` at startup and names the package.
Its message pins an older Arctic Inference release; the extra pins the one the plugin is tested with.

## Speculate with a draft model

With SGLang, a smaller causal language model that shares the target's tokenizer drafts the tokens on the target's chips:

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3-1.7B \
  --speculative-algorithm STANDALONE --speculative-draft-model-path Qwen/Qwen3-0.6B
```

[SGLang options](sglang.md#speculative-decoding) lists what the draft model shares with the target.

## Reuse cached prefixes

Both servers reuse the cache rows of a prompt prefix they computed before, as long as the cache holds them; the setting is on by default:

=== "vLLM"

    `--no-enable-prefix-caching` turns it off. With `--enable-prompt-tokens-details`, a response reports the reused tokens in `usage.prompt_tokens_details.cached_tokens`.

=== "SGLang"

    `--disable-radix-cache` turns it off. A `/generate` response reports the reused tokens in `meta_info.cached_tokens`.

## Offload the KV cache to host memory

Both servers copy the cache blocks of computed prompts to a tier of host memory, and copy them back into the device cache when a later prompt shares a prefix whose blocks the device has evicted, instead of computing it again:

=== "vLLM"

    ```bash
    vllm serve Qwen/Qwen3-0.6B --kv-offloading-size 16
    ```

    `--kv-offloading-size` sets the host tier in GiB, as on a GPU.
    The platform serves it through the plugin's `RNGDOffloadingConnector` instead of vLLM's `OffloadingConnector`, which copies with CUDA streams; vLLM's scheduler still decides what to store, load and evict.
    Options of vLLM's CPU offloading, such as `blocks_per_chunk` and `eviction_policy`, go in `--kv-transfer-config '{"kv_connector_extra_config": {...}}'` beside the size.
    vLLM completes a transfer once the worker of each rank reports it, and one worker drives every chip, so offloading serves one chip a replica.
    It also serves one replica: with data parallelism vLLM ranks each replica's worker among all of them.
    Sleep mode frees the cache storage the host tier copies, so startup refuses `--enable-sleep-mode` beside offloading.
    The `vllm:kv_offload_store_bytes` and `vllm:kv_offload_load_bytes` metrics count the bytes moved each way.

=== "SGLang"

    ```bash
    python -m sglang.launch_server --model-path Qwen/Qwen3-1.7B --enable-hierarchical-cache --hicache-ratio 2
    ```

    `--hicache-ratio` sizes the host tier as a multiple of the device cache, `--hicache-size` in GB; SGLang sizes it from the free host memory only for the device pools it builds itself, so one of them is required.
    The memory saver frees the cache storage the host tier copies, so startup refuses `--enable-memory-saver` beside the hierarchical cache.
    A `/generate` response reports reused tokens by tier in `meta_info.cached_tokens_details`.

Only the attention cache moves, so both servers refuse offloading at startup for a model with recurrent state, and vLLM also for one with encoder memory; the [vLLM](vllm.md#options) and [SGLang](sglang.md#options) option pages name the other refused settings.
A failed copy poisons the memory it touched, and neither server recovers once it observes the failure, so vLLM's `kv_load_failure_policy` takes only `fail`.
vLLM observes the failure as soon as the copy ends; SGLang may observe it only later, when a call or copy next uses that memory.

## Put the server to sleep

A sleeping server frees the device memory of the model's weights and KV cache for another process; freeing the cache forgets what it held, and the compiled kernels stay loaded.
Sleep mode supports text generation with Qwen3 BF16 checkpoints on one chip.
To wake to the weights it loaded, whatever happens to the checkpoint's files meanwhile, a server reads the checkpoint's tensors into host memory, loads them from there and keeps that copy for as long as it serves them: as many bytes as the checkpoint's weight files hold, 1.5 GB for Qwen3-0.6B and 16.4 GB for Qwen3-8B.
Without the copy, waking reads the checkpoint's files as they are then.

=== "vLLM"

    Start the server with sleep mode, which keeps the copy:

    ```bash
    VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-0.6B --enable-sleep-mode
    ```

    To put it to sleep and wake it from another shell:

    ```bash
    curl -X POST 'http://127.0.0.1:8000/sleep?level=1'
    curl -X POST http://127.0.0.1:8000/wake_up
    ```

    Level 1 keeps the copy and wakes from it; level 2 frees the copy too, and waking reads the checkpoint's files into a new one.
    `wake_up?tags=weights` and `wake_up?tags=kv_cache` wake one at a time, in either order; a wake that fails keeps what it restored, and the next wake restores the rest.
    `LLM.sleep` and `LLM.wake_up` do the same in Python, with `enable_sleep_mode=True`.
    Levels 1 and 2 need sleep mode, a rule of the plugin rather than of vLLM: only a load under sleep mode keeps the copy that level 1 wakes from.
    A server started without it refuses both levels; vLLM pauses scheduling before the refusal, and `wake_up` resumes it.
    Level 0 only pauses scheduling, as in vLLM.
    `reload_weights`, through `collective_rpc`, reads the checkpoint's files again and takes no other source.
    It frees the served weights' device memory before it loads the new ones, so the device needs no room for a second copy; in host memory, the old copy stays beside the new one until the new weights load.
    When that load fails, no weights are on the device: a later `reload_weights` loads them, and a level-1 sleep and wake restores the old ones from their copy.

=== "SGLang"

    Start the server with the memory saver, and with `--enable-weights-cpu-backup` to keep the copy:

    ```bash
    python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --enable-memory-saver --enable-weights-cpu-backup
    ```

    To release the memory and resume it from another shell:

    ```bash
    curl http://127.0.0.1:30000/release_memory_occupation -H 'Content-Type: application/json' -d '{}'
    curl http://127.0.0.1:30000/resume_memory_occupation -H 'Content-Type: application/json' -d '{}'
    ```

    `{"tags": ["weights"]}` and `{"tags": ["kv_cache"]}` release or resume one at a time, in either order.
    Releasing the weights alone keeps what the cache held, unless the weights come back from the files, which may then hold weights other than those the cache rows were computed with.
    A resume that fails, such as from files that no longer hold the checkpoint, stops the server, as any scheduler error does in SGLang.
    Without `--enable-memory-saver`, releasing frees nothing, as in SGLang; with speculative decoding, the memory saver is refused.

## Quantize a BF16 checkpoint at load

vLLM's per-tensor online FP8 quantizes a Qwen3-0.6B, Qwen3-1.7B or Qwen3-4B BF16 checkpoint as it loads:

```bash
vllm serve Qwen/Qwen3-0.6B --quantization fp8_per_tensor
```

The load stores each decoder projection's weight as E4M3 codes with FP32 scales, as vLLM's method computes them: one scale per projection, shared by the projections vLLM fuses into one matrix (a layer's query, key and value; its gate and up), the largest magnitude over 448.
Each projection restores the codes to BF16 before its product, so the activations stay in BF16, as vLLM computes this method without an FP8 matrix unit; the embedding and the vocabulary head stay in BF16.
Startup refuses Qwen3-8B, whose packed query, key and value projection has no FP8 form, before downloading its weights.
Startup also refuses the other methods, which quantize the activations too (`fp8_per_channel`, `fp8_per_block`) or use other formats (MX, NVFP4), and `--quantization` on a quantized checkpoint.
SGLang refuses `--quantization` for a BF16 checkpoint at startup.

## Serve long contexts

A request may hold as many tokens as the checkpoint serves, up to the model's compiled context window: its trained length, or the longer one its rotary scaling extends it to ([Extend the context past the trained length](#extend-the-context-past-the-trained-length)).
Without a context length option, the server takes the checkpoint's own length; startup stops and names the reason when:

- that length or the context length option exceeds the window: `requires max_model_len at most 131072` (vLLM `--max-model-len`) or `requires context_length at most 131072` (SGLang `--context-length`); either framework allows a length past the checkpoint's own only with `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` or `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`;
- with vLLM, the KV cache of one full-length request does not fit in device memory; pass `--max-model-len -1` to take the longest length that fits.

[Concepts](../concepts.md#compiled-limits) explains which limits compilation fixes.

## Extend the context past the trained length

A checkpoint serves a longer context than it was trained on under a rotary scaling that its configuration names, up to the window its decoder is compiled for.
To set one, override the checkpoint's `rope_parameters`; Qwen3-0.6B, trained on 32768 positions, then serves 131072, as its decoder is compiled for:

=== "vLLM"

    ```bash
    vllm serve Qwen/Qwen3-0.6B --hf-overrides '{"rope_parameters": {"rope_type": "yarn", "factor": 4.0,
      "original_max_position_embeddings": 32768, "rope_theta": 1000000.0}, "max_position_embeddings": 131072}'
    ```

=== "SGLang"

    ```bash
    python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --json-model-override-args \
      '{"rope_parameters": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768,
      "rope_theta": 1000000.0}, "max_position_embeddings": 131072}'
    ```

The override replaces the checkpoint's whole `rope_parameters`, so it repeats `rope_theta`.
A checkpoint whose own config names the scaling serves the same way, without an override; one that keeps its `max_position_embeddings`, as Qwen's model card edits it, serves a longer context under the context length options of [Serve long contexts](#serve-long-contexts).
The engine turns each rotary pair as Transformers does for the same configuration, and the context the server takes depends on the rope type:

| `rope_type` | Context | Also set |
| --- | --- | --- |
| `linear` | `factor` x `max_position_embeddings` | |
| `yarn` | `max_position_embeddings` | `max_position_embeddings` and `original_max_position_embeddings` |
| `llama3` | `max_position_embeddings` | as the checkpoint writes it |

With SGLang, a `yarn` override without `original_max_position_embeddings` takes `factor` x `max_position_embeddings`, which `--context-length` brings within the compiled window.
YaRN scales every rotation by its attention factor, the one the parameters give or the one `factor` and `mscale` imply.
As in Transformers, a missing `original_max_position_embeddings` is `max_position_embeddings`, and a null YaRN `factor` is `max_position_embeddings` over `original_max_position_embeddings`.
The compiled window bounds the context as for any checkpoint ([Serve long contexts](#serve-long-contexts)): 131072 positions for Qwen3 0.6B, 1.7B and 4B, and 40960 for Qwen3 8B.
Startup refuses a rope type the engine does not compute, such as `dynamic` or `longrope`, naming the ones it does.

## Run data-parallel replicas

Data parallelism serves a dense model as independent replicas, each on its own chips: replica `i` takes the chips of its tensor degree from position `i` times that degree of `FURIOSA_VISIBLE_CHIPS`, or of the installed chips when it is unset.

=== "vLLM"

    ```bash
    vllm serve Qwen/Qwen3-8B --data-parallel-size 2
    ```

=== "SGLang"

    ```bash
    python -m sglang.launch_server --model-path Qwen/Qwen3-8B --dp-size 2
    ```

Both plugins refuse speculative decoding with data parallelism; vLLM also refuses a mixture-of-experts model and its Ray backend with it.

## Report per-request metrics

Each response can carry its own timings beside the server's aggregated metrics:

=== "vLLM"

    `--enable-per-request-metrics` adds `metrics` to each response: `time_to_first_token_ms`, `mean_itl_ms`, `tokens_per_second` and the queue and generation times.

=== "SGLang"

    A `/generate` response reports `meta_info.e2e_latency` in seconds and its token counts.

## Serve LoRA adapters

With vLLM, each request may name a LoRA adapter of the served model, saved as PEFT saves it; requests of different adapters and of the base model share a step.
The engine adds an adapter's updates to the attention projections `q_proj`, `k_proj`, `v_proj` and `o_proj` and the MLP projections `gate_proj`, `up_proj` and `down_proj`, up to rank 32, at most two adapters a step.
[Qwen3 0.6B](../models/qwen3-0.6b.md) on one chip runs them, its tokens those of Transformers with each adapter merged by PEFT; `--enable-lora` with a quantized checkpoint, such as the block-scaled FP8 [Qwen3 8B](../models/qwen3-8b.md), or with an online quantization is refused at startup, as those loads run no adapter.

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM("Qwen/Qwen3-0.6B", enable_lora=True, max_loras=2, max_lora_rank=32)
sky = LoRARequest("sky", 1, "/path/to/adapter")
outputs = llm.generate(["Why is the sky blue?"], SamplingParams(temperature=0), lora_request=sky)
```

A loaded adapter stays on the device, out of the memory the KV cache leaves free.
`max_loras` sets the adapters a step runs, at most 2, and `max_cpu_loras` the adapters kept loaded, the least recently used unloaded first.
`llm.llm_engine.add_lora`, `pin_lora` and `remove_lora` act on the loaded adapters: a load also makes its adapter one of the `max_loras` active ones, as vLLM activates every adapter it adds, and a pinned adapter stays loaded and active until it is removed, so a load while `max_loras` pinned adapters are active is refused and changes nothing; a `LoRARequest` with `load_inplace=True` replaces the adapter of its ID.

The adapter directory holds `adapter_config.json` and `adapter_model.safetensors`.
An adapter that adapts another module, such as the embedding or the output head, or that uses DoRA, bias terms or per-module ranks is refused when it loads, naming what it holds, before any of it reaches the device; [vLLM options](vllm.md#options) lists the refused settings.

## Keep outputs independent of the batch

By default a call tiles each row's attention for all the rows it holds, so the last bits of a request's results can change with the requests that share its steps.
To make a request's results independent of its batch, serve the model batch invariant:

=== "vLLM"

    ```bash
    VLLM_BATCH_INVARIANT=1 vllm serve Qwen/Qwen3-0.6B
    ```

=== "SGLang"

    ```bash
    python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --enable-deterministic-inference
    ```

    The radix cache stays on: a request whose prefix the cache holds keeps the bits it has without the cache.

Each row's attention is then tiled by its own history alone: a request's tokens, embeddings and hidden rows keep their bits whatever other requests share its steps, wherever its rows lie in a step and however its prompt is chunked.
Requests stay greedy: sampling, including SGLang's seeded deterministic sampling, and logprobs are [not supported yet](#features-not-supported-yet).

The fixed tiling is slower where the default one adapts to the step's rows.
Measured on Qwen3-0.6B and Qwen3-8B, a call takes within 5% of the default time for histories up to 4,096 keys, 10 to 29% longer for rows just past 4,096 or 8,192 keys, which split into one more piece of attention work and a merge, and up to 14% less at 16,384 keys.

## Embed media on an encoder-only instance

An encoder-only vLLM instance, the producer half of a disaggregated encoder, embeds the images, video and audio of each request on the device and saves each item's rows through an encoder cache connector, for a language-only instance to read. The plugin runs Gemma 4 unified's media encoder this way through vLLM's `ECExampleConnector`:

```bash
vllm serve unsloth/gemma-4-12b-it-NVFP4 \
  --ec-transfer-config '{"ec_connector": "ECExampleConnector", "ec_role": "ec_producer", "ec_connector_extra_config": {"shared_storage_path": "/tmp/encoder-cache"}}'
```

Each item's rows, one for each of its placeholder tokens, are written to `<shared_storage_path>/<item hash>/encoder_cache.safetensors`. No instance reads them yet: the engine serves no language model of the checkpoint, so the plugin refuses the `ec_consumer` and `ec_both` roles, and SGLang serves neither half.

## Choose the chips

Set `FURIOSA_VISIBLE_CHIPS` to a comma-separated list of chip indices, such as `FURIOSA_VISIBLE_CHIPS=2,3`; the server then places the model on those chips in that order, and on chips `0`, `1` and so on when it is unset.

## What both plugins accept

- Each plugin overrides or checks some framework options and refuses unsupported values at startup: [vLLM options](vllm.md), [SGLang options](sglang.md). Other options keep the framework's behavior.
- The engine accepts a tensor degree of 1 or a chip count the model is compiled to split over, and a pipeline degree of 1; startup refuses any other with the engine's reason.
- The commands name checkpoints as examples; [Models](../models/index.md) lists every checkpoint that runs.

## Features not supported yet

Asking for a feature that is not supported yet has one of three outcomes:

- **Refused** before it runs, with the reason: an option the [vLLM](vllm.md#options) or [SGLang](sglang.md#options) page names stops startup, and so does a checkpoint config no served decoder holds, such as a rope type the engine does not compute; a request field [SGLang requests](sglang.md#requests) names fails its request with HTTP 400, and vLLM refuses a request that is not greedy or holds prompt embeddings.
- **Not verified**: tool calling and reasoning outputs with vLLM and `tool_choice` `auto` may work, as its parsers read the greedy answer, but no test runs them on RNGD yet.
- **Not refused**: any other way to ask may fail with an internal error or not take effect. A vLLM draft model (`"method": "draft_model"`) passes the option check, then fails at load.

Why each feature is not served yet:

- **Sampling, logprobs and logit changes**: structured outputs (and so a `required` or named tool choice), custom logits processors, watermarking, penalties and logit bias. A decoder call returns each row's greedy token, not the scores of the vocabulary.
- **Reranking and score heads**: SGLang's reranking reads the logprobs of two tokens; vLLM's reranking and either plugin's classification run a score head, which the engine loads but neither plugin calls yet.
- **Prompt embeddings**: a decoder call takes token IDs.
- **Multimodal inputs**: no served checkpoint answers about images, video or audio; Gemma 4 unified's media encoder runs [on an encoder-only instance](#embed-media-on-an-encoder-only-instance), but the engine serves no language model of it.
- **Speculative methods other than ngram and suffix** (vLLM, which also has no draft model yet) or NGRAM and STANDALONE (SGLang).
- **FP8 KV cache and quantized checkpoints**: the served checkpoints load BF16 weights and cache keys and values in BF16.
- **Online quantization** with SGLang, and with vLLM any method but `fp8_per_tensor`: the other methods quantize the activations too or use other formats, and the SGLang plugin passes no quantization to the load.
- **Tensor, pipeline, expert and context parallelism**, and serving across nodes: each served checkpoint runs on one chip.
- **LoRA with SGLang**: `--enable-lora` and `--lora-paths` are refused; the SGLang plugin runs no adapter yet.
- **Disaggregated prefill, KV connectors and a disaggregated encoder**: no cache row moves to another instance, and of a disaggregated encoder only the [encoder-only instance](#embed-media-on-an-encoder-only-instance) runs.
