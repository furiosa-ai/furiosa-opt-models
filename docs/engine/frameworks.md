# Integrating frameworks

A plugin turns each scheduler step into engine requests and returns the rows the framework needs from them.
This page shows that mapping for vLLM and SGLang; [Using the engine](python.md) introduces requests and the cache.

## Map a scheduled chunk to a request

Each scheduled sequence becomes one `engine.decoder.Request`: the blocks that hold it, a list for each cache group, the positions they already hold, its new tokens and the rows to return.
The framework assigns blocks, and the plugin translates its assignment into block IDs:

=== "vLLM"

    vLLM assigns block IDs directly, so they pass through unchanged:

    ```python
    blocks = [2, 5]  # the request's block IDs from vLLM's block table
    # 20 positions cached, in the cache's one group of layers
    request = engine.decoder.Request([blocks], 20, tokens, engine.decoder.Keep.LAST)
    ```

=== "SGLang"

    SGLang assigns token locations, and its page size equals the compiled block size, 16.
    A page's start location divided by 16 is its block ID:

    ```python
    pages = [32, 80]  # page-start locations from SGLang's request table
    blocks = [location // 16 for location in pages]  # [2, 5]
    request = engine.decoder.Request([blocks], 20, tokens, engine.decoder.Keep.LAST)
    ```

A scheduler step can hold more rows than one call takes, so the plugin splits it into bounded calls in order.

## Return the rows the framework needs

| Chunk | `keep` |
| --- | --- |
| Part of a prompt that continues in a later step | `Keep.NONE`: the call only writes keys and values |
| The last chunk of a prompt, or a decode step | `Keep.LAST`: the row the framework samples from |
| Prompt logprobs or pooling | `Keep.ALL`: every appended row |

A step whose requests are all greedy, without logprobs, penalties or other sampling modifiers, takes its tokens from `Decoder.forward` and keeps them on the device for the next step.
Every other step needs scores for the framework's sampler, which a decoder's calls do not return yet, so the plugins refuse it.
The complete conditions are in each plugin: [`greedy` in vLLM](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/integrations/furiosa-opt-vllm/furiosa_opt_vllm/generation/__init__.py) and [`device_argmax` in SGLang](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/integrations/furiosa-opt-sglang/furiosa_opt_sglang/generation/__init__.py).

## Keep the cache allocation alive

The framework owns request lifetimes and block assignments; the engine owns the cache's device storage.
Evicting a request releases its block assignments, not the cache allocation, and SGLang's offload and relocation copy token rows without moving the allocation.

## SGLang specifics

- SGLang rewrites some checkpoint geometry for its own kernels, so the plugin loads the engine from the original checkpoint config with the user's overrides.
- With the overlap schedule, SGLang schedules the next step and processes the last one while the device runs. Recurrent state and embeddings are read back within their step, so the plugin turns overlap off for them.
- Mixed chunks interleave prompt chunks with the running requests' decode rows in one step.

The full step logic is in [`generation/sequence.py` of vLLM](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/integrations/furiosa-opt-vllm/furiosa_opt_vllm/generation/sequence.py) and [of SGLang](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/integrations/furiosa-opt-sglang/furiosa_opt_sglang/generation/sequence.py).

## Encoder-only media

The [engine encoder contract](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-engine/src/encoder/mod.rs) returns one BF16 `[rows, output_width]` tensor per input item, in input order, with one output row per input row. A model must compute each row independently: the engine leaves unused compiled launch rows uninitialized and discards their outputs. The plugins rely on that contract for engine item-count, rank, and output-width guarantees.

`LoadOptions.encoder()` returns typed `EncoderLoadOptions`; its `load()` returns `Encoder`. Decoder `LoadOptions.load()` returns `Decoder`. The generated Python stubs preserve this distinction, so neither plugin needs a runtime type cast. See [vLLM's load options](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/integrations/furiosa-opt-vllm/furiosa_opt_vllm/admission.py) and [SGLang's encoder loader](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/integrations/furiosa-opt-sglang/furiosa_opt_sglang/encoder.py).

Both encoder-only loaders use one chip. SGLang rejects `tp_size != 1` during platform config resolution; vLLM now rejects `tensor_parallel_size != 1` before checkpoint verification. Neither silently drops a requested multi-chip layout.

### Framework boundary checks

The frameworks own scheduling, cache lifetime, and EPD token accounting. vLLM's plugin checks output rows against placeholders because it calls the engine outside vLLM's multimodal model callback. SGLang's EPD path owns request cardinality and returned token counts. Adapters validate only processor-specific tensor axes, padding masks, and video grouping needed to build typed engine inputs; nearby comments identify those boundaries. Recheck these hook contracts when updating the pinned [vLLM 0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/worker/utils.py) and [SGLang 0.5.22](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/disaggregation/encoder/server.py) versions.
