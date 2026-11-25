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

SGLang's platform rejects requested tensor parallelism other than one during config verification and sets the runtime `tp_size` to one before encoder loading ([platform setup](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/integrations/furiosa-opt-sglang/furiosa_opt_sglang/platform.py)). The encoder loader relies on that resolved config instead of repeating the same check.

### Checks delegated to the pinned frameworks

- **vLLM 0.31.0:** the encoder-only plugin calls the engine directly, outside vLLM's normal `SupportsMultiModal` model callback and its [`sanity_check_mm_encoder_outputs`](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/worker/utils.py). The engine contract supplies item count and tensor shape; the plugin checks output rows against each feature's `mm_position`, which the engine cannot know.
- **SGLang 0.5.22:** [`EncoderPreprocessOutput`](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/multimodal/encoder_preprocessing.py) rejects an empty item list. [`MMEncoder._prepare_encode_context`](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/disaggregation/encoder/server.py) rejects zero-item request groups and checks grid and `token_counts` lengths. `MMEncoder.slice_embedding` checks total output rows on cache misses; the direct path calls `_validate_embedding_token_count`. The adapter delegates these checks instead of repeating them.
- **SGLang preprocessing:** the adapter checks processor tensor layout while building items. [`EncoderPreprocessOutput`](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/multimodal/encoder_preprocessing.py) carries those normalized items unchanged to the model callback, so that callback does not recheck their grid, row, and mask alignment.

### Checks kept at the adapter and engine boundaries

The SGLang adapter checks batched feature/position/mask layouts, video grouping from the processor's video/frame axes, and integer audio lengths against the valid-frame mask while constructing EPD items. SGLang's EPD checks do not validate those processor-specific relationships. The vLLM adapter passes its declared processor fields to the engine's typed constructors; those enforce BF16 row input and image row/position counts. The vLLM plugin separately checks placeholder alignment because vLLM does not.

The engine validates public media inputs (BF16 row shape, exact buffer size, Python tensor device ownership, and position bounds). Those checks apply to every engine caller and are not implied by a framework processor contract. Unsupported modality/capability checks likewise remain at dispatch because neither framework nor engine can infer which paths a loaded model implements.

The SGLang hooks are version-specific. Recheck these contracts when changing the pinned framework versions: [vLLM 0.31.0](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/worker/utils.py) and [SGLang 0.5.22](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/disaggregation/encoder/server.py).
