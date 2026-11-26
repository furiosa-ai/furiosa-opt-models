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

## Offload cache pages

Frameworks keep cache selection, keys, eviction and tier lifecycle; the engine owns pinned host slots, page routing and DMA. The [serving guide](../serving/index.md#offload-the-kv-cache-to-host-memory) has setup recipes. The [vLLM](../serving/vllm.md#kv-cache-offloading) and [SGLang](../serving/sglang.md#hierarchical-cache) references define each adapter's options and framework-specific behavior.

### Filesystem layout and I/O

The file tier preserves each framework's existing byte layout.

| Framework | On-disk layout |
| --- | --- |
| SGLang | One raw cache page per file. |
| vLLM | Native layer-major chunks, including trailing chunk-alignment padding. |

The engine package's shared `AlignedFileIO` utility enables `O_DIRECT` only after querying the filesystem's memory and offset alignments and completing an aligned probe. Offset or vector-length misalignment falls back to buffered I/O from pinned slots. For a single-buffer transfer, a misaligned address uses a per-thread aligned buffer for direct I/O; for a vectored transfer, any misaligned address in that range falls back to buffered I/O. vLLM submits each chunk's layer ranges in one file-I/O batch, reusing at most one direct and one buffered file handle; the engine splits each range only at the operating system's vector limit.

### DMA batches and ownership

The shared copier has six workers and no job-count cap. An ordinary DMA batch contains up to two pages with distinct host slots; each page carries all requested layer routes. Slot uniqueness applies within a batch, so a later batch may reuse a slot. Pages are ordered by host slot within a batch to keep concurrent lock acquisition in one order.

The ordinary path moves KV data directly between pinned host slots and existing HBM pages. It does not copy KV through a CPU staging buffer. A transfer that reuses a host slot or device page while DMA still touches it fails with `BlockingIOError`. A live host-slot lease prevents DMA until Python or filesystem access ends, while transfers using unrelated slots and pages proceed independently.

For separately allocated slots, one per-slot mutex owns both the pinned buffer and its DMA/lease state. The direct path holds that mutex through DMA; the owned buffer needs no second host-buffer mutex. Each queue worker owns a clone of the same PDMA device handle, so it builds and submits transfers without taking the decoder's context mutex. Cloning the runtime context shares the opened device and its memory; it does not open another device. This preserves submission overlap with decoder calls while keeping each mutable PDMA context owned by one worker.

A shared backing uses the same direct route over per-slot byte ranges. The engine takes each slot's logical DMA gate, then holds the allocation-wide backing mutex while it builds borrowed runtime views and submits the request. It releases the backing mutex before awaiting DMA. This serializes setup and submission for transfers using that backing; DMA can still overlap after submission. The filesystem L3 adapters use separately allocated slots.

### Optional coalescing

Coalescing is a Furiosa opt-in that stages one copy's pages in pinned memory before DMA. The engine allocates one pinned buffer per chip for the full page batch, merges contiguous layer ranges, and scatters stores back to their slots. Each request owns its stages until completion, and coalesced requests have no separate admission cap, so transient pinned-memory use scales with concurrent staged bytes. This can reduce DMA commands at the cost of CPU gather/scatter and pinned memory. The vLLM and SGLang references document each adapter's coalescing scope.

For separately allocated slots, coalescing holds the per-slot mutex only while gathering a load into staging or scattering a completed store. It releases the mutex before staging DMA because DMA touches staging buffers, not host slots. A load has already snapshotted its host bytes; a store publishes host bytes only after DMA completes. The slot gate does not reserve a cache value for the full staged transfer, so callers must await completion before reusing that value. The filesystem L3 adapters use this separately allocated path.

With a shared backing, coalescing holds the per-slot gate and allocation-wide backing mutex only while gathering or scattering one slot. Both are released before staging DMA. Transfers using that allocation can contend during CPU staging.

### Runtime command batching

On the channel-DMA path, the runtime groups layer/page segments into DMA-BUF commands by chip and host allocation. A command can carry multiple segments up to the driver's per-command byte limit. Each internal runtime slot holds up to 64 commands; larger command lists use multiple slots in sequence. This limit applies to runtime command descriptors, not to queued jobs.

For a two-page request across eight chips, the runtime creates at most 16 base commands if each chip/host-slot group fits one driver command. Thirty-two layers produce 512 layer/page/chip segments, but those remain grouped into the same 16 commands. Coalescing uses one staging allocation per chip, so the same request needs at most eight base commands under that size assumption. Larger groups may require additional commands and runtime slots when the total exceeds 64.

No NPU measurement establishes the end-to-end performance of these paths.

### Completion and failures

`copy_in` and `copy_out` return after queueing their chunks, without waiting for runtime submission or DMA. Callers must retain and await a `Copy` before reusing its host slots or device pages. A failed copy poisons the memory it touched, and the engine does not recover that memory. Each framework adapter reports the failure through its native completion path.

`Copier.shutdown()` drains submitted copies and joins the workers while releasing the Python GIL. Callers must release every host-slot access lease before shutdown, because an active lease blocks DMA until the access ends.

## SGLang specifics

SGLang's scheduler and checkpoint conventions require adapter-specific handling beyond request mapping.

The plugin loads the engine from the original checkpoint config with the user's overrides because SGLang rewrites some geometry for its own kernels.

With overlap enabled, SGLang schedules the next step and processes the previous step while the device runs. The plugin disables overlap for recurrent state and embeddings because they are read back within their step.

Mixed chunks interleave prompt chunks with running requests' decode rows in one step.

The full step logic is in [`generation/sequence.py` of vLLM](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/integrations/furiosa-opt-vllm/furiosa_opt_vllm/generation/sequence.py) and [of SGLang](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/integrations/furiosa-opt-sglang/furiosa_opt_sglang/generation/sequence.py).
