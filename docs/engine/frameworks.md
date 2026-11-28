# Integrating frameworks

A serving plugin maps framework scheduling and cache policy onto `opt-engine` for model execution and KV transfers. [Using the engine](python.md) documents the Python interface, and the [serving guide](../serving/index.md) gives launch recipes.

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
See the [vLLM step implementation](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/integrations/furiosa-opt-vllm/furiosa_opt_vllm/generation/sequence.py) and [SGLang scheduling](../serving/sglang.md#model-loading-and-scheduling).

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

Framework adapters own cache selection, keys, eviction and tier lifecycle. They translate scheduled blocks into host-slot and device-block routes. The engine owns the pinned host memory and moves the bytes between those slots and the device. See [vLLM](../serving/vllm.md#kv-cache-offloading) and [SGLang](../serving/sglang.md#hierarchical-cache) for tier settings and byte formats.

### Aligned filesystem I/O

`AlignedFileIO` reads and writes the byte ranges requested by a filesystem tier. It enables `O_DIRECT` after querying the filesystem's memory and offset alignments and completing an aligned probe. Misaligned offsets or vector lengths use buffered I/O. A single buffer with a misaligned address uses a per-thread aligned buffer for direct I/O; a misaligned address in a vector switches the whole vector to buffered I/O. One batch can reuse a direct and a buffered file handle, and vectors split at the operating system's vector limit.

### DMA batches and ownership

Each `Copier` owns a six-worker queue, with ordinary DMA batches of up to two distinct host slots. A batch carries all requested layer routes for its pages. It acquires slot gates in index order, and a later batch can reuse a slot. The queue can hold any number of jobs.

The direct path transfers bytes between pinned slots and HBM while holding the relevant slot gates. Workers build DMA routes before taking the shared device context lock. They hold it through submission and wait for DMA completion outside it. DMA submission and decoder calls share that context lock.

### Optional coalescing

Coalescing reduces the number of DMA commands by gathering or scattering one copy through a temporary pinned allocation per chip. The engine merges contiguous device ranges and holds each slot gate only while reading or writing its bytes. A load gathers before DMA, and a store scatters after DMA. The request retains its temporary buffers until completion, so temporary pinned memory scales with the staged bytes of concurrent copies.

### Runtime command batching

On the channel-DMA path, the runtime groups layer/page segments into DMA-BUF commands by chip and host allocation. A command can carry multiple segments up to the driver's per-command byte limit. Each internal runtime slot holds up to 64 commands; larger command lists use multiple slots in sequence. This limit applies to runtime command descriptors, not to queued jobs.

For a two-page request across eight chips, the runtime creates at most 16 base commands if each chip/host-slot group fits one driver command. Thirty-two layers produce 512 layer/page/chip segments, but those remain grouped into the same 16 commands. Coalescing uses one staging allocation per chip, so the same request needs at most eight base commands under that size assumption. Larger groups may require additional commands and runtime slots when the total exceeds 64.

No NPU measurement establishes the end-to-end performance of these paths.

### Completion and failures

A `Copy` covers the whole request, including every DMA batch and staging step. Submission returns once the work is queued. Wait for its result before reusing the request's host slots or device pages. An overlapping device request raises `BlockingIOError` while the earlier copy is pending. A failed copy leaves its device pages reserved until the copier shuts down.

A view from `HostCache.access()` blocks DMA on its slot until the view and its derived views are released. The framework coordinates its host users and decides when a cache value can be reused. Release host views before calling `Copier.shutdown()`, which drains the queue and joins the workers.
