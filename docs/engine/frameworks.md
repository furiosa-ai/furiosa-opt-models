# Integrating frameworks

A plugin maps framework scheduling and cache policy onto `opt-engine`, which runs model work and owns the shared KV transfer path.
This page defines that boundary for vLLM and SGLang: request mapping, cache allocation ownership and filesystem offload mechanics. [Using the engine](python.md) documents the Python API; the [serving guide](../serving/index.md) gives launch recipes.

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

Frameworks keep cache selection, keys, eviction and tier lifecycle; the engine owns pinned host slots, page routing, DMA and aligned file I/O. This section is the source for those shared mechanics. The [serving guide](../serving/index.md#offload-the-kv-cache-to-host-memory) has launch recipes; the [vLLM](../serving/vllm.md#kv-cache-offloading) and [SGLang](../serving/sglang.md#hierarchical-cache) references define each adapter's options and framework-specific behavior.

### Filesystem layout and I/O

The file tier preserves each framework's existing byte layout.

| Framework | On-disk layout |
| --- | --- |
| SGLang | One raw cache page per file. |
| vLLM | Native layer-major chunks, including trailing chunk-alignment padding. |

`AlignedFileIO` enables `O_DIRECT` only after querying the filesystem's memory and offset alignments and completing an aligned probe. An offset or vector-length misalignment uses buffered I/O from pinned slots. A single buffer with a misaligned address uses a per-thread aligned buffer for direct I/O; for vectored I/O, one misaligned address switches the whole vector to buffered I/O. vLLM submits each chunk's layer ranges in one file-I/O batch, reusing at most one direct and one buffered file handle. The engine splits vectors only at the operating system's vector limit.

### DMA batches and ownership

Each `Copier` owns a six-worker queue with no job-count cap. An ordinary DMA batch contains up to two pages with distinct host slots; each page carries all requested layer routes. Slot uniqueness applies within a batch, so a later batch may reuse a slot. Pages are ordered by host slot within a batch to keep concurrent lock acquisition in one order.

The ordinary path moves KV data directly between pinned host slots and existing HBM pages, without a CPU staging copy. A device-page overlap with an in-flight copy fails with `BlockingIOError`; conflict checks read atomic copy status instead of locking each completion result. A host-slot conflict blocks on that slot's gate. A live host-slot lease prevents DMA until Python or filesystem access ends, while unrelated transfers proceed independently.

For separately allocated slots, one per-slot mutex owns the pinned buffer and its DMA/lease state. The direct path holds that mutex through DMA, so the buffer needs no second mutex. Each queue worker also owns a clone of the same PDMA device handle. It can build and submit transfers without taking the decoder's context mutex; the clone shares the opened device and memory rather than opening another device.

A pool backed by one contiguous allocation uses the same direct route over per-slot byte ranges. Each slot gate remains active through DMA. The engine holds the allocation-wide mutex only while building borrowed runtime views and submitting the request, then releases it before awaiting DMA. This serializes setup for transfers using that backing, while submitted DMA can still overlap. The filesystem L3 adapters use separately allocated slots.

### Optional coalescing

Coalescing is a Furiosa opt-in that stages one copy's pages in pinned memory before DMA. The engine allocates one pinned buffer per chip for the full page batch, merges contiguous layer ranges, and scatters stores back to their slots. Each request owns its stages until completion, and coalesced requests have no separate admission cap, so transient pinned-memory use scales with concurrent staged bytes. This can reduce DMA commands at the cost of CPU gather/scatter and pinned memory. The vLLM and SGLang references document each adapter's coalescing scope.

For separately allocated slots, coalescing holds the per-slot mutex only while gathering a load into staging or scattering a completed store. It releases the mutex before staging DMA because DMA touches staging buffers, not host slots. A load has already snapshotted its host bytes; a store publishes host bytes only after DMA completes. The slot gate does not reserve a cache value for the full staged transfer, so callers must await completion before reusing that value. The filesystem L3 adapters use this separately allocated path.

With a shared backing, coalescing holds the per-slot gate and allocation-wide backing mutex only while gathering or scattering one slot. Both are released before staging DMA. Transfers using that allocation can contend during CPU staging.

### Runtime command batching

On the channel-DMA path, the runtime groups layer/page segments into DMA-BUF commands by chip and host allocation. A command can carry multiple segments up to the driver's per-command byte limit. Each internal runtime slot holds up to 64 commands; larger command lists use multiple slots in sequence. This limit applies to runtime command descriptors, not to queued jobs.

For a two-page request across eight chips, the runtime creates at most 16 base commands if each chip/host-slot group fits one driver command. Thirty-two layers produce 512 layer/page/chip segments, but those remain grouped into the same 16 commands. Coalescing uses one staging allocation per chip, so the same request needs at most eight base commands under that size assumption. Larger groups may require additional commands and runtime slots when the total exceeds 64.

No NPU measurement establishes the end-to-end performance of these paths.

### Completion and failures

`copy_in` and `copy_out` return after queueing their chunks, without waiting for runtime submission or DMA. Callers must retain and await a `Copy` before reusing its host slots or device pages; this gives the caller transfer ordering and errors as well as safe cache reuse. A failed copy leaves every routed device page reserved for the copier's lifetime. Host-slot reuse remains governed by its gate and the caller's cache policy. Each framework adapter reports failures through its native completion path.

`Copier.shutdown()` drains submitted copies and joins the workers while releasing the Python GIL. Callers must release every host-slot access lease before shutdown, because an active lease blocks DMA until the access ends.

## SGLang specifics

SGLang's scheduler and checkpoint conventions require adapter-specific handling beyond request mapping.

The plugin loads the engine from the original checkpoint config with the user's overrides because SGLang rewrites some geometry for its own kernels.

With overlap enabled, SGLang schedules the next step and processes the previous step while the device runs. The plugin disables overlap for recurrent state and embeddings because they are read back within their step.

Mixed chunks interleave prompt chunks with running requests' decode rows in one step.

The full step logic is in [`generation/sequence.py` of vLLM](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/integrations/furiosa-opt-vllm/furiosa_opt_vllm/generation/sequence.py) and [of SGLang](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/integrations/furiosa-opt-sglang/furiosa_opt_sglang/generation/sequence.py).

### Mooncake Store upgrades

SGLang storage depends on private Store hooks and HiCache host-pool ownership; revalidate these boundaries after a SGLang upgrade.

- Review `FuriosaHiCacheMooncake` in `integrations/furiosa-opt-sglang/furiosa_opt_sglang/mooncake.py`.
- Check `_load_config`, `_tag_keys`, and `_batch_exist` signatures, initialization order, key semantics, and return values.
- Recheck `register_mem_pool_host`, `register_mem_host_pool_v2`, `batch_exists`, `batch_exists_v2`, `batch_get_v1/v2`, and `batch_set_v1/v2`, including `PoolName`, `PoolTransfer`, and `PoolTransferResult` shapes.
- Revalidate `_FuriosaHostPool` ownership and fields: `host_cache`, `_page_index`, `page_num`, `page_size`, `dtype`, `layer_num`, `start_layer`, `end_layer`, `chips`, and `sizes`.
- Check that `offload.py` selects the backend through SGLang's dynamic storage configuration and that `platform.py` accepts `mooncake`.
- Recheck the host-node lock through storage-write acknowledgment, synchronous backup writes, and prefetch completion before host-index release in `srt/managers/cache_controller.py` and `srt/mem_cache/unified_radix_cache.py`.

## vLLM specifics

vLLM storage uses its native Store scheduler and connector lifecycle; revalidate those upstream boundaries after a vLLM upgrade.

- Review `FuriosaMooncakeStoreConnector` and `FuriosaMooncakeStoreWorker` in `integrations/furiosa-opt-vllm/furiosa_opt_vllm/mooncake_store.py`.
- Revalidate the inherited `MooncakeStoreScheduler` and `MooncakeStoreConnector` lifecycle: lookup, block pinning, completion metadata, request retirement, events, and shutdown.
- Revalidate `ChunkedTokenDatabase`, `ReqMeta`, coordinator masks, Store layout, and projection of cache groups into request metadata.
- Recheck engine group mapping, page byte counts, one slab registration, bounded page leases, and uncertain-I/O retention. Keep supported cache groups and namespace inputs aligned with the [vLLM serving contract](../serving/vllm.md#shared-mooncake-store-prefix-cache).
- The connector replaces upstream's receiver; preserve its disk GET budget, usable-ratio, batch splitting, and oversized-key failure behavior.
- Recheck `MooncakeStoreConfig`, `setup_mooncake_store`, `ReplicateConfig`, Store disk-budget configuration, and `VLLM_MOONCAKE_LOAD_RECV_THREADS`.

## Shared Mooncake storage compatibility

The adapters share engine page-lifetime and result-validation rules while retaining their frameworks' native Store lifecycles.

### Dependency and version checks

Each plugin manifest owns its framework dependency and optional Mooncake range. The Mooncake contract workflow in `.github/workflows/test.yml` checks that both Mooncake ranges match and tests their minimum and newest allowed stable versions on RNGD.

### Shared pointer and page-format contracts

These invariants apply to both adapters and are implemented once in the shared engine modules.

- `HostRegionRegistrar` retains a `HostSlot` allocation while its address is registered. Successful unregistration releases it; failed unregistration keeps the registrar, Store owner, and allocations rooted until process exit.
- `MooncakeIOGuard` retains page leases and Store state when pointer-I/O completion is uncertain. Keep GET and PUT outcomes in the shared classifier; widen them only when the Mooncake contract proves pointer access has stopped, and update its tests.
- `ENGINE_PAGE_FORMAT_VERSION` in `crates/furiosa-opt-engine/python/furiosa_opt_engine/_mooncake.py` changes when serialized page bytes or byte order change. Framework cache identity is separate and is documented in each serving page.
- Existence responses must remain one integer `0` or `1` per key. Reject malformed responses before passing page pointers to a transfer call.
- The vLLM adapter uses `batch_is_exist`, `batch_get_into_multi_buffers`, and `batch_put_from_multi_buffers`. SGLang reaches `batch_is_exist`, `batch_get_into`, and `batch_put_from` through `MooncakeStore`.

### Transport and contract tests

The current RNGD contract suite establishes TCP storage only. It covers registration, PUT, GET, unregistration, and shutdown with real engine host memory; it does not establish RDMA support or measure performance.

| Framework | Upgrade contract tests |
| --- | --- |
| vLLM | `integrations/furiosa-opt-vllm/tests/test_mooncake.py`; `integrations/mooncake-contract-tests/test_vllm.py` |
| SGLang | `integrations/furiosa-opt-sglang/tests/test_cache_transfer.py`; `integrations/furiosa-opt-sglang/tests/test_mooncake.py`; `integrations/mooncake-contract-tests/test_sglang.py` |

Before documenting another storage transport, run the registration and pointer-I/O contracts on RNGD with that transport. Host fakes cannot validate DMA-heap registration. Never simulate RNGD kernels on CPU.
