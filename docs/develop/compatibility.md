# Mooncake compatibility and upgrades

Use this page to review Mooncake storage adapter contracts after changing SGLang, vLLM, or Mooncake
dependencies. It covers storage only; the [serving guide](../serving/index.md) owns the separate
KV-transfer support status. Serving pages own Store configuration, and the
[installation guide](../getting-started/installation.md) owns package and system dependencies.
Exact version ranges live in plugin manifests, which the Mooncake contract workflow reads.

## Adapter paths

The two storage adapters preserve their frameworks' native prefix-cache lifecycles.

| Framework | Storage integration | Upstream boundary |
| --- | --- | --- |
| vLLM | `FuriosaMooncakeStoreConnector` stages engine pages through its worker. | Reuses upstream Store scheduling and prefix-cache keys. |
| SGLang | HiCache uses `MooncakeStore` for engine pages. | Reuses upstream HiCache lookup and storage lifecycle. |

## Dependency declarations

These manifests own the versions used by builds and tests; keep this guide free of copied version
values.

| Dependency | Source of truth |
| --- | --- |
| SGLang | `integrations/furiosa-opt-sglang/pyproject.toml` |
| vLLM | Both architecture-specific wheel URLs in `integrations/furiosa-opt-vllm/pyproject.toml` |
| Mooncake Transfer Engine | The optional `mooncake` extras in both plugin `pyproject.toml` files; keep their ranges equal. |

## Framework upgrade contracts

Review the storage adapter's framework APIs whenever a framework dependency changes. This page does
not establish compatibility for disaggregated KV transfer.

### SGLang

SGLang storage depends on private `MooncakeStore` hooks and HiCache host-pool ownership details.

For storage, review `FuriosaHiCacheMooncake` in
`integrations/furiosa-opt-sglang/furiosa_opt_sglang/mooncake.py`:

- Check the private `_load_config`, `_tag_keys`, and `_batch_exist` hooks, including their signatures,
  initialization order, key semantics, and return values.
- Check the `register_mem_pool_host`, `register_mem_host_pool_v2`, `batch_exists`,
  `batch_exists_v2`, `batch_get_v1/v2`, and `batch_set_v1/v2` contracts, including the shapes of
  `PoolName`, `PoolTransfer`, and `PoolTransferResult`.
- Revalidate `_FuriosaHostPool` fields (`host_cache`, `_page_index`, `page_num`, `page_size`,
  `dtype`, `layer_num`, `start_layer`, `end_layer`, `chips`, and `sizes`), page-index mapping, and
  page byte counts.
- Check that `offload.py` still selects the backend through SGLang's dynamic storage configuration
  and that `platform.py` still accepts `mooncake`.

### vLLM

vLLM storage depends on upstream's native Store scheduler and key contract. Review
`FuriosaMooncakeStoreConnector` and `FuriosaMooncakeStoreWorker` in
`integrations/furiosa-opt-vllm/furiosa_opt_vllm/mooncake_store.py`:

- Revalidate the inherited `MooncakeStoreScheduler` and `MooncakeStoreConnector` lifecycle,
  including lookup, block pinning, completion metadata, request retirement, events and shutdown.
- Revalidate the worker's `ChunkedTokenDatabase`, `ReqMeta`, coordinator masks and Store layout;
  vLLM's cache groups are projected to the prefix-cacheable groups in request metadata.
- Recheck engine cache group mapping, page byte counts, one host slab registration, bounded page
  leases, and the uncertain-I/O retention path.
- Preserve the namespace over engine page format, vLLM cache specs, dtype, layout and topology.
  The checkpoint revision remains the operator's `cache_prefix` responsibility, matching upstream.
- Keep Paged full/sliding support explicit; the adapter rejects `State` and `Cross` cache groups.
- The adapter replaces upstream's receiver, so retain upstream's Mooncake disk GET budget,
  usable-ratio, batch-splitting and oversized-key failure behavior.

## Shared storage safety contracts

The framework must keep each engine block assigned until the native Store job stops reading it.
The upstream `MooncakeStoreScheduler` pins its save-job blocks and releases them after worker
completion metadata arrives; recheck this ordering in `store/scheduler.py` after an upgrade.

- The pinned SGLang release holds a host-node lock through storage-write acknowledgment. Its backup worker calls
  storage synchronously; prefetch reads also finish in the I/O thread before the scheduler releases
  host indices. Recheck `srt/managers/cache_controller.py` and
  `srt/mem_cache/unified_radix_cache.py` after an upgrade.
- The engine registrar retains a `HostSlot` allocation while its address is registered. Successful
  unregister releases it; failed unregister keeps the registrar, Store owner, and allocations rooted
  until process exit.

The engine page bytes and framework cache identity are separate parts of the stored key. The shared
`ENGINE_PAGE_FORMAT_VERSION` in `crates/furiosa-opt-engine/python/furiosa_opt_engine/_mooncake.py`
must change when serialized page bytes or byte order change; framework namespace behavior is
described in the [vLLM](../serving/vllm.md#shared-mooncake-store-prefix-cache) and
[SGLang](../serving/sglang.md#mooncake-storage) serving references.

`MooncakeIOGuard` retains page leases and Store state when I/O completion is uncertain. The accepted
GET/PUT outcomes are defined by the shared classifier in `_mooncake.py`; do not widen it unless the
new Mooncake contract proves pointer access has stopped, and update its failure tests with any
change.

## Mooncake package upgrades

Both plugin extras use the non-CUDA Mooncake Transfer Engine distribution and must keep equal
version ranges.

Review the package-level Store contract used by both storage adapters:

- Buffer registration, unregistration, and Store shutdown must preserve the registered host-page
  lifetime.
- vLLM uses `batch_is_exist`, `batch_get_into_multi_buffers`, and
  `batch_put_from_multi_buffers` from one contiguous staging slab; SGLang reaches
  `batch_is_exist`, `batch_get_into`, and
  `batch_put_from` through `MooncakeStore`.
- Existence responses must remain exactly one integer `0` or `1` per key. Any other response aborts
  before page pointers are passed to a transfer call.
- Recheck `MooncakeStoreConfig`, `setup_mooncake_store`, `ReplicateConfig`, the Store disk budget,
  and `VLLM_MOONCAKE_LOAD_RECV_THREADS` when changing the vLLM integration.

## Transport support

The current contract tests prove Mooncake storage over TCP only; they do not establish RDMA support.

The contract tests connect the Mooncake Store over TCP and cover registration, PUT, GET,
unregistration, and shutdown with real engine host memory on RNGD. They do not establish RDMA
support or measure performance. Before documenting another storage transport, repeat those checks
on RNGD with that transport; host fakes cannot validate DMA-heap registration.

## Version-change verification

The required Mooncake contract workflow in `.github/workflows/test.yml` builds wheels and tests the
minimum declared Mooncake version plus the newest stable allowed 0.3.x release on RNGD. It derives
the minimum from both plugin manifests and runs one matrix entry if minimum and latest match.

| Framework | Minimum/latest and RNGD contract tests |
| --- | --- |
| vLLM | `integrations/furiosa-opt-vllm/tests/test_mooncake.py`; `integrations/mooncake-contract-tests/test_vllm.py` |
| SGLang | `integrations/furiosa-opt-sglang/tests/test_cache_transfer.py`; `integrations/furiosa-opt-sglang/tests/test_mooncake.py`; `integrations/mooncake-contract-tests/test_sglang.py` |

Keep mocked adapter checks separate from RNGD registration and pointer-I/O checks. Never simulate
RNGD kernels on CPU.
