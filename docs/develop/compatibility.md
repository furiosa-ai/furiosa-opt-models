# Mooncake compatibility and upgrades

Use this page to review adapter contracts after changing SGLang, vLLM, or Mooncake dependencies.
Serving pages own supported behavior and Store configuration; the [installation guide](../getting-started/installation.md)
owns package and system dependencies. Exact version ranges live in plugin manifests, which the
Mooncake contract workflow reads.

## Adapter paths

KV storage and disaggregated KV transfer use different framework lifecycles, buffers, and completion
contracts. Do not infer that one path works because the other does.

| Framework | KV storage | Disaggregated KV transfer | Same-server support |
| --- | --- | --- | --- |
| vLLM | `FuriosaMooncakeStoreConnector` reuses upstream Store scheduling and keys; its worker stages engine pages. | `MooncakeConnector` transfers request KV through staging buffers. | One direct KV connector per worker: Store, P2P transfer and `--kv-offloading-size` offload cannot be composed. |
| SGLang | HiCache uses `MooncakeStore` for pages. | `MooncakeKVManager` and `MooncakeKVReceiver` follow the disaggregation lifecycle. | Decode-side HiCache is excluded during prefill/decode disaggregation; prefill-side storage has not been established. |

## Dependency declarations

These manifests own the versions used by builds and tests; keep this guide free of copied version
values.

| Dependency | Source of truth |
| --- | --- |
| SGLang | `integrations/furiosa-opt-sglang/pyproject.toml` |
| vLLM | Both architecture-specific wheel URLs in `integrations/furiosa-opt-vllm/pyproject.toml` |
| Mooncake Transfer Engine | The optional `mooncake` extras in both plugin `pyproject.toml` files; keep their ranges equal. |

## Framework upgrade contracts

Review the adapter-specific framework APIs whenever a framework dependency changes. Storage and
transfer adapters have separate upgrade surfaces and require separate regression coverage.

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

SGLang transfer depends on separate disaggregation hooks. If that adapter is in the release, review
`FuriosaMooncakeKVManager` and `FuriosaMooncakeKVReceiver` in the transfer implementation:

- Check the manager's staging registration, buffer initialization, peer registration, cache
  attachment, and staged-transfer methods against SGLang's current lifecycle.
- Revalidate `_staging_ctx`, `kv_args`, peer-registration metadata, `engine.batch_register`, and
  `_transfer_data` contracts.
- Check manager selection in `hooks.py` and rerun transfer scheduling, registration, page-layout,
  and completion tests.

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

The direct vLLM P2P transfer connector is a separate upgrade surface. If it is in the release,
review `FuriosaMooncakeConnector` in `integrations/furiosa-opt-vllm/furiosa_opt_vllm/mooncake.py`:

- Check its `MooncakeConnector` base-class hooks, registration methods, load submission, transfer
  results, and shutdown behavior.
- Revalidate connector metadata (`reqs_to_send`, `reqs_to_recv`) and worker fields
  (`block_len_per_layer`, `kv_block_len_per_layer`).
- Keep P2P transfer tests separate from Store tests, even when both are in
  `tests/test_mooncake.py`.

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
change. `TransferBatch` tracks engine DMA completion and does not replace this Store-I/O guard.

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

The direct transfer adapters force TCP: vLLM sets `mooncake_protocol=tcp`, while SGLang requires
`MC_FORCE_TCP=1`. This setting addresses RDMA registration failures for engine-pinned DMA-heap
staging memory. Storage uses a different registration path: it passes engine host-page addresses to
`MooncakeDistributedStore.register_buffer`, which Mooncake documents for zero-copy RDMA. Therefore,
the transfer adapters' TCP restriction does not establish the storage backend's supported transport.

Before documenting another storage transport, run an RNGD check for registration, PUT, GET,
unregistration, and shutdown with a real engine host slot under that transport. Host fakes cannot
validate DMA-heap registration. The existing contract job starts an ephemeral local Mooncake master
over TCP and checks multi-page vLLM and multi-key SGLang I/O; it does not benchmark performance.

## Version-change verification

The required Mooncake contract workflow in `.github/workflows/test.yml` builds wheels and tests the
minimum declared Mooncake version plus the newest stable allowed 0.3.x release on RNGD. It derives
the minimum from both plugin manifests and runs one matrix entry if minimum and latest match.

| Framework | Contract workflow | Additional adapter regression tests |
| --- | --- | --- |
| vLLM | `integrations/furiosa-opt-vllm/tests/test_mooncake.py`; `integrations/mooncake-contract-tests/test_vllm.py` | `integrations/furiosa-opt-vllm/tests/test_offload.py`; `integrations/furiosa-opt-vllm/tests/test_vllm_platform.py` |
| SGLang | `integrations/furiosa-opt-sglang/tests/test_cache_transfer.py`; `integrations/furiosa-opt-sglang/tests/test_mooncake.py`; `integrations/mooncake-contract-tests/test_sglang.py` | `integrations/furiosa-opt-sglang/tests/test_platform.py` |

Run the additional regression tests when changing their framework integration; the contract workflow
does not invoke those files. Keep mocked adapter checks separate from RNGD registration and pointer
I/O checks. Never simulate RNGD kernels on CPU.
