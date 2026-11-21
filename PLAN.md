# Model-neutral KV cache and L3 offload plan

## Objective and controlling decisions

Extend Furiosa's vLLM 0.31.0 and SGLang 0.5.22 integrations with filesystem-backed L3 for supported causal `Paged` attention, while preserving each framework's storage identity and lifecycle. Share page routing, host DMA submission, failure draining, and byte-layout copies in `furiosa_opt_engine.cache_transfer`; keep vLLM and SGLang adapters focused on framework-owned scheduling and filesystem interfaces. NIXL remains PR #56, stacked on this PR after the common core lands. Keep cache-kind support end-to-end: explicitly reject `State` and `Cross`; encoder-decoder serving remains separate.

The user has confirmed these decisions:

| Area | Decision | Evidence and effect |
|---|---|---|
| Target vLLM | Pin behavior to vLLM 0.31.0, the current public integration pin. | `integrations/furiosa-opt-vllm/pyproject.toml`; use the pinned API, not newer upstream behavior. |
| L3 policy | Follow vLLM filesystem-tier policy: `FileMapper` identity and namespace, config directory, hash sharing, unbounded `max_bytes`, and operator-managed stale-directory cleanup. | vLLM 0.31.0 source; no plugin LRU, byte cap, process-local namespace, or automatic cleanup. |
| SGLang filesystem policy | Reuse SGLang 0.5.22 `HiCacheFile` for `--hicache-storage-backend file`; pass its parsed extra config and preserve its key namespace and eviction policy. | The engine adapter supplies flat opaque pages to SGLang's storage API; it adds no competing file manager or cache identity. |
| Shared host core | Keep page-run routing, engine host DMA submission/failure draining, and layout-aware host byte copies framework-neutral in `furiosa_opt_engine.cache_transfer`. | vLLM and SGLang use the same primitives; scheduler events, key construction, and filesystem lifecycle remain with their framework adapters. |
| PR stack | PR #57 contains filesystem support. PR #56 contains NIXL and is rebased onto #57; NIXL-specific lifecycle stays in #56. | User instruction. Build and review #57's shared core before updating #56's base. |
| Tier manager | Reuse vLLM `TieringOffloadingManager`, `TieringOffloadingSpec`, and `FileSystemTierManager` over `SharedOffloadRegion`. | Pinned vLLM 0.31 creates the primary mmap in the scheduler process and separately maps it in the worker; `SecondaryTierManager` runs in the scheduler process. Engine `HostBuffer` slots belong to the worker, so direct secondary-tier access cannot satisfy this process contract. The stock filesystem tier already probes `O_DIRECT` and falls back to buffered I/O when unsupported. |
| Cache groups and kinds | Implement end-to-end support for `Paged` full/sliding-attention groups. Explicitly reject `State` and `Cross`; do not add storage-only kind support. | User retained `Paged`-only scope after learning that existing cross-attention modules are not connected to engine/vLLM encoder-decoder serving. |
| Physical layout | Validate all six vLLM `KVCacheLayout` permutations in vLLM's tensor metadata path; keep the engine's model-defined per-layer cache format. | `FuriosaModelRunner` replaces vLLM attention and allocates `engine.decoder.Cache`; vLLM's layer cache callbacks are no-ops. The selected layout describes vLLM tensors that do not carry live model bytes. `paging.initialize` validates upstream legality and tensor strides. |
| Generic metadata | Extend engine cache metadata when its current shape, grouping, page-size, or layout information is insufficient. | User decision. Keep metadata model-neutral and avoid vLLM types in Rust engine APIs. |
| Canonical layout | Include `canonical_layout`. Follow vLLM 0.31.0's `derive_canonical_mappings` and exact-tiling verification. Reuse across parallelism only when vLLM certifies it; otherwise use opaque per-rank bytes and a topology-specific namespace. | User decision; vLLM provides this certification/fallback path. |
| Per-layer offset cap | Keep signed-I32 offsets relative to independent per-layer HBM buffers, with each layer's addressable span at most `2^31` bytes. Revisit wider offsets only if a supported per-layer configuration demonstrates the limit is insufficient. | User decision. Current `furiosa-opt-std` indirect gather/scatter accepts I32 byte offsets; the engine allocates one buffer per layer. This is separate from ATU's 4 GiB peer-address translation window. |
| Validation | Add focused tests and run available local validation. Do not include an RNGD performance benchmark. | User decision. |
| TDMA | Separate documentation/research workstream; retain current 2 GiB per-layer I32 limit for now and make no TDMA code change. Revisit only when a supported configuration requires it. | `Cache::sizes` I32 byte-offset check is per layer; current allocations use chip-local `HbmBuffer::alloc`. The 4 GiB ATU entry maps peer DRAM and is not this limit. |
| Worktree and handoff | Put enough source, design, setup, and launch detail in `/root/furiosa-opt-models-cache.txt` and `/root/furiosa-opt-models-tdma.txt` for a fresh agent to begin without this conversation. Use a new worktree for implementation; leave public `main` untouched. | User instruction. Proposed worktree `/root/Works/furiosa-opt-models-cache`, branch `codex/cache-l3`, base `c982976100d18d7a9641d5a8c22b89cec4544049`. |

## Scope

The cache/L3 work changes engine cache description/allocation/views and two thin serving-framework adapters over one host-transfer/page-routing core. vLLM supports its legal full/sliding `Paged` groups and pinned layouts; SGLang supports its currently served single full-attention group. vLLM uses its stock shared-region filesystem tier; SGLang uses native `HiCacheFile`. NIXL is not implemented in #57. Do not add recurrent-state or encoder-decoder serving, change the 2 GiB TDMA decision, replace framework cache policies, or add RNGD performance acceptance criteria. `State` and `Cross` remain explicitly unsupported.

The engine API currently has a single `pool` per group, a contiguous `Range` of layer indices, one page size per pool, and one `HbmBuffer` allocation per layer. Its `Kind` enum describes `Paged`, `State`, and `Cross`, and the Python binding exposes all three, but `Batch::new` rejects every non-`Paged` kind before planning a call. `furiosa-opt-models` contains reusable T5/Whisper encoder-decoder compositions and cross-attention cache operations. They are not registered as a served engine model: `engine::Model` is `Decoder` or standalone `Encoder`, and the public vLLM runner is `Decoder`/`Media`; `admission.engine_cache()` rejects `is_encoder_decoder`. User decided not to add that serving feature to this L3 project. The public vLLM adapter currently constructs only full-attention specs, validates one group shape at the worker, and transfers only two compact layouts.

## Evidence and pinned interfaces

Use vLLM tag 0.31.0 as the pinned upstream artifact. Relevant files in that artifact:

- `vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py`: `get_offloading_group_ids`, `build_offloading_config`, group token spans, parallelism metadata, and canonical-layout static eligibility.
- `vllm/distributed/kv_transfer/kv_connector/v1/offloading/canonical_mapping.py`: per-layer map derivation, opaque fallback, and `_verify_tiling` exact-partition assertions.
- `vllm/v1/kv_offload/tiering/spec.py`, `tiering/manager.py`, `tiering/fs/manager.py`, `tiering/base.py`, and `file_mapper.py`: primary/secondary lifecycle, async job semantics, cache identity, and filesystem I/O contract.
- `vllm/v1/kv_offload/cpu/shared_offload_region.py`: cross-worker shared CPU primary storage and row layout.
- `vllm/v1/kv_cache_layout.py` and `vllm/v1/core/kv_cache_utils.py`: six `[L,B,H,N,C]` permutations, physical strides, and which mixed-page-size combinations are legal.

Relevant public repository sources at base `c982976100d18d7a9641d5a8c22b89cec4544049`:

- `crates/furiosa-opt-engine/src/decoder/cache.rs`: current `CacheSpec`, `Pool`, `Group`, page allocation, per-layer I32 offset cap, and `Cache::layers`.
- `crates/furiosa-opt-engine/src/decoder/batch/mod.rs` and `batch/operands.rs`: group semantics and page-table byte-offset assumptions.
- `crates/furiosa-opt-models/src/tensor/cache.rs`, `ops/attention/gqa/paged.rs`, and `ops/attention/gqa/mod.rs`: typed per-layer cache views, block stride, gather, and address generation.
- `crates/furiosa-opt-python/src/decoder.rs` and `tensor.rs`: Python cache API and `HbmBuffer` / `HostBuffer` memory behavior.
- `integrations/furiosa-opt-vllm/furiosa_opt_vllm/{cache,paging,model_runner,worker}.py` and `offload/{__init__,worker}.py`: current model-spec adapter, layout advertisement, cache initialization, and one-group transfer assumptions.

Prior internal implementation evidence:

- Internal PR #322 (`nemo/measured-snapshot`, head `21a3743fabbe7964b7d6f06b9fe6a228732f690d`) is a closed reference snapshot, not a merge candidate. Its Mistral-Nemo serving run used a disk tier on a 440-GB NVMe directory. Commit `90f18bf63554a1984c1345ce782efc9d1cd4e92d` implements a vLLM `SecondaryTierManager` that reads engine `HostCache` slot addresses directly and uses page-aligned bounce buffers for `O_DIRECT`, avoiding a copy into a separate shared-primary region. This proves that the direct slot path can work in that serving configuration; the old plugin also requires one cache group and `block_size_factor == 1`, and its vLLM 0.24.0 integration, process-local directories, custom LRU/`max_bytes`, and stale-directory cleanup do not satisfy this PR's 0.31.0 scope or policy.
- Internal commit `0b8af1d66` generalized host offload across cache groups but implemented custom scheduler/storage policy and rejects `canonical_layout`. Reuse its transfer/layout lessons, not its divergent max-byte, process-local-directory, and cleanup policies.

## Architecture and data flow

Preserve each framework's cache policy and adapt only cache description, physical addressing, and Furiosa transfers. The engine package owns reusable host-transfer primitives; framework adapters own request identifiers, tier scheduling, cache keys, and filesystem policy.

The shared host-transfer/page-routing core lives in `crates/furiosa-opt-engine/python/furiosa_opt_engine/cache_transfer.py`. It groups adjacent page copies, submits per-layer DMA transfers, drains earlier work if submission fails, and gathers/scatters certified page-byte routes. It imports no serving framework and owns no storage lifecycle. vLLM canonical mappings translate into the core's generic byte routes; SGLang file pages flatten and restore engine host buffers through the same byte-copy primitive.

1. **Describe cache layers without changing vLLM's block-table grouping.** Treat each vLLM block-table group as the logical request-cache unit. Map it to one or more engine storage groups when `UniformTypeKVCacheSpecs` has different per-layer page sizes; each layer descriptor names its engine pool/page format, logical shape, element width, and strides. Every engine storage group mapped to one vLLM group uses the same token block size and full/sliding attention semantics. Allow non-contiguous layers. At the adapter boundary, reuse that vLLM group's block IDs for each mapped engine storage group; do not invent separate vLLM block tables. Validate complete, unique layer coverage and reject `State`/`Cross` before allocation or offload setup.
2. **Keep engine payload layout independent from vLLM tensor layout.** The custom runner executes the engine decoder over `engine.decoder.Cache`; it does not register or access vLLM attention-layer payload tensors. `save_kv_layer` and `wait_for_layer_load` are no-ops, and offload copies engine `cache.layers`. Keep each engine layer in its model-defined `PageFormat`; validate vLLM's selected global layout and `KVCacheTensor` strides through pinned upstream rules without transposing engine pages or changing model views. Canonical mapping metadata describes the actual engine page strides.
3. **Keep addresses per layer.** Engine block tables contain byte offsets relative to the owning layer's HBM base. Page `b` starts at `b * engine_page_bytes`; the aggregate vLLM `block_stride` never scales engine offsets. Preserve the existing gather/scatter hot path. vLLM's six legal layouts affect only its metadata validation and do not add engine copies, transposes, or per-head address tables. Use upstream mixed-page-size rules and reject combinations vLLM cannot express.
4. **Describe every supported eligible group to vLLM.** `paging.describe` emits a per-layer vLLM `FullAttentionSpec` or `SlidingWindowSpec` from engine metadata. For supported per-layer heterogeneous specs, construct `UniformTypeKVCacheSpecs`. `paging.initialize` validates the exact layer set, pool/page sizes, group order, block size, and physical placement, and fails clearly for `State`/`Cross`. Select only `get_offloading_group_ids(kv_cache_config)` groups, preserving vLLM prefix-cache eligibility and transfer policy.
5. **Route host and disk operations through vLLM tiering.** Build the pinned vLLM `OffloadingConfig`, preserve `OffloadingConnectorScheduler`, and use a Furiosa `TieringOffloadingSpec` worker adapter with vLLM `TieringOffloadingManager`. The worker maps vLLM `GPULoadStoreSpec` group sizes/block indices to engine groups/layer views and copies only the selected blocks. Host rows follow vLLM's per-chunk aggregate layout. Async lifetime, store fences, partial-load reporting, request completion, and shutdown follow vLLM's worker/tier interfaces.
6. **Keep one authoritative host copy across the process boundary.** Pinned vLLM 0.31 runs `TieringOffloadingManager` and `SecondaryTierManager` in the scheduler process; `SharedOffloadRegion` is the primary storage contract, separately mapped by the worker. Engine `HostCache` slots are worker-owned and cannot be passed directly to the scheduler-side tier manager. Copy engine transfers into the shared mmap, which is the single host copy used by disk tiering. Do not add a separate persistent host cache.
7. **Reuse vLLM's filesystem tier and compatibility behavior.** Use stock vLLM `FileSystemTierManager` over `SharedOffloadRegion`, preserving vLLM `FileMapper`, `OffloadKey`, `blocks_per_chunk`, lookup/job lifecycle, and namespace semantics. The manager probes `O_DIRECT` and falls back to buffered I/O when unsupported. Test persistence, async failures, and lifetime through the pinned tier API. Do not copy PR #322's process-local directories, custom LRU/`max_bytes`, or stale-directory cleanup; vLLM 0.31.0 policy remains fixed.
8. **Derive canonical transfer maps from engine metadata.** Create metadata-only `torch` views (shape/stride/dtype, no KV contents) matching each engine layer's physical page view and call vLLM 0.31.0 `derive_canonical_mappings`. Its `_verify_tiling` must certify local coverage and canonical coverage. Use certified copy runs for compatible attention layouts. For unsupported/opaque pages, use rank-exclusive opaque bytes and set the offloading config non-portable so `FileMapper` retains topology/rank in its namespace. A missing/ambiguous page descriptor fails closed; never persist unverified bytes under a cross-parallel namespace.
9. **Preserve upstream persistence identity.** Pass vLLM's `root_dir`, `tokens_per_hash`, `blocks_per_file`/`blocks_per_chunk`, model, dtype, cache groups, engine, parallelism fields, canonical format, and config writing unchanged. Do not add `max_bytes`, LRU, process-only directory naming, or automatic stale directory removal. Operators remove old directories.
10. **Adapt SGLang's native filesystem tier.** Support `--hicache-storage-backend file` for the served full-attention cache. Pass SGLang's parsed `storage_backend_extra_config` into `HybridCacheController`; implement aligned flat page read/write over the engine's per-layer `HostBuffer`s. Preserve SGLang `HiCacheFile` key namespace, file path, and eviction policy. Reject other storage backends and draft pools before setup. File page assembly copies bytes only on the L3 path; model execution and host DMA retain engine page layout.
11. **Keep storage adapters separate from the common core.** vLLM's stock `FileSystemTierManager` and SGLang's `HiCacheFile` remain independent adapters. PR #56's NIXL connector is another adapter over the shared page-transfer core, added only after #57 is ready to serve as its base.

## TDMA disposition

No TDMA implementation change is planned. `CacheSpec::sizes` caps each layer's allocation at `1 << 31` because batch operands carry signed-I32 byte offsets. `paging.available` also caps the per-layer block count by `(2**31) // page_size_bytes`. Total cache over layers can exceed 2 GiB. Current `Cache::new` allocates chip-local `HbmBuffer::alloc`; `HbmBuffer::alloc_shared` is the peer-visible DRAM path, and pinned Furiosa lowering notes no inter-chip indirect DMA. The 4 GiB ATU entry is a peer DRAM translation window, not the per-layer local cache allocation. Keep this distinction in the TDMA handoff and revisit only if a supported layout/config exceeds the actual address representation.

The current 2 GiB per-layer cap protects a signed-I32 byte-offset contract. This design preserves separate engine layer buffers, so vLLM's global block-outer `block_stride` does not multiply an engine layer's base-relative offsets. Never silently wrap offsets or claim unsupported capacity; only revisit I32 if an actual supported per-layer cache exceeds the cap.

## Implementation task table

Every row names the concrete owner, action, dependency, and acceptance gate. “Owner” is the file/module boundary the implementation agent changes; pinned vLLM behavior remains governed by the exact upstream 0.31.0 artifact.

| ID | Owner | Action and dependency | Status / acceptance gate |
|---|---|---|---|
| C1 | `furiosa-opt-engine::decoder::CacheSpec` and Python stubs | Add only metadata needed for model-neutral `Paged` attention: stable vLLM logical group mapping, per-layer engine pool/page format, token block size, window, logical shape/byte strides, and non-contiguous layer membership. Allow one vLLM block-table group to map to multiple engine pool groups when page sizes differ; preserve one block-ID list at the vLLM boundary. Maintain Rust/Python parity; keep `State`/`Cross` unsupported by serving. | ready; tests reject malformed/duplicate/missing layers, inconsistent token block sizes, unsupported kinds, and invalid descriptors. |
| C2 | Engine `CacheSpec`/allocation, `tensor::cache::Blocks`, and `furiosa_opt_vllm.paging` | Keep the model-defined engine page view and one HBM allocation per layer. Validate all six vLLM layouts and actual `KVCacheTensor` strides independently. Keep block offsets base-relative to each engine layer and enforce the `2^31` byte bound. Depends on C1. | metadata tests cover all six upstream layouts; engine tests prove per-layer page bounds and no I32 wrap; canonical metadata views match actual engine page strides. No layout-specific copy or gather/scatter path. |
| C3 | `furiosa_opt_vllm.cache` / `paging` | Translate supported `Paged` full/sliding-attention layer specs into pinned vLLM groups; preserve legal `UniformTypeKVCacheSpecs`, map each vLLM group to its engine pool groups, and validate selected layout plus physical `KVCacheTensor` metadata. Reject `State`/`Cross`. Depends on C1/C2. | adapter tests cover heterogeneous pages, multiple groups, one vLLM block table feeding multiple engine pool groups, unsupported-kind errors, and all six legal vLLM metadata layouts. |
| C4 | `furiosa_opt_engine.cache_transfer` plus vLLM/SGLang host adapters | Centralize contiguous page routing, host DMA submission/draining, and byte-layout copies. Both adapters use these primitives while preserving their own scheduler contracts. Depends on C1-C3. | Tests cover adjacent and multi-chip pages, bidirectional page routing, canonical gather/scatter, and partial submission failure; no framework duplicates page-run or byte-copy mechanics. |
| C5 | Furiosa offloading spec adapter plus pinned vLLM tier API | Reuse `OffloadingConnectorScheduler` and `TieringOffloadingManager`; use the pinned shared primary region because scheduler-side secondary managers cannot access worker-owned engine buffers. Depends on C4. | shared-region requirement verified from pinned vLLM source; tests verify completion ordering, restart reuse, config namespace isolation, async completion, partial failure behavior, and operator-managed cleanup. |
| C6 | Canonical layout adapter (`pinned upstream artifact`) | Supply metadata views to `derive_canonical_mappings`, use vLLM's exact-tiling result, preserve canonical-format versioning, and force opaque topology-specific namespaces for uncertified mappings. Depends on C1-C5. | ready; tests prove certified TP reuse and that unsupported layouts never share cross-topology. |
| C7 | vLLM/SGLang integration tests and serving docs | Add host-only tests for both adapters and the common core; document supported cache kinds, file backend flags, and framework-owned storage behavior. | Local lint/format and focused host tests pass; NPU-only execution stays not run on machines without RNGD. |
| C8 | `integrations/furiosa-opt-sglang/furiosa_opt_sglang/offload.py` and `platform.py` | Enable only native `HiCacheFile`; pass extra config; expose aligned flat page views over engine host buffers; retain SGLang file identity/eviction. Depends on C4. | Host-only round-trip and CLI acceptance tests cover file pages/config; non-file backends and draft pools fail clearly. |
| T1 | `crates/furiosa-opt-engine/src/decoder/cache.rs` and TDMA handoff | Record evidence for the per-layer I32 limit and ATU separation; make no TDMA code change. Revisit only if a supported layout/config needs wider peer/local addressing. | ready; `/root/furiosa-opt-models-tdma.txt` states the no-change decision and evidence-based trigger. |
| H1 | `/root/furiosa-opt-models-cache.txt` and worktree setup | Include decisions, source paths, exact worktree path/branch/base, setup commands, first-task order, validation, explicit State/Cross rejection, no-main-edit rule, and PR target. | ready; a fresh agent can start using only the handoff. |

## Worktree and agent handoff

The active implementation worktree is `/root/Works/wt-fom-cache`, branch `codex/cache-l3`; PR #57 is already a draft. Keep `/root/Works/furiosa-opt-models` clean on `main`; all #57 edits stay in the active worktree. After #57 implementation and review, push it, rebase PR #56's NIXL branch onto this branch, and change #56's base to #57. Do not merge either PR. TDMA remains no-change per its separate handoff.

Fresh-agent first steps in the handoff:

1. Read `PLAN.md` and the cache handoff; verify branch/base/worktree and inspect current dirty files.
2. Read the pinned vLLM 0.31.0 sources listed above and the public cache files listed above. Read internal commits `90f18bf...` and `0b8af1d66` only for prior-art constraints.
3. Implement in dependency order C1 → C2 → C3 → C4 → C5/C6/C8 → C7. Preserve stock vLLM and SGLang storage behavior.
4. Add focused tests for each layer and run repository local formatting/lint/test commands applicable to changed Rust/Python packages. RNGD performance benchmark is excluded.
5. Review #57, then update PR #56 as a stacked NIXL adapter change; do not edit or commit in the clean main checkout.

## Interview status

The implementation interview is complete. All cache scope, layout, offset, L3 policy, and TDMA decisions are recorded above; there are no open questions.

## Risks, acceptance, and rollout

The primary correctness risk is confusing vLLM metadata strides with engine page strides. Validate `KVCacheTensor` using pinned upstream rules, but compute engine offsets from each model-defined per-layer page size and base. Never use aggregate vLLM `block_stride` for engine addresses; check all offset arithmetic before allocation.

The second risk is tier divergence: a disk tier may report a block present only after its store is durable and the scheduler's key is published. Use vLLM tier job lifecycle, transfer completion fences, and partial-load semantics over scheduler-owned `SharedOffloadRegion`. Filesystem cache identity must include upstream config/canonical-format fields exactly.

The third risk is overclaiming generality. Validate all six vLLM layouts and tensor strides with pinned upstream rules, while keeping engine page offsets tied to model-defined per-layer storage. Enforce vLLM's mixed-page-size restrictions. For canonical reuse, only upstream-certified maps share cross-parallel data; all others stay topology-specific.

Acceptance requires: all six legal vLLM layouts pass upstream legality and `KVCacheTensor` stride validation; engine cache pages retain model-defined format and per-layer base-relative I32 offsets; vLLM groups route to correct engine pages; SGLang file pages round-trip the aligned opaque engine host bytes; `State`/`Cross` fail clearly; both frameworks use the shared host-transfer/page-routing core and retain native filesystem policy; canonical layout invokes vLLM mapping and exact-tiling checks with topology-specific fallback; startup rejects invalid/overflowing plans without wrapping; focused tests and available local validations pass; docs match runtime behavior. Prove `paging.available` reports the 2 GiB per-layer limit. No RNGD performance benchmark is part of this PR.

## Final plan audit

The interview and handoff documents are complete. This plan-only checkpoint is committed in the clean designated worktree; implementation may begin from this point.

| Criterion | Status | Evidence |
|---|---|---|
| MECE | PASS | C1-C7 divide metadata, per-layer storage/view, vLLM adapter, transfer worker, filesystem tier, canonical mapping, and validation/docs; T1 isolates TDMA disposition. |
| Deduplication | PASS | User decisions appear once in the decision table; architecture and acceptance refer to their operational consequences. |
| Topic sentences | PASS | Major sections begin with the design decision governing their contents. |
| Top-down structure | PASS | Objective → evidence → architecture → TDMA boundary → tasks → handoff → risks → acceptance. |
| Feasibility | PASS | Backend selection has a compatibility gate; local CPU tests and RNGD-only checks are separated; C2 preserves current per-layer offset API. |
| Fidelity | PASS | vLLM behavior is pinned to 0.31.0; no vISA/vSFR change is planned; I32 is used only as the existing base-relative per-layer gather/scatter contract. |
| **Overall** | **PASS** | Interview decisions are resolved, handoffs are complete, and the plan-only checkpoint is committed in the clean implementation worktree. |
