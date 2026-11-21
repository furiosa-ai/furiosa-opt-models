# Model-neutral KV cache and L3 offload plan

## Objective and controlling decisions

Extend the Furiosa vLLM integration so vLLM 0.31.0 can offload every supported causal `Paged` attention cache group and legal physical cache layout to a persistent filesystem tier, while leaving vLLM responsible for hash identity, scheduling, eviction, promotion, and retention policy. Keep cache-kind support end-to-end: explicitly reject `State` and `Cross`; encoder-decoder serving remains a separate feature. Produce independent cache/L3 and TDMA handoffs, then delegate only the cache/L3 implementation in a worktree based on the public repository's `main`.

The user has confirmed these decisions:

| Area | Decision | Evidence and effect |
|---|---|---|
| Target vLLM | Pin behavior to vLLM 0.31.0, the current public integration pin. | `integrations/furiosa-opt-vllm/pyproject.toml`; use the pinned API, not newer upstream behavior. |
| L3 policy | Follow vLLM filesystem-tier policy: `FileMapper` identity and namespace, config directory, hash sharing, unbounded `max_bytes`, and operator-managed stale-directory cleanup. | vLLM 0.31.0 source; no plugin LRU, byte cap, process-local namespace, or automatic cleanup. |
| Tier manager | Reuse vLLM `TieringOffloadingManager` / `TieringOffloadingSpec` where the Furiosa worker boundary permits; select stock `FileSystemTierManager` or a thin `SecondaryTierManager` I/O adapter only after compatibility checks. | Upstream primary storage is a shared CPU region; current Furiosa transfers use engine-owned buffers. Keep vLLM semantics while retaining Furiosa I/O where needed. |
| Cache groups and kinds | Implement end-to-end support for `Paged` full/sliding-attention groups. Explicitly reject `State` and `Cross`; do not add storage-only kind support. | User retained `Paged`-only scope after learning that existing cross-attention modules are not connected to engine/vLLM encoder-decoder serving. |
| Physical layout | Support all six vLLM `KVCacheLayout` permutations in the engine cache allocation/view path, not only `LBHNC` and `LBNHC`. | User decision after learning the remaining four require engine cache allocation and model-view changes. Validate combinations using vLLM's layout rules. |
| Generic metadata | Extend engine cache metadata when its current shape, grouping, page-size, or layout information is insufficient. | User decision. Keep metadata model-neutral and avoid vLLM types in Rust engine APIs. |
| Canonical layout | Include `canonical_layout`. Follow vLLM 0.31.0's `derive_canonical_mappings` and exact-tiling verification. Reuse across parallelism only when vLLM certifies it; otherwise use opaque per-rank bytes and a topology-specific namespace. | User decision; vLLM provides this certification/fallback path. |
| Per-layer offset cap | Keep signed-I32 offsets relative to independent per-layer HBM buffers, with each layer's addressable span at most `2^31` bytes. Revisit wider offsets only if a supported per-layer configuration demonstrates the limit is insufficient. | User decision. Current `furiosa-opt-std` indirect gather/scatter accepts I32 byte offsets; the engine allocates one buffer per layer. This is separate from ATU's 4 GiB peer-address translation window. |
| Validation | Add focused tests and run available local validation. Do not include an RNGD performance benchmark. | User decision. |
| TDMA | Separate documentation/research workstream; retain current 2 GiB per-layer I32 limit for now and make no TDMA code change. Revisit only when a supported configuration requires it. | `Cache::sizes` I32 byte-offset check is per layer; current allocations use chip-local `HbmBuffer::alloc`. The 4 GiB ATU entry maps peer DRAM and is not this limit. |
| Worktree and handoff | Put enough source, design, setup, and launch detail in `/root/furiosa-opt-models-cache.txt` and `/root/furiosa-opt-models-tdma.txt` for a fresh agent to begin without this conversation. Use a new worktree for implementation; leave public `main` untouched. | User instruction. Proposed worktree `/root/Works/furiosa-opt-models-cache`, branch `codex/cache-l3`, base `c982976100d18d7a9641d5a8c22b89cec4544049`. |

## Scope

The cache/L3 work changes engine cache description/allocation/views, the vLLM cache adapter, and the offloading worker/storage adapter. It does not add recurrent-state or encoder-decoder model serving, change the 2 GiB TDMA decision or vLLM's cache policy, or add RNGD performance acceptance criteria. Every advertised cache kind must work through description, allocation, model execution, host movement, and disk movement. The accepted scope is `Paged` full/sliding attention; `State` and `Cross` stay explicitly unsupported.

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

- Internal commit `90f18bf63554a1984c1345ce782efc9d1cd4e92d` used an engine-aware disk tier because the engine host slots were not the contiguous primary `memoryview` expected by vLLM's filesystem tier; it used page-aligned bounce buffers for `O_DIRECT`.
- Internal commit `0b8af1d66` generalized host offload across cache groups but implemented custom scheduler/storage policy and rejects `canonical_layout`. Reuse its transfer/layout lessons, not its divergent max-byte, process-local-directory, and cleanup policies.

## Architecture and data flow

Preserve vLLM's offloading policy path and adapt only cache description, physical addressing, and Furiosa transfers.

1. **Describe cache layers without changing vLLM's block-table grouping.** Treat each vLLM block-table group as the logical request-cache unit. Map it to one or more engine storage groups when `UniformTypeKVCacheSpecs` has different per-layer page sizes; each layer descriptor names its engine pool/page format, logical shape, element width, and strides. Every engine storage group mapped to one vLLM group uses the same token block size and full/sliding attention semantics. Allow non-contiguous layers. At the adapter boundary, reuse that vLLM group's block IDs for each mapped engine storage group; do not invent separate vLLM block tables. Validate complete, unique layer coverage and reject `State`/`Cross` before allocation or offload setup.
2. **Separate logical cache metadata from physical placement.** The model declares each layer's logical page. At `initialize_kv_cache`, the adapter combines that with vLLM's selected `KVCacheLayout`, its `layer_view_order`, and `KVCacheTensor` metadata. The global six `[L,B,H,N,C]` layouts reduce to three per-layer orders after removing `L`: `BHNC` (`LBHNC`, `BLHNC`, `BHLNC`), `BNHC` (`LBNHC`, `BLNHC`), and `HBNC` (`LHBNC`). Keep engine-owned per-layer HBM allocations and represent each selected layout through its per-layer view; retain the full global layout for vLLM legality and canonical mapping. Do not mirror vLLM's global layer/block packing into one shared engine buffer. This keeps device offsets relative to a layer's own base and preserves the I32 contract.
3. **Apply layout to reads and writes.** Extend the engine cache-view and page-table path so a logical page can be read/written under each of the six permutations. The table's block IDs remain group-local; derive byte addresses relative to the selected layer view. Preserve current fast paths for `LBHNC` and `LBNHC`. Use vLLM's `validate_kv_cache_layout` rules for mixed page sizes; do not claim layouts for combinations upstream rejects. Verify that all six global vLLM layouts map correctly through independent per-layer allocations and their views.
4. **Describe every supported eligible group to vLLM.** `paging.describe` emits a per-layer vLLM `FullAttentionSpec` or `SlidingWindowSpec` from engine metadata. For supported per-layer heterogeneous specs, construct `UniformTypeKVCacheSpecs`. `paging.initialize` validates the exact layer set, pool/page sizes, group order, block size, and physical placement, and fails clearly for `State`/`Cross`. Select only `get_offloading_group_ids(kv_cache_config)` groups, preserving vLLM prefix-cache eligibility and transfer policy.
5. **Route host and disk operations through vLLM tiering.** Build the pinned vLLM `OffloadingConfig`, preserve `OffloadingConnectorScheduler`, and use a Furiosa `TieringOffloadingSpec` worker adapter with vLLM `TieringOffloadingManager`. The worker maps vLLM `GPULoadStoreSpec` group sizes/block indices to engine groups/layer views and copies only the selected blocks. Host rows follow vLLM's per-chunk aggregate layout. Async lifetime, store fences, partial-load reporting, request completion, and shutdown follow vLLM's worker/tier interfaces.
6. **Use one primary byte store for host and disk tiers.** vLLM's scheduler tier and worker must read/write the same `SharedOffloadRegion`; the filesystem tier receives its upstream `memoryview`. Test whether the engine can transfer to/from this region directly. If not, keep an engine `HostBuffer` staging adapter and copy to/from the shared primary region with explicit transfer completion fences. Avoid a second persistent host cache whose contents diverge from the scheduler's chunk IDs.
7. **Choose the filesystem I/O implementation by compatibility.** First exercise stock vLLM `FileSystemTierManager` against the pinned shared-region view and configured filesystem (`O_DIRECT` probe, aligned block transfers, buffered fallback, async failures). Reuse it if its buffer and performance constraints fit Furiosa's primary. If engine-specific host storage still requires bounce buffers or file I/O, implement a thin `SecondaryTierManager` adapter using vLLM `FileMapper`, `OffloadKey`, `blocks_per_chunk`, lookup/job lifecycle, and namespace semantics. Do not copy the internal custom LRU/cap/cleanup policy.
8. **Derive canonical transfer maps from engine metadata.** Create metadata-only `torch` views (shape/stride/dtype, no KV contents) matching each engine layer's physical page view and call vLLM 0.31.0 `derive_canonical_mappings`. Its `_verify_tiling` must certify local coverage and canonical coverage. Use certified copy runs for compatible attention layouts. For unsupported/opaque pages, use rank-exclusive opaque bytes and set the offloading config non-portable so `FileMapper` retains topology/rank in its namespace. A missing/ambiguous page descriptor fails closed; never persist unverified bytes under a cross-parallel namespace.
9. **Preserve upstream persistence identity.** Pass vLLM's `root_dir`, `tokens_per_hash`, `blocks_per_file`/`blocks_per_chunk`, model, dtype, cache groups, engine, parallelism fields, canonical format, and config writing unchanged. Do not add `max_bytes`, LRU, process-only directory naming, or automatic stale directory removal. Operators remove old directories.

## TDMA disposition

No TDMA implementation change is planned. `CacheSpec::sizes` caps each layer's allocation at `1 << 31` because batch operands carry signed-I32 byte offsets. `paging.available` also caps the per-layer block count by `(2**31) // page_size_bytes`. Total cache over layers can exceed 2 GiB. Current `Cache::new` allocates chip-local `HbmBuffer::alloc`; `HbmBuffer::alloc_shared` is the peer-visible DRAM path, and pinned Furiosa lowering notes no inter-chip indirect DMA. The 4 GiB ATU entry is a peer DRAM translation window, not the per-layer local cache allocation. Keep this distinction in the TDMA handoff and revisit only if a supported layout/config exceeds the actual address representation.

The current 2 GiB per-layer cap protects a signed-I32 byte-offset contract. This design preserves separate engine layer buffers, so vLLM's global block-outer `block_stride` does not multiply an engine layer's base-relative offsets. Never silently wrap offsets or claim unsupported capacity; only revisit I32 if an actual supported per-layer cache exceeds the cap.

## Implementation task table

Every row names the concrete owner, action, dependency, and acceptance gate. “Owner” is the file/module boundary the implementation agent changes; pinned vLLM behavior remains governed by the exact upstream 0.31.0 artifact.

| ID | Owner | Action and dependency | Status / acceptance gate |
|---|---|---|---|
| C1 | `furiosa-opt-engine::decoder::CacheSpec` and Python stubs | Add only metadata needed for model-neutral `Paged` attention: stable vLLM logical group mapping, per-layer engine pool/page format, token block size, window, logical shape/byte strides, and non-contiguous layer membership. Allow one vLLM block-table group to map to multiple engine pool groups when page sizes differ; preserve one block-ID list at the vLLM boundary. Maintain Rust/Python parity; keep `State`/`Cross` unsupported by serving. | ready; tests reject malformed/duplicate/missing layers, inconsistent token block sizes, unsupported kinds, and invalid descriptors. |
| C2 | Engine cache allocator and `tensor::cache::Blocks` (`Tensor plan`) | Keep one HBM buffer per layer; expose per-layer cache views and page address generation for all six vLLM layouts, retaining current contiguous fast paths. Use base-relative I32 offsets and enforce the `2^31` byte per-layer bound. Depends on C1. | ready; synthetic values prove each logical page maps to expected bytes without alias/overlap, all offsets stay within the layer buffer, and boundary values cannot wrap. |
| C3 | `furiosa_opt_vllm.cache` / `paging` | Translate supported `Paged` full/sliding-attention layer specs into pinned vLLM groups; preserve legal `UniformTypeKVCacheSpecs`, map each vLLM group to its engine pool groups, honor exact vLLM eligibility, selected layout, and physical cache metadata. Reject `State`/`Cross`. Depends on C1/C2. | ready; adapter tests cover heterogeneous pages, multiple groups, one vLLM block table feeding multiple engine pool groups, unsupported-kind errors, and all six legal layouts. |
| C4 | `furiosa_opt_vllm.offload.worker` and `offload.__init__` | Replace one-group assumptions with explicit vLLM-group/engine-group/layer/pool mapping; implement Furiosa worker against shared primary storage for supported attention groups, all transfer directions, canonical copy runs, job completion/fences, and failures. One scheduled vLLM block ID must select the corresponding page in every engine pool group mapped to that logical group. Depends on C1-C3. | ready; unit tests cover multi-group block ranges, shared block IDs across heterogeneous engine pools, chunk alignment, copy lifetime, load/store and failure reporting. |
| C5 | Furiosa offloading spec adapter plus pinned vLLM tier API | Reuse `OffloadingConnectorScheduler`, `TieringOffloadingManager`, and stock `FileSystemTierManager` when compatible; otherwise make only an I/O adapter over vLLM `SecondaryTierManager`/`FileMapper`. Depends on C4 and filesystem compatibility gate. | ready; temp-filesystem tests verify persistence/restart reuse, config namespace isolation, async completion, partial failure behavior, and operator-managed cleanup. |
| C6 | Canonical layout adapter (`pinned upstream artifact`) | Supply metadata views to `derive_canonical_mappings`, use vLLM's exact-tiling result, preserve canonical-format versioning, and force opaque topology-specific namespaces for uncertified mappings. Depends on C1-C5. | ready; tests prove certified TP reuse and that unsupported layouts never share cross-topology. |
| C7 | `integrations/furiosa-opt-vllm` tests and docs | Update admission/docs that currently reject canonical layout or extra groups; add focused engine, adapter, and filesystem tests; run available local validation. | ready; no RNGD performance benchmark; document unsupported combinations explicitly. |
| T1 | `crates/furiosa-opt-engine/src/decoder/cache.rs` and TDMA handoff | Record evidence for the per-layer I32 limit and ATU separation; make no TDMA code change. Revisit only if a supported layout/config needs wider peer/local addressing. | ready; `/root/furiosa-opt-models-tdma.txt` states the no-change decision and evidence-based trigger. |
| H1 | `/root/furiosa-opt-models-cache.txt` and worktree setup | Include decisions, source paths, exact worktree path/branch/base, setup commands, first-task order, validation, explicit State/Cross rejection, no-main-edit rule, and PR target. | ready; a fresh agent can start using only the handoff. |

## Worktree and agent handoff

After the final audit, create `/root/Works/furiosa-opt-models-cache` from `/root/Works/furiosa-opt-models` at public `origin/main` commit `c982976100d18d7a9641d5a8c22b89cec4544049`, on branch `codex/cache-l3`. Leave `/root/Works/furiosa-opt-models` on `main` clean. Put this audited `PLAN.md` in the new worktree and make the one plan checkpoint commit there immediately before implementation. Then start one fresh implementation agent whose only task is C1-C7, with the cache handoff as complete context. Do not assign TDMA implementation; its handoff records the no-change disposition. Agent must keep all code/commits in the new worktree, add focused tests, run local gates, and open a PR targeting public `main`; no merge or publish action without user approval.

Fresh-agent first steps in the handoff:

1. Read `PLAN.md` and the cache handoff; verify branch/base/worktree and clean status.
2. Read the pinned vLLM 0.31.0 sources listed above and the public cache files listed above. Read internal commits `90f18bf...` and `0b8af1d66` only for prior-art constraints.
3. Implement in dependency order C1 → C2 → C3 → C4 → C5/C6 → C7. Keep the vLLM I/O compatibility gate explicit before committing to the backend.
4. Add focused tests for each layer and run repository local formatting/lint/test commands applicable to changed Rust/Python packages. RNGD performance benchmark is excluded.
5. Review the complete diff and create a PR to public `main`; do not work on or commit to the existing main checkout.

## Interview status

The implementation interview is complete. All cache scope, layout, offset, L3 policy, and TDMA decisions are recorded above; there are no open questions.

## Risks, acceptance, and rollout

The primary correctness risk is confusing logical page size with physical address stride. Every allocation descriptor must prove that each logical layer/page maps to the expected bytes exactly once; legal vLLM layouts may interleave blocks or layers. Reject malformed layouts before allocation and check all offset arithmetic.

The second risk is tier divergence: a disk tier may report a block present only after its store is durable and the scheduler's key is published. Use vLLM tier job lifecycle, transfer completion fences, and partial-load semantics. Filesystem cache identity must include upstream config/canonical-format fields exactly.

The third risk is overclaiming generality. Advertise a layout only when engine cache views and address generation support it; enforce vLLM's mixed-page-size restrictions. For canonical reuse, only upstream-certified maps share cross-parallel data; all others stay topology-specific.

Acceptance requires: all six legal layout permutations pass synthetic address/layout tests for supported `Paged` attention; all supported full/sliding groups map to correct vLLM group specs and transfer ranges; `State`/`Cross` fail with clear unsupported-kind diagnostics; vLLM filesystem tier policy is reused or preserved by a thin I/O adapter; canonical layout invokes vLLM mapping and exact-tiling checks with a topology-specific fallback; startup rejects invalid/overflowing plans without wrapping; focused tests and available local validations pass; docs match runtime behavior. Prove offsets are relative to each layer's base and `paging.available` reports the 2 GiB per-layer limit. No RNGD performance benchmark is part of this PR.

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
