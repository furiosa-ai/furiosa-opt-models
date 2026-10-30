# Testing principles

Guarantee each invariant at the highest level that can hold it, and write a test only when no higher level can.
To run the tests, see [Testing and benchmarking](../develop/testing.md).

| Level | Guarantee | Fails |
| --- | --- | --- |
| 1. Type system | Illegal states do not compile | At build |
| 2. Checked construction | Input is refined once | At the boundary |
| 3. Runtime check | Outside answers are verified on entry | At the boundary |
| 4. Test | A contract that no type or boundary check enforces | In CI |

A level is higher when it fails earlier and covers every caller.

## Level 1: type system

Listed by how much each removes.

| Tool | Example |
| --- | --- |
| Axis-typed shapes | `geometry::Hidden<const N: usize>` in [`ops/attention/mod.rs`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-models/src/ops/attention/mod.rs) |
| Const assert | `Live::groups` in [`tensor/live.rs`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-models/src/tensor/live.rs) |
| Phantom types | `Rows<D, Row>` in [`decoder/batch.rs`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-engine/src/decoder/batch/mod.rs) |
| Newtypes, enums | `Keep` in [`decoder/mod.rs`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-engine/src/decoder/mod.rs) |
| Checked constructors | `qwen3::Size<N>` from its `Config` in [`qwen3.rs`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-models/src/models/decoder/attention/qwen3/mod.rs) |
| Typestate | Call order as one type per state |

- Exhaustive `match` makes a new variant a compile error at every use.
- Runtime values need a checked constructor; constants need a const assert.

## Level 2: checked construction

Bytes, configs, framework arguments, driver answers and request fields are untrusted.
Parse each once into a refined type with `TryFrom` or a validating `new`, and name what was refused and why.

- Past the boundary, take the refined type and do not check again.
- Inside it, a violated invariant is a bug: `assert!` it.

## Level 3: runtime check

When no type can hold a value, check it where it enters:
`LoadOptions::parse` in [`load.rs`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-engine/src/load.rs) rejects a chip count that no model in the registry serves with `LoadError::Unsupported`; a level 4 test only pins that the check fires.

## Level 4: tests

A test covers only:

- A parser's accepted and refused inputs.
- A contract with an outside party: hardware, compiler, framework, checkpoint format.
- Numerical agreement with a reference.

### One contract per test

Test at the narrowest surface a caller can observe, named for the guarantee (`test_prefix_cache`).

| Contract | Surface |
| --- | --- |
| Device kernel | `launch` on a real device, in [`ops/pointwise/add.rs`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-models/src/ops/pointwise/add.rs) |
| vLLM behavior | `LLM.generate`, in [`test_generation.py`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/integrations/furiosa-opt-vllm/tests/test_generation.py) |
| SGLang behavior | HTTP endpoints, in [`test_server.py`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/integrations/furiosa-opt-sglang/tests/test_server.py) |
| Parallelism limits | Startup, in [`test_parallelism.py`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/integrations/furiosa-opt-vllm/tests/test_parallelism.py) |
| Cache transfers | Copy hooks, in [`test_cache.py`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/integrations/furiosa-opt-sglang/tests/test_cache.py) |

Use real hardware and the pinned framework version, because a mock shares the code's wrong assumptions.

### Where a test lives

A Rust unit test sits inline, in a `#[cfg(test)] mod tests { ... }` of the file that owns the behavior.
Do not add a `tests.rs`, a `tests/` directory or a `#[path]` module for it; a test file of its own is for an integration test that exercises a public boundary across modules or crates.

### Assert against an oracle

Never compare with a recording of our own output, or re-implement the code under test.
`adds_elementwise` computes the expected sum on the host in `f32`.

Oracles are Transformers on the same checkpoint, the same request through another path (batched vs single, cache hit vs miss), framework semantics, and the upstream framework's [reference outputs](../develop/testing.md#reference-outputs) when it cannot run beside the test.

Tolerances are chosen for each comparison, not fitted to past runs:

- Log-probabilities are centered before cosine similarity.
- A greedy token passes within 5% of the top probability, which allows for BF16 rounding tying the top two.

### Pin each rejection

Each way to be refused gets its own case asserting the kind and the reason.

- `refusals_read_no_weight` checks the kind of each refused load choice: `InvalidData` for an unknown architecture, `Unsupported` for a refused one or a choice that no model in the registry serves.
- `test_refuses_parallelism_at_startup` matches each refused option by name.

## Checklist

Stop at the first yes:

1. A type can make it unrepresentable: do that, no test.
2. The input is untrusted: parse once, test accepted and refused inputs.
3. No type can hold it: check at the boundary, test each refusal.
4. A higher test already covers it: write nothing, delete duplicates.
5. Otherwise write one test, named for its contract, at the highest boundary, on hardware, against an oracle.

Speed is measured as [Models](../models/index.md#what-is-measured) describes.
