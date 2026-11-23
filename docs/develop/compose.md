# Model composition

A model is a tree of modules that all implement one trait, `Module`.
The tree also holds the model's weights, which load from a checkpoint by its field names.
It runs on the device whole, or a tile of rows at a time.
The worked example is Mistral NeMo ([`models/decoder/attention/mistral.rs`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-models/src/models/decoder/attention/mistral.rs)); to add a model, see [Adding a model](add-model.md).

## The `Module` contract

A module computes in one `forward`.
An op, a leaf operation that reads no env and holds no state, implements `Op` instead, and the blanket impl makes it a module of any env with the `&mut ()` state:

```rust
--8<-- "crates/furiosa-opt-models/src/module.rs:module"
```

Each argument has one role:

| Argument | Holds | Bound |
| --- | --- | --- |
| `self` | Weights, and constants read from the checkpoint's config | At load |
| `env` | Facts of the call that any module may read, such as positions, block tables and live rows | Per call, copied to each child |
| `state` | What the call lends for writing, such as KV cache blocks or recurrent state | Per call, borrowed |
| `input` | The data flowing through, and destinations the caller owns behind `&mut` | Per call |

- A module owns its weights, never its state, so one loaded tree serves any number of requests.
- A composite's state is its children's states: `&mut (AState, BState)` for `Sequential<A, B>`, `&mut [LayerState; N]` for `N` layers. A stateless module takes `&mut ()`.
- A composite passes `env` on as a generic `E`, so only the module that reads a fact names its type. Qwen3's base model, for example, prepares each call's attention context once and passes it to every layer as `env`.
- The tree below one `#[device]` entry fuses into one kernel.

## Four kinds of module

| Kind | Location | Examples |
|---|---|---|
| Leaf operation | `ops/` | `Add`, `RmsNorm`, `Linear` |
| Generic combinator | `modules/generic/`, re-exported at the crate root | `Sequential`, `Then`, `Layers` |
| Relation | `modules/` | `Attention`, `Residual` |
| Model definition | `models/` | `mistral::Model`, `qwen3::Catalog` |

- Only leaves touch hardware.
- A leaf is an `Op` unless it reads the call's env or has a module child; then it implements `Module` itself.
- Combinators and relations hold no model constants; they take their children as type parameters.
- A model definition fixes the constants, such as axis sizes, the layer count and the norm's epsilon.

A combinator gives each child its part of the input and threads the outputs forward.
`Sequential` passes its first child's output to the second by value, and `Then` lends it; a mismatch fails to compile at that pair:

```rust
--8<-- "crates/furiosa-opt-models/src/modules/generic/sequential.rs:sequential"
```

A relation is a combinator whose children have roles: [`Residual`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-models/src/modules/residual/mod.rs) computes `x + branch(norm(x))` from a norm, a branch and an add.

## Variant selection

A module or an op has one impl for each operand type it takes, and the compiler picks the impl from the operand and child types.
Nothing dispatches at run time.
These choices are types or values:

- **Weight form:** a type in [`tensor/form`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-models/src/tensor/form/mod.rs) that says how a weight is stored and restored to values, such as `BlockScaled` under the FP8 [`Block`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-models/src/tensor/form/fp8.rs) contract or NVFP4's [`Early` and `Late`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-models/src/tensor/form/nvfp4.rs).
- **Strategy:** a value in [`specialize/`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-models/src/specialize/mod.rs), such as a contraction's tiling, which the contraction looks up by its shape, or the `Strategy` that attention takes as a parameter.
- **Cache:** `Scatter` has an impl for each cache it writes into, such as a layer's paged blocks.
- **Placement:** a model alias takes `Parallel<N>`, tensor parallelism over `N` chips, as a type, so one alias yields the single-chip and the tensor-parallel model.
- **Rows:** a launch compiles for an envelope of `R` rows, and [`Live<X, R>`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-models/src/tensor/live.rs) carries the rows a call fills. Leaves bound their loops by these live rows, so one envelope serves any row count up to `R`.

## Traits beside `Module`

Only `forward` acts on the device.
A trait beside it is sound when its implementor's type alone decides what it says.
That leaves two shapes:

- **A public trait maps a data type to types and constants.** `Parallelism` gives a placement's local width and computes nothing, so any module may use it.
- **A private trait maps an operand type to the one step an operation does differently for it.** Flash attention stages whole spans for `Visible`, whose absent keys are a suffix, and clears every span for `Scattered`, whose absent keys may lie anywhere. The caller chooses by the type of the data it holds. The trait's methods act on the device, so it stays inside the operation.

What fails the test goes elsewhere:

| Instead of a trait | When |
|---|---|
| an enum or a const parameter | the cases differ by a value, not by a type |
| impls in the model definition | the impls list one model's sizes |
| a private method of the operation | several `forward` impls share a step |

## One model, top to bottom

NeMo stacks 40 pre-norm layers, normalizes, and projects to logits:

```rust
--8<-- "crates/furiosa-opt-models/src/models/decoder/attention/mistral.rs:composition"
```

`MistralForCausalLM` is the catalog's name for the upstream architecture: a template of generic modules, which NeMo's alias fills with sized leaves.
It expands to this nesting:

- `Readout`: `Normalized` over `Backbone`, then the head
    - `Backbone`: the embedding, then `Layers` of 40 layers
        - `PreNorm`: a `Residual` over attention, then a `Residual` over the gated MLP

See [`modules/`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-models/src/modules/mod.rs) and [`generic/layers.rs`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-models/src/modules/generic/layers.rs) for each level.

## The device entry

`furiosa_opt_engine::forward` is the `#[device]` entry that runs a module on the env, state and input of its launch:

```rust
--8<-- "crates/furiosa-opt-engine/src/function.rs:entry"
```

Each launch compiles the entry for its module and argument types, so a model adds no entry of its own.

- **A tensor the module writes must arrive behind `&mut`.** `DeviceSend` binds a tensor behind `&` read-only.
- **Work beyond the model is a module of the launched tree,** never a method of an argument. Qwen3's `rotary_emb`, which prepares the call's attention context, and the `Store` that moves its rows to HBM are modules of its tree.

## Weights and loading

A model's weights are the leaves of its tree.
`#[derive(Load)]` loads the tree from a checkpoint by its field names; [`load.rs`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-models/src/load/mod.rs) documents the trait, the derive and each attribute.

- **The model and its weights are one type.** A tree is generic over its memory ([`memory.rs`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-models/src/tensor/memory.rs)): `T<Hbm>` owns its leaves in HBM, and a call borrows it as `T<&Hbm>`, which kernels run on.
- **Nothing stays on host.** `checkpoint.load::<T<Hbm>>(&mut device)` moves the checkpoint into HBM one tensor at a time. Dropping the tree frees that HBM.
- **Fields carry the checkpoint's names.** `Named<"name", M>` names a generic child, as Qwen3's `Named<"embed_tokens", ..>` does ([`qwen3.rs`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-models/src/models/decoder/attention/qwen3/mod.rs)). Attributes cover what the tree does not share with the checkpoint, such as fused projections, config values or the scales of the [quantized formats](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-models/src/load/quantization/mod.rs).
- **Loads are strict.** A name the tree reads that the checkpoint lacks fails the load, and so does a checkpoint name that no field reads and no `ignore` prefix covers. A checkpoint that does not match the tree never loads in part.
- **Tying is decided by type.** A size's geometry type fixes whether its head ties to its embedding, and the config must agree, so an untied fine-tune of a tied size is another size type. A tied leaf shares its target's HBM.
- **Updates write in place.** An [`Update`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-models/src/load/update.rs) writes new weights into a loaded tree as their chunks arrive, keeping its structure and ties. If a write fails after the device took it, the tree's values are unknown until it loads again.

## Execution order

`forward` runs a module's whole output before the next module starts.
A decoder runs instead in the order its family's schedule writes, kept in [`schedule/`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-models/src/schedule/decoder/mod.rs) apart from the tree and its leaves: a leaf computes, the schedule says when.

Attention is a decoder's only step that reads other tokens' rows, so a launch's row tiles meet only there.
`Decoder` runs the loops: the first layer's stores over the embedded tiles, then each layer's attention once and, tile by tile, the work up to the next layer's store or the head.
That work is a `Tokenwise`: the end of one part, a layer after its attention or the embedded rows, then the start of the next, a layer up to its attention's store or the final norm.
A layer shape is cut at its attention, and its weights' landings are placed, only in its `Cut` impl, whose steps the `Tokenwise` impls run.

- **The order is written, not derived.** A schedule is one `forward` of plain loops, so the scheduler gets the order a reader sees. Polled streams kept the order in node state, and the device compile spent minutes on its branches.
- **A weight lands one part ahead of its use, never during an attention.** A paged attention's fold fills most of data memory: an MLP weight live there is refused at allocation. The next layer's projections land between this layer's MLP steps, the output projection's while the attention's last history is in flight.
- **A device loop cannot carry a struct whose fields it also reads apart.** A period loop that carries a staging struct and reads the struct's own stored rows is refused with `Loop inputs/outputs contain duplicate tensor indices`; keep the rows in a place of their own.
- **A capture-free closure is not a value on the device.** Passing one is refused with `unsupported aggregate constant`; split the step around the work instead.
- **A non-generic caller of device-only code fails the host build.** Host codegen monomorphizes tile sizes the device compile never reaches and fails their const checks; keep a root's `Module` impl generic in its env, so only the launches that run it compile it.

## The model registry

[The registry](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-engine/src/registry.rs) names every upstream architecture once, under its model family, with the decoders its checkpoints load as, or none.
Of the decoders a name lists, the one whose size serves the checkpoint loads; choosing it reads the config and the tensor names, never a weight.
A refused name fails as `Unsupported` and an unknown one as `InvalidData`, so an error tells a model the stack does not serve yet from a name that does not exist.
A load returns the chosen decoder as a `decoder::Served`, one object whatever its model, so the registry is a plain table and dispatches nothing; what a model computes stays in its family's code.

## On the host

The engine runs a loaded tree through a `Function`, which binds a borrowed tree to its compiled entry.
A `Function` is an `AsyncModule`, the host counterpart of `Module`: its call returns once the device accepted the launch, and its output stays a device tensor.
A decoder's call is one launch of its root: Qwen3's causal model runs its base model's loops, then its greedy head on the rows its requests keep.
[Architecture](../design/architecture.md#4-engine-host-composition) states the engine's guarantees.

To write an operation that has no kernel yet, see [Writing a kernel](kernel.md).
