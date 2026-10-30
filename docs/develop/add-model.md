# Adding a model

Add a model whose building blocks already exist in the models crate, so no new kernel is written.
The composition example is Mistral NeMo, and the later steps follow the Qwen3 files; to write the composition itself, see [Model composition](compose.md).

| Step | File |
|---|---|
| Define the composition | `crates/furiosa-opt-models/src/models/<family>/<model type>.rs` |
| Load its weights | the same file |
| Serve it from the engine | `crates/furiosa-opt-engine/src/decoder/<model type>.rs`, `crates/furiosa-opt-python/src/model/<model type>.rs` |
| Add it to the registry | [`registry.rs`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-engine/src/registry.rs) |
| List the checkpoint | `.github/benchmarks.json` |
| Document it | `docs/models/<name>.md`, `nav` in `mkdocs.yml` |
| Test it | `crates/furiosa-opt-engine/tests/<model type>.rs` |

## Define the composition

Create the model type's file in its family's directory of `models/`, declare it in that directory's `mod.rs` with `pub mod <model type>;`, and bind the checkpoint's catalog architecture to existing modules at the checkpoint's dimensions:

```rust
--8<-- "crates/furiosa-opt-models/src/models/decoder/attention/mistral.rs:composition"
```

The same file holds the config, which checks `config.json` against the compiled dimensions.
A model type that the engine serves also declares each published size, as [`qwen3.rs`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-models/src/models/decoder/attention/qwen3/mod.rs) does:

- Its classes are aliases over a size, the form of its projections, a placement, a KV cache form and a memory, such as `ForCausalLM<Size<"0.6B">, form::Affine, Parallel<1>, cache::Bf16, X>`.
- Each size implements `Geometry`, the values its checkpoints declare, and `Decode`, how a call of it compiles.
- Converting the parsed `Config` to a size checks every value the size fixes, and refuses a feature the model type does not serve as `Unsupported`.

Keep two conventions in model and catalog files:

- Declare items top-down, in this order:
    1. The model type the file defines, with its doc.
    2. Its parts, in the order the model uses them (embedding, layers, norm, head), each below its first use.
    3. The input, output and context types.
    4. Geometry axes and constants.
    5. `Config` and its parsing.
    6. Tests.
- Give every catalog template a `/// Source: <URL>` line that names the vLLM or Transformers file it follows, at a pinned commit.

## Load its weights

Derive `Load` on each struct the composition adds, and name each field as the checkpoint names it, so that `checkpoint.load::<T<Hbm>>(&mut device)` loads the whole tree strictly.
Where the checkpoint's names or shapes differ from the tree's, use the derive's attributes; [Weights and loading](compose.md#weights-and-loading) gives the rules and links their reference.

A checkpoint tensor that no field reads fails the load unless an `ignore` prefix covers it, so declare the prefixes of the tensors the model leaves unused.

## Serve it from the engine

A framework calls a model through its engine decoder and the Python class of its model type:

- The engine's one decoder serves every `Root` in [`decoder/mod.rs`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-engine/src/decoder/mod.rs), launching the root once a call. A root the decoder family's loop runs, such as `launch::Model`, is one already; another root implements `Root`: its size, where its rows land and the adapters it takes.
- The Python class names the model: it derives from `Decoder`, which serves every decoder alike, and adds nothing of its own. `register` in the binding's [`lib.rs`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-python/src/lib.rs) adds its submodule, and `LoadOptions.load` in [`model.rs`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-python/src/model/mod.rs) names the class of each architecture.
- Regenerate the Python type stubs with `cargo furiosa-opt run -p furiosa-opt-python --release --bin stub`, which loads the Python library: put its directory, `sysconfig.get_config_var("LIBDIR")`, on `LD_LIBRARY_PATH`. CI fails when the stubs differ from the code.

## Add it to the registry

In [the registry](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-engine/src/registry.rs), add `Implementation::of::<T>()` for each concrete type, one per size, to the row of each upstream architecture name the model serves.
A type is listed once it implements `Implements`, its size and the scope it reads a checkpoint at, and the engine's `Decoder`; a name without one stays refused, `&[]`.
[The model registry](compose.md#the-model-registry) states how a load chooses among them.

## List the checkpoint

Add a row to `models` in `.github/benchmarks.json`, copying a similar model's row:

```json
{
  "id": "org/Model-Name",
  "tier": "nightly",
  "task": "text",
  "doc": "model-name",
  "serve": ["--max-model-len", "40960"],
  "download": ["*.json", "*.safetensors"]
}
```

- `doc` names the model's page.
- `serve` adds server flags to the task's defaults.
- `quantization` lists a quantized checkpoint's `weights` and `activations` precisions, each naming the layers it applies to when they differ; a row without it runs BF16.
- `tier` sets how often CI runs the row.
- `forward` names the row's device test, which CI runs alone and measures at the row's tier.
- `serving: true` marks the one row the serving benchmark serves.
- `download` lists the files CI downloads as `hf download --include` patterns; without it, CI downloads every file.
- `revision` pins the checkpoint commit that CI downloads.
- `reference` names the [reference outputs](testing.md#reference-outputs) that the row's tests compare with.

## Document the model

Create `docs/models/<name>.md` and add it under `Models:` in the `nav` of `mkdocs.yml`; `mkdocs build --strict` fails on a page outside the `nav`.
The page holds only a title and the benchmark include; the include renders the serve command and dashboard from the inventory:

```markdown title="docs/models/mistral-nemo.md"
--8<-- "docs/models/mistral-nemo.md"
```

## Test it

Test the model type in `crates/furiosa-opt-engine/tests/<model type>.rs`, following [`qwen3_bf16.rs`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-engine/tests/qwen3_bf16.rs): load the checkpoint, compare a greedy continuation with Transformers' on the same checkpoint, and pin each refusal.
[Test a model](testing.md#test-a-model) runs the tests on a chip.
