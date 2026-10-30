# Architecture

The layers between a checkpoint and a Python call, and which way dependencies point.
For calls, see [Using the engine](../engine/python.md) and [Model composition](../develop/compose.md).

| Layer | Owns | Guarantees | Depends on |
| --- | --- | --- | --- |
| 1. Checkpoint | A directory's tensors and config | Malformed input refused | SDK |
| 2. Registry | Config to loaded model | Refusals before a weight is read | 1, 3 |
| 3. Models | Module trees and their weights | Compile-time shapes | 1, SDK |
| 4. Host composition | One launch per `Function`, decoder calls | Requests checked before a launch | 3, 5 |
| 5. Runtime memory | Device allocations and views | Freed with last view | Runtime |
| 6. Python surface | Checkpoints, load options, loaded models | Typed, device-resident | 2, 4, 5 |

No layer reaches back up.

## Boundaries

| Owner | Owns | Leaves to others |
| --- | --- | --- |
| Models | Module trees, their weights and how they load, the registry | Launches, requests, serving |
| Engine | Load options, launches, decoder calls and their cache storage, the Python binding | Scheduling, block assignment, sampling |
| Integrations | Framework conversions | Mathematics |
| Framework | Scheduling, cache blocks, sampling | Models, engine |

A loaded model is callable without a serving stack, and one engine serves frameworks that disagree on policy.

## 1. Checkpoint

Maps a directory's tensors and reads its config, and chooses no architecture.
A directory is one model, its `.safetensors` files beside `config.json`, or a diffusers pipeline, whose `model_index.json` names a folder per component; a component's tensor names take its folder's path as a prefix.
It refuses malformed files and a tensor name that two files hold.
Mapping reads no weight; a load converts and uploads one tensor at a time ([`checkpoint.rs`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-models/src/load/checkpoint.rs)).

## 2. Registry

Turns a checkpoint's config into the model it loads as, and loads it.

- **The registry:** every upstream architecture once, with the concrete types its checkpoints load as, or a refusal ([The model registry](../develop/compose.md#the-model-registry)).
- **`LoadOptions::parse`:** reads the config and the tensor names only, with no weight or device. It takes the architecture the options name, else the config's, and the one registered type that serves the config, then checks the load's choices: chips, KV cache form and layers.
- **`LoadOptions::load`:** loads that type strictly into HBM.
- **Refuses early:** an unknown architecture as invalid, and a refused architecture or a choice that no registered type serves as unsupported, before any weight is read.

## 3. Models crate

Numerical functions over device tensors, the weight trees they read and the registry, with no Python or serving knowledge.

- **Modules:** `ops/`, `modules/` and `models/` hold plain `Module`s, leaf `Op`s among them, composed by `Sequential`, `Then`, `Fork`, `Each`, `Layers` and tuples under one calling convention ([The `Module` contract](../develop/compose.md#the-module-contract)).
- **Weights:** a tree loads from a checkpoint by its field names, tensor by tensor into HBM, and nothing stays on host ([Weights and loading](../develop/compose.md#weights-and-loading)).
- **Execution order:** a decoder runs as its family's schedule writes it, plain loops over its layers and row tiles that meet only at attention ([Execution order](../develop/compose.md#execution-order)).
- **Checked:** shapes, dtypes and mappings at compile time, with no CPU fallback.

## 4. Engine host composition

Runs the layer 3 structure from the host, one device launch at a time.

- **Entry:** one `#[device]` function, `forward`, runs any module on the env, state and input its launch passes, so a model adds no entry ([The device entry](../develop/compose.md#the-device-entry)).
- **`Function`:** a borrowed tree bound to the entry that runs it.
- **`AsyncModule`:** the host counterpart of `Module`, which `Function` and every decoder implement; a call returns once the device accepted its launches.
- **No implicit tensor transfers:** operands and results stay on the device, and a call uploads only what it plans on the host, so latency and ownership stay attributable.
- **Decoder calls:** a decoder's call checks its requests on the host, nowhere else, and uploads their compact history: each request's rows, cached positions and blocks. It launches the body once, in the one row envelope the body compiles. The base model derives every layer's attention context from that history once, on the device, so no row count compiles a body of its own and the host builds no per-layer metadata.

## 5. Runtime memory

A thin wrapper over the runtime's `Device`, `HbmBuffer`, `HostBuffer` and typed tensors.
Views retain their allocations, so a row slice remains valid after its source object is dropped, and a device stays open while any buffer on it lives.

## 6. Python surface

A framework opens a `Checkpoint`, states its choices in `LoadOptions` and loads once onto a `Device` it opened.
The load returns a `Decoder` as an object of the model type's class, such as `qwen3.ForCausalLM`, whose `forward` takes the requests and the cache.
Checking the choices reads no weight and opens no device.

| Rung | Refused | When |
| --- | --- | --- |
| Type | Mismatched composition | Engine compile |
| Construction | Weights for another model | `LoadOptions.load` |
| Runtime | A request that breaks the call's rules, a tensor on another device | Each call |

## Enabling a model family

Compositions exist for the whole catalog, and a family runs once every layer below holds what it needs:

| Layer | Holds | Blocks when |
| --- | --- | --- |
| `ops/` | One kernel per operation | A leaf operation has no kernel |
| `modules/`, `models/` | Compositions, sized instances | No sized instance |
| Registry | The concrete types of each architecture | The architecture is `refused` |
| Engine | A decoder and a Python class for each model type | Either is missing |

An operation without a kernel is a typed contract whose `forward` is `todo!()`.
A model that uses it still compiles and loads, and its first launch fails with that operation's device panic.

[Models](../models/index.md) names the checkpoints that run.

## Measurement

[`.github/benchmarks.json`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/.github/benchmarks.json) feeds the [Models](../models/index.md) table and the checkpoints CI serves.
