# furiosa-opt-models

[![Rust reference](https://img.shields.io/badge/reference-Rust-orange)](https://developer.furiosa.ai/furiosa-opt-models/reference/rust/)

Build NPU functions from numerical operations and reusable module compositions.
Parameters belong to a module, and each call supplies operands and writable state.

- [Adding a model](https://developer.furiosa.ai/furiosa-opt-models/develop/add-model/)
- [Model composition](https://developer.furiosa.ai/furiosa-opt-models/develop/compose/)
- [Testing and benchmarking](https://developer.furiosa.ai/furiosa-opt-models/develop/testing/)
- [Architecture](https://developer.furiosa.ai/furiosa-opt-models/design/architecture/#3-models-crate)

## Repository layout

```text
src/
  lib.rs          Crate root; the Module traits and generic combinators are imported from it
  module.rs       Module trait and its device-call aliases
  ops/            Numerical operations and tensor layouts
  modules/        Reusable function structures
    generic/      General composition and layer containers
  models/         Model definitions, weight trees, and device entries
    parameters/   Shared parameter construction and validation
```

Add numerical behavior to an operation and connect it through a module composition.
Model definitions choose those compositions and the checkpoint parameters they consume.

