# furiosa-opt-engine

[![Rust reference](https://img.shields.io/badge/reference-Rust-orange)](https://developer.furiosa.ai/furiosa-opt-models/reference/rust/) [![Python reference](https://img.shields.io/badge/reference-Python-blue)](https://developer.furiosa.ai/furiosa-opt-models/reference/python/)

Load model weights and expose composed NPU functions to Python.
The engine owns checkpoint I/O, parameter upload and submission.

- [Installation](https://developer.furiosa.ai/furiosa-opt-models/getting-started/installation/): each framework plugin installs the engine, the `furiosa_opt_engine` Python package, as a dependency.
- [Using the engine](https://developer.furiosa.ai/furiosa-opt-models/engine/python/)
- [Architecture](https://developer.furiosa.ai/furiosa-opt-models/design/architecture/#4-engine-host-composition): why the engine is shaped this way.

## Repository layout

```text
src/
  load/           Checkpoint files and weight upload into HBM
```

To add a model, follow [Adding a model](https://developer.furiosa.ai/furiosa-opt-models/develop/add-model/).
