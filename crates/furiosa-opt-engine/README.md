# furiosa-opt-engine

[![Rust reference](https://img.shields.io/badge/reference-Rust-orange)](https://developer.furiosa.ai/furiosa-opt-models/reference/rust/) [![Python reference](https://img.shields.io/badge/reference-Python-blue)](https://developer.furiosa.ai/furiosa-opt-models/reference/python/)

Load furiosa-opt-models checkpoints onto RNGD and run them for a serving framework: the engine chooses the model a checkpoint loads as, loads its weights into device memory and submits its calls, while the framework schedules requests and assigns cache blocks.
Frameworks call it through the `furiosa_opt_engine` Python package, which the binding crate `furiosa-opt-python` builds.

- [Installation](https://developer.furiosa.ai/furiosa-opt-models/getting-started/installation/): each framework plugin installs the engine, the `furiosa_opt_engine` Python package, as a dependency.
- [Using the engine](https://developer.furiosa.ai/furiosa-opt-models/engine/python/)
- [Architecture](https://developer.furiosa.ai/furiosa-opt-models/design/architecture/#4-engine-host-composition): why the engine is shaped this way.

To add a model, follow [Adding a model](https://developer.furiosa.ai/furiosa-opt-models/develop/add-model/).
