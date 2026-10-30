# Environment variables

Variables that furiosa-opt-models, its build and its runtime read.
Framework variables, such as vLLM's, keep their framework's meaning.

## Build

| Variable | Effect |
| --- | --- |
| `CARGO` | The Cargo that builds the engine during `uv pip install`. Set it to `cargo-furiosa-opt` so model functions compile for RNGD. |
| `FURIOSA_OPT_OUT_DIR` | Directory where `cargo-furiosa-opt` writes and finds compiled kernels. |
| `RUSTFLAGS` | Compiler flags for the build. |

## Runtime

| Variable | Effect |
| --- | --- |
| `FURIOSA_VISIBLE_CHIPS` | Comma-separated chip indices, such as `2,3`, that automatic placement chooses from. Unset or empty, it chooses from every installed chip. An explicit engine `Placement` names its own chips. |
| `FURIOSA_OPT_PROFILE` | `info`, `debug` or `trace` records device spans at that level for a tracing subscriber installed before the functions load, as the forward tests' Chrome trace does. Another value prints a warning and runs unprofiled. |

## Tests

| Variable | Effect |
| --- | --- |
| `FURIOSA_OPT_REFERENCES` | Directory of downloaded [reference outputs](../develop/testing.md#reference-outputs), each in `<directory>/<tag>`. The tests of an inventory row that names a reference read it there. |
