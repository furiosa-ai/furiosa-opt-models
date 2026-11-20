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
| `FURIOSA_H3_CHECKPOINT` | Local MiniMax H3 checkpoint root or `FL2VA` directory for the opt-in [H3 tests](../serving/minimax-h3.md#private-reference-tests). No weights are downloaded by these tests. |
| `FURIOSA_H3_GOLDEN` | Private H3 reference directory containing `manifest.json`, `inputs.safetensors` and `outputs.safetensors` for the recorded request. The opt-in tests validate the checkpoint identity, workload, hashes and tensor contracts. There is no public reference-download fallback. |
