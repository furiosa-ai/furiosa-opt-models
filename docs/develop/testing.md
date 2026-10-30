{% from "commands.md" import vllm_serve, vllm_bench with context %}
{%- set served = benchmarks.models | selectattr("tier", "equalto", "gate") | selectattr("task", "equalto", "text") | first -%}
# Testing and benchmarking

Test a model and run the framework tests. You need an RNGD device, `uv` (fixtures download checkpoints with it) and `cargo-furiosa-opt` on your `PATH`; see [Installation](../getting-started/installation.md).

## Test a model

Run the model type's tests on a chip. They load the checkpoint, decode on the device and check the output against Transformers:

```bash
cargo furiosa-opt test -p furiosa-opt-engine --release --test qwen3_bf16 \
  -- --include-ignored --test-threads 1
```

A test that opens a device is ignored by default, so `--include-ignored` runs it with the others.
The first build on a new compiler revision is slow.
To test one numerical operation, run a models test such as `adds_elementwise` with `-p furiosa-opt-models`.

The engine's tests of a published checkpoint's config and strict load fetch no weights: the first run reads each checkpoint's `config.json` and safetensors headers from the Hugging Face hub, at the revision the test names, into the hub's cache (`$HF_HOME`, else `~/.cache/huggingface`).
Later runs read the cache and need no network.
A gated checkpoint needs the token that `hf auth login` stores there.

## Run the framework tests

Install a plugin as in [Installation](../getting-started/installation.md), the vLLM plugin with its `omni` extra, then run its tests on RNGD ([what they check](../design/testing.md)):

```bash
python -m pytest integrations/furiosa-opt-vllm/tests    # vLLM
python -m pytest integrations/furiosa-opt-sglang/tests  # SGLang
```

A test that compares with [reference outputs](#reference-outputs) reads them from [`FURIOSA_OPT_REFERENCES`](../reference/environment.md#tests).

## Reference outputs

A reference output is what the upstream framework returns for one fixed request, recorded on the hardware it runs on.
A test compares with it when that framework cannot run beside the test, as vLLM-Omni's GPU pipeline cannot run beside an RNGD test.
It is never our own output, which would only pin today's behavior ([Assert against an oracle](../design/testing.md#assert-against-an-oracle)).

Each reference is a release of this repository, tagged `references/<org>--<name>@<revision>` with the first seven characters of the checkpoint revision.
An inventory row names it in `reference`.
The [`checkpoints` action](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/.github/actions/checkpoints/action.yml) downloads the release into `$FURIOSA_OPT_REFERENCES/<tag>` and checks every file against the digests in its `manifest.json`; a test reads it through the `reference` fixture.
To run such a test locally, download the release the same way:

```bash
export FURIOSA_OPT_REFERENCES=~/.cache/furiosa-opt-models/references
tag=references/<org>--<checkpoint>@<revision>
gh release download "$tag" --repo furiosa-ai/furiosa-opt-models --dir "$FURIOSA_OPT_REFERENCES/$tag"
```

Publish a reference with a newly served checkpoint, and a new one for a new checkpoint revision or upstream version:

1. Record the request with the upstream framework at the checkpoint revision the row pins.
   Hooks that read values must not change the computation: a second request without them gives the same outputs.
2. Keep only what the test reads and asserts, in the smallest dtype that holds it, such as sampled video frames as `uint8`.
3. Write `manifest.json` with the capture command, the software versions, the checkpoint revision, the workload and the `sha256` of every other file.
   Attach the capture script and the environment's `pip freeze`, so the recording can be repeated.
4. Create the release:

    ```bash
    gh release create references/<org>--<name>@<revision> --prerelease \
      --title "Reference outputs: <org>/<name>@<revision>" --notes "<the request>" \
      manifest.json inputs.safetensors outputs.safetensors capture.py freeze.txt
    ```

5. Set the row's `revision` and `reference`.

A tag is never changed: CI caches a reference by its tag and never downloads it again, and past results were checked against it.
A new recording gets a new tag, and the row moves to it.

## Reproduce a benchmark

To reproduce kernel and forward figures, run a checkpoint's forward test with `FURIOSA_OPT_PROFILE=info`, which kernel spans need:

```bash
FURIOSA_OPT_PROFILE=info cargo furiosa-opt test -p furiosa-opt-engine \
  --release --test forward {{ served.forward }} -- --include-ignored --test-threads 1
```

The trace lands in `trace/<org>--<checkpoint>/forward.json`, a Chrome trace that [Perfetto](https://ui.perfetto.dev) opens.

To reproduce a serving figure, start the server as CI does, with the engine core in the server process:

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 {{ vllm_serve(served) }}
```

Once it logs `Application startup complete`, run one width of the pull-request workload from another shell:

```bash
{{ vllm_bench(served.id, benchmarks.tasks.text.serving.gate[0], benchmarks.tasks.text.serving.gate[0].widths[0]) }}
```

Stop the server with Ctrl+C after the measurement.
To reproduce the request latency of a task that no command serves, run its contract test, which writes `trace/<org>--<checkpoint>/request.json`.
CI runs every width of each workload through the [`serving` action](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/.github/actions/serving/action.yml); [Models](../models/index.md#what-is-measured) lists the workloads.

## Measurement implementation

How CI records the three levels defined in [Models](../models/index.md#what-is-measured):

| Level | Span | Recorded by |
| --- | --- | --- |
| Kernel | `NPU` span in a `forward` span | Chrome trace |
| Forward | `forward` span | Chrome trace |
| Serving | Client-observed request, or a whole request of another task | `vllm bench serve`, or the contract test's Chrome trace |

- A contract test of a task that `vllm bench serve` cannot send writes one complete `request` event per measured request to `trace/<org>--<checkpoint>/request.json` in the repository; the event's duration is the request's latency.
- Kernel time is `begin_cycle` to `end_cycle`, aggregated per kernel. Forward time is recorded in microseconds and published in milliseconds.
- The [`metrics` action](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/.github/actions/metrics/action.yml) turns traces and serving summaries into records, taking the median across steps and across requests. Steps run one at a time, so a kernel span belongs to the `forward` span it is recorded in.
- The [`datadog` action](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/.github/actions/datadog/action.yml) publishes them. For a pull request it compares against the baseline revision within each level's budget in [`.github/benchmarks.json`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/.github/benchmarks.json).
- Serving records median time to first token, time per output token and whole-request latency per concurrency width and workload, using `vllm.benchmarks.sweep.serve` in the [`serving` action](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/.github/actions/serving/action.yml).
