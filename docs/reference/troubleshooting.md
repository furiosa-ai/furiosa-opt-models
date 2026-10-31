# Troubleshooting

Each entry starts from the message you see, then gives its cause and the fix.

## Installation

### The kernel was expanded by another rev

```text
error: this kernel was expanded by rev <kernel rev>, but the driver is rev <driver rev>
```

The installed compiler is another snapshot than the one `Cargo.toml` pins, typically after pulling a change that moved the pin.
Install the pinned snapshot again, as [Install the compiler](../getting-started/installation.md#install-the-compiler) shows.

### The aarch64 C headers are missing

```text
fatal error: bits/wordsize.h: No such file or directory
```

The compiler links the NPU programs with the aarch64 cross toolchain, which it does not ship; install it with `sudo apt install gcc-aarch64-linux-gnu libc6-dev-arm64-cross`.

## Startup

### The context length exceeds the compiled window

```text
furiosa-opt-vllm requires max_model_len at most 131072
furiosa-opt-sglang requires context_length at most 131072
```

The server's context, the checkpoint's own length unless an option sets it, exceeds the window its decoder is compiled for.
Pass at most the number in the message as `--max-model-len` (vLLM) or `--context-length` (SGLang).

### One full-length request does not fit in cache memory

```text
ValueError: To serve at least one request with the model's max seq len (262144), (84.0 GiB KV cache is needed, which is larger than the available KV cache memory (35.09 GiB).
```

The window fits, but the KV cache of one request at that length does not fit in device memory.
Pass `--max-model-len -1` so that vLLM takes the longest length that fits, or pass the length the message estimates.

### An option is refused

```text
furiosa_opt_vllm.admission.Refused: furiosa-opt-vllm serves no LoRA adapter
```

The plugin cannot honor the option, and the message states what it accepts.
Remove the option, or set a value from [vLLM options](../serving/vllm.md#options) or [SGLang options](../serving/sglang.md#options).

### The checkpoint has no compiled cache dtype

```text
the model has no specialization for cache dtype ..., compiled ...
```

The decoder is compiled without that KV cache dtype.
Drop `--kv-cache-dtype` or pick a compiled one; the KV cache column of [Models](../models/index.md#available-models) lists them.

### No chip is free

```text
npu0 is already reserved by another process
invalid request: no placement candidates
```

Another process holds the chip, or no installed chip matches the selection.
Stop your own server that holds it, or select free chips with `FURIOSA_VISIBLE_CHIPS`; do not stop another user's process.
If `/dev/rngd` is missing, install the RNGD driver as described in [Installing Prerequisites](https://developer.furiosa.ai/latest/en/get_started/prerequisites.html).

### A chip index is malformed

```text
ValueError: Non-integer device ID 'a' is not supported by cpu.
ValueError: invalid literal for int() with base 10: 'a'
FURIOSA_VISIBLE_CHIPS: `a` is not a chip index
```

vLLM, the SGLang plugin and the runtime's automatic placement each read `FURIOSA_VISIBLE_CHIPS`, which takes comma-separated chip numbers, such as `0,1` ([Environment variables](environment.md)).

### Profiling records nothing

```text
furiosa-opt: FURIOSA_OPT_PROFILE requested profiling, but no subscriber listens on span::npu; install a tracing subscriber before loading functions; profiling is disabled
```

Device spans need a tracing subscriber that exists before the functions load.
Run a forward test, whose harness installs one ([Reproduce a benchmark](../develop/testing.md#reproduce-a-benchmark)).

## While serving

### Every request fails after one error

A device launch failed, which fails the device: every later call on it raises `OSError`.
Restart the server and resubmit the affected requests; with the engine API, follow [Errors and limits](../engine/python.md#errors-and-limits).

## Benchmarks

### A pull request comment lists a regression

Each line names a measurement, its base and pull request values, and the budget it exceeds ([Regression gate](../models/index.md#regression-gate)).
Start with the lowest level that shows a regression, then compare matched workloads before you assign a cause.

### The comment says the comparison is unavailable

```text
Measurements or comparison unavailable; see CI.
```

The benchmark job failed before it compared, for example while serving or publishing; the linked CI run shows the failed step.
