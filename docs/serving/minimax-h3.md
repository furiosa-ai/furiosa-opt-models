---
hide: [toc]
---

# MiniMax H3

MiniMax H3 generates video with stereo audio from text on four RNGD chips.
This setup and integration guide covers a manually validated profile with private
reference data.
The profile below has passed local native, vLLM-Omni and HTTP lifecycle checks,
including TeaCache. The original 77-token workload passes component, raw-output
and served-output GPU reference checks. H3 has no CI coverage and is not in the
published CI benchmark inventory.

## MiniMaxAI/MiniMax-H3

Text-to-video with audio · FL2VA · Four RNGD chips · One active request

### Selected profile

| Setting | Profile |
| --- | --- |
| Checkpoint | `MiniMaxAI/MiniMax-H3`, revision `42ed227ee7df40d41602854ae760620d6eb651fe`, FL2VA |
| Placement | Four explicitly selected RNGD chips; one framework worker |
| Requests | One active request; batch size one |
| Video | 1344 × 768, 124 frames at 24 FPS |
| Requested duration | Five seconds |
| Audio | Stereo, 32 kHz |
| Text | `Live` execution with maximum capacity 2048 tokens |
| Validation workload | 50 denoising steps; not a compile-time step limit |

The denoiser and video/audio decoders remain resident. The text encoder keeps a
memory-sized prefix of its first 50 layers resident and streams the rest across
the same four chips. It returns rows before final normalization. Text length counts tokens after tokenization and prompt formatting,
before padding. Runtime valid lengths control visibility; requests exceeding the
configured capacity are rejected.

Five seconds is the request value: 124 frames at 24 FPS represent about 5.167
seconds, and the audio profile contains 165600 samples per channel (5.175 seconds).
Returned duration therefore extends beyond the requested five seconds.

### Serving

Follow [Installation](../getting-started/installation.md) and use the
`furiosa-opt-vllm` plugin with its `omni` extra (vLLM-Omni `0.31.0rc1`). Point the
command at the `FL2VA` directory from the checkpoint revision above. Select four
free chips explicitly; replace the example IDs with your selected chips.
The command runs without TeaCache.

=== "vLLM-Omni"

    ```bash
    FURIOSA_VISIBLE_CHIPS=4,5,6,7 vllm serve /path/to/MiniMax-H3/FL2VA \
      --omni --trust-remote-code --task-type t2va --enforce-eager \
      --stage-init-timeout 900 --init-timeout 900 \
      --host 127.0.0.1 --port 8000
    ```

One Omni worker drives all four chips. Eager execution is required. The framework
loads checkpoint code to construct component skeletons, so `--trust-remote-code`
is required for this checkpoint. The explicit startup timeouts accommodate the
observed loading time; they are startup limits, not request latency targets.

### TeaCache

To enable the H3 TeaCache policy, append these flags to the serving command:

```bash
--cache-backend tea_cache --cache-config '{"rel_l1_thresh": 0.17}'
```

TeaCache reuses transformer residuals on selected denoising steps. The engine
retains the probe and residual rows, while Omni applies the scalar policy.
Each new generation resets the retained rows. Normal-threshold HTTP generation,
repeated decoded outputs and cancellation/recovery have passed local checks;
the recorded cached request passes the original served-output GPU bounds. These
checks do not establish quality across other prompts or cache thresholds.

### Validation status

Local checks cover four-chip component coexistence, text encoding through 2048
tokens, and short/2048-token framework calls. Forced-compute cache calls match
uncached output bits. Callback failures and cancellation restore storage and
allow the next call to succeed.

Both uncached and normal-threshold cached HTTP checks cover a complete 50-step
request, repeated decoded values, cancellation of an in-progress request with
successful recovery, and normal shutdown with worker release.

The recorded raw-output request uses saved GPU text conditioning and checks five
sampled frames and the full stereo waveform before media encoding. Served-output
checks also cover the plain-text path. Both use the original numerical bounds.
Component comparisons cover the text encoder, condition projection, refiner,
first joint block and heads at the original sampled steps. Numerical acceptance
uses the original 77-token workload and tolerances. Runtime-length checks through
2048 tokens are separate from GPU numerical comparisons; broader prompt and cache
quality coverage is not established by this workload.

Two corrected uncached HTTP requests took 255.41 and 256.44 seconds, averaging
255.92 seconds. The earlier ten-request Furiosa reference baseline averaged
255.40 seconds on the same host and four chips at 300 W. Both use the original
50-step workload with TeaCache disabled and time the complete HTTP response.
The measurements use each implementation's pinned SDK/compiler, and the two new
requests were not interleaved with the earlier baseline. The observed average
offset is 0.52 seconds; this comparison does not establish exact performance
parity or a statistically significant difference.

### Private reference tests

The H3 tests run only when explicitly enabled with `--h3`. Supply the local paths
described in [Environment variables](../reference/environment.md#tests), select
four free chips, and run:

```bash
FURIOSA_VISIBLE_CHIPS=4,5,6,7 \
FURIOSA_H3_CHECKPOINT=/path/to/MiniMax-H3/FL2VA \
FURIOSA_H3_GOLDEN=/private/h3-reference \
python -m pytest --h3 integrations/furiosa-opt-vllm/tests/test_minimax_h3.py
```

The selected tests fail when inputs are missing or incompatible. They do not
download weights or references, or save generated media. Keep reference files and
test logs in private storage when running this command in CI; H3 has no public
reference-release download in the inventory workflow.

TeaCache's synchronous decision callback cannot call a component of its own
pipeline on that thread; such calls raise `RuntimeError`. Calls from other threads
remain serialized through the same pipeline owner.

### Unsupported scope

Additional resolutions or media shapes, batching or concurrent requests,
image/video conditioning, LoRA, and FastH3 adapter fusion are outside this profile.
Extra encoder chips and separate resident text stages are also outside scope.

The H3 integration refuses sleep/wake, framework weight offloading, quantization
overrides, LoRA adapters, multiple Omni workers, and execution without eager mode.
The engine refuses KV-cache options, batch-invariant attention tiling, and
partial-layer loading. Lifecycle support in other models does not establish H3
support.

### Weights

MiniMax H3 weights are not included. Obtain them from MiniMax under the
[MiniMax H3 Community License and Acceptable Use Policy](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE).
