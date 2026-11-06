---
hide: [toc]
---

# Llama 3.1 8B NVFP4

{% include "benchmarks.md" %}

NVFP4 weights run with BF16 activations and BF16 KV cache. The checkpoint's activation and
FP8 cache calibration scales are validated but are not applied. The served context is 40960 tokens.

## Build and verify this branch

Follow the RNGD driver, Rust and cross-toolchain prerequisites in
[Installation](../getting-started/installation.md). This branch pins `furiosa-opt-std`
revision `8cc647d99896d35ad6b1192a387809c0ac95740b` and requires compiler
`v0.9.0+g2ccfb59b` with Rust `nightly-2026-05-01`. Install this exact public snapshot;
a newer preview compiler can reject kernels expanded by this branch's SDK.

```bash
git clone --branch feat/llama3.1-8b-nvfp4-4to16 --single-branch \
  https://github.com/furiosa-ai/furiosa-opt-models.git
cd furiosa-opt-models

compiler_version=v0.9.0+g2ccfb59b
compiler_package="cargo-furiosa-opt-${compiler_version}-x86_64-unknown-linux-gnu"
compiler_dir="$(mktemp -d)"
release="https://github.com/furiosa-ai/furiosa-opt-preview/releases/download/${compiler_version}"
curl --fail --location "${release}/${compiler_package}.tgz" \
  --output "${compiler_dir}/${compiler_package}.tgz"
curl --fail --location "${release}/SHA256SUMS" --output "${compiler_dir}/SHA256SUMS"
(cd "$compiler_dir" && sha256sum --check --ignore-missing SHA256SUMS)
tar -xzf "${compiler_dir}/${compiler_package}.tgz" -C "$compiler_dir"
export PATH="${compiler_dir}/${compiler_package}:$PATH"
```

Keep the compiler directory while using this shell, or install its two binaries into
an existing directory on `PATH`. The SDK downloads and verifies the matching mapping
and lowering libraries automatically; local prebuilt-library overrides are unnecessary.

Run the registry and NPU inference checks from the repository root:

```bash
FURIOSA_VISIBLE_CHIPS=0 cargo furiosa-opt test --locked \
  -p furiosa-opt-engine --release \
  --test llama31_nvfp4_config --test llama31_nvfp4 \
  -- --include-ignored --test-threads=1
```

Choose an available chip instead of `0` if needed. The tests use `uvx` to download
`nvidia/Llama-3.1-8B-Instruct-NVFP4` at revision
`bdb54e24298451af785c0ac63c1b485e9b7400a2`. They check CPU token references,
prefill/chunk boundaries, repeated mixed launches, BF16 cache precision and cache guards.

For Python inference without a serving framework, install the engine from this checkout:

```bash
uv venv && source .venv/bin/activate
CARGO=cargo-furiosa-opt uv pip install ./crates/furiosa-opt-engine
uv pip install huggingface-hub
```

Use [the engine API](../engine/python.md) with the NVIDIA checkpoint above and its
pinned revision. `LoadOptions` selects the NVFP4 implementation and BF16 cache from the
checkpoint configuration; no activation quantization or FP8 cache option is required.
