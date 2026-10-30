# Installation

This page installs the compiler and then the vLLM or SGLang plugin.
Installing a plugin also builds the engine and compiles every model function.

## Prerequisites

- Linux on x86_64.
- One or more RNGD devices with the RNGD driver of Furiosa SDK 2026.4 or later, installed as described in [Installing Prerequisites](https://developer.furiosa.ai/latest/en/get_started/prerequisites.html); `ls /dev/rngd` then lists the device nodes. The SDK documentation lists driver and firmware versions.
- For the vLLM plugin: Python `{{ vllm_project["requires-python"] }}` and glibc 2.39 or later, such as Ubuntu 24.04, because the plugin depends on vLLM's CPU wheel, which is built for `manylinux_2_39`.
- For the vLLM plugin: a C compiler, such as `sudo apt install gcc`. vLLM's CPU build samples on the host with triton-cpu and `torch.compile` kernels, which it compiles on first use.
- For the SGLang plugin: Python `{{ sglang_project["requires-python"] }}`.
- [uv](https://docs.astral.sh/uv/), [rustup](https://rustup.rs/), and the aarch64 toolchain the compiler links the NPU programs with, which it does not ship: `sudo apt install gcc-aarch64-linux-gnu libc6-dev-arm64-cross`.

## Compatibility

| Component | Version |
| --- | --- |
| RNGD driver | Furiosa SDK 2026.4 or later |
| vLLM plugin `furiosa-opt-vllm` {{ vllm_project.version }} | vLLM {{ vllm_version }}, Python `{{ vllm_project["requires-python"] }}` |
| SGLang plugin `furiosa-opt-sglang` {{ sglang_project.version }} | SGLang {{ sglang_version }}, Python `{{ sglang_project["requires-python"] }}` |
| Compiler | The `cargo-furiosa-opt` snapshot of furiosa-opt-preview [`{{ std_rev[:9] }}`](https://github.com/furiosa-ai/furiosa-opt-preview/commit/{{ std_rev }}), which [`Cargo.toml`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/Cargo.toml) pins |

## Install the compiler

This repository pins the tip of furiosa-opt-preview's default branch, `next`, where the next `cargo-furiosa-opt` release lands first: the models use compiler and `furiosa-opt-std` features that the latest release does not have yet. `cargo binstall` installs that snapshot:

```bash
curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash
cargo binstall --git https://github.com/furiosa-ai/furiosa-opt-preview cargo-furiosa-opt
```

The plugin build runs the compiler by name through `CARGO`.
A compiler of another snapshot refuses the kernels with `this kernel was expanded by rev …, but the driver is rev …`; run the second command again after pulling a change that moves the pin.

## Install a plugin

Clone the repository and create a Python environment for one framework, separate from other serving frameworks:

```bash
git clone https://github.com/furiosa-ai/furiosa-opt-models && cd furiosa-opt-models
uv venv && source .venv/bin/activate
```

Run the install commands of the next sections from the repository root.

### vLLM

<!-- --8<-- [start:vllm] -->
```bash
CARGO=cargo-furiosa-opt uv pip install --torch-backend cpu ./integrations/furiosa-opt-vllm
```
<!-- --8<-- [end:vllm] -->

Installation registers the platform plugin with vLLM.
The `omni` extra, `./integrations/furiosa-opt-vllm[omni]`, also installs vLLM-Omni, whose diffusion pipelines the plugin refuses at load: the engine loads none yet.
The plugin uses [vLLM's CPU build](https://docs.vllm.ai/en/stable/getting_started/installation/cpu/), whose torch comes from PyTorch's CPU index.

### SGLang

<!-- --8<-- [start:sglang] -->
```bash
CARGO=cargo-furiosa-opt uv pip install --prerelease=allow ./integrations/furiosa-opt-sglang
```
<!-- --8<-- [end:sglang] -->

Installation registers the platform and worker through SGLang's plugin entry points.

`--prerelease=allow` is needed because SGLang pins a pre-release dependency (`cuda-tile`), which uv does not resolve by default.

## Verify the installation

To check that the compiled engine loads, run this command; it prints a `LoadOptions` object without opening a device:

```bash
python -c "import furiosa_opt_engine as engine; print(engine.LoadOptions())"
```

[Quickstart](quickstart.md) checks that a server runs on RNGD.


## Build time

!!! note
    The first build compiles every model function and takes several minutes.
    Add `-v` to see its progress.

uv builds each install in a fresh Python environment, so reinstalling recompiles pyo3 and the engine.
To reuse the previous build, install `maturin` into your environment and add `--no-build-isolation`.

## Next steps

- [Quickstart](quickstart.md)
