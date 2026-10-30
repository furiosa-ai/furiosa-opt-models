---
hide: [navigation, toc]
---

<div class="hero" markdown>

<div class="hero__copy" markdown>

# Serve models on FuriosaAI RNGD

One model stack on the RNGD accelerator, for any serving framework.
vLLM and SGLang serve it today through their own Python and HTTP APIs, and FuriosaAI's own engine, Furiosa LLM, is next.

[Get started](getting-started/installation.md){ .md-button .md-button--primary }
[Supported models](models/index.md){ .md-button }

</div>

<div class="hero__install" markdown>

With `cargo-furiosa-opt` on your `PATH` ([Installation](getting-started/installation.md#install-the-compiler)), install a plugin from the repository root and serve a checkpoint:

=== "vLLM"

    --8<-- "docs/getting-started/installation.md:vllm"

    --8<-- "docs/getting-started/quickstart.md:vllm-serve"

=== "SGLang"

    --8<-- "docs/getting-started/installation.md:sglang"

    --8<-- "docs/getting-started/quickstart.md:sglang-serve"

</div>

</div>

## What do you want to do?

<div class="grid cards quad" markdown>

-   :octicons-server-24:{ .lg .middle } __Serve a model__

    ---

    Run an OpenAI-compatible server for text generation, embeddings or reranking.

    [Serving :octicons-arrow-right-24:](serving/index.md)

-   :octicons-graph-24:{ .lg .middle } __Check performance__

    ---

    See what CI measures and how it blocks a regression.

    [Performance :octicons-arrow-right-24:](models/index.md#how-availability-is-kept)

-   :octicons-cpu-24:{ .lg .middle } __Call the engine__

    ---

    Load a checkpoint and run it from Python without a serving framework.

    [Using the engine :octicons-arrow-right-24:](engine/python.md)

-   :octicons-plus-circle-24:{ .lg .middle } __Add a model__

    ---

    Compose a checkpoint's architecture from modules the models crate already has.

    [Adding a model :octicons-arrow-right-24:](develop/add-model.md)

</div>

## What you get

<div class="features" markdown>

- **Framework agnostic.** The models and the engine know no serving framework; a thin plugin adapts each one, which keeps its own APIs, scheduling and sampling.
- **Checkpoints, not conversions.** A Hugging Face checkpoint loads as it is when a compiled model fits its architecture; [Models](models/index.md) lists the ones that do.
- **One engine underneath.** The same compiled functions are callable from Python without a framework.
- **Measured continuously.** CI measures kernels, forward passes and serving on RNGD every night and on request for a pull request ([Models](models/index.md#how-availability-is-kept)).

</div>

## How it fits together

<div class="fit" markdown>

![Layer stack: vLLM and SGLang run on the furiosa-opt-vllm and furiosa-opt-sglang plugins, and Furiosa LLM, coming soon, runs on the engine directly. The plugins, the engine and the models make up the furiosa-opt-models repository; the models run on the RNGD accelerator.](assets/stack-light.svg#only-light)
![Layer stack: vLLM and SGLang run on the furiosa-opt-vllm and furiosa-opt-sglang plugins, and Furiosa LLM, coming soon, runs on the engine directly. The plugins, the engine and the models make up the furiosa-opt-models repository; the models run on the RNGD accelerator.](assets/stack-dark.svg#only-dark)

<div markdown>

The plugins incubate in this repository while the interface between frameworks and the engine is designed.
Once it settles, each moves to a dedicated repository, and this repository keeps the engine and the models.

[Concepts](concepts.md) explains what this split means for the options you can set.

</div>

</div>
