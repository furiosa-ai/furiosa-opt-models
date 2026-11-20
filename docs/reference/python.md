# Python reference

The engine is the `furiosa_opt_engine` package, which both plugins install as a dependency.
This page is generated from its type stubs and grouped by task.
For a walkthrough, see [Using the engine](../engine/python.md).

## Load a checkpoint

`LoadOptions` checks its choices against a `Checkpoint` before any weight is read, then loads the model the registry selects onto a `Device`.

::: furiosa_opt_engine.Checkpoint

::: furiosa_opt_engine.LoadOptions

## Device memory

::: furiosa_opt_engine.Placement

::: furiosa_opt_engine.Device

::: furiosa_opt_engine.HbmBuffer

::: furiosa_opt_engine.HostBuffer

::: furiosa_opt_engine.Transfer

::: furiosa_opt_engine.HostTensor

::: furiosa_opt_engine.HbmTensor

## Loaded models

A load of a text model returns an instance of its model class derived from `Decoder`.
Each decoder's `forward` returns its device outputs once the device accepted the call; a copy of pinned `HostBuffer` memory may return before it ends, and only `Transfer.wait`, every other copy, `HbmTensor.wait` and `Device.synchronize` wait for the device ([asynchronous calls](../engine/python.md#run-calls-asynchronously)).

::: furiosa_opt_engine.Decoder

::: furiosa_opt_engine.qwen3.ForCausalLM

::: furiosa_opt_engine.qwen3.Model

::: furiosa_opt_engine.qwen3.ForSequenceClassification

::: furiosa_opt_engine.llama.ForCausalLM

A load of a vision-language model's media encoder alone (`LoadOptions.encoder()`) returns an `Encoder`, whatever its model, whose calls turn the media below into rows of the language model's width.

::: furiosa_opt_engine.Encoder

::: furiosa_opt_engine.vision_language.Image

::: furiosa_opt_engine.vision_language.Audio

## Diffusion pipelines

A MiniMax H3 load returns one four-chip `Pipeline`. Its component views retain
the same pipeline and device. Component calls take the framework's arguments and
return completed host tensors; they do not use the decoder KV-cache API below.
See the [H3 profile](../serving/minimax-h3.md) for supported inputs and serving.

::: furiosa_opt_engine.minimax_h3.Component

::: furiosa_opt_engine.minimax_h3.Pipeline

::: furiosa_opt_engine.minimax_h3.Transformer

::: furiosa_opt_engine.minimax_h3.TextEncoder

::: furiosa_opt_engine.minimax_h3.VideoDecoder

::: furiosa_opt_engine.minimax_h3.AudioDecoder

## Requests and cache

A call takes one `Request` per sequence and the `Cache` they append to; `Cache(spec, blocks)` allocates the `CacheSpec` that a loaded model describes.

::: furiosa_opt_engine.decoder.Request

::: furiosa_opt_engine.decoder.Keep

::: furiosa_opt_engine.decoder.Output

::: furiosa_opt_engine.decoder.CacheSpec

::: furiosa_opt_engine.decoder.Cache

::: furiosa_opt_engine.decoder.Pool

::: furiosa_opt_engine.decoder.Group

::: furiosa_opt_engine.decoder.Kind
