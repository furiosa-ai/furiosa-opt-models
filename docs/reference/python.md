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

A load returns a `Decoder`, which serves every model alike, as an object of its model type's class from the submodule of that model type.
Every `forward` returns its device outputs once the device accepted the call; a copy of pinned `HostBuffer` memory may return before it ends, and only `Transfer.wait`, every other copy, `HbmTensor.wait` and `Device.synchronize` wait for the device ([asynchronous calls](../engine/python.md#run-calls-asynchronously)).

::: furiosa_opt_engine.Decoder
{% for class in decoder_classes %}
::: furiosa_opt_engine.{{ class }}
{% endfor %}
A load of a vision-language model's media encoder alone (`LoadOptions.encoder()`) returns an `Encoder`, whatever its model, whose calls turn the media below into rows of the language model's width.

::: furiosa_opt_engine.Encoder

::: furiosa_opt_engine.vision_language.Image

::: furiosa_opt_engine.vision_language.Audio

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
