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

`Device.placement` is the immutable physical placement used to open that device.

::: furiosa_opt_engine.HbmBuffer

::: furiosa_opt_engine.HostBuffer

::: furiosa_opt_engine.Transfer

::: furiosa_opt_engine.HostTensor

::: furiosa_opt_engine.HbmTensor

## Loaded models

A load returns a `Decoder`, which serves every model alike, as an object of its model type's class from the submodule of that model type.
Every `forward` returns its device outputs once the device accepted the call; a copy of pinned `HostBuffer` memory may return before it ends, and only `Transfer.wait`, every other copy, `HbmTensor.wait` and `Device.synchronize` wait for the device ([asynchronous calls](../engine/python.md#run-calls-asynchronously)).

::: furiosa_opt_engine.Decoder

::: furiosa_opt_engine.qwen3.ForCausalLM

::: furiosa_opt_engine.qwen3.Model

::: furiosa_opt_engine.qwen3.ForSequenceClassification

::: furiosa_opt_engine.llama.ForCausalLM

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

## Batch planning

Framework adapters use these shared helpers to map scheduled rows to decoder calls.
`Identity` carries each framework's request identifier through `Row`, `Request` and `Call` typing.

::: furiosa_opt_engine.batch.Batch

::: furiosa_opt_engine.batch.Framing

::: furiosa_opt_engine.batch.Call

::: furiosa_opt_engine.batch.Row

::: furiosa_opt_engine.batch.Request

::: furiosa_opt_engine.batch.Chunk

::: furiosa_opt_engine.batch.chunks

::: furiosa_opt_engine.batch.admit

::: furiosa_opt_engine.batch.operand

## Speculative drafting

`draft_device` reuses the target device for a default or exact match, and opens a device for an
explicit disjoint placement.
`SequenceId` carries a hashable framework request ID through `DraftRequest` and `DraftSession`.

::: furiosa_opt_engine.speculative.TokenSource

::: furiosa_opt_engine.placement.draft_device

::: furiosa_opt_engine.placement.has_distinct_chips

::: furiosa_opt_engine.placement.is_nonempty_pe_range

::: furiosa_opt_engine.placement.MAX_CHIP_ID

::: furiosa_opt_engine.placement.PE_COUNT

::: furiosa_opt_engine.speculative.same_tokenizer_vocabulary

::: furiosa_opt_engine.speculative.validate_draft_vocabulary

::: furiosa_opt_engine.speculative.validate_draft_decoder

::: furiosa_opt_engine.speculative.validate_mask_token

::: furiosa_opt_engine.speculative.cache_memory_budget

::: furiosa_opt_engine.speculative.cache_blocks_for_memory

::: furiosa_opt_engine.speculative.draft_blocks_for_shared_budget

::: furiosa_opt_engine.speculative.DraftRequest

::: furiosa_opt_engine.speculative.DraftSession
