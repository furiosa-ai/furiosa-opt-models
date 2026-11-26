# Using the engine

This page loads a checkpoint onto one RNGD chip, runs a three-token prompt and then appends the next token in the same KV cache, without a serving framework.
Install either plugin first ([Installation](../getting-started/installation.md)); both install the `furiosa_opt_engine` package, and the [Python reference](../reference/python.md) lists every signature.

## Load a model

```python
from huggingface_hub import snapshot_download

import furiosa_opt_engine as engine

checkpoint = engine.Checkpoint(snapshot_download("Qwen/Qwen3-0.6B"))
options = engine.LoadOptions()
options.parse(checkpoint)  # (1)!
device = engine.Device(engine.Placement([0], range(8)))  # (2)!
decoder = options.load(checkpoint, device)  # (3)!
```

1. Checks the options against the checkpoint's config and tensor names, reading no weight and opening no device.
2. Opens chip 0 and its PEs 0 to 7.
3. Loads the weights into the device's HBM and returns a `qwen3.ForCausalLM`, a subclass of `Decoder`.

`decoder.output` is `Output.TOKENS`: a call returns the greedy token ID of each row it keeps.
One call takes at most `decoder.rows` rows, and a request reaches at most `decoder.window` positions.
Dropping `decoder` frees its device memory.

## Run a prompt

```python
cache = engine.decoder.Cache(decoder.cache, 1)  # (1)!
prompt = engine.HostTensor([642, 643, 644]).to_hbm(device)  # (2)!
token = decoder.forward([engine.decoder.Request([[0]], 0, prompt)], cache)  # (3)!
```

1. One block of every layer `decoder.cache` lists; a block holds `decoder.cache.pools[0].block` positions, more than this page writes.
2. Token IDs of an already tokenized prompt, copied to the device as `int32`.
3. Block `0` of the cache's one group, no position cached yet, three tokens appended. `Keep.LAST`, the default, returns the last row's greedy token.

!!! warning "Keep token IDs within the vocabulary"
    The engine never reads an ID on the host, and an ID outside the model's vocabulary reads other device memory.

## Append the next token

```python
following = decoder.forward([engine.decoder.Request([[0]], 3, token)], cache)
print(token.to_host().numpy(), following.to_host().numpy())
```

It prints one token ID from each call.
The second request appends `token` after the three positions the first call wrote, so `cached` is `3` and the block is the same: position `p` writes slot `p % block` of the request's block `p // block`.
The engine cannot check what the block holds: `cached` must count positions this sequence already wrote.
`token` stays on the device, so the next call takes it without a copy to the host.

## Plan framework batches

`furiosa_opt_engine.batch` provides shared host-side helpers for adapters that turn framework rows into decoder calls:

- `Batch.parse` snapshots a flat token buffer into per-sequence rows and cache positions.
- `Call.plan` groups contiguous rows by sequence and reports which output rows to read.
- `chunks` fairly partitions per-sequence row counts into calls within `decoder.rows`, with one run per sequence in each call.
- `operand` checks token IDs against the vocabulary and converts them to `int32` for upload.

The adapters use these helpers where their framework's request and cache representation fits, then map the planned rows to `engine.decoder.Request` and `Keep` values.

## Draft with another decoder

`furiosa_opt_engine.speculative.DraftSession` runs a causal draft model over framework-owned token
and block tables. An adapter supplies each request's full known-token source, the prefix the target
cache guarantees, one block table per draft-cache group and its requested draft count. A token source
can read lazily from framework storage, so the adapter need not copy the whole token matrix.

Without a mask, the session feeds each proposed token into the next decoder call. With a mask token,
it appends the known tokens and masks together, then reads all proposed rows from those calls. Both
paths split work fairly across `decoder.rows`, cap proposals at the decoder window and every cache
group's block capacity, and retain draft cache prefixes between calls. Call `retire` when the
framework releases a request's blocks; call `retain` when only a supplied set of requests remains
active, or `reset` after clearing the framework cache pool.

The framework still owns speculative configuration, target scheduling, verification and accepted
token commits. Its adapter translates request IDs, tokens and block tables to `DraftRequest`, then
maps proposed token IDs back to framework order.

The engine also owns the checks both adapters share: draft and target vocabulary size, causal token
output, paged-cache block layout and minimum context. `validate_mask_token` checks mask IDs after an
adapter applies its framework's checkpoint-field priority. Cache helpers calculate bytes and block
limits from engine cache specs; adapters still choose each framework's memory budget.

## Run calls asynchronously

`forward` returns its output once the device accepted the call, before the device finishes, as PyTorch and JAX calls do.
Pass an output straight to the next call; `to_host()` waits for the calls that write it and copies it, `wait()` waits without a copy, and `device.synchronize()` waits for every call submitted before it.
`HostTensor.to_hbm` ends its copy before it returns.

A copy between a `HostBuffer` and an `HbmBuffer`, such as a host snapshot of a cache layer, returns a `Transfer` once submitted:

```python
layer = cache.layers[0]  # (1)!
host = engine.HostBuffer(len(layer))  # (2)!
transfer = layer.to_host(host)
transfer.wait()
```

1. The first cached layer's device storage.
2. `len(layer)` bytes for each chip of the device, here one.

## Run requests with LoRA adapters

A decoder whose projections take LoRA adapters describes them as `decoder.adapters`, an `engine.decoder.AdapterSpec`, and any other decoder as `None`; a load for adapters, `engine.LoadOptions().adapters()`, refuses any other one from its config.
An `Adapter` is a PEFT checkpoint loaded onto the decoder's device, and a call takes a list of adapters that each request names one of by its index:

```python
import json

spec = decoder.adapters
config = json.loads(open("/path/to/adapter/adapter_config.json").read())
adapter = engine.decoder.Adapter(spec, engine.Checkpoint("/path/to/adapter", config))  # (1)!
request = engine.decoder.Request([[0]], 0, prompt, adapter=0)  # (2)!
token = decoder.forward([request], cache, [adapter])
```

1. The directory holds `adapter_model.safetensors`, opened with its `adapter_config.json` as its config; an adapter of a rank above `AdapterSpec.RANK`, or of a module other than an attention or MLP projection, raises when it loads, before any factor reaches the device.
2. The request runs with the adapter at index 0 of the list; one with `adapter=None` runs the base model alone.

A list holds at most `AdapterSpec.SLOTS` adapters, and each costs its updates on every row of the call.
An adapter never changes: a call reads it where it lies, so putting it in a slot copies nothing, and an adapter can leave its slot and return to any slot later.

## Errors and limits

`parse` and `load` raise `ValueError` for a config that no listed model serves and `NotImplementedError` for an architecture or option the model list refuses, before any weight is read.
`forward` raises `ValueError` for a request that breaks a rule of the call, such as naming a slot that holds no adapter, and `NotImplementedError` for one that asks for what the model does not serve, before anything reaches the device.
A launch that fails on the device raises `RuntimeError` from `forward`, or later from `wait()` or `to_host()` of a tensor it writes.
Then restart the process, load the model and replay each sequence's tokens into a new cache.
