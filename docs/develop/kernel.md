# Writing a kernel

Write the device kernel of a leaf operation that a model needs and the models crate does not implement yet.
The example is `Add`; to build a model from operations that already have kernels, see [Model composition](compose.md) and [Adding a model](add-model.md).
Kernels are written in the Virtual ISA (VISA) of the [furiosa-opt SDK](https://github.com/furiosa-ai/furiosa-opt), whose documentation covers the tensors, mappings and engines on this page.

## Define the operation

Leaf operations live in `crates/furiosa-opt-models/src/ops/`.
They are the only modules that touch hardware ([Four kinds of module](compose.md#four-kinds-of-module)).
[`ops/mod.rs`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-models/src/ops/mod.rs) declares the modules in groups, such as elementwise functions, or attention and sequence mixing.
Declare a new file in its group with `pub mod <name>;`.

An operation computes one mathematical function, such as a linear, a softmax or an add.
It is a struct with one `Op` impl per operand type it accepts:

- The struct holds the operation's parameters, such as `RmsNorm`'s weight, and derives `DeviceSend`, so a `#[device]` entry can take it as an argument.
- Each impl takes the device and the operands, and names its result type `Output`. The blanket impl makes the operation a module of any env ([The `Module` contract](compose.md#the-module-contract)); an operation that reads the call's env or has a module child implements `Module` instead.
- The operand types fix each tensor's memory, element type and mapping, so an operand in another layout fails to compile.
  An `HbmTensor` lives in HBM, the device's DRAM, and a `DmTensor` in DM, its on-chip SRAM.
- Axis sizes are type parameters, such as `Add`'s `Hidden` and `Rows`, and the model definition fixes them.
- The doc comment states the mathematics and its precision, and the test checks the kernel against it.
- To load as part of a model, the struct derives `Load`, so each weight loads from the checkpoint tensor its field names; an operation without weights derives it too and reads nothing ([Weights and loading](compose.md#weights-and-loading)).

Give one computation one operation: what varies between its callers is a type or a value the operation takes ([Variant selection](compose.md#variant-selection)).

- Take a weight's storage and numerical contract as a type of `tensor/form`, not as an operation for each precision.
- Read a strategy or a tiling as a value of `specialize/`, not as a second implementation.
- Take a size as a type parameter, so the body has no branch on a size, a model or a layer count.
- Take the rows of a launch as a `Live<X, R>` of `tensor/live.rs`, whose live rows bound the loops, not as a row type of the operation's own.

## Example: `Add`

`Add` has one impl per operand layout, and the input type picks the one that runs:

```rust
--8<-- "crates/furiosa-opt-models/src/ops/pointwise/add.rs:add"
```

- HBM tensors reuse the elementwise kernel of `ops/pointwise/mod.rs`, which `Multiply` shares.
- DM rows (`Spread`) are the kernel itself.
  The sub context loads `rhs` into the vector register file, and the main context streams `lhs` through the vector engine, adds `rhs`, casts the sum to bf16 and commits it to DM.
- The vector engine computes on 32-bit lanes, so both terms are cast to f32 as they are fetched.
- Rows of one launch (`Live`) unwrap to the DM impl, and the sum keeps their live rows.

The DM chain passes the data through the fetch, collect, vector, cast and commit engines of the Tensor Unit; the SDK documents each.

## Operations without a kernel

An operation without a kernel is a typed contract whose `forward` is `todo!()`, usually with a message that describes the kernel to write.
To list these operations, search the crate:

```bash
grep -rn 'todo!(' crates/furiosa-opt-models/src
```

- `todo!()` type-checks as any output, so a family that uses the operation still compiles and loads.
- The family's first launch panics on the device, and the panic reaches the caller that awaits the launch.
- [Enabling a model family](../design/architecture.md#enabling-a-model-family) explains why leaf operations are the bottleneck.

Write the kernel in place of the `todo!()`.
Keep the impl's signature, which the compositions that use the operation compile against.

## Test the kernel on the chip

A leaf's `forward` compiles into the `#[device]` entry that launches it, so a test launches the operation through an entry too.
Write the test in a `#[cfg(test)] mod tests` of the operation's file, following `adds_elementwise` in [`ops/pointwise/add.rs`](https://github.com/furiosa-ai/furiosa-opt-models/blob/{{ git.commit }}/crates/furiosa-opt-models/src/ops/pointwise/add.rs).
Its entry, `summed`, moves the operands from HBM into the DM rows `Add` takes and writes the sum back to an HBM output that the host can read:

```rust title="crates/furiosa-opt-models/src/ops/pointwise/add.rs"
--8<-- "crates/furiosa-opt-models/src/ops/pointwise/add.rs:summed"
```

The test body runs on the host:

1. Fix the axis sizes with `axes!`, here `Pinned = 1024`, so the entry compiles for one shape.
2. Make the test a `#[tokio::test(flavor = "current_thread")]`, because uploads and launches are awaited.
3. Open the device with `Device::new(summed.topology())`, then upload the operands and a zeroed output with `to_hbm(&mut device.pdma)`.
4. Compute the expected values on the host from the doc comment's mathematics: `adds_elementwise` sums the bf16 terms in f32.
5. `launch` the entry, read the output back with `to_host`, and compare each element within a tolerance that the contract's rounding allows ([Assert against an oracle](../design/testing.md#assert-against-an-oracle)).

When the contract fixes the rounding, compare bits, as `sigmoid_rounds_once` in `ops/pointwise/activation.rs` does.
Hold `crate::tests::DEVICE` while a test uses the chip, as that test does, because device tests share one chip and run one at a time.

To run the test, use an RNGD device with `cargo-furiosa-opt` on your `PATH` ([Installation](../getting-started/installation.md#install-the-compiler)) and name the test's path:

```bash
cargo furiosa-opt test -p furiosa-opt-models --release --lib \
  ops::pointwise::add::tests::adds_elementwise -- --test-threads 1
```

- The first run compiles every `#[device]` entry the crate's tests reach; later runs recompile only the entries that changed.
- CI runs every models test on RNGD for each pull request.

## Lowering traps

The CPU backend that a plain `cargo` build selects checks shape algebra but not how memory moves, so a wrong kernel can pass it.

- **HBM-to-DM source strides must be 8-byte aligned.** The SDK's layout assertion checks only the destination, so the CPU backend accepts a misaligned source.
- **A collect packet is one 32-byte flit**, such as the eight f32 lanes `Add` collects. A wider packet does not lower.
- **Widening a vector to add lanes emits no instruction.** After `vector_widen_pad`, the new lanes hold stale values on the chip and zero on the CPU backend, so the result is a plausible, wrong token.

The first two stop the `cargo furiosa-opt` build, and only values read back from the chip show the third.
Run the operation's test, then the model's tests ([Test a model](testing.md#test-a-model)).
