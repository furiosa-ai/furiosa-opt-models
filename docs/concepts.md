# How a model runs on RNGD

A checkpoint runs as model functions that were compiled for RNGD when the plugin was built.
Compilation fixes their shapes ahead of time, so a server accepts only what the compiled functions can run and says so when it starts.
The framework keeps everything else: it schedules requests, assigns cache blocks and samples tokens.

## Compiled ahead of time

Installing a plugin compiles every model function, which is why the first build takes minutes.
Loading a checkpoint loads those compiled functions and uploads the checkpoint's weights; serving compiles nothing.
A checkpoint loads when its config matches a compiled geometry, and the load refuses it otherwise, naming the compiled value it expected.

## Compiled limits

Four limits come from compilation:

| Limit | Bounds | Where you meet it |
| --- | --- | --- |
| Context window | Prompt and output tokens of one request | Startup names the window when the context length exceeds it |
| Rows per call | Tokens one device call processes, prompt chunks and decode rows together | The plugin splits a larger scheduler step into several calls |
| Cache block | Tokens per KV cache block | vLLM's `--block-size` must equal it, and the SGLang plugin sets the page size to it |
| Chips | Chips the model is compiled to split over | A refused tensor-parallel degree names the accepted ones |

A step split into several calls runs them one after another, so its time grows with its rows.

## The framework and the engine

The framework decides which requests run in each step; the plugin turns that step into engine calls, and the engine runs them on the device.
A step of greedy requests takes its tokens from the device; the plugins refuse a request that needs token scores or changes the token distribution, since a decoder's calls return no scores yet.
[Integrating frameworks](engine/frameworks.md) gives each framework's eligibility rules, and [Architecture](design/architecture.md#boundaries) states what each part owns.

## Calls are asynchronous

An engine call checks its inputs, submits its launches and returns device tensors before they finish, as PyTorch and JAX calls do.
Only copies to the host and explicit waits block, so the next step is prepared while the device still runs the current one.
[Using the engine](engine/python.md#run-calls-asynchronously) lists the calls that wait.

## Refusals come first

Most refusals come before the device opens: unknown architectures, unsupported options, cache dtypes and placements.
The context window is checked once the model has loaded.
A refusal names the option or value and what is accepted; [Troubleshooting](reference/troubleshooting.md) lists the common ones with their fixes.
