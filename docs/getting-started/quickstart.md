# Quickstart

This page generates text with Qwen3-1.7B, first in a Python program and then over HTTP.
Install a plugin first; see [Installation](installation.md). The framework tabs switch together, so pick yours once.

## Generate in a Python program

Save the program as `generate.py`, then run `python generate.py`:

=== "vLLM"

    ```python
    from vllm import LLM, SamplingParams

    if __name__ == "__main__":
        model = LLM(model="Qwen/Qwen3-1.7B")
        result = model.generate(
            "The capital of Korea is",
            SamplingParams(temperature=0, max_tokens=12),
        )
        print(result[0].outputs[0].text)
    ```

=== "SGLang"

    ```python
    from sglang import Engine

    if __name__ == "__main__":
        model = Engine(
            model_path="Qwen/Qwen3-1.7B",
        )
        try:
            result = model.generate(
                "The capital of Korea is",
                sampling_params={"temperature": 0, "max_new_tokens": 12},
            )
            print(result["text"])
        finally:
            model.shutdown()
    ```

The first run downloads the checkpoint and uploads its weights to RNGD.
The program then prints the greedy continuation, for example:

```text
 Seoul. The capital of the United States is Washington, D
```

## Serve over HTTP

To expose the model as an OpenAI-compatible server:

=== "vLLM"

    <!-- --8<-- [start:vllm-serve] -->
    ```bash
    vllm serve Qwen/Qwen3-1.7B
    ```
    <!-- --8<-- [end:vllm-serve] -->

=== "SGLang"

    <!-- --8<-- [start:sglang-serve] -->
    ```bash
    python -m sglang.launch_server --model-path Qwen/Qwen3-1.7B
    ```
    <!-- --8<-- [end:sglang-serve] -->

To send a request from another shell once the server reports that it is ready:

=== "vLLM"

    ```bash
    curl http://127.0.0.1:8000/v1/completions \
      -H 'Content-Type: application/json' \
      -d '{"model": "Qwen/Qwen3-1.7B", "prompt": "The capital of Korea is", "max_tokens": 12, "temperature": 0}'
    ```

=== "SGLang"

    ```bash
    curl http://127.0.0.1:30000/v1/completions \
      -H 'Content-Type: application/json' \
      -d '{"model": "Qwen/Qwen3-1.7B", "prompt": "The capital of Korea is", "max_tokens": 12, "temperature": 0}'
    ```

The response's `choices[0].text` holds the generated continuation.

## Next steps

- [Serving](../serving/index.md): embeddings, reranking, an FP8 KV cache and long contexts.
- [Models](../models/index.md): which checkpoints run and how fast.
- [Using the engine](../engine/python.md): run a checkpoint without a framework.
