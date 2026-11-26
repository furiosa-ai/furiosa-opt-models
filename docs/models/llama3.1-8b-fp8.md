---
hide: [toc]
---

# Llama 3.1 8B FP8 dynamic

The checkpoint's E4M3 weights retain their per-output-channel scales. Activations use
dynamic per-token E4M3 quantization; each request retains its own BF16 KV cache.

Cross-request prefix caching must be disabled. Use `--no-enable-prefix-caching`
with vLLM or `--disable-radix-cache` with SGLang. One RNGD chip supports up to
40,960 context tokens, 1,024 new token rows and eight multi-token requests per launch.

{% include "benchmarks.md" %}
