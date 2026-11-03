---
hide: [toc]
---

# Llama 3.1 8B NVFP4

{% include "benchmarks.md" %}

NVFP4 weights run with BF16 activations and BF16 KV cache. The checkpoint's activation and
FP8 cache calibration scales are validated but are not applied. The served context is 40960 tokens.
