"""vLLM's online FP8 quantization serves a BF16 checkpoint as vLLM quantizes it at load."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest
from transformers import PreTrainedTokenizerBase
from vllm import LLM, SamplingParams

if TYPE_CHECKING:
    from collections.abc import Generator

    import torch
    from transformers import PreTrainedModel

QUESTIONS: Final = (
    "What is the capital of France? Answer with only the city name.",
    "Name three primary colors.",
)
TOKENS: Final = 16


@pytest.fixture(scope="module", params=["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"])
def checkpoint(request: pytest.FixtureRequest) -> str:
    """Each Qwen3 size this quantization serves that RNGD verifies."""
    name = request.param
    assert isinstance(name, str)
    return name


@pytest.fixture(scope="module")
def llm(checkpoint: str) -> Generator[LLM]:
    llm = LLM(
        checkpoint,
        dtype="bfloat16",
        generation_config="vllm",
        max_model_len=256,
        quantization="fp8_per_tensor",
    )
    yield llm
    llm.llm_engine.engine_core.shutdown()


def quantized(weight: torch.Tensor, amax: torch.Tensor) -> torch.Tensor:
    """`weight` restored from its E4M3 codes under the tensor scale of `amax`, as vLLM's per-tensor
    online method makes them: the scale `amax / 448` in FP32, each code its value times the scale's
    reciprocal, saturated; the projection restores each code times the scale in BF16."""
    import torch

    largest = torch.tensor(torch.finfo(torch.float8_e4m3fn).max, dtype=torch.float32)
    scale = amax.to(torch.float32) / largest
    codes = (weight.float() * (1.0 / scale)).clamp(-largest, largest).to(torch.float8_e4m3fn)
    return (codes.float() * scale).to(torch.bfloat16)


@pytest.fixture(scope="module")
def reference(checkpoint: str) -> PreTrainedModel:
    """Transformers' `checkpoint` with every projection vLLM quantizes restored from its codes: one
    scale for each layer vLLM fuses, as its model's `packed_modules_mapping` groups them."""
    import torch
    from transformers import AutoModelForCausalLM
    from vllm.model_executor.layers.quantization.utils.quant_utils import weight_amax
    from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM

    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16)
    # vLLM's layers: each group of projections it fuses, and each projection it leaves alone.
    groups = [*Qwen3ForCausalLM.packed_modules_mapping.values(), ["o_proj"], ["down_proj"]]
    with torch.inference_mode():
        for layer in model.model.layers:
            modules = {name.rsplit(".", 1)[-1]: module for name, module in layer.named_modules()}
            for group in groups:
                parts = [modules[name] for name in group]
                amax = torch.stack([weight_amax(part.weight) for part in parts]).max().reshape(1)
                for part in parts:
                    part.weight.copy_(quantized(part.weight, amax))
    return model


@pytest.mark.feature("Online quantization")
def test_generates_as_the_checkpoint_quantized_by_vllm(
    llm: LLM, reference: PreTrainedModel, tokenizer: PreTrainedTokenizerBase
) -> None:
    """`--quantization fp8_per_tensor` on a BF16 checkpoint generates the greedy tokens of the
    checkpoint quantized as vLLM quantizes it, each within 5% of the reference's top probability."""
    import torch

    prompts: list[str] = []
    for question in QUESTIONS:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert isinstance(prompt, str)
        prompts.append(prompt)
    greedy = SamplingParams(temperature=0, max_tokens=TOKENS)
    for prompt, actual in zip(prompts, llm.generate(prompts, greedy), strict=True):
        tokens = tokenizer(prompt, return_tensors="pt").input_ids
        generated = actual.outputs[0].token_ids
        assert len(generated) > 0
        with torch.inference_mode():
            logits = reference(torch.cat([tokens, torch.tensor([generated])], dim=-1)).logits[0]
        expected = logits[tokens.shape[1] - 1 : -1].float().log_softmax(-1)
        # BF16 rounding can tie the two highest scores; compare each chosen probability.
        chosen = expected[torch.arange(len(generated)), torch.tensor(generated)]
        assert ((chosen - expected.max(-1).values).exp() >= 0.95).all(), (prompt, generated)
