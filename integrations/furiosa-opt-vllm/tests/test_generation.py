"""vLLM generation returns the checkpoint's answer under chunked prefill, caching and sampling."""

from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase
    from vllm import LLM, RequestOutput

TASK = "text"
QUESTION = "What is the capital of France? Answer with only the city name."
BUDGET = 32
"""Tokens per scheduler step; a prompt longer than this is prefilled in chunks."""


@pytest.fixture(scope="module")
def config(row: dict[str, Any]) -> "PretrainedConfig":
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(row["id"])


@pytest.fixture(scope="module")
def window(config: "PretrainedConfig") -> int:
    """The checkpoint's sliding attention window, or 0 without one."""
    return getattr(config.get_text_config(), "sliding_window", None) or 0


@pytest.fixture(scope="module")
def tokenizer(row: dict[str, Any]) -> "PreTrainedTokenizerBase":
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(row["id"])


@pytest.fixture(scope="module")
def llm(options: dict[str, Any], window: int) -> Generator["LLM"]:
    from vllm import LLM

    # Twice the window lets a prompt pass the sliding window; a documented context, which is the
    # checkpoint's own when the row names none, is only ever raised to that.
    documented = options.get("max_model_len")
    context = {"max_model_len": 2 * window} if documented and 0 < documented < 2 * window else {}
    llm = LLM(
        **options
        | context
        | {
            "max_num_seqs": 2,
            "max_num_batched_tokens": BUDGET,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "enable_prompt_embeds": True,
            "max_logprobs": -1,
        }
    )
    yield llm
    llm.llm_engine.engine_core.shutdown()


@pytest.fixture(scope="module")
def reference(row: dict[str, Any], config: "PretrainedConfig") -> "PreTrainedModel":
    # Loading a checkpoint, vLLM registers its own configuration class for some architectures;
    # the reference builds from the one Transformers parsed.
    return reference_model(row["id"], config)


def render(tokenizer: "PreTrainedTokenizerBase", text: str) -> str:
    """The chat prompt for one user message; checkpoints without a chat template take it as is."""
    if not tokenizer.chat_template:
        return text
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def long_prompt(tokenizer: "PreTrainedTokenizerBase", window: int) -> str:
    """A question after a context spanning many prefill chunks and cache blocks, and the window."""
    context = "Explain briefly. " * 180
    while len(tokenizer(context).input_ids) <= window:
        context += context
    return render(tokenizer, context + QUESTION)


@pytest.mark.feature("Text generation")
@pytest.mark.feature("FP8 KV cache")
@pytest.mark.feature("Quantized checkpoints (FP8, NVFP4)")
def test_next_token_matches_transformers(
    llm: "LLM",
    reference: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    config: "PretrainedConfig",
) -> None:
    import torch
    from vllm import SamplingParams

    prompt = render(tokenizer, QUESTION)
    actual = llm.generate(prompt, SamplingParams(temperature=0, max_tokens=4, logprobs=-1))[0]
    tokens = tokenizer(prompt, return_tensors="pt").input_ids
    assert actual.outputs[0].logprobs is not None
    with torch.inference_mode():
        for token, probabilities in zip(
            actual.outputs[0].token_ids, actual.outputs[0].logprobs, strict=True
        ):
            assert probabilities is not None
            expected = reference(tokens).logits[0, -1].float()
            measured = torch.tensor(
                [probabilities[index].logprob for index in range(expected.numel())]
            )
            if getattr(config, "quantization_config", None):
                # Bound the sampler's distribution error after dequantizing stored weights.
                assert (measured.exp() - expected.softmax(0)).abs().sum() / 2 < 0.02
            else:
                # Centering removes log-softmax's shared offset from the framework output.
                assert (
                    torch.nn.functional.cosine_similarity(
                        measured - measured.mean(), expected - expected.mean(), dim=0
                    )
                    > 0.9995
                )
            # BF16 rounding can tie the two highest scores; compare the chosen probability.
            assert (expected[token] - expected.max()).exp() >= 0.95
            tokens = torch.cat([tokens, torch.tensor([[token]])], dim=-1)


@pytest.mark.feature("Prompt embeddings")
def test_prompt_embeds_match_token_ids(
    llm: "LLM", reference: "PreTrainedModel", tokenizer: "PreTrainedTokenizerBase"
) -> None:
    import torch
    from vllm import SamplingParams

    tokens = tokenizer(render(tokenizer, QUESTION), return_tensors="pt").input_ids
    # Longer than one step's budget, so both requests are prefilled in chunks.
    tokens = tokens.repeat(1, 1 + BUDGET // tokens.shape[1])
    with torch.inference_mode():
        embeddings = reference.get_input_embeddings()(tokens).squeeze(0)
        embedded, listed = llm.generate(
            [{"prompt_embeds": embeddings}, {"prompt_token_ids": tokens[0].tolist()}],
            SamplingParams(temperature=0, max_tokens=4),
        )
    assert embedded.outputs[0].token_ids == listed.outputs[0].token_ids


@pytest.mark.feature("Long contexts, up to the compiled window")
def test_long_context_matches_transformers(
    llm: "LLM", reference: "PreTrainedModel", tokenizer: "PreTrainedTokenizerBase", window: int
) -> None:
    import torch
    from vllm import SamplingParams

    prompt = long_prompt(tokenizer, window)
    actual = llm.generate(prompt, SamplingParams(temperature=0, max_tokens=1))[0]
    with torch.inference_mode():
        expected = reference(tokenizer(prompt, return_tensors="pt").input_ids).logits[0, -1]
    expected = expected.float()
    assert (expected[actual.outputs[0].token_ids[0]] - expected.max()).exp() >= 0.95


@pytest.mark.feature("Automatic prefix caching")
def test_prefix_cache(llm: "LLM", tokenizer: "PreTrainedTokenizerBase", window: int) -> None:
    from vllm import SamplingParams

    prompt = long_prompt(tokenizer, window)
    greedy = SamplingParams(temperature=0, max_tokens=4)
    # Earlier tests may have cached this prompt; the first request below must miss.
    assert llm.reset_prefix_cache()
    first = llm.generate(prompt, greedy)[0]
    cached = llm.generate(prompt, greedy)[0]
    assert cached.outputs[0].token_ids == first.outputs[0].token_ids
    assert cached.num_cached_tokens is not None and cached.num_cached_tokens > 0


def test_logprobs_keep_greedy_tokens(
    llm: "LLM", tokenizer: "PreTrainedTokenizerBase", window: int
) -> None:
    from vllm import SamplingParams

    # Requesting logprobs moves greedy decoding from the model call to host sampling;
    # both must choose the same tokens.
    prompt = long_prompt(tokenizer, window)
    argmax = llm.generate(prompt, SamplingParams(temperature=0, max_tokens=4))[0]
    scored = llm.generate(prompt, SamplingParams(temperature=0, max_tokens=4, logprobs=3))[0]
    assert scored.outputs[0].token_ids == argmax.outputs[0].token_ids
    assert scored.outputs[0].logprobs


def test_sampling_and_logprobs(llm: "LLM", tokenizer: "PreTrainedTokenizerBase") -> None:
    from vllm import SamplingParams

    prompts = [
        render(tokenizer, "What is the capital of France?"),
        render(tokenizer, "What is the capital of Japan?"),
    ]
    params = SamplingParams(
        temperature=0.8, seed=7, frequency_penalty=0.2, max_tokens=4, logprobs=3, prompt_logprobs=2
    )
    sampled = llm.generate(prompts, params)
    # A seeded request draws the same tokens again.
    repeated = llm.generate(prompts, params)
    assert [result.outputs[0].token_ids for result in repeated] == [
        result.outputs[0].token_ids for result in sampled
    ]
    for result in sampled:
        output = result.outputs[0]
        assert output.token_ids
        assert output.logprobs is not None and result.prompt_logprobs is not None
        # Each step lists the requested top logprobs, plus the chosen token when not among them.
        assert len(output.logprobs) == len(output.token_ids)
        for token, step in zip(output.token_ids, output.logprobs, strict=True):
            assert token in step and len(step) in (3, 4)
        assert len(result.prompt_logprobs) == len(result.prompt_token_ids or [])
        first, *rest = result.prompt_logprobs
        assert first is None
        assert all(step is not None and len(step) in (2, 3) for step in rest)


@pytest.mark.feature("Structured outputs")
def test_structured_choice(llm: "LLM", tokenizer: "PreTrainedTokenizerBase") -> None:
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    constrained = llm.generate(
        render(tokenizer, "What is the capital of France?"),
        SamplingParams(
            temperature=0,
            max_tokens=8,
            structured_outputs=StructuredOutputsParams(choice=["Paris", "Tokyo"]),
        ),
    )[0]
    assert constrained.outputs[0].text in {"Paris", "Tokyo"}


def test_min_p_of_one_samples_the_greedy_tokens(
    llm: "LLM", tokenizer: "PreTrainedTokenizerBase"
) -> None:
    from vllm import SamplingParams

    prompt = render(tokenizer, QUESTION)
    greedy = llm.generate(prompt, SamplingParams(temperature=0, max_tokens=4))[0]
    sampled = llm.generate(prompt, SamplingParams(temperature=1, seed=7, max_tokens=4, min_p=1))[0]
    assert sampled.outputs[0].token_ids == greedy.outputs[0].token_ids


def test_logit_bias_and_allowed_tokens_bound_the_choice(
    llm: "LLM", tokenizer: "PreTrainedTokenizerBase"
) -> None:
    from vllm import SamplingParams

    prompt = render(tokenizer, QUESTION)
    first, second = (
        tokenizer.encode(word, add_special_tokens=False)[0] for word in (" red", " blue")
    )
    biased, allowed = llm.generate(
        [prompt, prompt],
        [
            SamplingParams(temperature=0, max_tokens=4, logit_bias={first: 100}),
            SamplingParams(temperature=0, max_tokens=4, allowed_token_ids=[first, second]),
        ],
    )
    assert set(biased.outputs[0].token_ids) == {first}
    assert set(allowed.outputs[0].token_ids) <= {first, second}


def test_bad_words_never_appear(llm: "LLM", tokenizer: "PreTrainedTokenizerBase") -> None:
    from vllm import SamplingParams

    prompt = render(tokenizer, QUESTION)
    free = llm.generate(prompt, SamplingParams(temperature=0, max_tokens=8))[0]
    if "Paris" not in free.outputs[0].text:
        pytest.skip("the checkpoint does not answer with the banned word")
    banned = llm.generate(
        prompt, SamplingParams(temperature=0, max_tokens=8, bad_words=["Paris", " Paris"])
    )[0]
    assert "Paris" not in banned.outputs[0].text


def test_min_tokens_hold_back_the_stop(llm: "LLM", tokenizer: "PreTrainedTokenizerBase") -> None:
    from vllm import SamplingParams

    prompt = render(tokenizer, QUESTION)
    free = llm.generate(prompt, SamplingParams(temperature=0, max_tokens=32))[0]
    least = len(free.outputs[0].token_ids) + 4
    if free.outputs[0].finish_reason != "stop" or least > 32:
        pytest.skip("the checkpoint does not stop early on this prompt")
    held = llm.generate(prompt, SamplingParams(temperature=0, max_tokens=32, min_tokens=least))[0]
    assert len(held.outputs[0].token_ids) >= least


@pytest.mark.parametrize("bound", [{"top_k": 1}, {"top_p": 1e-6}], ids=["top_k", "top_p"])
def test_one_kept_token_samples_the_greedy_tokens(
    llm: "LLM", tokenizer: "PreTrainedTokenizerBase", bound: dict[str, float]
) -> None:
    from vllm import SamplingParams

    prompt = render(tokenizer, QUESTION)
    greedy = llm.generate(prompt, SamplingParams(temperature=0, max_tokens=4))[0]
    sampled = llm.generate(prompt, SamplingParams(temperature=1, seed=7, max_tokens=4, **bound))[0]
    assert sampled.outputs[0].token_ids == greedy.outputs[0].token_ids


def test_logprob_token_ids_match_the_full_distribution(
    llm: "LLM", tokenizer: "PreTrainedTokenizerBase"
) -> None:
    from vllm import SamplingParams

    prompt = render(tokenizer, QUESTION)
    asked = [tokenizer.encode(word, add_special_tokens=False)[0] for word in (" Paris", " Tokyo")]

    def generate(scored: dict[str, Any]) -> "RequestOutput":
        # A cached prefix leaves the prompt's last rows to a smaller call, which rounds them
        # otherwise, so each request computes the whole prompt.
        assert llm.reset_prefix_cache()
        return llm.generate(prompt, SamplingParams(temperature=0, max_tokens=4, **scored))[0]

    # vLLM answers every request of a step with the ids when one asks for them; ask apart.
    full, chosen = map(generate, ({"logprobs": -1}, {"logprob_token_ids": asked}))
    assert chosen.outputs[0].token_ids == full.outputs[0].token_ids
    assert chosen.outputs[0].logprobs is not None and full.outputs[0].logprobs is not None
    for token, step, every in zip(
        chosen.outputs[0].token_ids,
        chosen.outputs[0].logprobs,
        full.outputs[0].logprobs,
        strict=True,
    ):
        assert set(step) == {*asked, token}
        assert all(step[index].logprob == pytest.approx(every[index].logprob) for index in step)


def test_structured_json_and_regex(llm: "LLM", tokenizer: "PreTrainedTokenizerBase") -> None:
    import json
    import re

    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    schema = {
        "type": "object",
        "properties": {"city": {"type": "string", "maxLength": 16}},
        "required": ["city"],
        "additionalProperties": False,
    }
    pattern = "[A-Z][a-z]{1,15}"
    prompt = render(tokenizer, "What is the capital of France? Answer in JSON.")
    shaped, matched = llm.generate(
        [prompt, prompt],
        [
            SamplingParams(
                temperature=0,
                max_tokens=32,
                structured_outputs=StructuredOutputsParams(json=schema),
            ),
            SamplingParams(
                temperature=0,
                max_tokens=16,
                structured_outputs=StructuredOutputsParams(regex=pattern),
            ),
        ],
    )
    assert set(json.loads(shaped.outputs[0].text)) == {"city"}
    assert re.fullmatch(pattern, matched.outputs[0].text)


def test_parallel_samples_are_seeded_children(
    llm: "LLM", tokenizer: "PreTrainedTokenizerBase"
) -> None:
    from vllm import SamplingParams

    prompt = render(tokenizer, QUESTION)
    parallel = llm.generate(prompt, SamplingParams(n=3, temperature=1, seed=7, max_tokens=4))[0]
    # vLLM seeds child request i of a seeded request with the seed plus i.
    alone = llm.generate(
        [prompt] * 3, [SamplingParams(temperature=1, seed=7 + i, max_tokens=4) for i in range(3)]
    )
    assert [output.token_ids for output in parallel.outputs] == [
        result.outputs[0].token_ids for result in alone
    ]


def test_presence_and_frequency_penalties_lower_repeated_tokens(
    llm: "LLM", tokenizer: "PreTrainedTokenizerBase"
) -> None:
    from collections import Counter

    import torch
    from vllm import SamplingParams

    prompt = render(tokenizer, "Write a short poem about the sea.")
    free = llm.generate(prompt, SamplingParams(temperature=0, max_tokens=64))[0]
    # The largest penalties vLLM accepts, so that they change some greedy choice.
    presence, frequency = 2.0, 2.0
    penalized = llm.generate(
        prompt,
        SamplingParams(
            temperature=0,
            max_tokens=64,
            logprobs=-1,
            presence_penalty=presence,
            frequency_penalty=frequency,
        ),
    )[0].outputs[0]
    assert penalized.token_ids != free.outputs[0].token_ids
    assert penalized.logprobs is not None
    for at, (token, step) in enumerate(zip(penalized.token_ids, penalized.logprobs, strict=True)):
        # Raw logprobs differ from the logits by one offset a step, which penalties keep.
        scores = torch.tensor([step[index].logprob for index in range(len(step))])
        for seen, count in Counter(penalized.token_ids[:at]).items():
            scores[seen] -= presence + frequency * count
        # BF16 rounding can tie the two highest scores; compare the chosen probability.
        assert (scores[token] - scores.max()).exp() >= 0.95


def test_repetition_penalty_applies_to_its_request_alone(
    llm: "LLM", tokenizer: "PreTrainedTokenizerBase"
) -> None:
    from vllm import SamplingParams

    prompt = render(tokenizer, "Repeat the word hello ten times.")
    other = render(tokenizer, QUESTION)
    free = SamplingParams(temperature=0, max_tokens=8)
    penalized = SamplingParams(temperature=0, max_tokens=8, repetition_penalty=100.0)
    alone = llm.generate(prompt, penalized)[0]
    together, beside = llm.generate([prompt, other], [penalized, free])
    assert alone.outputs[0].token_ids != llm.generate(prompt, free)[0].outputs[0].token_ids
    assert together.outputs[0].token_ids == alone.outputs[0].token_ids
    assert beside.outputs[0].token_ids == llm.generate(other, free)[0].outputs[0].token_ids


def reference_model(checkpoint: str, config: "PretrainedConfig") -> "PreTrainedModel":
    """Evaluate checkpoint quantization with independent Torch operations."""
    from contextlib import ExitStack

    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    from transformers.initialization import no_init_weights
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

    class Linear(torch.nn.Module):
        def __init__(self, weight, input_scale, weight_scale, bias):
            super().__init__()
            self.register_buffer("weight", weight)
            self.register_buffer("input_scale", input_scale)
            self.register_buffer("weight_scale", weight_scale)
            self.register_buffer("bias", bias)

        def forward(self, value):
            limit = torch.finfo(torch.float8_e4m3fn).max
            quantized = (value.float() / self.input_scale).clamp(-limit, limit)
            quantized = quantized.to(torch.float8_e4m3fn).float()
            result = torch.nn.functional.linear(quantized, self.weight.float())
            result = result * self.input_scale * self.weight_scale
            if self.bias is not None:
                result = result + self.bias.float()
            return result.to(value.dtype)

    class Nvfp4(torch.nn.Module):
        """Inputs quantized as the weights are, by vLLM's reference NVFP4 quantizer; the FP32
        sum takes both tensor scales once."""

        def __init__(self, weight, input_scale, weight_scale_2, bias):
            super().__init__()
            self.register_buffer("weight", weight)
            self.register_buffer("input_scale", input_scale)
            self.register_buffer("weight_scale_2", weight_scale_2)
            self.register_buffer("bias", bias)

        def forward(self, value):
            from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
                ref_nvfp4_quant,
            )

            rows = value.reshape(-1, value.shape[-1])
            codes, scales = ref_nvfp4_quant(rows, 1 / self.input_scale, group)
            quantized = codes.view(len(rows), -1, group) * scales.view(len(rows), -1, 1)
            result = torch.nn.functional.linear(quantized.view(rows.shape), self.weight.float())
            result = result * (self.input_scale * self.weight_scale_2)
            if self.bias is not None:
                result = result + self.bias.float()
            return result.to(value.dtype).view(*value.shape[:-1], -1)

    if not getattr(config, "quantization_config", None):
        # A multimodal checkpoint's language model answers text alone through its image-text class.
        auto = (
            AutoModelForImageTextToText if hasattr(config, "text_config") else AutoModelForCausalLM
        )
        return auto.from_pretrained(checkpoint, dtype=torch.bfloat16).eval()
    directory = (
        Path(checkpoint) if Path(checkpoint).is_dir() else Path(snapshot_download(checkpoint))
    )
    language = config.get_text_config()
    # A language model without a causal LM class of its own runs in its multimodal class, whose
    # parameters keep the checkpoint's names.
    standalone = language.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    with no_init_weights():
        model = (
            AutoModelForCausalLM.from_config(language, dtype=torch.bfloat16)
            if standalone
            else AutoModelForImageTextToText.from_config(config, dtype=torch.bfloat16)
        )
    # NVFP4 stores FP4 E2M1 codes (sign bit, then magnitude) with one FP8 scale per 16 elements.
    e2m1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
    group = 16
    state = {}
    with ExitStack() as stack:
        sources = {
            name: source
            for path in sorted(directory.glob("model*.safetensors"))
            for source in [stack.enter_context(safe_open(path, framework="pt", device="cpu"))]
            for name in source.keys()
        }

        def read(name: str) -> "torch.Tensor":
            return sources[name].get_tensor(name)

        for name in model.state_dict():
            if name == "lm_head.weight" and config.get_text_config().tie_word_embeddings:
                state[name] = state["model.embed_tokens.weight"]
                continue
            key = (
                name.replace("model.", "model.language_model.", 1)
                if standalone and hasattr(config, "text_config")
                else name
            )
            prefix = key.removesuffix(".weight")
            if prefix + ".weight_scale_2" in sources:
                packed = read(f"{prefix}.weight")
                codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(packed.shape[0], -1)
                scale = read(f"{prefix}.weight_scale").float().repeat_interleave(group, -1)
                # Codes times E4M3 scales hold at most six significant bits, exact in BF16.
                weight = (e2m1[codes.long()] * scale).to(torch.bfloat16)
                module = name.removesuffix(".weight")
                scales = {
                    part: read(f"{prefix}.{part}").float()
                    for part in ("input_scale", "weight_scale_2")
                }
                model.set_submodule(
                    module, Nvfp4(weight, *scales.values(), model.get_submodule(module).bias)
                )
                state[name] = weight
                state |= {f"{module}.{part}": scale for part, scale in scales.items()}
                continue
            if prefix + ".weight_packed" in sources:
                packed = sources[prefix + ".weight_packed"].get_tensor(prefix + ".weight_packed")
                codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(packed.shape[0], -1)
                scale = sources[prefix + ".weight_scale"].get_tensor(prefix + ".weight_scale")
                global_scale = sources[prefix + ".weight_global_scale"].get_tensor(
                    prefix + ".weight_global_scale"
                )
                value = (
                    e2m1[codes.long()] * scale.float().repeat_interleave(group, -1) / global_scale
                )
            else:
                value = sources[key].get_tensor(key)
                if value.dtype == torch.float8_e4m3fn:
                    scale = sources[prefix + ".weight_scale"].get_tensor(prefix + ".weight_scale")
                    if prefix + ".input_scale" in sources:
                        input_scale = sources[prefix + ".input_scale"].get_tensor(
                            prefix + ".input_scale"
                        )
                        module = name.removesuffix(".weight")
                        model.set_submodule(
                            module,
                            Linear(value, input_scale, scale, model.get_submodule(module).bias),
                        )
                        state[name] = value
                        state[module + ".input_scale"] = input_scale
                        state[module + ".weight_scale"] = scale
                        continue
                    if scale.dim() == value.dim():
                        # Each scale covers an equal block of the weight's rows and columns.
                        for axis, count in enumerate(scale.shape):
                            scale = scale.repeat_interleave(value.shape[axis] // count, axis)
                    value = value.float() * scale.float()
            state[name] = value.to(torch.bfloat16)
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval()
