"""vLLM serves a checkpoint past the context it was trained on, under the rotary scaling that its
overrides set, as Transformers computes the same configuration."""

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

CHECKPOINT = "Qwen/Qwen3-0.6B"
"""A checkpoint whose config declares 40960 positions, of the 131072 its decoder is compiled for."""
DECLARED = 40960
THETA = 1000000.0


def prompt(tokenizer: "PreTrainedTokenizerBase") -> list[int]:
    """A question about a fact stated before more context than the checkpoint declares."""
    fact = "The vault's password is 7381, and the gardener's name is Ottilie."
    lines = " ".join(
        f"Line {line}: the river ran slowly past mill {line % 97}." for line in range(3200)
    )
    ids = tokenizer(f"{fact} {lines} Recall the opening line. The gardener's name is").input_ids
    assert DECLARED < len(ids) < 3 * DECLARED
    return ids


@pytest.mark.feature("Context extension past the trained length")
@pytest.mark.parametrize(
    "overrides",
    [
        # Qwen's YaRN recipe: four times the 32768 positions Qwen3 was trained on, which a YaRN
        # checkpoint writes as its own context.
        pytest.param(
            {
                "rope_parameters": {
                    "rope_type": "yarn",
                    "factor": 4.0,
                    "original_max_position_embeddings": 32768,
                    "rope_theta": THETA,
                },
                "max_position_embeddings": 131072,
            },
            id="yarn",
        ),
        pytest.param(
            {"rope_parameters": {"rope_type": "linear", "factor": 3.2, "rope_theta": THETA}},
            id="linear",
        ),
    ],
)
def test_extended_context_matches_transformers(
    overrides: dict[str, object], tokenizer: "PreTrainedTokenizerBase"
) -> None:
    """Each greedy token after a prompt past the declared context is within 5% of the top
    probability in Transformers with the same configuration, and vLLM serves the context the
    overrides extend to."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from vllm import LLM, SamplingParams

    ids = prompt(tokenizer)
    llm = LLM(
        model=CHECKPOINT,
        dtype="bfloat16",
        hf_overrides=overrides,
        max_num_seqs=1,
        max_num_batched_tokens=2048,
    )
    try:
        assert llm.llm_engine.model_config.max_model_len == 131072
        greedy = SamplingParams(temperature=0, max_tokens=4)
        output = llm.generate({"prompt_token_ids": ids}, greedy)[0].outputs[0]
    finally:
        llm.llm_engine.engine_core.shutdown()
    config = AutoConfig.from_pretrained(CHECKPOINT, **overrides)
    reference = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT, config=config, dtype=torch.bfloat16
    )
    tokens, cache = torch.tensor([ids]), None
    with torch.inference_mode():
        # Each token extends the cache, as Transformers generates.
        for token in output.token_ids:
            step = reference(tokens, past_key_values=cache, use_cache=True, logits_to_keep=1)
            expected, cache = step.logits[0, -1].float().log_softmax(0), step.past_key_values
            # BF16 rounding can tie the two highest scores.
            assert (expected[token] - expected.max()).exp() >= 0.95
            tokens = torch.tensor([[token]])
