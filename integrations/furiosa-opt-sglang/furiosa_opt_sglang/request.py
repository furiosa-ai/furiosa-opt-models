"""Translate framework requests into the checkpoint's input contract."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Final, Literal, assert_never, cast

from jinja2.sandbox import ImmutableSandboxedEnvironment
from pydantic import Field
from sglang.srt.entrypoints.openai.protocol import RerankContent
from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
from sglang.srt.sampling.sampling_params import SamplingParams

from furiosa_opt_sglang.batch import admit
from furiosa_opt_sglang.platform import Served, Unset

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    from transformers import PretrainedConfig


# Every generation request reads its tokens from a decoder call, which returns no scores.
GREEDY: Final = "serves only greedy decoding, without logprobs or logit changes"
EMBEDS: Final = "takes token IDs, not input embeddings"
OVERRIDES: Final = "takes token IDs, without embedding overrides"


class Generation(Served):
    """The fields of a generation request the plugin serves."""

    sampling_params: dict[str, object] = Field(
        description="requires per-request sampling parameters"
    )
    return_hidden_states: Unset
    session_params: Unset
    # SGLang dispatches on these fields' presence, so even an empty value counts.
    token_indices_to_pool: None
    input_embeds: None = Field(description=EMBEDS)
    positional_embed_overrides: None = Field(description=OVERRIDES)
    image_data: Unset
    video_data: Unset
    return_logprob: Unset = Field(description=GREEDY)
    return_sampling_mask: Unset = Field(description=GREEDY)
    custom_logit_processor: Unset = Field(description=GREEDY)


class Greedy(Served):
    """The sampling parameters the plugin serves: plain greedy decoding, the token a decoder call
    selects on the device."""

    # SGLang turns a zero temperature into top_k 1.
    top_k: Literal[1] = Field(description=f"requires temperature 0 or top_k 1: {GREEDY}")
    repetition_penalty: Literal[1] = Field(description=GREEDY)
    frequency_penalty: Unset = Field(description=GREEDY)
    presence_penalty: Unset = Field(description=GREEDY)
    min_new_tokens: Unset = Field(description=GREEDY)
    # SGLang biases the scores of a request given any logit bias, even an empty one.
    logit_bias: None = Field(description=GREEDY)
    regex: Unset = Field(description=GREEDY)
    json_schema: Unset = Field(description=GREEDY)
    ebnf: Unset = Field(description=GREEDY)
    structural_tag: Unset = Field(description=GREEDY)
    beam_width: Unset = Field(description=GREEDY)
    watermark: Unset = Field(description=GREEDY)


class Embedding(Served):
    """The fields of an embedding request the plugin serves."""

    # SGLang dispatches on these fields' presence, so even an empty value counts.
    token_indices_to_pool: None
    decision_layout: None
    input_embeds: None = Field(description=EMBEDS)
    positional_embed_overrides: None = Field(description=OVERRIDES)
    embed_overrides: None = Field(description=OVERRIDES)
    image_data: Unset
    video_data: Unset
    audio_data: Unset


def validate_request(
    manager: TokenizerManager,
    obj: GenerateReqInput | EmbeddingReqInput,
    input_ids: list[int] | None,
) -> None:
    """Refuse a request before tokenization when it carries inputs the plugin does not serve."""
    match obj:
        case GenerateReqInput(audio_data=audio):
            generation = Generation.parse(obj)
            # SGLang's sampling parameters parse the request's untyped values over the server's
            # preferred ones, which it declares a string and parses from JSON, as tokenization does.
            sampling = cast("Callable[..., SamplingParams]", SamplingParams)
            preferred = cast("dict[str, object] | None", manager.preferred_sampling_params) or {}
            _ = Greedy.parse(sampling(**(preferred | generation.sampling_params)))
        case EmbeddingReqInput():
            audio = None
            _ = Embedding.parse(obj)
        case _:
            assert_never(obj)
    acoustic = manager.model_config.is_encoder_decoder and hasattr(
        # SGLang leaves the checkpoint's Transformers config untyped.
        cast("PretrainedConfig", manager.model_config.hf_config),
        "num_mel_bins",
    )
    if audio and not acoustic:
        raise ValueError("audio inputs require an acoustic encoder-decoder model")
    if not input_ids and not audio:
        raise ValueError("furiosa-opt-sglang requires token IDs or audio")
    # SGLang admits any token ID, and the engine's own refusal would stop the scheduler while it
    # runs a step; audio requests carry SGLang's placeholder IDs instead.
    if input_ids and not audio:
        admit(input_ids, manager.model_config.vocab_size)


def render(
    chat_template: str,
    *,
    query: RerankContent,
    document: RerankContent,
    instruct: str | None,
) -> str:
    """Render a rerank query and document through the checkpoint's chat template.

    The checkpoint templates read `query` and `document` roles and an optional system instruction,
    unlike SGLang's generic renderer, so the plugin replaces it."""

    def text(content: RerankContent) -> str:
        match content:
            case str():
                return content
            case list():
                return "".join(part.text for part in content if part.type == "text")
            case _:
                assert_never(content)

    messages = [
        {"role": role, "content": text(content)}
        for role, content in [("query", query), ("document", document)]
    ]
    if instruct:
        messages.insert(0, {"role": "system", "content": instruct})
    return (
        ImmutableSandboxedEnvironment()
        .from_string(chat_template)
        .render(messages=messages, **({"instruct": instruct} if instruct else {}))
    )
