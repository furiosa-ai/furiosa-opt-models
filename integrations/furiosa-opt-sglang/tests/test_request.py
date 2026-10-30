"""Requests for readouts the plugin does not compute are refused before tokenization."""

from collections.abc import Callable
from typing import TypeAlias

import pytest
import torch
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.embed_types import PositionalEmbeds
from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import prepare_server_args
from transformers import PretrainedConfig

from furiosa_opt_sglang.platform import Refused
from furiosa_opt_sglang.request import validate_request

Value: TypeAlias = float | int | str | dict[str, float] | dict[str, bool]


@pytest.fixture
def manager() -> TokenizerManager:
    """A manager serving a text model; validation reads its model configuration and options."""
    config = ModelConfig.__new__(ModelConfig)
    config.hf_config, config.is_encoder_decoder, config.vocab_size = PretrainedConfig(), False, 32
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.model_config = config
    manager.server_args = prepare_server_args(["--model-path", "Qwen/Qwen3-1.7B"])
    manager.preferred_sampling_params = None
    return manager


# The sampling parameters of plain greedy decoding, which the plugin serves.
GREEDY = {"temperature": 0}


@pytest.mark.parametrize("positions", [[], [0], [1, 2]], ids=["empty", "first", "several"])
@pytest.mark.parametrize(
    ("build", "option"),
    [
        (
            lambda p: GenerateReqInput(
                input_ids=[1, 2], sampling_params=GREEDY, token_indices_to_pool=p
            ),
            "token_indices_to_pool",
        ),
        (
            lambda p: EmbeddingReqInput(input_ids=[1, 2], token_indices_to_pool=p),
            "token_indices_to_pool",
        ),
        (lambda p: EmbeddingReqInput(input_ids=[1, 2], decision_layout=p), "decision_layout"),
    ],
    ids=["generation", "embedding", "decision-layout"],
)
def test_refuses_unserved_readout(
    manager: TokenizerManager,
    build: Callable[[list[int]], GenerateReqInput | EmbeddingReqInput],
    option: str,
    positions: list[int],
) -> None:
    """A position selection is refused whenever it is given, even when it selects no position."""
    with pytest.raises(ValueError, match=f"does not support {option}"):
        validate_request(manager, build(positions), [1, 2])


@pytest.mark.parametrize(
    "request_input",
    [
        GenerateReqInput(input_ids=[1, 2], sampling_params=GREEDY),
        EmbeddingReqInput(input_ids=[1, 2]),
    ],
    ids=["generation", "embedding"],
)
def test_admits_request_without_selection(
    manager: TokenizerManager, request_input: GenerateReqInput | EmbeddingReqInput
) -> None:
    """A request that selects no positions is admitted."""
    validate_request(manager, request_input, [1, 2])


@pytest.mark.parametrize("ids", [[1, 32], [-1, 2]], ids=["past", "negative"])
def test_refuses_token_ids_outside_the_vocabulary(
    manager: TokenizerManager, ids: list[int]
) -> None:
    """A token ID outside the vocabulary is refused with its request, before the scheduler runs."""
    request = GenerateReqInput(input_ids=ids, sampling_params=GREEDY)
    with pytest.raises(ValueError, match="vocabulary"):
        validate_request(manager, request, ids)


@pytest.mark.feature("Multimodal inputs (images, video, audio)")
@pytest.mark.parametrize(
    ("request_input", "option"),
    [
        (GenerateReqInput(input_ids=[1, 2], sampling_params=GREEDY), "image_data"),
        (GenerateReqInput(input_ids=[1, 2], sampling_params=GREEDY), "video_data"),
        (EmbeddingReqInput(input_ids=[1, 2]), "audio_data"),
    ],
    ids=["image", "video", "embedding-audio"],
)
def test_refuses_multimodal(
    manager: TokenizerManager, request_input: GenerateReqInput | EmbeddingReqInput, option: str
) -> None:
    """Media inputs are refused by name before a text request is tokenized."""
    setattr(request_input, option, ["unserved-media"])
    with pytest.raises(Refused) as error:
        validate_request(manager, request_input, [1, 2])
    assert error.value.option == option


REFUSED = "serves only greedy decoding, without logprobs or logit changes"


@pytest.mark.parametrize(
    ("sampling", "option", "requirement"),
    [
        ({"temperature": 0.8}, "top_k", f"requires temperature 0 or top_k 1: {REFUSED}"),
        *(
            ({field: value}, field, REFUSED)
            for field, value in (
                ("repetition_penalty", 1.2),
                ("logit_bias", {"1": 2.0}),
                ("frequency_penalty", 0.2),
                ("presence_penalty", 0.2),
                ("min_new_tokens", 2),
                ("beam_width", 2),
                ("regex", "a"),
                ("json_schema", "{}"),
                ("ebnf", 'root ::= "a"'),
                ("structural_tag", "{}"),
                ("watermark", {"enabled": True}),
            )
        ),
        # SGLang biases the scores of a request given any logit bias, even an empty one.
        pytest.param({"logit_bias": {}}, "logit_bias", REFUSED, id="logit_bias-empty"),
    ],
    ids=lambda value: value if isinstance(value, str) and " " not in value else None,
)
@pytest.mark.feature("Text generation")
@pytest.mark.feature("Structured outputs")
@pytest.mark.feature("Watermarking")
def test_refuses_sampling(
    manager: TokenizerManager, sampling: dict[str, Value], option: str, requirement: str
) -> None:
    """A request's sampling other than plain greedy decoding, which the engine's calls return, is
    refused by name and reason."""
    request = GenerateReqInput(input_ids=[1, 2], sampling_params=GREEDY | sampling)
    with pytest.raises(Refused) as error:
        validate_request(manager, request, [1, 2])
    assert (error.value.option, error.value.requirement) == (option, requirement)


@pytest.mark.feature("Text generation")
def test_refuses_preferred_sampling(manager: TokenizerManager) -> None:
    """The server's preferred sampling counts where a request leaves a parameter out."""
    manager.preferred_sampling_params = {"temperature": 0.7}
    with pytest.raises(Refused) as error:
        validate_request(manager, GenerateReqInput(input_ids=[1, 2], sampling_params={}), [1, 2])
    assert error.value.option == "top_k"
    validate_request(manager, GenerateReqInput(input_ids=[1, 2], sampling_params=GREEDY), [1, 2])


@pytest.mark.parametrize(
    ("option", "value"),
    [
        pytest.param(
            "custom_logit_processor",
            "processor",
            marks=pytest.mark.feature("Custom logits processors"),
        ),
        pytest.param("return_logprob", True, marks=pytest.mark.feature("Text generation")),
        pytest.param("return_sampling_mask", True, marks=pytest.mark.feature("Text generation")),
    ],
)
def test_refuses_readout(manager: TokenizerManager, option: str, value: str | bool) -> None:
    """A request's own logit processor, logprobs or sampling mask, which read scores, are refused
    by name and reason."""
    request = GenerateReqInput(input_ids=[1, 2], sampling_params=GREEDY)
    setattr(request, option, value)
    with pytest.raises(Refused) as error:
        validate_request(manager, request, [1, 2])
    assert (error.value.option, error.value.requirement) == (option, REFUSED)


@pytest.mark.parametrize(
    "request_input",
    [
        GenerateReqInput(input_embeds=[[0.5, 0.5]], sampling_params=GREEDY),
        EmbeddingReqInput(input_embeds=[[0.5, 0.5]]),
    ],
    ids=["generation", "embedding"],
)
@pytest.mark.feature("Prompt embeddings")
def test_refuses_input_embeddings(
    manager: TokenizerManager, request_input: GenerateReqInput | EmbeddingReqInput
) -> None:
    """Embedding rows in place of token IDs are refused by name and reason."""
    with pytest.raises(Refused) as error:
        validate_request(manager, request_input, None)
    assert (error.value.option, error.value.requirement) == (
        "input_embeds",
        "takes token IDs, not input embeddings",
    )


OVERRIDE = PositionalEmbeds(embeds=torch.zeros((1, 8)), positions=[0])


@pytest.mark.parametrize(
    ("request_input", "option"),
    [
        (
            GenerateReqInput(
                input_ids=[1, 2], sampling_params=GREEDY, positional_embed_overrides=OVERRIDE
            ),
            "positional_embed_overrides",
        ),
        (
            EmbeddingReqInput(input_ids=[1, 2], positional_embed_overrides=OVERRIDE),
            "positional_embed_overrides",
        ),
        (
            EmbeddingReqInput(
                input_ids=[1, 2], embed_override_token_id=1, embed_overrides=[[torch.zeros(8)]]
            ),
            "embed_overrides",
        ),
    ],
    ids=["generation", "embedding", "embedding-unresolved"],
)
@pytest.mark.feature("Prompt embeddings")
def test_refuses_embedding_overrides(
    manager: TokenizerManager, request_input: GenerateReqInput | EmbeddingReqInput, option: str
) -> None:
    """Rows that replace the embeddings of some token positions are refused by name and reason in
    either form SGLang takes them."""
    with pytest.raises(Refused) as error:
        validate_request(manager, request_input, [1, 2])
    assert (error.value.option, error.value.requirement) == (
        option,
        "takes token IDs, without embedding overrides",
    )
