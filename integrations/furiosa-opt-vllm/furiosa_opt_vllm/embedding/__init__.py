"""Configure and complete vLLM embedding, pooling and score requests."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final, cast

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, TypeAdapter
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.layers.pooler.seqwise.poolers import pooler_for_classify
from vllm.v1.outputs import ModelRunnerOutput

from furiosa_opt_vllm.runner import Labels, Runner, Vocabulary

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from vllm.model_executor.layers.pooler.abstract import Pooler
    from vllm.tasks import PoolingTask
    from vllm.tokenizers.protocol import TokenizerLike
    from vllm.v1.outputs import PoolerOutput

TASKS: Final[tuple[PoolingTask, ...]] = ("embed", "classify", "token_embed", "token_classify")


class Classification(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    method: str | None = None
    classifier_from_token: tuple[str, ...] = ()


def configure(runner: Runner) -> None:
    config = runner.vllm_config.model_config
    if config.runner_type != "pooling":
        return
    pooling = config.pooler_config
    if pooling is None:
        raise ValueError("a pooling model requires a pooler configuration")
    with set_current_vllm_config(runner.vllm_config):
        settings = Classification.model_validate(config.hf_text_config.to_dict())
        match settings.method:
            case ("from_2_way_softmax" | "no_post_processing") as method:
                from vllm import tokenizers

                # vLLM's tokenizer factory leaves its dynamic options untyped.
                factory = cast("Callable[..., TokenizerLike]", tokenizers.get_tokenizer)
                tokenizer = factory(
                    config.tokenizer,
                    revision=config.tokenizer_revision,
                    tokenizer_mode=config.tokenizer_mode,
                    trust_remote_code=config.trust_remote_code,
                )
                tokens = settings.classifier_from_token
                indices = TypeAdapter(list[int]).validate_python(
                    [tokenizer.convert_tokens_to_ids(token) for token in tokens]
                )
                if not indices:
                    raise ValueError("classifier_from_token must name vocabulary tokens")
                match method:
                    case "from_2_way_softmax":
                        if len(indices) != 2:
                            raise ValueError("two-way classification requires two tokens")

                        def classifier(logits: torch.Tensor) -> torch.Tensor:
                            return logits[:, indices[1:]] - logits[:, indices[:1]]

                    case "no_post_processing":

                        def classifier(logits: torch.Tensor) -> torch.Tensor:
                            return logits[:, indices]

                # The classifier reads the vocabulary scores of the rows vLLM pools.
                runner.classifier = Vocabulary()
                runner.pooler = pooler_for_classify(pooling, classifier=classifier)
            # As vLLM converts a sequence classifier: one score head for sequences and tokens.
            case None if config.convert_type == "classify" or config.get_pooling_task(TASKS) in (
                "classify",
                "token_classify",
            ):
                if config.head_dtype != torch.float32:
                    raise ValueError(f"a score head computes in float32, not {config.head_dtype}")
                # The rows vLLM pools are the score head's, one column per label.
                runner.classifier = Labels()
                runner.pooler = DispatchPooler.for_seq_cls(pooling)
            case None:
                runner.pooler = DispatchPooler.for_embedding(pooling)
            case _:
                raise ValueError("classification requires a vocabulary readout")


def pool(
    runner: Runner,
    pooler: Pooler,
    scheduled: Mapping[str, int],
    features: Mapping[str, list[torch.Tensor]],
) -> ModelRunnerOutput:
    ids = list(runner.input_batch.req_ids)
    metadata = runner.input_batch.get_pooling_metadata()
    lengths = torch.tensor([runner.requests[req_id].num_computed_tokens for req_id in ids])
    metadata.build_pooling_cursor(
        np.array([scheduled[req_id] for req_id in ids], dtype=np.int32),
        lengths,
        device=torch.device("cpu"),
    )
    hidden = torch.stack([row for req_id in ids for row in features[req_id]])
    # nn.Module.__call__ erases the pooler's declared forward result type.
    pooled = cast("PoolerOutput", pooler(hidden, metadata))
    return ModelRunnerOutput(
        req_ids=ids,
        req_id_to_index={req_id: index for index, req_id in enumerate(ids)},
        pooler_output=[
            pooled[index] if lengths[index] == metadata.prompt_lens[index] else None
            for index in range(len(ids))
        ],
    )
