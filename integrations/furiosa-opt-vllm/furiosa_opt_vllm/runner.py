"""Own loaded engine resources and mutable vLLM request state for the runner lifetime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias, assert_never

import torch
from vllm.v1.worker.gpu.kv_connector import NO_OP_KV_CONNECTOR

import furiosa_opt_engine as engine
from furiosa_opt_vllm.additional import Section
from furiosa_opt_vllm.admission import speculative
from furiosa_opt_vllm.decoder import Decoder
from furiosa_opt_vllm.multimodal import Media

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vllm.config import VllmConfig
    from vllm.model_executor.layers.pooler.abstract import Pooler
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu.kv_connector import KVConnector
    from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

    from furiosa_opt_vllm import lora
    from furiosa_opt_vllm.admission import Speculative
    from furiosa_opt_vllm.generation import Collected
    from furiosa_opt_vllm.request import TokenScores
    from furiosa_opt_vllm.selection import Sampling, Speculation

    # The memory vLLM's sleep mode releases, by the tag vLLM names it with.
    Tag: TypeAlias = Literal["weights", "kv_cache"]


@dataclass(frozen=True, slots=True)
class Weights:
    """Weights sleep mode released: their copy in host memory, or `None` when waking reads the
    checkpoint's files."""

    copy: engine.Checkpoint | None


@dataclass(frozen=True, slots=True)
class Storage:
    """A KV cache sleep mode released: the spec and block count waking allocates."""

    spec: engine.decoder.CacheSpec
    blocks: int


@dataclass(frozen=True, slots=True)
class Labels:
    """Classify with the checkpoint's score head, one column per label."""


@dataclass(frozen=True, slots=True)
class Vocabulary:
    """Classify with the vocabulary head, combining the classifier tokens' columns."""


@dataclass(frozen=True, slots=True)
class Tokens:
    """The device argmax token of each request that completed its scheduled rows: the output of
    the model call that writes it and its row there."""

    values: Mapping[str, tuple[engine.HbmTensor, int]]


@dataclass(frozen=True, slots=True)
class Argmax:
    """The device argmax token of each completed request's sampled rows, in position order: one
    row, and one more per scheduled draft; each is the output of the call that writes it and its
    row there."""

    values: Mapping[str, list[tuple[engine.HbmTensor, int]]]


@dataclass(frozen=True, slots=True)
class Logits:
    """The vocabulary scores `[vocab]` of each completed request's sampled rows, in position
    order: one row, and one more per scheduled draft."""

    values: Mapping[str, list[torch.Tensor]]


class Runner:
    def __init__(self, vllm_config: VllmConfig) -> None:
        self.vllm_config: VllmConfig = vllm_config
        # The speculative method the platform admitted before the runner starts.
        self.speculative: Speculative | None = speculative(vllm_config, Section.parse(vllm_config))
        self.model: Decoder | Media | None = None
        self.cache: engine.decoder.Cache | None = None
        # vLLM's driver of the KV connector around each step, which does nothing without one.
        self.kv_connector: KVConnector = NO_OP_KV_CONNECTOR
        # The memory sleep mode released, by its tag, and what waking restores it from.
        self.asleep: dict[Tag, Weights | Storage] = {}
        # The LoRA adapters vLLM loads, when it enables LoRA.
        self.adapters: lora.Adapters | None = None
        self.requests: dict[str, CachedRequestState] = {}
        # req_id -> the logprobs of fixed token IDs the request asks for at its prompt rows.
        self.prompt_scores: dict[str, TokenScores] = {}
        self.draws: dict[str, dict[int, torch.Tensor]] = {}
        self.pending: tuple[SchedulerOutput, Argmax | Logits] | None = None
        # Greedy steps whose output vLLM has not collected, oldest first.
        self.issued: list[Collected] = []
        # req_id -> the uncollected output that selects the request's next token, and its row.
        self.selecting: dict[str, tuple[engine.HbmTensor, int]] = {}
        # Chosen once the target loads, since a draft model loads beside it.
        self.selection: Sampling | Speculation
        self.pooler: Pooler | None = None
        self.classifier: Labels | Vocabulary | None = None
        self.input_batch: InputBatch

    def extend(self, request: CachedRequestState, tokens: Sequence[int]) -> None:
        """Append `tokens` to `request`'s output and, while vLLM's batch holds the request, to its
        row there, which the proposers read; the batch copies earlier tokens when it admits one."""
        if (index := self.input_batch.req_id_to_index.get(request.req_id)) is not None:
            start = request.num_tokens
            self.input_batch.token_ids_cpu[index, start : start + len(tokens)] = tokens
            self.input_batch.num_tokens_no_spec[index] = start + len(tokens)
        request.output_token_ids.extend(tokens)

    def acknowledge(self, request: CachedRequestState, count: int) -> None:
        """Restore dropped draws and forget states before the scheduler's confirmed output."""
        states = self.draws.get(request.req_id, {})
        if count < len(request.output_token_ids) and request.generator is not None:
            if (state := states.get(count)) is not None:
                _ = request.generator.set_state(state)
        remaining = {position: state for position, state in states.items() if position >= count}
        if remaining:
            self.draws[request.req_id] = remaining
        else:
            self.draws.pop(request.req_id, None)

    @property
    def state(self) -> Decoder:
        match self.model:
            case Decoder():
                return self.model
            case None | Media():
                raise RuntimeError("no sequence model is loaded")
            case _:
                assert_never(self.model)
