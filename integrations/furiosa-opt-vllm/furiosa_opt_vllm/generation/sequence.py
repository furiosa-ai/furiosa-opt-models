"""Run scheduled text rows through the decoder's calls and the shared sequence planner."""

from __future__ import annotations

from itertools import groupby, islice
from typing import TYPE_CHECKING, Final, assert_never

import numpy as np
import torch

import furiosa_opt_engine as engine
from furiosa_opt_vllm import embedding, generation
from furiosa_opt_vllm.batch import Call, Row
from furiosa_opt_vllm.decoder import RELEASED, Decoder
from furiosa_opt_vllm.generation import Sampled, Scored, paged
from furiosa_opt_vllm.runner import Argmax, Logits, Runner

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput

# vLLM's draft at a scheduled position it has no draft for, which rejection refuses.
PLACEHOLDER: Final = -1


def token_ids(input: list[int] | NDArray[np.float32]) -> list[int]:
    match input:
        case list():
            return input
        case np.ndarray():
            raise ValueError("this entry requires token IDs, not embedding rows")
        case _:
            assert_never(input)


def forward(
    runner: Runner,
    model: Decoder,
    scheduler: SchedulerOutput,
) -> ModelRunnerOutput | None:
    match model:
        case Decoder(model=engine.Decoder(output=engine.decoder.Output.TOKENS) as language):
            pass
        case Decoder():
            language = None
        case _:
            assert_never(model)
    scheduled = dict(scheduler.num_scheduled_tokens)
    greedy = generation.device_argmax(runner, language, scheduled)
    adapters = runner.adapters
    slots = (
        adapters.activate(runner.requests[req_id] for req_id in scheduled)
        if adapters is not None
        else {}
    )
    pending = dict(scheduled)
    sampled: dict[str, list[tuple[engine.HbmTensor, int]]] = {}
    logits: dict[str, list[torch.Tensor]] = {}
    features: dict[str, list[torch.Tensor]] = {req_id: [] for req_id in scheduled}
    # The step's draft tokens by request and position, which follow the request's known tokens.
    drafts = {
        (req_id, runner.requests[req_id].num_tokens + offset): draft
        for req_id, proposed in scheduler.scheduled_spec_decode_tokens.items()
        for offset, draft in enumerate(proposed)
    }
    while pending:
        queries = (
            (
                req_id,
                position,
                request.prompt_embeds[position]
                if request.prompt_embeds is not None
                and position < request.num_prompt_tokens
                and (
                    request.prompt_token_ids is None
                    or (
                        request.prompt_is_token_ids is not None
                        and not request.prompt_is_token_ids[position]
                    )
                )
                else None,
            )
            for req_id, remaining in pending.items()
            for request in [runner.requests[req_id]]
            for position in range(
                request.num_computed_tokens, request.num_computed_tokens + remaining
            )
        )
        embedded, group = next(groupby(queries, key=lambda row: row[2] is not None))
        rows = list(islice(group, runner.state.rows))
        ids = [row[0] for row in rows]
        positions = [row[1] for row in rows]
        requests = [runner.requests[req_id] for req_id in ids]
        held = (
            []
            if embedded
            else [
                request.get_token_id(position)
                if position < request.num_tokens
                else drafts.get((req_id, position))
                for req_id, request, position in zip(ids, requests, positions, strict=True)
            ]
        )
        # A row past its request's known tokens and drafts takes the token an uncollected call
        # selects.
        fed = {row: runner.selecting[ids[row]] for row, id in enumerate(held) if id is None}
        input = (
            torch.stack([row[2] for row in rows if row[2] is not None])
            .detach()
            .float()
            .cpu()
            .numpy()
            if embedded
            # No sampled token depends on a fed row's ID or a draft placeholder's, which
            # rejection refuses, so either embeds as token 0.
            else [0 if id is None or id == PLACEHOLDER else id for id in held]
        )
        scoring = runner.pooler is None and any(
            generation.scores_prompt(runner, request) for request in requests
        )
        every = scoring or runner.pooler is not None
        planned = [
            Row(req_id, position, position + 1 >= request.num_tokens)
            for req_id, request, position in zip(ids, requests, positions, strict=True)
        ]
        final = any(row.complete for row in planned)
        call = Call[str].plan(planned, every=every)
        selected = {row: index for index, row in enumerate(call.kept)}
        read: Scored | Sampled | None = None
        hidden: engine.HbmTensor | None = None
        match model:
            case Decoder():
                if model.model is None:
                    raise RuntimeError(RELEASED)
                feed = paged.Feed(paged.upload(model, token_ids(input), fed), fed, every, slots)
                match model.model.output:
                    case engine.decoder.Output.HIDDEN:
                        hidden = paged.forward(runner, model.model.forward, call, feed)
                    case engine.decoder.Output.TOKENS:
                        if scoring or (final and greedy is None):
                            raise NotImplementedError(
                                "sampling and scoring read scores the engine's calls do not return"
                            )
                        read = Sampled(paged.forward(runner, model.model.forward, call, feed))
                    case engine.decoder.Output.SCORES:
                        raise NotImplementedError("a sequence classifier's calls are not served")
            case _:
                assert_never(model)
        pooling = runner.vllm_config.model_config.pooler_config
        # --8<-- [start:argmax]
        if hidden is None:
            pass  # The language model call read the step.
        elif (
            runner.classifier is not None
            and not final
            and pooling is not None
            and pooling.get_seq_pooling_type() == "LAST"
            and all(
                request.pooling_params is not None and request.pooling_params.task == "classify"
                for request in requests
            )
        ):
            pass  # A last-row sequence classifier pools only the call that completes a sequence.
        elif runner.pooler is not None:
            read = Scored(torch.as_tensor(hidden.to_host().numpy()))
        # --8<-- [end:argmax]
        for row, (req_id, request, position) in enumerate(
            zip(ids, requests, positions, strict=True)
        ):
            if (
                generation.scores_prompt(runner, request)
                and position + 1 < request.num_prompt_tokens
            ):
                # Scores of prompt rows make the step score every row.
                match read:
                    case Scored(scores=scores):
                        generation.record(runner, request, position, scores[row : row + 1])
                    case Sampled() | None:
                        raise RuntimeError("a prompt row was read without its scores")
                    case _:
                        assert_never(read)
            if runner.pooler is not None:
                match read:
                    case Scored(scores=scores):
                        features[req_id].append(scores[row])
                    case Sampled() | None if hidden is not None:
                        features[req_id].append(torch.zeros(hidden.shape[1]))
                    case Sampled() | None:
                        raise RuntimeError("a pooled row was read without its hidden rows")
                    case _:
                        assert_never(read)
            request.num_computed_tokens += 1
            pending[req_id] -= 1
            if pending[req_id] == 0:
                del pending[req_id]
            if request.num_computed_tokens >= request.num_tokens:
                # Greedy steps leave every completed row's token on the device.
                match read:
                    case Sampled(tokens=tokens):
                        sampled.setdefault(req_id, []).append((tokens, selected[row]))
                    case Scored(scores=scores):
                        logits.setdefault(req_id, []).append(scores[selected[row]])
                    case None:
                        raise RuntimeError("a completed row was read without its scores")
                    case _:
                        assert_never(read)
    if runner.pooler is not None:
        return embedding.pool(runner, runner.pooler, scheduled, features)
    runner.pending = (scheduler, Argmax(sampled) if greedy is not None else Logits(logits))
    return None
