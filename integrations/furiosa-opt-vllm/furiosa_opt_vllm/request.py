from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, assert_never

import torch
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu_input_batch import CachedRequestState

from furiosa_opt_vllm import generation
from furiosa_opt_vllm.decoder import Decoder
from furiosa_opt_vllm.multimodal import Media

if TYPE_CHECKING:
    from collections.abc import Callable

    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput

    from furiosa_opt_vllm.runner import Runner


def retire(runner: Runner, req_id: str) -> None:
    runner.selection.retire(req_id)
    _ = runner.prompt_scores.pop(req_id, None)


def update(runner: Runner, scheduler: SchedulerOutput) -> None:
    if scheduler.kv_cache_block_copies:
        raise RuntimeError("startup refuses every configuration that copies cache blocks")
    cached = scheduler.scheduled_cached_reqs
    # Requests vLLM sends in full; Model Runner V2's scheduler sends a resumed one as new.
    resent = {new.req_id for new in scheduler.scheduled_new_reqs} | cached.resumed_req_ids
    # A step in flight for a request vLLM resends carries tokens vLLM drops: read it now, so that
    # they extend the request's earlier state and not the tokens vLLM sends.
    if any(req_id in step.tokens.values for step in runner.issued for req_id in resent):
        generation.flush(runner)
    model = runner.model
    scheduled = scheduler.num_scheduled_tokens
    for req_id in scheduler.finished_req_ids:
        _ = runner.draws.pop(req_id, None)
        _ = runner.requests.pop(req_id, None)
        _ = runner.selecting.pop(req_id, None)
        _ = runner.input_batch.remove_request(req_id)
        retire(runner, req_id)
    for req_id in tuple(runner.input_batch.req_id_to_index):
        if req_id not in scheduled or req_id in resent:
            _ = runner.input_batch.remove_request(req_id)
    match model:
        case Media():
            media = True
        case None | Decoder():
            media = False
        case _:
            assert_never(model)
    for new in scheduler.scheduled_new_reqs:
        if (earlier := runner.requests.get(new.req_id)) is not None:
            runner.acknowledge(earlier, len((new.prefill_token_ids or [])[new.prompt_len :]))
        retire(runner, new.req_id)
        if new.mm_features and not media:
            raise ValueError("media features require a multimodal model")
        request = runner.requests[new.req_id] = admit(
            new,
            lambda: runner.vllm_config.model_config.get_vocab_size(),
            runner.requests.get(new.req_id),
        )
        if (asked := TokenScores.ask(request)) is not None:
            runner.prompt_scores[new.req_id] = asked
    for index, req_id in enumerate(cached.req_ids):
        request = runner.requests[req_id]
        runner.acknowledge(request, cached.num_output_tokens[index])
        request.num_computed_tokens = cached.num_computed_tokens[index]
        del request.output_token_ids[cached.num_output_tokens[index] :]
        if (blocks := cached.new_block_ids[index]) is not None:
            if req_id in cached.resumed_req_ids:
                request.block_ids = tuple(list(group) for group in blocks)
                retire(runner, req_id)
            else:
                for existing, added in zip(request.block_ids, blocks, strict=True):
                    existing.extend(added)
    for req_id in scheduled:
        if req_id not in runner.input_batch.req_id_to_index:
            _ = runner.input_batch.add_request(runner.requests[req_id])
    # Adding first fills removed indices; condensing an emptied batch keeps stale ones.
    runner.input_batch.condense()
    # vLLM's batch rows, which the samplers and proposers read, hold the tokens `Runner.extend`
    # writes; a rolled back output only shortens its row, and each step brings its drafts.
    for req_id in scheduled:
        request = runner.requests[req_id]
        index = runner.input_batch.req_id_to_index[req_id]
        runner.input_batch.num_tokens_no_spec[index] = request.num_tokens
        runner.input_batch.num_computed_tokens_cpu[index] = request.num_computed_tokens
        runner.input_batch.update_req_spec_token_ids(
            request, scheduler.scheduled_spec_decode_tokens
        )
    runner.input_batch.refresh_metadata()


def admit(
    new: NewRequestData, vocabulary: Callable[[], int], earlier: CachedRequestState | None
) -> CachedRequestState:
    """Restore the scheduler's tokens and preserve a resumed request's random stream."""
    params = new.sampling_params
    # The tokens a resumed request generated, which only Model Runner V2's scheduler sends.
    generated = (new.prefill_token_ids or [])[new.prompt_len :]
    if earlier is not None:
        generator = earlier.generator
    elif params is not None and params.seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(params.seed)
    else:
        generator = None
    request = CachedRequestState(
        req_id=new.req_id,
        prompt_token_ids=(list(new.prompt_token_ids) if new.prompt_token_ids is not None else None),
        prompt_embeds=new.prompt_embeds,
        prompt_is_token_ids=new.prompt_is_token_ids,
        mm_features=list(new.mm_features),
        sampling_params=params,
        pooling_params=new.pooling_params,
        generator=generator,
        block_ids=tuple(list(group) for group in new.block_ids),
        num_computed_tokens=new.num_computed_tokens,
        output_token_ids=generated,
        lora_request=new.lora_request,
    )
    prompt = params.prompt_logprobs if params is not None else None
    # A resumed request reported its prompt logprobs with its first token.
    if new.prompt_token_ids is not None and prompt is not None and not generated:
        count = vocabulary() if prompt == -1 else prompt
        request.in_progress_prompt_logprobs_cpu = LogprobsTensors.empty_cpu(
            request.num_prompt_tokens - 1, count + 1
        )
    return request


@dataclass(frozen=True, slots=True)
class TokenScores:
    """Requested token scores from prompt row `start` on, reported with the first output token."""

    token_ids: torch.Tensor
    start: int
    values: torch.Tensor

    @classmethod
    def ask(cls, request: CachedRequestState) -> TokenScores | None:
        params = request.sampling_params
        if params is None or params.prompt_logprob_token_ids is None or request.output_token_ids:
            return None
        start = params.prompt_logprob_start or 0
        if request.num_computed_tokens > start:
            return None
        ids = torch.tensor(params.prompt_logprob_token_ids, dtype=torch.long)
        rows = max(request.num_prompt_tokens - 1 - start, 0)
        return cls(ids, start, torch.empty((rows, len(ids)), dtype=torch.float32))

    def record(self, position: int, scores: torch.Tensor) -> None:
        if position < self.start:
            return
        _ = self.values[position - self.start].copy_(scores[self.token_ids])
