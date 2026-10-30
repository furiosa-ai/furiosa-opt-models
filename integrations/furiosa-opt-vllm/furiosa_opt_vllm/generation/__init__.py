"""Select device argmax or vLLM sampling without advancing unscheduled random streams.

A greedy step's tokens stay on the device until vLLM asks for its output, which it does after it
issues the next step: that step takes each token from the device as its input, so the device runs
the two calls back to back. A speculative step reads its greedy tokens at once, since accepting
its drafts and proposing the next ones take them on the host.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, assert_never, final

import torch
from typing_extensions import override
from vllm.sampling_params import SamplingType
from vllm.v1.outputs import AsyncModelRunnerOutput, LogprobsTensors, ModelRunnerOutput
from vllm.v1.sample.logits_processor import BUILTIN_LOGITS_PROCESSORS
from vllm.v1.sample.sampler import Sampler
from vllm.v1.structured_output.utils import apply_grammar_bitmask

from furiosa_opt_vllm.device import read
from furiosa_opt_vllm.replay import Replay
from furiosa_opt_vllm.runner import Argmax, Logits, Runner, Tokens
from furiosa_opt_vllm.selection import Sampling, Speculation

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.outputs import KVConnectorOutput, LogprobsLists, SamplingMaskLists
    from vllm.v1.worker.gpu_input_batch import CachedRequestState

    import furiosa_opt_engine as engine


@dataclass(frozen=True, slots=True)
class Scored:
    """Rows' vocabulary scores, or their hidden states when a pooler reads them."""

    scores: torch.Tensor


@dataclass(frozen=True, slots=True)
class Sampled:
    """Rows' greedy tokens from the device argmax, before the model call that writes them ends."""

    tokens: engine.HbmTensor


@final
class Collected(AsyncModelRunnerOutput):
    """A greedy step's output: its tokens read from the device once vLLM asks for them, or once
    a later step needs them on the host."""

    def __init__(
        self,
        runner: Runner,
        ids: list[str],
        tokens: Tokens,
        transferred: KVConnectorOutput | None,
    ) -> None:
        self.runner = runner
        self.ids = ids
        self.tokens = tokens
        # The KV connector's report of the step's transfers.
        self.transferred = transferred
        self.output: ModelRunnerOutput | None = None

    @override
    def get_output(self) -> ModelRunnerOutput:
        if self.output is None:
            self.output = collect(self.runner, self.ids, self.tokens)
            self.output.kv_connector_output = self.transferred
            self.runner.issued.remove(self)
        return self.output


def collect(runner: Runner, ids: list[str], tokens: Tokens) -> ModelRunnerOutput:
    """The step's device `tokens` on the host, reported for the step's requests `ids`; a request's
    selection by a later step stays."""
    host = read({req_id: [selected] for req_id, selected in tokens.values.items()})
    for req_id, selected in tokens.values.items():
        if runner.selecting.get(req_id) == selected:
            del runner.selecting[req_id]
    return report(runner, {req_id: host.get(req_id, []) for req_id in ids}, None, None)


def report(
    runner: Runner,
    sampled: Mapping[str, list[int]],
    logprobs: LogprobsLists | None,
    masks: SamplingMaskLists | None,
) -> ModelRunnerOutput:
    """Report step output in request order; the first token releases completed prompt scores."""
    prompt_logprobs: dict[str, LogprobsTensors | None] = {}
    token_scores: dict[str, torch.Tensor] = {}
    for req_id, tokens in sampled.items():
        # A finished request's state is gone; a live one's output grows in step order.
        if (request := runner.requests.get(req_id)) is None:
            continue
        runner.extend(request, tokens)
        if tokens and request.in_progress_prompt_logprobs_cpu is not None:
            prompt_logprobs[req_id] = request.in_progress_prompt_logprobs_cpu
            request.in_progress_prompt_logprobs_cpu = None
        if tokens and (asked := runner.prompt_scores.pop(req_id, None)) is not None:
            token_scores[req_id] = asked.values
    return ModelRunnerOutput(
        req_ids=list(sampled),
        req_id_to_index={req_id: index for index, req_id in enumerate(sampled)},
        sampled_token_ids=list(sampled.values()),
        logprobs=logprobs,
        prompt_logprobs_dict=prompt_logprobs,
        prompt_token_id_logprobs_dict=token_scores,
        sampling_masks=masks,
    )


def flush(runner: Runner) -> None:
    """Read every issued step's tokens to the host, for a step that needs them there."""
    for issued in list(runner.issued):
        _ = issued.get_output()


def greedy(params: SamplingParams) -> bool:
    return (
        params.sampling_type == SamplingType.GREEDY
        and params.logprobs is None
        and params.prompt_logprobs is None
        and params.prompt_logprob_token_ids is None
        and params.logprob_token_ids is None
        and params.thinking_token_budget is None
        and params.repetition_penalty == 1
        and params.frequency_penalty == 0
        and params.presence_penalty == 0
        and params.min_tokens == 0
        and not params.logit_bias
        and not params.allowed_token_ids
        and not params.bad_words
        and params.structured_outputs is None
        and params.trace_decode_token_ids is None
        and not params.extra_args
    )


def sample(
    runner: Runner, grammar_output: GrammarOutput | None
) -> ModelRunnerOutput | AsyncModelRunnerOutput:
    if runner.pending is None:
        raise RuntimeError("execute_model must precede sampling")
    scheduler, outputs = runner.pending
    runner.pending = None
    # As vLLM's runners do, the connector finishes the step's transfers once it is sampled.
    transferred = runner.kv_connector.post_forward(scheduler.finished_req_ids)
    ids = list(runner.input_batch.req_ids)
    logprobs = masks = None
    match outputs:
        case Argmax(values=values):
            match runner.selection:
                case Sampling():
                    # A request's one sampled row selects its next token, which the next call
                    # may take on the device.
                    tokens = Tokens({req_id: row for req_id, (row,) in values.items()})
                    runner.selecting.update(tokens.values)
                    collected = Collected(runner, ids, tokens, transferred)
                    runner.issued.append(collected)
                    return collected
                case Speculation() as speculation:
                    drawn = speculation.accept(values, runner.input_batch)
                case _:
                    assert_never(runner.selection)
        case Logits(values=values):
            drawn, logprobs, masks = draw(runner, values, scheduler, grammar_output)
        case _:
            assert_never(outputs)
    sampled = {req_id: drawn.get(req_id, []) for req_id in ids}
    output = report(runner, sampled, logprobs, masks)
    output.kv_connector_output = transferred
    runner.selection.propose(
        runner.input_batch, list(sampled.values()), scheduler.num_spec_tokens_to_schedule
    )
    return output


def draw(
    runner: Runner,
    values: Mapping[str, list[torch.Tensor]],
    scheduler: SchedulerOutput,
    grammar_output: GrammarOutput | None,
) -> tuple[dict[str, list[int]], LogprobsLists | None, SamplingMaskLists | None]:
    """The tokens vLLM's sampler, or its rejection, draws from the scores `values` of each
    request's sampled rows under the step's grammar, the logprobs of every batch row, and its
    supports when vLLM asks for sampling masks."""
    if not values:
        return {}, None, None
    ids = runner.input_batch.req_ids
    vocab = runner.vllm_config.model_config.get_vocab_size()
    # The sampler draws for every batch row: a request without sampled rows takes a zero row,
    # made only for it.
    scores = [values.get(req_id) or [torch.zeros(vocab)] for req_id in ids]
    logits = torch.stack([row for rows in scores for row in rows])
    if grammar_output is not None:
        apply_grammar_bitmask(scheduler, grammar_output, runner.input_batch, logits)
    # Preserve each seeded request's random stream when the sampler draws for idle rows.
    saved = {i: generator.get_state() for i, generator in runner.input_batch.generators.items()}
    for i, state in saved.items():
        if ids[i] in values:
            position = len(runner.requests[ids[i]].output_token_ids)
            runner.draws.setdefault(ids[i], {})[position] = state
    match runner.selection:
        case Sampling() as sampling:
            replay = Replay.of(
                [runner.requests[req_id] for req_id in ids],
                {row for row, req_id in enumerate(ids) if req_id in values},
                runner.vllm_config.model_config.return_sampling_mask,
            )
            drawn, logprobs, masks = sampling.sample(logits, runner.input_batch, replay)
        case Speculation() as speculation:
            # vLLM serves neither trace replay nor sampling masks with speculative decoding.
            drawn, logprobs = speculation.sample(
                logits, [len(rows) for rows in scores], runner.input_batch
            )
            masks = None
        case _:
            assert_never(runner.selection)
    for i, state in saved.items():
        if ids[i] not in values:
            _ = runner.input_batch.generators[i].set_state(state)
    return (
        {req_id: tokens for req_id, tokens in zip(ids, drawn, strict=True) if req_id in values},
        logprobs,
        masks,
    )


def scores_prompt(runner: Runner, request: CachedRequestState) -> bool:
    """Whether `request` asks for scores of its prompt rows: prompt logprobs, or the logprobs of
    fixed token IDs."""
    return (
        request.in_progress_prompt_logprobs_cpu is not None
        or request.req_id in runner.prompt_scores
    )


def record(
    runner: Runner, request: CachedRequestState, position: int, scores: torch.Tensor
) -> None:
    """Store the scores `request` asks for of prompt row `position` from that row's vocabulary
    `scores`, `[1, vocab]`, in logits or logprobs as vLLM's logprobs mode reports them."""
    mode = runner.vllm_config.model_config.logprobs_mode
    # Prompt rows skip the sampler's processing, so a processed mode reports a raw one's scores.
    match mode:
        case "raw_logits" | "processed_logits":
            values = scores.float()
        case "raw_logprobs" | "processed_logprobs":
            values = Sampler.compute_logprobs(scores)
        case _:
            assert_never(mode)
    if (prompt := request.in_progress_prompt_logprobs_cpu) is not None:
        target = torch.tensor([request.get_token_id(position + 1)], dtype=torch.long)
        part = Sampler.gather_logprobs(values, prompt.logprobs.shape[1] - 1, target)
        _ = prompt.logprob_token_ids[position : position + 1].copy_(part.logprob_token_ids)
        _ = prompt.logprobs[position : position + 1].copy_(part.logprobs)
        _ = prompt.selected_token_ranks[position : position + 1].copy_(part.selected_token_ranks)
    if (asked := runner.prompt_scores.get(request.req_id)) is not None:
        asked.record(position, values[0])


def device_argmax(
    runner: Runner, model: engine.Decoder | None, scheduled: Iterable[str]
) -> engine.Decoder | None:
    """`model` when its tokens may replace vLLM's sampler and greedy rejection: no pooling and an
    unmodified distribution."""
    unmodified = all(
        type(proc) in BUILTIN_LOGITS_PROCESSORS for proc in runner.input_batch.logitsprocs.all
    ) and all(
        params is not None and greedy(params)
        for params in (runner.requests[req_id].sampling_params for req_id in scheduled)
    )
    return model if runner.pooler is None and unmodified else None
