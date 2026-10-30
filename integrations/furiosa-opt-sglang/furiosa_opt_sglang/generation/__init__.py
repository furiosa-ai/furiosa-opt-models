"""Select each scheduled request's next token with SGLang's generation contract.

- A step returns once its engine calls are submitted, without waiting for them.
- A greedy step whose causal model selects tokens on the device (`Sampled`) leaves them there: the
  plugin's hook reads them as SGLang processes the step (`Sampled.collect`), and each request's next
  decode row takes its token on the device, so the next step need not wait for this one.
- Any other step reads scores for SGLang's sampler, which a decoder's calls do not return, so it
  raises.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, assert_never, cast, final

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.utils import GenerationBatchResult

import furiosa_opt_engine as engine
from furiosa_opt_sglang.batch import Batch
from furiosa_opt_sglang.decoder import RELEASED, Decoder
from furiosa_opt_sglang.generation import sequence
from furiosa_opt_sglang.generation.model import Model

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

    from furiosa_opt_sglang.runner import Runner

    # One engine call's output rows, with the (request, row) pair each holds.
    Output = tuple[list[tuple[int, int]], engine.HbmTensor]

# PyTorch erases element types from tensor list conversion.
integers = cast("Callable[[torch.Tensor], list[int]]", torch.Tensor.tolist)


@dataclass(frozen=True, slots=True)
class Schedule:
    """Copy scheduler framing and score selection before device execution."""

    inputs: Batch
    extend: bool
    logprob: bool
    starts: tuple[int, ...]
    top: tuple[int, ...]
    ids: tuple[tuple[int, ...], ...]
    targets: tuple[tuple[int, ...], ...]

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(len(tokens) for tokens in self.inputs.tokens)

    @classmethod
    def parse(cls, batch: ScheduleBatch) -> Schedule:
        extend = batch.forward_mode.is_extend()
        if not (extend or batch.forward_mode.is_decode()):
            raise ValueError(f"unsupported forward mode {batch.forward_mode}")
        lengths = batch.extend_lens if extend else [1] * len(batch.reqs)
        inputs = Batch.parse(integers(batch.input_ids), lengths, integers(batch.seq_lens))
        sources = batch.encoder_lens_cpu or [0] * len(lengths)
        return cls(
            inputs,
            extend,
            batch.return_logprob,
            tuple(batch.extend_logprob_start_lens) if extend else (),
            tuple(batch.top_logprobs_nums or [0] * len(lengths)),
            tuple(tuple(ids or ()) for ids in (batch.token_ids_logprobs or [()] * len(lengths))),
            tuple(
                tuple(
                    request.full_untruncated_fill_ids[position]
                    if position < len(request.full_untruncated_fill_ids)
                    else 0
                    for position in range(cached + source + 1, cached + source + length + 1)
                )
                for request, cached, source, length in zip(
                    batch.reqs, inputs.cached, sources, lengths, strict=True
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class Token:
    """A request slot's next token: a row of the tokens a greedy step selected on the device."""

    tokens: engine.HbmTensor
    row: int

    def operand(self) -> engine.HbmTensor:
        return self.tokens[self.row : self.row + 1]


@final
class Sampled(GenerationBatchResult):
    """A greedy step whose tokens stay on the device until SGLang processes the step."""

    def __init__(self, calls: tuple[Output, ...], size: int) -> None:
        # SGLang copies and relays these zeros before it processes the step; the decode rows that
        # follow take the device's tokens instead.
        super().__init__(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=torch.zeros(size, dtype=torch.int64),
            can_run_cuda_graph=False,
        )
        self.size = size
        self.calls = calls

    def collect(self) -> None:
        """Replace the step's tokens in the result SGLang is about to process, once."""
        if not self.calls:
            return
        ids = torch.zeros(self.size, dtype=torch.int64)
        for kept, tokens in self.calls:
            rows = torch.tensor([i for i, _ in kept])
            ids[rows] = torch.tensor(tokens.to_host().numpy()[: len(kept)], dtype=torch.int64)
        self.next_token_ids = ids
        self.calls = ()


def issue(runner: Runner, model: Model, batch: ScheduleBatch, *, overlap: bool) -> Sampled:
    """Submit the step's engine calls and return its result without waiting for them; with
    `overlap`, SGLang processes the step before this one while the device runs."""
    issued = runner.issued
    # SGLang processes a step before it issues the second one after it, or the next one without
    # overlap, and the plugin's hook collects a greedy step's tokens as SGLang processes it.
    while len(issued) > (1 if overlap else 0):
        match issued.popleft():
            case Sampled(calls=calls) if calls:
                raise RuntimeError("SGLang processed a greedy step without collecting its tokens")
            case Sampled():
                pass
    schedule = Schedule.parse(batch)
    sampling = batch.sampling_info
    # SGLang fills the step's grammars as it builds a ForwardBatch, which the engine call skips.
    sampling.grammars = [request.grammar for request in batch.reqs] if batch.has_grammar else None
    # SGLang's overlap loop hands the step a copy that holds its penalties; its other loop hands its
    # own, which keeps changing.
    if sampling.penalizer_orchestrator is not None:
        sampling = sampling.copy_for_forward()
    if device_argmax(model, sampling, logprob=schedule.logprob) is None:
        raise NotImplementedError(
            "sampling and scoring read scores the engine's calls do not return"
        )
    selected = runner.selected
    slots = integers(batch.req_pool_indices)
    # A decoding request's one row is the token its last step selected.
    decode, running = batch.forward_mode.is_decode(), set(batch.decoding_reqs or ())
    feed = {
        i: token.operand()
        for i, request in enumerate(batch.reqs)
        if (decode or request in running) and (token := selected.get(slots[i])) is not None
    }
    outputs = sequence.forward(runner, batch, schedule.inputs, every=schedule.logprob, feed=feed)
    calls = tuple((kept, tokens) for kept, tokens in outputs if kept)
    for kept, tokens in calls:
        selected.update((slots[i], Token(tokens, row)) for row, (i, _) in enumerate(kept))
    result = Sampled(calls, len(slots))
    issued.append(result)
    return result


def device_argmax(
    model: Model, sampling: SamplingBatchInfo, *, logprob: bool
) -> engine.Decoder | None:
    """The causal model whose device argmax selects the tokens SGLang's sampler would, if any: a
    greedy step without logprobs, penalties, grammar, logit bias or sampling masks."""
    match model:
        case Model(
            core=Decoder(model=engine.Decoder(output=engine.decoder.Output.TOKENS) as causal)
        ):
            pass
        case Model(core=Decoder(model=None)):
            raise RuntimeError(RELEASED)
        case Model():
            return None
        case _:
            assert_never(model)
    greedy = (
        sampling.is_all_greedy
        and not logprob
        and not sampling.grammars
        and sampling.logit_bias is None
        and sampling.acc_additive_penalties is None
        and sampling.acc_scaling_penalties is None
        and sampling.sampling_mask_batch_indices is None
        and not sampling.has_custom_logit_processor
        and not sampling.has_watermark_candidates
    )
    return causal if greedy else None
