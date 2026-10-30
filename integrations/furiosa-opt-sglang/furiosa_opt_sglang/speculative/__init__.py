"""Serve SGLang's synchronous speculative schedule with one chain of drafts a request.

- A proposer drafts each decoding request: a `Corpus` from the requests' own n-grams, a `Draft`
  from a draft model.
- The worker verifies every request's last token and drafts in one engine call, whose rows
  return the target's greedy tokens, and commits the matching drafts and the target's next token.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, final

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from typing_extensions import override

from furiosa_opt_sglang.batch import Batch
from furiosa_opt_sglang.generation import Sampled, integers, sequence

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs

    from furiosa_opt_sglang import worker
    from furiosa_opt_sglang.decoder import Decoder
    from furiosa_opt_sglang.speculative.ngram import Corpus
    from furiosa_opt_sglang.speculative.standalone import Draft


@dataclass(frozen=True, slots=True)
class Speculative:
    """A causal target, whose calls return each row's greedy token, and the proposer that drafts
    for it."""

    target: Decoder
    proposer: Corpus | Draft
    # How many tokens a verification holds a request: its last token and the drafts after it.
    width: int


@dataclass(frozen=True, slots=True)
class Decoding:
    """A decoding request as a proposer reads it."""

    rid: str
    # The target caches every token but the last.
    tokens: tuple[int, ...]
    # The most drafts the request's context leaves room for.
    drafts: int
    # The request's cache blocks up to its last draft.
    blocks: tuple[int, ...]

    @property
    def cached(self) -> int:
        return len(self.tokens) - 1

    @classmethod
    def parse(cls, request: Req, cached: int, room: int, blocks: tuple[int, ...]) -> Decoding:
        """`request`, of which SGLang caches `cached` tokens, with `room` positions from its last
        token on in `blocks`."""
        tokens = (*request.origin_input_ids, *request.output_ids)
        if cached != len(tokens) - 1:
            raise ValueError(f"SGLang caches {cached} of a decoding request's {len(tokens)} tokens")
        return cls(request.rid, tokens, room - 1, blocks)


@final
class Worker(BaseSpecWorker):
    """SGLang's speculative worker over the runner's speculative target."""

    # SGLang's scheduler reads the worker's draft worker; the proposer drafts instead.
    _draft_worker: None = None

    # SGLang constructs every speculative worker with this signature.
    def __init__(
        self, server_args: ServerArgs, gpu_id: int, nccl_port: int, target_worker: worker.Worker
    ) -> None:
        super().__init__()
        runner = target_worker.runner
        assert runner.speculative is not None, "the runner loads the proposer under speculation"
        assert runner.req_to_token_pool is not None, "SGLang builds its pools before this worker"
        self.speculative: Speculative = runner.speculative
        self.pool: ReqToTokenPool = runner.req_to_token_pool
        self.server_args: ServerArgs = server_args
        self._target_worker: worker.Worker = target_worker
        self.model_runner: ModelRunner = target_worker.model_runner

    @property
    @override
    def target_worker(self) -> worker.Worker:
        return self._target_worker

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish: Callable[[torch.Tensor], None] | None = None,
        grammar_barrier: Callable[[], None] | None = None,
        pp_proxy_tensors: PPProxyTensors | None = None,
    ) -> GenerationBatchResult:
        # The synchronous schedule publishes no future tokens and runs no grammar or pipeline.
        _ = on_publish, grammar_barrier, pp_proxy_tensors
        speculative = self.speculative
        width = speculative.width

        def pack(committed: tuple[tuple[int, ...], ...]) -> torch.Tensor:
            # SGLang reads a request's committed tokens at a stride of the verification's width.
            strided = torch.zeros((len(committed), width), dtype=torch.int64)
            for row, tokens in zip(strided, committed, strict=True):
                row[: len(tokens)] = torch.tensor(tokens)
            return strided.flatten()

        match batch.forward_mode:
            case ForwardMode.EXTEND:
                # A prompt step commits the target's one next token a request; a request that
                # prefills again lost the cache its proposer kept.
                speculative.proposer.forget({request.rid for request in batch.reqs})
                result = self.target_worker.forward_batch_generation(batch)
                if isinstance(result, Sampled):
                    result.collect()
                assert isinstance(result.next_token_ids, torch.Tensor)
                committed = tuple((int(token),) for token in result.next_token_ids)
                packed = pack(committed)
                # SGLang keeps a prompt step's sequence lengths as they are.
                new_seq_lens = batch.seq_lens
            case ForwardMode.DECODE:
                runner = self.target_worker.runner
                context = self.target_worker.model_config.context_len
                block = speculative.target.pool.block
                cached = integers(batch.seq_lens)

                def decoding(request: Req, start: int) -> Decoding:
                    room = min(width, context - start)
                    # A block's first slot names the block.
                    slots = self.pool.req_to_token[request.kv.req_pool_idx, : start + room : block]
                    return Decoding.parse(request, start, room, tuple(integers(slots // block)))

                decodings = tuple(
                    decoding(request, start)
                    for request, start in zip(batch.reqs, cached, strict=True)
                )
                chains = tuple(
                    (decoding.tokens[-1], *drafts)
                    for decoding, drafts in zip(
                        decodings, speculative.proposer.propose(decodings), strict=True
                    )
                )
                predicted: list[list[int | None]] = [[None] * len(chain) for chain in chains]
                # A verification reads its tokens back to the host; no row keeps a device-selected
                # one.
                runner.selected.clear()
                for rows, output in sequence.forward(
                    runner, batch, Batch(chains, tuple(cached)), every=True, feed={}
                ):
                    for (request, offset), token in zip(
                        rows, output.to_host().numpy()[: len(rows)], strict=True
                    ):
                        predicted[request][offset] = int(token)

                def commit(
                    chain: tuple[int, ...], targets: Sequence[int | None]
                ) -> tuple[int, ...]:
                    # A draft is kept while it equals the target's token at the position before it.
                    tokens = tuple(token for token in targets if token is not None)
                    if len(tokens) != len(chain):
                        raise RuntimeError("the engine returned no token for a chain position")
                    kept = next(
                        (
                            i
                            for i, (draft, token) in enumerate(
                                zip(chain[1:], tokens[:-1], strict=True)
                            )
                            if draft != token
                        ),
                        len(chain) - 1,
                    )
                    return tokens[: kept + 1]

                committed = tuple(
                    commit(chain, targets) for chain, targets in zip(chains, predicted, strict=True)
                )
                new_seq_lens = batch.seq_lens + torch.tensor([len(tokens) for tokens in committed])
                packed = pack(committed)
                result = GenerationBatchResult(
                    logits_output=LogitsProcessorOutput(next_token_logits=None),
                    next_token_ids=packed,
                    can_run_cuda_graph=False,
                )
            case mode:
                raise ValueError(f"speculative decoding schedules no {mode} step")
        result.accept_lens = torch.tensor([len(tokens) for tokens in committed])
        result.new_seq_lens = new_seq_lens
        result.next_draft_input = NgramVerifyInput(
            draft_token_num=width,
            new_seq_lens=new_seq_lens,
            accept_tokens=packed,
            accept_lens=result.accept_lens,
        )
        result.speculative_num_draft_tokens = width
        return result

    @override
    def clear_cache_pool(self) -> None:
        self.speculative.proposer.reset()
