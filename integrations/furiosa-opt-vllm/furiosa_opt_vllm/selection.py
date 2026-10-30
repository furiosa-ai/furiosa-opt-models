"""Draw each request's tokens and propose the drafts vLLM schedules next."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate
from typing import TYPE_CHECKING, assert_never

import torch
from typing_extensions import override
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer

from furiosa_opt_vllm.admission import Draft, Method
from furiosa_opt_vllm.device import read
from furiosa_opt_vllm.drafting import Drafter, Drafting
from furiosa_opt_vllm.watermark import Watermark

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from vllm.config import VllmConfig
    from vllm.config.model import LogprobsMode
    from vllm.v1.outputs import LogprobsLists, SamplingMaskLists
    from vllm.v1.sample.metadata import SamplingMetadata
    from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

    import furiosa_opt_engine as engine
    from furiosa_opt_vllm.admission import Speculative
    from furiosa_opt_vllm.replay import Replay


def choose(
    config: VllmConfig,
    speculative: Speculative | None,
    target: engine.decoder.CacheSpec | None,
    requests: Mapping[str, CachedRequestState],
) -> Sampling | Speculation:
    """Choose sampling or rejection; a draft model loads beside the target cache."""
    model = config.model_config

    def sampler() -> Sampler:
        return Sampler(model.logprobs_mode, model.use_fp64_gumbel)

    match speculative:
        case None if config.watermark_config is not None:
            watermark = Watermark.parse(config.watermark_config, model.use_fp64_gumbel)
            return Sampling(sampler(), (watermark, requests))
        case None:
            return Sampling(sampler())
        case Method.NGRAM:
            proposer: NgramProposer | SuffixDecodingProposer | Drafter = NgramProposer(config)
        case Method.SUFFIX:
            proposer = SuffixDecodingProposer(config)
        case Draft():
            proposer = Drafter.load(speculative, target)
        case _:
            assert_never(speculative)
    return Speculation(RejectionSampler(sampler()), proposer)


@dataclass(frozen=True, slots=True)
class Sampling:
    """vLLM's sampler, which draws each request's next token from its one sampled row, or vLLM's
    watermark token for a request that keeps a server's watermark."""

    sampler: Sampler
    watermark: tuple[Watermark, Mapping[str, CachedRequestState]] | None = None

    def sample(
        self, logits: torch.Tensor, batch: InputBatch, replay: Replay
    ) -> tuple[list[list[int]], LogprobsLists | None, SamplingMaskLists | None]:
        """Each request's token from its one row of `logits`, in batch order, as `replay` forces
        or draws it, its logprobs, and the supports `replay` reports."""
        masks = None
        if self.watermark is None and not replay.forced and replay.supported is None:
            result = self.sampler.forward(logits, batch.sampling_metadata)
        else:
            marked = (
                (self.watermark[0], self.watermark[0].mark(batch, self.watermark[1]))
                if self.watermark is not None
                else None
            )

            class Scoped(Sampler):
                @override
                def sample(
                    scoped,
                    logits: torch.Tensor,
                    sampling_metadata: SamplingMetadata,
                    logprobs_mode_override: LogprobsMode | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
                    nonlocal masks
                    tokens, distribution = super().sample(logits, sampling_metadata)
                    processed = logits
                    if not sampling_metadata.all_greedy:
                        assert distribution is not None
                        processed = distribution
                        if marked is not None:
                            tokens = marked[0].apply(processed, tokens, marked[1])
                    for row, token in replay.forced.items():
                        tokens[row] = token
                    masks = replay.supports(processed)
                    if (
                        sampling_metadata.max_num_logprobs is None
                        and not sampling_metadata.logprob_token_ids
                    ):
                        return tokens, None
                    match logprobs_mode_override or scoped.logprobs_mode:
                        case "raw_logits" | "raw_logprobs":
                            return tokens, None
                        case "processed_logits":
                            return tokens, processed
                        case "processed_logprobs":
                            return tokens, Sampler.compute_logprobs(processed)
                        case _:
                            assert_never(scoped.logprobs_mode)

            scoped = Scoped(self.sampler.logprobs_mode, self.sampler.use_fp64_gumbel)
            scoped.topk_topp_sampler = TopKTopPSampler(
                "processed_logits", self.sampler.use_fp64_gumbel
            )
            result = scoped.forward(logits, batch.sampling_metadata)
        logprobs = result.logprobs_tensors
        return (
            result.sampled_token_ids.tolist(),
            logprobs.tolists() if logprobs is not None else None,
            masks,
        )

    def capacity(self, blocks: int) -> int:
        """`blocks`, since sampling holds no cache."""
        return blocks

    def allocate(self, _blocks: int, _requests: Mapping[str, CachedRequestState]) -> None:
        """Nothing, since sampling holds no cache."""

    def retire(self, _req_id: str) -> None:
        """Nothing, since sampling holds no request's state."""

    def propose(self, _batch: InputBatch, _sampled: list[list[int]], _count: int) -> None:
        """Nothing, since sampling drafts no tokens."""


@dataclass(slots=True)
class Speculation:
    """Accept a draft prefix with vLLM rejection and propose the next step's drafts."""

    rejection: RejectionSampler
    # A draft model proposes once `allocate` gives it a cache.
    proposer: NgramProposer | SuffixDecodingProposer | Drafter | Drafting
    # The drafts proposed after the last sampled step, until vLLM takes them.
    drafts: DraftTokenIds | None = None

    def accept(
        self, tokens: Mapping[str, Sequence[tuple[engine.HbmTensor, int]]], batch: InputBatch
    ) -> dict[str, list[int]]:
        """The tokens greedy rejection keeps of each request's device argmax `tokens`, one per draft
        vLLM's `batch` holds for it and one more: each the output of the call that writes it and
        its row there."""
        host = read(tokens)
        return {
            req_id: accepted(host[req_id], batch.spec_token_ids[index])
            for index, req_id in enumerate(batch.req_ids)
            if req_id in host
        }

    def sample(
        self, logits: torch.Tensor, rows: Sequence[int], batch: InputBatch
    ) -> tuple[list[list[int]], LogprobsLists | None]:
        """Each request's tokens and their logprobs from its `rows` consecutive rows of
        `logits`, in batch order: one row per draft vLLM's batch holds for it, then one more."""
        # vLLM keeps more draft lists than the batch has requests.
        drafts = batch.spec_token_ids[: len(rows)]
        if any(count != len(tokens) + 1 for count, tokens in zip(rows, drafts, strict=True)):
            raise RuntimeError("a request's sampled rows must number one more than its drafts")
        ends = list(accumulate(rows))
        metadata = SpecDecodeMetadata(
            draft_token_ids=torch.tensor(
                [token for tokens in drafts for token in tokens], dtype=torch.int32
            ),
            num_draft_tokens=[count - 1 for count in rows],
            cu_num_draft_tokens=torch.tensor(
                list(accumulate(count - 1 for count in rows)), dtype=torch.int32
            ),
            cu_num_sampled_tokens=torch.tensor(ends, dtype=torch.int32),
            target_logits_indices=torch.tensor(
                [
                    index
                    for end, count in zip(ends, rows, strict=True)
                    for index in range(end - count, end - 1)
                ],
                dtype=torch.int32,
            ),
            bonus_logits_indices=torch.tensor([end - 1 for end in ends], dtype=torch.int32),
            logits_indices=torch.arange(ends[-1], dtype=torch.int32),
        )
        result = self.rejection.forward(metadata, None, logits, batch.sampling_metadata)
        return RejectionSampler.parse_output(
            result.sampled_token_ids, batch.vocab_size, logprobs_tensors=result.logprobs_tensors
        )

    def capacity(self, blocks: int) -> int:
        """The target's cache `blocks`, or fewer when a draft model's cache must hold as many."""
        match self.proposer:
            case NgramProposer() | SuffixDecodingProposer():
                return blocks
            case (Drafter() as drafter) | Drafting(drafter=drafter):
                return min(blocks, drafter.capacity())
            case _:
                assert_never(self.proposer)

    def allocate(self, blocks: int, requests: Mapping[str, CachedRequestState]) -> None:
        """Give a draft model a cache of `blocks` blocks, which `requests`' block tables address."""
        match self.proposer:
            case NgramProposer() | SuffixDecodingProposer():
                pass
            case (Drafter() as drafter) | Drafting(drafter=drafter):
                self.proposer = Drafting(drafter, blocks, requests)
            case _:
                assert_never(self.proposer)

    def retire(self, req_id: str) -> None:
        """Forget what a draft model's cache holds of `req_id`."""
        match self.proposer:
            # Suffix decoding ends a request once the batch no longer holds it.
            case NgramProposer() | SuffixDecodingProposer() | Drafter():
                pass
            case Drafting():
                self.proposer.retire(req_id)
            case _:
                assert_never(self.proposer)

    def propose(self, batch: InputBatch, sampled: list[list[int]], count: int) -> None:
        """Draft up to `count` tokens after each request's `sampled` tokens, in batch order, from
        vLLM's batch rows, which hold the sampled tokens."""
        match self.proposer:
            case NgramProposer():
                drafts = self.proposer.propose(
                    count, sampled, batch.num_tokens_no_spec, batch.token_ids_cpu
                )
            # Suffix decoding keys its cache by request ID: a request missing from the batch ends,
            # and its response stays to draft for later requests.
            case SuffixDecodingProposer():
                drafts = self.proposer.propose(count, batch, sampled)
            case Drafting():
                drafts = self.proposer.propose(batch, sampled, count)
            case Drafter():
                raise RuntimeError("initialize_kv_cache must precede drafting")
            case _:
                assert_never(self.proposer)
        self.drafts = DraftTokenIds(list(batch.req_ids), drafts)


def accepted(targets: Sequence[int], drafts: Sequence[int]) -> list[int]:
    """The tokens greedy rejection keeps from the greedy `targets` of a request's sampled rows,
    one per draft and one more: each draft's target while it equals the draft, then the first
    that differs or, after the last draft, the last row's."""
    kept = next(
        (
            index
            for index, (target, draft) in enumerate(zip(targets[:-1], drafts, strict=True))
            if target != draft
        ),
        len(drafts),
    )
    return list(targets[: kept + 1])
