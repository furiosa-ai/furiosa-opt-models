"""Apply vLLM's watermark to eligible rows of a step's processed logits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, assert_never

import torch
from vllm.v1.sample.ops.topk_topp_sampler import random_sample
from vllm.v1.watermarking import DualKeyGumbelWatermarker, Watermarker, create_watermarker
from vllm.v1.worker.gpu.sample.watermark import repeated_context_mask

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vllm.config.watermarking import WatermarkConfig
    from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch


@dataclass(frozen=True, slots=True)
class DualKey:
    """vLLM's dual-key watermark: key A's watermarker, and key B's, which a row draws with
    probability `alpha` in a race of float64 noise if `fp64`."""

    a: Watermarker
    b: Watermarker
    alpha: float
    fp64: bool


@dataclass(frozen=True, slots=True)
class Search:
    """The history searched for a row's context: the request's generated tokens, and its prompt
    with `prompt`, over the last `history` positions or, with `None`, all of them."""

    prompt: bool
    history: int | None


@dataclass(frozen=True, slots=True)
class Marked:
    """The batch `rows` of a step that draw vLLM's watermark token, the context of each, and the
    seeded random `streams` of some, by their index in `rows`."""

    rows: torch.Tensor
    contexts: torch.Tensor
    streams: dict[int, torch.Generator]


@dataclass(frozen=True, slots=True)
class Watermark:
    """A server's watermark: the watermarker a marked row draws with, the context width, and the
    search that leaves a row whose context repeats to the sampler, or `None` to search nowhere."""

    keys: Watermarker | DualKey
    width: int
    search: Search | None

    @classmethod
    def parse(cls, config: WatermarkConfig, fp64: bool) -> Watermark:
        """The watermark vLLM's `config` describes, whose race between two keys takes float64
        noise if `fp64`, as vLLM's does under --use-fp64-gumbel."""
        watermarker = create_watermarker(config)
        # vLLM routes between the keys on its device sampler's random stream, which needs Triton;
        # with an alpha of 0 or 1 it draws the one key itself.
        match watermarker:
            case DualKeyGumbelWatermarker(alpha=alpha) if 0 < alpha < 1:
                keys: Watermarker | DualKey = DualKey(
                    watermarker.draft_watermarker, watermarker.target_watermarker, alpha, fp64
                )
            case _:
                keys = watermarker
        history = config.deduplicate_contexts_max_history
        match config.deduplicate_contexts:
            case "none":
                search = None
            case "single_turn":
                search = Search(prompt=False, history=history)
            case "all":
                search = Search(prompt=True, history=history)
            case _:
                assert_never(config.deduplicate_contexts)
        return cls(keys, config.context_width, search)

    def mark(self, batch: InputBatch, requests: Mapping[str, CachedRequestState]) -> Marked:
        """The rows of `batch` that draw the watermark: each request that keeps it and samples at a
        temperature, unless its context repeats in the searched history."""
        rows = torch.tensor(
            [
                row
                for row, req_id in enumerate(batch.req_ids)
                if (params := requests[req_id].sampling_params) is not None
                and params.watermarking
                and params.temperature != 0
            ],
            dtype=torch.int64,
        )
        # A row's context is its last `width` generated tokens, -1 before the first.
        known = batch.num_tokens_no_spec_cpu_tensor[rows].long().unsqueeze(-1)
        positions = known + torch.arange(-self.width, 0)
        prompt = batch.num_prompt_tokens_cpu_tensor[rows].long().unsqueeze(-1)
        tokens = batch.token_ids_cpu_tensor[rows.unsqueeze(-1), positions.clamp(min=0)].long()
        contexts = torch.where(positions >= prompt, tokens, -1)
        match self.search:
            case None:
                pass
            case Search(prompt=searched, history=history):
                kept = ~repeated_context_mask(
                    batch.token_ids_cpu_tensor,
                    rows,
                    batch.num_prompt_tokens_cpu_tensor,
                    batch.num_tokens_no_spec_cpu_tensor,
                    contexts,
                    history,
                    include_prompt=searched,
                    skip_partial_context=searched,
                )
                rows, contexts = rows[kept], contexts[kept]
            case _:
                assert_never(self.search)
        streams = {
            index: batch.generators[int(row)]
            for index, row in enumerate(rows)
            if int(row) in batch.generators
        }
        return Marked(rows, contexts, streams)

    def apply(self, masked: torch.Tensor, sampled: torch.Tensor, marked: Marked) -> torch.Tensor:
        """Replace marked tokens; each row's stream selects a dual key."""
        logits = masked[marked.rows]
        match self.keys:
            case DualKey(a=a, b=b, alpha=alpha, fp64=fp64):
                odds = torch.tensor([1 - alpha, alpha]).repeat(len(marked.rows), 1)
                tokens = torch.where(
                    random_sample(odds, marked.streams, fp64) == 1,
                    b.sample(logits, marked.contexts).token_ids,
                    a.sample(logits, marked.contexts).token_ids,
                )
            case Watermarker() as watermarker:
                tokens = watermarker.sample(logits, marked.contexts).token_ids
            case _:
                assert_never(self.keys)
        return sampled.index_copy(0, marked.rows, tokens)
