"""Draft each request's next tokens with a causal language model loaded beside the target.

- `Drafter.load` loads the draft model once the target is loaded, refusing one whose cache cannot
  follow the target's: another vocabulary or block size, or a shorter window.
- `Drafting` is a drafter with its cache, which the target's block tables address, and the
  leading positions of each request that cache holds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, final

import furiosa_opt_engine as engine
from furiosa_opt_vllm import checkpoint
from furiosa_opt_vllm.admission import Refused
from furiosa_opt_vllm.batch import Call, Row, operand
from furiosa_opt_vllm.device import read

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

    from furiosa_opt_vllm.admission import Draft


@dataclass(frozen=True, slots=True)
class Drafter:
    """A loaded draft model whose cache takes the target's token IDs and block tables."""

    model: engine.relation.CausalLM[engine.Sequence]

    @classmethod
    def load(cls, draft: Draft, target: engine.CacheSpec | None) -> Drafter:
        """`draft`'s model on its placement, by default on the chips the target left free, for a
        target whose attention cache is `target`."""
        if target is None or target.vocabulary is None:
            raise Refused(
                "speculative_config.model",
                "speculative_config.model requires a target whose attention cache takes token IDs",
            )
        options = engine.LoadOptions().parallelism(
            tensor=draft.parallel.tensor_parallel_size,
            pipeline=draft.parallel.pipeline_parallel_size,
        )
        if draft.placement is not None:
            options = options.placement(
                engine.Placement(draft.placement.chips, range(*draft.placement.pes))
            )
        if draft.cache_dtype is not None:
            options = options.cache_dtype(draft.cache_dtype)
        loaded = checkpoint.load(
            checkpoint.checkout(draft.model.model, draft.model.revision, draft.download_dir),
            options,
            draft.model,
        )
        match loaded:
            case engine.relation.CausalLM(scores=engine.relation.Logits(body=engine.Sequence())):
                model = loaded
            case _:
                raise Refused(
                    "speculative_config.model",
                    "speculative_config.model requires a token-input CausalLM attention cache",
                )
        cache = model.scores.body.cache
        if cache.vocabulary != target.vocabulary or cache.block.tokens != target.block.tokens:
            raise Refused(
                "speculative_config.model",
                "speculative_config.model requires the target vocabulary and cache block size",
            )
        if cache.window < draft.max_model_len:
            raise Refused(
                "speculative_config.model",
                "speculative_config.model requires a context window at least max_model_len",
            )
        return cls(model)

    def capacity(self) -> int:
        """Cache blocks that fit in the free memory of the draft model's device."""
        body = self.model.scores.body
        return body.cache.blocks(max(0, body.device.available - body.cache.workspace))


@dataclass(frozen=True, slots=True)
class Plan:
    """What one request appends and drafts in a proposal: the known positions `start:end` its
    cache lacks, then `drafts` tokens."""

    req_id: str
    blocks: list[int]
    start: int
    end: int
    drafts: int


@final
class Drafting:
    """A drafter with a cache of `blocks` blocks, addressed by the block tables of the runner's
    `requests`."""

    def __init__(
        self, drafter: Drafter, blocks: int, requests: Mapping[str, CachedRequestState]
    ) -> None:
        self.drafter = drafter
        self.cache = engine.Cache(drafter.model.scores.body.cache, blocks)
        self.requests = requests
        # req_id -> how many of the request's leading positions the cache holds.
        self.cached: dict[str, int] = {}

    def propose(self, batch: InputBatch, sampled: list[list[int]], count: int) -> list[list[int]]:
        """Up to `count` greedy drafts after each request's known tokens in `batch`, in batch order;
        none for a request that sampled no token this step."""
        model = self.drafter.model
        cache = model.scores.body.cache
        plans: dict[int, Plan] = {}
        for index, tokens in enumerate(sampled):
            if not tokens or not count:
                continue
            req_id = batch.req_ids[index]
            blocks = self.requests[req_id].block_ids[0]
            end = int(batch.num_tokens_no_spec[index])
            # vLLM's blocks hold the known tokens and a slot per draft, but end at max_model_len.
            room = min(len(blocks) * cache.block.tokens, cache.window)
            assert end <= room, "the target's blocks hold every known token"
            # The final known token may replace a rejected draft, so it is always appended again.
            start = min(self.cached.get(req_id, 0), end - 1)
            plans[index] = Plan(req_id, blocks, start, end, min(count, room - end + 1))
        drafts = self.append(batch, plans)
        # Draft each later position from the previous draft, in calls of at most `rows` requests.
        for step in range(1, max((plan.drafts for plan in plans.values()), default=0)):
            live = [index for index, plan in plans.items() if plan.drafts > step]
            for first in range(0, len(live), cache.rows):
                part = live[first : first + cache.rows]
                output = model.forward(
                    [
                        engine.Request(
                            plans[index].blocks, plans[index].end + step - 1, last[row : row + 1]
                        )
                        for index in part
                        for last, row in [drafts[index][-1]]
                    ],
                    self.cache,
                )
                for row, index in enumerate(part):
                    drafts[index].append((output, row))
        # Read the drafts once every call is submitted, then record what the cache now holds.
        proposed = read(drafts)
        for plan in plans.values():
            self.cached[plan.req_id] = plan.end + plan.drafts - 1
        return [proposed.get(index, []) for index in range(len(sampled))]

    def retire(self, req_id: str) -> None:
        """Forget what the cache holds of `req_id`, whose blocks vLLM freed or replaced."""
        _ = self.cached.pop(req_id, None)

    def append(
        self, batch: InputBatch, plans: Mapping[int, Plan]
    ) -> dict[int, list[tuple[engine.HbmTensor, int]]]:
        """Append the known positions each planned request's cache lacks, in calls of at most
        `rows` rows; each request's first draft is on the device, in the output of the call that
        appends its last known position, at its row there."""
        model = self.drafter.model
        body = model.scores.body
        rows = [
            Row(index, position, position + 1 == plan.end)
            for index, plan in plans.items()
            for position in range(plan.start, plan.end)
        ]
        drafts: dict[int, list[tuple[engine.HbmTensor, int]]] = {index: [] for index in plans}
        for first in range(0, len(rows), body.cache.rows):
            chunk = rows[first : first + body.cache.rows]
            call = Call[int].plan(chunk, every=False)
            ids = engine.HostTensor.from_dlpack(
                operand(
                    [int(batch.token_ids_cpu[row.sequence, row.position]) for row in chunk],
                    body.cache.vocabulary,
                )
            ).to_hbm(body.device)
            requests = [
                engine.Request(
                    plans[run.sequence].blocks,
                    run.cached,
                    ids[run.rows[0] : run.rows[-1] + 1],
                    engine.Keep.LAST if run.complete else engine.Keep.NONE,
                )
                for run in call.requests
            ]
            if not call.kept:
                _ = body.forward(requests, self.cache)
                continue
            output = model.forward(requests, self.cache)
            for row, run in enumerate(run for run in call.requests if run.complete):
                drafts[run.sequence].append((output, row))
        return drafts
