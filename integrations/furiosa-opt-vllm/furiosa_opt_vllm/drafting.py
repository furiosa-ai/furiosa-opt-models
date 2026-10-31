"""Draft each request's next tokens with a causal language model loaded beside the target.

- `Drafter.load` loads the draft model once the target is loaded, on the target's chips unless
  a draft placement names others, refusing one whose cache cannot follow the target's: another
  vocabulary or block size, or a shorter window.
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
    from furiosa_opt_vllm.decoder import Decoder


@dataclass(frozen=True, slots=True)
class Drafter:
    """A loaded draft model, the device it runs on and the draft settings it loaded with."""

    model: engine.Decoder
    device: engine.Device
    draft: Draft

    @classmethod
    def load(cls, draft: Draft, target: Decoder | None) -> Drafter:
        """`draft`'s model beside `target`, a token-input decoder: on its device, whose memory
        the two caches share, unless the draft placement names other chips."""
        if target is None or target.model is None:
            raise Refused(
                "speculative_config.model",
                "speculative_config.model requires a target whose attention cache takes token IDs",
            )
        # The vocabulary decides before the draft model loads, so a refusal reads no weight.
        if draft.model.get_vocab_size() != target.vocabulary:
            raise Refused(
                "speculative_config.model",
                "speculative_config.model requires the target vocabulary",
            )
        options = engine.LoadOptions().chips(draft.parallel.tensor_parallel_size)
        if draft.cache_dtype is not None:
            options = options.kv_cache(draft.cache_dtype)
        device = target.device
        if draft.placement is not None:
            pes = range(*draft.placement.pes)
            device = engine.Device(engine.Placement(list(draft.placement.chips), pes))
        loaded = options.load(checkpoint.located(draft.model, draft.download_dir).mapped(), device)
        match loaded:
            case engine.Decoder(output=engine.decoder.Output.TOKENS):
                model = loaded
            case _:
                raise Refused(
                    "speculative_config.model",
                    "speculative_config.model requires a causal language model",
                )
        blocks = {pool.block for pool in target.model.cache.pools}
        if {pool.block for pool in model.cache.pools} != blocks:
            raise Refused(
                "speculative_config.model",
                "speculative_config.model requires the target's cache block size",
            )
        if model.window < draft.max_model_len:
            raise Refused(
                "speculative_config.model",
                "speculative_config.model requires a context window at least max_model_len",
            )
        return cls(model, device, draft)

    def capacity(self, blocks: int, block: int) -> int:
        """Cache blocks of a target whose `blocks` blocks of `block` bytes fill its free memory,
        when the drafter's cache must hold as many: on the target's device the two share it."""
        cache = self.model.cache
        size = sum(cache.pools[group.pool].bytes * len(group.layers) for group in cache.groups)
        if self.draft.placement is None:
            return blocks * block // (block + size)
        return min(blocks, max(0, self.device.available) // size)


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
        self.cache = engine.decoder.Cache(drafter.model.cache, blocks)
        self.requests = requests
        # req_id -> how many of the request's leading positions the cache holds.
        self.cached: dict[str, int] = {}

    def propose(self, batch: InputBatch, sampled: list[list[int]], count: int) -> list[list[int]]:
        """Up to `count` greedy drafts after each request's known tokens in `batch`, in batch order;
        none for a request that sampled no token this step."""
        model = self.drafter.model
        block = model.cache.pools[0].block
        plans: dict[int, Plan] = {}
        for index, tokens in enumerate(sampled):
            if not tokens or not count:
                continue
            req_id = batch.req_ids[index]
            blocks = self.requests[req_id].block_ids[0]
            end = int(batch.num_tokens_no_spec[index])
            # vLLM's blocks hold the known tokens and a slot per draft, but end at max_model_len.
            room = min(len(blocks) * block, model.window)
            assert end <= room, "the target's blocks hold every known token"
            # The final known token may replace a rejected draft, so it is always appended again.
            start = min(self.cached.get(req_id, 0), end - 1)
            plans[index] = Plan(req_id, blocks, start, end, min(count, room - end + 1))
        drafts = self.append(batch, plans)
        # Draft each later position from the previous draft, in calls of at most `rows` requests.
        for step in range(1, max((plan.drafts for plan in plans.values()), default=0)):
            live = [index for index, plan in plans.items() if plan.drafts > step]
            for first in range(0, len(live), model.rows):
                part = live[first : first + model.rows]
                output = model.forward(
                    [
                        engine.decoder.Request(
                            self.tables(plans[index]),
                            plans[index].end + step - 1,
                            last[row : row + 1],
                            engine.decoder.Keep.LAST,
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

    def tables(self, plan: Plan) -> list[list[int]]:
        """The block table of `plan`'s request for every group of the drafter's cache, which
        allocates as many blocks of each."""
        return [plan.blocks] * len(self.drafter.model.cache.groups)

    def append(
        self, batch: InputBatch, plans: Mapping[int, Plan]
    ) -> dict[int, list[tuple[engine.HbmTensor, int]]]:
        """Append the known positions each planned request's cache lacks, in calls of at most
        `rows` rows; each request's first draft is on the device, in the output of the call that
        appends its last known position, at its row there."""
        model = self.drafter.model
        rows = [
            Row(index, position, position + 1 == plan.end)
            for index, plan in plans.items()
            for position in range(plan.start, plan.end)
        ]
        drafts: dict[int, list[tuple[engine.HbmTensor, int]]] = {index: [] for index in plans}
        for first in range(0, len(rows), model.rows):
            chunk = rows[first : first + model.rows]
            call = Call[int].plan(chunk, every=False)
            ids = engine.HostTensor.from_dlpack(
                operand(
                    [int(batch.token_ids_cpu[row.sequence, row.position]) for row in chunk],
                    self.drafter.draft.model.get_vocab_size(),
                )
            ).to_hbm(self.drafter.device)
            output = model.forward(
                [
                    engine.decoder.Request(
                        self.tables(plans[run.sequence]),
                        run.cached,
                        ids[run.rows[0] : run.rows[-1] + 1],
                        engine.decoder.Keep.LAST if run.complete else engine.decoder.Keep.NONE,
                    )
                    for run in call.requests
                ],
                self.cache,
            )
            for row, run in enumerate(run for run in call.requests if run.complete):
                drafts[run.sequence].append((output, row))
        return drafts
