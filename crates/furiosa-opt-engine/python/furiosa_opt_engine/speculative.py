"""Framework-neutral execution of sequential and parallel speculative drafts."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, cast, final

import numpy as np
from numpy.typing import NDArray

import furiosa_opt_engine as engine
from furiosa_opt_engine.batch import Call, Row, chunks, operand

SequenceId = TypeVar("SequenceId", bound=Hashable)

__all__ = [
    "DraftRequest",
    "DraftSession",
    "SequenceId",
    "TokenSource",
    "cache_blocks_for_memory",
    "cache_memory_budget",
    "draft_blocks_for_shared_budget",
    "validate_draft_decoder",
    "validate_draft_vocabulary",
    "validate_mask_token",
    "same_tokenizer_vocabulary",
]


class TokenSource(Protocol):
    """An indexable token sequence, stable during one proposal; may expose CPU storage lazily."""

    def __len__(self) -> int: ...

    def __getitem__(self, position: int, /) -> int: ...


@dataclass(frozen=True, slots=True)
class DraftRequest(Generic[SequenceId]):
    """One request's known tokens, maximum reusable prefix, cache tables and desired draft count."""

    sequence: SequenceId
    tokens: TokenSource
    reusable: int
    blocks: Sequence[Sequence[int]]
    drafts: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "blocks", tuple(tuple(table) for table in self.blocks))
        if type(self.drafts) is not int or self.drafts < 0:
            raise ValueError("draft count must be a nonnegative integer")
        end = len(self.tokens)
        if end < 1:
            raise ValueError("a draft request must contain at least one known token")
        if type(self.reusable) is not int or not 0 <= self.reusable <= end:
            raise ValueError("reusable prefix length must lie within the known tokens")


@dataclass(frozen=True, slots=True)
class _Plan(Generic[SequenceId]):
    request: DraftRequest[SequenceId]
    start: int
    count: int

    @property
    def end(self) -> int:
        return len(self.request.tokens)


@dataclass(frozen=True, slots=True)
class _Held:
    """A sequence prefix written into the cache blocks it last named."""

    positions: int
    blocks: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class _OutputRow:
    output: engine.HbmTensor
    row: int


@final
class DraftSession(Generic[SequenceId]):
    """A draft decoder cache and the known-token prefix each sequence has written into it.

    Adapters translate framework requests into `DraftRequest`, then retire sequence IDs when
    their blocks are released. The session requires paged attention cache groups; it does not
    manage recurrent-state snapshots across speculative acceptance and rejection. With no mask,
    each proposed token feeds the next decoder call; with a mask, proposals follow known tokens
    and masks in the same calls. Proposals and lifecycle changes mutate shared state and must be
    serialized; token sources must remain stable until `propose` returns.
    """

    def __init__(
        self,
        model: engine.Decoder,
        device: engine.Device,
        blocks: int,
        vocabulary: int,
        mask: int | None = None,
    ) -> None:
        """Build a session for `model` and its token-upload `device`.

        `device` must be where `model` runs; `vocabulary` must describe its token IDs.
        """
        mask = validate_mask_token(mask, vocabulary)
        _require_token_output(model, "draft session")
        self._block_sizes: tuple[int, ...] = _paged_cache_blocks(model, "draft session")
        self._cache_layout = _cache_layout(model)
        self._cache_spec: engine.decoder.CacheSpec = model.cache
        self._blocks = blocks
        self._model: engine.Decoder | None = model
        self._device = device
        self._vocabulary = vocabulary
        self._mask = mask
        self._cache: engine.decoder.Cache | None = engine.decoder.Cache(model.cache, blocks)
        self._held: dict[SequenceId, _Held] = {}

    def release_weights(self) -> None:
        """Drop the draft model while retaining cache state for a weights-only sleep."""
        self._model = None

    def restore_weights(self, model: engine.Decoder) -> None:
        """Restore a compatible draft model after a weights-only sleep.

        The caller must preserve the original device and token vocabulary. If cache storage remains
        awake, the restored model must also preserve its cache layout.
        """
        _require_token_output(model, "draft session")
        block_sizes = _paged_cache_blocks(model, "draft session")
        if block_sizes != self._block_sizes:
            raise ValueError(
                f"draft cache block sizes changed from {self._block_sizes} to {block_sizes}"
            )
        if self._cache is not None and _cache_layout(model) != self._cache_layout:
            raise ValueError("draft cache layout changed while its KV storage remains awake")
        self._cache_layout = _cache_layout(model)
        self._cache_spec = model.cache
        self._model = model

    def release_cache(self) -> None:
        """Drop draft cache storage and its prefix index for a KV-cache sleep."""
        self._cache = None
        self._held.clear()

    def restore_cache(self) -> None:
        """Allocate empty draft cache storage after a KV-cache sleep."""
        if self._cache is None:
            self._cache = engine.decoder.Cache(self._cache_spec, self._blocks)

    def retire(self, sequence: SequenceId) -> None:
        """Forget the cached prefix of a sequence whose blocks the framework released."""
        _ = self._held.pop(sequence, None)

    def retain(self, sequences: Sequence[SequenceId]) -> None:
        """Keep prefixes only for sequence IDs the framework still owns."""
        active = set(sequences)
        self._held = {sequence: held for sequence, held in self._held.items() if sequence in active}

    def reset(self) -> None:
        """Forget all cached prefixes after the framework clears its cache pool."""
        self._held.clear()

    def propose(self, requests: Sequence[DraftRequest[SequenceId]]) -> tuple[tuple[int, ...], ...]:
        """Return each request's drafts in input order, bounded by cache and row capacity."""
        model, cache = self._ready()
        plans = self._plan(requests, model)
        # A failed decoder call may leave partial writes; retries must rebuild the prefix.
        for plan in plans.values():
            _ = self._held.pop(plan.request.sequence, None)
        drafted = self._append(plans, model, cache)
        if self._mask is None:
            self._draft_sequential(plans, drafted, model, cache)
        return self._read_and_remember(requests, plans, drafted)

    def _ready(self) -> tuple[engine.Decoder, engine.decoder.Cache]:
        model, cache = self._model, self._cache
        if model is None or cache is None:
            raise RuntimeError("draft weights and KV cache must be awake before drafting")
        return model, cache

    def _plan(
        self, requests: Sequence[DraftRequest[SequenceId]], model: engine.Decoder
    ) -> dict[int, _Plan[SequenceId]]:
        if len({request.sequence for request in requests}) != len(requests):
            raise ValueError("a drafting batch must contain each sequence at most once")

        groups = model.cache.groups
        if not groups:
            raise ValueError("a draft model must have at least one cache group")
        plans: dict[int, _Plan[SequenceId]] = {}
        for index, request in enumerate(requests):
            if not request.drafts:
                continue
            if len(request.blocks) != len(groups):
                tables, group_count = len(request.blocks), len(groups)
                raise ValueError(
                    f"got {tables} request tables for {group_count} draft cache groups"
                )
            capacities = tuple(
                len(table) * model.cache.pools[group.pool].block
                for table, group in zip(request.blocks, groups, strict=True)
            )
            capacity = min(model.window, *capacities)
            # The last known row yields draft 1 without another cache slot. If `end == capacity`,
            # it still fits. Recompute that row because rejection may replace its previous draft.
            available = max(0, capacity - len(request.tokens) + 1)
            count = min(request.drafts, available)
            if not count:
                continue
            held = self._held.get(request.sequence)
            # Framework cache state bounds reuse but does not populate this session's draft cache.
            start = 0
            if held is not None:
                start = min(request.reusable, len(request.tokens) - 1)
                reusable = min(held.positions, start)
                # Blocks after this prefix cannot affect which cached tokens are reusable.
                common = min(
                    block_size
                    * _common_blocks(
                        previous,
                        current,
                        (reusable + block_size - 1) // block_size,
                    )
                    for block_size, previous, current in zip(
                        self._block_sizes, held.blocks, request.blocks, strict=True
                    )
                )
                start = min(reusable, common)
            plans[index] = _Plan(request, start, count)
        return plans

    def _append(
        self,
        plans: dict[int, _Plan[SequenceId]],
        model: engine.Decoder,
        cache: engine.decoder.Cache,
    ) -> dict[int, list[_OutputRow]]:
        drafted: dict[int, list[_OutputRow]] = {index: [] for index in plans}
        sequences = tuple(plans)
        lengths = tuple(
            plan.end - plan.start + (plan.count - 1 if self._mask is not None else 0)
            for plan in plans.values()
        )
        for batch in chunks(lengths, model.rows):
            rows: list[Row[int]] = []
            for chunk in batch:
                index = sequences[chunk.sequence]
                plan = plans[index]
                rows.extend(
                    Row(index, position, position + 1 >= plan.end)
                    for position in range(
                        plan.start + chunk.start,
                        plan.start + chunk.start + chunk.length,
                    )
                )
            call = Call[int].plan(rows, every=False)
            ids = [
                int(plans[row.sequence].request.tokens[row.position])
                if self._mask is None or row.position < plans[row.sequence].end
                else self._mask
                for row in rows
            ]
            tokens = engine.HostTensor.from_dlpack(operand(ids, self._vocabulary)).to_hbm(
                self._device
            )
            kept_rows = [sum(rows[row].complete for row in run.rows) for run in call.requests]
            output = model.forward(
                [
                    engine.decoder.Request(
                        plans[run.sequence].request.blocks,
                        run.cached,
                        tokens[run.rows[0] : run.rows[-1] + 1],
                        range(len(run.rows) - kept, len(run.rows)),
                    )
                    for run, kept in zip(call.requests, kept_rows, strict=True)
                ],
                cache,
            )
            for output_row, row in enumerate(call.kept):
                drafted[rows[row].sequence].append(_OutputRow(output, output_row))
        return drafted

    def _draft_sequential(
        self,
        plans: dict[int, _Plan[SequenceId]],
        drafted: dict[int, list[_OutputRow]],
        model: engine.Decoder,
        cache: engine.decoder.Cache,
    ) -> None:
        widest = max((plan.count for plan in plans.values()), default=0)
        for step in range(1, widest):
            active = tuple(index for index, plan in plans.items() if plan.count > step)
            for batch in chunks((1,) * len(active), model.rows):
                indices = tuple(active[chunk.sequence] for chunk in batch)
                output = model.forward(
                    [
                        self._next_request(
                            plans[index], drafted[index], plans[index].end + step - 1
                        )
                        for index in indices
                    ],
                    cache,
                )
                for output_row, index in enumerate(indices):
                    drafted[index].append(_OutputRow(output, output_row))

    @staticmethod
    def _next_request(
        plan: _Plan[SequenceId], drafted: list[_OutputRow], cached: int
    ) -> engine.decoder.Request:
        previous = drafted[-1]
        return engine.decoder.Request(
            plan.request.blocks,
            cached,
            previous.output[previous.row : previous.row + 1],
            engine.decoder.Keep.LAST,
        )

    def _read_and_remember(
        self,
        requests: Sequence[DraftRequest[SequenceId]],
        plans: dict[int, _Plan[SequenceId]],
        drafted: dict[int, list[_OutputRow]],
    ) -> tuple[tuple[int, ...], ...]:
        # Kept rows from one call share its output tensor; transfer each output only once.
        outputs = {id(row.output): row.output for rows in drafted.values() for row in rows}
        hosts: dict[int, NDArray[np.int32]] = {
            identity: cast(NDArray[np.int32], output.to_host().numpy())
            for identity, output in outputs.items()
        }
        proposed = {
            index: tuple(int(cast(np.int32, hosts[id(row.output)][row.row])) for row in rows)
            for index, rows in drafted.items()
        }
        for index, plan in plans.items():
            if len(proposed[index]) != plan.count:
                raise RuntimeError(
                    f"draft decoder returned {len(proposed[index])} rows "
                    + f"for sequence {plan.request.sequence!r}; expected {plan.count}"
                )
        for plan in plans.values():
            # Mask rows occupy the speculative suffix; only known tokens survive the next step.
            positions = plan.end + plan.count - 1 if self._mask is None else plan.end
            self._held[plan.request.sequence] = _Held(positions, plan.request.blocks)
        return tuple(proposed.get(index, ()) for index in range(len(requests)))


def same_tokenizer_vocabulary(target: Mapping[str, int], draft: Mapping[str, int]) -> bool:
    """Whether target and draft tokenizers map every token to the same ID."""
    return target == draft


def validate_draft_vocabulary(target: int, draft: int) -> None:
    """Refuse a draft whose token IDs cannot index the target's vocabulary."""
    if target != draft:
        raise ValueError(
            f"draft vocabulary size {draft} does not match target vocabulary size {target}"
        )


def _require_token_output(decoder: engine.Decoder, role: str) -> None:
    """Require a decoder whose calls return token IDs."""
    if decoder.output != engine.decoder.Output.TOKENS:
        raise ValueError(f"{role} decoder must return token IDs")


def _paged_cache_blocks(decoder: engine.Decoder, role: str) -> tuple[int, ...]:
    """The block size of each paged attention group, refusing another cache layout."""
    groups = decoder.cache.groups
    if not groups or any(not isinstance(group.kind, engine.decoder.Kind.Paged) for group in groups):
        raise ValueError(f"{role} decoder requires paged attention cache groups")
    return tuple(decoder.cache.pools[group.pool].block for group in groups)


def _cache_layout(decoder: engine.Decoder) -> tuple[object, ...]:
    """Cache dtype and ordered paged-group formats that existing storage must match."""
    return (
        decoder.kv_cache,
        tuple(
            (
                tuple(group.layers),
                group.window,
                decoder.cache.pools[group.pool].block,
                decoder.cache.pools[group.pool].bytes,
            )
            for group in decoder.cache.groups
        ),
    )


def validate_draft_decoder(target: engine.Decoder, draft: engine.Decoder, context: int) -> None:
    """Require token output, matching paged-cache blocks and enough draft context."""
    _require_token_output(draft, "draft")
    if type(context) is not int or context < 1:
        raise ValueError("draft context must be a positive integer")
    target_blocks = _paged_cache_blocks(target, "target")
    draft_blocks = _paged_cache_blocks(draft, "draft")
    if draft_blocks != target_blocks:
        raise ValueError(
            "target and draft cache group block sizes differ "
            + f"(target {target_blocks}, draft {draft_blocks})"
        )
    if draft.window < context:
        raise ValueError(
            f"draft context window {draft.window} is shorter than required context {context}"
        )


def validate_mask_token(token: object | None, vocabulary: int) -> int | None:
    """Validate a mask ID shared by the draft and target token vocabularies."""
    if type(vocabulary) is not int or vocabulary < 1:
        raise ValueError("vocabulary size must be a positive integer")
    if token is None:
        return None
    if type(token) is not int or not 0 <= token < vocabulary:
        raise ValueError(
            f"mask token ID {token!r} must lie in the vocabulary of {vocabulary} tokens"
        )
    return token


def _cache_bytes_per_block(cache: engine.decoder.CacheSpec) -> int:
    """Bytes per shared request block across every cache group and layer."""
    groups = cache.groups
    if not groups:
        raise ValueError("cache must have at least one group")
    size = sum(cache.pools[group.pool].bytes * len(group.layers) for group in groups)
    if size <= 0:
        raise ValueError("cache bytes per block must be positive")
    return size


def _cache_max_blocks(cache: engine.decoder.CacheSpec) -> int:
    """Maximum block count before a cache layer's I32 byte offset overflows."""
    groups = cache.groups
    if not groups:
        raise ValueError("cache must have at least one group")
    pages = tuple(cache.pools[group.pool].bytes for group in groups)
    if any(page <= 0 for page in pages):
        raise ValueError("cache page bytes must be positive")
    return min((2**31) // page for page in pages)


def cache_blocks_for_memory(memory: int, *caches: engine.decoder.CacheSpec) -> int:
    """Blocks that fit `memory` when each cache uses the same request block IDs."""
    if type(memory) is not int:
        raise ValueError("cache memory must be an integer")
    if not caches:
        raise ValueError("at least one cache is required")
    bytes_per_block = sum(_cache_bytes_per_block(cache) for cache in caches)
    max_blocks = min(_cache_max_blocks(cache) for cache in caches)
    return min(max(0, memory) // bytes_per_block, max_blocks)


def draft_blocks_for_shared_budget(
    target_blocks: int, target_bytes_per_block: int, draft: engine.decoder.CacheSpec
) -> int:
    """Draft blocks beside a target within its already-resolved cache budget."""
    if (
        type(target_blocks) is not int
        or type(target_bytes_per_block) is not int
        or target_blocks < 0
        or target_bytes_per_block <= 0
    ):
        raise ValueError("target blocks must be nonnegative and block bytes must be positive")
    draft_bytes_per_block = _cache_bytes_per_block(draft)
    combined = target_bytes_per_block + draft_bytes_per_block
    return min(
        target_blocks * target_bytes_per_block // combined,
        _cache_max_blocks(draft),
    )


def cache_memory_budget(
    available: int, capacity: int, utilization: float, workspace: int = 0
) -> int:
    """Bytes left after configured headroom and any caller-reported workspace.

    Utilization leaves framework-configured headroom; it is not a measured peak scratch bound.
    """
    if not 0.0 <= utilization <= 1.0:
        raise ValueError(f"memory utilization must be in 0..1, not {utilization!r}")
    if available < 0 or capacity < 0 or workspace < 0:
        raise ValueError("available memory, device capacity and workspace must be nonnegative")
    headroom = capacity - int(capacity * utilization)
    return max(0, min(available, capacity) - workspace - headroom)


def _common_blocks(left: Sequence[int], right: Sequence[int], limit: int) -> int:
    """Count leading block IDs shared, up to `limit`."""
    for index in range(min(len(left), len(right), limit)):
        if left[index] != right[index]:
            return index
    return min(len(left), len(right), limit)
