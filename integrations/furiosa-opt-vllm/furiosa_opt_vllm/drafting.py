"""Draft each request's next tokens with a causal language model loaded beside the target.

- `Drafter.load` loads the draft model once the target is loaded, sharing its device for a matching
  placement and opening another for disjoint chips; it refuses partial overlap and a cache that
  cannot follow the target's token IDs, block size or context window.
- `Drafting` is a drafter with its cache, which the target's block tables address, and the
  leading positions of each request that cache holds.
- A draft model drafts each position after the first from the draft before it; under parallel
  drafting it reads its mask token there instead, in the calls that append the known tokens, so
  that one step's drafts are the outputs of one forward of the known tokens and the masks.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol, cast, final

import numpy as np

import furiosa_opt_engine as engine
from furiosa_opt_engine.placement import draft_device
from furiosa_opt_engine.speculative import (
    DraftRequest,
    DraftSession,
    cache_blocks_for_memory,
    cache_memory_budget,
    draft_blocks_for_shared_budget,
    same_tokenizer_vocabulary,
    validate_draft_decoder,
    validate_draft_vocabulary,
)
from furiosa_opt_vllm import checkpoint
from furiosa_opt_vllm.admission import Refused

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from numpy.typing import NDArray
    from vllm.config import ModelConfig
    from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

    from furiosa_opt_vllm.admission import Draft
    from furiosa_opt_vllm.checkpoint import Location
    from furiosa_opt_vllm.decoder import Decoder


class _VocabularyTokenizer(Protocol):
    """A vLLM tokenizer exposing the mapping used by the engine's integer token IDs."""

    def get_vocab(self) -> Mapping[str, int]: ...


def _tokenizer_vocabulary(config: ModelConfig) -> Mapping[str, int]:
    """The token-to-ID mapping of vLLM's resolved tokenizer settings."""
    from vllm import tokenizers

    factory = cast("Callable[..., _VocabularyTokenizer]", tokenizers.get_tokenizer)
    tokenizer = factory(
        config.tokenizer,
        revision=config.tokenizer_revision,
        tokenizer_mode=config.tokenizer_mode,
        trust_remote_code=config.trust_remote_code,
    )
    return tokenizer.get_vocab()


def _same_tokenizer_vocabulary(target: ModelConfig, draft: ModelConfig) -> bool:
    """Whether vLLM's resolved target and draft tokenizers use identical IDs."""
    return same_tokenizer_vocabulary(_tokenizer_vocabulary(target), _tokenizer_vocabulary(draft))


def _select_device(draft: Draft, target: Decoder) -> engine.Device:
    """Select the target device or an explicitly disjoint draft device and placement."""
    placement = (
        None
        if draft.placement is None
        else engine.Placement(list(draft.placement.chips), range(*draft.placement.pes))
    )
    try:
        return draft_device(
            target.device,
            placement,
            draft.parallel.tensor_parallel_size,
        )
    except ValueError as error:
        raise Refused(
            "additional_config.furiosa.draft_placement",
            str(error),
        ) from None


@dataclass(frozen=True, slots=True)
class Drafter:
    """A draft model and the checkpoint state needed to release and restore its weights."""

    model: engine.Decoder | None
    device: engine.Device
    target_device: engine.Device
    draft: Draft
    location: Location
    options: engine.LoadOptions
    copy: engine.Checkpoint | None = None

    @classmethod
    def load(
        cls,
        draft: Draft,
        target: Decoder | None,
        target_config: ModelConfig | None,
    ) -> Drafter:
        """Load `draft` beside token-input `target` after matching tokenizer IDs and cache shape."""
        if target is None or target.model is None:
            raise Refused(
                "speculative_config.model",
                "speculative_config.model requires a target whose attention cache takes token IDs",
            )
        if target_config is None:
            raise Refused("speculative_config.model", "requires the target tokenizer configuration")
        # The vocabulary decides before the draft model loads, so a refusal reads no weight.
        draft_vocabulary = draft.model.get_vocab_size()
        try:
            validate_draft_vocabulary(target.vocabulary, draft_vocabulary)
        except ValueError as error:
            raise Refused("speculative_config.model", str(error)) from None
        if not _same_tokenizer_vocabulary(target_config, draft.model):
            raise Refused(
                "speculative_config.model",
                "speculative_config.model requires matching target and draft token-to-ID mappings",
            )
        device = _select_device(draft, target)
        options = engine.LoadOptions().chips(draft.parallel.tensor_parallel_size)
        if draft.cache_dtype is not None:
            options = options.kv_cache(draft.cache_dtype)
        location = checkpoint.located(draft.model, draft.download_dir)
        mapped = location.mapped()
        copy = mapped.read() if target_config.enable_sleep_mode else None
        loaded = options.load(mapped if copy is None else copy, device)
        match loaded:
            case engine.Decoder():
                model = loaded
            case _:
                raise Refused(
                    "speculative_config.model",
                    "speculative_config.model requires a causal language model",
                )
        try:
            validate_draft_decoder(target.model, model, draft.max_model_len)
        except ValueError as error:
            raise Refused("speculative_config.model", str(error)) from None
        return cls(model, device, target.device, draft, location, options, copy)

    def release_weights(self, *, preserve_copy: bool) -> Drafter:
        """Drop loaded draft weights, keeping their host copy only for level-1 sleep."""
        return replace(self, model=None, copy=self.copy if preserve_copy else None)

    def synchronize(self) -> None:
        """Wait for the device that owns this draft model before releasing its resources."""
        self.device.synchronize()

    def restore_weights(self, target: Decoder, target_config: ModelConfig) -> Drafter:
        """Reload draft weights beside a restored target, preserving level-1 checkpoint state."""
        if self.model is not None:
            return self
        if target.model is None:
            raise RuntimeError("target weights must be awake before draft weights")
        source = self.copy
        if source is None:
            mapped = self.location.mapped()
            source = mapped.read() if target_config.enable_sleep_mode else mapped
        match self.options.load(source, self.device):
            case engine.Decoder() as model:
                pass
            case loaded:
                raise Refused(
                    "speculative_config.model",
                    "speculative_config.model requires a causal language model, got "
                    + type(loaded).__name__,
                )
        try:
            validate_draft_decoder(target.model, model, self.draft.max_model_len)
        except ValueError as error:
            raise Refused("speculative_config.model", str(error)) from None
        return replace(
            self,
            model=model,
            copy=source if target_config.enable_sleep_mode else None,
        )

    def capacity(self, blocks: int, target_bytes_per_block: int) -> int:
        """Target blocks that fit beside draft KV under each placement's cache budget.

        `blocks` is target capacity at `target_bytes_per_block`. Colocation splits that resolved
        budget; disjoint placement applies the framework's utilization headroom to the draft cache.
        """
        model = self.model
        if model is None:
            raise RuntimeError("draft weights must be awake before cache sizing")
        cache = model.cache
        if self.device.placement == self.target_device.placement:
            return draft_blocks_for_shared_budget(blocks, target_bytes_per_block, cache)
        available = cache_memory_budget(
            self.device.available,
            self.device.capacity,
            self.draft.memory_utilization,
        )
        return min(blocks, cache_blocks_for_memory(available, cache))


@dataclass(frozen=True, slots=True)
class _TokenRow:
    """A lazy row of vLLM's CPU token matrix, limited to its known-token prefix."""

    tokens: NDArray[np.int32]
    row: int
    end: int

    def __len__(self) -> int:
        return self.end

    def __getitem__(self, position: int, /) -> int:
        return int(cast(np.int32, self.tokens[self.row, position]))


@final
class Drafting:
    """A drafter with a cache of `blocks` blocks, addressed by the block tables of the runner's
    `requests`. The mapping must remain live so each proposal reads current block tables."""

    def __init__(
        self, drafter: Drafter, blocks: int, requests: Mapping[str, CachedRequestState]
    ) -> None:
        self._drafter = drafter
        self._requests = requests
        model = drafter.model
        if model is None:
            raise RuntimeError("draft weights must be awake before allocating their cache")
        self._session: DraftSession[str] = DraftSession(
            model,
            drafter.device,
            blocks,
            drafter.draft.model.get_vocab_size(),
            drafter.draft.mask,
        )

    def capacity(self, blocks: int, target_bytes_per_block: int) -> int:
        """Target blocks that fit beside this draft's cache."""
        return self._drafter.capacity(blocks, target_bytes_per_block)

    def with_cache(self, blocks: int, requests: Mapping[str, CachedRequestState]) -> Drafting:
        """Create a fresh drafting session for a newly sized cache and request table."""
        return Drafting(self._drafter, blocks, requests)

    def release_weights(self, *, preserve_copy: bool) -> None:
        """Release draft weights while keeping or dropping their host checkpoint copy."""
        self._session.release_weights()
        self._drafter = self._drafter.release_weights(preserve_copy=preserve_copy)

    def synchronize(self) -> None:
        """Wait for draft forwards before the runner releases draft memory."""
        self._drafter.synchronize()

    def restore_weights(self, target: Decoder, target_config: ModelConfig) -> None:
        """Restore draft weights and reconnect the live session to the loaded model."""
        drafter = self._drafter.restore_weights(target, target_config)
        model = drafter.model
        if model is None:
            raise RuntimeError("restoring draft weights returned no model")
        self._session.restore_weights(model)
        self._drafter = drafter

    def release_cache(self) -> None:
        """Release draft KV storage and forget prefixes whose cache contents were dropped."""
        self._session.release_cache()

    def restore_cache(self) -> None:
        """Allocate empty draft KV storage after vLLM restores its target cache."""
        self._session.restore_cache()

    def propose(self, batch: InputBatch, sampled: list[list[int]], count: int) -> list[list[int]]:
        """Up to `count` greedy drafts after each request's known tokens in `batch`, in batch order;
        none for a request that sampled no token this step."""
        requests: list[DraftRequest[str]] = []
        indices: list[int] = []
        for index, tokens in enumerate(sampled):
            if not tokens or not count:
                continue
            req_id = batch.req_ids[index]
            end = int(cast(np.int32, batch.num_tokens_no_spec[index]))
            requests.append(
                DraftRequest(
                    req_id,
                    _TokenRow(batch.token_ids_cpu, index, end),
                    end - 1,
                    self._requests[req_id].block_ids,
                    count,
                )
            )
            indices.append(index)
        proposals = self._session.propose(requests)
        result: list[list[int]] = [[] for _ in sampled]
        for index, proposal in zip(indices, proposals, strict=True):
            result[index] = list(proposal)
        return result

    def retire(self, req_id: str) -> None:
        """Forget what the cache holds of `req_id`, whose blocks vLLM freed or replaced."""
        self._session.retire(req_id)
